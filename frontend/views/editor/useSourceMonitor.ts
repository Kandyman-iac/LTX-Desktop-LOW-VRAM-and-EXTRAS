import { useState, useRef, useEffect, useCallback } from 'react'
import type { Asset, TimelineClip, Track } from '../../types/project'
import { DEFAULT_COLOR_CORRECTION } from '../../types/project'
import { resolveOverlaps } from './video-editor-utils'

interface UseSourceMonitorParams {
  currentTime: number
  tracks: Track[]
  pushUndo: () => void
  setClips: React.Dispatch<React.SetStateAction<TimelineClip[]>>
}

export function useSourceMonitor({ currentTime, tracks, pushUndo, setClips }: UseSourceMonitorParams) {
  const [sourceAsset, setSourceAsset] = useState<Asset | null>(null)
  const [sourceTime, setSourceTime] = useState(0)
  const [sourceIsPlaying, setSourceIsPlaying] = useState(false)
  const [sourceIn, setSourceIn] = useState<number | null>(null)
  const [sourceOut, setSourceOut] = useState<number | null>(null)
  const [showSourceMonitor, setShowSourceMonitor] = useState(false)
  const [activePanel, setActivePanel] = useState<'source' | 'timeline'>('timeline')
  const [sourceSplitPercent, setSourceSplitPercent] = useState(50)
  const sourceVideoRef = useRef<HTMLVideoElement>(null)
  const sourceAnimRef = useRef<number>(0)
  const sourceTimeRef = useRef(0)
  sourceTimeRef.current = sourceTime
  const sourceIsPlayingRef = useRef(false)
  sourceIsPlayingRef.current = sourceIsPlaying

  // Refs to avoid stale closures in callbacks
  const tracksRef = useRef(tracks)
  tracksRef.current = tracks
  const currentTimeRef = useRef(currentTime)
  currentTimeRef.current = currentTime
  const sourceAssetRef = useRef(sourceAsset)
  sourceAssetRef.current = sourceAsset
  const sourceInRef = useRef(sourceIn)
  sourceInRef.current = sourceIn
  const sourceOutRef = useRef(sourceOut)
  sourceOutRef.current = sourceOut

  // --- Drag from source monitor to timeline ---
  const handleSourceDragStart = useCallback((e: React.DragEvent) => {
    const asset = sourceAssetRef.current
    if (!asset) return
    const sIn = sourceInRef.current ?? 0
    const sDuration = asset.duration || 5
    const sOut = sourceOutRef.current ?? sDuration
    const insertDuration = sOut - sIn
    if (insertDuration <= 0) return
    e.dataTransfer.setData('sourcemonitordrag', JSON.stringify({
      assetId: asset.id,
      trimStart: sIn,
      trimEnd: sDuration - sOut,
      insertDuration,
    }))
    e.dataTransfer.setData('assetId', asset.id)
    e.dataTransfer.effectAllowed = 'copy'
  }, [])

  const loadSourceAsset = useCallback((asset: Asset) => {
    setSourceAsset(asset)
    setSourceTime(0)
    setSourceIn(null)
    setSourceOut(null)
    setSourceIsPlaying(false)
    setShowSourceMonitor(true)
  }, [])

  // --- Source Monitor: playback loop ---
  useEffect(() => {
    if (!sourceIsPlaying || !sourceVideoRef.current) {
      cancelAnimationFrame(sourceAnimRef.current)
      return
    }
    const video = sourceVideoRef.current
    const tick = () => {
      setSourceTime(video.currentTime)
      if (sourceOut !== null && video.currentTime >= sourceOut) {
        video.pause()
        setSourceIsPlaying(false)
        setSourceTime(sourceOut)
        return
      }
      if (!video.paused) sourceAnimRef.current = requestAnimationFrame(tick)
    }
    video.play().catch(() => {})
    sourceAnimRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(sourceAnimRef.current)
  }, [sourceIsPlaying, sourceOut])

  // --- Helper: build clips for insert/overwrite edits ---
  const buildEditClips = useCallback(() => {
    const asset = sourceAssetRef.current
    if (!asset) return null

    const sIn = sourceInRef.current ?? 0
    const sDuration = asset.duration || 5
    const sOut = sourceOutRef.current ?? sDuration
    const insertDuration = sOut - sIn
    if (insertDuration <= 0) return null

    const trks = tracksRef.current
    const time = currentTimeRef.current
    const isAudio = asset.type === 'audio'

    // Find target tracks: first unlocked, source-patched track of each kind
    const videoTrack = !isAudio
      ? trks.find(t => !t.locked && t.sourcePatched !== false && t.kind === 'video')
      : undefined
    const audioTrack = trks.find(t => !t.locked && t.sourcePatched !== false && t.kind === 'audio')

    // Validate we have the tracks we need
    if (!videoTrack && !audioTrack) return null
    if (isAudio && !audioTrack) return null
    if (!isAudio && !videoTrack) return null

    const videoTrackIndex = videoTrack ? trks.indexOf(videoTrack) : -1
    const audioTrackIndex = audioTrack ? trks.indexOf(audioTrack) : -1

    const videoClipId = `clip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const audioClipId = `clip-${Date.now()}-a-${Math.random().toString(36).substr(2, 9)}`

    const baseClip = {
      assetId: asset.id,
      startTime: time,
      duration: insertDuration,
      trimStart: sIn,
      trimEnd: sDuration - sOut,
      speed: 1,
      reversed: false,
      muted: false,
      volume: 1,
      asset,
      flipH: false as const,
      flipV: false as const,
      transitionIn: { type: 'none' as const, duration: 0.5 },
      transitionOut: { type: 'none' as const, duration: 0.5 },
      colorCorrection: { ...DEFAULT_COLOR_CORRECTION },
      opacity: 100,
    }

    const newClips: TimelineClip[] = []

    if (isAudio) {
      newClips.push({ ...baseClip, id: audioClipId, type: 'audio', trackIndex: audioTrackIndex })
    } else {
      const needsAudio = asset.type === 'video' && audioTrackIndex >= 0
      newClips.push({
        ...baseClip,
        id: videoClipId,
        type: asset.type === 'video' ? 'video' : 'image',
        trackIndex: videoTrackIndex,
        ...(needsAudio ? { linkedClipIds: [audioClipId] } : {}),
      })
      if (needsAudio) {
        newClips.push({
          ...baseClip,
          id: audioClipId,
          type: 'audio',
          trackIndex: audioTrackIndex,
          linkedClipIds: [videoClipId],
        })
      }
    }

    return { newClips, insertDuration, videoTrackIndex, audioTrackIndex, time }
  }, [])

  // --- 3-Point Editing: Insert Edit ---
  const handleInsertEdit = useCallback(() => {
    const result = buildEditClips()
    if (!result) return

    pushUndo()

    const { newClips, insertDuration, videoTrackIndex, audioTrackIndex, time } = result
    setClips(prev => {
      // Ripple clips on all targeted tracks
      const rippled = prev.map(c => {
        const isTargetTrack = (videoTrackIndex >= 0 && c.trackIndex === videoTrackIndex) ||
                              (audioTrackIndex >= 0 && c.trackIndex === audioTrackIndex)
        if (isTargetTrack && c.startTime >= time) {
          return { ...c, startTime: c.startTime + insertDuration }
        }
        return c
      })
      return [...rippled, ...newClips]
    })
  }, [pushUndo, buildEditClips])

  // --- 3-Point Editing: Overwrite Edit ---
  const handleOverwriteEdit = useCallback(() => {
    const result = buildEditClips()
    if (!result) return

    pushUndo()

    const { newClips } = result
    const newClipIds = new Set(newClips.map(c => c.id))
    setClips(prev => resolveOverlaps([...prev, ...newClips], newClipIds))
  }, [pushUndo, buildEditClips])

  return {
    sourceAsset, setSourceAsset,
    sourceTime, setSourceTime,
    sourceIsPlaying, setSourceIsPlaying,
    sourceIn, setSourceIn,
    sourceOut, setSourceOut,
    showSourceMonitor, setShowSourceMonitor,
    activePanel, setActivePanel,
    sourceSplitPercent, setSourceSplitPercent,
    sourceVideoRef, sourceAnimRef, sourceTimeRef, sourceIsPlayingRef,
    loadSourceAsset,
    handleSourceDragStart,
    handleInsertEdit,
    handleOverwriteEdit,
  }
}
