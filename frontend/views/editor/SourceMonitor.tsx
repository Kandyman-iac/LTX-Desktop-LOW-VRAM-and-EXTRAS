import React, { useState, useEffect, useRef } from 'react'
import { Play, Pause, Square, SkipBack, SkipForward, ChevronLeft, ChevronRight, Video, Music, X, GripHorizontal } from 'lucide-react'
import type { Asset } from '../../types/project'
import { formatTime } from './video-editor-utils'
import { Tooltip } from '../../components/ui/tooltip'

export interface SourceMonitorProps {
  sourceAsset: Asset | null
  sourceTime: number
  setSourceTime: (t: number | ((prev: number) => number)) => void
  sourceIsPlaying: boolean
  setSourceIsPlaying: (v: boolean) => void
  sourceIn: number | null
  sourceOut: number | null
  setSourceIn: (v: number | null | ((prev: number | null) => number | null)) => void
  setSourceOut: (v: number | null | ((prev: number | null) => number | null)) => void
  setShowSourceMonitor: (v: boolean) => void
  activePanel: 'source' | 'timeline'
  setActivePanel: (p: 'source' | 'timeline') => void
  sourceSplitPercent: number
  draggingMarker: 'timelineIn' | 'timelineOut' | 'sourceIn' | 'sourceOut' | null
  setDraggingMarker: React.Dispatch<React.SetStateAction<'timelineIn' | 'timelineOut' | 'sourceIn' | 'sourceOut' | null>>
  sourceVideoRef: React.RefObject<HTMLVideoElement | null>
  onInsertEdit: () => void
  onOverwriteEdit: () => void
  onSourceDragStart: (e: React.DragEvent) => void
}

export function SourceMonitor({
  sourceAsset,
  sourceTime,
  setSourceTime,
  sourceIsPlaying,
  setSourceIsPlaying,
  sourceIn,
  sourceOut,
  setSourceIn,
  setSourceOut,
  setShowSourceMonitor,
  activePanel,
  setActivePanel,
  sourceSplitPercent,
  setDraggingMarker,
  sourceVideoRef,
  onInsertEdit,
  onOverwriteEdit,
  onSourceDragStart,
}: SourceMonitorProps) {
  const [sourceReversePlaying, setSourceReversePlaying] = useState(false)
  const [videoCtxMenu, setVideoCtxMenu] = useState<{ x: number; y: number } | null>(null)
  const reverseRafRef = useRef<number | null>(null)
  const reverseLastRef = useRef<number | null>(null)

  useEffect(() => {
    if (!sourceReversePlaying) {
      if (reverseRafRef.current) cancelAnimationFrame(reverseRafRef.current)
      reverseRafRef.current = null
      reverseLastRef.current = null
      return
    }
    sourceVideoRef.current?.pause()
    const tick = (ts: number) => {
      if (!sourceReversePlaying) return
      if (reverseLastRef.current !== null) {
        const delta = (ts - reverseLastRef.current) / 1000
        const next = Math.max(0, (sourceVideoRef.current?.currentTime ?? sourceTime) - delta)
        if (sourceVideoRef.current) sourceVideoRef.current.currentTime = next
        setSourceTime(next)
        if (next <= 0) { setSourceReversePlaying(false); return }
      }
      reverseLastRef.current = ts
      reverseRafRef.current = requestAnimationFrame(tick)
    }
    reverseRafRef.current = requestAnimationFrame(tick)
    return () => { if (reverseRafRef.current) cancelAnimationFrame(reverseRafRef.current) }
  }, [sourceReversePlaying])

  return (
    <div
      className={`flex flex-col ${activePanel === 'source' ? 'ring-2 ring-blue-500 ring-inset' : 'border-r border-zinc-800'}`}
      style={{ width: `${sourceSplitPercent}%` }}
      onMouseDown={() => setActivePanel('source')}
    >
      {/* Header */}
      <div className="h-7 bg-zinc-900 border-b border-zinc-800 flex items-center justify-between px-3 flex-shrink-0">
        <span className="text-[11px] font-semibold text-zinc-400 tracking-wide">Clip Viewer</span>
        <Tooltip content="Close clip viewer" side="left">
          <button onClick={() => { setShowSourceMonitor(false); setSourceIsPlaying(false) }} className="text-zinc-500 hover:text-white">
            <X className="h-3.5 w-3.5" />
          </button>
        </Tooltip>
      </div>
      {/* Video Area */}
      <div
        className="flex-1 relative overflow-hidden bg-black flex items-center justify-center min-h-0 select-none"
        draggable={!!sourceAsset}
        onDragStart={sourceAsset ? onSourceDragStart : undefined}
        onContextMenu={(e) => {
          if (!sourceAsset) return
          e.preventDefault()
          setVideoCtxMenu({ x: e.clientX, y: e.clientY })
        }}
        onClick={() => setVideoCtxMenu(null)}
      >
        {sourceAsset ? (
          <>
            {sourceAsset.type === 'video' ? (
              <video
                ref={sourceVideoRef as React.RefObject<HTMLVideoElement>}
                src={sourceAsset.url}
                className="max-w-full max-h-full object-contain pointer-events-none"
                onTimeUpdate={() => {
                  if (sourceVideoRef.current) setSourceTime(sourceVideoRef.current.currentTime)
                }}
                onEnded={() => setSourceIsPlaying(false)}
                playsInline
              />
            ) : sourceAsset.type === 'image' ? (
              <img src={sourceAsset.url} alt="" className="max-w-full max-h-full object-contain pointer-events-none" />
            ) : (
              <div className="text-center text-zinc-500 pointer-events-none">
                <Music className="h-12 w-12 mx-auto mb-2" />
                <p className="text-sm">{sourceAsset.path?.split('/').pop() || 'Audio'}</p>
              </div>
            )}
            {/* Drag hint */}
            <div className="absolute bottom-1.5 right-1.5 flex items-center gap-1 text-[9px] text-zinc-600 pointer-events-none">
              <GripHorizontal className="h-2.5 w-2.5" />
              <span>drag to timeline</span>
            </div>
          </>
        ) : (
          <div className="text-center text-zinc-600">
            <Video className="h-10 w-10 mx-auto mb-2" />
            <p className="text-xs">Double-click an asset to load it here</p>
          </div>
        )}
        {/* Right-click context menu */}
        {videoCtxMenu && (
          <div
            className="fixed bg-zinc-800 border border-zinc-700 rounded-lg shadow-2xl py-1 z-[80] min-w-[170px] text-xs"
            style={{ left: videoCtxMenu.x, top: videoCtxMenu.y }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => { onInsertEdit(); setVideoCtxMenu(null) }}
              className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center justify-between gap-3"
            >
              <span>Insert at playhead</span>
              <span className="text-zinc-500 font-mono">,</span>
            </button>
            <button
              onClick={() => { onOverwriteEdit(); setVideoCtxMenu(null) }}
              className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center justify-between gap-3"
            >
              <span>Overwrite at playhead</span>
              <span className="text-zinc-500 font-mono">.</span>
            </button>
          </div>
        )}
      </div>
      {/* Scrub bar with In/Out markers */}
      {/* Premiere-style scrub bar with In/Out range */}
      {sourceAsset && (sourceAsset.type === 'video' || sourceAsset.type === 'audio') && (
        <div className="bg-zinc-900 border-t border-zinc-800 flex-shrink-0 relative px-2 py-1">
          {/* Scrub track */}
          <div
            id="source-scrub-bar"
            className="relative h-6 cursor-pointer group"
            onMouseDown={(e) => {
              const bar = e.currentTarget
              const rect = bar.getBoundingClientRect()
              const dur = sourceAsset.duration || 5
              const seek = (clientX: number) => {
                const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
                const t = frac * dur
                setSourceTime(t)
                if (sourceVideoRef.current) sourceVideoRef.current.currentTime = t
              }
              seek(e.clientX)
              const onMove = (ev: MouseEvent) => seek(ev.clientX)
              const onUp = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
              window.addEventListener('mousemove', onMove)
              window.addEventListener('mouseup', onUp)
            }}
          >
            {/* Base track line */}
            <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-1 bg-zinc-700 rounded-full" />

            {/* Dimmed regions outside In/Out (darker overlay) */}
            {sourceIn !== null && (
              <div
                className="absolute top-0 bottom-0 left-0 bg-black/50 rounded-l"
                style={{ width: `${(sourceIn / (sourceAsset.duration || 5)) * 100}%` }}
              />
            )}
            {sourceOut !== null && (
              <div
                className="absolute top-0 bottom-0 right-0 bg-black/50 rounded-r"
                style={{ width: `${100 - (sourceOut / (sourceAsset.duration || 5)) * 100}%` }}
              />
            )}

            {/* Selected range highlight */}
            {(sourceIn !== null || sourceOut !== null) && (
              <div
                className="absolute top-0 bottom-0 border-t-2 border-b-2 border-blue-400/70"
                style={{
                  left: `${((sourceIn ?? 0) / (sourceAsset!.duration || 5)) * 100}%`,
                  width: `${(((sourceOut ?? sourceAsset!.duration ?? 5) - (sourceIn ?? 0)) / (sourceAsset!.duration || 5)) * 100}%`,
                }}
              >
                <div className="absolute inset-0 top-1/2 -translate-y-1/2 h-1 bg-blue-400/40 rounded-full" />
              </div>
            )}

            {/* In bracket marker — draggable */}
            {sourceIn !== null && (
              <div
                className="absolute top-0 bottom-0 flex items-center z-10 cursor-ew-resize"
                style={{ left: `calc(${(sourceIn / (sourceAsset!.duration || 5)) * 100}% - 8px)`, width: 14 }}
                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setDraggingMarker('sourceIn') }}
              >
                <div className="w-1.5 h-full bg-blue-400 rounded-l-sm flex flex-col justify-between py-0.5 pointer-events-none ml-auto">
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-r" />
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-r" />
                </div>
              </div>
            )}

            {/* Out bracket marker — draggable */}
            {sourceOut !== null && (
              <div
                className="absolute top-0 bottom-0 flex items-center z-10 cursor-ew-resize"
                style={{ left: `${(sourceOut / (sourceAsset!.duration || 5)) * 100}%`, width: 14 }}
                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setDraggingMarker('sourceOut') }}
              >
                <div className="w-1.5 h-full bg-blue-400 rounded-r-sm flex flex-col justify-between py-0.5 pointer-events-none">
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-l -ml-1" />
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-l -ml-1" />
                </div>
              </div>
            )}

            {/* Playhead needle */}
            <div
              className="absolute top-0 bottom-0 z-20"
              style={{ left: `${(sourceTime / (sourceAsset.duration || 5)) * 100}%` }}
            >
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 bg-blue-400 clip-triangle" style={{ clipPath: 'polygon(50% 100%, 0% 0%, 100% 0%)' }} />
              <div className="absolute top-2 bottom-0 left-1/2 -translate-x-1/2 w-px bg-blue-400" />
            </div>
          </div>

          {/* In/Out timecode labels below scrub bar */}
          {(sourceIn !== null || sourceOut !== null) && (
            <div className="flex justify-between items-center mt-0.5 h-3.5">
              <span className="text-[9px] font-mono text-blue-400/80">
                {sourceIn !== null ? `IN ${formatTime(sourceIn)}` : ''}
              </span>
              <span className="text-[9px] font-mono text-zinc-500">
                {sourceIn !== null && sourceOut !== null
                  ? `Duration: ${formatTime(sourceOut - sourceIn)}`
                  : ''
                }
              </span>
              <span className="text-[9px] font-mono text-blue-400/80">
                {sourceOut !== null ? `OUT ${formatTime(sourceOut)}` : ''}
              </span>
            </div>
          )}
        </div>
      )}
      {/* Status bar: timecode | transport controls | duration */}
      <div className="h-8 bg-zinc-950 border-t border-zinc-800 flex items-center px-3 flex-shrink-0 gap-2">
        {/* Left: current timecode */}
        <span className="text-[12px] font-mono font-medium text-amber-400 tabular-nums tracking-tight select-none min-w-[90px]">
          {formatTime(sourceTime)}
        </span>

        {/* Center: transport controls */}
        <div className="flex-1 flex items-center justify-center gap-0.5">
          {/* Mark In */}
          <Tooltip content={sourceIn !== null ? `In: ${formatTime(sourceIn)}` : 'Set In (I)'} side="top">
            <button
              onClick={() => setSourceIn(prev => prev !== null && Math.abs(prev - sourceTime) < 0.01 ? null : sourceTime)}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${sourceIn !== null ? 'text-yellow-400' : 'text-zinc-500 hover:text-white hover:bg-zinc-800'}`}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="7,4 4,4 4,20 7,20" />
                <line x1="10" y1="12" x2="20" y2="12" />
                <polyline points="16,8 20,12 16,16" />
              </svg>
            </button>
          </Tooltip>
          <div className="w-px h-3 bg-zinc-700" />
          <Tooltip content="Go to start" side="top">
            <button
              onClick={() => { const t = sourceIn ?? 0; setSourceTime(t); if (sourceVideoRef.current) sourceVideoRef.current.currentTime = t }}
              className="h-6 w-6 flex items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <SkipBack className="h-3 w-3" />
            </button>
          </Tooltip>
          <Tooltip content="Step back" side="top">
            <button
              onClick={() => {
                setSourceReversePlaying(false)
                const t = Math.max(0, sourceTime - 1 / 24)
                setSourceTime(t)
                if (sourceVideoRef.current) sourceVideoRef.current.currentTime = t
              }}
              className="h-6 w-6 flex items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
          <Tooltip content="Play reverse" side="top">
            <button
              onClick={() => {
                if (sourceReversePlaying) {
                  setSourceReversePlaying(false)
                } else {
                  sourceVideoRef.current?.pause()
                  setSourceIsPlaying(false)
                  setSourceReversePlaying(true)
                }
              }}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${sourceReversePlaying ? 'text-blue-400' : 'text-zinc-500 hover:text-white hover:bg-zinc-800'}`}
            >
              <Play className="h-3 w-3 mr-0.5 rotate-180" />
            </button>
          </Tooltip>
          <Tooltip content="Stop" side="top">
            <button
              onClick={() => { setSourceReversePlaying(false); sourceVideoRef.current?.pause(); setSourceIsPlaying(false) }}
              className="h-6 w-6 flex items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <Square className="h-2.5 w-2.5" />
            </button>
          </Tooltip>
          <Tooltip content={sourceIsPlaying ? 'Pause' : 'Play'} side="top">
            <button
              onClick={() => {
                setSourceReversePlaying(false)
                if (sourceIsPlaying) {
                  sourceVideoRef.current?.pause()
                  setSourceIsPlaying(false)
                } else {
                  if (sourceVideoRef.current) {
                    if (sourceIn !== null && sourceTime < sourceIn) sourceVideoRef.current.currentTime = sourceIn
                    sourceVideoRef.current.play().catch(() => {})
                  }
                  setSourceIsPlaying(true)
                }
              }}
              className="h-6 w-6 flex items-center justify-center rounded text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              {sourceIsPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3 ml-0.5" />}
            </button>
          </Tooltip>
          <Tooltip content="Step forward" side="top">
            <button
              onClick={() => {
                const dur = sourceAsset?.duration || 5
                const t = Math.min(dur, sourceTime + 1 / 24)
                setSourceTime(t)
                if (sourceVideoRef.current) sourceVideoRef.current.currentTime = t
              }}
              className="h-6 w-6 flex items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
          <Tooltip content="Go to end" side="top">
            <button
              onClick={() => { const t = sourceOut ?? (sourceAsset?.duration || 5); setSourceTime(t); if (sourceVideoRef.current) sourceVideoRef.current.currentTime = t }}
              className="h-6 w-6 flex items-center justify-center rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              <SkipForward className="h-3 w-3" />
            </button>
          </Tooltip>
          <div className="w-px h-3 bg-zinc-700" />
          {/* Mark Out */}
          <Tooltip content={sourceOut !== null ? `Out: ${formatTime(sourceOut)}` : 'Set Out (O)'} side="top">
            <button
              onClick={() => setSourceOut(prev => prev !== null && Math.abs(prev - sourceTime) < 0.01 ? null : sourceTime)}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${sourceOut !== null ? 'text-yellow-400' : 'text-zinc-500 hover:text-white hover:bg-zinc-800'}`}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="17,4 20,4 20,20 17,20" />
                <line x1="14" y1="12" x2="4" y2="12" />
                <polyline points="8,8 4,12 8,16" />
              </svg>
            </button>
          </Tooltip>
          <div className="w-px h-3 bg-zinc-700 mx-0.5" />
          {/* Insert */}
          <Tooltip content="Insert Edit (,)" side="top">
            <button
              onClick={onInsertEdit}
              disabled={!sourceAsset}
              className="h-6 px-1 flex items-center rounded text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
            </button>
          </Tooltip>
          {/* Overwrite */}
          <Tooltip content="Overwrite Edit (.)" side="top">
            <button
              onClick={onOverwriteEdit}
              disabled={!sourceAsset}
              className="h-6 px-1 flex items-center rounded text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 12h6" /></svg>
            </button>
          </Tooltip>
        </div>

        {/* Right: total duration */}
        <span className="text-[12px] font-mono font-medium text-zinc-400 tabular-nums tracking-tight select-none min-w-[90px] text-right">
          {formatTime(sourceAsset?.duration || 0)}
        </span>
      </div>
    </div>
  )
}
