import { useState, useCallback, useEffect, useRef } from 'react'
import { backendFetch } from '../lib/backend'
import type { GenerationSettings } from '../components/SettingsPanel'
import type { ConditioningFrame } from '../components/MultiFrameConditioningPanel'

export interface QueuedJob {
  id: string
  status: 'pending' | 'running' | 'complete' | 'error' | 'cancelled'
  prompt: string
  result_path: string | null
  error: string | null
  created_at: number
  resolution: string
  duration: string
  fps: string
  model: string
  aspect_ratio: string
}

interface UseQueueResult {
  jobs: QueuedJob[]
  addToQueue: (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
    negativePrompt?: string,
    conditioningFrames?: ConditioningFrame[],
  ) => Promise<void>
  removeFromQueue: (jobId: string) => Promise<void>
  pendingCount: number
}

export function useQueue(): UseQueueResult {
  const [jobs, setJobs] = useState<QueuedJob[]>([])
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchJobs = useCallback(async () => {
    try {
      const res = await backendFetch('/api/queue')
      if (!res.ok) return
      const data = await res.json() as { jobs: QueuedJob[] }
      setJobs(data.jobs)
    } catch {
      // ignore polling errors
    }
  }, [])

  useEffect(() => {
    void fetchJobs()
    pollRef.current = setInterval(() => { void fetchJobs() }, 2000)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [fetchJobs])

  const addToQueue = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
    negativePrompt = '',
    conditioningFrames?: ConditioningFrame[],
  ) => {
    const body: Record<string, unknown> = {
      prompt,
      model: settings.model,
      duration: String(settings.duration),
      resolution: settings.videoResolution,
      fps: String(settings.fps),
      audio: String(settings.audio),
      cameraMotion: settings.cameraMotion,
      aspectRatio: settings.aspectRatio || '16:9',
    }
    if (settings.seed != null) body.seed = settings.seed
    if (negativePrompt) body.negativePrompt = negativePrompt
    const activeFrames = conditioningFrames?.filter(f => f.imagePath)
    if (activeFrames && activeFrames.length > 0) {
      const totalFrames = Math.round(parseFloat(String(settings.duration)) * settings.fps)
      body.conditioningImages = activeFrames.map(f => ({
        path: f.imagePath!,
        frameIdx: f.role === 'first' ? 0 : f.role === 'last' ? -1 : Math.round(totalFrames * f.position),
        strength: f.strength,
      }))
    } else if (imagePath) {
      body.imagePath = imagePath
    }
    if (audioPath) body.audioPath = audioPath

    await backendFetch('/api/queue/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    void fetchJobs()
  }, [fetchJobs])

  const removeFromQueue = useCallback(async (jobId: string) => {
    await backendFetch(`/api/queue/${jobId}`, { method: 'DELETE' })
    void fetchJobs()
  }, [fetchJobs])

  const pendingCount = jobs.filter(j => j.status === 'pending').length

  return { jobs, addToQueue, removeFromQueue, pendingCount }
}
