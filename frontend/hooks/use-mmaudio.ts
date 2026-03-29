import { useState, useRef, useCallback } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

export type MMAudioStatus = 'idle' | 'running' | 'complete' | 'error' | 'cancelled'

export interface MMAudioState {
  status: MMAudioStatus
  outputPath: string | null
  outputUrl: string | null
  error: string | null
  logTail: string
}

const INITIAL: MMAudioState = {
  status: 'idle',
  outputPath: null,
  outputUrl: null,
  error: null,
  logTail: '',
}

function pathToFileUrl(winPath: string): string {
  const normalized = winPath.replace(/\\/g, '/')
  return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
}

export function useMMAudio() {
  const [state, setState] = useState<MMAudioState>(INITIAL)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const poll = useCallback(async () => {
    try {
      const res = await backendFetch('/api/mmaudio/progress')
      if (!res.ok) return
      const data = await res.json()

      const status: MMAudioStatus = data.status
      const outputPath: string | null = data.output_path ?? null

      setState(prev => ({
        ...prev,
        status,
        outputPath,
        outputUrl: outputPath ? pathToFileUrl(outputPath) : prev.outputUrl,
        error: data.error ?? null,
        logTail: data.log_tail ?? '',
      }))

      if (status === 'complete' || status === 'error' || status === 'cancelled') {
        stopPolling()
      }
    } catch (err) {
      logger.error(`MMAudio poll failed: ${String(err)}`)
    }
  }, [stopPolling])

  const generate = useCallback(async (videoPath: string, prompt: string, duration: number, seed?: number) => {
    stopPolling()
    setState({ ...INITIAL, status: 'running' })

    try {
      const res = await backendFetch('/api/mmaudio/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoPath, prompt, duration, seed: seed ?? null }),
      })
      if (!res.ok) {
        const err = await res.text()
        setState(prev => ({ ...prev, status: 'error', error: err }))
        return
      }
    } catch (err) {
      setState(prev => ({ ...prev, status: 'error', error: String(err) }))
      return
    }

    pollRef.current = setInterval(poll, 1500)
  }, [stopPolling, poll])

  const cancel = useCallback(async () => {
    stopPolling()
    try {
      await backendFetch('/api/mmaudio/cancel', { method: 'POST' })
    } catch { /* ignore */ }
    setState(prev => ({ ...prev, status: 'cancelled' }))
  }, [stopPolling])

  const reset = useCallback(() => {
    stopPolling()
    setState(INITIAL)
  }, [stopPolling])

  return { state, generate, cancel, reset }
}
