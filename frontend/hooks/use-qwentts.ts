import { useState, useRef, useCallback } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

export type QwenTTSStatus = 'idle' | 'running' | 'complete' | 'error' | 'cancelled'

export interface QwenTTSState {
  status: QwenTTSStatus
  outputPath: string | null
  outputUrl: string | null
  error: string | null
  logTail: string
}

const INITIAL: QwenTTSState = {
  status: 'idle',
  outputPath: null,
  outputUrl: null,
  error: null,
  logTail: '',
}

function pathToFileUrl(p: string): string {
  const n = p.replace(/\\/g, '/')
  return n.startsWith('/') ? `file://${n}` : `file:///${n}`
}

export function useQwenTTS() {
  const [state, setState] = useState<QwenTTSState>(INITIAL)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const poll = useCallback(async () => {
    try {
      const res = await backendFetch('/api/qwentts/progress')
      if (!res.ok) return
      const data = await res.json()
      const status: QwenTTSStatus = data.status
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
      logger.error(`QwenTTS poll failed: ${String(err)}`)
    }
  }, [stopPolling])

  const generate = useCallback(async (params: {
    text: string
    language: string
    mode: 'custom_voice' | 'voice_clone'
    speaker: string
    instruct: string
    refAudioPath: string | null
    refText: string
    modelSize: '1.7b' | '0.6b'
  }) => {
    stopPolling()
    setState({ ...INITIAL, status: 'running' })
    try {
      const res = await backendFetch('/api/qwentts/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: params.text,
          language: params.language,
          mode: params.mode,
          speaker: params.speaker,
          instruct: params.instruct,
          ref_audio_path: params.refAudioPath,
          ref_text: params.refText,
          model_size: params.modelSize,
        }),
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
    try { await backendFetch('/api/qwentts/cancel', { method: 'POST' }) } catch { /* ignore */ }
    setState(prev => ({ ...prev, status: 'cancelled' }))
  }, [stopPolling])

  const reset = useCallback(() => {
    stopPolling()
    setState(INITIAL)
  }, [stopPolling])

  const unload = useCallback(async () => {
    try { await backendFetch('/api/qwentts/unload', { method: 'POST' }) } catch { /* ignore */ }
  }, [])

  return { state, generate, cancel, reset, unload }
}
