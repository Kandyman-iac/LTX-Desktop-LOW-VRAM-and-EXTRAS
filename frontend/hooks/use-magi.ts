import { useCallback, useEffect, useRef, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

export interface MagiSubmitParams {
  prompt: string
  imagePath: string
  seconds: number
  width: number
  height: number
  gpus: number
  seed?: number
  sr?: boolean
}

export interface MagiResult {
  videoPath: string
  videoUrl: string
}

interface UseMagiState {
  isGenerating: boolean
  status: string
  error: string | null
  result: MagiResult | null
  logTail: string
  srModelReady: boolean
}

const INITIAL: UseMagiState = {
  isGenerating: false,
  status: '',
  error: null,
  result: null,
  logTail: '',
  srModelReady: false,
}

export function useMagi() {
  const [state, setState] = useState<UseMagiState>(INITIAL)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const pollProgress = useCallback(async () => {
    try {
      const res = await backendFetch('/api/magi/progress')
      const data = await res.json() as { status: string; output_path?: string; error?: string; log_tail?: string; sr_model_ready?: boolean }
      const srReady = data.sr_model_ready ?? false

      if (data.status === 'complete' && data.output_path) {
        stopPolling()
        const p = data.output_path.replace(/\\/g, '/')
        const videoUrl = p.startsWith('/') ? `file://${p}` : `file:///${p}`
        setState({
          isGenerating: false,
          status: 'Generation complete!',
          error: null,
          result: { videoPath: data.output_path, videoUrl },
          logTail: data.log_tail ?? '',
          srModelReady: srReady,
        })
      } else if (data.status === 'error' || data.status === 'cancelled') {
        stopPolling()
        setState(s => ({
          ...s,
          isGenerating: false,
          status: '',
          error: data.error ?? data.status,
          logTail: data.log_tail ?? '',
          srModelReady: srReady,
        }))
      } else if (data.status === 'running') {
        setState(s => ({ ...s, logTail: data.log_tail ?? '', srModelReady: srReady }))
      } else {
        setState(s => ({ ...s, srModelReady: srReady }))
      }
    } catch (err) {
      logger.error(`[magi] Progress poll error: ${String(err)}`)
    }
  }, [stopPolling])

  const submitMagi = useCallback(async (params: MagiSubmitParams) => {
    if (!params.imagePath || !params.prompt.trim()) return

    setState({ isGenerating: true, status: 'Starting…', error: null, result: null, logTail: '' })

    try {
      const res = await backendFetch('/api/magi/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: params.prompt,
          image_path: params.imagePath,
          seconds: params.seconds,
          width: params.width,
          height: params.height,
          gpus: params.gpus,
          seed: params.seed ?? null,
          sr: params.sr ?? false,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        setState({ isGenerating: false, status: '', error: (data as { error?: string }).error ?? 'Failed to start', result: null, logTail: '' })
        return
      }
      setState(s => ({ ...s, status: 'Generating… (model load ~2 min first run)' }))
      // Poll every 3 seconds
      pollRef.current = setInterval(pollProgress, 3000)
    } catch (err) {
      setState({ isGenerating: false, status: '', error: (err as Error).message, result: null, logTail: '' })
    }
  }, [pollProgress])

  const cancelMagi = useCallback(async () => {
    stopPolling()
    try {
      await backendFetch('/api/magi/cancel', { method: 'POST' })
    } catch (_) { /* ignore */ }
    setState(s => ({ ...s, isGenerating: false, status: '' }))
  }, [stopPolling])

  const resetMagi = useCallback(() => {
    stopPolling()
    setState(INITIAL)
  }, [stopPolling])

  // Poll once on mount to populate sr_model_ready state before the user generates.
  useEffect(() => {
    void pollProgress()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => () => stopPolling(), [stopPolling])

  return {
    submitMagi,
    cancelMagi,
    resetMagi,
    isMagiGenerating: state.isGenerating,
    magiStatus: state.status,
    magiError: state.error,
    magiResult: state.result,
    magiLogTail: state.logTail,
    magiSrModelReady: state.srModelReady,
  }
}
