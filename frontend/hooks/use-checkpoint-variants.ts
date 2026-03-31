import { useCallback, useEffect, useState } from 'react'
import { backendFetch } from '../lib/backend'

export interface CheckpointVariant {
  id: string
  label: string
  description: string
  available: boolean
  pipeline_type: string  // "fast" | "dev"
  gguf_path: string
  use_fp8: boolean
  size_gb: number | null
}

interface UseCheckpointVariantsResult {
  variants: CheckpointVariant[]
  isLoading: boolean
  refresh: () => void
}

export function useCheckpointVariants(): UseCheckpointVariantsResult {
  const [variants, setVariants] = useState<CheckpointVariant[]>([])
  const [isLoading, setIsLoading] = useState(true)

  const refresh = useCallback(() => {
    setIsLoading(true)
    backendFetch('/api/models/checkpoint-variants')
      .then((r) => r.json())
      .then((data) => setVariants(data as CheckpointVariant[]))
      .catch(() => setVariants([]))
      .finally(() => setIsLoading(false))
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { variants, isLoading, refresh }
}
