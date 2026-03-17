import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

interface UseEnhancePromptReturn {
  isEnhancing: boolean
  enhanceError: string | null
  enhancePrompt: (prompt: string) => Promise<string | null>
}

export function useEnhancePrompt(): UseEnhancePromptReturn {
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [enhanceError, setEnhanceError] = useState<string | null>(null)

  const enhancePrompt = useCallback(async (prompt: string): Promise<string | null> => {
    if (!prompt.trim()) return null

    setIsEnhancing(true)
    setEnhanceError(null)

    try {
      const response = await backendFetch('/api/enhance-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Enhancement failed')
      }

      logger.info('Prompt enhanced successfully')
      return data.enhanced_prompt as string
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Enhancement failed'
      setEnhanceError(msg)
      logger.error(`Prompt enhancement failed: ${msg}`)
      return null
    } finally {
      setIsEnhancing(false)
    }
  }, [])

  return { isEnhancing, enhanceError, enhancePrompt }
}
