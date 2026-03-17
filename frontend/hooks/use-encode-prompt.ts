import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

interface UseEncodePromptReturn {
  isEncoding: boolean
  encodedPrompt: string | null
  encodeError: string | null
  encodePrompt: (prompt: string) => Promise<void>
  clearEncoded: () => void
}

export function useEncodePrompt(): UseEncodePromptReturn {
  const [isEncoding, setIsEncoding] = useState(false)
  const [encodedPrompt, setEncodedPrompt] = useState<string | null>(null)
  const [encodeError, setEncodeError] = useState<string | null>(null)

  const encodePrompt = useCallback(async (prompt: string) => {
    if (!prompt.trim()) return

    setIsEncoding(true)
    setEncodeError(null)

    try {
      const response = await backendFetch('/api/encode-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Encoding failed')
      }

      setEncodedPrompt(data.encoded_prompt ?? prompt.trim())
      logger.info('Prompt encoded successfully')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Encoding failed'
      setEncodeError(msg)
      logger.error(`Prompt encoding failed: ${msg}`)
    } finally {
      setIsEncoding(false)
    }
  }, [])

  const clearEncoded = useCallback(() => {
    setEncodedPrompt(null)
    setEncodeError(null)
  }, [])

  return { isEncoding, encodedPrompt, encodeError, encodePrompt, clearEncoded }
}
