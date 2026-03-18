import { useState, useCallback } from 'react'

const STORAGE_KEY = 'ltx_enhanced_prompt_history'
const MAX_HISTORY = 20

function loadHistory(): string[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return stored ? (JSON.parse(stored) as string[]) : []
  } catch {
    return []
  }
}

function saveHistory(history: string[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
  } catch {}
}

interface UseEnhancedPromptHistoryReturn {
  history: string[]
  addToHistory: (prompt: string) => void
  clearHistory: () => void
}

export function useEnhancedPromptHistory(): UseEnhancedPromptHistoryReturn {
  const [history, setHistory] = useState<string[]>(loadHistory)

  const addToHistory = useCallback((prompt: string) => {
    const trimmed = prompt.trim()
    if (!trimmed) return
    setHistory(prev => {
      // Deduplicate — move to front if already exists
      const deduped = prev.filter(p => p !== trimmed)
      const next = [trimmed, ...deduped].slice(0, MAX_HISTORY)
      saveHistory(next)
      return next
    })
  }, [])

  const clearHistory = useCallback(() => {
    setHistory([])
    try { localStorage.removeItem(STORAGE_KEY) } catch {}
  }, [])

  return { history, addToHistory, clearHistory }
}
