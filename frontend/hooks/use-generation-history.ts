import { useCallback } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'

export interface HistoryEntry {
  id: string
  timestamp: number
  prompt: string
  negativePrompt: string
  settings: GenerationSettings
  seedUsed: number | null
  videoPath: string | null
  /** Active LoRAs at the time of generation (filename + strength). */
  lorasUsed?: Array<{ name: string; strength: number }> | null
}

const HISTORY_KEY = 'ltx_generation_history'
const MAX_ENTRIES = 100

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : []
  } catch {
    return []
  }
}

function saveHistory(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)))
  } catch {
    // Storage quota exceeded — ignore
  }
}

export function useGenerationHistory() {
  const push = useCallback((entry: Omit<HistoryEntry, 'id' | 'timestamp'>) => {
    const entries = loadHistory()
    const newEntry: HistoryEntry = {
      ...entry,
      id: Math.random().toString(36).slice(2),
      timestamp: Date.now(),
    }
    saveHistory([newEntry, ...entries])
  }, [])

  const getAll = useCallback((): HistoryEntry[] => loadHistory(), [])

  const remove = useCallback((id: string) => {
    saveHistory(loadHistory().filter(e => e.id !== id))
  }, [])

  const clear = useCallback(() => {
    localStorage.removeItem(HISTORY_KEY)
  }, [])

  return { push, getAll, remove, clear }
}
