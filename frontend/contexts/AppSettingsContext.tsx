import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { backendFetch, resetBackendCredentials } from '../lib/backend'

export interface InferenceSettings {
  steps: number
  useUpscaler: boolean
}

export interface FastModelSettings {
  useUpscaler: boolean
  steps: number
}

export interface AppSettings {
  useTorchCompile: boolean
  loadOnStartup: boolean
  hasLtxApiKey: boolean
  userPrefersLtxApiVideoGenerations: boolean
  hasFalApiKey: boolean
  hasGeminiApiKey: boolean
  useLocalTextEncoder: boolean
  fastModel: FastModelSettings
  proModel: InferenceSettings
  promptCacheSize: number
  promptEnhancerEnabledT2V: boolean
  promptEnhancerEnabledI2V: boolean
  seedLocked: boolean
  lockedSeed: number
  modelsDir: string
  // VRAM optimisation settings
  blockSwapBlocksOnGpu: number
  useFp8Transformer: boolean
  attentionTileSize: number
  textEncoderDevice: string
  transformerDevice: string
  useMultiGpu: boolean
  ggufTransformerPath: string
  useAbliteratedEncoder: boolean
  civitaiLoras: string
  vaeSpatialTileSize: number
  vaeTemporalTileSize: number
  unloadTextEncoderAfterEncode: boolean
  enhancePromptLocally: boolean
}

export const DEFAULT_APP_SETTINGS: AppSettings = {
  useTorchCompile: false,
  loadOnStartup: true,
  hasLtxApiKey: false,
  userPrefersLtxApiVideoGenerations: false,
  hasFalApiKey: false,
  hasGeminiApiKey: false,
  useLocalTextEncoder: false,
  fastModel: { useUpscaler: true, steps: 8 },
  proModel: { steps: 20, useUpscaler: true },
  promptCacheSize: 1,
  promptEnhancerEnabledT2V: false,
  promptEnhancerEnabledI2V: false,
  seedLocked: false,
  lockedSeed: 42,
  modelsDir: '',
  blockSwapBlocksOnGpu: 0,
  useFp8Transformer: false,
  attentionTileSize: 0,
  textEncoderDevice: '',
  transformerDevice: '',
  useMultiGpu: false,
  ggufTransformerPath: '',
  useAbliteratedEncoder: false,
  civitaiLoras: '[]',
  vaeSpatialTileSize: 0,
  vaeTemporalTileSize: 0,
  unloadTextEncoderAfterEncode: false,
  enhancePromptLocally: false,
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
  saveLtxApiKey: (value: string) => Promise<void>
  saveFalApiKey: (value: string) => Promise<void>
  saveGeminiApiKey: (value: string) => Promise<void>
  forceApiGenerations: boolean
  shouldVideoGenerateWithLtxApi: boolean
}

const AppSettingsContext = createContext<AppSettingsContextValue | null>(null)

function toBackendProcessStatus(value: unknown): BackendProcessStatus | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { status?: unknown }
  if (record.status === 'alive' || record.status === 'restarting' || record.status === 'dead') {
    return record.status
  }
  return null
}

function normalizeAppSettings(data: Partial<AppSettings>): AppSettings {
  return {
    useTorchCompile: data.useTorchCompile ?? DEFAULT_APP_SETTINGS.useTorchCompile,
    loadOnStartup: data.loadOnStartup ?? DEFAULT_APP_SETTINGS.loadOnStartup,
    hasLtxApiKey: data.hasLtxApiKey ?? DEFAULT_APP_SETTINGS.hasLtxApiKey,
    userPrefersLtxApiVideoGenerations: data.userPrefersLtxApiVideoGenerations ?? DEFAULT_APP_SETTINGS.userPrefersLtxApiVideoGenerations,
    hasFalApiKey: data.hasFalApiKey ?? DEFAULT_APP_SETTINGS.hasFalApiKey,
    hasGeminiApiKey: data.hasGeminiApiKey ?? DEFAULT_APP_SETTINGS.hasGeminiApiKey,
    useLocalTextEncoder: data.useLocalTextEncoder ?? DEFAULT_APP_SETTINGS.useLocalTextEncoder,
    fastModel: data.fastModel ?? DEFAULT_APP_SETTINGS.fastModel,
    proModel: data.proModel ?? DEFAULT_APP_SETTINGS.proModel,
    promptCacheSize: data.promptCacheSize ?? DEFAULT_APP_SETTINGS.promptCacheSize,
    promptEnhancerEnabledT2V: data.promptEnhancerEnabledT2V ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabledT2V,
    promptEnhancerEnabledI2V: data.promptEnhancerEnabledI2V ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabledI2V,
    seedLocked: data.seedLocked ?? DEFAULT_APP_SETTINGS.seedLocked,
    lockedSeed: data.lockedSeed ?? DEFAULT_APP_SETTINGS.lockedSeed,
    modelsDir: data.modelsDir ?? DEFAULT_APP_SETTINGS.modelsDir,
    blockSwapBlocksOnGpu: data.blockSwapBlocksOnGpu ?? DEFAULT_APP_SETTINGS.blockSwapBlocksOnGpu,
    useFp8Transformer: data.useFp8Transformer ?? DEFAULT_APP_SETTINGS.useFp8Transformer,
    attentionTileSize: data.attentionTileSize ?? DEFAULT_APP_SETTINGS.attentionTileSize,
    textEncoderDevice: data.textEncoderDevice ?? DEFAULT_APP_SETTINGS.textEncoderDevice,
    transformerDevice: data.transformerDevice ?? DEFAULT_APP_SETTINGS.transformerDevice,
    useMultiGpu: data.useMultiGpu ?? DEFAULT_APP_SETTINGS.useMultiGpu,
    ggufTransformerPath: data.ggufTransformerPath ?? DEFAULT_APP_SETTINGS.ggufTransformerPath,
    useAbliteratedEncoder: data.useAbliteratedEncoder ?? DEFAULT_APP_SETTINGS.useAbliteratedEncoder,
    civitaiLoras: data.civitaiLoras ?? DEFAULT_APP_SETTINGS.civitaiLoras,
    vaeSpatialTileSize: data.vaeSpatialTileSize ?? DEFAULT_APP_SETTINGS.vaeSpatialTileSize,
    vaeTemporalTileSize: data.vaeTemporalTileSize ?? DEFAULT_APP_SETTINGS.vaeTemporalTileSize,
    unloadTextEncoderAfterEncode: data.unloadTextEncoderAfterEncode ?? DEFAULT_APP_SETTINGS.unloadTextEncoderAfterEncode,
    enhancePromptLocally: data.enhancePromptLocally ?? DEFAULT_APP_SETTINGS.enhancePromptLocally,
  }
}

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const [isLoaded, setIsLoaded] = useState(false)
  const [runtimePolicyLoaded, setRuntimePolicyLoaded] = useState(false)
  const [forceApiGenerations, setForceApiGenerations] = useState(false)
  const [backendProcessStatus, setBackendProcessStatus] = useState<BackendProcessStatus | null>(null)

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return

    let cancelled = false
    setRuntimePolicyLoaded(false)

    const fetchRuntimePolicy = async () => {
      try {
        const response = await backendFetch('/api/runtime-policy')
        if (!response.ok) {
          throw new Error(`Runtime policy fetch failed with status ${response.status}`)
        }

        const payload = (await response.json()) as { force_api_generations?: unknown }
        if (typeof payload.force_api_generations !== 'boolean') {
          throw new Error('Runtime policy response missing force_api_generations boolean')
        }

        if (!cancelled) {
          setForceApiGenerations(payload.force_api_generations)
        }
      } catch {
        if (!cancelled) {
          // Fail open — local-generation fork; don't gate the user if policy fetch fails.
          setForceApiGenerations(false)
        }
      } finally {
        if (!cancelled) {
          setRuntimePolicyLoaded(true)
        }
      }
    }

    void fetchRuntimePolicy()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const nextStatus = toBackendProcessStatus(value)
      if (!nextStatus || cancelled) {
        return
      }
      if (nextStatus === 'alive') {
        resetBackendCredentials()
      }
      setBackendProcessStatus(nextStatus)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data) => {
      applyStatus(data)
    })

    void window.electronAPI.getBackendHealthStatus()
      .then((snapshot) => {
        applyStatus(snapshot)
      })
      .catch(() => {
        // Snapshot is optional at startup; subscription continues to listen for pushes.
      })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  const refreshSettings = useCallback(async () => {
    const response = await backendFetch('/api/settings')
    if (!response.ok) {
      throw new Error(`Settings fetch failed with status ${response.status}`)
    }
    const data = await response.json()
    setSettings(normalizeAppSettings(data))
    setIsLoaded(true)
  }, [])

  useEffect(() => {
    if (isLoaded || backendProcessStatus !== 'alive') return

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const fetchSettings = async () => {
      try {
        await refreshSettings()
        if (cancelled) return
      } catch {
        if (!cancelled) {
          retryTimer = setTimeout(fetchSettings, 1000)
        }
      }
    }

    fetchSettings()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [backendProcessStatus, isLoaded, refreshSettings])

  useEffect(() => {
    if (!isLoaded || backendProcessStatus !== 'alive') return
    const syncTimer = setTimeout(async () => {
      try {
        const { hasLtxApiKey: _a, hasFalApiKey: _b, hasGeminiApiKey: _c, modelsDir: _d, ...syncPayload } = settings
        await backendFetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(syncPayload),
        })
      } catch {
        // Best-effort settings sync.
      }
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, isLoaded, settings])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    if (typeof patch === 'function') {
      setSettings((prev) => patch(prev))
      return
    }
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  const saveLtxApiKey = useCallback(async (value: string) => {
    const response = await backendFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ltxApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save LTX API key.')
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveGeminiApiKey = useCallback(async (value: string) => {
    const response = await backendFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ geminiApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save Gemini API key.')
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveFalApiKey = useCallback(async (value: string) => {
    const response = await backendFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ falApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save FAL API key.')
    }
    await refreshSettings()
  }, [refreshSettings])

  const shouldVideoGenerateWithLtxApi =
    forceApiGenerations || (settings.userPrefersLtxApiVideoGenerations && settings.hasLtxApiKey)

  const contextValue = useMemo<AppSettingsContextValue>(
    () => ({
      settings,
      isLoaded,
      runtimePolicyLoaded,
      updateSettings,
      refreshSettings,
      saveLtxApiKey,
      saveFalApiKey,
      saveGeminiApiKey,
      forceApiGenerations,
      shouldVideoGenerateWithLtxApi,
    }),
    [forceApiGenerations, isLoaded, refreshSettings, runtimePolicyLoaded, saveFalApiKey, saveGeminiApiKey, saveLtxApiKey, settings, shouldVideoGenerateWithLtxApi, updateSettings],
  )

  return <AppSettingsContext.Provider value={contextValue}>{children}</AppSettingsContext.Provider>
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext)
  if (!context) {
    throw new Error('useAppSettings must be used within AppSettingsProvider')
  }
  return context
}
