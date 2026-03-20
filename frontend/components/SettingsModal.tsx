import { AlertCircle, Check, Download, Film, Folder, Info, KeyRound, Plus, Settings, Sliders, Sparkles, X, Zap } from 'lucide-react'
import React, { useEffect, useRef, useState } from 'react'
import { Button } from './ui/button'
import { useAppSettings, type AppSettings } from '../contexts/AppSettingsContext'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'
import { ApiKeyHelperRow, LtxApiKeyInput, LtxApiKeyHelperRow } from './LtxApiKeyInput'

interface TextEncoderStatus {
  downloaded: boolean
  size_gb: number
  expected_size_gb: number
}

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: TabId
}

type TabId = 'general' | 'apiKeys' | 'inference' | 'promptEnhancer' | 'about'

export function SettingsModal({ isOpen, onClose, initialTab }: SettingsModalProps) {
  const { settings, updateSettings, saveLtxApiKey, saveFalApiKey, saveGeminiApiKey, forceApiGenerations } = useAppSettings()
  const onSettingsChange = (next: AppSettings) => updateSettings(next)
  const [activeTab, setActiveTab] = useState<TabId>('general')
  const [ltxApiKeyInput, setLtxApiKeyInput] = useState('')
  const ltxApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [focusLtxApiKeyInputOnTabChange, setFocusLtxApiKeyInputOnTabChange] = useState(false)
  const [falApiKeyInput, setFalApiKeyInput] = useState('')
  const falApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [geminiApiKeyInput, setGeminiApiKeyInput] = useState('')
  const geminiApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [textEncoderStatus, setTextEncoderStatus] = useState<TextEncoderStatus | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [appVersion, setAppVersion] = useState('')
  const [noticesText, setNoticesText] = useState<string | null>(null)
  const [noticesLoading, setNoticesLoading] = useState(false)
  const [showNotices, setShowNotices] = useState(false)
  const [modelLicenseText, setModelLicenseText] = useState<string | null>(null)
  const [modelLicenseLoading, setModelLicenseLoading] = useState(false)
  const [showModelLicense, setShowModelLicense] = useState(false)
  const [analyticsEnabled, setAnalyticsEnabled] = useState(false)
  const [projectAssetsPath, setProjectAssetsPath] = useState('')
  const [fp8ExportStatus, setFp8ExportStatus] = useState<{
    status: string
    progress: number
    error: string | null
    file_exists: boolean
  } | null>(null)
  const [fp8ExportStarted, setFp8ExportStarted] = useState(false)

  // Sync active tab with initialTab prop when modal opens
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab)
    }
  }, [isOpen, initialTab])

  useEffect(() => {
    if (!isOpen || activeTab !== 'apiKeys' || !focusLtxApiKeyInputOnTabChange) return

    const frameId = window.requestAnimationFrame(() => {
      ltxApiKeyInputRef.current?.focus()
    })
    setFocusLtxApiKeyInputOnTabChange(false)

    return () => {
      window.cancelAnimationFrame(frameId)
    }
  }, [activeTab, focusLtxApiKeyInputOnTabChange, isOpen])

  // Fetch app version when About tab is shown
  useEffect(() => {
    if (activeTab !== 'about' || appVersion) return
    window.electronAPI.getAppInfo().then(info => setAppVersion(info.version)).catch(() => {})
  }, [activeTab, appVersion])

  // Fetch analytics state when modal opens
  useEffect(() => {
    if (!isOpen) return
    window.electronAPI.getAnalyticsState()
      .then((state: { analyticsEnabled: boolean }) => setAnalyticsEnabled(state.analyticsEnabled))
      .catch(() => {})
    window.electronAPI.getProjectAssetsPath()
      .then((p: string) => setProjectAssetsPath(p))
      .catch(() => {})
  }, [isOpen])

  // Fetch text encoder status when modal opens
  useEffect(() => {
    if (!isOpen) return

    const fetchStatus = async () => {
      try {
        const response = await backendFetch('/api/models/status')
        if (response.ok) {
          const data = await response.json()
          setTextEncoderStatus(data.text_encoder_status)
        }
      } catch (e) {
        logger.error(`Failed to fetch text encoder status: ${e}`)
      }
    }

    fetchStatus()
    // Poll while downloading
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [isOpen, isDownloading])

  // Poll FP8 export status when inference tab is open
  useEffect(() => {
    if (!isOpen || activeTab !== 'inference') return

    const fetchFp8Status = async () => {
      try {
        const res = await backendFetch('/api/models/export-fp8/status')
        if (res.ok) {
          const data = await res.json()
          setFp8ExportStatus(data)
          if (data.status === 'running') setFp8ExportStarted(true)
        }
      } catch { /* ignore */ }
    }

    fetchFp8Status()
    const interval = setInterval(fetchFp8Status, 2000)
    return () => clearInterval(interval)
  }, [isOpen, activeTab])

  const handleExportFp8 = async () => {
    setFp8ExportStarted(true)
    try {
      await backendFetch('/api/models/export-fp8', { method: 'POST' })
    } catch (e) {
      logger.error(`Failed to start FP8 export: ${e}`)
    }
  }

  // Handle text encoder download
  const handleDownloadTextEncoder = async () => {
    setIsDownloading(true)
    setDownloadError(null)
    try {
      const response = await backendFetch('/api/text-encoder/download', { method: 'POST' })
      const data = await response.json()

      if (data.status === 'already_downloaded') {
        setTextEncoderStatus(prev => prev ? { ...prev, downloaded: true } : null)
      }
      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await backendFetch('/api/models/status')
          if (statusRes.ok) {
            const statusData = await statusRes.json()
            setTextEncoderStatus(statusData.text_encoder_status)
            if (statusData.text_encoder_status?.downloaded) {
              setIsDownloading(false)
              clearInterval(pollInterval)
            }
          }
        } catch {
          // ignore
        }
      }, 2000)

      // Timeout after 30 minutes
      setTimeout(() => {
        clearInterval(pollInterval)
        if (isDownloading) setIsDownloading(false)
      }, 30 * 60 * 1000)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : 'Download failed')
      setIsDownloading(false)
    }
  }

  if (!isOpen) return null

  const handleToggleTorchCompile = () => {
    onSettingsChange({
      ...settings,
      useTorchCompile: !settings.useTorchCompile,
    })
  }

  const handleToggleLoadOnStartup = () => {
    onSettingsChange({
      ...settings,
      loadOnStartup: !settings.loadOnStartup,
    })
  }

  const handleToggleLocalEncoder = () => {
    onSettingsChange({
      ...settings,
      useLocalTextEncoder: !settings.useLocalTextEncoder,
    })
  }

  const openApiKeysAndFocusLtxInput = () => {
    setActiveTab('apiKeys')
    setFocusLtxApiKeyInputOnTabChange(true)
  }

  const handlePromptCacheSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const size = Math.max(0, Math.min(1000, parseInt(e.target.value) || 100))
    onSettingsChange({
      ...settings,
      promptCacheSize: size,
    })
  }

  const handleFastUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      fastModel: { ...settings.fastModel, useUpscaler: !settings.fastModel?.useUpscaler },
    })
  }

  const handleProStepsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const steps = Math.max(1, Math.min(100, parseInt(e.target.value) || 20))
    onSettingsChange({
      ...settings,
      proModel: { ...settings.proModel, steps },
    })
  }

  const handleProUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      proModel: { ...settings.proModel, useUpscaler: !settings.proModel.useUpscaler },
    })
  }

  const handleBlockSwapChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Math.max(0, Math.min(48, parseInt(e.target.value) || 0))
    onSettingsChange({ ...settings, blockSwapBlocksOnGpu: v })
  }

  const handleAttentionTileSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Math.max(0, Math.min(16384, parseInt(e.target.value) || 0))
    onSettingsChange({ ...settings, attentionTileSize: v })
  }

  const handleVaeSpatialTileSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Math.max(0, Math.min(4096, parseInt(e.target.value) || 0))
    onSettingsChange({ ...settings, vaeSpatialTileSize: v })
  }

  const handleVaeTemporalTileSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Math.max(0, Math.min(512, parseInt(e.target.value) || 0))
    onSettingsChange({ ...settings, vaeTemporalTileSize: v })
  }

  // Prompt Enhancer handlers
  const handleTogglePromptEnhancer = (mode: 't2v' | 'i2v') => {
    if (mode === 't2v') {
      onSettingsChange({ ...settings, promptEnhancerEnabledT2V: !settings.promptEnhancerEnabledT2V })
    } else {
      onSettingsChange({ ...settings, promptEnhancerEnabledI2V: !settings.promptEnhancerEnabledI2V })
    }
  }
  const handleToggleEnhancePromptLocally = () => {
    onSettingsChange({ ...settings, enhancePromptLocally: !settings.enhancePromptLocally })
  }
  // Analytics handler
  const handleToggleAnalytics = () => {
    const next = !analyticsEnabled
    setAnalyticsEnabled(next)
    window.electronAPI.setAnalyticsEnabled(next).catch(() => {})
  }

  // Seed handlers
  const handleToggleSeedLock = () => {
    onSettingsChange({
      ...settings,
      seedLocked: !settings.seedLocked,
    })
  }

  const handleLockedSeedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0
    onSettingsChange({
      ...settings,
      lockedSeed: Math.max(0, Math.min(2147483647, value)),
    })
  }

  const handleRandomizeSeed = () => {
    onSettingsChange({
      ...settings,
      lockedSeed: Math.floor(Math.random() * 2147483647),
    })
  }

  const handleLoadModelLicense = async () => {
    setModelLicenseLoading(true)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setModelLicenseText(text)
      setShowModelLicense(true)
    } catch (e) {
      logger.error(`Failed to load model license: ${e}`)
    } finally {
      setModelLicenseLoading(false)
    }
  }

  const handleLoadNotices = async () => {
    setNoticesLoading(true)
    try {
      const text = await window.electronAPI.getNoticesText()
      setNoticesText(text)
      setShowNotices(true)
    } catch (e) {
      logger.error(`Failed to load notices: ${e}`)
    } finally {
      setNoticesLoading(false)
    }
  }

  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: Settings },
    { id: 'apiKeys' as TabId, label: 'API Keys', icon: KeyRound },
    { id: 'inference' as TabId, label: 'Inference', icon: Sliders },
    { id: 'promptEnhancer' as TabId, label: 'Prompt Enhancer', icon: Sparkles },
    { id: 'about' as TabId, label: 'About', icon: Info },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-zinc-400" />
            <h2 className="text-lg font-semibold text-white">Settings</h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-white border-b-2 border-blue-500 -mb-px'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-6 h-[60vh] overflow-y-auto">
          {activeTab === 'general' && (
            <>
              {/* Project Assets Path */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Project Assets Path</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Where generated video and image assets are saved. Each project gets a subfolder.
                </p>
                <div className="flex gap-2">
                  <div className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm truncate select-text">
                    {projectAssetsPath || <span className="text-zinc-600">Not set</span>}
                  </div>
                  <Button
                    variant="outline"
                    className="border-zinc-700 flex-shrink-0"
                    onClick={async () => {
                      const result = await window.electronAPI.openProjectAssetsPathChangeDialog()
                      if (result.success && result.path) {
                        setProjectAssetsPath(result.path)
                      }
                    }}
                  >
                    <Folder className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {!forceApiGenerations && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Film className="h-4 w-4 text-blue-400" />
                    <h3 className="text-sm font-semibold text-white">Videos Generation</h3>
                  </div>

                  <div
                    className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                      settings.userPrefersLtxApiVideoGenerations ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                    }`}
                    onClick={() => {
                      if (!settings.hasLtxApiKey) {
                        openApiKeysAndFocusLtxInput()
                        return
                      }
                      onSettingsChange({
                        ...settings,
                        userPrefersLtxApiVideoGenerations: !settings.userPrefersLtxApiVideoGenerations,
                      })
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-blue-400" />
                          <span className="text-sm font-medium text-white">Generate With API</span>
                        </div>
                        <p className="text-xs text-zinc-400 mt-1">
                          Use LTX API for video generation when an LTX API key is configured.
                        </p>
                      </div>
                      <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                        settings.userPrefersLtxApiVideoGenerations ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                      }`}>
                        {settings.userPrefersLtxApiVideoGenerations && <Check className="h-3 w-3 text-white" />}
                      </div>
                    </div>

                    {!settings.hasLtxApiKey && (
                      <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                        <AlertCircle className="h-3 w-3" />
                        API key required — configure it in the API Keys tab.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Text Encoding Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 7h3a5 5 0 0 1 5 5 5 5 0 0 1-5 5h-3m-6 0H6a5 5 0 0 1-5-5 5 5 0 0 1 5-5h3" />
                    <line x1="8" y1="12" x2="16" y2="12" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">Text Encoding</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Text encoding converts your prompt into data the AI understands. Choose how to do this.
                </p>

                {/* LTX API Option (Default) */}
                <div
                  className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                    !settings.useLocalTextEncoder ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                  }`}
                  onClick={() => {
                    if (!settings.useLocalTextEncoder) return
                    if (!settings.hasLtxApiKey) {
                      openApiKeysAndFocusLtxInput()
                      return
                    }
                    handleToggleLocalEncoder()
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Zap className="h-4 w-4 text-blue-400" />
                        <span className="text-sm font-medium text-white">LTX API</span>
                        <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">Recommended</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">
                        Fast cloud-based text encoding (~1 second). Requires an LTX API key configured in the API Keys tab.
                      </p>
                    </div>
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      !settings.useLocalTextEncoder ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                    }`}>
                      {!settings.useLocalTextEncoder && <Check className="h-3 w-3 text-white" />}
                    </div>
                  </div>

                  {/* Warning when selected but no key */}
                  {!settings.useLocalTextEncoder && !settings.hasLtxApiKey && (
                    <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                      <AlertCircle className="h-3 w-3" />
                      API key required — configure it in the API Keys tab.
                    </div>
                  )}

                  {/* Prompt Cache Size — only relevant for API text encoding */}
                  {!settings.useLocalTextEncoder && settings.hasLtxApiKey && (
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-700/50">
                      <div>
                        <label className="text-xs text-white">Prompt Cache</label>
                        <p className="text-xs text-zinc-500">Skip repeat encoding calls</p>
                      </div>
                      <input
                        type="number"
                        min="0"
                        max="1000"
                        value={settings.promptCacheSize ?? 100}
                        onChange={handlePromptCacheSizeChange}
                        onClick={(e) => e.stopPropagation()}
                        className="w-16 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  )}
                </div>

                {/* Local Encoder Option */}
                <div
                  className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                    settings.useLocalTextEncoder ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                  }`}
                  onClick={() => !settings.useLocalTextEncoder && handleToggleLocalEncoder()}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <svg className="h-4 w-4 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="4" y="4" width="16" height="16" rx="2" />
                          <path d="M9 9h6m-6 3h6m-6 3h4" />
                        </svg>
                        <span className="text-sm font-medium text-white">Local Encoder</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">
                        Run on your computer (~23 seconds). Requires 25 GB download.
                      </p>
                    </div>
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      settings.useLocalTextEncoder ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                    }`}>
                      {settings.useLocalTextEncoder && <Check className="h-3 w-3 text-white" />}
                    </div>
                  </div>

                  {/* Download Status - show when this option is selected */}
                  {settings.useLocalTextEncoder && (
                    <div className="mt-3 pt-3 border-t border-zinc-700/50">
                      {textEncoderStatus?.downloaded ? (
                        <div className="flex items-center gap-2 text-xs text-green-400">
                          <Check className="h-4 w-4" />
                          <span>Downloaded ({textEncoderStatus.size_gb} GB)</span>
                        </div>
                      ) : isDownloading ? (
                        <div className="flex items-center gap-2 text-xs text-blue-400">
                          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                          <span>Downloading text encoder...</span>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs text-amber-400">
                            <AlertCircle className="h-4 w-4" />
                            <span>Not downloaded ({textEncoderStatus?.expected_size_gb || 8} GB required)</span>
                          </div>
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDownloadTextEncoder()
                            }}
                            className="w-full bg-blue-600 hover:bg-blue-500 text-white text-xs"
                          >
                            <Download className="h-3 w-3 mr-2" />
                            Download Text Encoder
                          </Button>
                          {downloadError && (
                            <p className="text-xs text-red-400">{downloadError}</p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Load on Startup Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Preload models on startup
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Load AI models in the background after the app starts. The video model is loaded
                      and warmed up on GPU, and the image model is preloaded into CPU RAM for faster
                      first generation. When disabled, models load on first use (faster startup, slower
                      first generation). Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleLoadOnStartup}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.loadOnStartup ? 'bg-blue-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.loadOnStartup ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.loadOnStartup
                    ? 'bg-blue-500/10 text-blue-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.loadOnStartup ? 'bg-blue-400' : 'bg-zinc-600'
                  }`} />
                  {settings.loadOnStartup ? 'Models preload in background at startup' : 'Models load on first generation'}
                </div>
              </div>

              {/* Torch Compile Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Torch Compile
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Compiles the model for optimized inference. <span className="text-orange-400">Experimental:</span> First
                      generation can take 5-10+ minutes for compilation. Subsequent generations may be
                      20-40% faster. Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleTorchCompile}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.useTorchCompile ? 'bg-orange-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useTorchCompile ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.useTorchCompile
                    ? 'bg-orange-500/10 text-orange-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.useTorchCompile ? 'bg-orange-400' : 'bg-zinc-600'
                  }`} />
                  {settings.useTorchCompile ? 'Optimized inference (recommended)' : 'Standard inference'}
                </div>
              </div>

              {/* Seed Lock Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Lock Seed
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Use the same seed for reproducible generations. When unlocked, a random seed is used each time.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleSeedLock}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.seedLocked ? 'bg-emerald-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.seedLocked ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Seed input - only show when locked */}
                {settings.seedLocked && (
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      value={settings.lockedSeed ?? 42}
                      onChange={handleLockedSeedChange}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="Enter seed..."
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRandomizeSeed}
                      className="h-9 px-3 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                      title="Generate random seed"
                    >
                      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
                      </svg>
                    </Button>
                  </div>
                )}

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.seedLocked
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.seedLocked ? 'bg-emerald-400' : 'bg-zinc-600'
                  }`} />
                  {settings.seedLocked ? `Seed locked: ${settings.lockedSeed ?? 42}` : 'Random seed each generation'}
                </div>
              </div>

              {/* Anonymous Analytics Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10" />
                        <line x1="12" y1="20" x2="12" y2="4" />
                        <line x1="6" y1="20" x2="6" y2="14" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Anonymous Analytics
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Share anonymous usage data to help improve LTX Desktop.
                      Only basic technical information is collected — never personal data or generated content.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleAnalytics}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      analyticsEnabled ? 'bg-violet-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        analyticsEnabled ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

              </div>
            </>
          )}

          {activeTab === 'apiKeys' && (
            <>
              {/* LTX API Key Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">LTX API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your LTX API key is used for cloud text encoding, prompt enhancement, and Pro generation.
                  Add your key below to unlock these features.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={ltxApiKeyInputRef}
                      value={ltxApiKeyInput}
                      onChange={(e) => setLtxApiKeyInput(e.target.value)}
                      placeholder={settings.hasLtxApiKey ? 'Enter new key to replace...' : 'Enter your LTX API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = ltxApiKeyInput.trim()
                        if (!trimmed) return
                        void saveLtxApiKey(trimmed)
                        setLtxApiKeyInput('')
                      }}
                      disabled={!ltxApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <LtxApiKeyHelperRow stopPropagation />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasLtxApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasLtxApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* FAL API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <KeyRound className="h-4 w-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">FAL AI</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">Optional</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your FAL AI key is used for generating images with Z Image Turbo when API generations are enabled.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={falApiKeyInputRef}
                      value={falApiKeyInput}
                      onChange={(e) => setFalApiKeyInput(e.target.value)}
                      placeholder={settings.hasFalApiKey ? 'Enter new key to replace...' : 'Enter your FAL AI API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = falApiKeyInput.trim()
                        if (!trimmed) return
                        void saveFalApiKey(trimmed)
                        setFalApiKeyInput('')
                      }}
                      disabled={!falApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <ApiKeyHelperRow
                    stopPropagation
                    label="Get FAL API key"
                    onOpenKey={() => window.electronAPI.openFalApiKeyPage()}
                  />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasFalApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {settings.hasFalApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          Optional
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Gemini API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Gemini API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your Gemini API key is used for AI-powered prompt suggestions when filling timeline gaps.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <input
                      ref={geminiApiKeyInputRef}
                      type="password"
                      value={geminiApiKeyInput}
                      onChange={(e) => setGeminiApiKeyInput(e.target.value)}
                      placeholder={settings.hasGeminiApiKey ? 'Enter new key to replace...' : 'Enter your Gemini API key...'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = geminiApiKeyInput.trim()
                        if (!trimmed) return
                        void saveGeminiApiKey(trimmed)
                        setGeminiApiKeyInput('')
                      }}
                      disabled={!geminiApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasGeminiApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasGeminiApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <a
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Get Gemini API key →
                    </a>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeTab === 'inference' && (
            <>
              {/* Fast Model Settings */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-400" />
                  <h3 className="text-sm font-semibold text-white">Fast Model (Distilled)</h3>
                </div>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  {/* Steps */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Inference Steps</label>
                      <p className="text-xs text-zinc-500">Fewer = faster, lower quality. 8 = full quality</p>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="8"
                      value={settings.distilledNumSteps ?? 8}
                      onChange={(e) => {
                        const v = Math.max(1, Math.min(8, parseInt(e.target.value) || 8))
                        onSettingsChange({ ...settings, distilledNumSteps: v })
                      }}
                      className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                  </div>

                  {/* STG Scale */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">STG Scale</label>
                      <p className="text-xs text-zinc-500">0 = off. Improves prompt adherence (~2× slower). Try 0.5–1.5</p>
                    </div>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      value={settings.stgScale ?? 0}
                      onChange={(e) => {
                        const v = Math.max(0, Math.min(10, parseFloat(e.target.value) || 0))
                        onSettingsChange({ ...settings, stgScale: v })
                      }}
                      className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                  </div>

                  {/* STG Block Index — only show when STG is active */}
                  {(settings.stgScale ?? 0) > 0 && (
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm text-zinc-300">STG Block Index</label>
                        <p className="text-xs text-zinc-500">Transformer block to perturb (0–27). Default 19 works well</p>
                      </div>
                      <input
                        type="number"
                        min="0"
                        max="27"
                        value={settings.stgBlockIndex ?? 19}
                        onChange={(e) => {
                          const v = Math.max(0, Math.min(27, parseInt(e.target.value) || 19))
                          onSettingsChange({ ...settings, stgBlockIndex: v })
                        }}
                        className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-green-500"
                      />
                    </div>
                  )}

                  {/* Upscaler Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">2x Upscaler</label>
                      <p className="text-xs text-zinc-500">When off, generates at native resolution</p>
                    </div>
                    <button
                      onClick={handleFastUpscalerToggle}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.fastModel?.useUpscaler !== false ? 'bg-green-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          settings.fastModel?.useUpscaler !== false ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Summary */}
                <div className="text-xs text-zinc-500">
                  Current: {settings.distilledNumSteps ?? 8} steps
                  {(settings.stgScale ?? 0) > 0 ? `, STG ${settings.stgScale} (block ${settings.stgBlockIndex ?? 19})` : ''}
                  {', '}{settings.fastModel?.useUpscaler !== false ? 'with upscaler (2-stage, recommended)' : 'native resolution (experimental)'}
                </div>
              </div>

              {/* Pro Model Settings */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">Pro Model (Full)</h3>
                </div>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  {/* Steps */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Inference Steps</label>
                      <p className="text-xs text-zinc-500">More steps = better quality, slower</p>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={settings.proModel?.steps ?? 20}
                      onChange={handleProStepsChange}
                      className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  {/* Upscaler Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">2x Upscaler</label>
                      <p className="text-xs text-zinc-500">Doubles resolution in second pass</p>
                    </div>
                    <button
                      onClick={handleProUpscalerToggle}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.proModel?.useUpscaler !== false ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          settings.proModel?.useUpscaler !== false ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Summary */}
                <div className="text-xs text-zinc-500">
                  Current: {settings.proModel?.steps ?? 20} steps, {settings.proModel?.useUpscaler !== false ? 'with upscaler (2-stage, recommended)' : 'native resolution'}
                </div>
              </div>

              {/* Info Box */}
              <div className="bg-zinc-800/30 rounded-lg p-3 mt-4">
                <p className="text-xs text-zinc-400">
                  <span className="text-blue-400 font-medium">Tip:</span> Lower steps = faster but lower quality.
                  Higher steps = better quality but slower.
                </p>
              </div>

              {/* VRAM Optimisations */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="2" y="7" width="20" height="10" rx="2" />
                    <path d="M6 7V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v2" />
                    <circle cx="12" cy="12" r="1" fill="currentColor" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">VRAM Optimisations</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">Advanced</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Settings to reduce VRAM usage and enable longer generations. Requires app restart to take effect.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  {/* Multi-GPU */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Multi-GPU</label>
                      <p className="text-xs text-zinc-500">Transformer on cuda:0, text encoder on cuda:1</p>
                    </div>
                    <button
                      onClick={() => onSettingsChange({ ...settings, useMultiGpu: !settings.useMultiGpu })}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.useMultiGpu ? 'bg-amber-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useMultiGpu ? 'translate-x-5' : 'translate-x-0'
                      }`} />
                    </button>
                  </div>

                  {/* FP8 Transformer */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">FP8 Transformer</label>
                      <p className="text-xs text-zinc-500">Force FP8 quantisation (auto-enabled on supported GPUs)</p>
                    </div>
                    <button
                      onClick={() => onSettingsChange({ ...settings, useFp8Transformer: !settings.useFp8Transformer })}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.useFp8Transformer ? 'bg-amber-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useFp8Transformer ? 'translate-x-5' : 'translate-x-0'
                      }`} />
                    </button>
                  </div>

                  {/* Block Swap */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Block Swap (blocks on GPU)</label>
                      <p className="text-xs text-zinc-500">0 = disabled, max 48. Offloads transformer blocks to CPU RAM</p>
                    </div>
                    <input
                      type="number"
                      min="0"
                      max="48"
                      value={settings.blockSwapBlocksOnGpu ?? 0}
                      onChange={handleBlockSwapChange}
                      className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                    />
                  </div>

                  {/* Attention Tiling */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Attention Tile Size</label>
                      <p className="text-xs text-zinc-500">0 = disabled. Enables long videos (try 512–2048). Lower = less VRAM</p>
                    </div>
                    <input
                      type="number"
                      min="0"
                      max="16384"
                      step="64"
                      value={settings.attentionTileSize ?? 0}
                      onChange={handleAttentionTileSizeChange}
                      className="w-24 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                    />
                  </div>

                  {/* Abliterated Encoder */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Abliterated Text Encoder</label>
                      <p className="text-xs text-zinc-500">Use refusal-direction-removed Gemma encoder</p>
                    </div>
                    <button
                      onClick={() => onSettingsChange({ ...settings, useAbliteratedEncoder: !settings.useAbliteratedEncoder })}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.useAbliteratedEncoder ? 'bg-amber-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useAbliteratedEncoder ? 'translate-x-5' : 'translate-x-0'
                      }`} />
                    </button>
                  </div>

                  {/* GGUF Path */}
                  <div className="space-y-1.5">
                    <label className="text-sm text-white">GGUF Transformer Path</label>
                    <p className="text-xs text-zinc-500">Optional path to a GGUF transformer file (leave blank to use default)</p>
                    <input
                      type="text"
                      value={settings.ggufTransformerPath ?? ''}
                      onChange={(e) => onSettingsChange({ ...settings, ggufTransformerPath: e.target.value })}
                      onKeyDown={(e) => e.stopPropagation()}
                      placeholder="e.g. C:\models\transformer.gguf"
                      className="w-full px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-xs text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                    />
                  </div>

                  {/* FP8 Pre-quantized Export */}
                  <div className="space-y-2 pt-2 border-t border-zinc-700">
                    <div>
                      <label className="text-sm text-white">Pre-quantized FP8 Export</label>
                      <p className="text-xs text-zinc-500">
                        Saves a ~3 GB pre-quantized file so FP8 conversion is skipped on every launch.
                        {fp8ExportStatus?.file_exists && ' File already exists — re-export to refresh.'}
                      </p>
                    </div>
                    <div className="flex items-center gap-3 flex-wrap">
                      <Button
                        variant="outline"
                        className="border-zinc-600 text-xs h-7 px-3"
                        onClick={handleExportFp8}
                        disabled={fp8ExportStatus?.status === 'running'}
                      >
                        {fp8ExportStatus?.status === 'running'
                          ? `Exporting… ${Math.round((fp8ExportStatus.progress ?? 0) * 100)}%`
                          : fp8ExportStatus?.file_exists
                          ? 'Re-export FP8'
                          : 'Export FP8'}
                      </Button>
                      {fp8ExportStatus?.file_exists && fp8ExportStatus.status !== 'running' && (
                        <span className="text-xs text-emerald-400 flex items-center gap-1">
                          <Check className="h-3 w-3" /> Cached
                        </span>
                      )}
                      {fp8ExportStatus?.status === 'done' && fp8ExportStarted && (
                        <span className="text-xs text-emerald-400">Export complete — restart to use</span>
                      )}
                      {fp8ExportStatus?.status === 'error' && (
                        <span className="text-xs text-red-400">{fp8ExportStatus.error}</span>
                      )}
                    </div>
                  </div>

                  {/* VAE Tiling */}
                  <div className="space-y-2 pt-2 border-t border-zinc-700">
                    <div>
                      <label className="text-sm text-white">VAE Tiling</label>
                      <p className="text-xs text-zinc-500">Reduce VRAM during decode. 0 = library defaults (512px spatial, 64 frames temporal)</p>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <label className="text-xs text-zinc-400">Spatial tile (px)</label>
                        <input
                          type="number"
                          min="0"
                          max="4096"
                          step="32"
                          value={settings.vaeSpatialTileSize ?? 0}
                          onChange={handleVaeSpatialTileSizeChange}
                          className="w-full px-2 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                        />
                      </div>
                      <div className="space-y-1">
                        <label className="text-xs text-zinc-400">Temporal tile (frames)</label>
                        <input
                          type="number"
                          min="0"
                          max="512"
                          step="8"
                          value={settings.vaeTemporalTileSize ?? 0}
                          onChange={handleVaeTemporalTileSizeChange}
                          className="w-full px-2 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Unload text encoder after encode */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-xs text-zinc-300">Unload Text Encoder After Encode</label>
                      <p className="text-xs text-zinc-500">Free ~9GB RAM after encoding (single-GPU only). Slower between generations.</p>
                    </div>
                    <button
                      onClick={() => onSettingsChange({ ...settings, unloadTextEncoderAfterEncode: !settings.unloadTextEncoderAfterEncode })}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${settings.unloadTextEncoderAfterEncode ? 'bg-amber-500' : 'bg-zinc-600'}`}
                    >
                      <span className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${settings.unloadTextEncoderAfterEncode ? 'translate-x-4' : 'translate-x-1'}`} />
                    </button>
                  </div>

                  {/* Pipeline reload interval (OOM prevention) */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-xs text-zinc-300">Reload Pipeline Every N Gens</label>
                      <p className="text-xs text-zinc-500">0 = disabled. Force VRAM defrag after N generations to prevent OOM crashes (try 4–8)</p>
                    </div>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={settings.reloadPipelineEveryNGens ?? 0}
                      onChange={(e) => {
                        const v = Math.max(0, Math.min(100, parseInt(e.target.value) || 0))
                        onSettingsChange({ ...settings, reloadPipelineEveryNGens: v })
                      }}
                      className="w-20 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                    />
                  </div>
                </div>

                {/* Device overrides */}
                <div className="bg-zinc-800/30 rounded-lg p-3 space-y-3">
                  <p className="text-xs text-zinc-400 font-medium">Device Overrides <span className="text-zinc-600 font-normal">(expert, blank = auto)</span></p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                      <label className="text-xs text-zinc-400">Transformer Device</label>
                      <input
                        type="text"
                        value={settings.transformerDevice ?? ''}
                        onChange={(e) => onSettingsChange({ ...settings, transformerDevice: e.target.value })}
                        onKeyDown={(e) => e.stopPropagation()}
                        placeholder="cuda:0"
                        className="w-full px-2 py-1 bg-zinc-700 border border-zinc-700 rounded text-xs text-white placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-xs text-zinc-400">Text Encoder Device</label>
                      <input
                        type="text"
                        value={settings.textEncoderDevice ?? ''}
                        onChange={(e) => onSettingsChange({ ...settings, textEncoderDevice: e.target.value })}
                        onKeyDown={(e) => e.stopPropagation()}
                        placeholder="cuda:1"
                        className="w-full px-2 py-1 bg-zinc-700 border border-zinc-700 rounded text-xs text-white placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* LoRA Manager */}
              <LoraManager
                value={settings.civitaiLoras ?? '[]'}
                onChange={(v) => onSettingsChange({ ...settings, civitaiLoras: v })}
              />
            </>
          )}

          {activeTab === 'promptEnhancer' && (
            <>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Prompt Enhancer</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Automatically enhances your prompts with rich visual details, sound descriptions,
                  and motion cues to help generate higher quality videos.
                </p>

                {/* Local enhancement toggle — uses the already-loaded Gemma model */}
                <div
                  className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                  onClick={handleToggleEnhancePromptLocally}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-semibold text-purple-400 bg-purple-400/10 px-1.5 py-0.5 rounded">LOCAL</span>
                    <div>
                      <span className="text-sm text-zinc-200">On-device enhancement</span>
                      <p className="text-[10px] text-zinc-500 mt-0.5">
                        {settings.enhancePromptLocally
                          ? 'Gemma expands prompts locally before encoding — no API needed'
                          : 'Prompts sent to encoder as-is'}
                      </p>
                    </div>
                  </div>
                  <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                    settings.enhancePromptLocally ? 'bg-purple-500' : 'bg-zinc-700'
                  }`}>
                    <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                      settings.enhancePromptLocally ? 'translate-x-5' : 'translate-x-0'
                    }`} />
                  </div>
                </div>

                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-px bg-zinc-700/50" />
                  <span className="text-[10px] text-zinc-600">or via LTX API</span>
                  <div className="flex-1 h-px bg-zinc-700/50" />
                </div>

                {!settings.hasLtxApiKey ? (
                  <div className="space-y-4 mt-2">
                    <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4 space-y-3">
                      <div className="flex items-start gap-2.5">
                        <AlertCircle className="h-4 w-4 text-amber-400 mt-0.5 flex-shrink-0" />
                        <div className="space-y-2">
                          <p className="text-sm text-amber-300 font-medium">LTX API key required</p>
                          <p className="text-xs text-zinc-400 leading-relaxed">
                            Prompt enhancement runs server-side on the LTX API. To use this feature, you need to configure
                            an API key in the API Keys tab.
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab('apiKeys')}
                        className="w-full mt-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        Set API Key
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* T2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('t2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded">T2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Text-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledT2V ? 'Prompts will be enhanced before T2V generation' : 'T2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledT2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledT2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>

                    {/* I2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('i2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">I2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Image-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledI2V ? 'Prompts will be enhanced before I2V generation' : 'I2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledI2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledI2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab === 'about' && (
            <>
              {showModelLicense ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">LTX-2 Model License</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowModelLicense(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {modelLicenseText}
                  </pre>
                </div>
              ) : showNotices ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">Third-Party Notices</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowNotices(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {noticesText}
                  </pre>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* App Identity */}
                  <div className="text-center space-y-2">
                    <h3 className="text-lg font-bold text-white">LTX Desktop</h3>
                    <p className="text-sm text-zinc-400">Version {appVersion || '...'}</p>
                    <p className="text-xs text-zinc-500">AI-Powered Video Editor</p>
                  </div>

                  {/* License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-blue-400" />
                      <span className="text-sm font-medium text-white">License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      Licensed under the Apache License, Version 2.0
                    </p>
                  </div>

                  {/* LTX-2 Model License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                      </svg>
                      <span className="text-sm font-medium text-white">LTX-2 Model License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      The LTX-2 model is subject to the LTX-2 Community License Agreement, accepted during first-run setup.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadModelLicense}
                      disabled={modelLicenseLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {modelLicenseLoading ? 'Loading...' : 'View Model License'}
                    </Button>
                  </div>

                  {/* Third-Party Notices */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                      </svg>
                      <span className="text-sm font-medium text-white">Third-Party Notices</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      This application uses open-source software and AI models subject to their own license terms.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadNotices}
                      disabled={noticesLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {noticesLoading ? 'Loading...' : 'View Third-Party Notices'}
                    </Button>
                  </div>

                  {/* Copyright */}
                  <p className="text-center text-xs text-zinc-600">
                    Copyright © 2026 Lightricks
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-zinc-800 flex justify-end">
          <Button
            onClick={onClose}
            className="bg-zinc-700 hover:bg-zinc-600 text-white"
          >
            Done
          </Button>
        </div>
      </div>
    </div>
  )
}

export type { AppSettings, TabId as SettingsTabId }

// ── LoRA Manager ─────────────────────────────────────────────────────────────

interface LoraEntry {
  path: string
  strength: number
  enabled: boolean
}

function parseLoras(json: string): LoraEntry[] {
  try { return JSON.parse(json) as LoraEntry[] } catch { return [] }
}

function basename(p: string): string {
  return p.replace(/\\/g, '/').split('/').pop() ?? p
}

function LoraManager({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const loras = parseLoras(value)

  const save = (next: LoraEntry[]) => onChange(JSON.stringify(next))

  const handleAdd = async () => {
    const files = await window.electronAPI.showOpenFileDialog({
      title: 'Select LoRA (.safetensors)',
      filters: [{ name: 'LoRA weights', extensions: ['safetensors'] }],
      properties: ['openFile', 'multiSelections'],
    })
    if (!files || files.length === 0) return
    const newEntries: LoraEntry[] = files
      .filter(f => !loras.some(l => l.path === f))
      .map(f => ({ path: f, strength: 1.0, enabled: true }))
    save([...loras, ...newEntries])
  }

  const handleRemove = (i: number) => save(loras.filter((_, idx) => idx !== i))

  const handleToggle = (i: number) =>
    save(loras.map((l, idx) => idx === i ? { ...l, enabled: !l.enabled } : l))

  const handleStrength = (i: number, v: number) =>
    save(loras.map((l, idx) => idx === i ? { ...l, strength: v } : l))

  return (
    <div className="space-y-3 pt-4 border-t border-zinc-800">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg className="h-4 w-4 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
          <h3 className="text-sm font-semibold text-white">LoRA Manager</h3>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-400">Experimental</span>
        </div>
        <button
          onClick={() => { void handleAdd() }}
          className="flex items-center gap-1 px-2 py-1 text-xs text-violet-300 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-700/40 rounded-lg transition-colors"
        >
          <Plus size={11} />
          Add LoRA
        </button>
      </div>

      <p className="text-xs text-zinc-500">
        Load LTX-Video compatible LoRAs (.safetensors). CivitAI and diffusers formats are auto-detected.
        Changes take effect on next pipeline load (restart required).
      </p>

      <div className="space-y-2">
        {loras.length === 0 ? (
          <p className="text-xs text-zinc-600 italic px-1">No LoRAs added. Click "Add LoRA" to pick a .safetensors file.</p>
        ) : loras.map((lora, i) => (
          <div key={i} className={`rounded-lg p-3 space-y-2 border transition-colors ${lora.enabled ? 'bg-zinc-800/60 border-zinc-700/50' : 'bg-zinc-900/40 border-zinc-800/30 opacity-60'}`}>
            <div className="flex items-center gap-2">
              {/* Enable toggle */}
              <button
                onClick={() => handleToggle(i)}
                className={`relative inline-flex h-4 w-7 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${lora.enabled ? 'bg-violet-500' : 'bg-zinc-600'}`}
                title={lora.enabled ? 'Disable LoRA' : 'Enable LoRA'}
              >
                <span className={`inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform ${lora.enabled ? 'translate-x-3' : 'translate-x-0'}`} />
              </button>
              {/* Filename */}
              <span className="flex-1 text-xs text-zinc-300 truncate" title={lora.path}>
                {basename(lora.path)}
              </span>
              {/* Remove */}
              <button
                onClick={() => handleRemove(i)}
                className="text-zinc-600 hover:text-red-400 transition-colors flex-shrink-0"
                title="Remove LoRA"
              >
                <X size={13} />
              </button>
            </div>
            {/* Strength slider */}
            <div className="flex items-center gap-2 pl-9">
              <label className="text-[10px] text-zinc-500 w-12 flex-shrink-0">Strength</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={lora.strength}
                onChange={(e) => handleStrength(i, parseFloat(e.target.value))}
                disabled={!lora.enabled}
                className="flex-1 accent-violet-500 disabled:opacity-40"
              />
              <span className="text-[10px] text-zinc-400 w-8 text-right flex-shrink-0">
                {lora.strength.toFixed(2)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
