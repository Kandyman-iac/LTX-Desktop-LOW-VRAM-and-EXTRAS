import { useState, useRef, useEffect, useMemo } from 'react'
import { Sparkles, Trash2, Square, ImageIcon, ArrowLeft, Scissors, History, X, Film, CheckCircle, Cpu, AlertCircle } from 'lucide-react'
import { logger } from '../lib/logger'
import { ImageUploader } from '../components/ImageUploader'
import { AudioUploader } from '../components/AudioUploader'
import { VideoPlayer } from '../components/VideoPlayer'
import { ImageResult } from '../components/ImageResult'
import { SettingsPanel, type GenerationSettings } from '../components/SettingsPanel'
import { ModeTabs, type GenerationMode } from '../components/ModeTabs'
import { LtxLogo } from '../components/LtxLogo'
import { ModelStatusDropdown } from '../components/ModelStatusDropdown'
import { Textarea } from '../components/ui/textarea'
import { Button } from '../components/ui/button'
import { useGeneration } from '../hooks/use-generation'
import { useRetake } from '../hooks/use-retake'
import { useIcLora } from '../hooks/use-ic-lora'
import { useBackend } from '../hooks/use-backend'
import { useProjects } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { fileUrlToPath } from '../lib/url-to-path'
import { sanitizeForcedApiVideoSettings } from '../lib/api-video-options'
import { RetakePanel } from '../components/RetakePanel'
import { ICLoraPanel, CONDITIONING_TYPES, type ICLoraConditioningType } from '../components/ICLoraPanel'
import { useGenerationHistory, type HistoryEntry } from '../hooks/use-generation-history'
import { useEncodePrompt } from '../hooks/use-encode-prompt'
import { OutputBrowser } from '../components/OutputBrowser'

const DEFAULT_SETTINGS: GenerationSettings = {
  model: 'fast',
  duration: 5,
  videoResolution: '540p',
  fps: 24,
  audio: true,
  cameraMotion: 'none',
  aspectRatio: '16:9',
  seed: null,
  // Image settings
  imageResolution: '1080p',
  imageAspectRatio: '16:9',
  imageSteps: 4,
}

export function Playground() {
  const { goHome } = useProjects()
  const { forceApiGenerations, shouldVideoGenerateWithLtxApi, settings: appSettings } = useAppSettings()
  const [mode, setMode] = useState<GenerationMode>('text-to-video')
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')
  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedAudio, setSelectedAudio] = useState<string | null>(null)
  const [settings, setSettings] = useState<GenerationSettings>(() => ({ ...DEFAULT_SETTINGS }))
  const [showHistory, setShowHistory] = useState(false)
  const [showOutputBrowser, setShowOutputBrowser] = useState(false)
  const { push: pushHistory, getAll: getHistory, remove: removeHistory } = useGenerationHistory()
  const historyEntries = useMemo(() => (showHistory ? getHistory() : []), [showHistory, getHistory])

  const { status, processStatus } = useBackend()
  const { isEncoding, encodedPrompt, encodeError, encodePrompt, clearEncoded } = useEncodePrompt()

  // Determine whether the manual Encode Prompt workflow is relevant.
  // Shown for any local text encoder setup (single-GPU or multi-GPU).
  const showEncodeButton = appSettings.useLocalTextEncoder

  // Track whether the current prompt differs from the last encoded one.
  const isPromptChanged = showEncodeButton && encodedPrompt !== null && prompt.trim() !== encodedPrompt

  // Editable copy of the Gemma-enhanced prompt (populated after generation).
  const [editableEnhancedPrompt, setEditableEnhancedPrompt] = useState<string | null>(null)

  useEffect(() => {
    if (!shouldVideoGenerateWithLtxApi || mode === 'text-to-image') return
    setSettings((prev) => sanitizeForcedApiVideoSettings({ ...prev, model: 'fast' }))
  }, [mode, shouldVideoGenerateWithLtxApi])

  useEffect(() => {
    if (forceApiGenerations && mode === 'ic-lora') {
      setMode('text-to-video')
    }
  }, [forceApiGenerations, mode])

  // Force pro model + resolution when audio is attached (A2V only supports pro @ 1080p 16:9)
  useEffect(() => {
    if (selectedAudio && mode !== 'text-to-image') {
      setSettings(prev => {
        if (shouldVideoGenerateWithLtxApi) {
          return sanitizeForcedApiVideoSettings({ ...prev, model: 'pro' }, { hasAudio: true })
        }
        return prev.model !== 'pro' ? { ...prev, model: 'pro' } : prev
      })
    }
  }, [mode, selectedAudio, shouldVideoGenerateWithLtxApi]) // eslint-disable-line react-hooks/exhaustive-deps

  // Handle mode change
  const handleModeChange = (newMode: GenerationMode) => {
    setMode(newMode)
  }
  const {
    isGenerating,
    progress,
    statusMessage,
    videoUrl,
    videoPath,
    imageUrl,
    enhancedPrompt,
    error: generationError,
    generate,
    generateImage,
    cancel,
    reset,
  } = useGeneration({
    onGenerationSuccess: (info) => {
      pushHistory({
        prompt: info.prompt,
        negativePrompt: info.negativePrompt,
        settings: info.settings,
        seedUsed: info.seedUsed,
        videoPath: info.videoPath,
      })
      // Show the actual seed used so user can lock it to repeat the generation
      if (info.seedUsed != null && info.settings.seed == null) {
        setSettings(prev => ({ ...prev, seed: info.seedUsed }))
      }
    },
  })

  // Sync editableEnhancedPrompt whenever a new enhanced prompt arrives from generation.
  useEffect(() => {
    if (enhancedPrompt !== null) setEditableEnhancedPrompt(enhancedPrompt)
  }, [enhancedPrompt])

  const {
    submitRetake,
    resetRetake,
    isRetaking,
    retakeStatus,
    retakeError,
    retakeResult,
  } = useRetake()

  const {
    submitIcLora,
    resetIcLora,
    isIcLoraGenerating,
    icLoraStatus,
    icLoraError,
    icLoraResult,
  } = useIcLora()

  const [retakeInput, setRetakeInput] = useState({
    videoUrl: null as string | null,
    videoPath: null as string | null,
    startTime: 0,
    duration: 0,
    videoDuration: 0,
    ready: false,
  })
  const [retakePanelKey, setRetakePanelKey] = useState(0)
  const [icLoraInput, setIcLoraInput] = useState({
    videoUrl: null as string | null,
    videoPath: null as string | null,
    conditioningType: 'canny' as 'canny' | 'depth',
    conditioningStrength: 1.0,
    ready: false,
  })
  const [icLoraPanelKey, setIcLoraPanelKey] = useState(0)
  const [icLoraCondType, setIcLoraCondType] = useState<ICLoraConditioningType>('canny')
  const [icLoraStrength, setIcLoraStrength] = useState(1.0)

  // Ref to store generated image URL for "Create video" flow
  const generatedImageRef = useRef<string | null>(null)

  const handleGenerate = () => {
    if (mode === 'ic-lora') {
      if (!icLoraInput.videoPath || !icLoraInput.ready || !prompt.trim()) return
      submitIcLora({
        videoPath: icLoraInput.videoPath,
        conditioningType: icLoraCondType,
        conditioningStrength: icLoraStrength,
        prompt,
      })
      return
    }

    if (mode === 'retake') {
      if (!retakeInput.videoPath || retakeInput.duration < 2) return
      submitRetake({
        videoPath: retakeInput.videoPath,
        startTime: retakeInput.startTime,
        duration: retakeInput.duration,
        prompt,
        mode: 'replace_audio_and_video',
      })
      return
    }

    if (mode === 'text-to-image') {
      if (!prompt.trim()) return
      // Text-to-image behavior remains tied to raw forceApiGenerations in useGeneration.
      generateImage(prompt, settings)
    } else {
      const effectiveVideoSettings = shouldVideoGenerateWithLtxApi
        ? sanitizeForcedApiVideoSettings(settings)
        : settings
      // Auto-detect: if image is loaded → I2V, otherwise → T2V
      if (!prompt.trim()) return
      const imagePath = selectedImage ? fileUrlToPath(selectedImage) : null
      const audioPath = selectedAudio ? fileUrlToPath(selectedAudio) : null
      if (audioPath) effectiveVideoSettings.model = 'pro'
      generate(prompt, imagePath, effectiveVideoSettings, audioPath, negativePrompt)
    }
  }
  
  // Handle "Create video" from generated image
  const handleCreateVideoFromImage = () => {
    if (!imageUrl) {
      logger.error('No image URL available')
      return
    }

    // imageUrl is already a file:// URL — just pass it as the selected image path
    setSelectedImage(imageUrl)
    setMode('image-to-video')
    generatedImageRef.current = imageUrl
  }

  const handleClearAll = () => {
    clearEncoded()
    setEditableEnhancedPrompt(null)
    setPrompt('')
    setNegativePrompt('')
    setSelectedImage(null)
    setSelectedAudio(null)
    const baseDefaults = { ...DEFAULT_SETTINGS }
    const shouldSanitizeVideoSettings = shouldVideoGenerateWithLtxApi && mode !== 'text-to-image'
    setSettings(shouldSanitizeVideoSettings ? sanitizeForcedApiVideoSettings(baseDefaults) : baseDefaults)
    if (mode !== 'text-to-image') setMode('text-to-video')
    setRetakeInput({
      videoUrl: null,
      videoPath: null,
      startTime: 0,
      duration: 0,
      videoDuration: 0,
      ready: false,
    })
    setRetakePanelKey((prev) => prev + 1)
    setIcLoraInput({
      videoUrl: null,
      videoPath: null,
      conditioningType: 'canny',
      conditioningStrength: 1.0,
      ready: false,
    })
    setIcLoraPanelKey((prev) => prev + 1)
    setIcLoraCondType('canny')
    setIcLoraStrength(1.0)
    resetRetake()
    resetIcLora()
    reset()
  }

  const restoreFromHistory = (entry: HistoryEntry) => {
    setPrompt(entry.prompt)
    if (entry.negativePrompt) {
      setNegativePrompt(entry.negativePrompt)
      setShowNegativePrompt(true)
    }
    setSettings(entry.settings)
    setShowHistory(false)
  }

  const isRetakeMode = mode === 'retake'
  const isIcLoraMode = mode === 'ic-lora'
  const isVideoMode = mode === 'text-to-video' || mode === 'image-to-video'
  const isBusy = isRetakeMode ? isRetaking : isIcLoraMode ? isIcLoraGenerating : isGenerating
  const canGenerate = processStatus === 'alive' && !isBusy && (
    isRetakeMode
      ? retakeInput.ready && !!retakeInput.videoPath
      : isIcLoraMode
        ? icLoraInput.ready && !!icLoraInput.videoPath && !!prompt.trim()
        : !!prompt.trim()
  )

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-4">
          <button 
            onClick={goHome}
            className="p-2 rounded-lg hover:bg-zinc-800 transition-colors"
            title="Back to Home"
          >
            <ArrowLeft className="h-5 w-5 text-zinc-400" />
          </button>
          <div className="flex items-center gap-2.5">
            <LtxLogo className="h-6 w-auto text-white" />
            <span className="text-zinc-400 text-base font-medium tracking-wide leading-none pt-1 pl-1.5">Playground</span>
          </div>
        </div>
        
        <div className="flex items-center gap-4 pr-20">
          {/* Model Status Dropdown */}
          {!forceApiGenerations && <ModelStatusDropdown />}

          {/* GPU Info */}
          {status.gpuInfo && (
            <div className="text-sm text-zinc-500">
              {status.gpuInfo.name} ({(status.gpuInfo.vramUsed / 1024).toFixed(1)}GB / {Math.round(status.gpuInfo.vram / 1024)}GB)
            </div>
          )}

          {/* Output Browser button */}
          <button
            onClick={() => setShowOutputBrowser(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            title="Browse generated outputs"
          >
            <Film className="h-4 w-4" />
            Outputs
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Panel - Controls */}
        <div className="w-[500px] border-r border-zinc-800 p-6 overflow-y-auto">
          <div className="space-y-6">
            {/* Mode Tabs */}
            <ModeTabs
              mode={mode}
              onModeChange={handleModeChange}
              disabled={isBusy}
              showIcLora={!forceApiGenerations}
            />

            {/* Image Upload - Always shown in video mode (optional: makes it I2V) */}
            {isVideoMode && !isRetakeMode && (
              <>
                <ImageUploader
                  selectedImage={selectedImage}
                  onImageSelect={setSelectedImage}
                />
                <AudioUploader
                  selectedAudio={selectedAudio}
                  onAudioSelect={setSelectedAudio}
                />
              </>
            )}

            {isRetakeMode && (
              <RetakePanel
                resetKey={retakePanelKey}
                isProcessing={isRetaking}
                processingStatus={retakeStatus}
                onChange={(data) => setRetakeInput(data)}
              />
            )}

            {isIcLoraMode && (
              <>
                <ICLoraPanel
                  resetKey={icLoraPanelKey}
                  isProcessing={isIcLoraGenerating}
                  processingStatus={icLoraStatus}
                  conditioningType={icLoraCondType}
                  onConditioningTypeChange={setIcLoraCondType}
                  conditioningStrength={icLoraStrength}
                  onConditioningStrengthChange={setIcLoraStrength}
                  outputVideoUrl={icLoraResult?.videoUrl || null}
                  outputVideoPath={icLoraResult?.videoPath || null}
                  onChange={setIcLoraInput}
                />

                {/* Conditioning controls */}
                <div className="space-y-3 p-4 bg-zinc-900 border border-zinc-800 rounded-2xl">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-zinc-400">Conditioning Type</label>
                    <select
                      value={icLoraCondType}
                      onChange={(e) => setIcLoraCondType(e.target.value as ICLoraConditioningType)}
                      className="bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                    >
                      {CONDITIONING_TYPES.map(ct => (
                        <option key={ct.value} value={ct.value}>{ct.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-zinc-400">Strength</label>
                      <span className="text-xs text-zinc-400">{icLoraStrength.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.05"
                      value={icLoraStrength}
                      onChange={(e) => setIcLoraStrength(Number(e.target.value))}
                      className="w-full accent-blue-500"
                    />
                  </div>
                </div>
              </>
            )}

            {/* Prompt Input */}
            <div className="w-full">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[12px] font-semibold text-zinc-500 uppercase leading-4">Prompt</span>
                <button
                  onClick={() => setShowHistory(h => !h)}
                  className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
                  title="Prompt & settings history"
                >
                  <History size={12} />
                  History
                </button>
              </div>

              {/* History dropdown */}
              {showHistory && (
                <div className="mb-2 border border-zinc-700 rounded-lg bg-zinc-900 overflow-hidden">
                  {historyEntries.length === 0 ? (
                    <p className="text-xs text-zinc-500 px-3 py-3">No history yet. Successful generations will appear here.</p>
                  ) : (
                    <div className="max-h-64 overflow-y-auto divide-y divide-zinc-800">
                      {historyEntries.map((entry) => (
                        <div
                          key={entry.id}
                          className="flex items-start gap-2 px-3 py-2 hover:bg-zinc-800 cursor-pointer group"
                          onClick={() => restoreFromHistory(entry)}
                        >
                          <div className="flex-1 min-w-0">
                            <p className="text-xs text-zinc-200 truncate">{entry.prompt}</p>
                            <p className="text-[11px] text-zinc-500 mt-0.5">
                              {entry.settings.videoResolution} · {entry.settings.duration}s · {entry.settings.fps}fps
                              {entry.seedUsed != null ? ` · seed ${entry.seedUsed}` : ''}
                              {' · '}{new Date(entry.timestamp).toLocaleString()}
                            </p>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); removeHistory(entry.id); setShowHistory(false); setTimeout(() => setShowHistory(true), 0) }}
                            className="opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-zinc-400 transition-opacity flex-shrink-0 mt-0.5"
                          >
                            <X size={12} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <Textarea
                placeholder="Write a prompt..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                helperText="Longer, detailed prompts lead to better, more accurate results."
                charCount={prompt.length}
                maxChars={5000}
                disabled={isBusy}
              />

              {/* Encode Prompt button — single-GPU + local encoder only */}
              {showEncodeButton && (
                <div className="mt-2 flex items-center gap-2">
                  <button
                    onClick={() => encodePrompt(prompt)}
                    disabled={isEncoding || isBusy || !prompt.trim() || processStatus !== 'alive'}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      isEncoding
                        ? 'bg-zinc-700 text-zinc-300'
                        : encodedPrompt && !isPromptChanged
                        ? 'bg-emerald-800/60 text-emerald-300 hover:bg-emerald-800'
                        : isPromptChanged
                        ? 'bg-amber-800/60 text-amber-300 hover:bg-amber-800'
                        : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                    }`}
                    title={
                      isEncoding
                        ? 'Encoding prompt on GPU…'
                        : encodedPrompt && !isPromptChanged
                        ? 'Prompt encoded — click to re-encode'
                        : isPromptChanged
                        ? 'Prompt changed — click to re-encode on GPU'
                        : 'Encode prompt on GPU (ejects video model temporarily)'
                    }
                  >
                    {isEncoding ? (
                      <>
                        <Cpu className="h-3.5 w-3.5 animate-pulse" />
                        Encoding…
                      </>
                    ) : encodedPrompt && !isPromptChanged ? (
                      <>
                        <CheckCircle className="h-3.5 w-3.5" />
                        Encoded
                      </>
                    ) : isPromptChanged ? (
                      <>
                        <AlertCircle className="h-3.5 w-3.5" />
                        Re-encode prompt
                      </>
                    ) : (
                      <>
                        <Cpu className="h-3.5 w-3.5" />
                        Encode prompt
                      </>
                    )}
                  </button>
                  {encodeError && (
                    <span className="text-xs text-red-400 truncate" title={encodeError}>
                      {encodeError}
                    </span>
                  )}
                </div>
              )}

              {/* Enhanced prompt (from Gemma local enhancement) */}
              {editableEnhancedPrompt !== null && (
                <div className="mt-3 w-full">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] font-semibold text-purple-400 uppercase leading-4 flex items-center gap-1">
                      <Sparkles size={10} />
                      Enhanced prompt
                    </span>
                    <button
                      onClick={() => setEditableEnhancedPrompt(null)}
                      className="text-zinc-600 hover:text-zinc-400 transition-colors"
                      title="Dismiss"
                    >
                      <X size={12} />
                    </button>
                  </div>
                  <textarea
                    value={editableEnhancedPrompt}
                    onChange={(e) => setEditableEnhancedPrompt(e.target.value)}
                    disabled={isBusy}
                    rows={4}
                    className="w-full resize-none rounded-lg bg-zinc-900 border border-purple-900/50 px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-purple-700 disabled:opacity-50"
                  />
                  {showEncodeButton && (
                    <div className="mt-1.5 flex items-center gap-2">
                      <button
                        onClick={() => encodePrompt(editableEnhancedPrompt)}
                        disabled={isEncoding || isBusy || !editableEnhancedPrompt.trim() || processStatus !== 'alive'}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                          isEncoding
                            ? 'bg-zinc-700 text-zinc-300'
                            : encodedPrompt === editableEnhancedPrompt.trim()
                            ? 'bg-emerald-800/60 text-emerald-300 hover:bg-emerald-800'
                            : encodedPrompt !== null
                            ? 'bg-amber-800/60 text-amber-300 hover:bg-amber-800'
                            : 'bg-purple-900/60 text-purple-300 hover:bg-purple-900'
                        }`}
                        title="Encode enhanced prompt on GPU"
                      >
                        {isEncoding ? (
                          <><Cpu className="h-3.5 w-3.5 animate-pulse" />Encoding…</>
                        ) : encodedPrompt === editableEnhancedPrompt.trim() ? (
                          <><CheckCircle className="h-3.5 w-3.5" />Encoded</>
                        ) : (
                          <><Cpu className="h-3.5 w-3.5" />Encode enhanced</>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Negative prompt */}
              <div className="mt-2">
                <button
                  onClick={() => setShowNegativePrompt(p => !p)}
                  className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  <span className={`transition-transform ${showNegativePrompt ? 'rotate-90' : ''}`}>▶</span>
                  Negative prompt
                </button>
                {showNegativePrompt && (
                  <textarea
                    placeholder="What to avoid in the generation..."
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    disabled={isBusy}
                    rows={2}
                    className="mt-1.5 w-full resize-none rounded-lg bg-zinc-900 border border-zinc-700 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-600 disabled:opacity-50"
                  />
                )}
              </div>
            </div>

            {/* Settings */}
            {!isRetakeMode && !isIcLoraMode && (
              <SettingsPanel
                settings={settings}
                onSettingsChange={setSettings}
                disabled={isBusy}
                mode={mode}
                forceApiGenerations={shouldVideoGenerateWithLtxApi}
                hasAudio={!!selectedAudio}
              />
            )}

            {/* Error Display */}
            {(generationError || retakeError || icLoraError) && (
              <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm">
                {(generationError || retakeError || icLoraError)!.includes('TEXT_ENCODING_NOT_CONFIGURED') ? (
                  <div className="space-y-2">
                    <p className="text-red-400 font-medium">Text encoding not configured</p>
                    <p className="text-red-400/80">
                      To generate videos, you need to set up text encoding in Settings.
                    </p>
                  </div>
                ) : (generationError || retakeError || icLoraError)!.includes('TEXT_ENCODER_NOT_DOWNLOADED') ? (
                  <div className="space-y-2">
                    <p className="text-red-400 font-medium">Text encoder not downloaded</p>
                    <p className="text-red-400/80">
                      The local text encoder needs to be downloaded (~25 GB).
                    </p>
                  </div>
                ) : (
                  <span className="text-red-400">{generationError || retakeError || icLoraError}</span>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4">
              <Button
                variant="outline"
                onClick={handleClearAll}
                disabled={isBusy}
                className="flex items-center gap-2 border-zinc-700 bg-zinc-800 text-white hover:bg-zinc-700"
              >
                <Trash2 className="h-4 w-4" />
                Clear all
              </Button>
              
              {isGenerating ? (
                <Button
                  onClick={cancel}
                  className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white"
                >
                  <Square className="h-4 w-4" />
                  Stop generation
                </Button>
              ) : (
                <Button
                  onClick={handleGenerate}
                  disabled={!canGenerate}
                  className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 text-white disabled:bg-zinc-700 disabled:text-zinc-500"
                >
                  {isRetakeMode ? (
                    <>
                      <Scissors className="h-4 w-4" />
                      {isRetaking ? 'Retaking...' : 'Retake'}
                    </>
                  ) : isIcLoraMode ? (
                    <>
                      <Sparkles className="h-4 w-4" />
                      {isIcLoraGenerating ? 'Generating...' : 'Generate IC-LoRA'}
                    </>
                  ) : mode === 'text-to-image' ? (
                    <>
                      <ImageIcon className="h-4 w-4" />
                      Generate image
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      Generate video
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Result Preview */}
        <div className="flex-1 p-6">
          {mode === 'text-to-image' ? (
            <ImageResult
              imageUrl={imageUrl}
              isGenerating={isGenerating}
              progress={progress}
              statusMessage={statusMessage}
              onCreateVideo={handleCreateVideoFromImage}
            />
          ) : mode === 'retake' ? (
            <VideoPlayer
              videoUrl={retakeResult?.videoUrl || null}
              videoPath={retakeResult?.videoPath || null}
              videoResolution={settings.videoResolution}
              isGenerating={isRetaking}
              progress={0}
              statusMessage={retakeStatus}
            />
          ) : mode === 'ic-lora' ? (
            <VideoPlayer
              videoUrl={icLoraResult?.videoUrl || null}
              videoPath={icLoraResult?.videoPath || null}
              videoResolution={settings.videoResolution}
              isGenerating={isIcLoraGenerating}
              progress={0}
              statusMessage={icLoraStatus}
            />
          ) : (
            <VideoPlayer
              videoUrl={videoUrl}
              videoPath={videoPath}
              videoResolution={settings.videoResolution}
              isGenerating={isGenerating}
              progress={progress}
              statusMessage={statusMessage}
            />
          )}
        </div>
      </main>

      <OutputBrowser
        isOpen={showOutputBrowser}
        onClose={() => setShowOutputBrowser(false)}
      />
    </div>
  )
}
