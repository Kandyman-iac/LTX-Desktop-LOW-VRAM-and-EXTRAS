import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import {
  Trash2, Download, Image, Video, X,
  Heart, Film, Volume2, VolumeX, Sparkles,
  Clock, Monitor, ChevronUp, Scissors, Music,
  ChevronLeft, ChevronRight, Copy, Check, Lock, Shuffle,
  Wand2, Cpu, CheckCircle, AlertCircle, ListPlus, ChevronDown,
  Upload, Plus, Maximize2, Minimize2, Star, MessageSquare, FolderPlus,
} from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import type { GenSpaceRetakeSource } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useGeneration } from '../hooks/use-generation'
import { useRetake } from '../hooks/use-retake'
import { useIcLora } from '../hooks/use-ic-lora'
import type { ICLoraConditioningType } from '../components/ICLoraPanel'
import type { Asset } from '../types/project'
import { GenerationErrorDialog } from '../components/GenerationErrorDialog'
import { copyToAssetFolder } from '../lib/asset-copy'
import { fileUrlToPath } from '../lib/url-to-path'
import {
  FORCED_API_VIDEO_FPS,
  FORCED_API_VIDEO_RESOLUTIONS,
  getAllowedForcedApiDurations,
  sanitizeForcedApiVideoSettings,
} from '../lib/api-video-options'
import { logger } from '../lib/logger'
import type { ConditioningFrame } from '../components/MultiFrameConditioningPanel'
import { RetakePanel } from '../components/RetakePanel'
import { ICLoraPanel, CONDITIONING_TYPES } from '../components/ICLoraPanel'
import { FreeApiKeyBubble } from '../components/FreeApiKeyBubble'
import { useBackend } from '../hooks/use-backend'
import { useEnhancePrompt } from '../hooks/use-enhance-prompt'
import { useEnhancedPromptHistory } from '../hooks/use-enhanced-prompt-history'
import { useGenerationHistory } from '../hooks/use-generation-history'
import { useEncodePrompt } from '../hooks/use-encode-prompt'
import { useQueue, parseJobLoras } from '../hooks/use-queue'
import { QueuePanel } from '../components/QueuePanel'
import { COLOR_LABELS, getColorLabel } from './editor/video-editor-utils'

// Asset card with hover overlays
function AssetCard({
  asset,
  onDelete,
  onPlay,
  onDragStart,
  onCreateVideo,
  onRetake,
  onIcLora,
  onToggleFavorite,
  onSetRating,
  onSetNotes,
  onSetColorLabel,
  onSetBin,
  allBins,
}: {
  asset: Asset
  onDelete: () => void
  onPlay: () => void
  onDragStart: (e: React.DragEvent, asset: Asset) => void
  onCreateVideo?: (asset: Asset) => void
  onRetake?: (asset: Asset) => void
  onIcLora?: (asset: Asset) => void
  onToggleFavorite?: () => void
  onSetRating?: (rating: number | undefined) => void
  onSetNotes?: (notes: string) => void
  onSetColorLabel?: (color: string | undefined) => void
  onSetBin?: (bin: string | undefined) => void
  allBins?: string[]
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isMuted, setIsMuted] = useState(true)
  const [showNotesInput, setShowNotesInput] = useState(false)
  const [notesValue, setNotesValue] = useState(asset.notes ?? '')
  const [showBinInput, setShowBinInput] = useState(false)
  const [newBinValue, setNewBinValue] = useState('')
  const isFavorite = asset.favorite || false
  const colorDef = getColorLabel(asset.colorLabel)

  useEffect(() => {
    if (asset.type === 'video' && videoRef.current) {
      if (isHovered) {
        videoRef.current.play().catch(() => {})
      } else {
        videoRef.current.pause()
        videoRef.current.currentTime = 0
        setCurrentTime(0)
      }
    }
  }, [isHovered, asset.type])

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation()
    const a = document.createElement('a')
    a.href = asset.url
    a.download = asset.path.split('/').pop() || `${asset.type}-${asset.id}`
    a.click()
  }

  return (
    <div
      className="relative group cursor-pointer rounded-xl overflow-hidden bg-zinc-900"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => { setIsHovered(false); setShowNotesInput(false); setShowBinInput(false) }}
      onClick={onPlay}
      draggable={asset.type === 'image'}
      onDragStart={(e) => asset.type === 'image' && onDragStart(e, asset)}
    >
      {asset.type === 'video' ? (
        <video
          ref={videoRef}
          src={asset.url}
          className="w-full aspect-video object-contain"
          muted={isMuted}
          loop
          onTimeUpdate={handleTimeUpdate}
        />
      ) : (
        <img src={asset.url} alt="" className="w-full aspect-video object-contain" />
      )}

      {/* Color label bar — always visible when set */}
      {colorDef && (
        <div className="absolute top-0 left-0 right-0 h-1" style={{ backgroundColor: colorDef.color }} />
      )}

      {/* Star rating — always visible when rated */}
      {(asset.rating ?? 0) > 0 && !isHovered && (
        <div className="absolute bottom-2 left-2 flex items-center gap-0.5">
          {[1,2,3,4,5].map(n => (
            <Star key={n} className={`h-3 w-3 ${n <= (asset.rating ?? 0) ? 'text-yellow-400 fill-current' : 'text-zinc-600'}`} />
          ))}
        </div>
      )}

      {/* Notes indicator — always visible when set */}
      {asset.notes && !isHovered && (
        <div className="absolute bottom-2 right-2 p-1 rounded bg-black/40 backdrop-blur-md">
          <MessageSquare className="h-3 w-3 text-zinc-300" />
        </div>
      )}

      {/* Favorite heart - always visible when favorited */}
      {isFavorite && !isHovered && (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleFavorite?.() }}
          className="absolute top-2 left-2 p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white transition-colors z-10"
        >
          <Heart className="h-3.5 w-3.5 fill-current" />
        </button>
      )}

      {/* Hover overlay */}
      <div className={`absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/30 transition-opacity duration-200 ${
        isHovered ? 'opacity-100' : 'opacity-0'
      }`}>
        {/* Top buttons */}
        <div className="absolute top-2 left-2 right-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <button
              onClick={(e) => { e.stopPropagation(); onToggleFavorite?.() }}
              className={`p-1.5 rounded-lg backdrop-blur-md transition-colors ${
                isFavorite ? 'bg-white/20 text-white' : 'bg-black/40 text-white hover:bg-black/60'
              }`}
            >
              <Heart className={`h-3.5 w-3.5 ${isFavorite ? 'fill-current' : ''}`} />
            </button>

            {asset.type === 'image' && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); onCreateVideo?.(asset) }}
                  className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                >
                  <Film className="h-3 w-3" />
                  Create video
                </button>
              </>
            )}
            {asset.type === 'video' && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); onRetake?.(asset) }}
                  className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                >
                  <Scissors className="h-3 w-3" />
                  Retake
                </button>
                {onIcLora && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onIcLora(asset) }}
                    className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                  >
                    <Sparkles className="h-3 w-3" />
                    IC-LoRA
                  </button>
                )}
              </>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={handleDownload}
              className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Middle meta bar: color picker + star rating + notes + bin */}
        <div className="absolute left-2 right-2 top-12 flex items-center gap-1.5 flex-wrap" onClick={e => e.stopPropagation()}>
          {/* Color dots */}
          <div className="flex items-center gap-0.5 p-1 rounded-lg bg-black/50 backdrop-blur-md">
            <button
              onClick={() => onSetColorLabel?.(undefined)}
              className={`w-3.5 h-3.5 rounded-full border transition-all ${!asset.colorLabel ? 'border-white scale-110' : 'border-zinc-600 hover:border-white'} bg-zinc-700`}
              title="No color"
            />
            {COLOR_LABELS.map(cl => (
              <button
                key={cl.id}
                onClick={() => onSetColorLabel?.(asset.colorLabel === cl.id ? undefined : cl.id)}
                className={`w-3.5 h-3.5 rounded-full border transition-all ${asset.colorLabel === cl.id ? 'border-white scale-110' : 'border-transparent hover:border-white'}`}
                style={{ backgroundColor: cl.color }}
                title={cl.label}
              />
            ))}
          </div>

          {/* Star rating */}
          <div className="flex items-center gap-0.5 p-1 rounded-lg bg-black/50 backdrop-blur-md">
            {[1,2,3,4,5].map(n => (
              <button
                key={n}
                onClick={() => onSetRating?.(asset.rating === n ? undefined : n)}
                className="p-0.5"
              >
                <Star className={`h-3.5 w-3.5 transition-colors ${n <= (asset.rating ?? 0) ? 'text-yellow-400 fill-current' : 'text-zinc-500 hover:text-yellow-400'}`} />
              </button>
            ))}
          </div>

          {/* Notes toggle */}
          <button
            onClick={() => { setShowNotesInput(!showNotesInput); setNotesValue(asset.notes ?? '') }}
            className={`p-1.5 rounded-lg backdrop-blur-md transition-colors ${asset.notes ? 'bg-blue-500/30 text-blue-300' : 'bg-black/50 text-zinc-400 hover:text-white'}`}
            title="Notes"
          >
            <MessageSquare className="h-3 w-3" />
          </button>

          {/* Bin toggle */}
          <button
            onClick={() => { setShowBinInput(!showBinInput) }}
            className={`p-1.5 rounded-lg backdrop-blur-md transition-colors ${asset.bin ? 'bg-violet-500/30 text-violet-300' : 'bg-black/50 text-zinc-400 hover:text-white'}`}
            title={asset.bin ? `Bin: ${asset.bin}` : 'Add to bin'}
          >
            <FolderPlus className="h-3 w-3" />
          </button>
          {asset.bin && (
            <span className="text-[10px] text-violet-300 bg-violet-500/20 px-1.5 py-0.5 rounded">{asset.bin}</span>
          )}
        </div>

        {/* Notes input (expands below meta bar) */}
        {showNotesInput && (
          <div className="absolute left-2 right-2 top-[88px] z-20" onClick={e => e.stopPropagation()}>
            <div className="bg-zinc-900/95 border border-zinc-700 rounded-lg p-2">
              <textarea
                className="w-full bg-transparent text-xs text-white placeholder:text-zinc-500 resize-none outline-none"
                rows={3}
                placeholder="Add notes..."
                value={notesValue}
                onChange={e => setNotesValue(e.target.value)}
                autoFocus
              />
              <div className="flex justify-end gap-1.5 mt-1">
                <button
                  className="px-2 py-0.5 rounded text-xs text-zinc-400 hover:text-white"
                  onClick={() => setShowNotesInput(false)}
                >Cancel</button>
                <button
                  className="px-2 py-0.5 rounded text-xs bg-violet-600 text-white hover:bg-violet-500"
                  onClick={() => { onSetNotes?.(notesValue); setShowNotesInput(false) }}
                >Save</button>
              </div>
            </div>
          </div>
        )}

        {/* Bin picker */}
        {showBinInput && (
          <div className="absolute left-2 right-2 top-[88px] z-20" onClick={e => e.stopPropagation()}>
            <div className="bg-zinc-900/95 border border-zinc-700 rounded-lg p-2 space-y-1">
              {(allBins ?? []).map(bin => (
                <button
                  key={bin}
                  onClick={() => { onSetBin?.(asset.bin === bin ? undefined : bin); setShowBinInput(false) }}
                  className={`w-full text-left text-xs px-2 py-1 rounded transition-colors ${asset.bin === bin ? 'bg-violet-600 text-white' : 'text-zinc-300 hover:bg-zinc-700'}`}
                >
                  {bin}
                </button>
              ))}
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-zinc-800 border border-zinc-600 rounded px-2 py-1 text-xs text-white placeholder:text-zinc-500 outline-none"
                  placeholder="New bin..."
                  value={newBinValue}
                  onChange={e => setNewBinValue(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && newBinValue.trim()) {
                      onSetBin?.(newBinValue.trim())
                      setNewBinValue('')
                      setShowBinInput(false)
                    }
                  }}
                  autoFocus={!(allBins ?? []).length}
                />
                <button
                  className="px-2 py-1 rounded text-xs bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40"
                  disabled={!newBinValue.trim()}
                  onClick={() => { onSetBin?.(newBinValue.trim()); setNewBinValue(''); setShowBinInput(false) }}
                >Add</button>
              </div>
              {asset.bin && (
                <button
                  className="w-full text-left text-xs px-2 py-1 rounded text-zinc-500 hover:text-red-400 hover:bg-zinc-700 transition-colors"
                  onClick={() => { onSetBin?.(undefined); setShowBinInput(false) }}
                >Remove from bin</button>
              )}
            </div>
          </div>
        )}

        {/* Bottom controls for video */}
        {asset.type === 'video' && (
          <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <div className="px-2 py-1 rounded-lg bg-black/50 backdrop-blur-md text-white text-xs font-mono">
                {formatTime(currentTime)}
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setIsMuted(!isMuted) }}
                className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
              >
                {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>
        )}

        {/* Delete button (subtle, bottom right) */}
        <button
          onClick={(e) => { e.stopPropagation(); onDelete() }}
          className="absolute bottom-2 right-2 p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white/70 hover:bg-red-500/80 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

    </div>
  )
}

// Dropdown component for settings
function SettingsDropdown({ 
  trigger, 
  options, 
  value, 
  onChange,
  title 
}: { 
  trigger: React.ReactNode
  options: { value: string; label: string; disabled?: boolean; tooltip?: string; icon?: React.ReactNode }[]
  value: string
  onChange: (value: string) => void
  title: string
}) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])
  
  return (
    <div ref={dropdownRef} className="relative">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`flex shrink-0 items-center gap-1 whitespace-nowrap px-2 py-1.5 rounded-md transition-colors ${isOpen ? 'bg-zinc-700 hover:bg-zinc-700' : 'hover:bg-zinc-800'}`}
      >
        {trigger}
      </button>
      
      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[160px] shadow-xl z-[9999]">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">{title}</div>
          <div className="space-y-1">
            {options.map(option => (
              <div key={option.value} className="relative group/option">
                <button
                  onClick={() => { if (!option.disabled) { onChange(option.value); setIsOpen(false) } }}
                  className={`w-full flex items-center justify-between px-2 py-2 rounded-md transition-colors text-left ${
                    option.disabled
                      ? 'cursor-not-allowed'
                      : value === option.value ? 'bg-white/20 hover:bg-white/25' : 'hover:bg-zinc-700'
                  }`}
                >
                  <span className={`flex items-center gap-2.5 text-sm ${
                    option.disabled 
                      ? 'text-zinc-600' 
                      : value === option.value ? 'text-white' : 'text-zinc-400'
                  }`}>
                    {option.icon && <span className="flex-shrink-0">{option.icon}</span>}
                    {option.label}
                  </span>
                  {value === option.value && !option.disabled && (
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
                {option.disabled && option.tooltip && (
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-zinc-700 rounded text-xs text-zinc-300 whitespace-nowrap opacity-0 group-hover/option:opacity-100 pointer-events-none z-[10000] transition-opacity">
                    {option.tooltip}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Lightricks brand icon
function LightricksIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fillRule="evenodd" clipRule="evenodd" d="M17.0073 8.18934C16.3266 5.6556 14.9346 2.06903 12.3065 2.06903C9.27204 2.06903 6.86627 7.24621 5.45487 11.7948C4.79654 13.9203 4.35877 15.9049 4.17755 17.1736C4.10214 17.5829 4.06274 18.0044 4.06274 18.4347C4.06274 22.2903 7.22553 25.4338 11.1133 25.4338C15.5206 25.4338 23.9376 22.7073 23.9376 18.4347C23.9376 17.1179 23.1376 15.948 21.9018 14.9595L21.9039 14.9575C22.4493 13.7707 22.847 12.648 23.001 11.705C23.1934 10.5053 23.0074 9.5494 22.4429 8.88217C21.7692 8.07382 20.7107 7.85572 19.6586 7.84288C18.8826 7.84288 17.9777 7.96904 17.0073 8.18934ZM8.00176 9.17083C7.6945 9.93266 7.02317 11.7419 6.70157 12.9799C7.93005 11.9987 9.2965 11.1653 10.7091 10.4796C12.2325 9.73758 13.9171 9.06448 15.518 8.58411C15.08 6.98293 13.9585 3.62158 12.3129 3.62158C11.0298 3.62158 9.41958 5.69374 8.00176 9.17083ZM20.6201 14.083L20.6209 14.0786C21.0507 13.1163 21.3522 12.2118 21.4741 11.4547C21.5511 10.9607 21.5832 10.2872 21.2752 9.89577C20.9416 9.46599 20.1975 9.39543 19.6521 9.38901C18.9932 9.38901 18.2117 9.49943 17.3641 9.69208L17.3683 9.69702C17.586 10.7217 17.7526 11.772 17.8808 12.7968C18.8527 13.16 19.7877 13.5908 20.6201 14.083ZM15.8828 10.0897C14.6739 10.4588 13.4041 10.9464 12.209 11.4846C13.4346 11.588 14.8471 11.8527 16.2581 12.2608C16.1554 11.5367 16.0273 10.8061 15.8799 10.0948L15.8828 10.0897ZM11.1133 12.9816C8.07878 12.9816 5.60884 15.4258 5.60884 18.4347C5.60884 21.4435 8.07878 23.8878 11.1133 23.8878C13.8701 23.8878 16.3653 21.6639 16.6048 18.9158C16.7011 17.7546 16.669 15.9263 16.4637 13.9311C14.6294 13.3385 12.6763 12.9816 11.1133 12.9816ZM18.3883 22.2069C17.7984 22.4697 17.1711 22.7085 16.5284 22.9184C18.0872 21.3274 19.8832 18.8193 21.1982 16.3689L21.1997 16.3654C21.9756 17.0509 22.3915 17.7593 22.3915 18.4347C22.3915 19.6985 20.9288 21.0778 18.3883 22.2069ZM19.9493 15.4655L19.9473 15.4707C19.4291 16.4567 18.8221 17.4625 18.1833 18.4092C18.2214 17.4089 18.1892 16.0386 18.0611 14.5212C18.71 14.7948 19.3456 15.1021 19.9493 15.4655Z" fill="currentColor" />
    </svg>
  )
}

function ZitIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M19.113 12.2515H16.5605L14.008 8.63382L6.04545 19.9068H8.60348L14.0079 12.2518L16.5605 12.2515L11.156 19.9068H13.721L19.113 12.2515V15.8693L16.2716 19.9073V22.0063H2L14.008 5L19.113 12.2515Z" fill="currentColor"/>
      <path d="M26 22.0064L21.9704 22.0063V19.9151L19.113 15.8693V12.2515L26 22.0064Z" fill="currentColor"/>
    </svg>
  )
}

// Square icon for aspect ratio
function AspectIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="5" width="18" height="14" rx="2" />
    </svg>
  )
}

// ─── Inline frame conditioning (horizontal 3-column layout for GenSpace) ────

function InlineFrameSlot({
  frame,
  label,
  onUpdate,
  onRemove,
}: {
  frame: ConditioningFrame
  label: string
  onUpdate: (updates: Partial<ConditioningFrame>) => void
  onRemove?: () => void
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [extracting, setExtracting] = useState(false)
  const [videoPath, setVideoPath] = useState('')
  const [seekTime, setSeekTime] = useState('0')
  const [showExtractor, setShowExtractor] = useState(false)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const path = (file as File & { path?: string }).path ?? ''
    const url = path ? `file:///${path.replace(/\\/g, '/')}` : URL.createObjectURL(file)
    onUpdate({ imagePath: path || null, imageUrl: url })
  }

  const handleExtract = async () => {
    if (!videoPath.trim()) return
    setExtracting(true)
    try {
      const videoUrl = videoPath.startsWith('file://') ? videoPath : `file:///${videoPath.replace(/\\/g, '/')}`
      const result = await window.electronAPI.extractVideoFrame(videoUrl, parseFloat(seekTime) || 0, 512, 3)
      onUpdate({ imagePath: result.path, imageUrl: result.url })
      setShowExtractor(false)
    } catch (err) {
      console.error('Frame extraction failed:', err)
    } finally {
      setExtracting(false)
    }
  }

  const handlePickVideo = async () => {
    const files = await window.electronAPI.showOpenFileDialog({
      title: 'Select Video',
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'webm', 'avi', 'mkv'] }],
    })
    if (files?.[0]) setVideoPath(files[0])
  }

  return (
    <div className="flex flex-col gap-1.5">
      {/* Header row */}
      <div className="flex items-center justify-between h-4">
        <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide leading-none">{label}</span>
        {onRemove && (
          <button onClick={onRemove} className="text-zinc-600 hover:text-zinc-400">
            <X className="h-3 w-3" />
          </button>
        )}
      </div>

      {/* Thumbnail / upload zone */}
      <div
        className="relative rounded-lg border border-dashed border-zinc-700 bg-zinc-800 cursor-pointer overflow-hidden aspect-video hover:border-zinc-500 transition-colors"
        onClick={() => fileInputRef.current?.click()}
      >
        {frame.imageUrl ? (
          <>
            <img src={frame.imageUrl} className="w-full h-full object-contain" alt={label} />
            <button
              onClick={(e) => { e.stopPropagation(); onUpdate({ imagePath: null, imageUrl: null }) }}
              className="absolute top-1 right-1 p-0.5 rounded bg-black/60 text-zinc-300 hover:text-white"
            >
              <X className="h-3 w-3" />
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-1 text-zinc-600">
            <Upload className="h-4 w-4" />
          </div>
        )}
        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
      </div>

      {/* Strength slider */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-zinc-500 w-5 shrink-0">Str</span>
        <input
          type="range" min={0} max={1} step={0.05}
          value={frame.strength}
          onChange={(e) => onUpdate({ strength: parseFloat(e.target.value) })}
          className="flex-1 h-1 accent-blue-500"
        />
        <span className="text-[10px] text-zinc-400 w-7 text-right">{frame.strength.toFixed(2)}</span>
      </div>

      {/* Extract-from-video toggle */}
      <button
        className="flex items-center gap-1 text-[10px] text-zinc-600 hover:text-blue-400 transition-colors"
        onClick={() => setShowExtractor(v => !v)}
      >
        <Film className="h-3 w-3" />
        From video
      </button>
      {showExtractor && (
        <div className="flex gap-1 items-center">
          <button className="text-[10px] text-blue-400 hover:underline whitespace-nowrap" onClick={handlePickVideo}>
            Pick
          </button>
          {videoPath && (
            <span className="text-[10px] text-zinc-500 truncate flex-1">{videoPath.split(/[\\/]/).pop()}</span>
          )}
          <input
            type="number" min={0} step={0.1}
            value={seekTime}
            onChange={(e) => setSeekTime(e.target.value)}
            className="w-10 rounded border border-zinc-700 bg-zinc-800 px-1 py-0.5 text-[10px] text-white"
            placeholder="s"
          />
          <button
            onClick={handleExtract}
            disabled={!videoPath || extracting}
            className="px-1.5 py-0.5 rounded bg-blue-600 hover:bg-blue-500 text-[10px] text-white disabled:opacity-40"
          >
            {extracting ? '…' : 'Get'}
          </button>
        </div>
      )}
    </div>
  )
}

function InlineFrameConditioning({
  frames,
  onChange,
}: {
  frames: ConditioningFrame[]
  onChange: (frames: ConditioningFrame[]) => void
}) {
  const firstFrame = frames.find(f => f.role === 'first')!
  const middleFrame = frames.find(f => f.role === 'middle') ?? null
  const lastFrame = frames.find(f => f.role === 'last') ?? null

  const updateFrame = (role: ConditioningFrame['role'], updates: Partial<ConditioningFrame>) =>
    onChange(frames.map(f => f.role === role ? { ...f, ...updates } : f))

  const addMiddleFrame = () => {
    const nf: ConditioningFrame = { role: 'middle', imagePath: null, imageUrl: null, strength: 1.0, position: 0.5 }
    onChange([...frames.filter(f => f.role !== 'last'), nf, ...(lastFrame ? [lastFrame] : [])])
  }

  const removeMiddleFrame = () => onChange(frames.filter(f => f.role !== 'middle'))
  const addLastFrame = () => onChange([...frames, { role: 'last', imagePath: null, imageUrl: null, strength: 1.0, position: 1 }])
  const removeLastFrame = () => onChange(frames.filter(f => f.role !== 'last'))

  const placeholderSlot = (label: string, onClick: () => void) => (
    <div className="flex flex-col gap-1.5">
      <span className="h-4" />
      <button
        onClick={onClick}
        className="aspect-video rounded-lg border border-dashed border-zinc-700 hover:border-zinc-500 bg-zinc-800/30 hover:bg-zinc-800/60 transition-colors flex flex-col items-center justify-center gap-1 text-zinc-600 hover:text-zinc-400"
      >
        <Plus className="h-4 w-4" />
        <span className="text-[10px]">{label}</span>
      </button>
    </div>
  )

  return (
    <div className="space-y-2">
      {/* Three columns */}
      <div className="grid grid-cols-3 gap-3">
        <InlineFrameSlot label="First Frame" frame={firstFrame} onUpdate={u => updateFrame('first', u)} />
        {middleFrame
          ? <InlineFrameSlot label="Middle Frame" frame={middleFrame} onUpdate={u => updateFrame('middle', u)} onRemove={removeMiddleFrame} />
          : placeholderSlot('Middle frame', addMiddleFrame)
        }
        {lastFrame
          ? <InlineFrameSlot label="Last Frame" frame={lastFrame} onUpdate={u => updateFrame('last', u)} onRemove={removeLastFrame} />
          : placeholderSlot('Last frame', addLastFrame)
        }
      </div>

      {/* Middle frame position slider — spans the middle column by using an invisible label spacer */}
      {middleFrame && (
        <div className="grid grid-cols-3 gap-3">
          <div />
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-zinc-500 shrink-0">Position</span>
            <input
              type="range" min={0.05} max={0.95} step={0.05}
              value={middleFrame.position}
              onChange={(e) => updateFrame('middle', { position: parseFloat(e.target.value) })}
              className="flex-1 h-1 accent-blue-500"
            />
            <span className="text-[10px] text-zinc-400 w-7 text-right shrink-0">{Math.round(middleFrame.position * 100)}%</span>
          </div>
          <div />
        </div>
      )}
    </div>
  )
}

// Prompt bar component matching the design
// Two-row layout: prompt row on top, settings row below
function PromptBar({
  mode,
  onModeChange,
  canUseIcLora,
  prompt,
  onPromptChange,
  onGenerate,
  isGenerating,
  inputImage,
  onInputImageChange,
  inputAudio,
  onInputAudioChange,
  settings,
  onSettingsChange,
  shouldVideoGenerateWithLtxApi,
  canGenerate,
  buttonLabel,
  buttonIcon,
  icLoraCondType,
  onIcLoraCondTypeChange,
  icLoraStrength,
  onIcLoraStrengthChange,
  seedInput,
  onSeedInputChange,
  seedLocked,
  onToggleSeedLock,
  negativePrompt = '',
  onNegativePromptChange,
  onEnhance,
  isEnhancing = false,
  enhanceError = null,
  editableEnhancedPrompt,
  onEditableEnhancedPromptChange,
  onEncode,
  isEncoding = false,
  encodedPrompt,
  isPromptChanged = false,
  showEncodeButton = false,
  onAddToQueue,
  pendingJobCount = 0,
  numSteps,
  onNumStepsChange,
  stgScale,
  onStgScaleChange,
  stgBlockIndex,
  onStgBlockIndexChange,
  blockSwap,
  onBlockSwapChange,
  processStatus = 'alive',
  useEnhancedView = false,
  onUseEnhancedViewChange,
  conditioningFrames,
  onConditioningFramesChange,
}: {
  mode: 'image' | 'video' | 'retake' | 'ic-lora'
  onModeChange: (mode: 'image' | 'video' | 'retake' | 'ic-lora') => void
  canUseIcLora: boolean
  prompt: string
  onPromptChange: (prompt: string) => void
  onGenerate: () => void
  isGenerating: boolean
  canGenerate: boolean
  buttonLabel: string
  buttonIcon: React.ReactNode
  inputImage: string | null
  onInputImageChange: (url: string | null) => void
  inputAudio: string | null
  onInputAudioChange: (url: string | null) => void
  settings: {
    model: string
    duration: number
    videoResolution: string
    fps: number
    aspectRatio: string
    imageResolution: string
    variations: number
    audio?: boolean
  }
  onSettingsChange: (settings: any) => void
  shouldVideoGenerateWithLtxApi: boolean
  icLoraCondType?: ICLoraConditioningType
  onIcLoraCondTypeChange?: (type: ICLoraConditioningType) => void
  icLoraStrength?: number
  onIcLoraStrengthChange?: (strength: number) => void
  seedInput?: string
  onSeedInputChange?: (v: string) => void
  seedLocked?: boolean
  onToggleSeedLock?: () => void
  negativePrompt?: string
  onNegativePromptChange?: (v: string) => void
  onEnhance?: () => void
  isEnhancing?: boolean
  enhanceError?: string | null
  editableEnhancedPrompt?: string | null
  onEditableEnhancedPromptChange?: (v: string) => void
  onEncode?: () => void
  isEncoding?: boolean
  encodedPrompt?: string | null
  isPromptChanged?: boolean
  showEncodeButton?: boolean
  onAddToQueue?: () => void
  pendingJobCount?: number
  numSteps?: number
  onNumStepsChange?: (v: number) => void
  stgScale?: number
  onStgScaleChange?: (v: number) => void
  stgBlockIndex?: number
  onStgBlockIndexChange?: (v: number) => void
  blockSwap?: number
  onBlockSwapChange?: (v: number) => void
  processStatus?: string
  useEnhancedView?: boolean
  onUseEnhancedViewChange?: (v: boolean) => void
  conditioningFrames?: ConditioningFrame[]
  onConditioningFramesChange?: (frames: ConditioningFrame[]) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isAudioDragOver, setIsAudioDragOver] = useState(false)
  const [showNegativePrompt, setShowNegativePrompt] = useState(false)
  const [showSampler, setShowSampler] = useState(false)

  const [isExpanded, setIsExpanded] = useState(false)
  const isRetake = mode === 'retake'
  const isIcLora = mode === 'ic-lora'
  const LOCAL_MAX_DURATION: Record<string, number> = { '540p': 30, '720p': 30, '1080p': 10 }
  const localMaxDuration = LOCAL_MAX_DURATION[settings.videoResolution] ?? 30
  const videoDurationOptions = shouldVideoGenerateWithLtxApi
    ? [...getAllowedForcedApiDurations(settings.model, settings.videoResolution, settings.fps)]
    : [5, 6, 8, 10, 15, 20, 25, 30].filter(d => d <= localMaxDuration)
  const videoResolutionOptions = shouldVideoGenerateWithLtxApi
    ? (inputAudio ? ['1080p'] : [...FORCED_API_VIDEO_RESOLUTIONS])
    : ['540p', '720p', '1080p']
  const videoFpsOptions = shouldVideoGenerateWithLtxApi ? [...FORCED_API_VIDEO_FPS] : [24, 25, 50]

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'image') {
        onInputImageChange(asset.url)
      }
    }
  }

  const handleAudioDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsAudioDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'audio') {
        onInputAudioChange(asset.url)
      }
    }

    // Handle file drops
    const file = e.dataTransfer.files?.[0]
    if (file) {
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (['mp3', 'wav', 'ogg', 'aac', 'flac', 'm4a'].includes(ext || '')) {
        const filePath = (file as any).path as string | undefined
        if (filePath) {
          const normalized = filePath.replace(/\\/g, '/')
          const fileUrl = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
          onInputAudioChange(fileUrl)
        }
      }
    }
  }

  const handleAudioFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const filePath = (file as any).path as string | undefined
      if (filePath) {
        const normalized = filePath.replace(/\\/g, '/')
        const fileUrl = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
        onInputAudioChange(fileUrl)
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      // In Electron, File objects have a .path property with the full filesystem path
      const filePath = (file as any).path as string | undefined
      if (filePath) {
        const normalized = filePath.replace(/\\/g, '/')
        const fileUrl = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
        onInputImageChange(fileUrl)
      } else {
        const url = URL.createObjectURL(file)
        onInputImageChange(url)
      }
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isGenerating && canGenerate) {
      e.preventDefault()
      onGenerate()
    }
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-visible">
      {/* Reference strip — image and audio drop zones above prompt */}
      {mode === 'video' && !isRetake && !isIcLora && (
        <div className="flex items-center gap-2 px-3 pt-2.5 pb-1">
          {/* Input image drop zone */}
          <div
            className={`relative w-10 h-10 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isDragOver ? 'border-blue-500 bg-blue-500/10' : inputImage ? 'border-blue-600' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            title={inputImage ? 'Image attached — click to change' : 'Attach reference image (I2V)'}
          >
            {inputImage ? (
              <>
                <img src={inputImage} alt="" className="w-full h-full object-contain rounded-md" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputImageChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Image className="h-4 w-4 text-zinc-500" />
            )}
            <input ref={inputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
          </div>

          {/* Audio drop zone */}
          <div
            className={`relative w-10 h-10 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isAudioDragOver ? 'border-emerald-500 bg-emerald-500/10' : inputAudio ? 'border-emerald-600' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsAudioDragOver(true) }}
            onDragLeave={() => setIsAudioDragOver(false)}
            onDrop={handleAudioDrop}
            onClick={() => audioInputRef.current?.click()}
            title={inputAudio ? 'Audio attached — click to change' : 'Attach audio for A2V'}
          >
            {inputAudio ? (
              <>
                <Music className="h-4 w-4 text-emerald-400" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputAudioChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Music className="h-4 w-4 text-zinc-500" />
            )}
            <input ref={audioInputRef} type="file" accept=".mp3,.wav,.ogg,.aac,.flac,.m4a" onChange={handleAudioFileSelect} className="hidden" />
          </div>

          {inputImage && <span className="text-[10px] text-blue-400 truncate max-w-[120px]">Image attached</span>}
          {inputAudio && <span className="text-[10px] text-emerald-400 truncate max-w-[120px]">Audio attached</span>}
        </div>
      )}

      {/* Frame conditioning — first / middle / last above prompt */}
      {mode === 'video' && !isRetake && !isIcLora && !shouldVideoGenerateWithLtxApi && conditioningFrames && onConditioningFramesChange && (
        <div className="border-t border-zinc-800/60 px-3 py-3">
          <InlineFrameConditioning
            frames={conditioningFrames}
            onChange={onConditioningFramesChange}
          />
        </div>
      )}

      {/* Prompt area */}
      <div className="px-3 pt-1 pb-0">
        {/* Original / Enhanced tab toggle — shared textarea space */}
        {editableEnhancedPrompt && !isRetake && !isIcLora && (
          <div className="flex items-center gap-0.5 pb-0.5">
            <button
              onClick={() => onUseEnhancedViewChange?.(false)}
              className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                !useEnhancedView ? 'bg-zinc-700 text-white' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              Original
            </button>
            <button
              onClick={() => onUseEnhancedViewChange?.(true)}
              className={`flex items-center gap-0.5 px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                useEnhancedView ? 'bg-purple-600/30 text-purple-300' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <Wand2 className="h-2.5 w-2.5" />
              Enhanced
            </button>
            <button
              onClick={() => { onEditableEnhancedPromptChange?.(''); onUseEnhancedViewChange?.(false) }}
              className="ml-auto text-zinc-600 hover:text-zinc-400"
              title="Clear enhanced prompt"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        )}

        {/* Textarea with expand toggle */}
        <div className="relative">
          <textarea
            value={useEnhancedView && editableEnhancedPrompt ? editableEnhancedPrompt : prompt}
            onChange={(e) => {
              if (useEnhancedView && editableEnhancedPrompt != null) {
                onEditableEnhancedPromptChange?.(e.target.value)
              } else {
                onPromptChange(e.target.value)
              }
            }}
            onKeyDown={handleKeyDown}
            placeholder={mode === 'retake'
              ? "Describe what should happen in the selected section..."
              : mode === 'ic-lora'
                ? "Describe the style or transformation to apply..."
              : mode === 'image'
                ? "A close-up of a woman talking on the phone..."
                : "The woman sips from a cup of coffee..."
            }
            className={`w-full bg-transparent text-sm placeholder:text-zinc-500 focus:outline-none py-2 resize-none leading-5 transition-all duration-150 ${
              isExpanded ? 'h-[200px]' : 'h-[70px]'
            } ${useEnhancedView && editableEnhancedPrompt ? 'text-purple-200' : 'text-white'}`}
          />
          <button
            onClick={() => setIsExpanded(v => !v)}
            className="absolute top-1.5 right-0 text-zinc-600 hover:text-zinc-400 transition-colors"
            title={isExpanded ? 'Collapse prompt' : 'Expand prompt'}
          >
            {isExpanded ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>

        {/* Enhance / Encode / Negative prompt toggle buttons */}
        {!isRetake && !isIcLora && (
          <div className="flex items-center gap-1.5 pb-1.5 flex-wrap">
            {onEnhance && (
              <button
                onClick={onEnhance}
                disabled={isEnhancing || isGenerating || !prompt.trim() || processStatus !== 'alive'}
                title="Enhance prompt with Gemma"
                className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  isEnhancing ? 'bg-purple-900/60 text-purple-300' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
                }`}
              >
                <Wand2 className={`h-3 w-3 ${isEnhancing ? 'animate-pulse' : ''}`} />
                {isEnhancing ? 'Enhancing…' : 'Enhance'}
              </button>
            )}
            {showEncodeButton && onEncode && (
              <button
                onClick={onEncode}
                disabled={isEncoding || isGenerating || !(useEnhancedView ? editableEnhancedPrompt : prompt)?.trim() || processStatus !== 'alive'}
                title={isEncoding ? 'Encoding…' : encodedPrompt && !isPromptChanged ? 'Encoded — click to re-encode' : isPromptChanged ? 'Prompt changed — re-encode' : `Encode ${useEnhancedView ? 'enhanced' : ''} prompt on GPU`}
                className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                  isEncoding ? 'bg-zinc-700 text-zinc-300'
                  : encodedPrompt && !isPromptChanged ? 'bg-emerald-900/60 text-emerald-300 hover:bg-emerald-800/60'
                  : isPromptChanged ? 'bg-amber-900/60 text-amber-300 hover:bg-amber-800/60'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
                }`}
              >
                {isEncoding ? <Cpu className="h-3 w-3 animate-pulse" /> : encodedPrompt && !isPromptChanged ? <CheckCircle className="h-3 w-3" /> : isPromptChanged ? <AlertCircle className="h-3 w-3" /> : <Cpu className="h-3 w-3" />}
                {isEncoding ? 'Encoding…' : encodedPrompt && !isPromptChanged ? 'Encoded' : isPromptChanged ? 'Re-encode' : 'Encode'}
              </button>
            )}
            {enhanceError && <span className="text-[11px] text-red-400 truncate">{enhanceError}</span>}
            <button
              onClick={() => setShowNegativePrompt(v => !v)}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] transition-colors ${showNegativePrompt ? 'bg-zinc-700 text-zinc-300' : 'text-zinc-500 hover:text-zinc-300'}`}
              title="Negative prompt"
            >
              <ChevronDown className={`h-3 w-3 transition-transform ${showNegativePrompt ? 'rotate-180' : ''}`} />
              Negative
            </button>
          </div>
        )}
      </div>

      {/* Negative prompt collapsible */}
      {showNegativePrompt && !isRetake && !isIcLora && (
        <div className="border-t border-zinc-800/60 px-3 py-2">
          <textarea
            value={negativePrompt}
            onChange={(e) => onNegativePromptChange?.(e.target.value)}
            placeholder="What to avoid in the generation..."
            disabled={isGenerating}
            rows={2}
            className="w-full bg-transparent text-white text-xs placeholder:text-zinc-500 focus:outline-none resize-none leading-5 disabled:opacity-50"
          />
        </div>
      )}

      {/* Sampler: steps + STG scale + STG block index */}
      {showSampler && !isRetake && !isIcLora && !shouldVideoGenerateWithLtxApi && (
        <div className="border-t border-zinc-800/60 px-3 py-2.5 flex flex-wrap items-center gap-x-5 gap-y-2">
          {/* Steps */}
          {numSteps != null && onNumStepsChange && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide w-12 shrink-0">Steps</span>
              <div className="flex items-center gap-1">
                {[1,2,3,4,5,6,7,8].map(n => (
                  <button
                    key={n}
                    onClick={() => onNumStepsChange(n)}
                    className={`w-6 h-6 rounded text-[11px] font-medium transition-colors ${
                      numSteps === n
                        ? 'bg-white text-black'
                        : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STG Scale */}
          {stgScale != null && onStgScaleChange && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide w-12 shrink-0">STG</span>
              <div className="flex items-center gap-1">
                {[0, 0.5, 1, 1.5, 2, 2.5, 3].map(n => (
                  <button
                    key={n}
                    onClick={() => onStgScaleChange(n)}
                    className={`px-1.5 h-6 rounded text-[11px] font-medium transition-colors ${
                      stgScale === n
                        ? 'bg-white text-black'
                        : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STG Block Index */}
          {stgBlockIndex != null && onStgBlockIndexChange && stgScale != null && stgScale > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide w-12 shrink-0">Layer</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => onStgBlockIndexChange(Math.max(0, stgBlockIndex - 1))}
                  className="w-6 h-6 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white text-sm leading-none flex items-center justify-center"
                >−</button>
                <span className="w-7 text-center text-sm text-white tabular-nums">{stgBlockIndex}</span>
                <button
                  onClick={() => onStgBlockIndexChange(Math.min(35, stgBlockIndex + 1))}
                  className="w-6 h-6 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white text-sm leading-none flex items-center justify-center"
                >+</button>
              </div>
              <span className="text-[10px] text-zinc-600">/ 35</span>
            </div>
          )}

          {/* Block Swap */}
          {blockSwap != null && onBlockSwapChange && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide w-12 shrink-0">Swap</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => onBlockSwapChange(Math.max(0, blockSwap - 1))}
                  className="w-6 h-6 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white text-sm leading-none flex items-center justify-center"
                >−</button>
                <span className="w-7 text-center text-sm text-white tabular-nums">{blockSwap}</span>
                <button
                  onClick={() => onBlockSwapChange(Math.min(36, blockSwap + 1))}
                  className="w-6 h-6 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white text-sm leading-none flex items-center justify-center"
                >+</button>
              </div>
              <span className="text-[10px] text-zinc-500">blocks on GPU</span>
            </div>
          )}
        </div>
      )}


      {/* Bottom row: Mode selector + Settings */}
      <div className="flex items-center gap-0.5 px-1.5 py-1.5 border-t border-zinc-800/60 text-xs text-zinc-400">
        {/* Mode dropdown */}
        <SettingsDropdown
          title="MODE"
          value={mode}
          onChange={(v) => onModeChange(v as 'image' | 'video' | 'retake' | 'ic-lora')}
          options={[
            { value: 'image', label: 'Generate Images', icon: <Image className="h-4 w-4" /> },
            { value: 'video', label: 'Generate Videos', icon: <Video className="h-4 w-4" /> },
            { value: 'retake', label: 'Retake', icon: <Scissors className="h-4 w-4" /> },
            ...(canUseIcLora ? [{ value: 'ic-lora', label: 'IC-LoRA', icon: <Sparkles className="h-4 w-4" /> }] : []),
          ]}
          trigger={
            <>
              {mode === 'image' ? <Image className="h-3.5 w-3.5" /> : mode === 'retake' ? <Scissors className="h-3.5 w-3.5" /> : mode === 'ic-lora' ? <Sparkles className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
              <span className="text-zinc-300 font-medium">{mode === 'image' ? 'Image' : mode === 'retake' ? 'Retake' : mode === 'ic-lora' ? 'IC-LoRA' : 'Video'}</span>
              <ChevronUp className="h-3 w-3 text-zinc-500" />
            </>
          }
        />
        
        <div className="flex-1" />
        
        {isRetake ? (
          <div className="text-[10px] text-zinc-500 pr-2">Trim in the panel above, then retake</div>
        ) : isIcLora ? (
          <>
            <SettingsDropdown
              title="CONDITIONING TYPE"
              value={icLoraCondType || 'canny'}
              onChange={(v) => onIcLoraCondTypeChange?.(v as ICLoraConditioningType)}
              options={CONDITIONING_TYPES.map(ct => ({ value: ct.value, label: ct.label }))}
              trigger={
                <>
                  <span className="text-zinc-300 font-medium">{CONDITIONING_TYPES.find(ct => ct.value === icLoraCondType)?.label || 'Canny Edges'}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsDropdown
              title="STRENGTH"
              value={String(icLoraStrength ?? 1.0)}
              onChange={(v) => onIcLoraStrengthChange?.(parseFloat(v))}
              options={[
                { value: '0.5', label: '0.50' },
                { value: '0.75', label: '0.75' },
                { value: '1', label: '1.00' },
                { value: '1.25', label: '1.25' },
                { value: '1.5', label: '1.50' },
                { value: '2', label: '2.00' },
              ]}
              trigger={
                <>
                  <span className="text-zinc-500 text-[10px]">STR</span>
                  <span className="text-zinc-300 font-medium">{(icLoraStrength ?? 1.0).toFixed(2)}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
          </>
        ) : mode === 'image' ? (
          <>
            {/* Model indicator */}
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-800/50">
              <ZitIcon className="h-3.5 w-3.5" />
              <span className="text-zinc-300 font-medium">Z-Image Turbo</span>
            </div>
            
            {/* Resolution dropdown */}
            <SettingsDropdown
              title="IMAGE RESOLUTION"
              value={settings.imageResolution}
              onChange={(v) => onSettingsChange({ ...settings, imageResolution: v })}
              options={[
                { value: '1080p', label: '1080p' },
                { value: '1440p', label: '1440p' },
                { value: '2048p', label: '2048p' },
              ]}
              trigger={
                <>
                  <Monitor className="h-3.5 w-3.5" />
                  <span>{settings.imageResolution.replace('p', '')}</span>
                </>
              }
            />
            
            {/* Aspect ratio dropdown */}
            <SettingsDropdown
              title="RATIO"
              value={settings.aspectRatio}
              onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
              options={[
                { value: '16:9', label: '16:9' },
                { value: '1:1', label: '1:1' },
                { value: '9:16', label: '9:16' },
              ]}
              trigger={
                <>
                  <AspectIcon className="h-3.5 w-3.5" />
                  <span>{settings.aspectRatio}</span>
                </>
              }
            />
            
          </>
        ) : (
          <>
            <SettingsDropdown
              title="MODEL"
              value={settings.model}
              onChange={(v) => onSettingsChange({ ...settings, model: v })}
              options={
                shouldVideoGenerateWithLtxApi
                  ? [
                      { value: 'fast', label: 'LTX-2.3 Fast (API)', disabled: !!inputAudio, tooltip: inputAudio ? 'Fast model is not available for Audio-to-Video' : undefined },
                      { value: 'pro', label: 'LTX-2.3 Pro (API)' },
                    ]
                  : [
                      { value: 'fast', label: 'LTX 2.3 Fast' },
                    ]
              }
              trigger={
                <>
                  <LightricksIcon className="h-3.5 w-3.5" />
                  <span className="text-zinc-300 font-medium">
                    {shouldVideoGenerateWithLtxApi
                      ? (settings.model === 'pro' ? 'LTX-2.3 Pro (API)' : 'LTX-2.3 Fast (API)')
                      : 'LTX 2.3 Fast'}
                  </span>
                </>
              }
            />

            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            
            {/* Duration dropdown */}
            <SettingsDropdown
              title="DURATION"
              value={String(settings.duration)}
              onChange={(v) => onSettingsChange({ ...settings, duration: parseFloat(v) })}
              options={videoDurationOptions.map((value) => ({ value: String(value), label: `${value} Sec` }))}
              trigger={
                <>
                  <Clock className="h-3.5 w-3.5" />
                  <span>{settings.duration}s</span>
                </>
              }
            />
            
            {/* Resolution dropdown */}
            <SettingsDropdown
              title="RESOLUTION"
              value={settings.videoResolution}
              onChange={(v) => {
                const maxDur = LOCAL_MAX_DURATION[v] ?? 30
                const clampedDuration = settings.duration > maxDur ? maxDur : settings.duration
                onSettingsChange({ ...settings, videoResolution: v, duration: clampedDuration })
              }}
              options={videoResolutionOptions.map((value) => ({ value, label: value }))}
              trigger={
                <>
                  <Monitor className="h-3.5 w-3.5" />
                  <span>{settings.videoResolution.replace('p', '')}</span>
                </>
              }
            />

            {shouldVideoGenerateWithLtxApi && (
              <SettingsDropdown
                title="FPS"
                value={String(settings.fps)}
                onChange={(v) => onSettingsChange({ ...settings, fps: parseInt(v) })}
                options={videoFpsOptions.map((value) => ({ value: String(value), label: `${value}` }))}
                trigger={
                  <>
                    <Film className="h-3.5 w-3.5" />
                    <span>{settings.fps} FPS</span>
                  </>
                }
              />
            )}
            
            {/* Aspect Ratio dropdown */}
            <SettingsDropdown
              title="ASPECT RATIO"
              value={settings.aspectRatio}
              onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
              options={inputAudio
                ? [{ value: '16:9', label: '16:9' }]
                : [
                    { value: '16:9', label: '16:9' },
                    { value: '9:16', label: '9:16' },
                  ]
              }
              trigger={
                <>
                  <AspectIcon className="h-3.5 w-3.5" />
                  <span>{settings.aspectRatio}</span>
                </>
              }
            />

            {/* Seed input + lock */}
            {onSeedInputChange && onToggleSeedLock && (
              <>
                <div className="w-px h-4 bg-zinc-700 mx-0.5" />
                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-md border transition-colors ${
                  seedLocked
                    ? 'bg-blue-600/10 border-blue-600/30'
                    : 'border-transparent hover:border-zinc-700'
                }`}>
                  <button
                    onClick={onToggleSeedLock}
                    title={seedLocked ? 'Seed locked — click to randomise' : 'Click to lock seed'}
                    className={`transition-colors ${seedLocked ? 'text-blue-400' : 'text-zinc-600 hover:text-zinc-300'}`}
                  >
                    {seedLocked ? <Lock className="h-3 w-3" /> : <Shuffle className="h-3 w-3" />}
                  </button>
                  <input
                    type="text"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    placeholder="seed"
                    value={seedInput ?? ''}
                    onChange={(e) => {
                      const v = e.target.value.replace(/\D/g, '')
                      onSeedInputChange(v)
                    }}
                    className="w-16 bg-transparent text-[10px] font-mono text-zinc-300 placeholder-zinc-600 outline-none"
                  />
                </div>
              </>
            )}

          {/* Sampler toggle (local video only) */}
          {!shouldVideoGenerateWithLtxApi && numSteps != null && (
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
          )}
          {!shouldVideoGenerateWithLtxApi && numSteps != null && onNumStepsChange && (
            <button
              onClick={() => setShowSampler(v => !v)}
              title="Steps &amp; STG settings"
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] transition-colors ${
                showSampler ? 'bg-zinc-700 text-zinc-300' : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <ChevronUp className={`h-3 w-3 transition-transform ${showSampler ? '' : 'rotate-180'}`} />
              Sampler
            </button>
          )}

          </>
        )}

        {/* +Queue button */}
        {onAddToQueue && mode === 'video' && !isRetake && !isIcLora && (
          <button
            onClick={onAddToQueue}
            disabled={isGenerating || !canGenerate || !prompt.trim()}
            title="Add to queue"
            className="flex items-center gap-1 px-2 py-1.5 rounded-md text-xs text-zinc-400 hover:text-white hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
          >
            <ListPlus className="h-3.5 w-3.5" />
            {pendingJobCount > 0 && <span className="text-[10px] font-mono bg-zinc-700 px-1 rounded">{pendingJobCount}</span>}
          </button>
        )}

        {/* Generate button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating || !canGenerate}
          className={`flex items-center gap-1.5 ml-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all flex-shrink-0 ${
            isGenerating || !canGenerate
              ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
              : 'bg-white text-black hover:bg-zinc-200'
          }`}
        >
          <span className={isGenerating ? 'animate-pulse' : ''}>{buttonIcon}</span>
          {buttonLabel}
        </button>
      </div>
    </div>
  )
}

// Gallery size icon components
function GridSmallIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="4" height="4" rx="0.5" />
      <rect x="8" y="2" width="4" height="4" rx="0.5" />
      <rect x="14" y="2" width="4" height="4" rx="0.5" />
      <rect x="20" y="2" width="2" height="4" rx="0.5" />
      <rect x="2" y="8" width="4" height="4" rx="0.5" />
      <rect x="8" y="8" width="4" height="4" rx="0.5" />
      <rect x="14" y="8" width="4" height="4" rx="0.5" />
      <rect x="20" y="8" width="2" height="4" rx="0.5" />
      <rect x="2" y="14" width="4" height="4" rx="0.5" />
      <rect x="8" y="14" width="4" height="4" rx="0.5" />
      <rect x="14" y="14" width="4" height="4" rx="0.5" />
      <rect x="20" y="14" width="2" height="4" rx="0.5" />
    </svg>
  )
}

function GridMediumIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="6" height="6" rx="1" />
      <rect x="10" y="2" width="6" height="6" rx="1" />
      <rect x="18" y="2" width="4" height="6" rx="1" />
      <rect x="2" y="10" width="6" height="6" rx="1" />
      <rect x="10" y="10" width="6" height="6" rx="1" />
      <rect x="18" y="10" width="4" height="6" rx="1" />
      <rect x="2" y="18" width="6" height="4" rx="1" />
      <rect x="10" y="18" width="6" height="4" rx="1" />
      <rect x="18" y="18" width="4" height="4" rx="1" />
    </svg>
  )
}

function GridLargeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="9" height="9" rx="1.5" />
      <rect x="13" y="2" width="9" height="9" rx="1.5" />
      <rect x="2" y="13" width="9" height="9" rx="1.5" />
      <rect x="13" y="13" width="9" height="9" rx="1.5" />
    </svg>
  )
}

type GallerySize = 'small' | 'medium' | 'large'

const gallerySizeClasses: Record<GallerySize, string> = {
  small: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7',
  medium: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5',
  large: 'grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3',
}

const DEFAULT_VIDEO_SETTINGS = {
  model: 'fast',
  duration: 5,
  videoResolution: '540p',
  fps: 24,
  aspectRatio: '16:9',
  imageResolution: '1080p',
  variations: 1,
  audio: true,
  cameraMotion: 'none',
}

export function GenSpace() {
  const {
    currentProject,
    currentProjectId,
    addAsset,
    addTakeToAsset,
    deleteAsset,
    toggleFavorite,
    updateAsset,
    genSpaceEditImageUrl,
    setGenSpaceEditImageUrl,
    setGenSpaceEditMode,
    genSpaceAudioUrl,
    setGenSpaceAudioUrl,
    genSpaceRetakeSource,
    setGenSpaceRetakeSource,
    setPendingRetakeUpdate,
    genSpaceIcLoraSource,
    setGenSpaceIcLoraSource,
    setPendingIcLoraUpdate,
  } = useProjects()
  const { shouldVideoGenerateWithLtxApi, forceApiGenerations, settings: appSettings, updateSettings } = useAppSettings()
  const { processStatus } = useBackend()
  const { jobs: queueJobs, addToQueue, removeFromQueue, pendingCount: queuePendingCount } = useQueue()
  const { isEncoding, encodedPrompt, encodePrompt } = useEncodePrompt()
  const { isEnhancing, enhanceError, enhancePrompt } = useEnhancePrompt()
  const { addToHistory: addToEnhanceHistory } = useEnhancedPromptHistory()
  const { push: pushHistory } = useGenerationHistory()
  const showEncodeButton = appSettings.useLocalTextEncoder
  const [negativePrompt, setNegativePrompt] = useState('')
  const [editableEnhancedPrompt, setEditableEnhancedPrompt] = useState<string | null>(null)
  const [useEnhancedView, setUseEnhancedView] = useState(false)
  const [conditioningFrames, setConditioningFrames] = useState<ConditioningFrame[]>([
    { role: 'first', imagePath: null, imageUrl: null, strength: 1.0, position: 0 },
  ])
  const [numSteps, setNumSteps] = useState(() => appSettings.distilledNumSteps ?? 4)
  const [stgScale, setStgScale] = useState(() => appSettings.stgScale ?? 0)
  const [stgBlockIndex, setStgBlockIndex] = useState(() => appSettings.stgBlockIndex ?? 19)
  const [mode, setMode] = useState<'image' | 'video' | 'retake' | 'ic-lora'>('video')
  const [prompt, setPrompt] = useState('')
  const [inputImage, setInputImage] = useState<string | null>(null)
  const [inputAudio, setInputAudio] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null)
  const [copiedPrompt, setCopiedPrompt] = useState(false)
  const [showFavorites, setShowFavorites] = useState(false)
  const [filterBin, setFilterBin] = useState<string | null>(null)
  const [filterColor, setFilterColor] = useState<string | null>(null)
  const [filterMinRating, setFilterMinRating] = useState(0)
  const [gallerySize, setGallerySize] = useState<GallerySize>('medium')
  const [showSizeMenu, setShowSizeMenu] = useState(false)
  const sizeMenuRef = useRef<HTMLDivElement>(null)
  const persistedVideoKeyRef = useRef<string | null>(null)
  const persistedQueueJobIds = useRef<Set<string>>(new Set())
  const retakeSubmissionRef = useRef<{
    prompt: string
    input: {
      videoPath: string | null
      startTime: number
      duration: number
      videoDuration: number
    }
  } | null>(null)
  const icLoraSubmissionRef = useRef<{
    prompt: string
    input: {
      videoPath: string
      conditioningType: ICLoraConditioningType
      conditioningStrength: number
    }
  } | null>(null)
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_VIDEO_SETTINGS }))
  const [seedInput, setSeedInput] = useState('')
  const [seedLocked, setSeedLocked] = useState(false)
  const applyForcedVideoSettings = useCallback(
    (next: { model: string; duration: number; videoResolution: string; fps: number; audio: boolean; aspectRatio: string; imageResolution: string; variations: number; cameraMotion: string }) => {
      if (!shouldVideoGenerateWithLtxApi || mode !== 'video') return next
      return sanitizeForcedApiVideoSettings(next, { hasAudio: !!inputAudio })
    },
    [inputAudio, mode, shouldVideoGenerateWithLtxApi],
  )
  
  const activePromptForEncode = (useEnhancedView && editableEnhancedPrompt) ? editableEnhancedPrompt : prompt
  const isPromptChanged = showEncodeButton && encodedPrompt !== null && activePromptForEncode.trim() !== encodedPrompt

  const {
    generate,
    generateImage,
    isGenerating,
    progress,
    statusMessage,
    videoUrl,
    videoPath,
    imageUrls,
    imagePaths,
    error,
    reset,
    enhancedPrompt,
  } = useGeneration({
    onGenerationSuccess: ({ seedUsed }) => {
      if (seedUsed != null && !seedLocked) setSeedInput(String(seedUsed))
    },
  })

  // Sync editable enhanced prompt when generation returns one
  useEffect(() => {
    if (enhancedPrompt !== null) {
      setEditableEnhancedPrompt(enhancedPrompt)
      addToEnhanceHistory(enhancedPrompt)
    }
  }, [enhancedPrompt, addToEnhanceHistory])

  const {
    submitRetake,
    resetRetake,
    isRetaking,
    retakeStatus,
    retakeError,
    retakeResult,
  } = useRetake()

  const [retakeInput, setRetakeInput] = useState({
    videoUrl: null as string | null,
    videoPath: null as string | null,
    startTime: 0,
    duration: 0,
    videoDuration: 0,
    ready: false,
  })
  const [retakePanelKey, setRetakePanelKey] = useState(0)
  const [retakeInitial, setRetakeInitial] = useState<{
    videoUrl: string | null
    videoPath: string | null
    duration?: number
  }>({ videoUrl: null, videoPath: null, duration: undefined })
  const [activeRetakeSource, setActiveRetakeSource] = useState<GenSpaceRetakeSource | null>(null)
  const [activeIcLoraSource, setActiveIcLoraSource] = useState<{
    assetId?: string
    linkedClipIds?: string[]
  } | null>(null)
  const [icLoraInput, setIcLoraInput] = useState({
    videoUrl: null as string | null,
    videoPath: null as string | null,
    conditioningType: 'canny' as ICLoraConditioningType,
    conditioningStrength: 1.0,
    ready: false,
  })
  const [icLoraPanelKey, setIcLoraPanelKey] = useState(0)
  const [icLoraCondType, setIcLoraCondType] = useState<ICLoraConditioningType>('canny')
  const [icLoraStrength, setIcLoraStrength] = useState(1.0)
  const [icLoraInitial, setIcLoraInitial] = useState<{
    videoUrl: string | null
    videoPath: string | null
  }>({ videoUrl: null, videoPath: null })

  const {
    submitIcLora,
    resetIcLora,
    isIcLoraGenerating,
    icLoraStatus,
    icLoraError,
    icLoraResult,
  } = useIcLora()
  
  // Handle incoming frame from the Video Editor for editing
  useEffect(() => {
    if (genSpaceEditImageUrl) {
      setMode('video')
      setInputImage(genSpaceEditImageUrl)
      setPrompt('')
      setGenSpaceEditImageUrl(null)
      setGenSpaceEditMode(null)
    }
  }, [genSpaceEditImageUrl, setGenSpaceEditImageUrl, setGenSpaceEditMode])

  // Handle incoming audio from the Video Editor for A2V
  useEffect(() => {
    if (genSpaceAudioUrl) {
      setMode('video')
      setInputAudio(genSpaceAudioUrl)
      setPrompt('')
      setGenSpaceAudioUrl(null)
    }
  }, [genSpaceAudioUrl, setGenSpaceAudioUrl])

  useEffect(() => {
    if (!genSpaceRetakeSource) return
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(genSpaceRetakeSource)
    setRetakeInitial({
      videoUrl: genSpaceRetakeSource.videoUrl,
      videoPath: genSpaceRetakeSource.videoPath,
      duration: genSpaceRetakeSource.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
    setGenSpaceRetakeSource(null)
  }, [genSpaceRetakeSource, setGenSpaceRetakeSource])

  useEffect(() => {
    if (!genSpaceIcLoraSource) return
    if (forceApiGenerations) {
      setGenSpaceIcLoraSource(null)
      return
    }
    setMode('ic-lora')
    setPrompt('')
    setActiveIcLoraSource({
      assetId: genSpaceIcLoraSource.assetId,
      linkedClipIds: genSpaceIcLoraSource.linkedClipIds,
    })
    setIcLoraInitial({
      videoUrl: genSpaceIcLoraSource.videoUrl,
      videoPath: genSpaceIcLoraSource.videoPath,
    })
    setIcLoraPanelKey((prev) => prev + 1)
    setGenSpaceIcLoraSource(null)
  }, [genSpaceIcLoraSource, forceApiGenerations, setGenSpaceIcLoraSource])

  useEffect(() => {
    if (forceApiGenerations && mode === 'ic-lora') {
      setMode('video')
    }
  }, [forceApiGenerations, mode])

  useEffect(() => {
    if (!shouldVideoGenerateWithLtxApi || mode !== 'video') return
    setSettings((prev) => applyForcedVideoSettings({ ...prev, model: 'fast' }))
  }, [applyForcedVideoSettings, mode, shouldVideoGenerateWithLtxApi])

  useEffect(() => {
    if (retakeError) {
      setLocalError(retakeError)
    }
  }, [retakeError])

  useEffect(() => {
    if (icLoraError) {
      setLocalError(icLoraError)
    }
  }, [icLoraError])

  // Force pro model + resolution when audio is attached (A2V only supports pro @ 1080p 16:9)
  useEffect(() => {
    if (inputAudio) {
      setSettings(prev => applyForcedVideoSettings({ ...prev, model: 'pro', aspectRatio: '16:9' }))
    }
  }, [inputAudio]) // eslint-disable-line react-hooks/exhaustive-deps

  // Only show assets that were generated (have generationParams), not imported files
  const assets = (currentProject?.assets || []).filter(a => a.generationParams)
  const [lastPrompt, setLastPrompt] = useState('')
  
  // When video generation completes, add to project assets
  useEffect(() => {
    if (!videoUrl || !videoPath || !currentProjectId || isGenerating) return

    const generationKey = `${videoUrl}|${videoPath}`
    if (persistedVideoKeyRef.current === generationKey) return
    persistedVideoKeyRef.current = generationKey

    const genMode = inputAudio
      ? 'audio-to-video'
      : inputImage ? 'image-to-video' : 'text-to-video'
    const savedVideoSettings = applyForcedVideoSettings(settings)

    ;(async () => {
      try {
        const copied = await copyToAssetFolder(videoPath, currentProjectId)
        const finalPath = copied?.path ?? videoPath
        const finalUrl = copied?.url ?? videoUrl
        addAsset(currentProjectId, {
          type: 'video',
          path: finalPath,
          url: finalUrl,
          prompt: lastPrompt,
          resolution: savedVideoSettings.videoResolution,
          duration: savedVideoSettings.duration,
          generationParams: {
            mode: genMode as 'text-to-video' | 'image-to-video' | 'audio-to-video',
            prompt: lastPrompt,
            model: savedVideoSettings.model,
            duration: savedVideoSettings.duration,
            resolution: savedVideoSettings.videoResolution,
            fps: savedVideoSettings.fps,
            audio: savedVideoSettings.audio || false,
            cameraMotion: 'none',
            imageAspectRatio: savedVideoSettings.aspectRatio,
            imageSteps: 4,
            inputImageUrl: inputImage || undefined,
            inputAudioUrl: inputAudio || undefined,
          },
          takes: [{
            url: finalUrl,
            path: finalPath,
            createdAt: Date.now(),
          }],
          activeTakeIndex: 0,
        })
        reset()
      } catch (err) {
        persistedVideoKeyRef.current = null
        logger.error(`Failed to persist generated video asset: ${err}`)
      }
    })()
  }, [videoUrl, videoPath, currentProjectId, isGenerating, applyForcedVideoSettings, settings, inputImage, inputAudio, lastPrompt, addAsset, reset])

  // When a queued job completes, add its result to the gallery
  useEffect(() => {
    if (!currentProjectId) return
    const newlyCompleted = queueJobs.filter(
      j => j.status === 'complete' && j.result_path && !persistedQueueJobIds.current.has(j.id)
    )
    if (newlyCompleted.length === 0) return

    for (const job of newlyCompleted) {
      persistedQueueJobIds.current.add(job.id)
      const resultPath = job.result_path!
      pushHistory({
        prompt: job.prompt,
        negativePrompt: '',
        settings: {
          model: (job.model as 'fast' | 'pro' | 'dev') || 'fast',
          duration: parseInt(job.duration) || 5,
          videoResolution: job.resolution || '720p',
          fps: parseInt(job.fps) || 24,
          audio: false,
          cameraMotion: 'none',
          aspectRatio: job.aspect_ratio || '16:9',
          seed: null,
          imageResolution: '1080p',
          imageAspectRatio: '16:9',
          imageSteps: 4,
        },
        seedUsed: null,
        videoPath: resultPath,
        lorasUsed: parseJobLoras(job.civitai_loras_snapshot),
      })
      ;(async () => {
        try {
          const copied = await copyToAssetFolder(resultPath, currentProjectId)
          const finalPath = copied?.path ?? resultPath
          const normalized = finalPath.replace(/\\/g, '/')
          const finalUrl = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
          const jobDuration = parseInt(job.duration) || 5
          const jobFps = parseInt(job.fps) || 24
          addAsset(currentProjectId, {
            type: 'video',
            path: finalPath,
            url: finalUrl,
            prompt: job.prompt,
            resolution: job.resolution || '720p',
            duration: jobDuration,
            generationParams: {
              mode: 'text-to-video',
              prompt: job.prompt,
              model: job.model || 'fast',
              duration: jobDuration,
              resolution: job.resolution || '720p',
              fps: jobFps,
              audio: false,
              cameraMotion: 'none',
              imageAspectRatio: job.aspect_ratio || '16:9',
              imageSteps: 4,
            },
            takes: [{ url: finalUrl, path: finalPath, createdAt: Date.now() }],
            activeTakeIndex: 0,
          })
        } catch (err) {
          persistedQueueJobIds.current.delete(job.id)
          logger.error(`Failed to persist queued job asset: ${err}`)
        }
      })()
    }
  }, [queueJobs, currentProjectId, addAsset, pushHistory])

  // When retake completes, add as take or new asset
  useEffect(() => {
    if (!retakeResult || !currentProjectId || isRetaking) return
    const submission = retakeSubmissionRef.current
    if (!submission) return
    retakeSubmissionRef.current = null

    ;(async () => {
      const usedPrompt = submission.prompt
      const usedInput = submission.input
      const copied = await copyToAssetFolder(retakeResult.videoPath, currentProjectId)
      const finalPath = copied?.path ?? retakeResult.videoPath
      const finalUrl = copied?.url ?? retakeResult.videoUrl

      if (activeRetakeSource?.assetId) {
        const sourceAsset = currentProject?.assets?.find(a => a.id === activeRetakeSource.assetId)
        if (sourceAsset) {
          const newTakeIndex = sourceAsset.takes ? sourceAsset.takes.length : 1
          addTakeToAsset(currentProjectId, sourceAsset.id, {
            url: finalUrl,
            path: finalPath,
            createdAt: Date.now(),
          })
          if (activeRetakeSource.linkedClipIds?.length) {
            setPendingRetakeUpdate({
              assetId: sourceAsset.id,
              clipIds: activeRetakeSource.linkedClipIds,
              newTakeIndex,
            })
          }
        }
      } else {
        addAsset(currentProjectId, {
          type: 'video',
          path: finalPath,
          url: finalUrl,
          prompt: usedPrompt,
          resolution: '',
          duration: usedInput.duration,
          generationParams: {
            mode: 'retake',
            prompt: usedPrompt,
            model: 'pro',
            duration: usedInput.duration,
            resolution: '',
            fps: 24,
            audio: true,
            cameraMotion: 'none',
            retakeVideoPath: finalPath,
            retakeStartTime: usedInput.startTime,
            retakeDuration: usedInput.duration,
            retakeMode: 'replace_audio_and_video',
          },
          takes: [{ url: finalUrl, path: finalPath, createdAt: Date.now() }],
          activeTakeIndex: 0,
        })
        setMode('video')
      }

      setActiveRetakeSource(null)
      resetRetake()
    })()
  }, [retakeResult, isRetaking, currentProjectId, currentProject?.assets, activeRetakeSource, addAsset, addTakeToAsset, setPendingRetakeUpdate, resetRetake])

  useEffect(() => {
    if (!icLoraResult || !currentProjectId || isIcLoraGenerating) return
    const submission = icLoraSubmissionRef.current
    if (!submission) return
    icLoraSubmissionRef.current = null

    ;(async () => {
      const copied = await copyToAssetFolder(icLoraResult.videoPath, currentProjectId)
      const finalPath = copied?.path ?? icLoraResult.videoPath
      const finalUrl = copied?.url ?? icLoraResult.videoUrl

      if (activeIcLoraSource?.assetId) {
        const sourceAsset = currentProject?.assets?.find(a => a.id === activeIcLoraSource.assetId)
        if (sourceAsset) {
          const newTakeIndex = sourceAsset.takes ? sourceAsset.takes.length : 1
          addTakeToAsset(currentProjectId, sourceAsset.id, {
            url: finalUrl,
            path: finalPath,
            createdAt: Date.now(),
          })
          if (activeIcLoraSource.linkedClipIds?.length) {
            setPendingIcLoraUpdate({
              assetId: sourceAsset.id,
              clipIds: activeIcLoraSource.linkedClipIds,
              newTakeIndex,
            })
          }
        }
      } else {
        addAsset(currentProjectId, {
          type: 'video',
          path: finalPath,
          url: finalUrl,
          prompt: submission.prompt,
          resolution: '',
          generationParams: {
            mode: 'ic-lora',
            prompt: submission.prompt,
            model: 'fast',
            duration: 0,
            resolution: '',
            fps: 24,
            audio: false,
            cameraMotion: 'none',
            icLoraVideoPath: submission.input.videoPath,
            icLoraConditioningType: submission.input.conditioningType,
            icLoraConditioningStrength: submission.input.conditioningStrength,
          },
          takes: [{ url: finalUrl, path: finalPath, createdAt: Date.now() }],
          activeTakeIndex: 0,
        })
      }

      setActiveIcLoraSource(null)
    })()
  }, [icLoraResult, isIcLoraGenerating, currentProjectId, currentProject?.assets, activeIcLoraSource, addAsset, addTakeToAsset, setPendingIcLoraUpdate])
  
  // When image generation/editing completes, add all images to project assets
  useEffect(() => {
    if (imageUrls.length > 0 && currentProjectId && !isGenerating) {
      const genMode = 'text-to-image'
      ;(async () => {
        for (let i = 0; i < imageUrls.length; i++) {
          const imageUrl = imageUrls[i]
          const imgPath = imagePaths[i] || null
          const exists = assets.some(a => a.url === imageUrl)
          if (!exists) {
            const copied = imgPath ? await copyToAssetFolder(imgPath, currentProjectId) : null
            const finalPath = copied?.path ?? imgPath ?? imageUrl
            const finalUrl = copied?.url ?? imageUrl
            addAsset(currentProjectId, {
              type: 'image',
              path: finalPath,
              url: finalUrl,
              prompt: lastPrompt,
              resolution: settings.imageResolution,
              generationParams: {
                mode: genMode,
                prompt: lastPrompt,
                model: 'fast',
                duration: 5,
                resolution: settings.imageResolution,
                fps: 24,
                audio: false,
                cameraMotion: 'none',
                imageAspectRatio: settings.aspectRatio,
                imageSteps: 4,
              },
              takes: [{
                url: finalUrl,
                path: finalPath,
                createdAt: Date.now(),
              }],
              activeTakeIndex: 0,
            })
          }
        }
      })()
    }
  }, [imageUrls, imagePaths, currentProjectId, isGenerating])
  
  const handleGenerate = async () => {
    if (mode === 'ic-lora') {
      if (!prompt.trim() || !icLoraInput.videoPath || !icLoraInput.ready) return
      icLoraSubmissionRef.current = {
        prompt,
        input: {
          videoPath: icLoraInput.videoPath,
          conditioningType: icLoraCondType,
          conditioningStrength: icLoraStrength,
        },
      }
      await submitIcLora({
        videoPath: icLoraInput.videoPath,
        conditioningType: icLoraCondType,
        conditioningStrength: icLoraStrength,
        prompt,
      })
      return
    }

    if (mode === 'retake') {
      if (!retakeInput.videoPath || retakeInput.duration < 2) return
      retakeSubmissionRef.current = {
        prompt,
        input: {
          videoPath: retakeInput.videoPath,
          startTime: retakeInput.startTime,
          duration: retakeInput.duration,
          videoDuration: retakeInput.videoDuration,
        },
      }
      await submitRetake({
        videoPath: retakeInput.videoPath,
        startTime: retakeInput.startTime,
        duration: retakeInput.duration,
        prompt,
        mode: 'replace_audio_and_video',
      })
      return
    }

    if (!prompt.trim()) return

    // Save the prompt before generation starts
    setLastPrompt(prompt)

    if (mode === 'image') {
      generateImage(
        prompt,
        {
          model: 'fast' as 'fast' | 'pro' | 'dev',
          duration: 5,
          videoResolution: settings.videoResolution,
          fps: 24,
          audio: false,
          cameraMotion: 'none',
          imageResolution: settings.imageResolution,
          imageAspectRatio: settings.aspectRatio,
          imageSteps: appSettings.distilledNumSteps ?? 4,
          variations: settings.variations,
          seed: null,
        }
      )
    } else {
      // Generate video (t2v if no image/audio, i2v if image, a2v if audio)
      // Extract filesystem path from the file:// URL for the backend
      const imagePath = inputImage ? fileUrlToPath(inputImage) : null
      const audioPath = inputAudio ? fileUrlToPath(inputAudio) : null
      const videoSettings = applyForcedVideoSettings(settings)
      if (audioPath) videoSettings.model = 'pro'

      generate(
        prompt,
        imagePath,
        {
          model: videoSettings.model as 'fast' | 'pro' | 'dev',
          duration: videoSettings.duration,
          videoResolution: videoSettings.videoResolution,
          fps: videoSettings.fps,
          audio: videoSettings.audio || false,
          cameraMotion: videoSettings.cameraMotion ?? 'none',
          aspectRatio: videoSettings.aspectRatio,
          imageResolution: videoSettings.imageResolution,
          imageAspectRatio: videoSettings.aspectRatio,
          imageSteps: appSettings.distilledNumSteps ?? 4,
          seed: seedInput ? parseInt(seedInput) : null,
          numSteps: shouldVideoGenerateWithLtxApi ? undefined : numSteps,
          stgScale: shouldVideoGenerateWithLtxApi ? undefined : stgScale,
          stgBlockIndex: shouldVideoGenerateWithLtxApi ? undefined : stgBlockIndex,
        },
        audioPath,
        negativePrompt,
        useEnhancedView ? editableEnhancedPrompt : null,
        conditioningFrames,
      )
    }
  }
  
  const handleAddToQueue = () => {
    if (!prompt.trim() || mode !== 'video') return
    const videoSettings = applyForcedVideoSettings(settings)
    const imagePath = inputImage ? fileUrlToPath(inputImage) : null
    const audioPath = inputAudio ? fileUrlToPath(inputAudio) : null
    if (audioPath) videoSettings.model = 'pro'
    void addToQueue(
      prompt,
      imagePath,
      {
        model: videoSettings.model as 'fast' | 'pro' | 'dev',
        duration: videoSettings.duration,
        videoResolution: videoSettings.videoResolution,
        fps: videoSettings.fps,
        audio: videoSettings.audio || false,
        cameraMotion: videoSettings.cameraMotion ?? 'none',
        aspectRatio: videoSettings.aspectRatio,
        imageResolution: videoSettings.imageResolution,
        imageAspectRatio: videoSettings.aspectRatio,
        imageSteps: appSettings.distilledNumSteps ?? 4,
        seed: seedInput ? parseInt(seedInput) : null,
        numSteps: shouldVideoGenerateWithLtxApi ? undefined : numSteps,
        stgScale: shouldVideoGenerateWithLtxApi ? undefined : stgScale,
      },
      audioPath,
      negativePrompt,
      conditioningFrames,
    )
  }

  const handleDelete = (assetId: string) => {
    if (currentProjectId) {
      deleteAsset(currentProjectId, assetId)
    }
  }
  
  const handleDragStart = (e: React.DragEvent, asset: Asset) => {
    e.dataTransfer.setData('asset', JSON.stringify(asset))
    e.dataTransfer.setData('assetId', asset.id)
    e.dataTransfer.effectAllowed = 'copy'
  }
  
  const handleCreateVideo = (imageAsset: Asset) => {
    setMode('video')
    setInputImage(imageAsset.url)
    setPrompt(`${imageAsset.prompt || 'The scene comes to life...'}`)
  }

  const handleRetake = (videoAsset: Asset) => {
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(null)
    setRetakeInitial({
      videoUrl: videoAsset.url,
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
  }

  const handleIcLora = (videoAsset: Asset) => {
    if (forceApiGenerations) return
    setMode('ic-lora')
    setPrompt('')
    setActiveIcLoraSource(null)
    setIcLoraInitial({ videoUrl: videoAsset.url, videoPath: videoAsset.path })
    setIcLoraPanelKey((prev) => prev + 1)
  }

  const isRetakeMode = mode === 'retake'
  const isIcLoraMode = mode === 'ic-lora'
  const canSubmit = isRetakeMode
    ? retakeInput.ready && !!retakeInput.videoPath && !isRetaking
    : isIcLoraMode
      ? !!prompt.trim() && icLoraInput.ready && !!icLoraInput.videoPath && !isIcLoraGenerating
      : !!prompt.trim()
  const promptButtonLabel = isRetakeMode ? 'Retake' : isIcLoraMode ? 'Generate' : 'Generate'
  const promptButtonIcon = isRetakeMode
    ? <Scissors className="h-3.5 w-3.5" />
    : isIcLoraMode
      ? <Sparkles className="h-3.5 w-3.5" />
    : <Sparkles className={`h-3.5 w-3.5 ${isGenerating ? 'animate-pulse' : ''}`} />
  const promptGenerating = isRetakeMode ? isRetaking : isIcLoraMode ? isIcLoraGenerating : isGenerating
  
  // Close size menu on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sizeMenuRef.current && !sizeMenuRef.current.contains(e.target as Node)) {
        setShowSizeMenu(false)
      }
    }
    if (showSizeMenu) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showSizeMenu])

  const bins = useMemo(() => [...new Set(assets.filter(a => a.bin).map(a => a.bin!))], [assets])
  const filteredAssets = useMemo(() => assets.filter(a => {
    if (showFavorites && !a.favorite) return false
    if (filterBin && a.bin !== filterBin) return false
    if (filterColor && a.colorLabel !== filterColor) return false
    if (filterMinRating > 0 && (a.rating ?? 0) < filterMinRating) return false
    return true
  }), [assets, showFavorites, filterBin, filterColor, filterMinRating])
  const favoriteCount = assets.filter(a => a.favorite).length
  const isLibraryMode = mode === 'video' || mode === 'image'

  // Navigation for the asset preview modal
  const selectedIndex = selectedAsset ? filteredAssets.findIndex(a => a.id === selectedAsset.id) : -1
  const canGoPrev = selectedIndex > 0
  const canGoNext = selectedIndex >= 0 && selectedIndex < filteredAssets.length - 1

  const goToPrev = useCallback(() => {
    if (canGoPrev) setSelectedAsset(filteredAssets[selectedIndex - 1])
  }, [canGoPrev, filteredAssets, selectedIndex])

  const goToNext = useCallback(() => {
    if (canGoNext) setSelectedAsset(filteredAssets[selectedIndex + 1])
  }, [canGoNext, filteredAssets, selectedIndex])

  // Keyboard navigation for the preview modal
  useEffect(() => {
    if (!selectedAsset) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); goToPrev() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); goToNext() }
      else if (e.key === 'Escape') setSelectedAsset(null)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [selectedAsset, goToPrev, goToNext])

  return (
    <div className="h-full relative bg-zinc-950">

      {/* Empty state */}
      {isLibraryMode && assets.length === 0 && !isGenerating && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <div className="w-24 h-24 rounded-2xl border-2 border-dashed border-zinc-700 flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 text-zinc-600" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Start Creating</h3>
          <p className="text-zinc-500 max-w-md">
            Use the prompt bar below to generate images and videos.
            Drag assets into the input box to use them as references.
          </p>
        </div>
      )}

      {/* No results empty state (filters active) */}
      {isLibraryMode && filteredAssets.length === 0 && assets.length > 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <Heart className="h-12 w-12 text-zinc-700 mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">No matches</h3>
          <p className="text-zinc-500 text-sm">
            Try adjusting your filters.
          </p>
        </div>
      )}

      {/* Assets area — full width, no background, above the prompt bar */}
      {isLibraryMode && (assets.length > 0 || isGenerating) && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] flex flex-col px-4 pt-4">
          {/* Top bar */}
          <div className="flex items-center justify-end pb-2 gap-2 flex-wrap">
            {/* Bin pills */}
            {bins.length > 0 && (
              <div className="flex items-center gap-1 flex-wrap">
                <button
                  onClick={() => setFilterBin(null)}
                  className={`px-2 py-1 rounded-md text-xs transition-colors ${!filterBin ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-800'}`}
                >
                  All
                </button>
                {bins.map(bin => (
                  <button
                    key={bin}
                    onClick={() => setFilterBin(filterBin === bin ? null : bin)}
                    className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-colors ${filterBin === bin ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-800'}`}
                  >
                    <FolderPlus className="h-3 w-3" />
                    {bin}
                  </button>
                ))}
              </div>
            )}

            {/* Color filter dots */}
            <div className="flex items-center gap-0.5 p-1 rounded-md bg-zinc-900">
              <button
                onClick={() => setFilterColor(null)}
                className={`w-4 h-4 rounded-full border transition-all ${!filterColor ? 'border-white scale-110' : 'border-zinc-600 hover:border-white'} bg-zinc-700`}
                title="All colors"
              />
              {COLOR_LABELS.map(cl => (
                <button
                  key={cl.id}
                  onClick={() => setFilterColor(filterColor === cl.id ? null : cl.id)}
                  className={`w-4 h-4 rounded-full border transition-all ${filterColor === cl.id ? 'border-white scale-110' : 'border-transparent hover:border-white'}`}
                  style={{ backgroundColor: cl.color }}
                  title={cl.label}
                />
              ))}
            </div>

            {/* Star filter */}
            <div className="flex items-center gap-0.5 p-1 rounded-md bg-zinc-900">
              {[0,1,2,3,4,5].map(n => (
                <button
                  key={n}
                  onClick={() => setFilterMinRating(filterMinRating === n ? 0 : n)}
                  className="p-0.5"
                  title={n === 0 ? 'All ratings' : `${n}+ stars`}
                >
                  {n === 0
                    ? <span className={`text-xs font-medium ${filterMinRating === 0 ? 'text-white' : 'text-zinc-500 hover:text-white'}`}>All</span>
                    : <Star className={`h-3.5 w-3.5 transition-colors ${n <= filterMinRating ? 'text-yellow-400 fill-current' : 'text-zinc-500 hover:text-yellow-400'}`} />
                  }
                </button>
              ))}
            </div>

            <button
              onClick={() => setShowFavorites(!showFavorites)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                showFavorites
                  ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
            >
              <Heart className={`h-4 w-4 ${showFavorites ? 'fill-current' : ''}`} />
              Favorites
              {favoriteCount > 0 && (
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                  showFavorites ? 'bg-red-500/30 text-red-300' : 'bg-zinc-800 text-zinc-500'
                }`}>
                  {favoriteCount}
                </span>
              )}
            </button>

            <div ref={sizeMenuRef} className="relative">
              <button
                onClick={() => setShowSizeMenu(!showSizeMenu)}
                className={`p-2 rounded-md transition-colors ${
                  showSizeMenu ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
                }`}
              >
                {gallerySize === 'small' ? <GridSmallIcon className="h-4 w-4" /> :
                 gallerySize === 'medium' ? <GridMediumIcon className="h-4 w-4" /> :
                 <GridLargeIcon className="h-4 w-4" />}
              </button>

              {showSizeMenu && (
                <div className="absolute top-full mt-2 right-0 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[160px] shadow-xl z-50">
                  {([
                    { value: 'small' as GallerySize, label: 'Small', icon: GridSmallIcon },
                    { value: 'medium' as GallerySize, label: 'Medium', icon: GridMediumIcon },
                    { value: 'large' as GallerySize, label: 'Large', icon: GridLargeIcon },
                  ]).map(option => (
                    <button
                      key={option.value}
                      onClick={() => { setGallerySize(option.value); setShowSizeMenu(false) }}
                      className={`w-full flex items-center justify-between px-2 py-2.5 rounded-md transition-colors text-left ${gallerySize === option.value ? 'bg-white/20 hover:bg-white/25' : 'hover:bg-zinc-700'}`}
                    >
                      <div className="flex items-center gap-3">
                        <option.icon className={`h-4 w-4 ${gallerySize === option.value ? 'text-white' : 'text-zinc-500'}`} />
                        <span className={`text-sm ${gallerySize === option.value ? 'text-white font-medium' : 'text-zinc-400'}`}>
                          {option.label}
                        </span>
                      </div>
                      {gallerySize === option.value && (
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Assets grid — fills remaining space, scrollable */}
          <div className="overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable] flex-1">
            <div className={`grid ${gallerySizeClasses[gallerySize]} gap-4`}>
              {isGenerating && (
                <div className="relative rounded-xl overflow-hidden bg-zinc-800 aspect-video">
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="relative w-16 h-16 mb-3">
                      <div className="absolute inset-0 rounded-full border-2 border-violet-500/30" />
                      <div className="absolute inset-0 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
                      <div className="absolute inset-2 rounded-full bg-zinc-800 flex items-center justify-center">
                        <Sparkles className="h-6 w-6 text-violet-400" />
                      </div>
                    </div>
                    <p className="text-sm text-zinc-400">{statusMessage || 'Generating...'}</p>
                    {progress > 0 && (
                      <div className="w-32 h-1 bg-zinc-800 rounded-full mt-2 overflow-hidden">
                        <div className="h-full bg-violet-500 transition-all" style={{ width: `${progress}%` }} />
                      </div>
                    )}
                    {enhancedPrompt && (
                      <div className="mt-3 mx-4 px-3 py-2 bg-zinc-900/80 border border-purple-500/20 rounded-lg max-w-xs">
                        <p className="text-[10px] text-purple-400 font-medium mb-1">Enhanced prompt</p>
                        <p className="text-[10px] text-zinc-400 leading-relaxed line-clamp-4">{enhancedPrompt}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {filteredAssets.map(asset => (
                <AssetCard
                  key={asset.id}
                  asset={asset}
                  onDelete={() => handleDelete(asset.id)}
                  onPlay={() => setSelectedAsset(asset)}
                  onDragStart={handleDragStart}
                  onCreateVideo={handleCreateVideo}
                  onRetake={handleRetake}
                  onIcLora={!forceApiGenerations ? handleIcLora : undefined}
                  onToggleFavorite={() => currentProjectId && toggleFavorite(currentProjectId, asset.id)}
                  onSetRating={(rating) => currentProjectId && updateAsset(currentProjectId, asset.id, { rating })}
                  onSetNotes={(notes) => currentProjectId && updateAsset(currentProjectId, asset.id, { notes })}
                  onSetColorLabel={(colorLabel) => currentProjectId && updateAsset(currentProjectId, asset.id, { colorLabel })}
                  onSetBin={(bin) => currentProjectId && updateAsset(currentProjectId, asset.id, { bin })}
                  allBins={bins}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {mode === 'retake' && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <RetakePanel
            initialVideoUrl={retakeInitial.videoUrl}
            initialVideoPath={retakeInitial.videoPath}
            initialDuration={retakeInitial.duration}
            resetKey={retakePanelKey}
            fillHeight
            isProcessing={isRetaking}
            processingStatus={retakeStatus}
            onChange={(data) => setRetakeInput(data)}
          />
        </div>
      )}

      {mode === 'ic-lora' && !forceApiGenerations && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <ICLoraPanel
            initialVideoUrl={icLoraInitial.videoUrl}
            initialVideoPath={icLoraInitial.videoPath}
            resetKey={icLoraPanelKey}
            fillHeight
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
        </div>
      )}

      {/* Floating prompt panel — wider, responsive, centered */}
      <div className="absolute bottom-5 left-1/2 w-[min(700px,calc(100%-2rem))] -translate-x-1/2">

        <FreeApiKeyBubble
          forceApiGenerations={forceApiGenerations}
          hasLtxApiKey={appSettings.hasLtxApiKey}
          isGenerating={isGenerating}
        />

        <QueuePanel jobs={queueJobs} onRemove={removeFromQueue} />

        {/* Prompt bar */}
        <PromptBar
          mode={mode}
          onModeChange={setMode}
          canUseIcLora={!forceApiGenerations}
          prompt={prompt}
          onPromptChange={setPrompt}
          onGenerate={handleGenerate}
          isGenerating={promptGenerating}
          canGenerate={canSubmit}
          buttonLabel={promptButtonLabel}
          buttonIcon={promptButtonIcon}
          inputImage={inputImage}
          onInputImageChange={setInputImage}
          inputAudio={inputAudio}
          onInputAudioChange={setInputAudio}
          settings={settings}
          onSettingsChange={(nextSettings) => setSettings(applyForcedVideoSettings(nextSettings))}
          shouldVideoGenerateWithLtxApi={shouldVideoGenerateWithLtxApi}
          icLoraCondType={icLoraCondType}
          onIcLoraCondTypeChange={setIcLoraCondType}
          icLoraStrength={icLoraStrength}
          onIcLoraStrengthChange={setIcLoraStrength}
          seedInput={seedInput}
          onSeedInputChange={(v) => { setSeedInput(v); if (v) setSeedLocked(true); else setSeedLocked(false) }}
          seedLocked={seedLocked}
          onToggleSeedLock={() => { setSeedLocked(v => !v) }}
          negativePrompt={negativePrompt}
          onNegativePromptChange={setNegativePrompt}
          onEnhance={async () => {
            const result = await enhancePrompt(prompt)
            if (result !== null) {
              setEditableEnhancedPrompt(result)
              setUseEnhancedView(true)
              addToEnhanceHistory(result)
            }
          }}
          isEnhancing={isEnhancing}
          enhanceError={enhanceError}
          editableEnhancedPrompt={editableEnhancedPrompt}
          onEditableEnhancedPromptChange={(v) => {
            setEditableEnhancedPrompt(v || null)
            if (!v) setUseEnhancedView(false)
          }}
          onEncode={() => encodePrompt(activePromptForEncode)}
          isEncoding={isEncoding}
          encodedPrompt={encodedPrompt}
          isPromptChanged={isPromptChanged}
          showEncodeButton={showEncodeButton}
          onAddToQueue={handleAddToQueue}
          pendingJobCount={queuePendingCount}
          numSteps={numSteps}
          onNumStepsChange={setNumSteps}
          stgScale={stgScale}
          onStgScaleChange={setStgScale}
          stgBlockIndex={stgBlockIndex}
          onStgBlockIndexChange={setStgBlockIndex}
          blockSwap={appSettings.blockSwapBlocksOnGpu ?? 0}
          onBlockSwapChange={(v) => updateSettings({ blockSwapBlocksOnGpu: v })}
          processStatus={processStatus ?? undefined}
          useEnhancedView={useEnhancedView}
          onUseEnhancedViewChange={setUseEnhancedView}
          conditioningFrames={conditioningFrames}
          onConditioningFramesChange={setConditioningFrames}
        />
      </div>
      
      {/* Asset preview modal */}
      {selectedAsset && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
          onClick={() => setSelectedAsset(null)}
        >
          {/* Previous button */}
          <button
            onClick={(e) => { e.stopPropagation(); goToPrev() }}
            disabled={!canGoPrev}
            className={`absolute left-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
              canGoPrev
                ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
                : 'bg-white/5 text-zinc-600 cursor-default'
            }`}
          >
            <ChevronLeft className="h-6 w-6" />
          </button>

          {/* Next button */}
          <button
            onClick={(e) => { e.stopPropagation(); goToNext() }}
            disabled={!canGoNext}
            className={`absolute right-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
              canGoNext
                ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
                : 'bg-white/5 text-zinc-600 cursor-default'
            }`}
          >
            <ChevronRight className="h-6 w-6" />
          </button>

          {/* Content area */}
          <div className="relative max-w-5xl w-full max-h-full px-20 py-8" onClick={e => e.stopPropagation()}>
            {/* Top bar: counter + close */}
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-zinc-500 font-medium">
                {selectedIndex + 1} / {filteredAssets.length}
              </span>
              <button
                onClick={() => setSelectedAsset(null)}
                className="p-2 rounded-md text-zinc-400 hover:text-white transition-colors"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {selectedAsset.type === 'video' ? (
              <video
                key={selectedAsset.id}
                src={selectedAsset.url}
                controls
                autoPlay
                className="w-full rounded-xl object-contain max-h-[75vh]"
              />
            ) : (
              <img
                key={selectedAsset.id}
                src={selectedAsset.url}
                alt=""
                className="w-full rounded-xl object-contain max-h-[75vh]"
              />
            )}
            <div className="mt-4 text-center">
              <div className="inline-flex items-start gap-2 max-w-full">
                <p className="text-zinc-300">{selectedAsset.prompt}</p>
                {selectedAsset.prompt && (
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(selectedAsset.prompt)
                      setCopiedPrompt(true)
                      setTimeout(() => setCopiedPrompt(false), 2000)
                    }}
                    className="shrink-0 p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"
                    title="Copy prompt"
                  >
                    {copiedPrompt ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                )}
              </div>
              <p className="text-zinc-500 text-sm mt-1">
                {selectedAsset.resolution} • {selectedAsset.duration ? `${selectedAsset.duration}s` : 'Image'}
              </p>
            </div>
          </div>
        </div>
      )}

      {(error || localError) && (
        <GenerationErrorDialog
          error={(error || localError)!}
          onDismiss={() => {
            if (error) reset()
            if (localError) {
              setLocalError(null)
              resetRetake()
              resetIcLora()
            }
          }}
        />
      )}
    </div>
  )
}
