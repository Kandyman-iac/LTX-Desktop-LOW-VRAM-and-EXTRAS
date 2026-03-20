import { useState, useRef } from 'react'
import { Plus, X, Film, Upload } from 'lucide-react'
import { Button } from './ui/button'

export interface ConditioningFrame {
  role: 'first' | 'middle' | 'last'
  imagePath: string | null   // filesystem path for backend
  imageUrl: string | null    // file:// URL for display
  strength: number           // 0.0–1.0
  position: number           // 0.0–1.0, for middle frames only (fraction through video)
}

interface FrameSlotProps {
  frame: ConditioningFrame
  label: string
  showPosition: boolean
  onUpdate: (updates: Partial<ConditioningFrame>) => void
  onRemove?: () => void
}

function FrameSlot({ frame, label, showPosition, onUpdate, onRemove }: FrameSlotProps) {
  const [extracting, setExtracting] = useState(false)
  const [videoPath, setVideoPath] = useState('')
  const [seekTime, setSeekTime] = useState('0')
  const [showExtractor, setShowExtractor] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

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
    <div className="border border-zinc-700 rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-400">{label}</span>
        {onRemove && (
          <Button variant="ghost" size="sm" onClick={onRemove} className="h-5 w-5 p-0 text-zinc-500 hover:text-zinc-300">
            <X className="h-3 w-3" />
          </Button>
        )}
      </div>

      {/* Thumbnail / upload area */}
      <div className="flex gap-2">
        <div
          className="relative h-16 w-24 flex-shrink-0 rounded border border-zinc-600 bg-zinc-800 cursor-pointer overflow-hidden"
          onClick={() => fileInputRef.current?.click()}
        >
          {frame.imageUrl ? (
            <img src={frame.imageUrl} className="h-full w-full object-cover" alt={label} />
          ) : (
            <div className="flex h-full items-center justify-center text-zinc-500">
              <Upload className="h-4 w-4" />
            </div>
          )}
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
        </div>

        <div className="flex flex-col gap-1 flex-1 min-w-0">
          {/* Strength */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 w-12">Strength</span>
            <input
              type="range" min={0} max={1} step={0.05}
              value={frame.strength}
              onChange={e => onUpdate({ strength: parseFloat(e.target.value) })}
              className="flex-1 h-1 accent-blue-500"
            />
            <span className="text-xs text-zinc-400 w-8 text-right">{frame.strength.toFixed(2)}</span>
          </div>

          {/* Position (middle frames only) */}
          {showPosition && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500 w-12">Position</span>
              <input
                type="range" min={0.05} max={0.95} step={0.05}
                value={frame.position}
                onChange={e => onUpdate({ position: parseFloat(e.target.value) })}
                className="flex-1 h-1 accent-blue-500"
              />
              <span className="text-xs text-zinc-400 w-8 text-right">{Math.round(frame.position * 100)}%</span>
            </div>
          )}

          {/* Extract from video toggle */}
          <button
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-blue-400 self-start"
            onClick={() => setShowExtractor(v => !v)}
          >
            <Film className="h-3 w-3" />
            Extract from video
          </button>
        </div>
      </div>

      {/* Video extractor */}
      {showExtractor && (
        <div className="flex gap-1 items-center">
          <button
            className="text-xs text-blue-400 hover:underline whitespace-nowrap"
            onClick={handlePickVideo}
          >
            Pick video
          </button>
          {videoPath && <span className="text-xs text-zinc-500 truncate flex-1">{videoPath.split(/[\\/]/).pop()}</span>}
          <input
            type="number" min={0} step={0.1}
            value={seekTime}
            onChange={e => setSeekTime(e.target.value)}
            className="w-14 rounded border border-zinc-600 bg-zinc-800 px-1 py-0.5 text-xs text-white"
            placeholder="sec"
          />
          <Button size="sm" onClick={handleExtract} disabled={!videoPath || extracting}
            className="h-6 px-2 text-xs bg-blue-600 hover:bg-blue-500">
            {extracting ? '…' : 'Get'}
          </Button>
        </div>
      )}

      {frame.imageUrl && (
        <button
          className="text-xs text-zinc-500 hover:text-zinc-300"
          onClick={() => onUpdate({ imagePath: null, imageUrl: null })}
        >
          Clear
        </button>
      )}
    </div>
  )
}

interface MultiFrameConditioningPanelProps {
  frames: ConditioningFrame[]
  onChange: (frames: ConditioningFrame[]) => void
}

const MAX_MIDDLE_FRAMES = 3

export function MultiFrameConditioningPanel({ frames, onChange }: MultiFrameConditioningPanelProps) {
  const firstFrame = frames.find(f => f.role === 'first')!
  const middleFrames = frames.filter(f => f.role === 'middle')
  const lastFrame = frames.find(f => f.role === 'last')

  const update = (role: ConditioningFrame['role'], index: number, updates: Partial<ConditioningFrame>) => {
    onChange(frames.map((f, i) => {
      if (f.role !== role) return f
      const roleFrames = frames.filter(x => x.role === role)
      const targetFrame = roleFrames[index]
      if (f !== targetFrame) return f
      return { ...f, ...updates }
    }))
  }

  const addMiddleFrame = () => {
    if (middleFrames.length >= MAX_MIDDLE_FRAMES) return
    const newFrame: ConditioningFrame = {
      role: 'middle',
      imagePath: null, imageUrl: null,
      strength: 1.0,
      position: middleFrames.length === 0 ? 0.5 : middleFrames.length === 1 ? 0.33 : 0.67,
    }
    onChange([...frames.filter(f => f.role !== 'last'), newFrame, ...(lastFrame ? [lastFrame] : [])])
  }

  const removeMiddleFrame = (idx: number) => {
    const updated = [...frames]
    const mIdx = updated.findIndex((f, i) => f.role === 'middle' && frames.filter(x => x.role === 'middle').indexOf(f) === idx)
    updated.splice(mIdx, 1)
    onChange(updated)
  }

  const addLastFrame = () => {
    const newFrame: ConditioningFrame = { role: 'last', imagePath: null, imageUrl: null, strength: 1.0, position: 1 }
    onChange([...frames, newFrame])
  }

  const removeLastFrame = () => {
    onChange(frames.filter(f => f.role !== 'last'))
  }

  return (
    <div className="space-y-2">
      <FrameSlot
        frame={firstFrame}
        label="First Frame"
        showPosition={false}
        onUpdate={updates => update('first', 0, updates)}
      />

      {middleFrames.map((mf, i) => (
        <FrameSlot
          key={i}
          frame={mf}
          label={`Middle Frame ${middleFrames.length > 1 ? i + 1 : ''}`}
          showPosition
          onUpdate={updates => update('middle', i, updates)}
          onRemove={() => removeMiddleFrame(i)}
        />
      ))}

      {lastFrame && (
        <FrameSlot
          frame={lastFrame}
          label="Last Frame"
          showPosition={false}
          onUpdate={updates => update('last', 0, updates)}
          onRemove={removeLastFrame}
        />
      )}

      <div className="flex gap-2">
        {middleFrames.length < MAX_MIDDLE_FRAMES && (
          <button
            onClick={addMiddleFrame}
            className="flex items-center gap-1 text-xs text-zinc-400 hover:text-blue-400"
          >
            <Plus className="h-3 w-3" /> Middle frame
          </button>
        )}
        {!lastFrame && (
          <button
            onClick={addLastFrame}
            className="flex items-center gap-1 text-xs text-zinc-400 hover:text-blue-400"
          >
            <Plus className="h-3 w-3" /> Last frame
          </button>
        )}
      </div>
    </div>
  )
}
