import { useRef, useState } from 'react'
import { Upload, X } from 'lucide-react'

export const MAGI_RESOLUTION_PRESETS = [
  { label: '448×256  (16:9 fastest)', width: 448, height: 256 },
  { label: '640×384  (16:9 fast)',    width: 640, height: 384 },
  { label: '768×448  (16:9 slow)',    width: 768, height: 448 },
  { label: '512×512  (square)',       width: 512, height: 512 },
  { label: '256×448  (9:16 fast)',    width: 256, height: 448 },
  { label: '384×640  (9:16 slow)',    width: 384, height: 640 },
  { label: '448×768  (9:16 slowest)', width: 448, height: 768 },
]

export interface MagiPanelState {
  imagePath: string | null
  seconds: number
  width: number
  height: number
  gpus: number
  seed?: number
  sr: boolean
  ready: boolean
}

interface MagiPanelProps {
  isProcessing: boolean
  srModelReady?: boolean
  onChange: (state: MagiPanelState) => void
}

export function MagiPanel({ isProcessing, srModelReady = false, onChange }: MagiPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [imagePath, setImagePath] = useState<string | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [seconds, setSeconds] = useState(5)
  const [resIdx, setResIdx] = useState(0)
  const [gpus, setGpus] = useState(1)
  const [seedStr, setSeedStr] = useState('')
  const [sr, setSr] = useState(false)

  function emit(
    overrides: Partial<{ imagePath: string | null; seconds: number; resIdx: number; gpus: number; seedStr: string; sr: boolean }>
  ) {
    const ip = overrides.imagePath !== undefined ? overrides.imagePath : imagePath
    const ri = overrides.resIdx ?? resIdx
    const preset = MAGI_RESOLUTION_PRESETS[ri]
    const s = overrides.seedStr ?? seedStr
    const seedVal = s.trim() ? parseInt(s, 10) : undefined
    const srVal = overrides.sr !== undefined ? overrides.sr : sr
    onChange({
      imagePath: ip ?? null,
      seconds: overrides.seconds ?? seconds,
      width: preset.width,
      height: preset.height,
      gpus: overrides.gpus ?? gpus,
      seed: seedVal,
      sr: srVal,
      ready: !!ip,
    })
  }

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const path = (file as File & { path?: string }).path ?? ''
    const url = path ? `file:///${path.replace(/\\/g, '/')}` : URL.createObjectURL(file)
    setImagePath(path || null)
    setImageUrl(url)
    emit({ imagePath: path || null })
  }

  const clearImage = () => {
    setImagePath(null)
    setImageUrl(null)
    emit({ imagePath: null })
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="space-y-4 p-4 bg-zinc-900 border border-zinc-800 rounded-2xl">
      <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
        MagiHuman — Conditioning Image
      </p>

      {/* Image upload */}
      <div
        className="relative h-28 flex items-center justify-center rounded-lg border border-dashed border-zinc-600 bg-zinc-800 cursor-pointer overflow-hidden hover:border-zinc-400 transition-colors"
        onClick={() => !isProcessing && fileInputRef.current?.click()}
      >
        {imageUrl ? (
          <>
            <img src={imageUrl} className="h-full w-full object-contain" alt="Conditioning" />
            {!isProcessing && (
              <button
                className="absolute top-1 right-1 p-0.5 rounded-full bg-zinc-900/80 text-zinc-300 hover:text-white"
                onClick={(e) => { e.stopPropagation(); clearImage() }}
                title="Remove image"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center gap-1.5 text-zinc-500">
            <Upload className="h-5 w-5" />
            <span className="text-xs">Click to pick image</span>
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={handleFile}
        disabled={isProcessing}
      />

      {/* Duration */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-zinc-400">Duration</label>
          <span className="text-xs text-zinc-400">{seconds}s</span>
        </div>
        <input
          type="range" min={1} max={30} step={1} value={seconds}
          disabled={isProcessing}
          onChange={(e) => { const v = Number(e.target.value); setSeconds(v); emit({ seconds: v }) }}
          className="w-full accent-blue-500"
        />
      </div>

      {/* Resolution */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-zinc-400">Resolution</label>
        <select
          value={resIdx}
          disabled={isProcessing}
          onChange={(e) => { const v = Number(e.target.value); setResIdx(v); emit({ resIdx: v }) }}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500 disabled:opacity-50"
        >
          {MAGI_RESOLUTION_PRESETS.map((p, i) => (
            <option key={i} value={i}>{p.label}</option>
          ))}
        </select>
      </div>

      {/* GPUs */}
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-zinc-400">GPUs</label>
        <div className="flex gap-1">
          {[1, 2].map((g) => (
            <button
              key={g}
              disabled={isProcessing}
              onClick={() => { setGpus(g); emit({ gpus: g }) }}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors disabled:opacity-50 ${
                gpus === g ? 'bg-blue-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* SR toggle */}
      <div className="flex items-center justify-between">
        <div>
          <label className={`text-xs font-medium ${srModelReady ? 'text-zinc-400' : 'text-zinc-600'}`}>
            SR 2× upscale
          </label>
          <p className="text-[10px] text-zinc-600 mt-0.5">
            {srModelReady ? 'Generates at 2× resolution via 540p_sr' : 'Model not downloaded — run: bash ~/download_sr.sh'}
          </p>
        </div>
        <button
          disabled={isProcessing || !srModelReady}
          onClick={() => { const v = !sr; setSr(v); emit({ sr: v }) }}
          className={`relative w-9 h-5 rounded-full transition-colors disabled:opacity-50 ${sr && srModelReady ? 'bg-blue-600' : 'bg-zinc-700'}`}
        >
          <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${sr && srModelReady ? 'translate-x-4' : ''}`} />
        </button>
      </div>

      {/* Seed */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-zinc-400">Seed (optional)</label>
        <input
          type="number"
          value={seedStr}
          disabled={isProcessing}
          onChange={(e) => { setSeedStr(e.target.value); emit({ seedStr: e.target.value }) }}
          placeholder="Random"
          className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-blue-500 disabled:opacity-50"
        />
      </div>
    </div>
  )
}
