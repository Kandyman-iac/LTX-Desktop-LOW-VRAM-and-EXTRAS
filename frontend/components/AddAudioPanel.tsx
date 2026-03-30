import { useCallback, useEffect, useRef, useState } from 'react'
import { Film, Loader2, Music, Upload, X, Check, AlertCircle } from 'lucide-react'
import { backendFetch } from '../lib/backend'
import type { MMAudioState } from '../hooks/use-mmaudio'
import type { PrismAudioState } from '../hooks/use-prismaudio'

export type AudioEngine = 'mmaudio' | 'prismaudio'

export interface AddAudioPanelState {
  videoPath: string | null
  videoUrl: string | null
  engine: AudioEngine
  prompt: string
  ready: boolean
}

interface RecentOutput {
  filename: string
  path: string
  modified_at: number
  prompt?: string | null
}

interface AddAudioPanelProps {
  isProcessing: boolean
  audioState: MMAudioState | PrismAudioState
  onChange: (state: AddAudioPanelState) => void
}

function pathToFileUrl(p: string): string {
  const n = p.replace(/\\/g, '/')
  return n.startsWith('/') ? `file://${n}` : `file:///${n}`
}

export function AddAudioPanel({ isProcessing, audioState, onChange }: AddAudioPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [sourceTab, setSourceTab] = useState<'outputs' | 'upload'>('outputs')
  const [engine, setEngine] = useState<AudioEngine>('mmaudio')
  const [prompt, setPrompt] = useState('')

  const [videoPath, setVideoPath] = useState<string | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)

  // Recent outputs
  const [outputs, setOutputs] = useState<RecentOutput[]>([])
  const [outputsLoading, setOutputsLoading] = useState(false)
  const [outputsError, setOutputsError] = useState<string | null>(null)

  const fetchOutputs = useCallback(async () => {
    setOutputsLoading(true)
    setOutputsError(null)
    try {
      const res = await backendFetch('/api/outputs?page=1&page_size=24')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setOutputs(data.entries ?? [])
    } catch (e) {
      setOutputsError(String(e))
    } finally {
      setOutputsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (sourceTab === 'outputs') fetchOutputs()
  }, [sourceTab, fetchOutputs])

  function emit(overrides: Partial<{ videoPath: string | null; videoUrl: string | null; engine: AudioEngine; prompt: string }>) {
    const vp = overrides.videoPath !== undefined ? overrides.videoPath : videoPath
    const vu = overrides.videoUrl !== undefined ? overrides.videoUrl : videoUrl
    const eng = overrides.engine ?? engine
    const pr = overrides.prompt ?? prompt
    onChange({ videoPath: vp, videoUrl: vu, engine: eng, prompt: pr, ready: !!vp })
  }

  function selectVideo(path: string) {
    const url = pathToFileUrl(path)
    setVideoPath(path)
    setVideoUrl(url)
    emit({ videoPath: path, videoUrl: url })
  }

  function clearVideo() {
    setVideoPath(null)
    setVideoUrl(null)
    emit({ videoPath: null, videoUrl: null })
  }

  function handleEngine(eng: AudioEngine) {
    setEngine(eng)
    emit({ engine: eng })
  }

  function handlePrompt(p: string) {
    setPrompt(p)
    emit({ prompt: p })
  }

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    // Electron gives us the real path via webkitRelativePath fallback or window.electronAPI
    const anyFile = file as File & { path?: string }
    const path = anyFile.path ?? ''
    if (path) {
      selectVideo(path)
    } else {
      // Fallback: create an object URL (won't work for backend path, but shows preview)
      const url = URL.createObjectURL(file)
      setVideoPath(file.name)
      setVideoUrl(url)
      emit({ videoPath: file.name, videoUrl: url })
    }
  }

  const isComplete = audioState.status === 'complete'
  const isRunning = audioState.status === 'running'

  return (
    <div className="space-y-4">
      {/* Engine selector */}
      <div>
        <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">Audio Engine</label>
        <div className="flex gap-1 p-1 bg-zinc-900 border border-zinc-800 rounded-lg">
          {(['mmaudio', 'prismaudio'] as AudioEngine[]).map(eng => (
            <button
              key={eng}
              onClick={() => handleEngine(eng)}
              disabled={isProcessing}
              className={`flex-1 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-50 ${
                engine === eng ? 'bg-violet-700 text-white' : 'text-zinc-400 hover:text-white'
              }`}
            >
              {eng === 'mmaudio' ? 'MMAudio' : 'PrismAudio'}
            </button>
          ))}
        </div>
        <p className="text-[11px] text-zinc-600 mt-1">
          {engine === 'mmaudio'
            ? 'Synchronized ambient/music — runs in WSL'
            : 'Foley & SFX with spatial audio — WSL or Windows (conda)'}
        </p>
      </div>

      {/* Video source */}
      <div>
        <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">Source Video</label>

        {videoPath ? (
          /* Selected video */
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-2.5 flex items-center gap-2">
            <Film className="h-4 w-4 text-violet-400 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-xs text-white truncate">{videoPath.replace(/.*[\\/]/, '')}</p>
              {isComplete && (
                <p className="text-[11px] text-emerald-400 flex items-center gap-1 mt-0.5">
                  <Check className="h-3 w-3" /> Audio generated
                </p>
              )}
            </div>
            {!isRunning && (
              <button onClick={clearVideo} className="text-zinc-500 hover:text-white flex-shrink-0">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        ) : (
          /* Source picker tabs */
          <div className="border border-zinc-800 rounded-lg overflow-hidden">
            {/* Tab bar */}
            <div className="flex border-b border-zinc-800">
              {(['outputs', 'upload'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setSourceTab(tab)}
                  className={`flex-1 py-2 text-xs font-medium transition-colors ${
                    sourceTab === tab
                      ? 'bg-zinc-800 text-white'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {tab === 'outputs' ? 'Recent Outputs' : 'Upload File'}
                </button>
              ))}
            </div>

            {/* Outputs list */}
            {sourceTab === 'outputs' && (
              <div className="max-h-56 overflow-y-auto bg-zinc-900">
                {outputsLoading && (
                  <div className="flex items-center justify-center py-6 gap-2 text-xs text-zinc-500">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading…
                  </div>
                )}
                {outputsError && (
                  <div className="flex items-center gap-2 px-3 py-3 text-xs text-red-400">
                    <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
                    {outputsError}
                  </div>
                )}
                {!outputsLoading && !outputsError && outputs.length === 0 && (
                  <p className="px-3 py-4 text-xs text-zinc-500 text-center">
                    No generated videos yet — generate one in Video mode first.
                  </p>
                )}
                {outputs.map(out => (
                  <button
                    key={out.path}
                    onClick={() => selectVideo(out.path)}
                    className="w-full flex items-center gap-2.5 px-3 py-2 hover:bg-zinc-800 transition-colors text-left border-b border-zinc-800/50 last:border-0"
                  >
                    <Film className="h-3.5 w-3.5 text-zinc-500 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-white truncate">{out.filename}</p>
                      {out.prompt && (
                        <p className="text-[10px] text-zinc-500 truncate mt-0.5">{out.prompt}</p>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}

            {/* Upload */}
            {sourceTab === 'upload' && (
              <div className="bg-zinc-900 p-3">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/*"
                  className="hidden"
                  onChange={handleFileUpload}
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full flex flex-col items-center gap-2 py-5 border-2 border-dashed border-zinc-700 rounded-lg hover:border-violet-600 hover:bg-zinc-800/50 transition-colors"
                >
                  <Upload className="h-5 w-5 text-zinc-500" />
                  <span className="text-xs text-zinc-400">Click to browse</span>
                  <span className="text-[11px] text-zinc-600">MP4, MOV, WebM</span>
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Prompt */}
      <div>
        <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">
          Sound Description <span className="font-normal text-zinc-600 normal-case">(optional)</span>
        </label>
        <textarea
          value={prompt}
          onChange={e => handlePrompt(e.target.value)}
          disabled={isProcessing}
          placeholder={engine === 'mmaudio'
            ? 'e.g. dramatic orchestral score, distant thunder, ocean waves'
            : 'e.g. footsteps on gravel, door creaking, city ambience'}
          rows={2}
          className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 resize-none disabled:opacity-50"
        />
      </div>

      {/* Progress / log tail */}
      {isRunning && audioState.logTail && (
        <div className="bg-black/40 rounded-lg px-2.5 py-2 max-h-20 overflow-y-auto">
          <div className="flex items-center gap-2 mb-1">
            <Loader2 className="h-3 w-3 animate-spin text-violet-400 flex-shrink-0" />
            <span className="text-[11px] text-zinc-400">Generating audio…</span>
          </div>
          <pre className="text-[10px] text-zinc-600 whitespace-pre-wrap">{audioState.logTail.split('\n').slice(-4).join('\n')}</pre>
        </div>
      )}

      {audioState.status === 'error' && (
        <div className="flex items-start gap-2 bg-red-900/20 border border-red-900/40 rounded-lg px-3 py-2">
          <AlertCircle className="h-3.5 w-3.5 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-red-400">{audioState.error || 'Generation failed'}</p>
        </div>
      )}

      {isComplete && (
        <div className="flex items-center gap-2 bg-emerald-900/20 border border-emerald-900/40 rounded-lg px-3 py-2">
          <Check className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />
          <p className="text-xs text-emerald-400">Audio generated — result shown in the player →</p>
        </div>
      )}

      <div className="flex items-center gap-1.5 text-[11px] text-zinc-600">
        <Music className="h-3 w-3" />
        <span>Requires {engine === 'mmaudio' ? 'MMAudio (WSL)' : 'PrismAudio (WSL/Windows)'} — see Settings → Tools</span>
      </div>
    </div>
  )
}
