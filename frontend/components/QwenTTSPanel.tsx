import { useState, useRef } from 'react'
import { Loader2, Download, Mic, Upload, X } from 'lucide-react'
import type { QwenTTSState } from '../hooks/use-qwentts'

const SPEAKERS_BY_LANGUAGE: Record<string, string[]> = {
  English:    ['Ryan', 'Aiden'],
  Chinese:    ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric'],
  Japanese:   ['Ono_Anna'],
  Korean:     ['Sohee'],
}
const ALL_LANGUAGES = [
  'English', 'Chinese', 'Japanese', 'Korean',
  'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian',
]
const DEFAULT_SPEAKER_FOR: Record<string, string> = {
  English: 'Ryan', Chinese: 'Vivian', Japanese: 'Ono_Anna', Korean: 'Sohee',
}

export interface QwenTTSPanelState {
  text: string
  language: string
  mode: 'custom_voice' | 'voice_clone'
  speaker: string
  instruct: string
  refAudioPath: string | null
  refText: string
  modelSize: '1.7b' | '0.6b'
  ready: boolean
}

interface QwenTTSPanelProps {
  isProcessing: boolean
  ttsState: QwenTTSState
  onChange: (state: QwenTTSPanelState) => void
}

export function QwenTTSPanel({ ttsState, onChange }: QwenTTSPanelProps) {
  const refAudioInputRef = useRef<HTMLInputElement>(null)

  const [text, setText] = useState('')
  const [language, setLanguage] = useState('English')
  const [mode, setMode] = useState<'custom_voice' | 'voice_clone'>('custom_voice')
  const [speaker, setSpeaker] = useState('Ryan')
  const [instruct, setInstruct] = useState('')
  const [refAudioPath, setRefAudioPath] = useState<string | null>(null)
  const [refAudioName, setRefAudioName] = useState<string | null>(null)
  const [refText, setRefText] = useState('')
  const [modelSize, setModelSize] = useState<'1.7b' | '0.6b'>('1.7b')

  function emit(overrides: Partial<QwenTTSPanelState> = {}) {
    const state: QwenTTSPanelState = {
      text, language, mode, speaker, instruct,
      refAudioPath, refText, modelSize,
      ready: text.trim().length > 0 && (mode === 'custom_voice' || !!refAudioPath),
      ...overrides,
    }
    onChange(state)
  }

  function handleLanguageChange(lang: string) {
    setLanguage(lang)
    const speakers = SPEAKERS_BY_LANGUAGE[lang]
    const newSpeaker = speakers
      ? (DEFAULT_SPEAKER_FOR[lang] ?? speakers[0])
      : speaker
    setSpeaker(newSpeaker)
    emit({ language: lang, speaker: newSpeaker })
  }

  function handleRefAudioFile(file: File) {
    const path = (file as { path?: string }).path ?? null
    setRefAudioPath(path)
    setRefAudioName(file.name)
    emit({ refAudioPath: path })
  }

  const speakersForLang = SPEAKERS_BY_LANGUAGE[language] ?? []

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex gap-1 p-1 bg-zinc-800 rounded-lg">
        {(['custom_voice', 'voice_clone'] as const).map(m => (
          <button
            key={m}
            onClick={() => { setMode(m); emit({ mode: m }) }}
            className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-colors ${
              mode === m ? 'bg-zinc-600 text-white' : 'text-zinc-400 hover:text-white'
            }`}
          >
            {m === 'custom_voice' ? 'Preset Voice' : 'Voice Clone'}
          </button>
        ))}
      </div>

      {/* Text input */}
      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Text to synthesise</label>
        <textarea
          value={text}
          onChange={e => { setText(e.target.value); emit({ text: e.target.value }) }}
          placeholder="Enter text…"
          rows={4}
          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-500 resize-none focus:outline-none focus:border-zinc-500"
        />
      </div>

      {/* Language */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-zinc-400 w-16 shrink-0">Language</label>
        <select
          value={language}
          onChange={e => handleLanguageChange(e.target.value)}
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white focus:outline-none focus:border-zinc-500"
        >
          {ALL_LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>

      {mode === 'custom_voice' ? (
        <>
          {/* Speaker */}
          {speakersForLang.length > 0 && (
            <div className="flex items-center gap-3">
              <label className="text-xs text-zinc-400 w-16 shrink-0">Speaker</label>
              <select
                value={speaker}
                onChange={e => { setSpeaker(e.target.value); emit({ speaker: e.target.value }) }}
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white focus:outline-none focus:border-zinc-500"
              >
                {speakersForLang.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          )}
          {/* Style instruction */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-zinc-400 w-16 shrink-0">Style</label>
            <input
              type="text"
              value={instruct}
              onChange={e => { setInstruct(e.target.value); emit({ instruct: e.target.value }) }}
              placeholder="e.g. speak slowly and warmly"
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
            />
          </div>
        </>
      ) : (
        <>
          {/* Reference audio upload */}
          <div>
            <label className="text-xs text-zinc-400 mb-1 block">Reference audio (3–10s WAV/MP3)</label>
            <input
              ref={refAudioInputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={e => { if (e.target.files?.[0]) handleRefAudioFile(e.target.files[0]) }}
            />
            {refAudioPath ? (
              <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800 rounded-lg border border-zinc-700">
                <Mic className="h-3.5 w-3.5 text-green-400 shrink-0" />
                <span className="text-xs text-zinc-300 truncate flex-1">{refAudioName}</span>
                <button onClick={() => { setRefAudioPath(null); setRefAudioName(null); emit({ refAudioPath: null }) }}>
                  <X className="h-3.5 w-3.5 text-zinc-500 hover:text-white" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => refAudioInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 px-3 py-2.5 border border-dashed border-zinc-600 rounded-lg text-xs text-zinc-400 hover:border-zinc-400 hover:text-zinc-200 transition-colors"
              >
                <Upload className="h-3.5 w-3.5" />
                Upload reference audio
              </button>
            )}
          </div>
          {/* Reference transcript */}
          <div>
            <label className="text-xs text-zinc-400 mb-1 block">Reference transcript <span className="text-zinc-600">(improves quality)</span></label>
            <input
              type="text"
              value={refText}
              onChange={e => { setRefText(e.target.value); emit({ refText: e.target.value }) }}
              placeholder="What is said in the reference audio…"
              className="w-full bg-zinc-800 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
            />
          </div>
        </>
      )}

      {/* Model size */}
      <div className="flex items-center gap-3">
        <label className="text-xs text-zinc-400 w-16 shrink-0">Model</label>
        <div className="flex gap-1">
          {(['1.7b', '0.6b'] as const).map(sz => (
            <button
              key={sz}
              onClick={() => { setModelSize(sz); emit({ modelSize: sz }) }}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                modelSize === sz ? 'bg-zinc-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:text-white'
              }`}
            >
              {sz}
            </button>
          ))}
        </div>
      </div>

      {/* Status / output */}
      {ttsState.status === 'running' && (
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          <span>{ttsState.logTail ? ttsState.logTail.split('\n').at(-1) : 'Generating…'}</span>
        </div>
      )}

      {ttsState.status === 'complete' && ttsState.outputUrl && (
        <div className="space-y-2">
          <audio controls src={ttsState.outputUrl} className="w-full h-8" />
          <a
            href={ttsState.outputUrl}
            download
            className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300"
          >
            <Download className="h-3 w-3" />
            Download WAV
          </a>
        </div>
      )}

      {ttsState.status === 'error' && ttsState.error && (
        <p className="text-xs text-red-400 break-words">{ttsState.error}</p>
      )}
    </div>
  )
}
