import React, { useState } from 'react'
import { ChevronLeft, ChevronRight, Plus, RotateCcw, X, Zap } from 'lucide-react'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

export interface InferenceOverrides {
  numSteps: number
  stgScale: number
  stgBlockIndex: number
}

interface InferencePanelProps {
  overrides: InferenceOverrides
  defaults: InferenceOverrides
  onChange: (overrides: InferenceOverrides) => void
  disabled?: boolean
}

// ── LoRA helpers ──────────────────────────────────────────────────────────────

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

// ── FlushVramButton ───────────────────────────────────────────────────────────

function FlushVramButton() {
  const [status, setStatus] = useState<'idle' | 'flushing' | 'done' | 'error'>('idle')
  const [freedMb, setFreedMb] = useState<number | null>(null)

  const handleFlush = async () => {
    setStatus('flushing')
    setFreedMb(null)
    try {
      const res = await backendFetch('/api/system/clear-vram', { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json() as { freed_mb: number }
      setFreedMb(data.freed_mb)
      setStatus('done')
    } catch (err) {
      logger.error(`VRAM flush failed: ${String(err)}`)
      setStatus('error')
    } finally {
      setTimeout(() => setStatus('idle'), 3000)
    }
  }

  return (
    <div className="flex items-center justify-between">
      <div>
        <label className="text-[10px] text-zinc-300">Flush VRAM Cache</label>
        <p className="text-[10px] text-zinc-500">
          {status === 'done' && freedMb !== null
            ? `Freed ${freedMb} MB from allocator cache`
            : status === 'error'
            ? 'Flush failed — check logs'
            : 'Return cached GPU memory to driver'}
        </p>
      </div>
      <button
        onClick={() => { void handleFlush() }}
        disabled={status === 'flushing'}
        className={`px-3 py-1 rounded text-[10px] font-medium transition-colors ${
          status === 'flushing' ? 'bg-zinc-600 text-zinc-400 cursor-not-allowed' :
          status === 'done'     ? 'bg-green-600 text-white' :
          status === 'error'    ? 'bg-red-600 text-white' :
                                  'bg-amber-600 hover:bg-amber-500 text-white'
        }`}
      >
        {status === 'flushing' ? 'Flushing…' : status === 'done' ? 'Done' : status === 'error' ? 'Error' : 'Flush'}
      </button>
    </div>
  )
}

// ── LoraManager ───────────────────────────────────────────────────────────────

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
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-[10px] text-zinc-500">
          LTX-Video compatible LoRAs. Changes take effect on next pipeline load.
        </p>
        <button
          onClick={() => { void handleAdd() }}
          className="flex-shrink-0 flex items-center gap-1 px-2 py-1 text-[10px] text-violet-300 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-700/40 rounded-lg transition-colors"
        >
          <Plus size={10} />
          Add
        </button>
      </div>

      <div className="space-y-2">
        {loras.length === 0 ? (
          <p className="text-[10px] text-zinc-600 italic px-1">No LoRAs added.</p>
        ) : loras.map((lora, i) => (
          <div
            key={i}
            className={`rounded-lg p-2.5 space-y-2 border transition-colors ${lora.enabled ? 'bg-zinc-800/60 border-zinc-700/50' : 'bg-zinc-900/40 border-zinc-800/30 opacity-60'}`}
          >
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleToggle(i)}
                className={`relative inline-flex h-4 w-7 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${lora.enabled ? 'bg-violet-500' : 'bg-zinc-600'}`}
                title={lora.enabled ? 'Disable' : 'Enable'}
              >
                <span className={`inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform ${lora.enabled ? 'translate-x-3' : 'translate-x-0'}`} />
              </button>
              <span className="flex-1 text-[10px] text-zinc-300 truncate" title={lora.path}>
                {basename(lora.path)}
              </span>
              <button
                onClick={() => handleRemove(i)}
                className="text-zinc-600 hover:text-red-400 transition-colors flex-shrink-0"
                title="Remove"
              >
                <X size={12} />
              </button>
            </div>
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

// ── Toggle helper ─────────────────────────────────────────────────────────────

function Toggle({ on, onToggle, color = 'amber' }: { on: boolean; onToggle: () => void; color?: 'amber' | 'green' }) {
  const bg = on
    ? color === 'green' ? 'bg-green-500' : 'bg-amber-500'
    : 'bg-zinc-700'
  return (
    <button
      onClick={onToggle}
      className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${bg}`}
    >
      <span
        className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${on ? 'translate-x-4' : 'translate-x-0'}`}
      />
    </button>
  )
}

// ── Collapsible section header ────────────────────────────────────────────────

function SectionHeader({
  open,
  onToggle,
  icon,
  label,
  badge,
}: {
  open: boolean
  onToggle: () => void
  icon: React.ReactNode
  label: string
  badge?: React.ReactNode
}) {
  return (
    <button
      className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-zinc-800/30 transition-colors"
      onClick={onToggle}
    >
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs font-semibold text-white">{label}</span>
        {badge}
      </div>
      <ChevronRight
        className={`h-3.5 w-3.5 text-zinc-500 transition-transform ${open ? 'rotate-90' : ''}`}
      />
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function InferencePanel({ overrides, defaults, onChange, disabled = false }: InferencePanelProps) {
  const { settings: gs, updateSettings } = useAppSettings()
  const [collapsed, setCollapsed] = useState(false)
  const [vramOpen, setVramOpen] = useState(false)
  const [loraOpen, setLoraOpen] = useState(false)

  const isModified =
    overrides.numSteps !== defaults.numSteps ||
    overrides.stgScale !== defaults.stgScale ||
    overrides.stgBlockIndex !== defaults.stgBlockIndex

  const activeLoras = parseLoras(gs.civitaiLoras ?? '[]').filter(l => l.enabled).length

  if (collapsed) {
    return (
      <div className="flex-shrink-0 flex flex-col items-center pt-4">
        <button
          onClick={() => setCollapsed(false)}
          className="flex flex-col items-center gap-2 text-zinc-500 hover:text-zinc-300 transition-colors"
          title="Show inference settings"
        >
          <ChevronLeft className="h-4 w-4" />
          <Zap className="h-4 w-4 text-green-400" />
          {isModified && (
            <span className="w-2 h-2 rounded-full bg-blue-500" title="Overrides active" />
          )}
        </button>
      </div>
    )
  }

  return (
    <div className="w-72 flex-shrink-0 flex flex-col bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-green-400" />
          <span className="text-sm font-semibold text-white">Inference</span>
          {isModified && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-600/30 text-blue-400 font-medium">
              overrides
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isModified && (
            <button
              onClick={() => onChange({ ...defaults })}
              disabled={disabled}
              title="Reset per-gen overrides to global defaults"
              className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300 disabled:opacity-40 transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Reset
            </button>
          )}
          <button
            onClick={() => setCollapsed(true)}
            className="text-zinc-500 hover:text-zinc-300 transition-colors"
            title="Collapse"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex flex-col overflow-y-auto min-h-0 flex-1">

        {/* ── Fast Model (per-generation) ── */}
        <div className="p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Zap className="h-3.5 w-3.5 text-green-400" />
            <span className="text-xs font-semibold text-white">Fast Model</span>
            <span className="text-[10px] text-zinc-500">per-generation</span>
          </div>

          <div className="bg-zinc-800/50 rounded-lg p-3 space-y-3">
            {/* Steps */}
            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs text-white">Steps</label>
                <p className="text-[10px] text-zinc-500">Fewer = faster, 8 = full quality</p>
              </div>
              <input
                type="number"
                min={1}
                max={8}
                value={overrides.numSteps}
                disabled={disabled}
                onChange={(e) => {
                  const v = Math.max(1, Math.min(8, parseInt(e.target.value) || 8))
                  onChange({ ...overrides, numSteps: v })
                }}
                className="w-14 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded-lg text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
              />
            </div>

            {/* STG Scale */}
            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs text-white">STG Scale</label>
                <p className="text-[10px] text-zinc-500">0 = off. Try 0.5–1.5</p>
              </div>
              <input
                type="number"
                min={0}
                max={10}
                step={0.1}
                value={overrides.stgScale}
                disabled={disabled}
                onChange={(e) => {
                  const v = Math.max(0, Math.min(10, parseFloat(e.target.value) || 0))
                  onChange({ ...overrides, stgScale: v })
                }}
                className="w-14 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded-lg text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
              />
            </div>

            {/* STG Block Index — only when STG active */}
            {overrides.stgScale > 0 && (
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-xs text-zinc-300">STG Block</label>
                  <p className="text-[10px] text-zinc-500">Block to perturb (0–47 for 22B)</p>
                </div>
                <input
                  type="number"
                  min={0}
                  max={47}
                  value={overrides.stgBlockIndex}
                  disabled={disabled}
                  onChange={(e) => {
                    const v = Math.max(0, Math.min(47, parseInt(e.target.value) || 28))
                    onChange({ ...overrides, stgBlockIndex: v })
                  }}
                  className="w-14 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded-lg text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50"
                />
              </div>
            )}
          </div>

          <p className="text-[10px] text-zinc-500">
            {overrides.numSteps} steps
            {overrides.stgScale > 0 ? `, STG ${overrides.stgScale} (block ${overrides.stgBlockIndex})` : ''}
          </p>
        </div>

        {/* ── VRAM Optimisations (collapsible, global) ── */}
        <div className="border-t border-zinc-800">
          <SectionHeader
            open={vramOpen}
            onToggle={() => setVramOpen(v => !v)}
            icon={
              <svg className="h-3.5 w-3.5 text-amber-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="7" width="20" height="10" rx="2" />
                <path d="M6 7V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v2" />
                <circle cx="12" cy="12" r="1" fill="currentColor" />
              </svg>
            }
            label="VRAM Optimisations"
            badge={<span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">Global</span>}
          />

          {vramOpen && (
            <div className="px-4 pb-4 space-y-3">
              <p className="text-[10px] text-zinc-500">Changes save immediately but require app restart to take effect.</p>

              <div className="bg-zinc-800/50 rounded-lg p-3 space-y-3">
                {/* Multi-GPU */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-xs text-white">Multi-GPU</label>
                    <p className="text-[10px] text-zinc-500">Transformer on cuda:0, encoder on cuda:1</p>
                  </div>
                  <Toggle
                    on={gs.useMultiGpu ?? false}
                    onToggle={() => updateSettings({ ...gs, useMultiGpu: !gs.useMultiGpu })}
                  />
                </div>

                {/* FP8 Transformer */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-xs text-white">FP8 Transformer</label>
                    <p className="text-[10px] text-zinc-500">Force FP8 quantisation</p>
                  </div>
                  <Toggle
                    on={gs.useFp8Transformer ?? false}
                    onToggle={() => updateSettings({ ...gs, useFp8Transformer: !gs.useFp8Transformer })}
                  />
                </div>

                {/* Block Swap */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-xs text-white">Block Swap</label>
                    <p className="text-[10px] text-zinc-500">0 = off, max 48. Blocks on GPU</p>
                  </div>
                  <input
                    type="number"
                    min={0}
                    max={48}
                    value={gs.blockSwapBlocksOnGpu ?? 0}
                    onChange={(e) => updateSettings({ ...gs, blockSwapBlocksOnGpu: Math.max(0, Math.min(48, parseInt(e.target.value) || 0)) })}
                    className="w-14 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded-lg text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                  />
                </div>

                {/* Attention Tiling */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-xs text-white">Attention Tile</label>
                    <p className="text-[10px] text-zinc-500">0 = off. Try 512–2048</p>
                  </div>
                  <input
                    type="number"
                    min={0}
                    max={16384}
                    step={64}
                    value={gs.attentionTileSize ?? 0}
                    onChange={(e) => updateSettings({ ...gs, attentionTileSize: Math.max(0, Math.min(16384, parseInt(e.target.value) || 0)) })}
                    className="w-16 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded-lg text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-amber-500"
                  />
                </div>

                {/* Abliterated Encoder */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-xs text-white">Abliterated Encoder</label>
                    <p className="text-[10px] text-zinc-500">Refusal-direction-removed Gemma</p>
                  </div>
                  <Toggle
                    on={gs.useAbliteratedEncoder ?? false}
                    onToggle={() => updateSettings({ ...gs, useAbliteratedEncoder: !gs.useAbliteratedEncoder })}
                  />
                </div>

                {/* GGUF Path */}
                <div className="space-y-1">
                  <label className="text-xs text-white">GGUF Path</label>
                  <input
                    type="text"
                    value={gs.ggufTransformerPath ?? ''}
                    onChange={(e) => updateSettings({ ...gs, ggufTransformerPath: e.target.value })}
                    onKeyDown={(e) => e.stopPropagation()}
                    placeholder="e.g. C:\models\transformer.gguf"
                    className="w-full px-2 py-1 bg-zinc-700 border border-zinc-600 rounded-lg text-[10px] text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                  />
                </div>

                {/* VAE Tiling */}
                <div className="space-y-1.5 pt-2 border-t border-zinc-700">
                  <label className="text-xs text-white">VAE Tiling</label>
                  <p className="text-[10px] text-zinc-500">0 = library defaults (512px / 64 frames)</p>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                      <label className="text-[10px] text-zinc-400">Spatial (px)</label>
                      <input
                        type="number"
                        min={0}
                        max={4096}
                        step={32}
                        value={gs.vaeSpatialTileSize ?? 0}
                        onChange={(e) => updateSettings({ ...gs, vaeSpatialTileSize: Math.max(0, Math.min(4096, parseInt(e.target.value) || 0)) })}
                        className="w-full px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-1 focus:ring-amber-500"
                      />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[10px] text-zinc-400">Temporal (frames)</label>
                      <input
                        type="number"
                        min={0}
                        max={512}
                        step={8}
                        value={gs.vaeTemporalTileSize ?? 0}
                        onChange={(e) => updateSettings({ ...gs, vaeTemporalTileSize: Math.max(0, Math.min(512, parseInt(e.target.value) || 0)) })}
                        className="w-full px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-1 focus:ring-amber-500"
                      />
                    </div>
                  </div>
                </div>

                {/* Unload Text Encoder */}
                <div className="flex items-center justify-between pt-2 border-t border-zinc-700">
                  <div>
                    <label className="text-[10px] text-zinc-300">Unload Text Encoder</label>
                    <p className="text-[10px] text-zinc-500">Free ~9GB after encode (single-GPU)</p>
                  </div>
                  <Toggle
                    on={gs.unloadTextEncoderAfterEncode ?? false}
                    onToggle={() => updateSettings({ ...gs, unloadTextEncoderAfterEncode: !gs.unloadTextEncoderAfterEncode })}
                  />
                </div>

                {/* Reload Pipeline Every N */}
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-[10px] text-zinc-300">Reload Pipeline Every N</label>
                    <p className="text-[10px] text-zinc-500">0 = off. VRAM defrag (try 4–8)</p>
                  </div>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={gs.reloadPipelineEveryNGens ?? 0}
                    onChange={(e) => updateSettings({ ...gs, reloadPipelineEveryNGens: Math.max(0, Math.min(100, parseInt(e.target.value) || 0)) })}
                    className="w-14 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-1 focus:ring-amber-500"
                  />
                </div>

                {/* Flush VRAM */}
                <div className="pt-2 border-t border-zinc-700">
                  <FlushVramButton />
                </div>
              </div>

              {/* Device Overrides */}
              <div className="bg-zinc-800/30 rounded-lg p-3 space-y-2">
                <p className="text-[10px] text-zinc-400 font-medium">
                  Device Overrides <span className="text-zinc-600 font-normal">(expert, blank = auto)</span>
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-[10px] text-zinc-400">Transformer</label>
                    <input
                      type="text"
                      value={gs.transformerDevice ?? ''}
                      onChange={(e) => updateSettings({ ...gs, transformerDevice: e.target.value })}
                      onKeyDown={(e) => e.stopPropagation()}
                      placeholder="cuda:0"
                      className="w-full px-2 py-1 bg-zinc-700 border border-zinc-700 rounded text-[10px] text-white placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-zinc-400">Text Encoder</label>
                    <input
                      type="text"
                      value={gs.textEncoderDevice ?? ''}
                      onChange={(e) => updateSettings({ ...gs, textEncoderDevice: e.target.value })}
                      onKeyDown={(e) => e.stopPropagation()}
                      placeholder="cuda:1"
                      className="w-full px-2 py-1 bg-zinc-700 border border-zinc-700 rounded text-[10px] text-white placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── LoRA Manager (collapsible, global) ── */}
        <div className="border-t border-zinc-800">
          <SectionHeader
            open={loraOpen}
            onToggle={() => setLoraOpen(v => !v)}
            icon={
              <svg className="h-3.5 w-3.5 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            }
            label="LoRA Manager"
            badge={
              activeLoras > 0 ? (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/20 text-violet-400">
                  {activeLoras} active
                </span>
              ) : undefined
            }
          />

          {loraOpen && (
            <div className="px-4 pb-4">
              <LoraManager
                value={gs.civitaiLoras ?? '[]'}
                onChange={(v) => updateSettings({ ...gs, civitaiLoras: v })}
              />
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-[10px] text-zinc-600 leading-tight border-t border-zinc-800 px-4 py-3 flex-shrink-0">
          Per-gen overrides snapshot at queue time. VRAM/LoRA changes are global — restart required.
        </p>
      </div>
    </div>
  )
}
