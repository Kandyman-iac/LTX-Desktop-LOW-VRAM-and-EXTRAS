import { useState, useEffect, useCallback, useRef } from 'react'
import { X, Film, RefreshCw, ChevronLeft, ChevronRight, Star, FolderInput, Check, ChevronDown, ChevronUp } from 'lucide-react'
import { backendFetch } from '../lib/backend'
import { useProjects } from '../contexts/ProjectContext'
import { COLOR_LABELS } from '../views/editor/video-editor-utils'

interface OutputEntry {
  filename: string
  path: string
  size_bytes: number
  modified_at: number
  prompt: string | null
  negative_prompt: string | null
  model: string | null
  resolution: string | null
  width: number | null
  height: number | null
  num_frames: number | null
  duration_seconds: number | null
  fps: number | null
  seed: number | null
  aspect_ratio: string | null
  camera_motion: string | null
  timestamp: string | null
  loras: { name: string; strength: number }[] | null
  render_time_seconds: number | null
}

interface OutputsResponse {
  entries: OutputEntry[]
  total: number
  page: number
  page_size: number
}

interface OutputBrowserProps {
  isOpen: boolean
  onClose: () => void
}

// ── Gallery annotations (stored in localStorage) ──────────────────────────────

interface GalleryAnnotation {
  rating?: number       // 1–5
  colorLabel?: string   // e.g. 'violet', 'red', …
  notes?: string
}

const STORAGE_KEY = 'ltx_gallery_annotations'

function loadAnnotations(): Record<string, GalleryAnnotation> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) as Record<string, GalleryAnnotation> : {}
  } catch {
    return {}
  }
}

function saveAnnotations(data: Record<string, GalleryAnnotation>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
  } catch { /* ignore */ }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function pathToFileUrl(p: string): string {
  const normalized = p.replace(/\\/g, '/')
  return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
}

// ── Star rating widget ────────────────────────────────────────────────────────

function StarRating({
  value,
  onChange,
  size = 'md',
}: {
  value: number | undefined
  onChange: (v: number | undefined) => void
  size?: 'sm' | 'md'
}) {
  const [hovered, setHovered] = useState<number | null>(null)
  const sz = size === 'sm' ? 'h-3 w-3' : 'h-4 w-4'
  const active = hovered ?? value ?? 0

  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((i) => (
        <button
          key={i}
          onMouseEnter={() => setHovered(i)}
          onMouseLeave={() => setHovered(null)}
          onClick={() => onChange(value === i ? undefined : i)}
          title={`${i} star${i > 1 ? 's' : ''}${value === i ? ' — click to clear' : ''}`}
          className="text-zinc-600 hover:text-yellow-400 transition-colors"
        >
          <Star
            className={`${sz} transition-colors ${i <= active ? 'fill-yellow-400 text-yellow-400' : ''}`}
          />
        </button>
      ))}
    </div>
  )
}

// ── Video card ────────────────────────────────────────────────────────────────

function VideoCard({
  entry,
  annotation,
  onClick,
}: {
  entry: OutputEntry
  annotation: GalleryAnnotation
  onClick: () => void
}) {
  const fileUrl = pathToFileUrl(entry.path)
  const colorDef = COLOR_LABELS.find(c => c.id === annotation.colorLabel)

  return (
    <div
      className={`group cursor-pointer rounded-lg overflow-hidden bg-zinc-900 border transition-colors ${
        colorDef ? `${colorDef.border} border-opacity-70` : 'border-zinc-800 hover:border-zinc-600'
      }`}
      onClick={onClick}
    >
      {/* Color label stripe at top */}
      {colorDef && (
        <div className="h-0.5 w-full" style={{ backgroundColor: colorDef.color }} />
      )}

      {/* Video thumbnail */}
      <div className="relative aspect-video bg-zinc-950 overflow-hidden">
        <video
          src={fileUrl}
          className="w-full h-full object-cover"
          muted
          preload="metadata"
          onMouseOver={(e) => { e.currentTarget.play().catch(() => {}) }}
          onMouseOut={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0 }}
        />
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/30">
          <Film className="h-8 w-8 text-white" />
        </div>
        {entry.resolution && (
          <span className="absolute bottom-1 right-1 text-[10px] bg-black/60 text-zinc-300 px-1 rounded">
            {entry.resolution}
          </span>
        )}
        {entry.duration_seconds != null && (
          <span className="absolute bottom-1 left-1 text-[10px] bg-black/60 text-zinc-300 px-1 rounded">
            {entry.duration_seconds}s
          </span>
        )}
      </div>

      <div className="p-2 space-y-1">
        {entry.prompt ? (
          <p className="text-xs text-zinc-300 truncate" title={entry.prompt}>{entry.prompt}</p>
        ) : (
          <p className="text-xs text-zinc-600 italic">{entry.filename}</p>
        )}
        <div className="flex items-center gap-2">
          {/* Stars (read-only on card) */}
          {annotation.rating ? (
            <div className="flex gap-0.5">
              {[1, 2, 3, 4, 5].map(i => (
                <Star
                  key={i}
                  className={`h-2.5 w-2.5 ${i <= annotation.rating! ? 'fill-yellow-400 text-yellow-400' : 'text-zinc-700'}`}
                />
              ))}
            </div>
          ) : null}
          <div className="flex items-center gap-2 text-[11px] text-zinc-500 ml-auto">
            {entry.model && <span>{entry.model}</span>}
            <span>{formatBytes(entry.size_bytes)}</span>
          </div>
        </div>
        {/* LoRA pills */}
        {entry.loras && entry.loras.length > 0 && (
          <div className="flex flex-wrap gap-1 pt-0.5">
            {entry.loras.map((l, i) => (
              <span
                key={i}
                className="text-[9px] px-1 py-0.5 rounded bg-violet-500/15 text-violet-400 border border-violet-500/20 truncate max-w-[90px]"
                title={`${l.name} — strength ${l.strength.toFixed(2)}`}
              >
                {l.name.replace(/\.safetensors$/i, '')}
              </span>
            ))}
          </div>
        )}
        {annotation.notes && (
          <p className="text-[10px] text-zinc-500 truncate" title={annotation.notes}>
            {annotation.notes}
          </p>
        )}
      </div>
    </div>
  )
}

// ── Lightbox modal ────────────────────────────────────────────────────────────

function LightboxModal({
  entry,
  annotation,
  onAnnotationChange,
  onClose,
}: {
  entry: OutputEntry
  annotation: GalleryAnnotation
  onAnnotationChange: (a: GalleryAnnotation) => void
  onClose: () => void
}) {
  const fileUrl = pathToFileUrl(entry.path)
  const { projects, addAsset } = useProjects()
  const [projectMenuOpen, setProjectMenuOpen] = useState(false)
  const [sentToProject, setSentToProject] = useState<string | null>(null)
  const [promptExpanded, setPromptExpanded] = useState(false)
  const projectMenuRef = useRef<HTMLDivElement>(null)

  // Close project dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (projectMenuRef.current && !projectMenuRef.current.contains(e.target as Node)) {
        setProjectMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSendToProject = (projectId: string) => {
    const project = projects.find(p => p.id === projectId)
    if (!project) return
    addAsset(projectId, {
      type: 'video',
      path: entry.path,
      url: fileUrl,
      prompt: entry.prompt ?? '',
      resolution: entry.resolution ?? '',
      duration: entry.duration_seconds ?? undefined,
      rating: annotation.rating,
      colorLabel: annotation.colorLabel,
      notes: annotation.notes,
    })
    setSentToProject(project.name)
    setProjectMenuOpen(false)
    setTimeout(() => setSentToProject(null), 2500)
  }

  const isLongPrompt = (entry.prompt?.length ?? 0) > 120

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative max-w-5xl w-full mx-auto rounded-xl overflow-hidden bg-zinc-900 shadow-2xl flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 z-10 p-1.5 rounded-lg bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
        >
          <X className="h-4 w-4" />
        </button>

        {/* Video — fixed, never scrolls away */}
        <video
          src={fileUrl}
          controls
          autoPlay
          className="w-full max-h-[55vh] object-contain bg-black flex-shrink-0"
        />

        {/* Scrollable info + annotation panel */}
        <div className="overflow-y-auto flex-1 p-4 space-y-3">

          {/* Prompt (collapsible when long) */}
          {entry.prompt && (
            <div>
              <p className={`text-sm text-zinc-200 ${!promptExpanded && isLongPrompt ? 'line-clamp-2' : ''}`}>
                {entry.prompt}
              </p>
              {isLongPrompt && (
                <button
                  onClick={() => setPromptExpanded(v => !v)}
                  className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-zinc-300 mt-0.5 transition-colors"
                >
                  {promptExpanded
                    ? <><ChevronUp className="h-3 w-3" /> Show less</>
                    : <><ChevronDown className="h-3 w-3" /> Show more</>
                  }
                </button>
              )}
            </div>
          )}

          {/* Generation metadata */}
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-zinc-500">
            {entry.model && <span><span className="text-zinc-400">Model:</span> {entry.model}</span>}
            {entry.resolution && <span><span className="text-zinc-400">Res:</span> {entry.resolution}</span>}
            {entry.duration_seconds != null && <span><span className="text-zinc-400">Duration:</span> {entry.duration_seconds}s</span>}
            {entry.fps != null && <span><span className="text-zinc-400">FPS:</span> {entry.fps}</span>}
            {entry.seed != null && <span><span className="text-zinc-400">Seed:</span> {entry.seed}</span>}
            {entry.aspect_ratio && <span><span className="text-zinc-400">Aspect:</span> {entry.aspect_ratio}</span>}
            {entry.render_time_seconds != null && <span><span className="text-zinc-400">Render:</span> {entry.render_time_seconds}s</span>}
            {entry.camera_motion && entry.camera_motion !== 'none' && (
              <span><span className="text-zinc-400">Camera:</span> {entry.camera_motion}</span>
            )}
            <span className="text-zinc-600">{entry.filename} · {formatBytes(entry.size_bytes)}</span>
          </div>

          {/* LoRAs used */}
          {entry.loras && entry.loras.length > 0 && (
            <div className="flex items-start gap-2">
              <span className="text-xs text-zinc-400 flex-shrink-0 mt-0.5">LoRAs</span>
              <div className="flex flex-wrap gap-1.5">
                {entry.loras.map((l, i) => (
                  <span
                    key={i}
                    className="text-xs px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300 border border-violet-500/25"
                    title={l.name}
                  >
                    {l.name.replace(/\.safetensors$/i, '')}
                    <span className="text-violet-500 ml-1">{l.strength.toFixed(2)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* ── Annotation controls ── */}
          <div className="border-t border-zinc-800 pt-3 space-y-3">

            {/* Row: stars + color labels + send-to-project */}
            <div className="flex items-center gap-4 flex-wrap">

              {/* Star rating */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-400">Rating</span>
                <StarRating
                  value={annotation.rating}
                  onChange={(v) => onAnnotationChange({ ...annotation, rating: v })}
                />
              </div>

              {/* Color labels */}
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-zinc-400">Label</span>
                <div className="flex gap-1">
                  <button
                    onClick={() => onAnnotationChange({ ...annotation, colorLabel: undefined })}
                    title="No label"
                    className={`w-4 h-4 rounded-full border transition-all ${
                      !annotation.colorLabel ? 'border-white scale-110' : 'border-zinc-600 hover:border-zinc-400'
                    } bg-zinc-800`}
                  />
                  {COLOR_LABELS.map(c => (
                    <button
                      key={c.id}
                      onClick={() => onAnnotationChange({
                        ...annotation,
                        colorLabel: annotation.colorLabel === c.id ? undefined : c.id,
                      })}
                      title={c.label}
                      className={`w-4 h-4 rounded-full border transition-all ${
                        annotation.colorLabel === c.id ? 'border-white scale-125' : 'border-transparent hover:scale-110'
                      }`}
                      style={{ backgroundColor: c.color }}
                    />
                  ))}
                </div>
              </div>

              {/* Send to project */}
              <div className="ml-auto relative" ref={projectMenuRef}>
                {sentToProject ? (
                  <span className="flex items-center gap-1.5 text-xs text-green-400">
                    <Check className="h-3.5 w-3.5" />
                    Added to {sentToProject}
                  </span>
                ) : (
                  <button
                    onClick={() => setProjectMenuOpen(v => !v)}
                    disabled={projects.length === 0}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-zinc-300 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    title={projects.length === 0 ? 'No projects — create one first' : 'Send clip to a project'}
                  >
                    <FolderInput className="h-3.5 w-3.5" />
                    Send to Project
                  </button>
                )}

                {projectMenuOpen && (
                  <div className="absolute right-0 bottom-full mb-1 w-52 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl overflow-hidden z-10">
                    <p className="px-3 py-1.5 text-[10px] text-zinc-500 border-b border-zinc-700">
                      Select project to add clip to
                    </p>
                    {projects.map(p => (
                      <button
                        key={p.id}
                        onClick={() => handleSendToProject(p.id)}
                        className="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-700 transition-colors flex items-center justify-between"
                      >
                        <span className="truncate">{p.name}</span>
                        <span className="text-[10px] text-zinc-600 ml-2 flex-shrink-0">
                          {p.assets.length} asset{p.assets.length !== 1 ? 's' : ''}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Notes */}
            <div className="space-y-1">
              <label className="text-xs text-zinc-400">Notes</label>
              <textarea
                value={annotation.notes ?? ''}
                onChange={(e) => onAnnotationChange({ ...annotation, notes: e.target.value || undefined })}
                placeholder="Add notes about this clip…"
                rows={2}
                className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-xs text-zinc-200 placeholder-zinc-600 resize-none focus:outline-none focus:border-zinc-500"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

const PAGE_SIZE = 24

export function OutputBrowser({ isOpen, onClose }: OutputBrowserProps) {
  const [data, setData] = useState<OutputsResponse | null>(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<OutputEntry | null>(null)
  const [annotations, setAnnotations] = useState<Record<string, GalleryAnnotation>>({})

  // Load annotations from localStorage when opening
  useEffect(() => {
    if (isOpen) {
      setAnnotations(loadAnnotations())
    }
  }, [isOpen])

  const fetchPage = useCallback(async (p: number) => {
    setLoading(true)
    setError(null)
    try {
      const res = await backendFetch(`/api/outputs?page=${p}&page_size=${PAGE_SIZE}`)
      if (!res.ok) throw new Error(`Request failed: ${res.status}`)
      const json: OutputsResponse = await res.json()
      setData(json)
      setPage(p)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load outputs')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isOpen) {
      void fetchPage(1)
    }
  }, [isOpen, fetchPage])

  const handleAnnotationChange = useCallback((filename: string, a: GalleryAnnotation) => {
    setAnnotations(prev => {
      const next = { ...prev, [filename]: a }
      // Prune empty entries
      if (!a.rating && !a.colorLabel && !a.notes) {
        delete next[filename]
      }
      saveAnnotations(next)
      return next
    })
    // Also update selected entry's annotation in lightbox if open
  }, [])

  if (!isOpen) return null

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1

  return (
    <>
      <div className="fixed inset-0 z-[70] flex flex-col bg-zinc-950">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-3">
            <Film className="h-5 w-5 text-zinc-400" />
            <h2 className="text-base font-semibold text-white">Output Browser</h2>
            {data && (
              <span className="text-sm text-zinc-500">{data.total} video{data.total !== 1 ? 's' : ''}</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => fetchPage(page)}
              disabled={loading}
              className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors disabled:opacity-50"
              title="Refresh"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-2">
                <p className="text-red-400 text-sm">{error}</p>
                <button
                  onClick={() => fetchPage(page)}
                  className="text-xs text-zinc-400 hover:text-zinc-200 underline"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : loading && !data ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="h-8 w-8 text-zinc-600 animate-spin" />
            </div>
          ) : data && data.entries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-zinc-500 space-y-2">
              <Film className="h-12 w-12 text-zinc-700" />
              <p className="text-sm">No generated videos yet.</p>
              <p className="text-xs text-zinc-600">Videos will appear here after you generate them.</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {data?.entries.map((entry) => (
                <VideoCard
                  key={entry.filename}
                  entry={entry}
                  annotation={annotations[entry.filename] ?? {}}
                  onClick={() => setSelected(entry)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Pagination */}
        {data && data.total > PAGE_SIZE && (
          <div className="flex items-center justify-center gap-4 px-6 py-3 border-t border-zinc-800 flex-shrink-0">
            <button
              onClick={() => fetchPage(page - 1)}
              disabled={page <= 1 || loading}
              className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors disabled:opacity-30"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span className="text-sm text-zinc-400">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => fetchPage(page + 1)}
              disabled={page >= totalPages || loading}
              className="p-1.5 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors disabled:opacity-30"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>

      {selected && (
        <LightboxModal
          entry={selected}
          annotation={annotations[selected.filename] ?? {}}
          onAnnotationChange={(a) => handleAnnotationChange(selected.filename, a)}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  )
}
