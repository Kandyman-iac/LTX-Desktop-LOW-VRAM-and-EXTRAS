import { useState, useEffect, useCallback } from 'react'
import { X, Film, RefreshCw, ChevronLeft, ChevronRight } from 'lucide-react'
import { backendFetch } from '../lib/backend'

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

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function pathToFileUrl(p: string): string {
  const normalized = p.replace(/\\/g, '/')
  return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
}

function VideoCard({ entry, onClick }: { entry: OutputEntry; onClick: () => void }) {
  const fileUrl = pathToFileUrl(entry.path)

  return (
    <div
      className="group cursor-pointer rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 hover:border-zinc-600 transition-colors"
      onClick={onClick}
    >
      {/* Video thumbnail via video element */}
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

      <div className="p-2 space-y-0.5">
        {entry.prompt ? (
          <p className="text-xs text-zinc-300 truncate" title={entry.prompt}>{entry.prompt}</p>
        ) : (
          <p className="text-xs text-zinc-600 italic">{entry.filename}</p>
        )}
        <div className="flex items-center gap-2 text-[11px] text-zinc-500">
          {entry.model && <span>{entry.model}</span>}
          {entry.seed != null && <span>seed {entry.seed}</span>}
          <span className="ml-auto">{formatBytes(entry.size_bytes)}</span>
        </div>
      </div>
    </div>
  )
}

function LightboxModal({ entry, onClose }: { entry: OutputEntry; onClose: () => void }) {
  const fileUrl = pathToFileUrl(entry.path)

  return (
    <div
      className="fixed inset-0 z-[80] flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative max-w-5xl w-full mx-4 rounded-xl overflow-hidden bg-zinc-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 z-10 p-1.5 rounded-lg bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
        >
          <X className="h-4 w-4" />
        </button>

        <video
          src={fileUrl}
          controls
          autoPlay
          className="w-full max-h-[70vh] object-contain bg-black"
        />

        <div className="p-4 space-y-2">
          {entry.prompt && (
            <p className="text-sm text-zinc-200">{entry.prompt}</p>
          )}
          {entry.negative_prompt && (
            <p className="text-xs text-zinc-500">
              <span className="text-zinc-400">Negative:</span> {entry.negative_prompt}
            </p>
          )}
          <div className="flex flex-wrap gap-3 text-xs text-zinc-500">
            {entry.model && <span><span className="text-zinc-400">Model:</span> {entry.model}</span>}
            {entry.resolution && <span><span className="text-zinc-400">Res:</span> {entry.resolution}</span>}
            {entry.duration_seconds != null && <span><span className="text-zinc-400">Duration:</span> {entry.duration_seconds}s</span>}
            {entry.fps != null && <span><span className="text-zinc-400">FPS:</span> {entry.fps}</span>}
            {entry.seed != null && <span><span className="text-zinc-400">Seed:</span> {entry.seed}</span>}
            {entry.aspect_ratio && <span><span className="text-zinc-400">Aspect:</span> {entry.aspect_ratio}</span>}
            {entry.camera_motion && entry.camera_motion !== 'none' && (
              <span><span className="text-zinc-400">Camera:</span> {entry.camera_motion}</span>
            )}
            <span className="ml-auto text-zinc-600">{entry.filename} · {formatBytes(entry.size_bytes)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

const PAGE_SIZE = 24

export function OutputBrowser({ isOpen, onClose }: OutputBrowserProps) {
  const [data, setData] = useState<OutputsResponse | null>(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<OutputEntry | null>(null)

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
        <LightboxModal entry={selected} onClose={() => setSelected(null)} />
      )}
    </>
  )
}
