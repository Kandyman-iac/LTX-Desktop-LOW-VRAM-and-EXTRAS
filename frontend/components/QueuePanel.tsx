import { useState } from 'react'
import { ChevronDown, ChevronUp, X, Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react'
import { Button } from './ui/button'
import type { QueuedJob } from '../hooks/use-queue'

interface QueuePanelProps {
  jobs: QueuedJob[]
  onRemove: (jobId: string) => void
}

function pathToFileUrl(p: string): string {
  const normalized = p.replace(/\\/g, '/')
  return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
}

function StatusIcon({ status }: { status: QueuedJob['status'] }) {
  if (status === 'pending') return <Clock className="h-4 w-4 text-zinc-400" />
  if (status === 'running') return <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
  if (status === 'complete') return <CheckCircle className="h-4 w-4 text-green-400" />
  if (status === 'error') return <AlertCircle className="h-4 w-4 text-red-400" />
  return <X className="h-4 w-4 text-zinc-500" />
}

function StatusLabel({ status }: { status: QueuedJob['status'] }) {
  const labels: Record<QueuedJob['status'], string> = {
    pending: 'Pending',
    running: 'Running',
    complete: 'Done',
    error: 'Failed',
    cancelled: 'Cancelled',
  }
  const colors: Record<QueuedJob['status'], string> = {
    pending: 'text-zinc-400',
    running: 'text-blue-400',
    complete: 'text-green-400',
    error: 'text-red-400',
    cancelled: 'text-zinc-500',
  }
  return <span className={`text-xs font-medium ${colors[status]}`}>{labels[status]}</span>
}

export function QueuePanel({ jobs, onRemove }: QueuePanelProps) {
  const [expanded, setExpanded] = useState(true)

  const activeJobs = jobs.filter(j => j.status !== 'cancelled')
  if (activeJobs.length === 0) return null

  const runningCount = activeJobs.filter(j => j.status === 'running').length
  const pendingCount = activeJobs.filter(j => j.status === 'pending').length

  return (
    <div className="mt-4 border border-zinc-700 rounded-lg bg-zinc-900 overflow-hidden">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center justify-between px-4 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800"
      >
        <span className="flex items-center gap-2">
          Queue
          {runningCount > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-blue-600 text-white text-xs">{runningCount} running</span>
          )}
          {pendingCount > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-300 text-xs">{pendingCount} pending</span>
          )}
        </span>
        {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>

      {expanded && (
        <div className="divide-y divide-zinc-800">
          {activeJobs.map(job => (
            <div key={job.id} className="flex items-start gap-3 px-4 py-3">
              <StatusIcon status={job.status} />
              <div className="flex-1 min-w-0">
                <p className="text-xs text-zinc-300 truncate">{job.prompt}</p>
                <div className="flex items-center gap-2 mt-0.5">
                  <StatusLabel status={job.status} />
                  {job.status === 'complete' && job.result_path && (
                    <a
                      href={pathToFileUrl(job.result_path)}
                      className="text-xs text-blue-400 hover:underline"
                      onClick={e => {
                        e.preventDefault()
                        window.electronAPI?.showItemInFolder?.(job.result_path!)
                      }}
                    >
                      Show in folder
                    </a>
                  )}
                  {job.status === 'error' && job.error && (
                    <span className="text-xs text-red-400 truncate">{job.error}</span>
                  )}
                </div>
              </div>
              {(job.status === 'pending' || job.status === 'complete' || job.status === 'error') && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onRemove(job.id)}
                  className="h-6 w-6 p-0 text-zinc-500 hover:text-zinc-300"
                >
                  <X className="h-3 w-3" />
                </Button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
