import { useEffect, useState } from 'react'
import { Ban, Check, CircleDashed, Clapperboard, FileText, Loader2, MessageSquareText, PenLine, ScanSearch, Sparkles, Square, UserRoundCheck, X } from 'lucide-react'
import { api } from './api'

type TimelineEvent = { kind: string; label: string; status: string; at: string | null; seconds: number; job_id: string | null; detail: string | null }
type TimelineData = { events: TimelineEvent[]; total_seconds: number }

// One colour per kind of processing, so done and pending work can be told apart at a glance.
export const KINDS: Record<string, { color: string; icon: typeof Check }> = {
  created: { color: '#8b95a7', icon: Clapperboard },
  character_boxes: { color: '#f0a64a', icon: Square },
  character_identity: { color: '#e2709c', icon: UserRoundCheck },
  visual_analysis: { color: '#6aa8ff', icon: ScanSearch },
  consistency_summary: { color: '#b496ff', icon: Sparkles },
  review_answers: { color: '#4cc9c0', icon: MessageSquareText },
  answers_saved: { color: '#7bd88f', icon: PenLine },
  cut_summary: { color: '#e5d25a', icon: FileText },
}
const STATUS_ICON: Record<string, typeof Check> = { done: Check, failed: X, cancelled: Ban, running: Loader2, queued: CircleDashed, pending: CircleDashed }

export function formatDuration(seconds: number) {
  const s = Math.round(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}

function formatTime(at: string | null) {
  return at ? new Date(at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : 'time not recorded'
}

export default function AnalysisTimeline({ cutId, refreshKey }: { cutId: string; refreshKey: string }) {
  const [data, setData] = useState<TimelineData | null>(null)
  const [error, setError] = useState('')
  const [bump, setBump] = useState(0)
  useEffect(() => { const on = () => setBump(value => value + 1); window.addEventListener('framewise:cut-timeline-changed', on); return () => window.removeEventListener('framewise:cut-timeline-changed', on) }, [])
  const active = !!data?.events.some(event => event.status === 'running' || event.status === 'queued')
  useEffect(() => {
    let cancelled = false
    const load = () => api<TimelineData>(`/cuts/${cutId}/timeline`).then(value => { if (!cancelled) { setData(value); setError('') } }).catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : 'Timeline unavailable') })
    void load()
    const timer = active ? setInterval(load, 5000) : undefined
    return () => { cancelled = true; if (timer) clearInterval(timer) }
  }, [cutId, refreshKey, active, bump])
  if (error && !data) return <nav aria-label="Analysis timeline" className="border-b hairline px-5 py-2 text-xs text-[#ffb6bc]">Analysis timeline unavailable: {error}</nav>
  if (!data) return <nav aria-label="Analysis timeline" className="border-b hairline px-5 py-2 text-xs muted">Loading analysis timeline…</nav>
  const total = Math.max(data.total_seconds, 1)
  return <nav aria-label="Analysis timeline" className="shrink-0 border-b hairline px-5 py-2.5">
    <div className="mb-1.5 flex items-baseline justify-between gap-3"><h3 className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#a58ae8]">Analysis timeline</h3><p className="text-[11px] muted">Total time spent: <span className="font-semibold text-white">{formatDuration(data.total_seconds)}</span> <span className="hidden sm:inline">· queue time excluded</span></p></div>
    <ol className="scrollbar flex gap-1 overflow-x-auto pb-1">
      {data.events.map((event, index) => {
        const kind = KINDS[event.kind] ?? KINDS.created
        const Icon = kind.icon
        const StatusIcon = STATUS_ICON[event.status] ?? Check
        const pending = event.status === 'pending' || event.status === 'queued'
        const bad = event.status === 'failed' || event.status === 'cancelled'
        const grow = event.seconds > 0 ? Math.max(1, (event.seconds / total) * 24) : 0
        const title = `${event.label} — ${event.status}\n${formatTime(event.at)} · ${formatDuration(event.seconds)}${event.detail ? `\n${event.detail}` : ''}`
        return <li key={`${event.job_id ?? event.kind}-${index}`} title={title} style={{ flexGrow: grow, flexBasis: grow ? undefined : 'auto', minWidth: '7.5rem', borderColor: kind.color, background: pending ? 'transparent' : `${kind.color}${bad ? '14' : '26'}`, borderStyle: pending ? 'dashed' : 'solid', opacity: pending ? 0.6 : 1 }} className="flex shrink-0 items-center gap-2 rounded-md border px-2 py-1.5">
          <Icon size={14} style={{ color: kind.color }} className="shrink-0" aria-hidden />
          <span className="min-w-0 flex-1"><span className="block truncate text-[11px] font-semibold leading-tight">{event.label}</span><span className="block truncate text-[10px] leading-tight muted">{event.status === 'pending' ? 'Pending' : event.status === 'queued' ? 'Queued' : formatDuration(event.seconds)}{event.status !== 'pending' && event.at ? ` · ${new Date(event.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : ''}</span></span>
          <StatusIcon size={13} className={`shrink-0 ${event.status === 'running' ? 'animate-spin' : ''} ${bad ? 'text-[#ff8a94]' : event.status === 'done' ? 'text-[#7bd88f]' : 'muted'}`} aria-label={event.status} />
        </li>
      })}
    </ol>
  </nav>
}
