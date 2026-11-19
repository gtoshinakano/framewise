/** Types and pure helpers of the episode queue (backend `runs.py`): words, numbers and grouping, no React. */

export type RunStatus = 'queued' | 'running' | 'stopping' | 'done' | 'failed' | 'cancelled'
export type QueueSummary = {
  busy: boolean; since: string | null
  /** Individual GPU steps of this episode in its current busy period. */
  done: number; failed: number; running: number; waiting: number; stopping: number; cancelled: number; total: number; finished: number
  /** Failures nobody has run again or dismissed. */
  attention: number
  /** Queued or running media work (CPU), which the counter leaves out. */
  other_active: number
}
export type RunCut = { id: string; number: number | null; title: string; archived: boolean; missing: boolean; start_time: number | null; end_time: number | null; completed: boolean }
export type Run = {
  /** Cursor of the history pages (newest first). */
  id: string; seq: number; kind: string; label: string; group: string; gpu: boolean; status: RunStatus; stage: string; error: string | null
  priority_class: 'chat' | 'interactive' | 'batch'
  cuts: RunCut[]; cut_count: number; cut_span: { first: number; last: number } | null
  sequence: { id: string; title: string; step: string | null; shot_key: string | null; shot_number: number | null } | null
  range: { start: number; end: number } | null
  units: { done: number; total: number }
  created_at: string | null; started_at: string | null; finished_at: string | null; duration_seconds: number | null
  retry_of: string | null; dismissed: boolean
  actions: { cancel: boolean; retry: boolean; dismiss: boolean; delete: boolean }
}
export type ChatRun = {
  id: string; kind: 'chat'; label: string; status: 'queued' | 'running' | 'stopping'; stage: string; provider: string; model: string; title: string
  cut: { id: string; number: number | null; title: string; archived: boolean }
  sequence: { id: string; title: string | null } | null
  created_at: string | null; started_at: string | null; duration_seconds: number | null
}
export type WaitingGroup = { group: string; kind: string; label: string; gpu: boolean; count: number; cancel: boolean }
export type OtherEpisode = { episode_id: string; title: string; project_title: string; running: number; waiting: number; current: Run | null }
export type ChatApiRequest = { id: string; status: 'queued' | 'running' | 'stopping'; stage: string; model: string; created_at: string | null; started_at: string | null; duration_seconds: number | null }
export type QueueState = {
  summary: QueueSummary; running: Run[]; waiting: Run[]; waiting_total: number; waiting_groups: WaitingGroup[]
  attention: Run[]; chat: ChatRun[]; elsewhere: { episodes: OtherEpisode[]; chat_api: ChatApiRequest[] }
}
export type HistoryFilter = 'all' | 'done' | 'failed' | 'cancelled'
export type RunPage = { runs: Run[]; next: number | null; counts: { done: number; failed: number; cancelled: number } }

const plural = (count: number, one: string, many = `${one}s`) => `${count} ${count === 1 ? one : many}`
const pad = (value: number) => String(value).padStart(2, '0')

/** A run's length in minutes and seconds: `42s`, `5m 03s`, `75m 10s` (never hours). */
export function formatRunTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return ''
  const total = Math.max(0, Math.floor(seconds))
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${pad(total % 60)}s`
}

/** `M:SS` on the episode clock, for a stretch of audio. */
export function clockText(seconds: number) {
  const total = Math.max(0, Math.round(seconds))
  return `${Math.floor(total / 60)}:${pad(total % 60)}`
}

export const parseUtc = (value: string | null | undefined) => value ? Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(value) ? value : `${value.replace(' ', 'T')}Z`) : NaN

/** Seconds a run has taken so far (running) or took (finished). */
export function runSeconds(run: Pick<Run, 'status' | 'started_at' | 'duration_seconds'>, now: number): number | null {
  if ((run.status === 'running' || run.status === 'stopping') && run.started_at) {
    const start = parseUtc(run.started_at)
    if (Number.isFinite(start)) return Math.max(0, (now - start) / 1000)
  }
  return run.duration_seconds
}

export function cutText(cut: Pick<RunCut, 'number' | 'title' | 'archived' | 'missing'>, withTitle = true) {
  const name = cut.missing ? 'Deleted cut' : cut.archived ? `Archived cut${cut.number ? ` (was ${cut.number})` : ''}` : cut.number ? `Cut ${cut.number}` : 'Cut'
  return withTitle && cut.title ? `${name} · ${cut.title}` : name
}

/** What a run works on, in words: its cut, cuts, sequence shot, stretch of audio, or the whole episode. */
export function targetText(run: Pick<Run, 'cuts' | 'cut_count' | 'cut_span' | 'sequence' | 'range'>): string {
  const cut = run.cuts[0]
  if (run.sequence) {
    const shot = run.sequence.shot_number ? ` · Shot ${run.sequence.shot_number}` : ''
    return `Sequence “${run.sequence.title || 'Untitled'}”${shot}${cut ? ` · ${cutText(cut, false)}` : ''}`
  }
  if (run.cut_count === 1 && cut) return cutText(cut)
  if (run.cut_count > 1) {
    const span = run.cut_span
    return span && span.first !== span.last ? `Cuts ${span.first}–${span.last} (${run.cut_count})` : plural(run.cut_count, 'cut')
  }
  if (run.range) return `${clockText(run.range.start)}–${clockText(run.range.end)} of the audio`
  return 'Whole episode'
}

export const STATUS_TEXT: Record<RunStatus, string> = { queued: 'Waiting', running: 'Running', stopping: 'Stopping', done: 'Done', failed: 'Failed', cancelled: 'Cancelled' }
export type StatusTone = 'green' | 'purple' | 'amber' | 'red' | 'grey'
/** Cancelled is grey: it is the user's decision, never a failure. */
export const STATUS_TONE: Record<RunStatus, StatusTone> = { queued: 'amber', running: 'purple', stopping: 'amber', done: 'green', failed: 'red', cancelled: 'grey' }

/** `3/12 steps` for a run that holds several GPU steps, else nothing. */
export function unitsText(run: Pick<Run, 'units' | 'status'>) {
  return run.units.total > 1 ? `${run.status === 'done' ? run.units.total : run.units.done}/${run.units.total} steps` : ''
}

/** The header counter: finished/total GPU steps while the episode is busy, nothing when idle. */
export function counterText(summary: QueueSummary | null | undefined) {
  return summary?.busy && summary.total > 0 ? `${summary.finished}/${summary.total}` : ''
}

/** The counter spelled out, for its tooltip and screen readers. */
export function counterTitle(summary: QueueSummary | null | undefined) {
  if (!summary?.busy || !summary.total) return summary?.other_active ? 'Media work is running in this episode. Open the queue to see it.' : 'No GPU work is queued in this episode. Open the queue to see its history.'
  const finished = summary.failed ? `${summary.done} done, ${summary.failed} failed` : `${summary.done} done`
  const parts = [`${summary.finished} of ${summary.total} GPU steps finished in this episode: ${finished}`]
  if (summary.running) parts.push(`${summary.running} running`)
  if (summary.waiting) parts.push(`${summary.waiting} waiting`)
  if (summary.stopping) parts.push(`${plural(summary.stopping, 'cancelled run')} still stopping`)
  return `${parts.join(' · ')}.`
}

export const attentionText = (count: number) => `${count} failed`

export function attentionTitle(count: number) {
  return count === 1
    ? '1 run failed and was not run again. Open the queue to retry or dismiss it.'
    : `${count} runs failed and were not run again. Open the queue to retry or dismiss them.`
}

/** Start time in the viewer's time zone: `14:03` today, `Yesterday 09:12`, else `12 Nov 14:03`. */
export function localTime(value: string | null | undefined, now = Date.now()) {
  const at = parseUtc(value)
  if (!Number.isFinite(at)) return ''
  const date = new Date(at), today = new Date(now)
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`
  const day = (when: Date) => `${when.getFullYear()}-${when.getMonth()}-${when.getDate()}`
  if (day(date) === day(today)) return time
  if (day(date) === day(new Date(now - 86_400_000))) return `Yesterday ${time}`
  return `${date.getDate()} ${date.toLocaleString('en', { month: 'short' })} ${time}`
}

/** A stage worth showing under the run name (not just its status repeated). */
export function stageText(run: Pick<Run, 'stage' | 'status'>) {
  if (run.status === 'stopping') return 'Stopping — finishing the current step'
  const stage = (run.stage || '').trim()
  if (!stage || /^(queued|starting|complete|cancelled|failed)$/i.test(stage)) return ''
  return stage.replace(/^Failed\s*·\s*/i, '')
}

/** Merge a freshly loaded first page into the runs already shown, keeping the older pages the user loaded. */
export function mergeFirstPage(shown: Run[], first: Run[]): Run[] {
  if (!first.length) return []
  const oldest = first[first.length - 1].seq
  return [...first, ...shown.filter(run => run.seq < oldest)]
}
