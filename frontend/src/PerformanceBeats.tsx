import { useState } from 'react'
import { ChevronDown, Drama, RotateCcw } from 'lucide-react'
import { api, type AnalysisResult, type Job, type PerformanceBeats as Beats } from './api'
import { ActionButton, StaleChip, StatusChip } from './RunUi'

const DELIVERY: Record<string, string> = { on_screen: 'lip-synced', off_screen: 'off screen', inner_voice: 'inner voice', narrative: 'narration' }

/**
 * The cut's performance beats: the timed acting plan the text model compiled from the keyframes. Speech shows as who and
 * how it is delivered, never the words. Stale when the keyframes were analysed again after the beats were written.
 */
export function PerformanceBeats({ cutId, results, jobs, disabled, onStarted }: { cutId: string; results: AnalysisResult[]; jobs: Job[]; disabled: boolean; onStarted: () => Promise<void> }) {
  const [queueing, setQueueing] = useState(false)
  const [error, setError] = useState('')
  const keyframes = results.find(result => result.settings.structured_type === 'keyframe_analysis' && !result.superseded_by)
  const result = results.find(result => result.settings.structured_type === 'performance_beats' && !result.superseded_by)
  const visual = jobs.some(job => job.pipeline === 'analyze' && ['queued', 'running'].includes(job.status) && job.cuts?.some(item => item.id === cutId))
  if (!keyframes && !result && !visual) return null
  let beats: Beats | null = null
  try { beats = result ? JSON.parse(result.content) as Beats : null } catch { beats = null }
  const pending = jobs.find(job => job.pipeline === 'performance_beats' && ['queued', 'running'].includes(job.status) && job.cuts?.some(item => item.id === cutId))
  const staleReason = result?.stale_reason || (beats && keyframes && beats.based_on?.keyframe_analysis !== keyframes.id ? 'The keyframe analysis changed' : '')
  async function rewrite() {
    setQueueing(true); setError('')
    try { await api(`/cuts/${cutId}/performance-beats`, { method: 'POST' }); await onStarted() } catch (failure) { setError((failure as Error).message) } finally { setQueueing(false) }
  }
  const status = pending ? <StatusChip tone={pending.status === 'running' ? 'purple' : 'amber'}>{pending.status === 'running' ? 'Writing' : 'Queued'}</StatusChip>
    : staleReason ? <StaleChip reason={staleReason} />
    : beats ? <StatusChip tone="green">Current</StatusChip> : <StatusChip tone="grey">Not written</StatusChip>
  return <details open className="group shrink-0 rounded-lg border hairline bg-[#171c25] text-xs">
    <summary className="flex cursor-pointer list-none items-center gap-1.5 px-3 py-2 font-semibold text-[#d5dbe5]"><ChevronDown size={14} className="-rotate-90 transition-transform group-open:rotate-0" aria-hidden />
      <Drama size={14} className="text-[#f08a5d]" aria-hidden />Performance beats<span className="ml-auto">{status}</span></summary>
    <div className="space-y-2 border-t hairline p-3">
      {beats?.summary && <p className="text-[#d5dbe5]">{beats.summary}</p>}
      {!beats && <p className="muted">The keyframes have no beats yet. They are written after the keyframe analysis, or with the button below.</p>}
      {beats && <ol className="space-y-2">{beats.beats.map((beat, index) => <li key={index} className="rounded border hairline p-2">
        <div className="flex flex-wrap items-center gap-1.5"><strong className="tabular-nums text-[#e8ebf1]">{beat.start.toFixed(2)}–{beat.end.toFixed(2)}s</strong>
          {beat.intent && <span className="text-[#f0a37f]">{beat.intent}</span>}<span className="muted ml-auto">camera: {beat.camera.move}</span></div>
        <ul className="mt-1 space-y-0.5">{beat.characters.map(person => <li key={person.character}><span className="font-semibold">{person.character}</span>
          <span className="muted"> · body </span>{person.body}<span className="muted"> · face </span>{person.face}<span className="muted"> · gaze </span>{person.gaze}<span className="muted"> · mouth </span>{person.mouth}</li>)}</ul>
        {beat.speech.length > 0 && <div className="mt-1 flex flex-wrap gap-1">{beat.speech.map(line => <span key={line.line} className="rounded bg-[#2a3140] px-1.5 py-0.5 text-[10px]" title={`${line.start.toFixed(2)}–${line.end.toFixed(2)}s`}>
          {line.line} · {line.speaker} · {DELIVERY[line.delivery] ?? line.delivery}</span>)}</div>}
      </li>)}</ol>}
      {error && <p role="alert" className="text-[#ff9fa7]">{error}</p>}
      {(keyframes || visual) && <ActionButton compact icon={<RotateCcw size={13} />} pending={queueing} disabled={disabled} title="Queue beats from the current keyframe analysis" onClick={() => void rewrite()}>{beats ? 'Write beats again' : 'Write beats'}</ActionButton>}
    </div>
  </details>
}
