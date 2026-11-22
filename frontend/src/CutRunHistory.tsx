import { memo, useState } from 'react'
import { Ban, Eye, History, RotateCcw } from 'lucide-react'
import { api, type Job } from './api'
import { cutRunOutcome } from './cutRunHistory'
import { durationText } from './dataView'
import { RunDetailsDialog } from './RunDetailsDialog'
import { ActionButton, Elapsed, JobChip, StatusChip, formatElapsed, jobTone, type Tone } from './RunUi'
import type { StageRow } from './runState'

const stageLabels: Record<string, string> = { subtitle_ocr: 'Subtitle detection and OCR', initial_frame: 'Initial frame', keyframe_analysis: 'Keyframe analysis', video_analysis: 'Video analysis', last_frame: 'Last frame' }
// Stages a run still has to start. Runs from before the last frame stopped being analyzed list their own last_frame stage row.
const STAGE_ORDER = [['initial_frame', 'Initial frame'], ['keyframe_analysis', 'Keyframe analysis']] as const
const EDGE: Record<Tone, string> = { green: 'border-l-[#7bdcb2]', purple: 'border-l-[#c3aeff]', amber: 'border-l-[#e6c58b]', red: 'border-l-[#ff9fa7]', grey: 'border-l-[#4a5262]', blue: 'border-l-[#6aa8ff]' }
const PREVIEW = 5

const runName = (job: Job) => job.pipeline === 'regen' ? 'Cut summary' : job.pipeline === 'consistency_summary' ? 'Consistency summary' : job.pipeline === 'performance_beats' ? 'Performance beats' : job.stages?.length === 1 ? stageLabels[job.stages[0]] : 'Visual analysis'
const stageTone = (status: string): Tone => status === 'done' || status === 'complete' ? 'green' : status === 'running' ? 'purple' : status === 'failed' || status === 'interrupted' ? 'red' : 'grey'

function minutes(seconds: number) { return formatElapsed(seconds) }

export const CutRunHistory = memo(function CutRunHistory({ cutId, jobs, stages, onChanged, loaded = true }: { cutId: string; jobs: Job[]; stages: StageRow[]; onChanged: () => Promise<void>; /** False while the cut's full job history is still being read: the list may be incomplete. */ loaded?: boolean }) {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [working, setWorking] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [showAll, setShowAll] = useState(false)
  const runs = jobs.filter(job => ['analyze', 'regen', 'consistency_summary'].includes(job.pipeline) && job.cuts?.some(cut => cut.id === cutId))
  const visible = showAll ? runs : runs.slice(0, PREVIEW)
  async function action(job: Job, kind: 'cancel' | 'retry') {
    setWorking(job.id); setError('')
    try { await api(`/jobs/${job.id}/${kind}`, { method: 'POST' }); void onChanged() }
    catch (reason) { setError((reason as Error).message) } finally { setWorking(null) }
  }
  return <section aria-label="Cut run history" className="rounded-xl border hairline bg-[#171c25] p-3 text-xs">
    <h3 className="flex items-center gap-2 font-semibold"><History size={14} className="text-[#bca7ff]" /> Run history <span className="rounded-full bg-[#232a36] px-1.5 text-[11px] font-normal text-[#aeb5c1]">{runs.length}</span><span className="muted ml-auto text-[11px] font-normal" title="Cancelling stops after the current model operation. Completed work can be retried.">Times in UTC</span></h3>
    {error && <p role="alert" className="mt-2 text-[#ff9fa7]">{error}</p>}
    {runs.length === 0 ? <p className="muted mt-3 rounded-md border border-dashed hairline px-3 py-4 text-center">{loaded ? 'No analysis runs recorded for this cut yet.' : 'Loading the runs of this cut…'}</p> :
      <div className="mt-2"><table className="w-full border-separate border-spacing-y-1.5 text-left">
        <thead className="muted text-[10px] uppercase tracking-wider"><tr><th className="w-28 px-2 font-semibold">Run</th><th className="px-2 font-semibold">Step and stages</th></tr></thead>
        <tbody>{visible.map(job => {
          const outcome = cutRunOutcome(job, cutId)
          if (!outcome) return null
          const when = job.finished_at ? ['Finished', job.finished_at] : job.started_at ? ['Started', job.started_at] : ['Queued', job.created_at]
          const mine = stages.filter(stage => stage.job_id === job.id)
          const running = job.status === 'running' || job.status === 'queued'
          // What may be done comes from the backend's job registry (job_kinds.py).
          const canCancel = !!job.actions?.cancel && running
          const canRetry = !!job.actions?.retry
          const waiting = job.workflow === 'single_cut_v2' ? STAGE_ORDER.filter(([key]) => (!job.stages || job.stages.includes(key)) && !mine.some(stage => stage.stage_key === key)) : []
          return <tr key={job.id} className="align-top">
            <td className={`rounded-l-lg border-l-4 bg-[#1e2430] px-2 py-2 ${EDGE[jobTone(outcome.status)]}`}>
              <button type="button" onClick={() => setSelectedJob(job)} className="block text-left font-semibold leading-4 text-[#c3aeff] hover:underline" title="Open run details">{runName(job)}</button>
              <span className="mt-1.5 block"><JobChip status={outcome.status} /></span>
              <time className="muted mt-1.5 block text-[11px] leading-4" dateTime={`${when[1].replace(' ', 'T')}Z`} title={`${when[0]} ${when[1]} UTC`}>{when[0]} {when[1].slice(5, 16)}</time>
              <span className="block text-[11px] text-[#aeb5c1]">{running && job.started_at ? <Elapsed since={job.started_at} /> : durationText(job.started_at, job.finished_at)}</span>
            </td>
            <td className="rounded-r-lg bg-[#1e2430] px-2 py-2">
              <p className="break-words text-[#d5dbe5]"><span className="muted">Step: </span>{outcome.step}</p>
              {(mine.length > 0 || waiting.length > 0) && <ul className="mt-1.5 flex flex-wrap gap-1">{mine.map(stage => <li key={stage.id}><StatusChip tone={stageTone(stage.status)} title={stage.error || undefined}>{stage.label} · {stage.status} · {minutes(stage.elapsed_seconds)}</StatusChip></li>)}{waiting.map(([key, label]) => <li key={key}><StatusChip tone="grey">{label} · {job.status === 'failed' ? 'not started' : 'waiting'}</StatusChip></li>)}</ul>}
              {outcome.error && <p className="mt-1.5 whitespace-pre-wrap break-words rounded-md border border-[#63363e] bg-[#2b191f] px-2 py-1 text-[#ffb6bc]">{outcome.error}</p>}
              <div className="mt-2 flex flex-wrap justify-end gap-1">
                <ActionButton compact icon={<Eye size={13} />} onClick={() => setSelectedJob(job)} title="Open run details" aria-label={`Details of ${runName(job)}`}>Details</ActionButton>
                {canCancel && <ActionButton compact variant="danger" icon={<Ban size={13} />} pending={working === job.id} disabled={working !== null} onClick={() => void action(job, 'cancel')} title="Cancel run (stops after the current model operation)">Cancel</ActionButton>}
                {canRetry && <ActionButton compact icon={<RotateCcw size={13} />} pending={working === job.id} disabled={working !== null} onClick={() => void action(job, 'retry')} title="Retry the remaining work">Retry</ActionButton>}
              </div>
            </td>
          </tr>
        })}</tbody>
      </table></div>}
    {runs.length > PREVIEW && <button type="button" className="mt-1 text-[11px] text-[#c3aeff] hover:underline" onClick={() => setShowAll(!showAll)}>{showAll ? 'Show fewer' : `Show all ${runs.length} runs`}</button>}
    {selectedJob && <RunDetailsDialog job={selectedJob} cutId={cutId} stages={stages.filter(stage => stage.job_id === selectedJob.id)} onClose={() => setSelectedJob(null)} />}
  </section>
})
