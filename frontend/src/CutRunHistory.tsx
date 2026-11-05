import { History } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, type Job } from './api'
import { cutRunOutcome } from './cutRunHistory'

const colors: Record<string, string> = {
  Succeeded: 'text-[#7bdcb2]', Failed: 'text-[#ff9fa7]', Cancelled: 'text-[#e6c58b]',
  Running: 'text-[#c3aeff]', Queued: 'text-[#c3aeff]', Waiting: 'text-[#c3aeff]',
}

export function CutRunHistory({ cutId, jobs }: { cutId: string; jobs: Job[] }) {
  const [stages, setStages] = useState<{ id: string; job_id: string; stage_key: string; label: string; status: string; elapsed_seconds: number; error: string | null }[]>([])
  useEffect(() => {
    let active = true
    const refresh = () => { void api<typeof stages>(`/cuts/${cutId}/analysis-stages`).then(data => { if (active) setStages(data) }).catch(() => {}) }
    setStages([]); refresh()
    const timer = setInterval(refresh, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [cutId])
  const runs = jobs.filter(job => ['analyze', 'regen', 'consistency_summary'].includes(job.pipeline) && job.cuts?.some(cut => cut.id === cutId))
  return <section aria-label="Cut run history" className="rounded-lg border hairline p-3 text-xs">
    <h3 className="flex items-center gap-2 font-semibold"><History size={14} className="text-[#bca7ff]" /> Run history <span className="muted font-normal">({runs.length})</span></h3>
    <p className="muted mt-2 leading-5">Shows the last recorded step and outcome for each run of this cut. Times are UTC.</p>
    {runs.length === 0 ? <p className="muted mt-3">No analysis runs recorded for this cut yet.</p> :
      <ol className="mt-3 space-y-2">{runs.map(job => {
        const outcome = cutRunOutcome(job, cutId)
        if (!outcome) return null
        const when = job.finished_at ? ['Run finished', job.finished_at] : job.started_at ? ['Run started', job.started_at] : ['Queued', job.created_at]
        return <li key={job.id} className="rounded-md border hairline bg-[#171c25] p-3">
          <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1"><strong>{job.pipeline === 'regen' ? 'Summary' : job.pipeline === 'consistency_summary' ? 'Consistency summary' : 'Video analysis'} · <span className={colors[outcome.status] || 'text-[#c8cfda]'}>{outcome.status}</span></strong><time className="muted" dateTime={`${when[1].replace(' ', 'T')}Z`}>{when[0]} {when[1]} UTC</time></div>
          <p className="mt-2 break-words text-[#d5dbe5]"><span className="muted">Step: </span>{outcome.step}</p>
          {stages.some(stage => stage.job_id === job.id) && <ol className="mt-2 space-y-1">{stages.filter(stage => stage.job_id === job.id).map(stage => <li key={stage.id} className="flex flex-wrap justify-between gap-2"><span>{stage.label} · {stage.status}</span><span className="muted tabular-nums">{Math.floor(stage.elapsed_seconds / 60)}m {Math.floor(stage.elapsed_seconds % 60)}s</span>{stage.error && <span className="w-full break-words text-[#ffb6bc]">{stage.error}</span>}</li>)}</ol>}
          {job.workflow === 'single_cut_v2' && <ul className="mt-2 space-y-1">{([['subtitle_ocr', 'Subtitle detection and OCR'], ['initial_frame', 'Initial frame'], ['video_analysis', 'Video analysis'], ['last_frame', 'Last frame']] as const).filter(([key]) => !stages.some(stage => stage.job_id === job.id && stage.stage_key === key)).map(([key, label]) => <li key={key} className="muted">{label} · {job.status === 'failed' ? 'Not started' : 'Waiting'}</li>)}</ul>}
          {outcome.error && <p className="mt-2 whitespace-pre-wrap break-words text-[#ffb6bc]">{outcome.error}</p>}
        </li>
      })}</ol>}
  </section>
}
