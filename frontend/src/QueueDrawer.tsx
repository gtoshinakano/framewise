import { useEffect, useRef } from 'react'
import { Ban, ListChecks, X } from 'lucide-react'
import type { Cut, Job } from './api'
import { AnalysisHistory } from './AnalysisHistory'

/**
 * A slide-over that keeps "Runs, queue & history" one click away from anywhere in the editor.
 * The history list itself is the unchanged AnalysisHistory component.
 */
export function QueueDrawer({ jobs, cuts, onClose, onSelectCut, onChanged, onCancelJob, onCancelAllQueued, cancelling }: {
  jobs: Job[]
  cuts: Cut[]
  onClose: () => void
  onSelectCut: (cut: Cut) => void
  onChanged: () => Promise<void>
  onCancelJob: (jobId: string) => void
  onCancelAllQueued: () => void
  cancelling: boolean
}) {
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', key)
    return () => { window.removeEventListener('keydown', key); opener?.focus() }
  }, [onClose])
  const waitingAudio = jobs.filter(job => ['transcribe', 'voice_sample'].includes(job.pipeline) && job.status === 'queued')
  const visual = jobs.filter(job => job.pipeline === 'analyze' && (job.status === 'queued' || job.status === 'running'))
  const queued = jobs.filter(job => job.status === 'queued').length
  return <>
    <div className="drawer-backdrop" onMouseDown={onClose} aria-hidden />
    <div ref={panel} role="dialog" aria-modal="true" aria-labelledby="queue-drawer-title" tabIndex={-1} className="drawer outline-none">
      <div className="flex items-center justify-between gap-3 border-b hairline px-4 py-3">
        <h2 id="queue-drawer-title" className="flex items-center gap-2 text-sm font-semibold"><ListChecks size={16} className="text-[#bca7ff]" aria-hidden /> Runs, queue &amp; history</h2>
        <div className="flex items-center gap-2">
          <button type="button" className="icon-btn icon-btn-md icon-btn-text tone-danger" disabled={cancelling} onClick={onCancelAllQueued} title="Cancel all queued jobs across all episodes; running jobs continue"><Ban size={15} aria-hidden /> Cancel all queued{queued ? ` (${queued} here)` : ''}</button>
          <button type="button" className="icon-btn icon-btn-md ghost" onClick={onClose} aria-label="Close runs, queue and history"><X size={17} /></button>
        </div>
      </div>
      <div className="scrollbar min-h-0 flex-1 overflow-y-auto py-4">
        {[...waitingAudio.map(job => ({ job, text: 'Waiting for the audio worker.', action: 'Cancel queued job' })),
          ...visual.map(job => ({ job, text: `Visual analysis ${job.status}${job.stage ? `: ${job.stage}` : ''}`, action: 'Cancel run' }))].map(({ job, text, action }) =>
          <div key={job.id} className="mx-4 mb-3 flex items-center justify-between gap-3 rounded-lg border hairline p-3 text-xs"><span>{text}</span><button className="btn btn-quiet !px-3 !py-1.5 !text-xs" onClick={() => onCancelJob(job.id)}>{action}</button></div>)}
        <AnalysisHistory jobs={jobs} cuts={cuts} onSelectCut={cut => { onSelectCut(cut); onClose() }} onChanged={onChanged} defaultOpen className="mx-4" />
      </div>
    </div>
  </>
}
