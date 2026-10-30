import { useState } from 'react'
import { RefreshCw, Sparkles } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'

export const SUMMARY_TITLE = 'Cut summary'

export function CutSummaryPanel({ cutId, current, job, disabled, onStarted }: { cutId: string; current?: AnalysisResult; job?: Job; disabled: boolean; onStarted: () => Promise<void> }) {
  const [instruction, setInstruction] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const running = job?.pipeline === 'regen' && (job.status === 'queued' || job.status === 'running')
  async function start(mode: 'edit' | 'regenerate') {
    setWorking(true); setError('')
    try {
      await api(`/cuts/${cutId}/regenerate-summary`, { method: 'POST', body: JSON.stringify({ mode, instruction }) })
      if (mode === 'edit') setInstruction('')
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  const failure = job?.pipeline === 'regen' && job.status === 'failed' ? job.error : null
  return <section className="flex min-h-0 flex-1 flex-col gap-3 rounded-xl border hairline bg-[#1b202a] p-4" aria-label="Cut summary">
    <div><h3 className="flex items-center gap-2 text-xs font-semibold"><Sparkles size={14} className="text-[#bca7ff]" /> Cut summary</h3><p className="muted mt-2 text-[11px] leading-5">Rewrite the cut summary from available evidence and context. Frames are not analyzed again, and the previous version stays in history. Summaries use the same topics in a fixed order. Each topic has a qualitative confidence assessment, not a calibrated probability.</p></div>
    <div tabIndex={0} aria-label="Cut summary content" className="scrollbar min-h-40 flex-1 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[#333a45] bg-[#151a22] p-3 text-sm leading-6">{current ? current.content : <p className="muted text-xs">No cut summary yet. Regenerate builds one from the available results and context.</p>}</div>
    <label className="block"><span className="label">Instruction</span><textarea className="field min-h-16 resize-y" maxLength={5000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: emphasize the uncertain camera movement" /></label>
    {running && <p role="status" className="text-xs text-[#c3aeff]">{job?.stage || 'Waiting for the inference worker'}…</p>}
    {(error || failure) && <p role="alert" className="text-xs text-[#ff9fa7]">{error || failure}</p>}
    <div className="flex gap-2">
      <button className="btn btn-primary flex-1" disabled={working || disabled || running || !current || !instruction.trim()} onClick={() => start('edit')} title="Edit the current summary following your instruction"><Sparkles size={14} /> Edit current</button>
      <button className="btn btn-quiet flex-1" disabled={working || disabled || running} onClick={() => start('regenerate')} title="Write a new summary from the available results and context; your instruction is used as direction"><RefreshCw size={14} /> Regenerate</button>
    </div>
    <p className="muted text-[11px]">Sends the cut's results and selected context to the OpenRouter model chosen in Run settings.</p>
  </section>
}
