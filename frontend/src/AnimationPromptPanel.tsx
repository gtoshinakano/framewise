import { useEffect, useState } from 'react'
import { Check, Copy, RefreshCw, Save, Sparkles } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'
import { SummaryRequestEditor } from './SummaryRequestEditor'

export const SUMMARY_TITLE = 'Cut summary'

export function CutSummaryPanel({ cutId, current, loaded, jobs, disabled, onStarted, onSaved }: { cutId: string; current?: AnalysisResult; loaded: boolean; jobs: Job[]; disabled: boolean; onStarted: () => Promise<void>; onSaved: (result: AnalysisResult) => void }) {
  const [instruction, setInstruction] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [draft, setDraft] = useState({ id: current?.id, original: current?.content || '', content: current?.content || '' })
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  useEffect(() => {
    setDraft(previous => previous.content === previous.original ? { id: current?.id, original: current?.content || '', content: current?.content || '' } : previous)
  }, [current?.id, current?.content])
  useEffect(() => { setCopyStatus('idle') }, [draft.content])
  const changedElsewhere = draft.id !== current?.id
  async function saveSummary() {
    if (!draft.content.trim() || changedElsewhere) return
    setWorking(true); setError('')
    try {
      const result = draft.id
        ? await api<AnalysisResult>(`/analysis-results/${draft.id}`, { method: 'PUT', body: JSON.stringify({ title: SUMMARY_TITLE, content: draft.content }) })
        : await api<AnalysisResult>(`/cuts/${cutId}/analysis-results`, { method: 'POST', body: JSON.stringify({ kind: 'cut_description', title: SUMMARY_TITLE, content: draft.content }) })
      setDraft({ id: result.id, original: result.content, content: result.content })
      onSaved(result)
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  const cutJobs = jobs.filter(job => job.pipeline === 'regen' && job.cuts.some(cut => cut.id === cutId))
  const activeJob = cutJobs.find(job => job.status === 'queued' || job.status === 'running')
  const latestJob = cutJobs[0]
  const sourceJob = jobs.find(job => job.id === current?.job_id)
  const recordedSeconds = current?.settings.summary_generation_seconds
  const jobSeconds = sourceJob?.pipeline === 'regen' && sourceJob.started_at && sourceJob.finished_at ? (Date.parse(sourceJob.finished_at.replace(' ', 'T') + 'Z') - Date.parse(sourceJob.started_at.replace(' ', 'T') + 'Z')) / 1000 : null
  const generationSeconds = typeof recordedSeconds === 'number' && Number.isFinite(recordedSeconds) && recordedSeconds >= 0 ? recordedSeconds : jobSeconds != null && Number.isFinite(jobSeconds) && jobSeconds >= 0 ? jobSeconds : null
  const timeLabel = generationSeconds == null ? 'Unavailable for this result' : `${generationSeconds.toFixed(1)}s${typeof recordedSeconds === 'number' ? '' : ' (total job time)'}`
  async function start(mode: 'edit' | 'regenerate') {
    setWorking(true); setError('')
    try {
      await api(`/cuts/${cutId}/regenerate-summary`, { method: 'POST', body: JSON.stringify({ mode, instruction }) })
      if (mode === 'edit') setInstruction('')
      await onStarted()
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function cancelSummary() {
    if (!activeJob) return
    setWorking(true); setError('')
    try { await api(`/jobs/${activeJob.id}/cancel`, { method: 'POST' }); await onStarted() }
    catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function copySummary() {
    if (!draft.content) return
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable')
      await navigator.clipboard.writeText(draft.content)
      setCopyStatus('copied')
      return
    } catch { /* The app may be served over HTTP, where the Clipboard API is unavailable. */ }
    const previousFocus = document.activeElement as HTMLElement | null
    const field = document.createElement('textarea')
    field.value = draft.content
    field.setAttribute('readonly', '')
    field.style.position = 'fixed'; field.style.left = '-9999px'
    document.body.appendChild(field)
    try {
      field.focus(); field.select()
      setCopyStatus(document.execCommand('copy') ? 'copied' : 'failed')
    } catch { setCopyStatus('failed') }
    finally { field.remove(); previousFocus?.focus() }
  }
  const failure = !activeJob && latestJob?.status === 'failed' && latestJob.stage !== 'Cancelled' ? latestJob.error : null
  return <section className="flex min-h-0 flex-1 flex-col gap-3 rounded-xl border hairline bg-[#1b202a] p-4" aria-label="Cut summary">
    <div><div className="flex items-center justify-between gap-2"><h3 className="flex items-center gap-2 text-xs font-semibold"><Sparkles size={14} className="text-[#bca7ff]" /> Cut summary</h3><div className="flex items-center gap-2"><span role="status" className={`text-[11px] ${copyStatus === 'failed' ? 'text-[#ff9fa7]' : 'text-[#7bdcb2]'}`}>{copyStatus === 'copied' ? 'Copied' : copyStatus === 'failed' ? 'Could not copy; select the text instead.' : ''}</span><button type="button" disabled={!draft.content} onClick={copySummary} title="Copy cut summary" aria-label="Copy cut summary" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746] hover:text-white">{copyStatus === 'copied' ? <Check size={16} /> : <Copy size={16} />}</button></div></div><p className="muted mt-2 text-[11px] leading-5">Edit and save the summary directly, or queue a rewrite using the available evidence and context. Previous versions stay in history. Generated summaries can include transcript segments grouped by speaker with qualitative confidence; run audio transcription first to supply speech evidence.</p></div>
    <textarea aria-label="Cut summary content" className="scrollbar min-h-40 flex-1 resize-y overflow-y-auto rounded-lg border border-[#333a45] bg-[#151a22] p-3 text-sm leading-6" maxLength={50000} value={draft.content} disabled={!loaded || working} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Write a cut summary, or queue one from the available results and context." />
    {changedElsewhere && <div className="text-xs text-[#ffc980]" role="alert">The saved summary changed while you were editing. Copy your draft before loading the latest version. <button type="button" className="underline" onClick={() => setDraft({ id: current?.id, original: current?.content || '', content: current?.content || '' })}>Load latest</button></div>}
    <button type="button" className="btn btn-primary" disabled={!loaded || working || changedElsewhere || !draft.content.trim() || draft.content === draft.original} onClick={saveSummary}><Save size={14} /> Save summary</button>
    {current && <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[#aeb5c1]" aria-label="Summary generation details"><span>Model: <strong className="text-[#e8ebf1]">{current.model_id || (current.origin === 'manual_edit' ? 'Manually edited' : 'Unavailable')}</strong></span><span title="Time spent generating the summary, excluding time waiting in the queue and frame extraction. Initial analyses include the evidence assessment step.">Generation time: <strong className="text-[#e8ebf1]">{timeLabel}</strong></span></div>}
    <label className="block"><span className="label">Instruction</span><textarea className="field min-h-16 resize-y" maxLength={5000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: emphasize the uncertain camera movement" /></label>
    {activeJob && <div className="flex flex-wrap items-center justify-between gap-2"><p role="status" className="text-xs text-[#c3aeff]">{activeJob.status === 'queued' ? 'Summary queued for this cut' : activeJob.stage || 'Generating summary'}…</p><button type="button" className="btn btn-quiet px-3 py-2" disabled={working} onClick={cancelSummary}>Cancel summary</button></div>}
    {(error || failure) && <p role="alert" className="text-xs text-[#ff9fa7]">{error || failure}</p>}
    <div className="flex gap-2">
      <button className="btn btn-primary flex-1" disabled={working || disabled || !!activeJob || !current || !instruction.trim()} onClick={() => start('edit')} title="Edit the current summary following your instruction"><Sparkles size={14} /> Queue edit</button>
      <button className="btn btn-quiet flex-1" disabled={working || disabled || !!activeJob} onClick={() => start('regenerate')} title="Write a new summary from the available results and context; your instruction is used as direction"><RefreshCw size={14} /> Queue summary</button>
    </div>
    <p className="muted text-[11px]">Sends the cut's results and selected context to the OpenRouter model chosen in Run settings.</p>
    <SummaryRequestEditor cutId={cutId} disabled={disabled || !!activeJob} onStarted={onStarted} />
  </section>
}
