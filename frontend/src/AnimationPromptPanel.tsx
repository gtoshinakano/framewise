import { useEffect, useState } from 'react'
import { Check, Copy, RefreshCw, Save, Sparkles } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'
import { SummaryRequestEditor } from './SummaryRequestEditor'
import { ResultMetrics } from './ResultMetrics'

export const SUMMARY_TITLE = 'Cut summary'

const summaryHeadings = new Set(['overview', 'opening composition', 'main characters', 'secondary characters', 'character outfits', 'character actions', 'camera position', 'camera movement', 'setting and scenario', 'lighting', 'on-screen text', 'audio and voices', 'spoken dialogue by speaker', 'suggested animation direction', 'evidence and uncertainties'])

function visibleSummary(result?: AnalysisResult) {
  const text = result?.content || ''
  if (result?.origin !== 'model') return text
  let skipping = false
  return text.split('\n').filter(line => {
    const heading = line.trim().replace(/^#{1,6}\s*/, '').split(' · Confidence:')[0].replace(/:$/, '').trim().toLowerCase()
    if (heading === 'suggested composition frame') { skipping = true; return false }
    if (skipping && (summaryHeadings.has(heading) || /^\s*#{1,6}\s+/.test(line))) skipping = false
    return !skipping
  }).join('\n').trim()
}

export function CutSummaryPanel({ cutId, current, loaded, jobs, disabled, onStarted, onSaved, textProvider }: { cutId: string; current?: AnalysisResult; loaded: boolean; jobs: Job[]; disabled: boolean; textProvider?: 'local' | 'openrouter'; onStarted: () => Promise<void>; onSaved: (result: AnalysisResult) => void }) {
  const [instruction, setInstruction] = useState('')
  const [working, setWorking] = useState(false)
  const [savingInstructions, setSavingInstructions] = useState(false)
  const [instructionsError, setInstructionsError] = useState('')
  const [humanInstructions, setHumanInstructions] = useState('')
  const [savedInstructions, setSavedInstructions] = useState('')
  const [instructionsLoaded, setInstructionsLoaded] = useState(false)
  useEffect(() => {
    let active = true
    setInstructionsLoaded(false)
    api<{ content: string }>(`/cuts/${cutId}/brief-instructions`).then(value => { if (active) { setHumanInstructions(value.content); setSavedInstructions(value.content); setInstructionsLoaded(true) } }).catch(err => { if (active) setError((err as Error).message) })
    return () => { active = false }
  }, [cutId])
  async function saveInstructions() {
    setSavingInstructions(true); setInstructionsError('')
    try { const value = await api<{ content: string }>(`/cuts/${cutId}/brief-instructions`, { method: 'PUT', body: JSON.stringify({ content: humanInstructions }) }); setSavedInstructions(value.content) }
    catch (err) { setInstructionsError((err as Error).message) } finally { setSavingInstructions(false) }
  }
  const [error, setError] = useState('')
  const [draft, setDraft] = useState({ id: current?.id, original: visibleSummary(current), content: visibleSummary(current) })
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  useEffect(() => {
    setDraft(previous => previous.content === previous.original ? { id: current?.id, original: visibleSummary(current), content: visibleSummary(current) } : previous)
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
    <div><div className="flex items-center justify-between gap-2"><h3 className="flex items-center gap-2 text-xs font-semibold"><Sparkles size={14} className="text-[#bca7ff]" /> Cut summary</h3><div className="flex items-center gap-2"><span role="status" className={`text-[11px] ${copyStatus === 'failed' ? 'text-[#ff9fa7]' : 'text-[#7bdcb2]'}`}>{copyStatus === 'copied' ? 'Copied' : copyStatus === 'failed' ? 'Could not copy; select the text instead.' : ''}</span><button type="button" disabled={!draft.content} onClick={copySummary} title="Copy cut summary" aria-label="Copy cut summary" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746] hover:text-white">{copyStatus === 'copied' ? <Check size={16} /> : <Copy size={16} />}</button></div></div><p className="muted mt-2 text-[11px] leading-5">Write a production brief or generate one from available evidence. Save answers to suggested checks and production decisions below before generating. Previous versions stay in history.</p></div>
    <textarea aria-label="Cut summary content" className="scrollbar min-h-40 flex-1 resize-y overflow-y-auto rounded-lg border border-[#333a45] bg-[#151a22] p-3 text-sm leading-6" maxLength={50000} value={draft.content} disabled={!loaded || working} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Write a cut summary, or queue one from the available results and context." />
    {changedElsewhere && <div className="text-xs text-[#ffc980]" role="alert">The saved summary changed while you were editing. Copy your draft before loading the latest version. <button type="button" className="underline" onClick={() => setDraft({ id: current?.id, original: visibleSummary(current), content: visibleSummary(current) })}>Load latest</button></div>}
    <button type="button" className="btn btn-primary" disabled={!loaded || working || changedElsewhere || !draft.content.trim() || draft.content === draft.original} onClick={saveSummary}><Save size={14} /> Save summary</button>
    {current && <ResultMetrics result={current} />}
    <label className="block"><span className="label">Answers to suggested checks and production decisions</span><textarea aria-label="Answers to suggested checks and production decisions" className="field min-h-24 resize-y" maxLength={50000} value={humanInstructions} disabled={!instructionsLoaded || savingInstructions} onChange={event => setHumanInstructions(event.target.value)} placeholder="Record confirmations, corrections, and production decisions for this cut." /></label>
    <button type="button" className="btn btn-quiet" disabled={!instructionsLoaded || savingInstructions || humanInstructions === savedInstructions} onClick={saveInstructions}><Save size={14} /> {savingInstructions ? 'Saving answers…' : 'Save answers'}</button>
    {instructionsError && <p role="alert" className="text-xs text-[#ff9fa7]">{instructionsError}</p>}
    {activeJob && <p className="muted text-[11px]">Saved answers will be used the next time you queue a summary. The queued summary keeps its original answers.</p>}
    <label className="block"><span className="label">Instruction</span><textarea className="field min-h-16 resize-y" maxLength={5000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: emphasize the uncertain camera movement" /></label>
    {activeJob && <div className="flex flex-wrap items-center justify-between gap-2"><p role="status" className="text-xs text-[#c3aeff]">{activeJob.status === 'queued' ? 'Summary queued for this cut' : activeJob.stage || 'Generating summary'}…</p><button type="button" className="btn btn-quiet px-3 py-2" disabled={working} onClick={cancelSummary}>Cancel summary</button></div>}
    {(error || failure) && <p role="alert" className="text-xs text-[#ff9fa7]">{error || failure}</p>}
    <div className="flex gap-2">
      <button className="btn btn-primary flex-1" disabled={working || savingInstructions || disabled || !!activeJob || !instructionsLoaded || humanInstructions !== savedInstructions || !current || !instruction.trim()} onClick={() => start('edit')} title="Edit the current summary following your instruction"><Sparkles size={14} /> Queue edit</button>
      <button className="btn btn-quiet flex-1" disabled={working || savingInstructions || disabled || !!activeJob || !instructionsLoaded || humanInstructions !== savedInstructions} onClick={() => start('regenerate')} title="Write a new summary from the available results and context; your instruction is used as direction"><RefreshCw size={14} /> Queue summary</button>
    </div>
    <p className="muted text-[11px]">Uses the text engine selected in the top navigation.</p>
    <SummaryRequestEditor textProvider={textProvider} cutId={cutId} disabled={disabled || !!activeJob} onStarted={onStarted} />
  </section>
}
