import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { BookOpen, Check, CircleCheck, Circle, Clock, Copy, Loader2, Pencil, RefreshCw, Save, Sparkles, TriangleAlert, Undo2, Wand2 } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'
import { SummaryRequestEditor } from './SummaryRequestEditor'
import { ResultMetrics } from './ResultMetrics'
import { ActionButton, StaleChip, StatusChip, WarningLine } from './RunUi'
import { useEpisodeRevision } from './episodeEvents'
import { Disclosure, StepPanel } from './ReviewUi'
import type { AnswersApi } from './ConsistencySummary'
import { Markdown } from './Markdown'
import { formatAgo } from './queueView'

export const SUMMARY_TITLE = 'Cut summary'

const VIEW_KEY = 'framewise.summaryView'
type SummaryView = 'read' | 'edit'
function storedView(): SummaryView | null {
  try { const value = localStorage.getItem(VIEW_KEY); return value === 'read' || value === 'edit' ? value : null } catch { return null }
}
function rememberView(value: SummaryView) {
  try { localStorage.setItem(VIEW_KEY, value) } catch { /* Private windows can block storage; the choice then lasts for this visit only. */ }
}
// Timeline lines start with "[0.4–2.7s]"; shown as code in the reading view so the times stand out.
const timelineMarks = (text: string) => text.replace(/^(\[\d+(?:\.\d+)?\s?[–-]\s?\d+(?:\.\d+)?s\])/gm, '`$1`')

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

/**
 * The Results workbench: the Review column handed in as `review`, then the answers and the Cut summary. All three share
 * this component's state because the summary is written from the saved answers.
 */
export function CutSummaryPanel({ cutId, review, editorSlot, current, loaded, jobs, disabled, onStarted, onSaved, textProvider }: { cutId: string; review: ReactNode | ((answers: AnswersApi) => ReactNode); editorSlot: ReactNode; current?: AnalysisResult; loaded: boolean; jobs: Job[]; disabled: boolean; textProvider?: 'local' | 'openrouter'; onStarted: () => Promise<void>; onSaved: (result: AnalysisResult) => void }) {
  const [instruction, setInstruction] = useState('')
  const [working, setWorking] = useState(false)
  const [savingInstructions, setSavingInstructions] = useState(false)
  const [instructionsError, setInstructionsError] = useState('')
  const [humanInstructions, setHumanInstructions] = useState('')
  const [savedInstructions, setSavedInstructions] = useState('')
  const [savedUpdatedAt, setSavedUpdatedAt] = useState<string | undefined>(undefined)
  const [instructionsLoaded, setInstructionsLoaded] = useState(false)
  const [previousAnswers, setPreviousAnswers] = useState<string | null>(null)
  const [reviewError, setReviewError] = useState('')
  const [reviewWorking, setReviewWorking] = useState(false)
  const [reviewDraft, setReviewDraft] = useState<{ job_id: string | null; content: string; stale_reason?: string | null } | null>(null)
  const reviewRevision = useEpisodeRevision(['data', 'stale'])
  useEffect(() => { setReviewDraft(null); setPreviousAnswers(null); setReviewError('') }, [cutId])
  useEffect(() => {
    let active = true
    setInstructionsLoaded(false)
    api<{ content: string; updated_at?: string }>(`/cuts/${cutId}/brief-instructions`).then(value => { if (active) { setHumanInstructions(value.content); setSavedInstructions(value.content); setSavedUpdatedAt(value.updated_at); setInstructionsLoaded(true) } }).catch(err => { if (active) setError((err as Error).message) })
    return () => { active = false }
  }, [cutId])
  const savedRef = useRef(savedInstructions)
  savedRef.current = savedInstructions
  useEffect(() => {
    const reload = () => {
      api<{ content: string; updated_at?: string }>(`/cuts/${cutId}/brief-instructions`).then(value => {
        const previous = savedRef.current
        setSavedInstructions(value.content)
        setSavedUpdatedAt(value.updated_at)
        setHumanInstructions(current => current === previous ? value.content : current)
      }).catch(() => {})
    }
    window.addEventListener('framewise:brief-instructions-changed', reload)
    return () => window.removeEventListener('framewise:brief-instructions-changed', reload)
  }, [cutId])
  async function saveInstructions() {
    setSavingInstructions(true); setInstructionsError('')
    try { const value = await api<{ content: string; updated_at?: string }>(`/cuts/${cutId}/brief-instructions`, { method: 'PUT', body: JSON.stringify({ content: humanInstructions }) }); setSavedInstructions(value.content); setSavedUpdatedAt(value.updated_at); window.dispatchEvent(new Event('framewise:cut-timeline-changed')) }
    catch (err) { setInstructionsError((err as Error).message) } finally { setSavingInstructions(false) }
  }
  const [error, setError] = useState('')
  const [saveError, setSaveError] = useState('')
  const [draft, setDraft] = useState({ id: current?.id, original: visibleSummary(current), content: visibleSummary(current) })
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  useEffect(() => {
    setDraft(previous => previous.content === previous.original ? { id: current?.id, original: visibleSummary(current), content: visibleSummary(current) } : previous)
  }, [current?.id, current?.content])
  useEffect(() => { setCopyStatus('idle') }, [draft.content])
  const changedElsewhere = draft.id !== current?.id
  async function saveSummary() {
    if (!draft.content.trim() || changedElsewhere) return
    setWorking(true); setSaveError('')
    try {
      const result = draft.id
        ? await api<AnalysisResult>(`/analysis-results/${draft.id}`, { method: 'PUT', body: JSON.stringify({ title: SUMMARY_TITLE, content: draft.content }) })
        : await api<AnalysisResult>(`/cuts/${cutId}/analysis-results`, { method: 'POST', body: JSON.stringify({ kind: 'cut_description', title: SUMMARY_TITLE, content: draft.content }) })
      setDraft({ id: result.id, original: result.content, content: result.content })
      onSaved(result)
    } catch (err) { setSaveError((err as Error).message) } finally { setWorking(false) }
  }
  const reviewJobs = jobs.filter(job => job.pipeline === 'review_answers' && job.cuts.some(cut => cut.id === cutId))
  const activeReview = reviewJobs.find(job => job.status === 'queued' || job.status === 'running')
  const latestReview = reviewJobs[0]
  const reviewFailure = !activeReview && latestReview?.status === 'failed' ? latestReview.error : null
  useEffect(() => {
    if (!instructionsLoaded) return
    let active = true
    Promise.all([api<{ job_id: string | null; content: string; stale_reason?: string | null }>(`/cuts/${cutId}/review-answers`),
      api<{ content: string; updated_at?: string }>(`/cuts/${cutId}/brief-instructions`)]).then(([draft, saved]) => {
      if (!active) return
      setReviewDraft(draft)
      const previous = savedRef.current
      setSavedInstructions(saved.content)
      setSavedUpdatedAt(saved.updated_at)
      setHumanInstructions(current => current === previous ? saved.content : current)
    }).catch(err => { if (active) setReviewError((err as Error).message) })
    return () => { active = false }
  }, [reviewRevision, instructionsLoaded, cutId])
  async function generateAnswers() {
    setReviewWorking(true); setReviewError('')
    try { await api(`/cuts/${cutId}/review-answers`, { method: 'POST' }); await onStarted() }
    catch (err) { setReviewError((err as Error).message) } finally { setReviewWorking(false) }
  }
  async function cancelReview() {
    if (!activeReview) return
    setReviewWorking(true); setReviewError('')
    try { await api(`/jobs/${activeReview.id}/cancel`, { method: 'POST' }); await onStarted() }
    catch (err) { setReviewError((err as Error).message) } finally { setReviewWorking(false) }
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
  const failure = !activeJob && latestJob?.status === 'failed' ? latestJob.error : null
  const answersDirty = humanInstructions !== savedInstructions
  const summaryDirty = draft.content !== draft.original
  const [view, setView] = useState<SummaryView>(() => storedView() ?? 'read')
  function chooseView(next: SummaryView) { setView(next); rememberView(next) }
  // The choice is a preference; an empty summary can only be edited.
  const reading = view === 'read' && draft.content.trim() !== ''
  const readable = useMemo(() => timelineMarks(draft.content), [draft.content])

  const answersChip = !instructionsLoaded ? null
    : previousAnswers !== null && answersDirty ? <StatusChip tone="purple" icon={<Wand2 size={11} aria-hidden />} title="The draft replaced your previous text. Check it, edit it, then save.">AI draft · review and save</StatusChip>
      : answersDirty ? <StatusChip tone="amber" icon={<Circle size={11} aria-hidden />} title="The summary is written from the saved answers">Unsaved changes</StatusChip>
        : humanInstructions.trim() ? <StatusChip tone="green" icon={<CircleCheck size={11} aria-hidden />}>Saved</StatusChip>
          : <StatusChip tone="grey" icon={<Circle size={11} aria-hidden />}>No answers yet</StatusChip>
  const summaryChip = activeJob ? <StatusChip tone="purple" icon={activeJob.status === 'queued' ? <Clock size={11} aria-hidden /> : <Loader2 size={11} className="animate-spin" aria-hidden />}>{activeJob.status === 'queued' ? 'Queued' : 'Generating…'}</StatusChip>
    : summaryDirty ? <StatusChip tone="amber" icon={<Circle size={11} aria-hidden />}>Unsaved edits</StatusChip>
      : current?.stale_reason ? <StaleChip reason={current.stale_reason} />
        : current ? <StatusChip tone="green" icon={<CircleCheck size={11} aria-hidden />} title={`Source: ${current.origin}${current.model_id ? ` · ${current.model_id}` : ''}`}>Saved</StatusChip>
        : <StatusChip tone="grey" icon={<Circle size={11} aria-hidden />}>Not generated yet</StatusChip>

  // Same conditions as before, now with the reason in words instead of a silently disabled button.
  const queueBlocked = working || savingInstructions || disabled || !instructionsLoaded || answersDirty
  const blockedReason = answersDirty ? 'Save your answers first. The summary is written from the saved version.'
    : disabled ? 'Analysis is off for this cut, or the cut is archived.' : ''
  const editHint = !queueBlocked && (!current ? 'Queue edit needs a saved summary to change.' : !instruction.trim() ? 'Add an instruction to use Queue edit.' : '')

  return <div role="group" aria-label="Review, answers and Cut summary" className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)] gap-4 pt-3 lg:grid-cols-2 lg:grid-rows-[minmax(0,1fr)] 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.1fr)]">
    <div className="grid min-h-0 grid-rows-[minmax(0,1fr)] lg:col-start-1 lg:row-start-1">{typeof review === 'function' ? review({ text: humanInstructions, onChange: setHumanInstructions, ready: instructionsLoaded && !savingInstructions }) : review}</div>
    <div className="scrollbar flex flex-col gap-4 lg:col-start-2 lg:row-start-1 lg:-mt-3 lg:min-h-0 lg:overflow-y-auto lg:pt-3 2xl:contents">
      <StepPanel className="2xl:col-start-2 2xl:row-start-1" step={2} kind="review_answers" title="Your answers" subtitle="Decisions and checks" updated={savedUpdatedAt ? `last saved ${formatAgo(savedUpdatedAt)}` : undefined} tag={answersChip}
        actions={<ActionButton compact icon={<Wand2 size={13} aria-hidden />} pending={reviewWorking} disabled={!instructionsLoaded || disabled} onClick={generateAnswers} title="Draft answers from the current review and transcript">Draft with AI</ActionButton>}>
        <div className="flex min-h-[15rem] flex-1 flex-col gap-2">
          <label htmlFor={`answers-${cutId}`} className="label !mb-0">Answers to suggested checks and production decisions</label>
          <textarea id={`answers-${cutId}`} aria-label="Answers to suggested checks and production decisions" className="editor-area min-h-[12rem] flex-1" maxLength={50000} value={humanInstructions} disabled={!instructionsLoaded || savingInstructions} onChange={event => setHumanInstructions(event.target.value)} placeholder="Record confirmations, corrections, and production decisions for this cut. One bullet per item in the Review column works well." />
          {activeReview && <div className="flex items-center justify-between gap-2"><p role="status" className="text-[11px] text-[#c3aeff]">{activeReview.status === 'queued' ? 'Review answers queued' : activeReview.stage || 'Drafting review answers'}…</p><ActionButton compact disabled={reviewWorking} onClick={cancelReview}>Cancel</ActionButton></div>}
          {reviewDraft?.content && reviewDraft.content !== humanInstructions && reviewDraft.content !== savedInstructions && <div className="flex items-center gap-2">
            <ActionButton compact icon={<Wand2 size={13} aria-hidden />} onClick={() => { setPreviousAnswers(humanInstructions); setHumanInstructions(reviewDraft.content) }}>Use AI draft</ActionButton>
            {reviewDraft.stale_reason && <StaleChip reason={reviewDraft.stale_reason} />}
          </div>}
          {previousAnswers !== null && <button type="button" className="inline-flex items-center gap-1 self-start text-[11px] text-[#c3aeff] underline" onClick={() => { setHumanInstructions(previousAnswers); setPreviousAnswers(null) }}><Undo2 size={12} aria-hidden /> Restore previous answers</button>}
          {(reviewError || reviewFailure) && <p role="alert" className="text-xs text-[#ff9fa7]">{reviewError || reviewFailure}</p>}
          {instructionsError && <p role="alert" className="text-xs text-[#ff9fa7]">{instructionsError}</p>}
          <ActionButton variant={answersDirty ? 'primary' : 'quiet'} icon={<Save size={14} aria-hidden />} pending={savingInstructions} disabled={!instructionsLoaded || !answersDirty} onClick={saveInstructions} className="w-full">{savingInstructions ? 'Saving answers…' : 'Save answers'}</ActionButton>
        </div>
        <div className="shrink-0 space-y-2.5 rounded-lg border hairline bg-[#171c25] p-3">
          <h4 className="flex items-center gap-1.5 text-xs font-semibold"><Sparkles size={14} className="text-[#e5d25a]" aria-hidden /> Generate the Cut summary</h4>
          <label className="block"><span className="label !mb-1.5">Instruction · optional</span><textarea className="field min-h-16 resize-y !text-[13px]" maxLength={5000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: emphasize the uncertain camera movement" /></label>
          {activeJob && <div className="flex flex-wrap items-center justify-between gap-2"><p role="status" className="text-xs text-[#c3aeff]">{activeJob.status === 'queued' ? 'Summary queued for this cut' : activeJob.stage || 'Generating summary'}…</p><ActionButton compact disabled={working} onClick={cancelSummary}>Cancel summary</ActionButton></div>}
          {(error || failure) && <p role="alert" className="text-xs text-[#ff9fa7]">{error || failure}</p>}
          <div className="flex gap-2">
            <ActionButton variant="primary" icon={<RefreshCw size={14} aria-hidden />} className="flex-1" disabled={queueBlocked} onClick={() => start('regenerate')} title="Write a new summary from the available results and context; your instruction is used as direction">Queue summary</ActionButton>
            <ActionButton icon={<Sparkles size={14} aria-hidden />} className="flex-1" disabled={queueBlocked || !current || !instruction.trim()} onClick={() => start('edit')} title="Edit the current summary following your instruction">Queue edit</ActionButton>
          </div>
          {blockedReason ? <WarningLine level="warn">{blockedReason}</WarningLine> : editHint ? <WarningLine level="info">{editHint}</WarningLine> : null}
          <p className="muted text-[11px]">Uses the text engine selected in the top navigation.</p>
        </div>
        <div className="shrink-0"><SummaryRequestEditor textProvider={textProvider} cutId={cutId} disabled={disabled} onStarted={onStarted} /></div>
      </StepPanel>

      <StepPanel className="2xl:col-start-3 2xl:row-start-1" step={3} kind="cut_summary" title="Cut summary" subtitle="Final production brief" updated={current?.created_at ? formatAgo(current.created_at) : undefined} tag={summaryChip}
        actions={<>
          <div role="group" aria-label="Summary view" className="flex rounded-lg border hairline p-0.5 text-[11px] font-semibold">
            <button type="button" aria-pressed={reading} disabled={!draft.content.trim()} onClick={() => chooseView('read')} title={draft.content.trim() ? 'Read the summary formatted' : 'Nothing to read yet'} className={`inline-flex items-center gap-1 rounded-md px-2 py-1 ${reading ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}><BookOpen size={12} aria-hidden /> Read</button>
            <button type="button" aria-pressed={!reading} onClick={() => chooseView('edit')} title="Edit the summary text" className={`inline-flex items-center gap-1 rounded-md px-2 py-1 ${!reading ? 'bg-[#322a48] text-white' : 'text-[#aeb5c1] hover:text-white'}`}><Pencil size={12} aria-hidden /> Edit</button>
          </div>
          <ActionButton compact icon={copyStatus === 'copied' ? <Check size={14} className="text-[#7bdcb2]" aria-hidden /> : <Copy size={14} aria-hidden />} disabled={!draft.content} onClick={copySummary} title={copyStatus === 'copied' ? 'Copied' : 'Copy cut summary'} aria-label="Copy cut summary"><span role="status" className="sr-only">{copyStatus === 'copied' ? 'Copied' : ''}</span></ActionButton>
        </>}
        footer={<ActionButton variant="primary" icon={<Save size={14} aria-hidden />} className="w-full" disabled={!loaded || working || changedElsewhere || !draft.content.trim() || !summaryDirty} onClick={saveSummary}>Save summary</ActionButton>}>
        {changedElsewhere && <div className="flex items-start gap-2 rounded-lg border border-[#6b5a2f] bg-[#2a2314] px-3 py-2 text-xs text-[#f0d9a5]" role="alert"><TriangleAlert size={14} className="mt-0.5 shrink-0" aria-hidden /><span>The saved summary changed while you were editing. Copy your draft before loading the latest version. <button type="button" className="underline" onClick={() => setDraft({ id: current?.id, original: visibleSummary(current), content: visibleSummary(current) })}>Load latest</button></span></div>}
        {copyStatus === 'failed' && <p role="status" className="text-xs text-[#ff9fa7]">Could not copy; select the text instead.</p>}
        {reading
          ? <div className="scrollbar min-h-[24rem] flex-1 rounded-lg border border-[#303640] bg-[#11151d] p-4 lg:min-h-0 lg:overflow-y-auto 2xl:min-h-0" aria-label="Cut summary (formatted)"><Markdown className="md-summary" text={readable} /></div>
          : <textarea aria-label="Cut summary content" className="editor-area min-h-[24rem] flex-1 lg:min-h-0 2xl:min-h-0" maxLength={50000} value={draft.content} disabled={!loaded || working} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Write a cut summary, or queue one from the available results and context." />}
        {current && <Disclosure label="Generation details"><ResultMetrics result={current} /></Disclosure>}
        {saveError && <p role="alert" className="text-xs text-[#ff9fa7]">{saveError}</p>}
      </StepPanel>
    </div>
    {editorSlot}
  </div>
}
