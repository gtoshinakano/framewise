import { Fragment, useMemo, useState, type ReactNode } from 'react'
import { Bot, Check, CircleCheck, Eye, EyeOff, FileText, ListChecks, RefreshCw, ScanSearch, ShieldCheck, TriangleAlert, UserRoundCheck, X } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'
import { ResultMetrics } from './ResultMetrics'
import { ActionButton, Skeleton, StatusChip, WarningLine } from './RunUi'
import { Disclosure, StepPanel } from './ReviewUi'
import { answerLabel, parseConsistency, selection, setAnswer, timeParts, topicLabel, type ConsistencyKey, type ConsistencySection, type Question, type VisualCheck } from './consistencyView'
import { formatAgo } from './queueView'
import { DecisionStrip } from './AgentNotes'
import { answerFor } from './cutNeedsView'
import type { Decision } from './inboxView'

/** What the agents did with this cut's questions: their applied answers, and which questions wait for Claude. */
export type AgentMarks = { decisions: Decision[]; claudeQuestions: ReadonlySet<string>; onChanged: () => void }

// What the user works from comes first. Each field keeps its stored name's wording and gets its own colour and icon,
// matching the tones used on the Run settings tab; the label and icon always carry the meaning, never colour alone.
const ORDER: ConsistencyKey[] = ['human_review', 'recommended_checks', 'confusing', 'clear', 'summary']
// A result with questions has no Human review or Suggested checks text: the questions are what the person decides.
const QUESTION_ORDER: ConsistencyKey[] = ['confusing', 'clear', 'summary']
const LOOK: Record<ConsistencyKey, { label: string; none: string; icon: typeof FileText; accent: string; tint: string; ink: string }> = {
  human_review: { label: 'Human review', none: 'Nothing to decide', icon: UserRoundCheck, accent: '#f0a64a', tint: '#2a2314', ink: '#f0d9a5' },
  recommended_checks: { label: 'Suggested checks', none: 'No checks needed', icon: ListChecks, accent: '#6aa8ff', tint: '#141f33', ink: '#b4d2ff' },
  confusing: { label: 'Conflicting or unclear', none: 'No conflicts identified', icon: TriangleAlert, accent: '#ff8a94', tint: '#2b191f', ink: '#ffc0c6' },
  clear: { label: 'Clear evidence', none: 'Nothing recorded', icon: CircleCheck, accent: '#70d5aa', tint: '#12261f', ink: '#a4ebcd' },
  summary: { label: 'Summary', none: 'Nothing recorded', icon: FileText, accent: '#b496ff', tint: '#1c1830', ink: '#d6c9ff' },
}

/** Text with its timestamps picked out, so they can be found again and compared with the playhead. */
function Timed({ text, ink }: { text: string; ink: string }) {
  return <>{timeParts(text).map((part, index) => part.time
    ? <span key={index} style={{ color: ink }} className="whitespace-nowrap rounded bg-white/[.06] px-1 font-mono text-[12px] tabular-nums">{part.text}</span>
    : <Fragment key={index}>{part.text}</Fragment>)}</>
}

function Field({ section }: { section: ConsistencySection }) {
  const look = LOOK[section.key]
  const Icon = look.icon
  const count = section.items.length
  if (section.none) {
    return <div className="flex items-start gap-2 rounded-lg border hairline px-3 py-2">
      <CircleCheck size={15} className="mt-0.5 shrink-0 text-[#70d5aa]" aria-hidden />
      <p className="min-w-0 text-[12px] leading-5 text-[#aeb5c1]"><span className="font-semibold text-[#9be8c8]">{look.label}</span> — {look.none}{section.note && <span className="mt-0.5 block text-[#929aa9]"><Timed text={section.note} ink={look.ink} /></span>}</p>
    </div>
  }
  return <section aria-label={look.label} style={{ borderLeftColor: look.accent, background: look.tint }} className="rounded-lg border-l-[3px] px-3.5 py-3">
    <h4 style={{ color: look.ink }} className="mb-2 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[.1em]"><Icon size={14} aria-hidden />{look.label}
      {section.key === 'human_review' && count > 0 && <StatusChip tone="amber" className="ml-auto normal-case tracking-normal">{count} to decide</StatusChip>}
      {section.key === 'recommended_checks' && count > 0 && <StatusChip tone="blue" className="ml-auto normal-case tracking-normal">{count} to check</StatusChip>}
    </h4>
    {!section.present ? <p className="muted text-[13px]">Unavailable</p>
      : section.items.length ? <ol className="space-y-2.5">{section.items.map((item, index) => <li key={index} className="flex gap-2.5">
        <span style={{ background: look.accent }} className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-[#11111a]" aria-hidden>{index + 1}</span>
        <p className="min-w-0 flex-1 break-words text-[13px] leading-[1.65] text-[#e8ebf1]"><Timed text={item} ink={look.ink} /></p>
      </li>)}</ol>
      : <div className="space-y-2">{section.paragraphs.map((paragraph, index) => <p key={index} className="break-words text-[13px] leading-[1.65] text-[#e8ebf1]"><Timed text={paragraph} ink={look.ink} /></p>)}</div>}
  </section>
}

/** What the review column needs of "Your answers": the text, a way to change it (unsaved) and whether it may be changed now. */
export type AnswersApi = { text: string; onChange: (next: string) => void; ready: boolean }

function QuestionCard({ number, question, all, answers, agent }: { number: number; question: Question; all: Question[]; answers: AnswersApi; agent?: AgentMarks }) {
  const [own, setOwn] = useState('')
  const picked = selection(answers.text, question, all)
  const look = LOOK.human_review
  const locked = !answers.ready
  const choose = (text: string | null) => answers.onChange(setAnswer(answers.text, question, text, all))
  const answered = agent ? answerFor(agent.decisions, question.id) : null
  const forClaude = !picked && !!agent?.claudeQuestions.has(question.id)
  const open = !picked && !forClaude
  return <li className={`flex gap-2.5 ${open ? '-ml-2 rounded-md border-l-2 border-[#f0a64a] pl-1.5' : ''}`}>
    <span style={{ background: look.accent }} className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-[#11111a]" aria-hidden>{number}</span>
    <div className="min-w-0 flex-1 space-y-2">
      <div className="flex flex-wrap items-center gap-1.5"><StatusChip tone="amber">{topicLabel(question.topic)}</StatusChip>{question.character && <StatusChip tone="blue">{question.character}</StatusChip>}
        {open && <StatusChip tone="amber" icon={<UserRoundCheck size={11} aria-hidden />} className="ml-auto" title="No answer to this question is written in Your answers yet">Needs your answer</StatusChip>}
        {forClaude && <StatusChip tone="purple" icon={<Bot size={11} aria-hidden />} className="ml-auto" title="Claude (the second reviewer) answers this next; it comes back to you only if Claude is unsure">Claude will review</StatusChip>}</div>
      <p id={`question-${question.id}`} className="break-words text-[13px] font-medium leading-[1.55] text-[#f2f4f8]">{question.question}</p>
      <div role="radiogroup" aria-labelledby={`question-${question.id}`} className="grid gap-1.5">
        {question.options.map(option => {
          const on = picked?.option === option.id
          return <button key={option.id} type="button" role="radio" aria-checked={on} disabled={locked} onClick={() => choose(on ? null : option.text)}
            title={on ? 'Click again to clear this answer' : 'Write this as the answer'}
            className={`flex w-full items-start gap-2 rounded-lg border px-2.5 py-1.5 text-left text-[12.5px] leading-5 transition-colors disabled:opacity-50 ${on ? 'border-[#f0a64a] bg-[#3a2c12] text-white' : 'border-[#343b49] bg-[#171c25] text-[#d6dbe5] hover:border-[#6b5a2f]'}`}>
            <span className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border text-[10px] font-bold ${on ? 'border-[#f0a64a] bg-[#f0a64a] text-[#11111a]' : 'border-[#4a5262] text-[#929aa9]'}`} aria-hidden>{on ? <Check size={10} /> : option.id}</span>
            <span className="min-w-0 flex-1 break-words">{option.text}</span>
            {question.recommended === option.id && <StatusChip tone="green" className="shrink-0" title="The evidence favours this option">Evidence favours</StatusChip>}
          </button>
        })}
      </div>
      {answered && <DecisionStrip decision={answered.decision} note={answered.note} onChanged={agent!.onChanged}>answered {answered.option ? <b>{answered.option}</b> : null} — ‘{answered.text}’</DecisionStrip>}
      {picked?.custom && <p className="flex items-start gap-1.5 rounded-lg border border-[#4a4363] bg-[#221c33] px-2.5 py-1.5 text-[12px] text-[#d6c9ff]"><span className="min-w-0 flex-1 break-words">Your own answer: {picked.custom}</span><button type="button" onClick={() => choose(null)} disabled={locked} className="shrink-0 text-[#c3aeff] hover:text-white" aria-label={`Clear your answer to ${answerLabel(question)}`}><X size={13} aria-hidden /></button></p>}
      <form className="flex gap-1.5" onSubmit={event => { event.preventDefault(); if (own.trim()) { choose(own); setOwn('') } }}>
        <input className="field min-w-0 flex-1 !px-2 !py-1 !text-[12px]" value={own} maxLength={400} disabled={locked} onChange={event => setOwn(event.target.value)} aria-label={`Your own answer to: ${question.question}`} placeholder="None of these? Write your own answer" />
        <button type="submit" className="btn btn-quiet !px-2 !py-1 !text-[12px]" disabled={locked || !own.trim()}>Use</button>
      </form>
      {question.why && <p className="muted text-[11.5px] leading-4">{question.why}</p>}
    </div>
  </li>
}

const VERDICT: Record<VisualCheck['verdict'], { label: string; tone: 'red' | 'amber' | 'green' }> = {
  contradicts: { label: 'Picture disagrees', tone: 'red' },
  unclear: { label: 'Picture unclear', tone: 'amber' },
  agrees: { label: 'Picture agrees', tone: 'green' },
}

/** What the reviewer saw when it compared the stored readings with pictures of the cut; agreements stay folded away. */
function VisualChecks({ checks }: { checks: VisualCheck[] }) {
  const flagged = checks.filter(check => check.verdict !== 'agrees')
  const agreed = checks.length - flagged.length
  const row = (check: VisualCheck, index: number) => <li key={index} className="space-y-1">
    <div className="flex flex-wrap items-center gap-1.5">
      <StatusChip tone={VERDICT[check.verdict].tone}>{VERDICT[check.verdict].label}</StatusChip>
      <span className="rounded bg-white/[.06] px-1 font-mono text-[11.5px] text-[#b4d2ff]">{check.picture}</span>
      {check.character && <StatusChip tone="blue">{check.character}</StatusChip>}
    </div>
    {check.claim && <p className="break-words text-[12.5px] leading-5 text-[#aeb5c1]"><span className="font-semibold text-[#929aa9]">Stored: </span>{check.claim}</p>}
    {check.seen && <p className="break-words text-[12.5px] leading-5 text-[#e8ebf1]"><span className="font-semibold text-[#b4d2ff]">Seen: </span>{check.seen}</p>}
  </li>
  // A contradiction is something for the person to look at: it gets the amber edge the other open items have.
  const disagrees = checks.some(check => check.verdict === 'contradicts')
  return <section data-need="picture-check" aria-label="Checked against the pictures" className={`rounded-lg border-l-[3px] bg-[#141f33] px-3.5 py-3 ${disagrees ? 'border-l-[#f0a64a] ring-1 ring-inset ring-[#6b5a2f]' : 'border-l-[#6aa8ff]'}`}>
    <h4 className="mb-2 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[.1em] text-[#b4d2ff]"><Eye size={14} aria-hidden />Checked against the pictures
      <StatusChip tone={flagged.some(check => check.verdict === 'contradicts') ? 'red' : 'green'} className="ml-auto normal-case tracking-normal">{flagged.length ? `${flagged.length} flagged` : 'All agree'}</StatusChip></h4>
    {checks.length === 0 ? <p className="muted text-[12px]">No picture showed a face, pose or mouth the evidence describes.</p>
      : <>{flagged.length > 0 && <ol className="space-y-3">{flagged.map(row)}</ol>}
        {agreed > 0 && <Disclosure label={`${agreed} reading${agreed === 1 ? '' : 's'} the pictures confirm`}><ol className="space-y-3">{checks.filter(check => check.verdict === 'agrees').map(row)}</ol></Disclosure>}</>}
  </section>
}

function Decisions({ questions, answers, agent }: { questions: Question[]; answers: AnswersApi; agent?: AgentMarks }) {
  const look = LOOK.human_review
  const answered = questions.filter(question => selection(answers.text, question, questions)).length
  return <section data-need="decisions" aria-label="Decisions" style={{ borderLeftColor: look.accent, background: look.tint }} className="rounded-lg border-l-[3px] px-3.5 py-3">
    <h4 style={{ color: look.ink }} className="mb-2 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[.1em]"><UserRoundCheck size={14} aria-hidden />Decisions
      <StatusChip tone={answered === questions.length ? 'green' : 'amber'} className="ml-auto normal-case tracking-normal">{answered}/{questions.length} answered</StatusChip></h4>
    <p className="muted mb-3 text-[11.5px] leading-4">Pick an answer and it is written into Your answers (unsaved until you save there). Nothing is decided for you.</p>
    <ol className="space-y-4">{questions.map((question, index) => <QuestionCard key={question.id} number={index + 1} question={question} all={questions} answers={answers} agent={agent} />)}</ol>
  </section>
}

export function ConsistencySummary({ cutId, result, loaded, jobs, disabled, onStarted, lead, extra, answers, agent }: { cutId: string; result?: AnalysisResult; loaded: boolean; jobs: Job[]; disabled: boolean; onStarted: () => Promise<void>; lead?: ReactNode; extra?: ReactNode; answers?: AnswersApi; agent?: AgentMarks }) {
  const [working, setWorking] = useState(false)
  const [message, setMessage] = useState('')
  const view = useMemo(() => result ? parseConsistency(result.content) : null, [result?.content])
  const activeJob = jobs.find(job => job.pipeline === 'consistency_summary' && (job.status === 'queued' || job.status === 'running') && job.cuts.some(cut => cut.id === cutId))
  async function queue() {
    setWorking(true); setMessage('')
    try {
      // The beats are rewritten first and the review waits for them, so it reads the current acting plan.
      await api(`/cuts/${cutId}/consistency-summary?with_beats=true`, { method: 'POST' })
      setMessage('Performance beats and consistency summary queued.'); await onStarted()
    } catch (error) { setMessage((error as Error).message) } finally { setWorking(false) }
  }
  const rerun = <ActionButton compact icon={<RefreshCw size={13} aria-hidden />} pending={working} disabled={disabled} onClick={() => void queue()} title="Rewrite the performance beats, then write a new consistency summary for this cut">{result ? 'Re-run' : 'Queue'}</ActionButton>
  const pictures = typeof result?.settings?.pictures === 'number' ? result.settings.pictures : 0
  // Only reviews written since pictures were offered record `vision`; older ones say nothing about it.
  const sight = !result || typeof result.settings?.vision !== 'boolean' ? null
    : result.settings.vision ? <StatusChip tone="blue" icon={<Eye size={11} aria-hidden />} title="The reviewer looked at pictures of this cut">{pictures} pictures</StatusChip>
      : <StatusChip tone="amber" icon={<EyeOff size={11} aria-hidden />} title="The text model that wrote this review could not read images, so it judged from the text evidence alone">Checked without pictures</StatusChip>
  const status = activeJob ? <StatusChip tone="purple" title="A consistency summary for this cut is being written; this text is the previous version.">{activeJob.status === 'queued' ? 'Queued' : 'Updating…'}</StatusChip> : sight
  // "All clear" is only said when the model explicitly found nothing to decide or check, never because a field is missing.
  const quiet = !!view && (view.format === 'questions' ? view.questions.length === 0 : view.sections.human_review.none && view.sections.recommended_checks.none)
  const allClear = !!view && quiet && !view.conflicts
  return <StepPanel step={1} kind="consistency_summary" title="Review" subtitle="Consistency summary · read first" updated={result?.created_at ? formatAgo(result.created_at) : undefined} tag={status} actions={rerun}>
    {!loaded ? <div className="space-y-3" aria-busy="true" aria-label="Loading consistency summary"><Skeleton className="h-9" /><Skeleton className="h-28" /><Skeleton className="h-28" /><Skeleton className="h-20" /></div>
      : !result || !view ? <div className="flex flex-1 flex-col items-center justify-center gap-2 px-4 py-10 text-center">
        <ScanSearch size={28} className="text-[#b496ff]" aria-hidden />
        <p className="text-sm font-semibold">No consistency summary yet</p>
        <p className="muted max-w-xs text-xs leading-5">It is written from this cut's visual analysis, transcript and subtitles. Run the analysis from Run settings, or queue it here once results exist.</p>
      </div>
      : !view.valid ? <>
        <WarningLine level="warn">This result is not a valid JSON object, so it cannot be laid out here. Repair it in Analysis fields below, or queue a new one.</WarningLine>
        <pre className="scrollbar max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg border hairline bg-[#151a22] p-3 text-xs leading-5 text-[#c8cdd8]">{result.content}</pre>
      </>
      : <>
        {allClear
          ? <div className="flex items-center gap-2 rounded-lg border border-[#2f6b55] bg-[#12261f] px-3 py-2 text-xs font-semibold text-[#9be8c8]" role="status"><ShieldCheck size={15} aria-hidden />All clear — nothing needs your review</div>
          : (view.decisions > 0 || view.checks > 0 || view.conflicts) && <div className="flex flex-wrap items-center gap-1.5" role="status" aria-label="What needs attention">
            <span className="muted text-[11px] font-semibold uppercase tracking-[.1em]">To do</span>
            {view.decisions > 0 && <StatusChip tone="amber" icon={<UserRoundCheck size={11} aria-hidden />}>{view.decisions} to decide</StatusChip>}
            {view.checks > 0 && <StatusChip tone="blue" icon={<ListChecks size={11} aria-hidden />}>{view.checks} to check</StatusChip>}
            {view.conflicts && <StatusChip tone="red" icon={<TriangleAlert size={11} aria-hidden />}>Conflicts found</StatusChip>}
            {quiet && <StatusChip tone="green" icon={<CircleCheck size={11} aria-hidden />}>No decisions needed</StatusChip>}
          </div>}
        {/* What the person decides comes first; the picture check is the evidence behind it. */}
        {view.format === 'questions' && view.questions.length > 0 && <Decisions questions={view.questions} answers={answers ?? { text: '', onChange: () => {}, ready: false }} agent={agent} />}
        {view.visualChecks && <VisualChecks checks={view.visualChecks} />}
        {(view.format === 'questions' ? QUESTION_ORDER : ORDER).map(key => <Field key={key} section={view.sections[key]} />)}
        <Disclosure label="Generation details"><ResultMetrics result={result} /></Disclosure>
        <p className="muted text-[11px] leading-4">To edit this result or see earlier versions, use Analysis fields below.</p>
      </>}
    {message && <p role="status" className="text-xs text-[#c3aeff]">{message}</p>}
    {lead}
    {extra}
  </StepPanel>
}
