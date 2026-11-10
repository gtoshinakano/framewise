import { Fragment, useMemo, useState, type ReactNode } from 'react'
import { CircleCheck, FileText, ListChecks, RefreshCw, ScanSearch, ShieldCheck, TriangleAlert, UserRoundCheck } from 'lucide-react'
import { api, type AnalysisResult, type Job } from './api'
import { ResultMetrics } from './ResultMetrics'
import { ActionButton, Skeleton, StatusChip, WarningLine } from './RunUi'
import { Disclosure, StepPanel } from './ReviewUi'
import { parseConsistency, timeParts, type ConsistencyKey, type ConsistencySection } from './consistencyView'

// What the user works from comes first. Each field keeps its stored name's wording and gets its own colour and icon,
// matching the tones used on the Run settings tab; the label and icon always carry the meaning, never colour alone.
const ORDER: ConsistencyKey[] = ['human_review', 'recommended_checks', 'confusing', 'clear', 'summary']
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

export function ConsistencySummary({ cutId, result, loaded, jobs, disabled, onStarted, extra }: { cutId: string; result?: AnalysisResult; loaded: boolean; jobs: Job[]; disabled: boolean; onStarted: () => Promise<void>; extra?: ReactNode }) {
  const [working, setWorking] = useState(false)
  const [message, setMessage] = useState('')
  const view = useMemo(() => result ? parseConsistency(result.content) : null, [result?.content])
  const activeJob = jobs.find(job => job.pipeline === 'consistency_summary' && (job.status === 'queued' || job.status === 'running') && job.cuts.some(cut => cut.id === cutId))
  async function queue() {
    setWorking(true); setMessage('')
    try {
      await api(`/cuts/${cutId}/consistency-summary`, { method: 'POST' })
      setMessage('Consistency summary queued.'); await onStarted()
    } catch (error) { setMessage((error as Error).message) } finally { setWorking(false) }
  }
  const rerun = <ActionButton compact icon={<RefreshCw size={13} aria-hidden />} pending={working} disabled={disabled || !!activeJob} onClick={() => void queue()} title="Queue a new consistency summary for this cut">{result ? 'Re-run' : 'Queue'}</ActionButton>
  const status = activeJob ? <StatusChip tone="purple" title="A consistency summary for this cut is being written; this text is the previous version.">{activeJob.status === 'queued' ? 'Queued' : 'Updating…'}</StatusChip> : null
  // "All clear" is only said when the model explicitly found nothing to decide or check, never because a field is missing.
  const quiet = !!view && view.sections.human_review.none && view.sections.recommended_checks.none
  const allClear = !!view && quiet && !view.conflicts
  return <StepPanel step={1} kind="consistency_summary" title="Review" subtitle="Consistency summary · read first" tag={status} actions={rerun}>
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
        {ORDER.map(key => <Field key={key} section={view.sections[key]} />)}
        <Disclosure label="Generation details"><ResultMetrics result={result} /></Disclosure>
        <p className="muted text-[11px] leading-4">To edit this result or see earlier versions, use Analysis fields below.</p>
      </>}
    {message && <p role="status" className="text-xs text-[#c3aeff]">{message}</p>}
    {extra}
  </StepPanel>
}
