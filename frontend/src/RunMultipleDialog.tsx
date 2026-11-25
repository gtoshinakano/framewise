import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CheckCircle2, Circle, Inbox, Layers, LoaderCircle, MinusCircle, PictureInPicture2, Play, Square, XCircle, X } from 'lucide-react'
import { api, jsonBody, type Cut } from './api'
import { editorPath, type DialogTab, type RunSection } from './cutRoute'
import { SegmentedControl, Switch } from './EditorUi'
import { useEpisodeEvents } from './episodeEvents'
import { formatAgo } from './queueView'
import { ActionButton, StatusChip, type Tone } from './RunUi'
import { SUBTITLE_LANGUAGES } from './SubtitlePanel'
import { useToast } from './Toasts'
import {
  STAGES, STAGE_INTRO, STAGE_LABEL, STATUS_LABEL, cutNumbers, defaultStage, evaluateCounts, groupHeld, humanInbox, inboxText, plural,
  progressText, shownRun, startButton, stepLinks, stepText, stepTone, type BatchRun, type HeldCut, type PlanStep, type Preview, type StageKey,
} from './runMultipleView'

const STATUS_TONE: Record<BatchRun['status'], Tone> = { running: 'purple', finished: 'green', failed: 'red', cancelled: 'grey' }
const POLL_MS = 5000

/**
 * Run Multiple (backend batch_runs.py, docs/autopilot/HARNESS.md D37): the episode's batch stages in one place. Before
 * anything runs it shows what each step will do and which cuts wait for the person and why; while a run goes and once it
 * is over, what each step did and found. Prepare runs once; Evaluate needs Prepare and runs only on the cuts it can.
 */
export function RunMultipleDialog({ episodeId, cuts, subtitleLanguage, initialStage, onClose, onOpenInbox, onChanged, onFloatHeld }: {
  episodeId: string
  cuts: Cut[]
  subtitleLanguage: string | null
  initialStage?: StageKey
  onClose: () => void
  onOpenInbox: () => void
  onChanged: () => void
  /** Keep the Evaluate stage's "Waiting for you" list on screen (WaitingForYou.tsx) and close this dialog. */
  onFloatHeld: () => void
}) {
  const toast = useToast()
  const panel = useRef<HTMLDivElement>(null)
  const [runs, setRuns] = useState<BatchRun[] | null>(null)
  const [stage, setStage] = useState<StageKey | null>(null)
  const [previews, setPreviews] = useState<Partial<Record<StageKey, Preview>>>({})
  const [previewError, setPreviewError] = useState('')
  const [subtitles, setSubtitles] = useState({ on: false, language: subtitleLanguage || '' })
  const [followed, setFollowed] = useState<string | null>(null)
  const [working, setWorking] = useState<string | null>(null)
  const byNumber = useMemo(() => new Map(cuts.map(cut => [cut.cut_index + 1, cut.id])), [cuts])
  const subtitleChoice = subtitles.on && subtitles.language.trim().length >= 2 ? subtitles.language.trim() : ''

  const loadRuns = useCallback(async () => {
    try { setRuns(await api<BatchRun[]>(`/episodes/${episodeId}/batch-runs`)) } catch (err) { toast({ tone: 'error', message: (err as Error).message }) }
  }, [episodeId, toast])
  const loadPreview = useCallback(async (which: StageKey) => {
    const params = new URLSearchParams({ stage: which })
    if (which === 'prepare' && subtitleChoice) params.set('subtitles', subtitleChoice)
    try {
      const preview = await api<Preview>(`/episodes/${episodeId}/batch-runs/preview?${params}`)
      setPreviews(current => ({ ...current, [which]: preview }))
      setPreviewError('')
    } catch (err) {
      setPreviewError((err as Error).message)
    }
  }, [episodeId, subtitleChoice])

  useEffect(() => { void loadRuns() }, [loadRuns])
  const prepared = !!(previews.prepare?.prepared ?? previews.evaluate?.prepared ?? runs?.some(run => run.stage === 'prepare' && run.status === 'finished'))
  useEffect(() => { if (runs && stage === null) setStage(defaultStage(runs, prepared, initialStage)) }, [runs, stage, prepared, initialStage])
  const run = runs && stage ? shownRun(runs, stage, followed) : undefined
  const running = runs?.find(item => item.status === 'running')
  useEffect(() => { if (stage && !run) void loadPreview(stage) }, [stage, run?.id, loadPreview])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (stage && stage !== 'prepare_cuts' && !previews.prepare_cuts) void loadPreview('prepare_cuts') }, [stage])  // eslint-disable-line react-hooks/exhaustive-deps

  // A running run moves on in the server: follow it (events, and a slow poll as the fallback), and refresh the editor once it ends.
  useEpisodeEvents(change => { if (change.jobs || change.data) { void loadRuns(); if (stage && !run) void loadPreview(stage) } })
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => void loadRuns(), POLL_MS)
    return () => clearInterval(timer)
  }, [running, loadRuns])
  const wasRunning = useRef<string | null>(null)
  useEffect(() => {
    if (running) { wasRunning.current = running.id; setFollowed(running.id); return }
    if (wasRunning.current) { wasRunning.current = null; onChanged(); void loadPreview('prepare_cuts'); if (stage) void loadPreview(stage) }
  }, [running?.id])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape' && !document.querySelector('.popover')) onClose() }
    window.addEventListener('keydown', key)
    return () => { window.removeEventListener('keydown', key); opener?.focus() }
  }, [onClose])

  function openCut(cutId: string | undefined, tab: DialogTab, section?: RunSection) {
    if (!cutId) return
    window.location.hash = `#${editorPath({ episodeId, cutId, analysis: true, tab, section })}`
    onClose()
  }

  async function start() {
    if (!stage) return
    setWorking('start')
    try {
      const body: Record<string, unknown> = { stage }
      if (stage === 'prepare' && subtitleChoice) body.subtitles = { language: subtitleChoice }
      const started = await api<BatchRun>(`/episodes/${episodeId}/batch-runs`, jsonBody(body))
      setFollowed(started.id)
      setRuns(current => [started, ...(current ?? []).filter(item => item.id !== started.id)])
      toast({ tone: 'success', message: `${STAGE_LABEL[stage]} started. You can close this window: it goes on in the server.` })
    } catch (err) {
      toast({ tone: 'error', message: (err as Error).message })
      void loadPreview(stage)
    } finally {
      setWorking(null)
    }
  }

  async function cancel(target: BatchRun) {
    setWorking('cancel')
    try {
      const stopped = await api<BatchRun>(`/batch-runs/${target.id}/cancel`, { method: 'POST' })
      setRuns(current => (current ?? []).map(item => item.id === stopped.id ? stopped : item))
      toast({ message: `${STAGE_LABEL[target.stage]} cancelled. Its queued jobs were cancelled too.` })
    } catch (err) {
      toast({ tone: 'error', message: (err as Error).message })
    } finally {
      setWorking(null)
    }
  }

  const preview = stage ? previews[stage] : undefined
  const button = stage ? startButton(stage, preview ?? null) : null
  const history = (runs ?? []).filter(item => item.stage === stage && item.id !== run?.id)
  const changed = previews.prepare_cuts ? new Set(previews.prepare_cuts.steps.flatMap(step => step.state === 'will_run' ? (step.cuts ?? []).map(item => item.cut) : [])).size : 0

  return <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="run-multiple-title" className="flex max-h-[92vh] w-full max-w-3xl flex-col rounded-xl border hairline bg-[#1b202a] shadow-2xl outline-none">
      <header className="shrink-0 border-b hairline px-5 pb-3 pt-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="run-multiple-title" className="flex items-center gap-2 text-base font-semibold"><Layers size={17} className="text-[#b496ff]" aria-hidden /> Run Multiple</h2>
            <p className="muted mt-1 text-xs leading-5">Runs one stage on many cuts. Before it starts you see what each step will do and which cuts wait for you; while it runs and afterwards, what each step did and found.</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close Run Multiple" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button>
        </div>
        {stage && <SegmentedControl label="Stage" value={stage} onChange={setStage} className="mt-3 w-full" options={STAGES.map(key => ({
          value: key, label: STAGE_LABEL[key],
          badge: key === 'prepare' ? <StageBadge text={running?.stage === 'prepare' ? 'running' : prepared ? 'done' : 'not run'} tone={running?.stage === 'prepare' ? 'purple' : prepared ? 'green' : 'amber'} />
            : key === 'prepare_cuts' ? (!prepared ? <StageBadge text="after Prepare" tone="grey" /> : changed ? <StageBadge text={String(changed)} tone="amber" /> : undefined)
              : !prepared ? <StageBadge text="after Prepare" tone="grey" /> : running?.stage === 'evaluate' ? <StageBadge text="running" tone="purple" /> : undefined,
        }))} />}
      </header>

      <div className="scrollbar min-h-0 flex-1 overflow-y-auto px-5 py-4 text-xs">
        {!stage ? <p className="muted">Loading…</p> : <>
          <p className="muted mb-3 leading-5">{STAGE_INTRO[stage]}</p>
          {run ? <RunView run={run} byNumber={byNumber} onOpenCut={openCut} onOpenInbox={() => { onClose(); onOpenInbox() }} onFloat={onFloatHeld}
            onPreview={() => { setFollowed(null); void loadPreview(stage) }} />
            : previewError ? <p role="alert" className="text-[#ff9fa7]">{previewError}</p>
              : !preview ? <p className="muted flex items-center gap-1.5"><LoaderCircle size={13} className="animate-spin" aria-hidden /> Working out what would run…</p>
                : <PlanView stage={stage} preview={preview} onOpenCut={openCut} onFloat={onFloatHeld} />}
          {history.length > 0 && <section className="mt-5">
            <h3 className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#8f97a6]">Earlier runs</h3>
            <ul className="space-y-1.5">{history.slice(0, 8).map(item => <li key={item.id}>
              <details className="rounded-lg border hairline bg-[#171c25] px-3 py-2">
                <summary className="flex cursor-pointer items-center justify-between gap-2"><span className="flex items-center gap-2"><StatusChip tone={STATUS_TONE[item.status]}>{STATUS_LABEL[item.status]}</StatusChip>{formatAgo(item.finished_at || item.created_at)}</span>{item.error && <span className="truncate text-[#ff9fa7]">{item.error}</span>}</summary>
                <StepList steps={item.steps} byNumber={byNumber} onOpenCut={openCut} />
              </details>
            </li>)}</ul>
          </section>}
        </>}
      </div>

      {stage && <footer className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-t hairline px-5 py-3">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2 text-xs">
          {stage === 'prepare' && !run && preview?.available && <>
            <Switch checked={subtitles.on} onChange={on => setSubtitles({ ...subtitles, on })} label="Also read burned-in subtitles" hint="Optional supporting material: nothing waits for subtitles and they never make anything out of date" />
            {subtitles.on && <><input className="field !h-8 !w-44 !py-1 !text-xs" list="run-multiple-languages" placeholder="Language, e.g. Japanese" value={subtitles.language}
              onChange={event => setSubtitles({ ...subtitles, language: event.target.value })} aria-label="Subtitle language" />
              <datalist id="run-multiple-languages">{SUBTITLE_LANGUAGES.map(name => <option key={name} value={name} />)}</datalist></>}
          </>}
          {stage === 'evaluate' && !run && preview && button && !button.disabled && !!preview.held?.length &&
            <span className="muted">{plural(preview.held.length, 'cut')} waiting for you {preview.held.length === 1 ? 'is' : 'are'} skipped. Settle them first, or evaluate the others now.</span>}
          {button?.disabled && run?.status !== 'running' && button.why && <span className="muted">{button.why}</span>}
        </div>
        <div className="flex shrink-0 gap-2">
          <button type="button" className="btn btn-quiet" onClick={onClose}>Close</button>
          {run?.status === 'running'
            ? <ActionButton icon={<Square size={13} />} pending={working === 'cancel'} disabled={!!working} onClick={() => void cancel(run)}>Cancel run</ActionButton>
            : button && <ActionButton variant="primary" icon={<Play size={14} />} pending={working === 'start'} disabled={button.disabled || !!working || (stage === 'prepare' && subtitles.on && !subtitleChoice)}
              title={button.why || undefined} onClick={() => void start()}>{run ? 'Run again' : button.label}</ActionButton>}
        </div>
      </footer>}
    </div>
  </div>
}

function StageBadge({ text, tone }: { text: string; tone: Tone }) {
  return <StatusChip tone={tone} className="!px-1.5 !py-0 !text-[10px]">{text}</StatusChip>
}

type OpenCut = (cutId: string | undefined, tab: DialogTab, section?: RunSection) => void

/** Before a run: what each step will do, and for Evaluate what will be evaluated and what waits for the person. */
function PlanView({ stage, preview, onOpenCut, onFloat }: { stage: StageKey; preview: Preview; onOpenCut: OpenCut; onFloat: () => void }) {
  const counts = stage === 'evaluate' ? evaluateCounts(preview) : null
  return <>
    {!preview.available && <p className="mb-3 rounded-lg border border-[#4a4030] bg-[#2a2418] px-3 py-2 text-[#f5c678]">{preview.reason}</p>}
    {counts && preview.available && <div className="mb-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Figure value={counts.cuts} label="cuts will be evaluated" />
      <Figure value={counts.visual} label="need visual analysis" />
      <Figure value={counts.beats + counts.consistency} label="need only beats or the review now" />
      <Figure value={counts.held} label="wait for you" warn={counts.held > 0} />
    </div>}
    <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-[#8f97a6]">What will happen</h3>
    <StepList steps={preview.steps} />
    {stage === 'evaluate' && !!preview.held?.length && <HeldList held={preview.held} onOpenCut={onOpenCut} onFloat={onFloat}
      intro="These cuts are skipped until you settle what they wait for. Click a cut to open it where the fix is." />}
  </>
}

function Figure({ value, label, warn = false }: { value: number; label: string; warn?: boolean }) {
  return <div className="rounded-lg border hairline bg-[#171c25] px-3 py-2">
    <div className={`text-lg font-semibold tabular-nums ${warn ? 'text-[#f5c678]' : ''}`}>{value}</div>
    <div className="muted text-[11px] leading-4">{label}</div>
  </div>
}

/** A run going on, or one that just ended: its steps with what they found, then what needs the person now. */
function RunView({ run, byNumber, onOpenCut, onOpenInbox, onFloat, onPreview }: { run: BatchRun; byNumber: Map<number, string>; onOpenCut: OpenCut; onOpenInbox: () => void; onFloat: () => void; onPreview: () => void }) {
  const summary = run.summary ?? {}
  const people = humanInbox(summary.inbox)
  return <>
    <div className="mb-2 flex flex-wrap items-center gap-2">
      <StatusChip tone={STATUS_TONE[run.status]} icon={run.status === 'running' ? <LoaderCircle size={11} className="animate-spin" aria-hidden /> : undefined}>{STATUS_LABEL[run.status]}</StatusChip>
      <span className="muted">started {formatAgo(run.created_at)}{run.finished_at ? `, ended ${formatAgo(run.finished_at)}` : ''}</span>
      {run.status === 'running' ? <span className="muted">It goes on in the server if you close this window.</span>
        : <button type="button" className="ml-auto text-[#c3aeff] hover:underline" onClick={onPreview}>See what would run now</button>}
    </div>
    {run.error && <p role="alert" className="mb-2 text-[#ff9fa7]">{run.error}</p>}
    <StepList steps={run.steps} byNumber={byNumber} onOpenCut={onOpenCut} />
    {run.status !== 'running' && <section className="mt-4 rounded-lg border hairline bg-[#171c25] p-3">
      <h3 className="mb-1.5 text-sm font-semibold">Needs you now</h3>
      {inboxText(summary.inbox) ? <p className="flex flex-wrap items-center gap-2">{inboxText(summary.inbox)}
        {people > 0 && <ActionButton compact icon={<Inbox size={13} />} onClick={onOpenInbox}>Open the Inbox</ActionButton>}</p>
        : <p className="muted flex items-center gap-1.5"><CheckCircle2 size={13} className="text-[#7bdcb2]" aria-hidden /> No questions in the Inbox.</p>}
      {!!summary.failures && <p className="mt-1 text-[#ff9fa7]">{plural(summary.failures, 'failed run')} in the queue (Queue → Needs attention).</p>}
      {!!summary.held?.length && <HeldList held={summary.held} onOpenCut={onOpenCut} onFloat={onFloat} intro="Not evaluated: these wait for you. Settle them, then evaluate again; only they will run." />}
    </section>}
  </>
}

const STEP_ICON = {
  done: <CheckCircle2 size={15} className="text-[#7bdcb2]" aria-hidden />,
  active: <LoaderCircle size={15} className="animate-spin text-[#b496ff] motion-reduce:animate-none" aria-hidden />,
  waiting: <Circle size={15} className="text-[#8f97a6]" aria-hidden />,
  quiet: <MinusCircle size={15} className="text-[#5c6472]" aria-hidden />,
  failed: <XCircle size={15} className="text-[#ff9fa7]" aria-hidden />,
}

function StepList({ steps, byNumber, onOpenCut }: { steps: PlanStep[]; byNumber?: Map<number, string>; onOpenCut?: OpenCut }) {
  return <ol className="divide-y divide-[#2a303b] rounded-lg border hairline bg-[#171c25]">{steps.map(step => {
    const tone = stepTone(step.state)
    const progress = step.state === 'running' ? step.progress : undefined
    const planned = (step.state === 'will_run' || step.state === 'pending' || step.state === 'running') ? step.cuts ?? [] : []
    return <li key={step.step} className={`flex gap-2.5 px-3 py-2 ${tone === 'quiet' ? 'opacity-70' : ''}`}>
      <span className="mt-0.5 shrink-0">{STEP_ICON[tone]}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2"><span className="font-semibold">{step.label}</span>{progress && <span className="muted shrink-0 tabular-nums">{progressText(progress)}</span>}</div>
        <p className={tone === 'failed' ? 'text-[#ff9fa7]' : 'muted'}>{stepText(step)}{planned.length > 0 && !step.summary ? ` · cuts ${cutNumbers(planned.map(item => item.cut))}` : ''}</p>
        {progress && progress.total > 0 && <div className="mt-1 h-1 overflow-hidden rounded bg-[#2a303b]"><div className="h-full bg-[#b496ff]" style={{ width: `${Math.round(100 * progress.finished / progress.total)}%` }} /></div>}
        {step.start_error && <p className="text-[#ff9fa7]">Could not queue: {step.start_error}</p>}
        {step.errors?.map(item => <p key={item.cut_id} className="text-[#ff9fa7]">Cut {item.cut}: {item.error}</p>)}
        {byNumber && onOpenCut && stepLinks(step).map(link => <div key={link.label} className="mt-1 flex flex-wrap items-center gap-1">
          <span className="muted">{link.label}:</span>
          {link.cuts.map(number => <CutChip key={number} number={number} onClick={() => onOpenCut(byNumber.get(number), link.tab, link.section)} disabled={!byNumber.get(number)} />)}
        </div>)}
      </div>
    </li>
  })}</ol>
}

function HeldList({ held, intro, onOpenCut, onFloat }: { held: HeldCut[]; intro: string; onOpenCut: OpenCut; onFloat: () => void }) {
  return <section className="mt-4">
    <div className="mb-1 flex items-center justify-between gap-2">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[#f5c678]">Waiting for you · {plural(held.length, 'cut')}</h3>
      <button type="button" onClick={onFloat} className="flex items-center gap-1 text-[#c3aeff] hover:underline"
        title="Keep this list in a small window over the editor while you settle the cuts"><PictureInPicture2 size={13} aria-hidden /> Keep on screen</button>
    </div>
    <p className="muted mb-2">{intro}</p>
    <div className="space-y-2">{groupHeld(held).map(group => <div key={group.key} className="rounded-lg border hairline bg-[#171c25] px-3 py-2">
      <p className="mb-1 font-semibold">{group.label} <span className="muted font-normal">({group.cuts.length})</span></p>
      <div className="flex flex-wrap gap-1">{group.cuts.map(item => <CutChip key={item.cut_id} number={item.cut} title={item.reasons.join('; ')}
        onClick={() => onOpenCut(item.cut_id, group.tab, group.section)} />)}</div>
    </div>)}</div>
  </section>
}

/** A cut number that opens the cut; `current` marks the cut the editor is on. */
export function CutChip({ number, title, disabled, current = false, onClick }: { number: number; title?: string; disabled?: boolean; current?: boolean; onClick: () => void }) {
  return <button type="button" onClick={onClick} disabled={disabled} title={title ? `Cut ${number}: ${title}` : `Open cut ${number}`} aria-current={current || undefined}
    className={`rounded-md border px-1.5 py-0.5 font-mono text-[11px] tabular-nums hover:bg-[#2b3342] disabled:opacity-50 ${current ? 'border-[#f5c678] bg-[#3a3020] text-[#f5c678]' : 'border-[#3a4352] bg-[#202632] text-[#c3aeff]'}`}>{number}</button>
}
