import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Check, ChevronDown, Clapperboard, Copy, Download, History as HistoryIcon, LoaderCircle, MessageSquare, Pencil, Play, RefreshCw, Save, Sparkles, Square, WandSparkles, X } from 'lucide-react'
import { mediaUrl } from './api'
import CutAssistant from './CutAssistant'
import DialogueTab from './DialogueTab'
import { PlayerProvider, PlayerSlot, SourcePlayer, usePlayer, usePlayerValue } from './SourcePlayer'
import SequenceTimeline from './SequenceTimeline'
import type { DiffSegment } from './assistantApi'
import { DiffText } from './DiffText'
import { CopyButton, copyText } from './Markdown'
import { sequenceApi, shotFrameUrl, type Bible, type BoundaryChoice, type Delivery, type RunStep, type SequenceBoundary, type SequenceDetail, type SequenceShot, type ShotDialogue, type ShotHistory } from './sequenceApi'
import { boundaryChoices, changeAt, changeSize, clock, deciderLabel, DELIVERIES, deliveryLabel, edgeTags, hasOwnVoices, issueInstruction, issuesFor, progress, promptSections, revisionLabel, shotStatus, speakerRanges, stage, staleText, timecode, timecodeSegments, versionNote, type Tone } from './sequenceView'

type Tab = 'dialogue' | 'direction' | 'shots' | 'assistant'
const TONES: Record<Tone, string> = { good: 'bg-[#173a2b] text-[#9ff0c8]', busy: 'bg-[#2b2442] text-[#d6c8ff]', warn: 'bg-[#4a3a17] text-[#ffd59a]', bad: 'bg-[#3d2025] text-[#ffaaa7]', idle: 'bg-[#262b35] text-[#aeb5c1]' }
const DELIVERY_COLOURS: Record<Delivery, string> = { on_screen: 'bg-[#2f7a57]', off_screen: 'bg-[#4b5263]', inner_voice: 'bg-[#6a55a8]', narrative: 'bg-[#a8793a]' }
const ORIGIN: Record<string, string> = { model: 'written by the director', manual_edit: 'edited by you', assistant_edit: 'edited with the assistant' }

/** The sequence id as a chip that copies it in one click (`/api/v1/sequences/{id}` and the AI Director address a sequence by it). */
function CopyId({ id }: { id: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')
  return <button type="button" onClick={async () => { setState(await copyText(id) ? 'copied' : 'failed'); setTimeout(() => setState('idle'), 1600) }}
    title={state === 'failed' ? 'Copy failed' : 'Copy the sequence id'} aria-label={`Copy sequence id ${id}`}
    className="inline-flex min-w-0 items-center gap-1 rounded px-1 py-0.5 font-mono text-[11px] text-[#8b93a1] hover:bg-[#272e3a] hover:text-white">
    <span className="truncate">{id}</span>
    {state === 'copied' ? <Check size={11} className="shrink-0 text-[#7bdcb2]" aria-hidden /> : state === 'failed' ? <X size={11} className="shrink-0 text-[#ffaaa7]" aria-hidden /> : <Copy size={11} className="shrink-0" aria-hidden />}
    <span role="status" className="sr-only">{state === 'copied' ? 'Copied' : state === 'failed' ? 'Copy failed' : ''}</span>
  </button>
}

/** The sequence workspace: direct the range (bible and shot plan), review, write the shot prompts, and download. */
export default function SequenceDialog({ sequenceId, episodeId, proxyReady, onClose, onChanged }: {
  sequenceId: string; episodeId: string; proxyReady: boolean; onClose: () => void; onChanged: () => void
}) {
  const [detail, setDetail] = useState<SequenceDetail | null>(null)
  const [tab, setTab] = useState<Tab>('direction')
  const tabChosen = useRef(false)
  // One video for the whole workspace: it loads once, here, and every tab only moves its playhead.
  const player = useMemo(() => new SourcePlayer(), [])
  useEffect(() => { if (proxyReady) player.load(mediaUrl(episodeId, 'proxy')) }, [player, proxyReady, episodeId])
  useEffect(() => () => player.dispose(), [player])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [working, setWorking] = useState('')
  const [error, setError] = useState('')
  const [exporting, setExporting] = useState('')
  const [assistantMounted, setAssistantMounted] = useState(false)
  const panel = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    try { setDetail(await sequenceApi.get(sequenceId)) } catch (err) { setError((err as Error).message) }
  }, [sequenceId])
  useEffect(() => { void load() }, [load])
  const active = !!detail?.jobs.some(job => job.status === 'queued' || job.status === 'running')
  // While its jobs run the workspace follows them; the episode editor polls on its own schedule.
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => { void load() }, 2500)
    return () => clearInterval(timer)
  }, [active, load])
  // When the last job finishes, tell the editor so its sequence list and timeline lane catch up.
  const wasActive = useRef(false)
  useEffect(() => { if (wasActive.current && !active) onChanged(); wasActive.current = active }, [active, onChanged])
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape' && document.querySelectorAll('[aria-modal="true"]').length <= 1) onClose() }
    window.addEventListener('keydown', onKey)
    return () => { window.removeEventListener('keydown', onKey); opener?.focus() }
  }, [onClose])
  useEffect(() => { if (tab === 'assistant') setAssistantMounted(true) }, [tab])
  // Open on the Dialogue tab while the dialogue still waits for the user; later visits keep whatever tab was last used.
  useEffect(() => {
    if (!detail || tabChosen.current) return
    tabChosen.current = true
    if (!detail.bible && detail.compilation?.has_speech && detail.compilation.state !== 'confirmed') setTab('dialogue')
  }, [detail])

  /** Run a change, show its error, and reload: every action answers with the new state or needs a fresh read. */
  const act = useCallback(async (label: string, action: () => Promise<SequenceDetail | unknown>) => {
    setWorking(label); setError('')
    try {
      const result = await action()
      if (result && typeof result === 'object' && 'sequence' in result) setDetail(result as SequenceDetail)
      else await load()
      onChanged()
      return true
    } catch (err) { setError((err as Error).message); return false } finally { setWorking('') }
  }, [load, onChanged])
  const run = useCallback((step: RunStep, extra: { shot_keys?: string[]; instruction?: string } = {}) => act(step, () => sequenceApi.run(sequenceId, { step, ...extra })), [act, sequenceId])

  async function download() {
    setExporting('Queued'); setError('')
    try {
      const url = await sequenceApi.exportZip(sequenceId, setExporting)
      window.location.assign(url)
    } catch (err) { setError((err as Error).message) } finally { setExporting('') }
  }

  const shots = detail?.shots || []
  const selected = shots.find(shot => shot.key === selectedKey) || shots[0]
  const where = detail ? stage(detail) : null
  const counts = detail ? progress(detail) : null
  const flagged = useMemo(() => (detail?.review?.content.issues || []).filter(issue => issue.severity === 'fix' && shots.some(shot => shot.key === issue.shot_key)), [detail?.review, shots])
  const info = detail?.sequence
  const label = info ? `Sequence · cuts ${info.first_cut ?? '?'}–${info.last_cut ?? '?'} · ${clock(info.duration)}` : 'Sequence'

  function reviseFlagged() {
    // One rewrite per flagged shot, each with the review's suggestion for it as the instruction.
    return act('revise', async () => {
      for (const shot of shots) {
        const issues = issuesFor(flagged, shot)
        if (issues.length && !shot.job) await sequenceApi.run(sequenceId, { step: 'shots', shot_keys: [shot.key], instruction: issueInstruction(issues), instruction_source: 'review' })
      }
    })
  }

  return <PlayerProvider player={player}><div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Sequence ${info?.title || ''}`} className="flex h-[94vh] w-[96vw] max-w-[1800px] flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="grid grid-cols-[1fr_auto] items-center gap-3 border-b hairline px-5 py-3 lg:grid-cols-[1fr_auto_1fr]">
        <div className="min-w-0"><div className="flex min-w-0 items-center gap-2"><p className="shrink-0 text-[11px] font-semibold uppercase tracking-[.14em] text-[#e0a96d]">{label}</p><CopyId id={sequenceId} /></div><h2 className="truncate text-base font-semibold">{info?.title || 'Loading…'}</h2></div>
        <div className="order-3 col-span-2 flex justify-center lg:order-none lg:col-span-1"><div className="flex gap-1 rounded-lg border hairline p-1" role="tablist">
          {([['dialogue', `Dialogue${detail?.compilation ? { none: '', running: ' · compiling…', review: ' · review', confirmed: ' ✓' }[detail.compilation.state] : ''}`], ['direction', 'Direction'], ['shots', `Shots${counts?.shots ? ` · ${counts.ready}/${counts.shots}` : ''}`], ['assistant', 'Assistant']] as const).map(([value, text]) =>
            <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`rounded-md px-5 py-1.5 text-xs font-semibold ${tab === value ? 'bg-[#3a2f1f] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{text}</button>)}
        </div></div>
        <div className="flex items-center justify-end gap-2">
          <button type="button" className="btn btn-quiet px-3 py-2" disabled={!shots.length || !!exporting} onClick={() => void download()} title={detail?.ready ? 'Download sequence.json, the audio slices, the initial frames and the prompts' : 'Download what exists now; shots that are not written have no prompt yet'}>
            {exporting ? <LoaderCircle size={14} className="animate-spin" /> : <Download size={14} />} {exporting ? `Export · ${exporting}` : 'Download ZIP'}</button>
          <button onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close sequence"><X size={18} /></button>
        </div>
      </header>

      {detail && where && counts && <div className="flex flex-wrap items-center gap-3 border-b hairline bg-[#171c25] px-5 py-2.5">
        <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${TONES[where.key === 'ready' ? 'good' : where.key === 'directing' || where.key === 'writing' || where.key === 'compiling' ? 'busy' : where.key === 'review' ? 'warn' : where.key === 'direct' || where.key === 'write' ? 'idle' : 'warn']}`}>
          {where.key === 'directing' || where.key === 'writing' || where.key === 'compiling' ? <LoaderCircle size={12} className="animate-spin" /> : where.key === 'ready' ? <Check size={12} /> : where.key === 'direct' || where.key === 'write' ? <Clapperboard size={12} /> : <AlertTriangle size={12} />}
          {where.hint}</span>
        <span className="muted text-[11px]">{detail.bible ? 'Bible ✓' : 'Bible —'} · {detail.plan ? `Plan ${counts.shots} shots ✓` : 'Plan —'} · Prompts {counts.written}/{counts.shots}{detail.review ? ` · Review ${detail.review.content.issues.length} issue${detail.review.content.issues.length === 1 ? '' : 's'}` : ''}</span>
        <span className="ml-auto flex flex-wrap items-center gap-2">
          {(where.key === 'directing' || where.key === 'writing' || where.key === 'compiling') && <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void act('cancel', () => sequenceApi.cancel(sequenceId))}><Square size={12} /> Stop</button>}
          {where.key === 'review' && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" onClick={() => setTab('dialogue')}><Check size={13} /> Review dialogue</button>}
          {(where.key === 'direct' || where.key === 'redirect') && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void run('direct')} title="The director writes the sequence bible and the shot plan, then waits for your review"><WandSparkles size={13} /> Direct sequence</button>}
          {where.key === 'write' && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void run('shots')} title="Writes every shot that is missing, stale or failing its format check, two at a time, then reviews continuity"><Sparkles size={13} /> Write shot prompts ({counts.needWriting})</button>}
          {(where.key === 'write' || where.key === 'ready') && counts.written > 0 && <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void run('review')} title="One pass over all written shots for continuity, speech and timing problems"><RefreshCw size={12} /> Review continuity</button>}
          {(where.key === 'write' || where.key === 'ready') && flagged.length > 0 && <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void reviseFlagged()} title="Rewrites each shot the review marked as wrong, with the review's suggestion as the instruction">Revise flagged shots ({new Set(flagged.map(issue => issue.shot_key)).size})</button>}
        </span>
      </div>}
      {error && <div role="alert" className="flex items-start gap-3 border-b border-[#63363e] bg-[#2b191f] px-5 py-2 text-xs text-[#ffb6bc]"><span className="flex-1 whitespace-pre-wrap">{error}</span><button onClick={() => setError('')} aria-label="Dismiss"><X size={14} /></button></div>}

      {!detail ? <div className="muted p-12 text-sm">Loading sequence…</div> : <>
        {assistantMounted && <CutAssistant sequence={{ id: sequenceId, label }} active={tab === 'assistant'} onResultsChanged={() => { void load(); onChanged() }} />}
        {tab === 'dialogue' && <DialogueTab detail={detail} proxyReady={proxyReady} working={!!working} act={act} run={run} onContinue={() => setTab('direction')} />}
        {tab === 'direction' && <DirectionTab detail={detail} working={!!working || active} act={act} run={run} onOpenShot={key => { setSelectedKey(key); setTab('shots') }} onOpenDialogue={() => setTab('dialogue')} />}
        {tab === 'shots' && <ShotsTab detail={detail} selected={selected} onSelect={setSelectedKey} proxyReady={proxyReady} working={!!working} act={act} run={run} onOpenAssistant={() => setTab('assistant')} />}
      </>}
    </div>
  </div></PlayerProvider>
}

type Act = (label: string, action: () => Promise<SequenceDetail | unknown>) => Promise<boolean>
type Run = (step: RunStep, extra?: { shot_keys?: string[]; instruction?: string }) => Promise<boolean>

function Panel({ title, subtitle, actions, children, className = '' }: { title: string; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return <section aria-label={title} className={`flex min-h-0 flex-col rounded-xl border hairline bg-[#1b202a] ${className}`}>
    <header className="flex shrink-0 flex-wrap items-center gap-2 border-b hairline px-4 py-3">
      <div className="mr-auto min-w-0"><h3 className="text-sm font-semibold">{title}</h3>{subtitle && <p className="muted text-[11px]">{subtitle}</p>}</div>
      {actions}
    </header>
    <div className="scrollbar min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
  </section>
}

/** How long a generation took: seconds below a minute, then minutes and seconds. */
function took(seconds: number) {
  const whole = Math.max(1, Math.round(seconds))
  return whole < 60 ? `${whole} s` : `${Math.floor(whole / 60)} min ${String(whole % 60).padStart(2, '0')} s`
}

function provenance(result: { origin: string; model_id: string | null; created_at: string; completed_by_app?: string | null }) {
  return `${ORIGIN[result.origin] || result.origin}${result.origin === 'model' && result.model_id ? ` (${result.model_id.split('/').pop()})` : ''} · ${result.created_at.slice(0, 16)}`
}

/** A JSON result edited as text; it is saved only if it parses, and the server checks it like a model answer. */
function JsonEditor({ value, onSave, onCancel, saving }: { value: unknown; onSave: (content: unknown) => void; onCancel: () => void; saving: boolean }) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2))
  const [problem, setProblem] = useState('')
  function save() {
    try { const parsed = JSON.parse(text); setProblem(''); onSave(parsed) } catch (err) { setProblem(`Not valid JSON: ${(err as Error).message}`) }
  }
  return <div className="flex min-h-0 flex-1 flex-col gap-2">
    <textarea className="editor-area scrollbar min-h-[24rem] flex-1 resize-y font-mono !text-xs" spellCheck={false} value={text} onChange={event => setText(event.target.value)} aria-label="Edit as JSON" />
    {problem && <p role="alert" className="text-[11px] text-[#ffaaa7]">{problem}</p>}
    <div className="flex justify-end gap-2"><button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" onClick={onCancel}>Cancel</button><button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={saving} onClick={save}><Save size={13} /> Save revision</button></div>
  </div>
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <div><p className="label !mb-1">{label}</p><div className="text-[13px] leading-6 text-[#e8ebf1]">{children}</div></div>
}

function DirectionTab({ detail, working, act, run, onOpenShot, onOpenDialogue }: { detail: SequenceDetail; working: boolean; act: Act; run: Run; onOpenShot: (key: string) => void; onOpenDialogue: () => void }) {
  const { sequence, bible, plan, shots } = detail
  const [title, setTitle] = useState(sequence.title)
  const [instruction, setInstruction] = useState(sequence.instruction)
  const [editingBible, setEditingBible] = useState(false)
  useEffect(() => { setTitle(sequence.title); setInstruction(sequence.instruction) }, [sequence.id, sequence.title, sequence.instruction])
  const dirty = title.trim() !== sequence.title || instruction.trim() !== sequence.instruction
  const boundaries = detail.timeline?.boundaries || []
  // The hard cut after a shot, when speech crosses it or you placed its boundary: only after the last part of a cut.
  const boundaryAfter = (shot: SequenceShot, next?: SequenceShot) => next && next.cut_id !== shot.cut_id ? boundaries.find(item => item.after_cut_id === shot.cut_id && (item.speech || item.fixed)) : undefined
  return <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:overflow-hidden">
    <div className="flex min-h-0 flex-col gap-4 lg:overflow-hidden">
      <Panel title="Brief" subtitle="Your intent for this short video; the director follows it" className="shrink-0"
        actions={dirty && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={working || !title.trim()} onClick={() => void act('brief', () => sequenceApi.update(sequence.id, { title: title.trim(), instruction: instruction.trim() }))}><Save size={13} /> Save</button>}>
        <label className="block"><span className="label">Title</span><input className="field" maxLength={200} value={title} onChange={event => setTitle(event.target.value)} /></label>
        <label className="mt-3 block"><span className="label">Direction</span><textarea className="field min-h-20 resize-y" maxLength={8000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="What this short video should achieve, its tone, what to emphasise…" /></label>
        {detail.missing.length > 0 && <p className="mt-3 text-[11px] leading-5 text-[#ffd59a]">Cuts without a first-frame and video analysis: {detail.missing.join(', ')}. Analyse them before directing.</p>}
      </Panel>
      <Panel title="Sequence bible" subtitle={bible ? provenance(bible) : 'What the sequence is, who is in it, and the rules every shot follows'} className="flex-1"
        actions={bible && !editingBible && <><button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" disabled={working} onClick={() => void run('bible')} title="Write the bible again from the current analysis; the plan and the shots stay until you redo them"><RefreshCw size={12} /> Rewrite</button><button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" onClick={() => setEditingBible(true)}><Pencil size={12} /> Edit</button></>}>
        {!bible ? <p className="muted text-xs leading-5">Not written yet. “Direct sequence” writes it together with the shot plan.</p>
          : editingBible ? <JsonEditor value={bible.content} saving={working} onCancel={() => setEditingBible(false)} onSave={content => void act('bible', () => sequenceApi.revise(bible.id, content)).then(saved => { if (saved) setEditingBible(false) })} />
            : <BibleView bible={bible.content} />}
      </Panel>
    </div>
    <Panel title="Shot plan" subtitle={plan ? `${shots.length} shots, one per cut · ${provenance(plan)}` : 'One shot per cut, along the original audio'}
      actions={plan && <button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" disabled={working} onClick={() => void run('plan')} title="Plan the shots again from the current bible: the director decides every hard cut that falls inside speech again"><RefreshCw size={12} /> Replan</button>}>
      {!plan ? <p className="muted text-xs leading-5">Not planned yet. Every cut becomes one shot, in order; only a cut longer than 15 seconds is divided into parts of one take. Where a hard cut falls inside speech, the director decides whether the boundary stays on the cut (the phrase is divided there, for example when someone interrupts) or moves into the silence just before or after the phrase so that it stays whole in one shot.</p> : <>
        {detail.plan_problems.length > 0 && <p className="mb-3 rounded-lg bg-[#4a3a17] p-2 text-[11px] leading-5 text-[#ffd59a]">The plan no longer fits the cuts: {detail.plan_problems.join('; ')}.</p>}
        {plan.completed_by_app && <p className="mb-3 rounded-lg bg-[#262b35] p-2 text-[11px] leading-5 text-[#c9cfda]">The director's answer was not valid, so the app completed the plan. Boundaries marked “app” were chosen by the app.</p>}
        {plan.content.notes && <p className="muted mb-3 text-xs leading-5">{plan.content.notes}</p>}
        <ol className="space-y-2">{shots.map((shot, index) => {
          const status = shotStatus(shot)
          const boundary = boundaryAfter(shot, shots[index + 1])
          return <Fragment key={shot.key}>
            <li className="rounded-lg border border-[#2f3643] bg-[#161b24] p-3">
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => onOpenShot(shot.key)} className="text-sm font-semibold hover:text-[#e0a96d]" title="Open this shot">Shot {shot.index}</button>
                <span className="font-mono text-[11px] text-[#aeb5c1]">{clock(shot.t0)}–{clock(shot.t1)} · {shot.duration.toFixed(2)} s</span>
                <span className="muted text-[11px]">cut {shot.cut}{shot.scene ? ` · ${shot.scene}` : ''}</span>
                {edgeTags(shot).map(tag => <Tag key={tag.label} tone="busy" title={tag.title}>{tag.label}</Tag>)}
                {shot.trim_seconds !== null && <Tag tone="idle" title={`Generated at ${shot.generate_seconds} s and trimmed to ${shot.trim_seconds} s`}>short</Tag>}
                {shot.fixed_voices.length > 0 && <Tag tone="idle" title="You decided how some speakers are heard in this shot">your voices</Tag>}
                <span className="ml-auto"><Tag tone={status.tone}>{status.label}</Tag></span>
              </div>
              <p className="mt-2 text-[13px] leading-5">{shot.intent || <span className="muted">No intent written.</span>}</p>
              <p className="muted mt-1 text-[11px]">Initial frame: {frameText(shot)}{shot.lines.length ? ` · ${shot.lines.length} spoken line${shot.lines.length > 1 ? 's' : ''}` : ' · no speech'}</p>
            </li>
            {boundary && <BoundaryRow boundary={boundary} working={working} onOpenDialogue={onOpenDialogue} onChoose={choice => void act('boundary', () => sequenceApi.setBoundary(sequence.id, boundary.after_cut_id, choice))} />}
          </Fragment>
        })}</ol>
        <p className="muted mt-3 text-[11px] leading-5">Moving a boundary changes the two shots beside it: their prompts must be written again. Every other shot keeps its prompt. For a different intent, ask the assistant or edit and rewrite the shot.</p>
      </>}
    </Panel>
  </div>
}

/** A hard cut that falls inside speech, between two shots of the plan: where its boundary sits, who chose it and why, and the other positions. */
function BoundaryRow({ boundary, working, onChoose, onOpenDialogue }: { boundary: SequenceBoundary; working: boolean; onChoose: (choice: BoundaryChoice) => void; onOpenDialogue: () => void }) {
  if (boundary.fixed) {
    const shift = boundary.fixed.shift
    return <li className="ml-6 rounded-lg border border-[#7d6542] bg-[#1d1a14] px-3 py-2" aria-label={`Hard cut between cut ${boundary.after_cut} and cut ${boundary.before_cut}, placed by you`}>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="font-mono text-[#aeb5c1]">Hard cut {clock(boundary.t)}</span>
        <span className="min-w-0 flex-1 text-[#e8d2b0]">Placed by you at {clock(boundary.fixed.t)}{Math.abs(shift) > 0.002 ? ` (${shift < 0 ? '−' : '+'}${Math.abs(shift).toFixed(2)} s)` : ''}{boundary.speech ? ` · ${boundary.speech.lines.join(' / ')}` : ''}</span>
        <button type="button" className="btn btn-quiet !px-2 !py-1 !text-[11px]" onClick={onOpenDialogue} title="Boundaries you placed are moved on the Dialogue tab's timeline">Change in Dialogue</button>
      </div>
    </li>
  }
  const speech = boundary.speech!
  const by = deciderLabel(boundary.by)
  const refused = Object.entries(boundary.unavailable).map(([choice, why]) => `${choice.replace('_', ' ')}: ${why}`).join('\n')
  return <li className="ml-6 rounded-lg border border-dashed border-[#3a4150] bg-[#12161e] px-3 py-2" aria-label={`Hard cut between cut ${boundary.after_cut} and cut ${boundary.before_cut}`}>
    <div className="flex flex-wrap items-center gap-2 text-[11px]">
      <span className="font-mono text-[#aeb5c1]">Hard cut {clock(boundary.t)}</span>
      <span className="min-w-0 flex-1 text-[#c9cfda]"><span className="font-semibold">{speech.character_name || speech.speaker_label || 'Unknown speaker'}</span> · {speech.lines.join(' / ')}: “{speech.words_before}” <span className="text-[#e0a96d]">|</span> “{speech.words_after}”</span>
      <select className="field !w-auto !px-2 !py-1 !text-[11px]" aria-label={`Boundary at the hard cut after cut ${boundary.after_cut}`} title={refused || undefined}
        disabled={working || !boundary.choice} value={boundary.choice ?? ''} onChange={event => onChoose(event.target.value as BoundaryChoice)}>
        {!boundary.choice && <option value="">Not planned</option>}
        {boundaryChoices(boundary).map(item => <option key={item.choice} value={item.choice}>{item.label}</option>)}
      </select>
    </div>
    {(by || boundary.reason) && <p className="muted mt-1 text-[11px] leading-5">{by === 'you' ? 'Your choice' : by ? `Chosen by the ${by}` : 'Reason'}{boundary.reason ? `: ${boundary.reason}` : ''}</p>}
  </li>
}

function Tag({ tone, title, children }: { tone: Tone; title?: string; children: ReactNode }) {
  return <span title={title} className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${TONES[tone]}`}>{children}</span>
}

function BibleView({ bible }: { bible: Bible }) {
  return <div className="space-y-4">
    <Field label="Logline">{bible.logline}</Field>
    <Field label="Synopsis">{bible.synopsis}</Field>
    <div className="grid gap-4 sm:grid-cols-2"><Field label="Spoken language">{bible.spoken_language}</Field><Field label="Pacing">{bible.pacing || '—'}</Field></div>
    {bible.arc && <Field label="Arc">{bible.arc}</Field>}
    <Field label="Scenes"><ul className="space-y-1.5">{bible.scenes.map(scene => <li key={scene.id}><span className="font-semibold">{scene.id}</span> · cuts {scene.from_cut}–{scene.to_cut} · {scene.location}<span className="muted"> · {[scene.time_of_day, scene.lighting, scene.mood].filter(Boolean).join(' · ')}{scene.scenario ? ` · scenario “${scene.scenario}”` : ''}</span></li>)}</ul></Field>
    <Field label="Characters">{bible.characters.length ? <ul className="space-y-2">{bible.characters.map(person => <li key={person.name}><span className="font-semibold">{person.name}</span>{person.role ? <span className="muted"> · {person.role}</span> : null}<br /><span className="text-xs leading-5 text-[#c9cfda]">{person.anchors}{person.outfit ? ` · wears ${person.outfit}` : ''}</span></li>)}</ul> : 'No named character is on screen.'}</Field>
    <Field label="Voices">{bible.voices.length ? <ul className="space-y-1.5">{bible.voices.map(voice => <li key={voice.speaker_label}><span className="font-mono text-xs">{voice.speaker_label}</span> → <span className="font-semibold">{voice.character || (voice.kind === 'narrator' ? 'Narrator' : 'unknown')}</span><span className="muted"> · {voice.source === 'reviewed' ? 'your reviewed dialogue' : voice.source === 'transcript' ? 'your transcript link' : `director, ${voice.confidence}`}{voice.reason && voice.source === 'director' ? ` · ${voice.reason}` : ''}</span></li>)}</ul> : 'Nobody speaks in this range.'}</Field>
    {bible.continuity.length > 0 && <Field label="Continuity rules"><ul className="list-disc space-y-1 pl-5">{bible.continuity.map((rule, index) => <li key={index}>{rule}</li>)}</ul></Field>}
    {bible.open_questions.length > 0 && <Field label="Open questions"><ul className="list-disc space-y-1 pl-5 text-[#ffd59a]">{bible.open_questions.map((question, index) => <li key={index}>{question}</li>)}</ul></Field>}
  </div>
}

function ShotsTab({ detail, selected, onSelect, proxyReady, working, act, run, onOpenAssistant }: {
  detail: SequenceDetail; selected?: SequenceShot; onSelect: (key: string) => void; proxyReady: boolean; working: boolean; act: Act; run: Run; onOpenAssistant: () => void
}) {
  // Whether the opening still covers the shared video; kept while the user moves from shot to shot.
  const [showFrame, setShowFrame] = useState(!proxyReady)
  const player = usePlayer()
  // A seek into another shot waits here until that shot's detail has set the player's range (which moves to its start).
  const pending = useRef<Pending | null>(null)
  const takePending = useCallback((key: string) => {
    const want = pending.current
    if (!want || want.key !== key) return null
    // Cleared after the effect has run (twice under StrictMode), not inside it.
    setTimeout(() => { if (pending.current === want) pending.current = null }, 0)
    return want
  }, [])
  const pick = useCallback((shot: SequenceShot, seconds: number, play: boolean) => {
    const time = shot.start_time + seconds
    setShowFrame(false)
    if (shot.key === selected?.key) { if (play) player.playFrom(time); else player.seek(time) }
    else { pending.current = { key: shot.key, time, play }; onSelect(shot.key) }
  }, [player, selected?.key, onSelect])
  useEffect(() => { if (selected) document.getElementById(`shot-nav-${selected.key}`)?.scrollIntoView({ block: 'nearest' }) }, [selected?.key])
  if (!detail.shots.length) return <div className="muted p-12 text-sm leading-6">There is no shot plan yet. Direct the sequence on the Direction tab first.</div>
  return <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-5 lg:overflow-hidden">
    <SequenceTimeline detail={detail} selectedKey={selected?.key} onPick={pick} />
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[17rem_minmax(0,1fr)] lg:overflow-hidden">
    <nav aria-label="Shots" className="scrollbar min-h-0 space-y-1.5 lg:overflow-y-auto lg:pr-1">
      {detail.shots.map(shot => {
        const status = shotStatus(shot)
        const issues = issuesFor(detail.review?.content.issues, shot)
        return <button key={shot.key} id={`shot-nav-${shot.key}`} type="button" onClick={() => onSelect(shot.key)} aria-current={selected?.key === shot.key} className={`flex w-full items-center gap-2.5 rounded-lg border p-2 text-left ${selected?.key === shot.key ? 'border-[#e0a96d] bg-[#2a2319]' : 'border-[#2f3643] bg-[#161b24] hover:border-[#5a4a35]'}`}>
          <img src={shotFrameUrl(detail.sequence.id, shot.key)} alt="" loading="lazy" className="h-12 w-20 shrink-0 rounded bg-black object-cover" />
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-1.5"><span className="text-xs font-semibold">Shot {shot.index}</span><Tag tone={status.tone}>{status.label}</Tag>{issues.length > 0 && <Tag tone={issues.some(issue => issue.severity === 'fix') ? 'bad' : 'warn'}>{issues.length} issue{issues.length > 1 ? 's' : ''}</Tag>}</span>
            <span className="muted block truncate font-mono text-[10px]">{clock(shot.t0)}–{clock(shot.t1)} · {shot.duration.toFixed(1)} s{shot.result?.generation_seconds ? ` · written in ${took(shot.result.generation_seconds)}` : ''}</span>
            <span className="muted block truncate text-[10px]">{shot.intent}</span>
          </span>
        </button>
      })}
    </nav>
    {selected && <ShotDetail key={selected.key} detail={detail} shot={selected} proxyReady={proxyReady} showFrame={showFrame} onShowFrame={setShowFrame} takePending={takePending} working={working} act={act} run={run} onOpenAssistant={onOpenAssistant} />}
    </div>
  </div>
}

/** A seek (or play) requested for a shot that is not selected yet; `time` is in episode seconds. */
type Pending = { key: string; time: number; play: boolean }

function ShotDetail({ detail, shot, proxyReady, showFrame, onShowFrame, takePending, working, act, run, onOpenAssistant }: {
  detail: SequenceDetail; shot: SequenceShot; proxyReady: boolean; showFrame: boolean; onShowFrame: (value: boolean) => void
  takePending: (key: string) => Pending | null; working: boolean; act: Act; run: Run; onOpenAssistant: () => void
}) {
  const player = usePlayer()
  // Selecting a shot only moves the playhead of the video that is already loaded; a click on the timeline then seeks or plays inside it.
  useEffect(() => {
    player.setRange({ start: shot.start_time, end: shot.end_time }, true)
    const want = takePending(shot.key)
    if (want) { if (want.play) player.playFrom(want.time); else player.seek(want.time) }
  }, [player, shot.key, shot.start_time, shot.end_time, takePending])
  // A timecode of the prompt is a second of this shot: play the source from there.
  const playAt = useCallback((seconds: number) => { onShowFrame(false); player.playFrom(shot.start_time + seconds) }, [player, shot.start_time, onShowFrame])
  const result = shot.result
  const [view, setView] = useState<'read' | 'raw' | 'edit'>('read')
  const [instruction, setInstruction] = useState('')
  // The versions below the prompt; opened by a rewrite so the change is in view when it lands.
  const [historyOpen, setHistoryOpen] = useState(false)
  const issues = issuesFor(detail.review?.content.issues, shot)
  const dialogue: ShotDialogue[] = result?.derived.dialogue || shot.lines
  const sections = useMemo(() => result?.prompt ? promptSections(result.prompt) : [], [result?.prompt])
  const status = shotStatus(shot)
  const busy = working || !!shot.job
  const canQueue = !busy && !!detail.bible
  const instructionField = useRef<HTMLTextAreaElement>(null)
  const queue = () => void run('shots', { shot_keys: [shot.key], instruction: instruction.trim() }).then(queued => { if (queued) { setInstruction(''); if (result) setHistoryOpen(true) } })
  const note = result ? versionNote(result) : null
  // The rewrite instruction stays pinned under the scrolling shot details.
  return <div className="flex min-h-0 flex-col gap-3 lg:overflow-hidden">
  <div className="scrollbar flex min-h-0 flex-1 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
    <div className="grid shrink-0 gap-4 xl:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <div>
        <PlayerSlot className="aspect-video">
          {(showFrame || !proxyReady) && <img src={shotFrameUrl(detail.sequence.id, shot.key)} alt={`Initial frame of shot ${shot.index}`} className="absolute inset-0 z-10 h-full w-full bg-black object-contain" />}
          {proxyReady && <ShotClock shot={shot} sequenceStart={detail.sequence.start_time} />}
        </PlayerSlot>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" disabled={!proxyReady} onClick={() => { onShowFrame(false); player.playRange(shot.start_time, shot.end_time) }} title="Play this shot of the source video"><Play size={12} /> Play shot</button>
          <button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" disabled={!proxyReady} onClick={() => onShowFrame(!showFrame)}>{showFrame ? 'Show source video' : 'Show initial frame'}</button>
          <span className="muted text-[11px]">Initial frame: {frameText(shot)}{shot.frame.strategy === 'frame_at_time' ? ' (described from the motion, not observed)' : ''}</span>
        </div>
      </div>
      <div className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center gap-2"><h3 className="text-base font-semibold">Shot {shot.index}</h3><Tag tone={status.tone}>{status.label}</Tag>{edgeTags(shot).map(tag => <Tag key={tag.label} tone="busy" title={tag.title}>{tag.label}</Tag>)}</div>
        <p className="font-mono text-xs text-[#aeb5c1]">{clock(shot.t0)}–{clock(shot.t1)} in the sequence · {shot.duration.toFixed(2)} s · cut {shot.cut}{shot.source_cuts.length > 1 ? ` (audio of cuts ${shot.source_cuts.join(', ')})` : ''}</p>
        {shot.trim_seconds !== null && <p className="text-[11px] leading-5 text-[#c9cfda]">Shorter than H3's minimum: generate {shot.generate_seconds} s (the prompt ends on a hold) and trim to {shot.trim_seconds} s.</p>}
        <p className="text-[13px] leading-5">{shot.intent}</p>
        {[shot.opens, shot.closes].filter(edge => edge.reason).map((edge, index) => <p key={index} className="muted text-xs leading-5">{edge === shot.opens ? 'Opens' : 'Closes'}: {edge.reason}</p>)}
        <div>
          <p className="label !mb-1">Dialogue</p>
          {dialogue.length ? <ul className="space-y-1">{dialogue.map(line => <li key={line.line} className="text-xs leading-5">
            <span className="font-mono text-[10px] text-[#8b93a3]">{line.line} {line.t0.toFixed(2)}–{line.t1.toFixed(2)}</span> <span className="font-semibold">{line.character_name || line.speaker_label || 'Unknown'}</span>
            {line.delivery && <Tag tone={line.delivery === 'on_screen' ? 'good' : 'idle'} title={line.fixed_by_user ? 'You decided this' : 'Decided by the director'}>{deliveryLabel(line.delivery)}{line.subject ? ` · S${line.subject}` : ''}</Tag>} <span className="text-[#e8ebf1]">{line.text}</span>
            {(line.crosses_start || line.crosses_end) && <span className="muted"> ({line.crosses_start ? 'continues from the previous shot' : 'continues in the next shot'})</span>}
          </li>)}</ul> : <p className="muted text-xs">Nobody speaks during this shot.</p>}
        </div>
      </div>
    </div>

    {shot.speakers.length > 0 && <VoicesPanel detail={detail} shot={shot} working={working} act={act} />}
    {shot.stale.length > 0 && <p className="shrink-0 rounded-lg bg-[#4a3a17] p-2.5 text-xs leading-5 text-[#ffd59a]">{staleText(shot.stale)}</p>}
    {issues.length > 0 && <div className="shrink-0 rounded-lg border border-[#5b4a2a] bg-[#241d12] p-3">
      <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-[#ffd59a]"><MessageSquare size={13} /> Review</p>
      <ul className="space-y-1.5">{issues.map((issue, index) => <li key={index} className="text-xs leading-5"><Tag tone={issue.severity === 'fix' ? 'bad' : 'warn'}>{issue.severity}</Tag> {issue.problem}{issue.suggestion && <span className="muted"> → {issue.suggestion}</span>}</li>)}</ul>
      <button type="button" className="mt-2 !text-[11px] text-[#e0a96d] underline" onClick={() => { setInstruction(issueInstruction(issues)); instructionField.current?.focus() }}>Use as the instruction below</button>
    </div>}

    <section className="shrink-0 rounded-xl border hairline bg-[#1b202a]" aria-label="H3 prompt">
      <header className="flex flex-wrap items-center gap-2 border-b hairline px-4 py-3">
        <div className="mr-auto min-w-0"><h3 className="text-sm font-semibold">MiniMax H3 prompt</h3><p className="muted text-[11px]">{result ? `${provenance(result)}${result.generation_seconds ? ` · written in ${took(result.generation_seconds)}` : ''}${result.check?.stats ? ` · ${result.check.stats.description_words ?? '?'} words · ${result.check.stats.chars ?? '?'} characters` : ''}` : 'Not written yet'}</p></div>
        {result && <>
          <div className="flex gap-1 rounded-lg border hairline p-0.5" role="tablist" aria-label="Prompt view">{([['read', 'Read'], ['raw', 'Text'], ['edit', 'Edit direction']] as const).map(([value, text]) =>
            <button key={value} role="tab" aria-selected={view === value} onClick={() => setView(value)} className={`rounded-md px-2.5 py-1 !text-[11px] font-semibold ${view === value ? 'bg-[#3a2f1f] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{text}</button>)}</div>
          {result.prompt && <CopyButton text={result.prompt} label="Copy prompt" />}
        </>}
      </header>
      <div className="p-4">
        {!result ? <p className="muted text-xs leading-5">{shot.job ? 'The director is writing this shot…' : 'This shot has no prompt yet. “Write shot prompts” writes every missing shot; the button below writes only this one.'}</p> : <>
          {note && <p className="mb-3 border-l-2 border-[#e0a96d] pl-2.5 text-xs leading-5 text-[#e8ebf1]">{note}</p>}
          {result.check && !result.check.ok && <ul className="mb-3 list-disc space-y-1 rounded-lg bg-[#3d2025] p-2.5 pl-6 text-xs leading-5 text-[#ffaaa7]">{result.check.errors.map((item, index) => <li key={index}>{item}</li>)}</ul>}
          {!!result.check?.warnings.length && <ul className="mb-3 list-disc space-y-1 rounded-lg bg-[#262b35] p-2.5 pl-6 text-[11px] leading-5 text-[#c9cfda]">{result.check.warnings.map((item, index) => <li key={index}>{item}</li>)}</ul>}
          {view === 'edit' ? <>
            <p className="muted mb-2 text-[11px] leading-5">You edit the direction; the app renders the prompt again and checks it. Write names, not labels: labels, timecodes and spoken words are added for you.</p>
            <JsonEditor value={result.content} saving={working} onCancel={() => setView('read')} onSave={content => void act('shot', () => sequenceApi.revise(result.id, content)).then(saved => { if (saved) setView('read') })} />
          </> : view === 'raw' ? <pre className="scrollbar max-h-[36rem] overflow-auto whitespace-pre-wrap rounded-lg border border-[#2b3240] bg-[#0f131a] p-3 font-mono text-[11px] leading-5 text-[#c9cfda]">{(result.prompt || '').split('\n').map((line, index) => <Fragment key={index}>{index > 0 && '\n'}<PromptLine line={line} shot={shot} disabled={!proxyReady} onPlay={playAt} /></Fragment>)}</pre>
            : <div className="space-y-4">{sections.map(section => <div key={section.name}><p className="label !mb-1">{section.title}</p><div className="space-y-1.5 text-[13px] leading-6 text-[#e8ebf1]">{section.lines.map((line, index) => <p key={index} className={/^\d{2}:\d{2}/.test(line) ? 'border-l-2 border-[#5a4a35] pl-2.5' : ''}><PromptLine line={line} shot={shot} disabled={!proxyReady} onPlay={playAt} /></p>)}</div></div>)}</div>}
          {result.derived.references && <p className="muted mt-4 text-[11px] leading-5">References to attach, in this order: {result.derived.references.map(reference => `${reference.label} ${reference.role === 'initial_frame' ? 'initial frame' : reference.role === 'character_sheet' ? `sheet of ${reference.name}` : reference.role === 'scenario' ? `scenario “${reference.name}”` : `audio ${result.derived.audio_file || ''}`}`).join(' · ')}</p>}
          {result.derived.still_prompt && <details className="mt-3 rounded-lg border hairline bg-[#171c25] text-xs"><summary className="cursor-pointer px-3 py-2 font-semibold text-[#aeb5c1] hover:text-white">Still image prompt for the initial frame</summary><div className="border-t hairline p-3"><pre className="whitespace-pre-wrap font-mono text-[11px] leading-5 text-[#c9cfda]">{result.derived.still_prompt}</pre><div className="mt-2"><CopyButton text={result.derived.still_prompt} label="Copy still prompt" /></div></div></details>}
          {result.content.open_questions.length > 0 && <div className="mt-3"><p className="label !mb-1">Open questions</p><ul className="list-disc space-y-1 pl-5 text-xs leading-5 text-[#ffd59a]">{result.content.open_questions.map((question, index) => <li key={index}>{question}</li>)}</ul></div>}
          <PromptHistory sequenceId={detail.sequence.id} shot={shot} open={historyOpen} onToggle={setHistoryOpen} onOpenAssistant={onOpenAssistant} />
        </>}
      </div>
    </section>
  </div>
  <footer className="sticky bottom-0 z-30 flex shrink-0 flex-wrap items-end gap-2 rounded-xl border hairline bg-[#1b202a] p-3 shadow-[0_-6px_18px_rgba(0,0,0,0.35)]" aria-label={result ? 'Rewrite this shot' : 'Write this shot'}>
    <label className="min-w-0 flex-1"><span className="label !mb-1 flex items-center justify-between gap-2">Instruction for a rewrite (optional, remembered for every later rewrite of this shot)<span className="font-normal normal-case tracking-normal text-[#8b93a3]">Ctrl+Enter queues it</span></span>
      <textarea ref={instructionField} className="field max-h-48 min-h-[3.25rem] resize-y !py-2 !text-xs leading-5" rows={2} maxLength={4000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: she is off screen for the first line; keep the camera static"
        onKeyDown={event => {
          if (event.key !== 'Enter' || !event.ctrlKey || event.altKey || event.metaKey || event.shiftKey || event.nativeEvent.isComposing) return
          event.preventDefault()
          if (!event.repeat && canQueue) queue()
        }} /></label>
    <button type="button" className="btn btn-quiet !py-2 !text-xs" disabled={!canQueue} onClick={queue} title="The director writes this shot again from the current bible, plan and analysis, following your instruction and the ones you gave before (Ctrl+Enter in the instruction)"><Sparkles size={13} /> {result ? 'Rewrite this shot' : 'Write this shot'}</button>
  </footer>
  </div>
}

/** A line of the prompt with each timecode as a button that plays the source from that second of the shot. */
function PromptLine({ line, shot, disabled, onPlay }: { line: string; shot: SequenceShot; disabled: boolean; onPlay: (seconds: number) => void }) {
  return <>{timecodeSegments(line).map((part, index) => {
    if (part.t === null) return part.text
    // The hold that pads a shot shorter than H3's minimum has no source picture behind it.
    const beyond = part.t > shot.duration + 0.0005
    return <button key={index} type="button" disabled={disabled || beyond} onClick={() => onPlay(part.t!)}
      title={beyond ? `${part.text} is past the end of this shot (${timecode(shot.duration)}): the generated hold, trimmed away` : `Play the source from ${part.text} of this shot (episode ${clock(shot.start_time + part.t)})`}
      className={`rounded px-0.5 font-mono tabular-nums ${beyond ? 'cursor-default text-[#8b93a3]' : 'bg-[#123840] text-[#a6eef7] hover:bg-[#1d5967] hover:text-white disabled:cursor-default disabled:bg-transparent disabled:text-inherit'}`}>{part.text}</button>
  })}</>
}

/** The playhead on the shot's own clock (the prompt's MM:SS.mmm), over the video; only this badge re-renders while it plays. */
function ShotClock({ shot, sequenceStart }: { shot: SequenceShot; sequenceStart: number }) {
  const [at, inSequence] = usePlayerValue(time => `${timecode(Math.min(shot.duration, Math.max(0, time - shot.start_time)))}|${clock(time - sequenceStart)}`).split('|')
  return <div className="pointer-events-none absolute left-2 top-2 z-20 rounded-md bg-black/75 px-2 py-1 font-mono tabular-nums text-white" title="Playhead from the start of this shot, as the prompt writes its times" aria-label={`Playhead at ${at} of the shot`}>
    <span className="text-sm">{at}</span><span className="text-[11px] text-[#aeb5c1]"> / {timecode(shot.duration)}</span>
    <span className="block text-[10px] text-[#aeb5c1]">sequence {inSequence}</span>
  </div>
}

/** Every version of a shot's prompt below the current one, newest first: what caused it and what changed against the version before. */
function PromptHistory({ sequenceId, shot, open, onToggle, onOpenAssistant }: {
  sequenceId: string; shot: SequenceShot; open: boolean; onToggle: (open: boolean) => void; onOpenAssistant: () => void
}) {
  const [history, setHistory] = useState<ShotHistory | null>(null)
  const [error, setError] = useState('')
  const resultId = shot.result?.id
  // Loaded only while open, and again when a new version lands or a rewrite is queued or starts.
  useEffect(() => {
    if (!open) return
    let live = true
    sequenceApi.shotHistory(sequenceId, shot.key).then(value => { if (live) { setHistory(value); setError('') } }, (err: Error) => { if (live) setError(err.message) })
    return () => { live = false }
  }, [open, sequenceId, shot.key, resultId, shot.job])
  const count = history?.revisions.length ?? 0
  const waiting = history?.proposals.length ?? 0
  return <details open={open} onToggle={event => onToggle(event.currentTarget.open)} className="group/history mt-4 rounded-lg border hairline bg-[#171c25] text-xs">
    <summary className="flex cursor-pointer list-none items-center gap-1.5 px-3 py-2 font-semibold text-[#aeb5c1] hover:text-white">
      <ChevronDown size={14} className="-rotate-90 transition-transform group-open/history:rotate-0" aria-hidden /><HistoryIcon size={13} aria-hidden />
      Instructions and changes{count ? ` · ${count} version${count > 1 ? 's' : ''}` : ''}
      {waiting > 0 && <Tag tone="busy">{waiting} from the assistant, not applied</Tag>}
    </summary>
    <div className="space-y-2 border-t hairline p-3">
      {error && <p role="alert" className="text-[11px] text-[#ffaaa7]">{error}</p>}
      {!history && !error && <p className="muted flex items-center gap-1.5 text-[11px]"><LoaderCircle size={12} className="animate-spin" /> Loading the versions…</p>}
      {history?.pending && <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[#5a4a8c] bg-[#1a1730] px-3 py-2 text-[#d6c8ff]">
        <LoaderCircle size={13} className="animate-spin" /><span className="font-semibold">{history.pending.status === 'running' ? 'Rewriting now' : 'Rewrite queued'}</span>
        <span className="min-w-0 flex-1 text-[#e8ebf1]">{history.pending.instruction ? `with your instruction: “${history.pending.instruction}”` : history.pending.trigger === 'voices' ? 'after you changed the voices' : history.remembered.length ? 'following your earlier instructions' : 'without an instruction'}</span>
      </div>}
      {!!history?.remembered.length && <div className="rounded-lg border border-[#3d5a45] bg-[#142019] px-3 py-2 text-[#cfe8d6]">
        <p className="font-semibold">Remembered for every rewrite of this shot</p>
        <ol className="mt-1 list-decimal space-y-0.5 pl-4 text-[#e8ebf1]">{history.remembered.map((text, index) => <li key={index}>{text}</li>)}</ol>
        <p className="muted mt-1 text-[11px]">Every rewrite follows all of these. Where two differ, the later one wins, so write a new instruction to change one.</p>
      </div>}
      {history?.proposals.map(proposal => <ChangeRow key={proposal.id} open={false} diff={proposal.diff} prompt={proposal.prompt}
        heading={<><Tag tone={proposal.stale ? 'warn' : 'busy'}>{proposal.stale ? 'Outdated' : 'Not applied'}</Tag><span className="font-semibold text-[#e8ebf1]">Proposed in the Assistant tab</span><Size diff={proposal.diff} /><span className="muted ml-auto font-mono text-[10px]">{proposal.created_at.slice(0, 16)}</span></>}
        why={<Why asked={proposal.request} said={proposal.summary} />}
        actions={<button type="button" className="!text-[11px] text-[#c3aeff] underline" onClick={onOpenAssistant}>{proposal.stale ? 'Open the Assistant to ask again' : 'Open the Assistant to apply or discard'}</button>} />)}
      {history?.revisions.map((revision, index) => <ChangeRow key={revision.id} open={index === 0} diff={revision.diff} prompt={revision.prompt}
        heading={<><span className="font-mono text-[11px] font-semibold text-[#e8ebf1]">v{revision.number}</span>{index === 0 && <Tag tone="good">Current</Tag>}
          <span className="font-semibold text-[#e8ebf1]">{revisionLabel(revision)}</span><Size diff={revision.diff} />{revision.check_ok === false && <Tag tone="bad">format errors</Tag>}
          <span className="muted ml-auto font-mono text-[10px]">{revision.created_at.slice(0, 16)}{revision.generation_seconds ? ` · ${took(revision.generation_seconds)}` : ''}</span></>}
        why={revision.assistant ? <Why asked={revision.assistant.request} said={revision.assistant.summary} />
          : revision.instruction ? <Why instruction={revision.instruction} review={revision.instruction_source === 'review'} earlier={revision.earlier_instructions} />
          : revision.trigger === 'voices' ? <p className="muted text-[11px]">You changed how a speaker is heard in this shot; the director wrote it again around your voices.</p>
          : revision.number > 1 && revision.origin === 'model' ? <p className="muted text-[11px]">{revision.earlier_instructions ? `No new instruction: the director wrote it again from the current bible, plan and analysis, following your ${revision.earlier_instructions === 1 ? 'earlier instruction' : `${revision.earlier_instructions} earlier instructions`}.` : 'No instruction: the director wrote it again from the current bible, plan and analysis.'}</p> : null} />)}
      {history && count <= 1 && !history.pending && !waiting && <p className="muted text-[11px]">This is the first version. Rewrites, your edits and changes from the Assistant tab will appear here with what changed.</p>}
    </div>
  </details>
}

/** One version (or proposed version) of a prompt: what caused it, then its change as a diff, or the whole prompt on request. */
function ChangeRow({ heading, why, diff, prompt, open, actions }: { heading: ReactNode; why?: ReactNode; diff: DiffSegment[] | null; prompt: string; open: boolean; actions?: ReactNode }) {
  const [full, setFull] = useState(false)
  const changed = !!diff?.some(part => part.op !== 'equal')
  return <details open={open} className="group/change rounded-lg border border-[#2b3240] bg-[#11151d]">
    <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2 px-3 py-2 hover:bg-[#161b24]"><ChevronDown size={13} className="-rotate-90 text-[#8b93a3] transition-transform group-open/change:rotate-0" aria-hidden />{heading}</summary>
    <div className="space-y-2 border-t border-[#2b3240] px-3 py-2.5">
      {why}
      {full || changed ? <div className={`scrollbar overflow-y-auto whitespace-pre-wrap rounded-lg border border-[#2b3240] bg-[#0f131a] p-2.5 font-mono text-[11px] leading-5 ${full ? 'max-h-[28rem] text-[#c9cfda]' : 'max-h-72 text-[#9ea6b5]'}`}>
        {full ? prompt : <DiffText segments={diff || []} />}
      </div> : <p className="muted text-[11px]">{diff ? 'The prompt did not change.' : 'The first version of this prompt.'}</p>}
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" className="!text-[11px] text-[#e0a96d] underline" onClick={() => setFull(!full)}>{full ? (changed ? 'Show changes' : 'Hide prompt') : changed ? 'Show full prompt' : 'Show prompt'}</button>
        <CopyButton text={prompt} label="Copy this version" />
        {actions}
      </div>
    </div>
  </details>
}

/** What the user asked for: a rewrite instruction, or a chat request and the assistant's summary of its change. */
function Why({ instruction, review, earlier, asked, said }: { instruction?: string; review?: boolean; earlier?: number | null; asked?: string | null; said?: string }) {
  return <div className="space-y-1 text-[11px] leading-5">
    {instruction && <p className="border-l-2 border-[#e0a96d] pl-2.5 text-[#e8ebf1]"><span className="muted">{review ? 'Suggestion of the review (used once): ' : 'Your instruction: '}</span>“{instruction}”</p>}
    {!!earlier && <p className="muted pl-2.5">Also followed your {earlier === 1 ? 'earlier instruction' : `${earlier} earlier instructions`} for this shot.</p>}
    {asked && <p className="border-l-2 border-[#e0a96d] pl-2.5 text-[#e8ebf1]"><span className="muted">You asked: </span>“{asked}”</p>}
    {said && <p className="border-l-2 border-[#5a4a8c] pl-2.5 text-[#d6c8ff]"><span className="muted">Assistant: </span>{said}</p>}
  </div>
}

function Size({ diff }: { diff: DiffSegment[] | null }) {
  if (!diff) return null
  const size = changeSize(diff)
  return <Tag tone="idle" title="Words added and removed against the version before">{size.added || size.removed ? `+${size.added} −${size.removed} words` : 'no change'}</Tag>
}

function frameText(shot: SequenceShot) {
  if (shot.frame.strategy === 'cut_first_frame') return `first frame of cut ${shot.frame.cut}`
  return shot.frame.continues_previous_shot ? `continues the take of the previous part of cut ${shot.frame.cut}` : `a moment into cut ${shot.frame.cut}, after the phrase the previous shot finished`
}

/** How each speaker of a shot is heard, as ranges over the shot; a change is saved on the plan and rewrites the shot. */
function VoicesPanel({ detail, shot, working, act }: { detail: SequenceDetail; shot: SequenceShot; working: boolean; act: Act }) {
  const player = usePlayer()
  const written = shot.result?.derived.voices ?? shot.result?.content.voices
  const rewriting = shot.job ? (shot.job === 'running' ? 'The director is rewriting this shot with your voices…' : 'A rewrite with your voices is queued…') : ''
  const save = (speaker: string, segments: { from: number; delivery: Delivery }[] | null) => void act('voices', () => sequenceApi.setVoices(detail.sequence.id, shot.key, speaker, segments))
  return <section className="shrink-0 rounded-xl border hairline bg-[#1b202a] p-4" aria-label="Voices in this shot">
    <div className="mb-3 flex flex-wrap items-baseline gap-2">
      <h3 className="text-sm font-semibold">Voices in this shot</h3>
      <p className="muted text-[11px]">How each speaker is heard in this take. Your choice is kept and the shot is rewritten around it.</p>
      {rewriting && <span className="ml-auto flex items-center gap-1.5 text-[11px] text-[#d6c8ff]"><LoaderCircle size={12} className="animate-spin" />{rewriting}</span>}
    </div>
    <ul className="space-y-3">{shot.speakers.map(speaker => {
      const ranges = speakerRanges(written, shot.fixed_voices || [], speaker, shot.duration)
      const whole = ranges.length === 1 ? ranges[0].delivery : null
      const own = hasOwnVoices(shot, speaker)
      return <li key={speaker} className="rounded-lg border border-[#2f3643] bg-[#161b24] p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold">{speaker}</span>
          <span className="muted text-[10px]">{own ? 'set by you' : ranges.length ? 'decided by the director' : 'not decided yet'}</span>
          <span className="ml-auto flex flex-wrap gap-1" role="group" aria-label={`How ${speaker} is heard for the whole shot`}>
            {DELIVERIES.map(item => <button key={item.value} type="button" title={item.hint} disabled={working} aria-pressed={whole === item.value}
              onClick={() => save(speaker, [{ from: 0, delivery: item.value }])}
              className={`rounded-md border px-2 py-1 text-[11px] font-semibold ${whole === item.value ? 'border-[#e0a96d] bg-[#3a2f1f] text-white' : 'border-[#2f3643] text-[#aeb5c1] hover:text-white'}`}>{item.label}</button>)}
          </span>
        </div>
        <div className="relative mt-2 flex h-5 overflow-hidden rounded bg-[#0f131a]" aria-hidden="true">
          {ranges.map(range => <div key={range.from} className={`${DELIVERY_COLOURS[range.delivery]} flex items-center overflow-hidden border-r border-[#0f131a] px-1 text-[10px] font-semibold text-white`}
            style={{ width: `${(range.to - range.from) / shot.duration * 100}%` }} title={`${clock(range.from)}–${clock(range.to)} · ${deliveryLabel(range.delivery)}${range.mine ? ' (yours)' : ''}`}>
            <span className="truncate">{range.mine ? '● ' : ''}{deliveryLabel(range.delivery)}</span></div>)}
        </div>
        {ranges.length > 1 && <p className="muted mt-1 text-[10px]">{ranges.map(range => `${range.from.toFixed(1)}–${range.to.toFixed(1)} s ${deliveryLabel(range.delivery).toLowerCase()}`).join(' · ')}</p>}
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <label className="muted flex items-center gap-1.5 text-[11px]">From the playhead, change to
            <select className="field !w-auto !px-2 !py-1 !text-[11px]" value="" disabled={working} aria-label={`Change how ${speaker} is heard from the playhead`}
              onChange={event => {
                const delivery = event.target.value as Delivery
                const at = Math.min(shot.duration - 0.1, Math.max(0, player.time() - shot.start_time))
                if (delivery) save(speaker, changeAt(ranges, at, delivery))
              }}>
              <option value="">…</option>
              {DELIVERIES.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </label>
          {own && <button type="button" className="!text-[11px] text-[#e0a96d] underline" disabled={working} onClick={() => save(speaker, null)}>Let the director decide</button>}
        </div>
      </li>
    })}</ul>
  </section>
}
