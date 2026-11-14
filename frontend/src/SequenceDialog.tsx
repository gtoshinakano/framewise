import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Check, Clapperboard, Download, LoaderCircle, Merge, MessageSquare, Pencil, Play, RefreshCw, Save, Sparkles, Square, WandSparkles, X } from 'lucide-react'
import { mediaUrl } from './api'
import CutAssistant from './CutAssistant'
import { CopyButton } from './Markdown'
import { sequenceApi, shotFrameUrl, type Bible, type RunStep, type SequenceDetail, type SequenceShot, type ShotDialogue } from './sequenceApi'
import { canMerge, clock, issueInstruction, issuesFor, progress, promptSections, shotStatus, splitChoices, stage, staleText, type Tone } from './sequenceView'

type Tab = 'direction' | 'shots' | 'assistant'
const TONES: Record<Tone, string> = { good: 'bg-[#173a2b] text-[#9ff0c8]', busy: 'bg-[#2b2442] text-[#d6c8ff]', warn: 'bg-[#4a3a17] text-[#ffd59a]', bad: 'bg-[#3d2025] text-[#ffaaa7]', idle: 'bg-[#262b35] text-[#aeb5c1]' }
const DELIVERY: Record<string, string> = { lip_sync: 'lip sync', off_screen: 'off screen', inner_voice: 'inner voice', narration: 'narration' }
const ORIGIN: Record<string, string> = { model: 'written by the director', manual_edit: 'edited by you', assistant_edit: 'edited with the assistant' }

/** The sequence workspace: direct the range (bible and shot plan), review, write the shot prompts, and download. */
export default function SequenceDialog({ sequenceId, episodeId, proxyReady, onClose, onChanged }: {
  sequenceId: string; episodeId: string; proxyReady: boolean; onClose: () => void; onChanged: () => void
}) {
  const [detail, setDetail] = useState<SequenceDetail | null>(null)
  const [tab, setTab] = useState<Tab>('direction')
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
        if (issues.length && !shot.job) await sequenceApi.run(sequenceId, { step: 'shots', shot_keys: [shot.key], instruction: issueInstruction(issues) })
      }
    })
  }

  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Sequence ${info?.title || ''}`} className="flex h-[94vh] w-[96vw] max-w-[1800px] flex-col overflow-hidden rounded-2xl border hairline bg-[#131720] shadow-2xl outline-none">
      <header className="grid grid-cols-[1fr_auto] items-center gap-3 border-b hairline px-5 py-3 lg:grid-cols-[1fr_auto_1fr]">
        <div className="min-w-0"><p className="text-[11px] font-semibold uppercase tracking-[.14em] text-[#e0a96d]">{label}</p><h2 className="truncate text-base font-semibold">{info?.title || 'Loading…'}</h2></div>
        <div className="order-3 col-span-2 flex justify-center lg:order-none lg:col-span-1"><div className="flex gap-1 rounded-lg border hairline p-1" role="tablist">
          {([['direction', 'Direction'], ['shots', `Shots${counts?.shots ? ` · ${counts.ready}/${counts.shots}` : ''}`], ['assistant', 'Assistant']] as const).map(([value, text]) =>
            <button key={value} role="tab" aria-selected={tab === value} onClick={() => setTab(value)} className={`rounded-md px-5 py-1.5 text-xs font-semibold ${tab === value ? 'bg-[#3a2f1f] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{text}</button>)}
        </div></div>
        <div className="flex items-center justify-end gap-2">
          <button type="button" className="btn btn-quiet px-3 py-2" disabled={!shots.length || !!exporting} onClick={() => void download()} title={detail?.ready ? 'Download sequence.json, the audio slices, the initial frames and the prompts' : 'Download what exists now; shots that are not written have no prompt yet'}>
            {exporting ? <LoaderCircle size={14} className="animate-spin" /> : <Download size={14} />} {exporting ? `Export · ${exporting}` : 'Download ZIP'}</button>
          <button onClick={onClose} className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" aria-label="Close sequence"><X size={18} /></button>
        </div>
      </header>

      {detail && where && counts && <div className="flex flex-wrap items-center gap-3 border-b hairline bg-[#171c25] px-5 py-2.5">
        <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${TONES[where.key === 'ready' ? 'good' : where.key === 'directing' || where.key === 'writing' ? 'busy' : where.key === 'direct' || where.key === 'write' ? 'idle' : 'warn']}`}>
          {where.key === 'directing' || where.key === 'writing' ? <LoaderCircle size={12} className="animate-spin" /> : where.key === 'ready' ? <Check size={12} /> : where.key === 'direct' || where.key === 'write' ? <Clapperboard size={12} /> : <AlertTriangle size={12} />}
          {where.hint}</span>
        <span className="muted text-[11px]">{detail.bible ? 'Bible ✓' : 'Bible —'} · {detail.plan ? `Plan ${counts.shots} shots ✓` : 'Plan —'} · Prompts {counts.written}/{counts.shots}{detail.review ? ` · Review ${detail.review.content.issues.length} issue${detail.review.content.issues.length === 1 ? '' : 's'}` : ''}</span>
        <span className="ml-auto flex flex-wrap items-center gap-2">
          {(where.key === 'directing' || where.key === 'writing') && <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void act('cancel', () => sequenceApi.cancel(sequenceId))}><Square size={12} /> Stop</button>}
          {(where.key === 'direct' || where.key === 'redirect') && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void run('direct')} title="The director writes the sequence bible and the shot plan, then waits for your review"><WandSparkles size={13} /> Direct sequence</button>}
          {where.key === 'write' && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void run('shots')} title="Writes every shot that is missing, stale or failing its format check, two at a time, then reviews continuity"><Sparkles size={13} /> Write shot prompts ({counts.needWriting})</button>}
          {(where.key === 'write' || where.key === 'ready') && counts.written > 0 && <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void run('review')} title="One pass over all written shots for continuity, speech and staging problems"><RefreshCw size={12} /> Review continuity</button>}
          {(where.key === 'write' || where.key === 'ready') && flagged.length > 0 && <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={!!working} onClick={() => void reviseFlagged()} title="Rewrites each shot the review marked as wrong, with the review's suggestion as the instruction">Revise flagged shots ({new Set(flagged.map(issue => issue.shot_key)).size})</button>}
        </span>
      </div>}
      {error && <div role="alert" className="flex items-start gap-3 border-b border-[#63363e] bg-[#2b191f] px-5 py-2 text-xs text-[#ffb6bc]"><span className="flex-1 whitespace-pre-wrap">{error}</span><button onClick={() => setError('')} aria-label="Dismiss"><X size={14} /></button></div>}

      {!detail ? <div className="muted p-12 text-sm">Loading sequence…</div> : <>
        {assistantMounted && <CutAssistant sequence={{ id: sequenceId, label }} active={tab === 'assistant'} onResultsChanged={() => { void load(); onChanged() }} />}
        {tab === 'direction' && <DirectionTab detail={detail} working={!!working || active} act={act} run={run} onOpenShot={key => { setSelectedKey(key); setTab('shots') }} />}
        {tab === 'shots' && <ShotsTab detail={detail} selected={selected} onSelect={setSelectedKey} episodeId={episodeId} proxyReady={proxyReady} working={!!working} act={act} run={run} />}
      </>}
    </div>
  </div>
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

function DirectionTab({ detail, working, act, run, onOpenShot }: { detail: SequenceDetail; working: boolean; act: Act; run: Run; onOpenShot: (key: string) => void }) {
  const { sequence, bible, plan, shots } = detail
  const [title, setTitle] = useState(sequence.title)
  const [instruction, setInstruction] = useState(sequence.instruction)
  const [editingBible, setEditingBible] = useState(false)
  useEffect(() => { setTitle(sequence.title); setInstruction(sequence.instruction) }, [sequence.id, sequence.title, sequence.instruction])
  const dirty = title.trim() !== sequence.title || instruction.trim() !== sequence.instruction
  const candidates = detail.timeline?.candidates || []
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
    <Panel title="Shot plan" subtitle={plan ? `${shots.length} shots · ${provenance(plan)}` : 'Consecutive shots of at most 15 seconds along the original audio'}
      actions={plan && <button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" disabled={working} onClick={() => void run('plan')} title="Plan the shots again from the current bible"><RefreshCw size={12} /> Replan</button>}>
      {!plan ? <p className="muted text-xs leading-5">Not planned yet. Each shot is one continuous H3 take: the director keeps the hard cuts by default, merges short cuts when one take can carry them, and splits cuts longer than 15 seconds at pauses.</p> : <>
        {detail.plan_problems.length > 0 && <p className="mb-3 rounded-lg bg-[#4a3a17] p-2 text-[11px] leading-5 text-[#ffd59a]">The plan no longer fits the cuts: {detail.plan_problems.join('; ')}. Direct the sequence again.</p>}
        {plan.completed_by_app && <p className="mb-3 rounded-lg bg-[#262b35] p-2 text-[11px] leading-5 text-[#c9cfda]">The director's plan was not valid, so the app completed it from the hard cuts and pauses. Shots marked “auto” were added.</p>}
        {plan.content.notes && <p className="muted mb-3 text-xs leading-5">{plan.content.notes}</p>}
        <ol className="space-y-2">{shots.map((shot, index) => {
          const status = shotStatus(shot)
          const choices = splitChoices(shot, candidates)
          return <li key={shot.key} className="rounded-lg border border-[#2f3643] bg-[#161b24] p-3">
            <div className="flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => onOpenShot(shot.key)} className="text-sm font-semibold hover:text-[#e0a96d]" title="Open this shot">Shot {shot.index}</button>
              <span className="font-mono text-[11px] text-[#aeb5c1]">{clock(shot.t0)}–{clock(shot.t1)} · {shot.duration.toFixed(1)} s</span>
              <span className="muted text-[11px]">cut{shot.source_cuts.length > 1 ? 's' : ''} {shot.source_cuts.join(', ')}{shot.scene ? ` · ${shot.scene}` : ''}</span>
              {shot.merged && <Tag tone="busy">merged</Tag>}
              {(shot.starts_mid_cut || shot.ends_mid_cut) && <Tag tone="busy">split</Tag>}
              {shot.trim_seconds !== null && <Tag tone="idle" title={`Generated at ${shot.generate_seconds} s and trimmed to ${shot.trim_seconds} s`}>short</Tag>}
              {shot.auto_fixed && <Tag tone="warn">auto</Tag>}
              <Tag tone={status.tone}>{status.label}</Tag>
              <span className="ml-auto flex items-center gap-1.5">
                <button type="button" className="btn btn-quiet !px-2 !py-1 !text-[11px]" disabled={working || !canMerge(shots, index, detail.limits.shot_seconds)} onClick={() => void act('merge', () => sequenceApi.merge(sequence.id, shot.key))} title={canMerge(shots, index, detail.limits.shot_seconds) ? 'Join this shot with the next one as one continuous take' : 'Together they would exceed 15 seconds, or there is no next shot'}><Merge size={12} /> Merge next</button>
                <select className="field !w-auto !px-2 !py-1 !text-[11px]" aria-label={`Split shot ${shot.index} at`} disabled={working || !choices.length} value="" onChange={event => { const time = Number(event.target.value); if (event.target.value) void act('split', () => sequenceApi.split(sequence.id, shot.key, time)) }}>
                  <option value="">{choices.length ? 'Split at…' : 'No split point'}</option>
                  {choices.map(choice => <option key={choice.time} value={choice.time}>{choice.label}</option>)}
                </select>
              </span>
            </div>
            <p className="mt-2 text-[13px] leading-5">{shot.intent || <span className="muted">No intent written.</span>}</p>
            {shot.staging && <p className="muted mt-1 text-xs leading-5">Staging: {shot.staging}</p>}
            <p className="muted mt-1 text-[11px]">Initial frame: {shot.frame.strategy === 'cut_first_frame' ? `first frame of cut ${shot.frame.cut}` : `continues the take of the previous shot inside cut ${shot.frame.cut}`}{shot.lines.length ? ` · ${shot.lines.length} spoken line${shot.lines.length > 1 ? 's' : ''}` : ' · no speech'}</p>
          </li>
        })}</ol>
        <p className="muted mt-3 text-[11px] leading-5">Changing a shot's boundaries makes it a new shot: its prompt must be written again. For other changes (intent, staging, which cut supplies the frame) ask the assistant, or edit and rewrite the shot.</p>
      </>}
    </Panel>
  </div>
}

function Tag({ tone, title, children }: { tone: Tone; title?: string; children: ReactNode }) {
  return <span title={title} className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${TONES[tone]}`}>{children}</span>
}

function BibleView({ bible }: { bible: Bible }) {
  return <div className="space-y-4">
    <Field label="Logline">{bible.logline}</Field>
    <Field label="Synopsis">{bible.synopsis}</Field>
    <Field label="Style lock">{bible.style}</Field>
    <div className="grid gap-4 sm:grid-cols-2"><Field label="Spoken language">{bible.spoken_language}</Field><Field label="Pacing">{bible.pacing || '—'}</Field></div>
    {bible.arc && <Field label="Arc">{bible.arc}</Field>}
    <Field label="Scenes"><ul className="space-y-1.5">{bible.scenes.map(scene => <li key={scene.id}><span className="font-semibold">{scene.id}</span> · cuts {scene.from_cut}–{scene.to_cut} · {scene.location}<span className="muted"> · {[scene.time_of_day, scene.lighting, scene.mood].filter(Boolean).join(' · ')}{scene.scenario ? ` · scenario “${scene.scenario}”` : ''}</span></li>)}</ul></Field>
    <Field label="Characters">{bible.characters.length ? <ul className="space-y-2">{bible.characters.map(person => <li key={person.name}><span className="font-semibold">{person.name}</span>{person.role ? <span className="muted"> · {person.role}</span> : null}<br /><span className="text-xs leading-5 text-[#c9cfda]">{person.anchors}{person.outfit ? ` · wears ${person.outfit}` : ''}</span></li>)}</ul> : 'No named character is on screen.'}</Field>
    <Field label="Voices">{bible.voices.length ? <ul className="space-y-1.5">{bible.voices.map(voice => <li key={voice.speaker_label}><span className="font-mono text-xs">{voice.speaker_label}</span> → <span className="font-semibold">{voice.character || (voice.kind === 'narrator' ? 'Narrator' : 'unknown')}</span><span className="muted"> · {voice.source === 'transcript' ? 'your transcript link' : `director, ${voice.confidence}`}{voice.reason && voice.source !== 'transcript' ? ` · ${voice.reason}` : ''}</span></li>)}</ul> : 'Nobody speaks in this range.'}</Field>
    {bible.continuity.length > 0 && <Field label="Continuity rules"><ul className="list-disc space-y-1 pl-5">{bible.continuity.map((rule, index) => <li key={index}>{rule}</li>)}</ul></Field>}
    {bible.open_questions.length > 0 && <Field label="Open questions"><ul className="list-disc space-y-1 pl-5 text-[#ffd59a]">{bible.open_questions.map((question, index) => <li key={index}>{question}</li>)}</ul></Field>}
  </div>
}

function ShotsTab({ detail, selected, onSelect, episodeId, proxyReady, working, act, run }: {
  detail: SequenceDetail; selected?: SequenceShot; onSelect: (key: string) => void; episodeId: string; proxyReady: boolean; working: boolean; act: Act; run: Run
}) {
  if (!detail.shots.length) return <div className="muted p-12 text-sm leading-6">There is no shot plan yet. Direct the sequence on the Direction tab first.</div>
  return <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-5 lg:grid-cols-[17rem_minmax(0,1fr)] lg:overflow-hidden">
    <nav aria-label="Shots" className="scrollbar min-h-0 space-y-1.5 lg:overflow-y-auto lg:pr-1">
      {detail.shots.map(shot => {
        const status = shotStatus(shot)
        const issues = issuesFor(detail.review?.content.issues, shot)
        return <button key={shot.key} type="button" onClick={() => onSelect(shot.key)} aria-current={selected?.key === shot.key} className={`flex w-full items-center gap-2.5 rounded-lg border p-2 text-left ${selected?.key === shot.key ? 'border-[#e0a96d] bg-[#2a2319]' : 'border-[#2f3643] bg-[#161b24] hover:border-[#5a4a35]'}`}>
          <img src={shotFrameUrl(detail.sequence.id, shot.key)} alt="" loading="lazy" className="h-12 w-20 shrink-0 rounded bg-black object-cover" />
          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-1.5"><span className="text-xs font-semibold">Shot {shot.index}</span><Tag tone={status.tone}>{status.label}</Tag>{issues.length > 0 && <Tag tone={issues.some(issue => issue.severity === 'fix') ? 'bad' : 'warn'}>{issues.length} issue{issues.length > 1 ? 's' : ''}</Tag>}</span>
            <span className="muted block truncate font-mono text-[10px]">{clock(shot.t0)}–{clock(shot.t1)} · {shot.duration.toFixed(1)} s</span>
            <span className="muted block truncate text-[10px]">{shot.intent}</span>
          </span>
        </button>
      })}
    </nav>
    {selected && <ShotDetail key={selected.key} detail={detail} shot={selected} episodeId={episodeId} proxyReady={proxyReady} working={working} act={act} run={run} />}
  </div>
}

/** The proxy video bound to one shot's interval of the episode. */
function RangeVideo({ episodeId, start, end }: { episodeId: string; start: number; end: number }) {
  return <video className="h-full w-full" controls preload="metadata" src={mediaUrl(episodeId, 'proxy')} aria-label="Play the source of this shot"
    onLoadedMetadata={event => { event.currentTarget.currentTime = start }}
    onSeeking={event => { const player = event.currentTarget; if (player.currentTime < start) player.currentTime = start; else if (player.currentTime > end) player.currentTime = end }}
    onPlay={event => { if (event.currentTarget.currentTime >= end) event.currentTarget.currentTime = start }}
    onTimeUpdate={event => { const player = event.currentTarget; if (player.currentTime >= end && !player.paused) { player.pause(); if (player.currentTime > end) player.currentTime = end } }} />
}

function ShotDetail({ detail, shot, episodeId, proxyReady, working, act, run }: { detail: SequenceDetail; shot: SequenceShot; episodeId: string; proxyReady: boolean; working: boolean; act: Act; run: Run }) {
  const result = shot.result
  const [view, setView] = useState<'read' | 'raw' | 'edit'>('read')
  const [instruction, setInstruction] = useState('')
  const [showVideo, setShowVideo] = useState(false)
  const issues = issuesFor(detail.review?.content.issues, shot)
  const dialogue: ShotDialogue[] = result?.derived.dialogue || shot.lines
  const sections = useMemo(() => result?.prompt ? promptSections(result.prompt) : [], [result?.prompt])
  const status = shotStatus(shot)
  const busy = working || !!shot.job
  return <div className="scrollbar flex min-h-0 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
    <div className="grid shrink-0 gap-4 xl:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <div>
        <div className="relative aspect-video overflow-hidden rounded-xl border hairline bg-black">
          {showVideo && proxyReady ? <RangeVideo episodeId={episodeId} start={shot.start_time} end={shot.end_time} /> : <img src={shotFrameUrl(detail.sequence.id, shot.key)} alt={`Initial frame of shot ${shot.index}`} className="h-full w-full object-contain" />}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <button type="button" className="btn btn-quiet !px-2.5 !py-1.5 !text-xs" disabled={!proxyReady} onClick={() => setShowVideo(value => !value)}>{showVideo ? 'Show initial frame' : <><Play size={12} /> Play source</>}</button>
          <span className="muted text-[11px]">{shot.frame.strategy === 'cut_first_frame' ? `Initial frame: first frame of cut ${shot.frame.cut}` : `Initial frame: where the previous shot stopped inside cut ${shot.frame.cut} (described from the motion, not observed)`}</span>
        </div>
      </div>
      <div className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center gap-2"><h3 className="text-base font-semibold">Shot {shot.index}</h3><Tag tone={status.tone}>{status.label}</Tag>{shot.merged && <Tag tone="busy">merged</Tag>}{(shot.starts_mid_cut || shot.ends_mid_cut) && <Tag tone="busy">split</Tag>}</div>
        <p className="font-mono text-xs text-[#aeb5c1]">{clock(shot.t0)}–{clock(shot.t1)} in the sequence · {shot.duration.toFixed(2)} s · cut{shot.source_cuts.length > 1 ? 's' : ''} {shot.source_cuts.join(', ')}</p>
        {shot.trim_seconds !== null && <p className="text-[11px] leading-5 text-[#c9cfda]">Shorter than H3's minimum: generate {shot.generate_seconds} s (the prompt ends on a hold) and trim to {shot.trim_seconds} s.</p>}
        <p className="text-[13px] leading-5">{shot.intent}</p>
        {shot.staging && <p className="muted text-xs leading-5">Staging: {shot.staging}</p>}
        <div>
          <p className="label !mb-1">Dialogue</p>
          {dialogue.length ? <ul className="space-y-1">{dialogue.map(line => <li key={line.line} className="text-xs leading-5">
            <span className="font-mono text-[10px] text-[#8b93a3]">{line.line} {line.t0.toFixed(2)}–{line.t1.toFixed(2)}</span> <span className="font-semibold">{line.character_name || line.speaker_label || 'Unknown'}</span>
            {line.delivery && <Tag tone={line.delivery === 'lip_sync' ? 'good' : 'idle'}>{DELIVERY[line.delivery]}{line.subject ? ` · S${line.subject}` : ''}</Tag>} <span className="text-[#e8ebf1]">{line.text}</span>
            {(line.crosses_start || line.crosses_end) && <span className="muted"> ({line.crosses_start ? 'continues from the previous shot' : 'continues in the next shot'})</span>}
          </li>)}</ul> : <p className="muted text-xs">Nobody speaks during this shot.</p>}
        </div>
      </div>
    </div>

    {shot.stale.length > 0 && <p className="shrink-0 rounded-lg bg-[#4a3a17] p-2.5 text-xs leading-5 text-[#ffd59a]">{staleText(shot.stale)}</p>}
    {issues.length > 0 && <div className="shrink-0 rounded-lg border border-[#5b4a2a] bg-[#241d12] p-3">
      <p className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold text-[#ffd59a]"><MessageSquare size={13} /> Review</p>
      <ul className="space-y-1.5">{issues.map((issue, index) => <li key={index} className="text-xs leading-5"><Tag tone={issue.severity === 'fix' ? 'bad' : 'warn'}>{issue.severity}</Tag> {issue.problem}{issue.suggestion && <span className="muted"> → {issue.suggestion}</span>}</li>)}</ul>
      <button type="button" className="mt-2 !text-[11px] text-[#e0a96d] underline" onClick={() => setInstruction(issueInstruction(issues))}>Use as the instruction below</button>
    </div>}

    <section className="shrink-0 rounded-xl border hairline bg-[#1b202a]" aria-label="H3 prompt">
      <header className="flex flex-wrap items-center gap-2 border-b hairline px-4 py-3">
        <div className="mr-auto min-w-0"><h3 className="text-sm font-semibold">MiniMax H3 prompt</h3><p className="muted text-[11px]">{result ? `${provenance(result)}${result.check?.stats ? ` · ${result.check.stats.description_words ?? '?'} words · ${result.check.stats.chars ?? '?'} characters` : ''}` : 'Not written yet'}</p></div>
        {result && <>
          <div className="flex gap-1 rounded-lg border hairline p-0.5" role="tablist" aria-label="Prompt view">{([['read', 'Read'], ['raw', 'Text'], ['edit', 'Edit direction']] as const).map(([value, text]) =>
            <button key={value} role="tab" aria-selected={view === value} onClick={() => setView(value)} className={`rounded-md px-2.5 py-1 !text-[11px] font-semibold ${view === value ? 'bg-[#3a2f1f] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{text}</button>)}</div>
          {result.prompt && <CopyButton text={result.prompt} label="Copy prompt" />}
        </>}
      </header>
      <div className="p-4">
        {!result ? <p className="muted text-xs leading-5">{shot.job ? 'The director is writing this shot…' : 'This shot has no prompt yet. “Write shot prompts” writes every missing shot; the button below writes only this one.'}</p> : <>
          {result.check && !result.check.ok && <ul className="mb-3 list-disc space-y-1 rounded-lg bg-[#3d2025] p-2.5 pl-6 text-xs leading-5 text-[#ffaaa7]">{result.check.errors.map((item, index) => <li key={index}>{item}</li>)}</ul>}
          {!!result.check?.warnings.length && <ul className="mb-3 list-disc space-y-1 rounded-lg bg-[#262b35] p-2.5 pl-6 text-[11px] leading-5 text-[#c9cfda]">{result.check.warnings.map((item, index) => <li key={index}>{item}</li>)}</ul>}
          {view === 'edit' ? <>
            <p className="muted mb-2 text-[11px] leading-5">You edit the direction; the app renders the prompt again and checks it. Write names, not labels: labels, timecodes and spoken words are added for you.</p>
            <JsonEditor value={result.content} saving={working} onCancel={() => setView('read')} onSave={content => void act('shot', () => sequenceApi.revise(result.id, content)).then(saved => { if (saved) setView('read') })} />
          </> : view === 'raw' ? <pre className="scrollbar max-h-[36rem] overflow-auto whitespace-pre-wrap rounded-lg border border-[#2b3240] bg-[#0f131a] p-3 font-mono text-[11px] leading-5 text-[#c9cfda]">{result.prompt}</pre>
            : <div className="space-y-4">{sections.map(section => <div key={section.name}><p className="label !mb-1">{section.title}</p><div className="space-y-1.5 text-[13px] leading-6 text-[#e8ebf1]">{section.lines.map((line, index) => <p key={index} className={/^\d{2}:\d{2}/.test(line) ? 'border-l-2 border-[#5a4a35] pl-2.5' : ''}>{line}</p>)}</div></div>)}</div>}
          {result.derived.references && <p className="muted mt-4 text-[11px] leading-5">References to attach, in this order: {result.derived.references.map(reference => `${reference.label} ${reference.role === 'initial_frame' ? 'initial frame' : reference.role === 'character_sheet' ? `sheet of ${reference.name}` : reference.role === 'scenario' ? `scenario “${reference.name}”` : `audio ${result.derived.audio_file || ''}`}`).join(' · ')}</p>}
          {result.derived.still_prompt && <details className="mt-3 rounded-lg border hairline bg-[#171c25] text-xs"><summary className="cursor-pointer px-3 py-2 font-semibold text-[#aeb5c1] hover:text-white">Still image prompt for the initial frame</summary><div className="border-t hairline p-3"><pre className="whitespace-pre-wrap font-mono text-[11px] leading-5 text-[#c9cfda]">{result.derived.still_prompt}</pre><div className="mt-2"><CopyButton text={result.derived.still_prompt} label="Copy still prompt" /></div></div></details>}
          {result.content.open_questions.length > 0 && <div className="mt-3"><p className="label !mb-1">Open questions</p><ul className="list-disc space-y-1 pl-5 text-xs leading-5 text-[#ffd59a]">{result.content.open_questions.map((question, index) => <li key={index}>{question}</li>)}</ul></div>}
        </>}
      </div>
      <footer className="flex flex-wrap items-end gap-2 border-t hairline p-3">
        <label className="min-w-0 flex-1"><span className="label !mb-1">Instruction for a rewrite (optional)</span><input className="field !py-2 !text-xs" maxLength={4000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder="For example: she is off screen for the first line; keep the camera static" /></label>
        <button type="button" className="btn btn-quiet !py-2 !text-xs" disabled={busy || !detail.bible} onClick={() => void run('shots', { shot_keys: [shot.key], instruction: instruction.trim() }).then(queued => { if (queued) setInstruction('') })} title="The director writes this shot again from the current bible, plan and analysis"><Sparkles size={13} /> {result ? 'Rewrite this shot' : 'Write this shot'}</button>
      </footer>
    </section>
  </div>
}
