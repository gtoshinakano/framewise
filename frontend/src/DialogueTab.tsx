import { memo, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Check, Crosshair, LoaderCircle, Merge, Minus, Pause, Play, Plus, RefreshCw, Save, Scissors, Trash2, Undo2, WandSparkles, X } from 'lucide-react'
import { applyVoice, changed, counts, diff, isNew, lineAt, mergeNext, nudge, payload, problemsById, sameContent, setEdge, setSpeaker, setText, speakerOptions, splitAt, visibleLines, type Filter } from './compilationView'
import DialogueTimeline from './DialogueTimeline'
import { boundaryMarkers, limits, lineColour, openingFrame, placePoint, pointWarnings, samePoints, type Marker } from './dialogueTimeline'
import { PlayerSlot, usePlayer, usePlayerValue } from './SourcePlayer'
import { sequenceApi, type Compilation, type CompiledLine, type SequenceCut, type SequenceDetail, type ShotPoint } from './sequenceApi'
import { clock } from './sequenceView'
import { TimecodeInput } from './TimecodeInput'

type Act = (label: string, action: () => Promise<SequenceDetail | unknown>) => Promise<boolean>
type Run = (step: 'compile' | 'direct') => Promise<boolean>
type Tone = 'good' | 'busy' | 'warn' | 'bad' | 'idle'
const CHIP: Record<Tone, string> = { good: 'bg-[#173a2b] text-[#9ff0c8]', busy: 'bg-[#2b2442] text-[#d6c8ff]', warn: 'bg-[#4a3a17] text-[#ffd59a]', bad: 'bg-[#3d2025] text-[#ffaaa7]', idle: 'bg-[#262b35] text-[#aeb5c1]' }
const Chip = ({ tone, title, children }: { tone: Tone; title?: string; children: ReactNode }) => <span title={title} className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${CHIP[tone]}`}>{children}</span>

/** A number edited as text, committed on blur or Enter, so typing "1." does not jump. */
function NumberField({ value, label, onCommit }: { value: number; label: string; onCommit: (value: number) => void }) {
  const [text, setTextValue] = useState(value.toFixed(2))
  const editing = useRef(false)
  useEffect(() => { if (!editing.current) setTextValue(value.toFixed(2)) }, [value])
  const commit = () => { editing.current = false; const parsed = Number(text); if (Number.isFinite(parsed) && text.trim() !== '') onCommit(parsed); else setTextValue(value.toFixed(2)) }
  return <input aria-label={label} inputMode="decimal" className="field !w-[4.6rem] !px-1.5 !py-1 text-center font-mono !text-xs" value={text}
    onFocus={() => { editing.current = true }} onChange={event => setTextValue(event.target.value)} onBlur={commit} onKeyDown={event => { if (event.key === 'Enter') { event.currentTarget.blur() } }} />
}

function TimeEdit({ label, value, onChange, onNudge, onPlayhead }: { label: string; value: number; onChange: (value: number) => void; onNudge: (delta: number) => void; onPlayhead: () => void }) {
  const small = 'rounded p-1 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white'
  return <span className="inline-flex items-center gap-0.5" role="group" aria-label={label}>
    <button type="button" className={small} onClick={() => onNudge(-0.1)} aria-label={`${label} earlier by 0.1 s`} title="0.1 s earlier"><Minus size={12} /></button>
    <NumberField value={value} label={label} onCommit={onChange} />
    <button type="button" className={small} onClick={() => onNudge(0.1)} aria-label={`${label} later by 0.1 s`} title="0.1 s later"><Plus size={12} /></button>
    <button type="button" className={small} onClick={onPlayhead} aria-label={`Set ${label} to the playhead`} title="Set to the video's playhead"><Crosshair size={12} /></button>
  </span>
}

type RowProps = {
  line: CompiledLine; selected: boolean; active: boolean; playing: boolean; colour: string; shotsInside: string; problems: string[]; duration: number; added: boolean
  options: { value: string; label: string }[]; characters: { id: string; name: string }[]
  onSelect: (id: string) => void; onPlay: (line: CompiledLine) => void; onEdit: (id: string, edit: (line: CompiledLine) => CompiledLine) => void
  onMerge: (id: string) => void; onSplit: (id: string) => void; onDelete: (id: string) => void; playhead: () => number
}

const Row = memo(function Row({ line, selected, active, playing, colour, shotsInside, problems, duration, added, options, characters, onSelect, onPlay, onEdit, onMerge, onSplit, onDelete, playhead }: RowProps) {
  const edited = changed(line)
  const marks = edited ? diff(line.original_text, line.text) : []
  const border = active ? 'border-[#e0a96d] bg-[#2a2319]' : selected ? 'border-[#7d6542] bg-[#1d222c]' : 'border-[#2f3643] bg-[#161b24]'
  const icon = 'rounded p-1.5 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white disabled:opacity-40'
  return <li id={`line-${line.id}`} data-line={line.id} className={`rounded-lg border border-l-4 p-3 ${border}`} style={{ borderLeftColor: colour }} onClick={() => onSelect(line.id)}>
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <button type="button" className="btn btn-quiet !px-2 !py-1 !text-[11px]" onClick={() => onPlay(line)} aria-pressed={playing}
        title={playing ? 'Pause (Space)' : 'Play this line of the source video (Space)'}>{playing ? <><Pause size={12} /> Pause</> : <><Play size={12} /> Play</>}</button>
      <TimeEdit label={`${line.id} start`} value={line.t0} onChange={value => onEdit(line.id, item => setEdge(item, 't0', value, duration))} onNudge={delta => onEdit(line.id, item => nudge(item, 't0', delta, duration))} onPlayhead={() => onEdit(line.id, item => setEdge(item, 't0', playhead(), duration))} />
      <span className="muted text-[11px]">to</span>
      <TimeEdit label={`${line.id} end`} value={line.t1} onChange={value => onEdit(line.id, item => setEdge(item, 't1', value, duration))} onNudge={delta => onEdit(line.id, item => nudge(item, 't1', delta, duration))} onPlayhead={() => onEdit(line.id, item => setEdge(item, 't1', playhead(), duration))} />
      <span className="muted font-mono text-[10px]">{(line.t1 - line.t0).toFixed(1)} s · {line.id}{line.speaker_label ? ` · ${line.speaker_label}` : ''}</span>
      {added && <Chip tone="good" title="A line of the cut you added; review it before saving">new</Chip>}
      {edited && <Chip tone="busy" title="The words differ from the recognised text">corrected</Chip>}
      {line.confidence === 'low' && <Chip tone="warn" title={line.note || 'Qwen was not sure'}>unsure</Chip>}
      {line.continues_next && <Chip tone="warn" title="This utterance continues in the next line">continues</Chip>}
      {shotsInside && <Chip tone="busy" title="A shot boundary falls inside this line: its words are divided between two shots">{shotsInside} starts inside</Chip>}
      <span className="ml-auto flex items-center">
        <button type="button" className={icon} onClick={event => { event.stopPropagation(); onMerge(line.id) }} aria-label={`Merge ${line.id} with the next line`} title="Merge with the next line"><Merge size={13} /></button>
        <button type="button" className={icon} onClick={event => { event.stopPropagation(); onSplit(line.id) }} aria-label={`Split ${line.id} at the playhead`} title="Split at the video's playhead"><Scissors size={13} /></button>
        <button type="button" className={`${icon} hover:!text-[#ffaaa7]`} onClick={event => { event.stopPropagation(); onDelete(line.id) }} aria-label={`Delete ${line.id}`} title="Delete this line"><Trash2 size={13} /></button>
      </span>
    </div>
    <div className="mt-2 grid gap-3 md:grid-cols-[13rem_minmax(0,1fr)]">
      <div className="space-y-2">
        <select className="field !py-1.5 !text-xs" aria-label={`Speaker of ${line.id}`} value={line.character || ''} onChange={event => onEdit(line.id, item => setSpeaker(item, event.target.value || null, characters))}>
          {options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </div>
      <div className="min-w-0 space-y-1.5">
        <textarea lang="ja" className="field resize-y !py-1.5 text-[13px] leading-6" rows={Math.max(2, Math.ceil(line.text.length / 40))} aria-label={`Words of ${line.id}`} value={line.text} onChange={event => onEdit(line.id, item => setText(item, event.target.value))} />
        {edited && <p className="text-[11px] leading-5 text-[#8b93a3]"><span className="mr-1">Recognised:</span>{marks.map((part, index) => part.kind === 'same' ? <span key={index}>{part.text}</span>
          : part.kind === 'del' ? <del key={index} className="rounded bg-[#3d2025] px-0.5 text-[#ffaaa7]">{part.text}</del> : <ins key={index} className="rounded bg-[#173a2b] px-0.5 text-[#9ff0c8] no-underline">{part.text}</ins>)}</p>}
        {line.subtitle_text && line.subtitle_text !== line.text && <p className="text-[11px] leading-5 text-[#8b93a3]"><span className="mr-1">Subtitle:</span><span className="text-[#c9cfda]">{line.subtitle_text}</span>
          <button type="button" className="ml-2 text-[#e0a96d] underline" onClick={() => onEdit(line.id, item => setText(item, item.subtitle_text || item.text))}>Use subtitle</button></p>}
        {line.note && <p className="text-[11px] leading-5 text-[#8b93a3]">Qwen: {line.note}</p>}
        {problems.map(problem => <p key={problem} role="alert" className="text-[11px] text-[#ffaaa7]">{problem}</p>)}
      </div>
    </div>
  </li>
})

/** The selected shot boundary: where it sits, the frame its shot opens on, and what to look at before keeping it. */
function BoundaryCard({ marker, markers, cuts, lines, origin, duration, disabled, onPlace, onRelease, onClose, onSave }: {
  marker: Marker; markers: Marker[]; cuts: SequenceCut[]; lines: CompiledLine[]; origin: number; duration: number; disabled: boolean
  onPlace: (marker: Marker, t: number) => void; onRelease: (marker: Marker) => void; onClose: () => void; onSave?: () => void
}) {
  const player = usePlayer()
  const playhead = usePlayerValue(time => Math.round((time - origin) * 25) / 25)
  const cut = cuts[marker.index + 1]
  const frame = cut ? openingFrame(marker, cut, origin) : null
  const range = limits(marker.index, markers, cuts)
  const warnings = pointWarnings(marker, cuts, lines)
  const shift = marker.t - marker.hard
  const where = Math.abs(shift) <= 0.002 ? 'on the hard cut' : `${Math.abs(shift).toFixed(2)} s ${shift < 0 ? 'before' : 'after'} the hard cut (${clock(marker.hard)})`
  const who = marker.source === 'you' ? 'Placed by you: the director keeps it.' : marker.source === 'plan'
    ? `Where the plan has it${marker.by === 'director' ? ', chosen by the director' : marker.by === 'app' ? ', chosen by the app' : ''}. Drag it, or place it here, to make it yours.`
    : marker.speech ? 'Speech crosses this hard cut: the director decides where the boundary sits unless you place it.' : 'No speech crosses this hard cut, so the boundary stays on it unless you place it.'
  const small = 'btn btn-quiet !px-2.5 !py-1.5 !text-[11px]'
  return <section aria-label={`Start of shot ${marker.shot}`} className="rounded-xl border border-[#7d6542] bg-[#1b202a] p-4">
    <header className="mb-3 flex items-start gap-2">
      <div className="min-w-0 flex-1">
        <h3 className="text-sm font-semibold">Shot {marker.shot} starts at {clock(marker.t)}</h3>
        <p className="muted text-[11px] leading-5">{where} · {who}</p>
      </div>
      <button type="button" className="rounded p-1 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white" onClick={onClose} aria-label="Close the boundary"><X size={14} /></button>
    </header>
    {frame && <figure className="mb-3">
      <img src={frame.url} alt={`The frame shot ${marker.shot} opens on`} className="aspect-video w-full rounded-lg bg-black object-contain" />
      <figcaption className={`mt-1 text-[11px] leading-5 ${frame.derived ? 'text-[#ffd59a]' : 'text-[#aeb5c1]'}`}>Opens on: {frame.caption}</figcaption>
    </figure>}
    <TimecodeInput compact label="Starts at (sequence time)" value={marker.t} min={range.min} max={range.max} playhead={Math.min(range.max, Math.max(range.min, playhead))} disabled={disabled}
      onChange={value => onPlace(marker, value)} />
    <div className="mt-3 flex flex-wrap gap-1.5">
      <button type="button" className={small} onClick={() => player.playRange(origin + Math.max(0, marker.t - 2), origin + Math.min(duration, marker.t + 2))} title="Play two seconds on either side of the boundary"><Play size={12} /> Play across</button>
      {marker.source !== 'you' && <button type="button" className={small} disabled={disabled} onClick={() => onPlace(marker, marker.t)} title="Keep the boundary here as yours; the director will not move it">Place here</button>}
      {marker.source === 'you' && <button type="button" className={small} disabled={disabled} onClick={() => onRelease(marker)} title="Let the director decide this boundary again"><Undo2 size={12} /> Hand back to the director</button>}
      {onSave && <button type="button" className="btn btn-primary !px-2.5 !py-1.5 !text-[11px]" disabled={disabled} onClick={onSave}><Save size={12} /> Save boundaries</button>}
    </div>
    {warnings.length > 0 && <ul className="mt-3 space-y-1 text-[11px] leading-5 text-[#ffd59a]">{warnings.map(warning => <li key={warning} className="flex gap-1.5"><AlertTriangle size={12} className="mt-1 shrink-0" />{warning}</li>)}</ul>}
  </section>
}

/** The playhead in sequence time; it re-renders on its own while the video plays. */
function PlayheadClock({ origin }: { origin: number }) {
  const tenths = usePlayerValue(time => Math.round(Math.max(0, time - origin) * 10))
  return <>{clock(tenths / 10)}</>
}

/** Review and correct the dialogue Qwen compiled from the transcript, the subtitles and the pictures; Direction starts from the saved result. */
export default function DialogueTab({ detail, working, act, run, onContinue, proxyReady }: {
  detail: SequenceDetail; working: boolean; act: Act; run: Run; onContinue: () => void; proxyReady: boolean
}) {
  const view = detail.compilation
  const info = detail.sequence
  const result = view.result
  const duration = detail.timeline?.duration ?? info.duration
  const player = usePlayer()
  const [draft, setDraft] = useState<Compilation | null>(result ? result.content : null)
  const [baseId, setBaseId] = useState<string | undefined>(result?.id)
  const [selected, setSelected] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  useEffect(() => { if (result && result.id !== baseId) { setDraft(result.content); setBaseId(result.id) } }, [result, baseId])
  useEffect(() => { player.setRange({ start: info.start_time, end: info.end_time }) }, [player, info.start_time, info.end_time])
  // The shot boundaries placed on the timeline: a draft until saved, reset when the saved ones change.
  const savedKey = JSON.stringify(info.shot_points ?? [])
  const [points, setPoints] = useState<ShotPoint[]>(() => info.shot_points ?? [])
  const [pointsBase, setPointsBase] = useState(savedKey)
  useEffect(() => { if (savedKey !== pointsBase) { setPoints(JSON.parse(savedKey) as ShotPoint[]); setPointsBase(savedKey) } }, [savedKey, pointsBase])
  const [selectedMarker, setSelectedMarker] = useState<string | null>(null)
  // The line whose Play button was pressed last; its button shows Pause while the video plays.
  const [playingId, setPlayingId] = useState<string | null>(null)
  const paused = usePlayerValue((_, isPaused) => isPaused)

  const characters = useMemo(() => view.characters || [], [view.characters])
  const options = useMemo(() => speakerOptions(characters), [characters])
  const lines = useMemo(() => draft?.lines ?? [], [draft])
  const problems = useMemo(() => problemsById(lines, duration), [lines, duration])
  const dialogueDirty = !!draft && !!result && !sameContent(draft, result.content)
  const pointsDirty = !samePoints(points, JSON.parse(savedKey) as ShotPoint[])
  const dirty = dialogueDirty || pointsDirty
  // Lines of a cut added at the end are new until the direction is redone; they count as lines to review.
  const newFrom = info.extension ? info.extension.from_time - info.start_time : null
  const stats = counts(lines, newFrom)
  const shown = useMemo(() => visibleLines(lines, filter, newFrom), [lines, filter, newFrom])
  const activeId = usePlayerValue(time => lineAt(lines, time - info.start_time)?.id ?? null)
  const cuts = detail.cuts
  const markers = useMemo(() => boundaryMarkers(detail.timeline?.boundaries ?? [], detail.shots, points, info.start_time), [detail.timeline, detail.shots, points, info.start_time])
  const marker = markers.find(item => item.id === selectedMarker) ?? null
  const colours = useMemo(() => new Map(lines.map((line, index) => [line.id, lineColour(index)])), [lines])
  // Shot boundaries that fall inside a line divide its words between two shots.
  const inside = useMemo(() => new Map(lines.map(line => [line.id, markers.filter(item => line.t0 + 0.05 < item.t && item.t < line.t1 - 0.05).map(item => `shot ${item.shot}`).join(', ')])), [lines, markers])
  const state = view.state
  const compileJob = detail.jobs.find(job => job.step === 'compile' && (job.status === 'queued' || job.status === 'running'))
  const failedJob = !compileJob && view.state === 'none' ? detail.jobs.find(job => job.step === 'compile' && job.status === 'failed') : undefined
  const busy = working || !!detail.jobs.some(job => job.status === 'queued' || job.status === 'running')
  const playhead = useCallback(() => player.time() - info.start_time, [player, info.start_time])

  const edit = useCallback((id: string, change: (line: CompiledLine) => CompiledLine) => setDraft(current => current && { ...current, lines: current.lines.map(line => line.id === id ? change(line) : line) }), [])
  const merge = useCallback((id: string) => setDraft(current => current && { ...current, lines: mergeNext(current.lines, current.lines.findIndex(line => line.id === id)) }), [])
  const split = useCallback((id: string) => { const time = playhead(); setDraft(current => current && { ...current, lines: splitAt(current.lines, current.lines.findIndex(line => line.id === id), time) }) }, [playhead])
  const remove = useCallback((id: string) => setDraft(current => current && { ...current, lines: current.lines.filter(line => line.id !== id) }), [])
  // Play a line, or pause it while it plays; pressed again inside the line, it goes on from the playhead.
  const playingRef = useRef<string | null>(null)
  const play = useCallback((line: CompiledLine) => {
    setSelected(line.id)
    if (!player.paused && playingRef.current === line.id) { player.pause(); return }
    const at = player.time() - info.start_time
    const from = playingRef.current === line.id && at > line.t0 + 0.05 && at < line.t1 - 0.05 ? at : line.t0
    playingRef.current = line.id
    setPlayingId(line.id)
    player.playRange(info.start_time + from, info.start_time + line.t1)
  }, [player, info.start_time])
  const forget = () => { playingRef.current = null; setPlayingId(null) }
  const place = useCallback((item: Marker, t: number) => { setSelectedMarker(item.id); setPoints(current => placePoint(current, item.after_cut_id, info.start_time + t)) }, [info.start_time])
  const release = useCallback((item: Marker) => setPoints(current => placePoint(current, item.after_cut_id, null)), [])
  const pickLine = useCallback((line: CompiledLine) => {
    setSelected(line.id)
    if (!visibleLines([line], filter, newFrom).length) setFilter('all')
    player.seek(info.start_time + line.t0)
    requestAnimationFrame(() => document.getElementById(`line-${line.id}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }))
  }, [player, info.start_time, filter, newFrom])

  // Keyboard: j/k (or the arrows) move between lines, Space plays the selected one; typing in a field is left alone.
  function onKeyDown(event: React.KeyboardEvent) {
    const target = event.target as HTMLElement
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || event.metaKey || event.ctrlKey || event.altKey) return
    const index = shown.findIndex(line => line.id === selected)
    const go = (next: number) => {
      const line = shown[Math.max(0, Math.min(shown.length - 1, next))]
      if (!line) return
      setSelected(line.id); player.seek(info.start_time + line.t0)
      document.getElementById(`line-${line.id}`)?.scrollIntoView({ block: 'nearest' })
    }
    if (event.key === 'j' || event.key === 'ArrowDown') { event.preventDefault(); go(index + 1) }
    else if (event.key === 'k' || event.key === 'ArrowUp') { event.preventDefault(); go(index < 0 ? 0 : index - 1) }
    else if (event.key === ' ' && index >= 0) { event.preventDefault(); play(shown[index]) }
    else if (event.key === 'Escape' && selectedMarker) { event.stopPropagation(); setSelectedMarker(null) }
  }

  // The dialogue and the placed boundaries are saved together; the boundaries do not need the dialogue to be confirmed again.
  const save = () => act('save-dialogue', async () => {
    let latest: SequenceDetail | undefined
    if (dialogueDirty && result && draft) latest = await sequenceApi.revise(result.id, payload(draft))
    if (pointsDirty) latest = await sequenceApi.setShotPoints(info.id, points)
    return latest
  })
  async function confirm() {
    if (!result || !draft) return
    if (state === 'confirmed' && !dirty) { onContinue(); return }
    const ok = await act('confirm-dialogue', async () => {
      let id = result.id
      if (dialogueDirty) id = (await sequenceApi.revise(id, payload(draft))).compilation.result?.id || id
      const confirmed = await sequenceApi.confirmDialogue(info.id, id)
      return pointsDirty ? sequenceApi.setShotPoints(info.id, points) : confirmed
    })
    if (!ok) return
    onContinue()
    // After a cut was added the server directs the sequence again by itself once the dialogue is confirmed.
    if (!detail.bible && !info.extension) void run('direct')
  }
  function rerun() {
    const warning = dirty ? 'Qwen will compile the dialogue again and your unsaved edits will be lost. Continue?' : 'Qwen will compile the dialogue again from the transcript and the subtitles. The current version stays in the history, but must be saved again. Continue?'
    if (window.confirm(warning)) void run('compile')
  }
  function useOriginal() {
    if (window.confirm('Use the original transcript as the dialogue, without corrections? Qwen\'s compilation and your edits will be replaced.')) void act('use-source', () => sequenceApi.useSourceDialogue(info.id))
  }

  const voices = useMemo(() => {
    const labels = [...new Set([...(draft?.voices || []).map(voice => voice.speaker_label), ...lines.map(line => line.speaker_label).filter(Boolean)])]
    return labels.map(label => ({ label, voice: draft?.voices.find(voice => voice.speaker_label === label), count: lines.filter(line => line.speaker_label === label).length }))
  }, [draft, lines])

  const nextCut = detail.next_cut
  const rewrite = !!info.extension?.rewrite_shots || detail.shots.some(shot => shot.result)
  const directed = !!(detail.bible || detail.plan)
  const addBlocked = !nextCut ? '' : !nextCut.fits ? `With cut ${nextCut.number} the sequence would last more than ${clock(detail.limits.sequence_seconds)}`
    : !nextCut.ready ? `Cut ${nextCut.number} needs a first-frame and a keyframe analysis first` : dirty ? 'Save your changes first' : busy ? 'Wait for the running jobs to finish' : ''
  function addCut() {
    if (!nextCut || addBlocked) return
    const then = directed ? ` Once the new lines are saved (at once if nobody speaks in it), the bible and the shot plan are written again${rewrite ? ', then every shot prompt is rewritten and reviewed' : ''}.` : ''
    if (window.confirm(`Add cut ${nextCut.number} (${nextCut.seconds.toFixed(1)} s) to the end of this sequence? Qwen compiles its lines and adds them to your dialogue; your edits stay.${then}`))
      void act('extend', () => sequenceApi.extend(info.id))
  }
  const addButton = nextCut && <span title={addBlocked || `Add cut ${nextCut.number} (${nextCut.seconds.toFixed(1)} s) to the end of the sequence`}>
    <button type="button" className="btn btn-quiet !px-2.5 !py-1 !text-[11px]" disabled={!!addBlocked} onClick={addCut}><Plus size={12} /> Add cut {nextCut.number}</button></span>
  const speechHint = !info.extension && nextCut?.speech ? nextCut.speech : null
  const added = info.extension ? (info.extension.cuts.length > 1 ? `Cuts ${info.extension.cuts[0]}–${info.extension.cuts[info.extension.cuts.length - 1]}` : `Cut ${info.extension.cuts[0]}`) : ''
  const banner = view.stale ? 'The transcript or the subtitles changed after this dialogue was compiled. Re-run Qwen to pick the changes up (your edits would be replaced).' : detail.sequence.cuts_changed ? 'The cuts of this sequence changed. Direct the sequence again after reviewing the dialogue.' : ''

  return <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-5 lg:grid-cols-[minmax(0,24rem)_15rem_minmax(0,1fr)] lg:overflow-hidden" onKeyDown={onKeyDown}>
    <aside className="scrollbar min-h-0 space-y-4 lg:overflow-y-auto lg:pr-1">
      <PlayerSlot className="aspect-video">{!proxyReady && <div className="absolute inset-0 grid place-items-center bg-black/80 p-4 text-center text-xs text-[#aeb5c1]">The preview video is not ready yet.</div>}</PlayerSlot>
      <p className="muted text-[11px] leading-5">Playhead <PlayheadClock origin={info.start_time} /> in the sequence. <kbd className="rounded bg-[#262b35] px-1">J</kbd>/<kbd className="rounded bg-[#262b35] px-1">K</kbd> move between lines, <kbd className="rounded bg-[#262b35] px-1">Space</kbd> plays or pauses the selected line. The crosshair next to a time sets it to the playhead. On the timeline, drag a shot to place where it starts.</p>
      {marker && <BoundaryCard marker={marker} markers={markers} cuts={cuts} lines={lines} origin={info.start_time} duration={duration} disabled={busy}
        onPlace={place} onRelease={release} onClose={() => setSelectedMarker(null)} onSave={!draft && pointsDirty ? () => void save() : undefined} />}
      {draft && <section aria-label="Voices" className="rounded-xl border hairline bg-[#1b202a] p-4">
        <h3 className="text-sm font-semibold">Voices</h3>
        <p className="muted mb-3 text-[11px] leading-5">Decide who owns each voice once; it is applied to every line of that voice, and the bible keeps your decision.</p>
        {voices.length ? <ul className="space-y-2">{voices.map(({ label, voice, count }) => <li key={label} className="flex items-center gap-2">
          <span className="min-w-0 flex-1"><span className="block truncate font-mono text-[11px]">{label}</span><span className="muted text-[10px]">{count} line{count === 1 ? '' : 's'}{voice?.reason ? ` · ${voice.reason}` : ''}</span></span>
          <select className="field !w-36 !py-1 !text-xs" aria-label={`Character for ${label}`} value={voice?.character || ''} onChange={event => setDraft(current => current && applyVoice(current, label, event.target.value || null, characters))}>
            {options.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select></li>)}</ul> : <p className="muted text-xs">No voice labels in this range.</p>}
      </section>}
      {view.subtitles.length > 0 && <details className="rounded-xl border hairline bg-[#1b202a] text-xs"><summary className="cursor-pointer px-4 py-3 font-semibold text-[#aeb5c1] hover:text-white">Subtitles read from the picture ({view.subtitles.length})</summary>
        <ul className="scrollbar max-h-64 space-y-1 overflow-y-auto border-t hairline p-3">{view.subtitles.map(sub => <li key={sub.id}>
          <button type="button" className="w-full text-left hover:text-[#e0a96d]" onClick={() => { forget(); player.playRange(info.start_time + sub.t0, info.start_time + sub.t1) }}><span className="font-mono text-[10px] text-[#8b93a3]">{sub.t0.toFixed(1)}–{sub.t1.toFixed(1)}</span> {sub.text}</button></li>)}</ul></details>}
    </aside>

    {detail.timeline ? <DialogueTimeline duration={duration} origin={info.start_time} cuts={cuts} lines={lines} markers={markers} selectedLine={selected} selectedMarker={selectedMarker} disabled={busy}
      onSelectLine={pickLine} onSelectMarker={item => setSelectedMarker(item.id)} onPlace={place} onRelease={release} />
      : <section className="muted rounded-xl border hairline bg-[#161b24] p-4 text-xs">The cuts of this sequence are missing, so there is no timeline.</section>}

    <section aria-label="Dialogue" className="flex min-h-0 flex-col rounded-xl border hairline bg-[#1b202a]">
      <header className="flex shrink-0 flex-wrap items-center gap-2 border-b hairline px-4 py-3">
        <div className="mr-auto min-w-0">
          <h3 className="text-sm font-semibold">Dialogue</h3>
          <p className="muted text-[11px]">{state === 'confirmed' ? 'Saved: Direction uses this dialogue' : state === 'review' ? 'Compiled by Qwen: review it, then save' : state === 'running' ? 'Qwen is aligning the transcript with the subtitles' : 'Not compiled yet'}
            {result ? ` · ${result.origin === 'model' ? 'Qwen' : result.origin === 'source' ? 'original transcript' : 'edited by you'}${result.generation_seconds ? ` · ${Math.round(result.generation_seconds)} s` : ''}` : ''}</p>
        </div>
        {draft && <>
          <Chip tone={stats.review ? 'warn' : 'good'} title="Lines that are unsure, have no speaker, or continue in the next line">{stats.review} to review</Chip>
          <Chip tone="busy">{stats.changed} corrected</Chip>
          <div className="flex gap-1 rounded-lg border hairline p-0.5" role="tablist" aria-label="Filter lines">{([['all', `All ${stats.total}`], ['review', 'To review'], ['changed', 'Corrected']] as const).map(([value, text]) =>
            <button key={value} role="tab" aria-selected={filter === value} onClick={() => setFilter(value)} className={`rounded-md px-2.5 py-1 !text-[11px] font-semibold ${filter === value ? 'bg-[#3a2f1f] text-white' : 'text-[#aeb5c1] hover:text-white'}`}>{text}</button>)}</div>
        </>}
        {!speechHint && addButton}
      </header>
      {banner && <p role="status" className="flex shrink-0 items-start gap-2 bg-[#4a3a17] px-4 py-2 text-xs leading-5 text-[#ffd59a]"><AlertTriangle size={14} className="mt-0.5 shrink-0" /> {banner}</p>}
      {info.extension && state !== 'running' && <p role="status" className="flex shrink-0 items-start gap-2 bg-[#173a2b] px-4 py-2 text-xs leading-5 text-[#9ff0c8]"><Plus size={14} className="mt-0.5 shrink-0" />
        <span>{added} added. {stats.added ? `Its ${stats.added} line${stats.added > 1 ? 's are' : ' is'} marked new; review ${stats.added > 1 ? 'them' : 'it'}, then Save & continue: ` : 'Save & continue: '}
          {directed ? `the bible and the shot plan are written again${rewrite ? ', then every shot prompt is rewritten and reviewed' : ''}.` : 'the director starts from this dialogue.'}</span></p>}
      {speechHint && nextCut && <p role="status" className="flex shrink-0 flex-wrap items-center gap-2 bg-[#2b2442] px-4 py-2 text-xs leading-5 text-[#d6c8ff]"><AlertTriangle size={14} className="shrink-0" />
        <span className="min-w-0 flex-1">Speech runs on past the end of the sequence into cut {nextCut.number}: “{speechHint.text_before}” <strong className="text-white">“{speechHint.words_after}”</strong></span>{addButton}</p>}

      {state === 'running' ? <div className="flex flex-1 flex-col items-center justify-center gap-3 p-10 text-center"><LoaderCircle size={22} className="animate-spin text-[#d6c8ff]" /><p className="text-sm font-semibold">Qwen is compiling the dialogue…</p><p className="muted max-w-md text-xs leading-5">{compileJob?.stage || 'Queued'}. It matches the transcript with the subtitles, corrects the wording, joins lines that were cut in half and proposes speakers. You review it here when it finishes; whether a speaker is seen speaking is decided per shot later.</p></div>
        : !draft ? <div className="flex flex-1 flex-col items-center justify-center gap-3 p-10 text-center">
          <WandSparkles size={22} className="text-[#e0a96d]" /><p className="text-sm font-semibold">The dialogue has not been compiled yet</p>
          {failedJob?.error && <p role="alert" className="max-w-md text-xs leading-5 text-[#ffaaa7]">The last attempt failed: {failedJob.error}</p>}
          <p className="muted max-w-md text-xs leading-5">Qwen reconciles the recognised speech with the burned-in subtitles so that wording, speakers and timing are right before the sequence is directed.</p>
          <div className="flex gap-2"><button type="button" className="btn btn-primary !text-xs" disabled={busy} onClick={() => void run('compile')}><WandSparkles size={13} /> Compile dialogue</button><button type="button" className="btn btn-quiet !text-xs" disabled={busy} onClick={useOriginal}>Use the original transcript</button></div>
        </div>
          : <>
            <ol className="scrollbar min-h-0 flex-1 space-y-2 overflow-y-auto p-3" aria-label="Dialogue lines">
              {shown.map(line => <Row key={line.id} line={line} selected={selected === line.id} active={activeId === line.id} playing={!paused && playingId === line.id}
                colour={colours.get(line.id) ?? lineColour(0)} shotsInside={inside.get(line.id) ?? ''} problems={problems[line.id] || []} duration={duration} added={isNew(line, newFrom)} options={options} characters={characters}
                onSelect={setSelected} onPlay={play} onEdit={edit} onMerge={merge} onSplit={split} onDelete={remove} playhead={playhead} />)}
              {!shown.length && <li className="muted p-6 text-center text-sm">{lines.length ? 'No line matches this filter.' : 'Nobody speaks in this range.'}</li>}
            </ol>
            {draft.open_questions.length > 0 && <div className="shrink-0 border-t hairline px-4 py-2"><p className="label !mb-1">Open questions from Qwen</p><ul className="list-disc space-y-0.5 pl-5 text-xs leading-5 text-[#ffd59a]">{draft.open_questions.map((question, index) => <li key={index}>{question}</li>)}</ul></div>}
            <footer className="flex shrink-0 flex-wrap items-center gap-2 border-t hairline p-3">
              <span className="mr-auto text-[11px] text-[#aeb5c1]">{dirty ? 'Unsaved changes' : state === 'confirmed' ? <span className="inline-flex items-center gap-1 text-[#9ff0c8]"><Check size={12} /> Saved</span> : 'No changes'}{Object.keys(problems).length ? ` · ${Object.keys(problems).length} line${Object.keys(problems).length > 1 ? 's' : ''} need fixing` : ''}
                {points.length > 0 && ` · ${points.length} shot boundar${points.length > 1 ? 'ies' : 'y'} placed by you`}{pointsDirty && detail.plan ? ' · saving moves the planned shots, which must then be written again' : ''}</span>
              <button type="button" className="btn btn-quiet !py-2 !text-xs" disabled={busy} onClick={useOriginal} title="Replace everything with the original transcript, uncorrected">Use original transcript</button>
              <button type="button" className="btn btn-quiet !py-2 !text-xs" disabled={busy} onClick={rerun} title="Qwen compiles the dialogue again"><RefreshCw size={13} /> Re-run Qwen</button>
              <button type="button" className="btn btn-quiet !py-2 !text-xs" disabled={busy || !dirty || Object.keys(problems).length > 0} onClick={() => void save()}><Save size={13} /> Save</button>
              <button type="button" className="btn btn-primary !py-2 !text-xs" disabled={busy || Object.keys(problems).length > 0} onClick={() => void confirm()}
                title="Save this dialogue as final; the director starts from it">{state === 'confirmed' && !dirty ? 'Continue to Direction' : 'Save & continue to Direction'}</button>
            </footer>
          </>}
    </section>
  </div>
}
