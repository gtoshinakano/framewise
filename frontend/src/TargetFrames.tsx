import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, PencilLine, ScanFace, Trash2, XCircle } from 'lucide-react'
import { api, type ContextItem, type KeyframeAnalysis, type KeyframeAnalysisState, type KeyframeSegment } from './api'
import { BoxEditor, type EditorBox } from './BoxEditor'
import { boxColor } from './boxColors'
import { cachedCharacters, loadCharacters } from './runCache'
import { ActionButton, StatusChip } from './RunUi'
import { boxesInFrame, canDraw, detectReason, framesPerSlot, isJobActive, mergeTargets, slotChip, slotName, type Box, type TargetBox, type TargetDetections, type TargetFrameInfo, type TargetRun } from './targetView'

type TargetFrame = { source_frame_index: number; relative_timestamp_seconds: number; reason: string }
export type Selection = { revision?: string | null; status: string; error?: string | null; result_status?: string; result: null | { shot_change_score: number; change_level: string; coverage_limited: boolean; selected_frames: TargetFrame[] } }

export function useTargetSelection(cutId: string, boundaryKey: string) {
  const [selection, setSelection] = useState<Selection | null>(null)
  const [error, setError] = useState('')
  const current = useRef(0)
  const load = useCallback(() => {
    const request = ++current.current
    void api<Selection>(`/cuts/${cutId}/keyframe-selection`).then(value => {
      if (request === current.current) { setSelection(value); setError('') }
    }).catch(reason => { if (request === current.current) setError((reason as Error).message) })
  }, [cutId])
  useEffect(() => {
    setSelection(null); setError(''); load()
    return () => { current.current++ }
  }, [load, boundaryKey])
  useEffect(() => {
    window.addEventListener('framewise:keyframes-changed', load)
    return () => window.removeEventListener('framewise:keyframes-changed', load)
  }, [load])
  useEffect(() => {
    if (selection?.status !== 'queued' && selection?.status !== 'running') return
    const timer = setInterval(load, 3000)
    return () => clearInterval(timer)
  }, [load, selection?.status])
  return { selection, error }
}

/**
 * The people found on the evaluated frames. `revision` is the evaluation's key: a different set of frames
 * deletes what was found on the old ones, so the tab reads again when it changes. `onChanged` runs when a
 * detection job finishes, so the timeline's unknown-people marker follows.
 */
export function useTargetDetections(cutId: string, boundaryKey: string, revision: string, onChanged?: () => void) {
  const [data, setData] = useState<TargetDetections | null>(null)
  const [error, setError] = useState('')
  const current = useRef(0)
  const changed = useRef(onChanged); changed.current = onChanged
  const load = useCallback(async () => {
    const request = ++current.current
    try {
      const value = await api<TargetDetections>(`/cuts/${cutId}/target-detections`)
      if (request === current.current) { setData(value); setError('') }
    } catch (reason) { if (request === current.current) setError((reason as Error).message) }
  }, [cutId])
  useEffect(() => {
    setData(null); setError(''); void load()
    return () => { current.current++ }
  }, [load, boundaryKey, revision])
  useEffect(() => {
    const reload = () => void load()
    window.addEventListener('framewise:keyframes-changed', reload)
    return () => window.removeEventListener('framewise:keyframes-changed', reload)
  }, [load])
  const active = isJobActive(data?.job)
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => void load(), 3000)
    return () => clearInterval(timer)
  }, [active, load])
  const wasActive = useRef(false)
  useEffect(() => {
    if (wasActive.current && !active) changed.current?.()
    wasActive.current = active
  }, [active])
  return { data, error, reload: load, setData }
}
export type TargetDetectionState = ReturnType<typeof useTargetDetections>

/** The cut's keyframe analysis (what changes between these frames); read again when the frames change or a job finishes. */
export function useKeyframeAnalysis(cutId: string, revision: string) {
  const [data, setData] = useState<KeyframeAnalysisState | null>(null)
  const current = useRef(0)
  const load = useCallback(() => {
    const request = ++current.current
    void api<KeyframeAnalysisState>(`/cuts/${cutId}/keyframe-analysis`).then(value => { if (request === current.current) setData(value) }).catch(() => {})
  }, [cutId])
  useEffect(() => { setData(null); load(); return () => { current.current++ } }, [load, revision])
  useEffect(() => {
    window.addEventListener('framewise:keyframes-changed', load)
    return () => window.removeEventListener('framewise:keyframes-changed', load)
  }, [load])
  const active = isJobActive(data?.job)
  useEffect(() => {
    if (!active) return
    const timer = setInterval(load, 5000)
    return () => clearInterval(timer)
  }, [active, load])
  return data
}

const CAMERA_TEXT: Record<string, string> = { static: 'static camera', unclear: 'camera not measurable' }
const cameraText = (move: string) => CAMERA_TEXT[move] ?? move.replace('_', ' ')

/** What changed from one keyframe to the next: the measured camera, then per character the visible motions and changed parts. */
function PairChanges({ segment, slotOf }: { segment: KeyframeSegment; slotOf: (name: string) => number | null }) {
  const people = segment.characters.filter(person => person.changes.length || ['walking', 'speaking', 'blinking'].some(flag => person[flag as 'walking'] === 'yes'))
  return <div className="space-y-1 rounded-lg border border-dashed hairline px-2 py-1.5 text-[11px]" aria-label={`Changes ${segment.from} to ${segment.to}`}>
    <div className="flex flex-wrap items-center gap-1.5"><strong className="text-[#d5dbe5]">{segment.from}→{segment.to}</strong><span className="muted">{segment.start.toFixed(2)}–{segment.end.toFixed(2)}s</span>
      <StatusChip tone={segment.camera.move === 'static' || segment.camera.move === 'unclear' ? 'grey' : 'purple'} title="Measured from the background pixels, people masked out">{cameraText(segment.camera.move)}</StatusChip></div>
    {!people.length && !segment.other_changes.length && <p className="muted">No visible change.</p>}
    {people.map(person => {
      const slot = slotOf(person.name)
      return <div key={person.name} className="pl-2" style={slot !== null ? { borderLeft: `3px solid ${boxColor(slot)}` } : undefined}>
        <span className="font-semibold text-[#d5dbe5]">{person.name}</span>
        {(['walking', 'speaking', 'blinking'] as const).filter(flag => person[flag] === 'yes').map(flag => <span key={flag} className="ml-1.5 rounded bg-[#2a3140] px-1 py-0.5 text-[10px]">{flag}</span>)}
        <ul className="mt-0.5 space-y-0.5">{person.changes.map((change, index) => <li key={index}><span className="muted">{change.part.replace('_', ' ')}:</span> {change.description}
          <span className={`ml-1 text-[10px] ${change.source === 'dwpose' ? 'text-[#7bd88f]' : 'text-[#8b93a3]'}`} title={change.source === 'dwpose' ? 'Measured from DWPose body keypoints' : 'Read from the images by Qwen3.5'}>{change.source === 'dwpose' ? 'keypoints' : 'vision model'}</span></li>)}</ul>
      </div>
    })}
    {segment.other_changes.map((other, index) => <p key={index}><span className="muted">{other.kind}:</span> {other.description}</p>)}
  </div>
}

const frameUrl = (cutId: string, ordinal: number, revision?: string | null) => `/api/cuts/${encodeURIComponent(cutId)}/keyframe-selection/frames/${ordinal}?revision=${encodeURIComponent(revision || '')}`

export function TargetFrames({ cutId, selection, error, projectId, detections, episodeBusy, onChanged }: {
  cutId: string; selection: Selection | null; error: string; projectId: string
  detections: TargetDetectionState; episodeBusy: boolean; onChanged: () => void
}) {
  const { data, error: loadError, reload, setData } = detections
  const [characters, setCharacters] = useState<ContextItem[]>(() => cachedCharacters(projectId) ?? [])
  const [editError, setEditError] = useState('')
  const [saving, setSaving] = useState(0)
  const [queueing, setQueueing] = useState(false)
  const [drawing, setDrawing] = useState<number | null>(null)
  const chain = useRef<Promise<void>>(Promise.resolve())
  useEffect(() => { void loadCharacters(projectId).then(setCharacters).catch(() => {}) }, [projectId])

  const run = data?.run || null
  const keyframes = useKeyframeAnalysis(cutId, selection?.revision || '')
  const analysis = keyframes?.result ?? null
  const slotOf = (name: string) => live?.slots.find(slot => slot.character_name === name)?.slot ?? null
  const active = isJobActive(data?.job)
  const evaluated = !!selection?.result
  const live = run && !run.stale ? run : null
  const locked = saving > 0 || active || !live
  const reason = detectReason({ evaluated, episodeBusy, pending: active, saving: saving > 0 })

  /** One edit at a time, in order. Changes paint at once; the server's answer replaces them. */
  function edit(task: () => Promise<unknown>, paint?: (current: TargetRun) => TargetRun) {
    if (paint) setData(value => value && value.run ? { ...value, run: paint(value.run) } : value)
    setSaving(count => count + 1); setEditError('')
    chain.current = chain.current.then(async () => {
      try { await task() } catch (reason) { setEditError(`Could not save: ${(reason as Error).message}`) } finally {
        setSaving(count => count - 1)
        await reload(); onChanged()
      }
    })
  }
  const label = (slot: number, characterId: string | null) => edit(
    () => api(`/target-slots/${live!.id}/${slot}`, { method: 'PUT', body: JSON.stringify({ character_context_id: characterId }) }),
    current => ({ ...current, slots: current.slots.map(item => item.slot === slot ? { ...item, character_context_id: characterId, character_name: characters.find(entry => entry.id === characterId)?.title ?? null, label_source: characterId ? 'manual' : null } : item) }))
  const move = (id: string, box: Box) => edit(
    () => api(`/target-detections/${id}/box`, { method: 'PUT', body: JSON.stringify({ box }) }),
    current => ({ ...current, detections: current.detections.map(item => item.id === id ? { ...item, box, source: 'manual' } : item) }))
  const remove = (id: string) => edit(
    () => api(`/target-detections/${id}`, { method: 'DELETE' }),
    current => ({ ...current, detections: current.detections.filter(item => item.id !== id) }))
  /** The detector took one person for two colours: all of `slot`'s boxes join `into`, and `slot` goes away. */
  function merge(slot: number, into: number) {
    const from = live!.slots.find(item => item.slot === slot), to = live!.slots.find(item => item.slot === into)
    if (!window.confirm(`Merge colour ${slot + 1}${from?.character_name ? ` (${from.character_name})` : ''} into colour ${into + 1}${to?.character_name ? ` (${to.character_name})` : ''}? All its boxes become colour ${into + 1} and colour ${slot + 1} is removed.`)) return
    edit(() => api(`/target-slots/${live!.id}/${slot}/merge`, { method: 'POST', body: JSON.stringify({ into }) }))
  }
  const reslot = (id: string, slot: number) => edit(() => api(`/target-detections/${id}/slot`, { method: 'PUT', body: JSON.stringify({ slot }) }))
  const draw = (ordinal: number, box: Box) => { setDrawing(null); edit(() => api(`/target-runs/${live!.id}/boxes`, { method: 'POST', body: JSON.stringify({ frame_ordinal: ordinal, box }) })) }
  async function detect() {
    setQueueing(true); setEditError('')
    try { await api(`/cuts/${cutId}/target-detection`, { method: 'POST' }); await reload(); onChanged() } catch (failure) { setEditError((failure as Error).message) } finally { setQueueing(false) }
  }

  const counts = live ? framesPerSlot(live) : new Map<number, number>()
  const framesWithBoxes = live ? new Set(live.detections.map(item => item.frame_ordinal)).size : 0
  return <section aria-label="Target frames" className="space-y-3 rounded-xl border hairline bg-[#171c25] p-3">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h3 className="text-sm font-semibold">Target frames</h3>
      {evaluated && <ActionButton compact icon={<ScanFace size={13} />} pending={queueing || active} disabled={!!reason} title={reason || (live ? 'Find the people again. The names you set and the boxes you drew are kept.' : 'Find the people on these frames, in the order of the first frame')} onClick={() => void detect()}>{active ? 'Detecting' : live ? 'Detect again' : 'Detect characters'}</ActionButton>}
    </div>
    {error && <p role="alert" className="text-xs text-[#ffb6bc]">{error}</p>}
    {!selection && !error && <p className="muted text-xs">Loading target frames…</p>}
    {selection?.status === 'missing' && <p className="muted text-xs">Evaluate this cut in Run settings to select target frames.</p>}
    {(selection?.status === 'queued' || selection?.status === 'running') && <p role="status" className="text-xs text-[#c3aeff]">{selection.status === 'queued' ? 'Evaluation queued' : 'Evaluating this cut…'}</p>}
    {selection?.status === 'failed' && <p role="alert" className="text-xs text-[#ffb6bc]">Evaluation failed: {selection.error}</p>}
    {selection?.status === 'cancelled' && <p className="muted text-xs">Evaluation cancelled.</p>}
    {selection && !selection.result && selection.result_status === 'stale' && selection.status === 'done' && <p className="muted text-xs">Selection is stale. Evaluate the cut again.</p>}
    {selection?.result && <>
      <p className="muted text-xs">Change: {selection.result.change_level} ({selection.result.shot_change_score.toFixed(1)} / 100)</p>
      {selection.result.coverage_limited && <p className="text-xs text-[#e6c58b]">Some detected changes may not be represented in the selected frames.</p>}
      {active && <p role="status" className="flex items-center gap-2 rounded-lg border border-[#4a4363] bg-[#221c33] px-3 py-2 text-xs text-[#c3aeff]"><Loader2 size={14} className="animate-spin" aria-hidden />{data?.job?.stage || 'Detection queued'}…</p>}
      {data?.job?.status === 'failed' && <p role="alert" className="flex items-start gap-2 rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-xs text-[#ffb6bc]"><XCircle size={14} className="mt-0.5 shrink-0" aria-hidden />{data.job.error || 'Detection failed.'}</p>}
      {(editError || loadError) && <p role="alert" className="text-xs text-[#ff9fa7]">{editError || loadError}</p>}
      {run?.stale && <p className="rounded-lg border border-[#6b5a2f] bg-[#2a2314] px-3 py-2 text-xs text-[#e6c58b]">The cut's timing or evaluation changed since these people were found. Detect characters again; evaluating different frames removes them.</p>}
      {!run && !active && data && <p className="muted rounded-lg border border-dashed hairline px-3 py-3 text-center text-xs">No people found on these frames yet. Press Detect characters. The first frame's boxes set the order and the colours.</p>}
      {live && <div className="space-y-1.5 rounded-lg border hairline p-2" role="group" aria-label="People in this cut">
        <div className="flex flex-wrap items-center justify-between gap-1"><strong className="text-xs text-[#d5dbe5]">People · {live.slots.length}</strong><span className="muted text-[11px]">One colour is one person in every frame. A name applies to all of its boxes and is never used as a matching example.</span></div>
        {!live.slots.length && <p className="muted text-xs">Nobody was found on these frames. Use Draw box on a frame to add someone.</p>}
        {live.slots.map(slot => {
          const chip = slotChip(slot)
          const joins = mergeTargets(live, slot.slot)
          return <div key={slot.slot} className="flex flex-wrap items-center gap-1.5 rounded-lg bg-[#1e2430] p-1.5 text-xs" style={{ borderLeft: `4px solid ${boxColor(slot.slot)}` }}>
            <span className="w-5 shrink-0 text-center font-semibold" style={{ color: boxColor(slot.slot) }}>{slot.slot + 1}</span>
            <select aria-label={`Character for colour ${slot.slot + 1}`} className="field !w-auto min-w-32 flex-1 !px-2 !py-1.5 !text-xs" value={slot.character_context_id || ''} disabled={locked}
              onChange={event => label(slot.slot, event.target.value || null)}><option value="">Unknown</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select>
            <StatusChip tone={chip.tone} title={chip.title}>{chip.text}</StatusChip>
            {joins.length > 0 && <select aria-label={`Colour ${slot.slot + 1} is the same person as`} className="field !w-auto !px-2 !py-1.5 !text-xs" value="" disabled={locked}
              title="The detector took one person for two colours. Pick the colour they really are: all these boxes join it. Where both have a box in a frame, the one you drew or moved stays."
              onChange={event => { if (event.target.value !== '') merge(slot.slot, Number(event.target.value)) }}><option value="">Same person as…</option>{joins.map(item => <option key={item.slot} value={item.slot}>{item.slot + 1}. {slotName(item)}</option>)}</select>}
            <span className="muted text-[11px]">{slot.origin === 'first_frame' ? 'In the first frame' : 'Appears later'} · {counts.get(slot.slot) || 0} of {framesWithBoxes || live.frames.length} frames</span>
          </div>
        })}
      </div>}
      {analysis && <KeyframeHeader state={keyframes!} analysis={analysis} />}
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">{selection.result.selected_frames.map((frame, index) => {
        const info = live?.frames.find(item => item.ordinal === frame.source_frame_index)
        const described = keyframes?.status === 'current' ? analysis?.frames.find(item => item.ordinal === frame.source_frame_index) : undefined
        const after = described ? analysis?.segments.find(segment => segment.from === described.label) : undefined
        return <div key={frame.source_frame_index} className="space-y-1.5">
          <FrameCard index={index} frame={frame} src={frameUrl(cutId, frame.source_frame_index, selection.revision)} info={info} run={live}
            drawing={drawing === frame.source_frame_index} locked={locked} onDrawing={on => setDrawing(on ? frame.source_frame_index : null)}
            onDraw={box => draw(frame.source_frame_index, box)} onMove={move} onRemove={remove} onReslot={reslot} described={described} slotOf={slotOf} />
          {after && <PairChanges segment={after} slotOf={slotOf} />}
        </div>
      })}</div>
    </>}
  </section>
}

/** Which keyframe analysis this tab shows, and whether it still belongs to these frames and people. */
function KeyframeHeader({ state, analysis }: { state: KeyframeAnalysisState; analysis: KeyframeAnalysis }) {
  const method = analysis.method === 'hybrid' ? 'Qwen3.5 + DWPose keypoints' : 'Qwen3.5 only (DWPose not installed)'
  return <div className="space-y-1 rounded-lg border hairline px-3 py-2 text-xs">
    <div className="flex flex-wrap items-center gap-1.5"><strong className="text-[#d5dbe5]">Keyframe analysis</strong><StatusChip tone="grey" title="How the body changes were found">{method}</StatusChip>
      {state.run?.roster_source && <StatusChip tone="grey" title="Where the names came from">{state.run.roster_source === 'target_slots' ? 'Names from these frames' : state.run.roster_source === 'first_frame' ? 'Names from the first frame' : 'No names'}</StatusChip>}
      {state.status === 'stale' && <StatusChip tone="amber" title="The frames were evaluated again after this analysis; run Keyframes again">Stale</StatusChip>}</div>
    {analysis.summary && <p className="muted">{analysis.summary}</p>}
    {state.status === 'stale' && <p className="text-[#e6c58b]">These frames changed since the analysis; its states and changes are hidden until Keyframes runs again.</p>}
  </div>
}

function FrameCard({ index, frame, src, info, run, drawing, locked, onDrawing, onDraw, onMove, onRemove, onReslot, described, slotOf }: {
  index: number; frame: TargetFrame; src: string; info?: TargetFrameInfo; run: TargetRun | null; drawing: boolean; locked: boolean
  onDrawing: (on: boolean) => void; onDraw: (box: Box) => void; onMove: (id: string, box: Box) => void; onRemove: (id: string) => void; onReslot: (id: string, slot: number) => void
  described?: KeyframeAnalysis['frames'][number]; slotOf: (name: string) => number | null
}) {
  const boxes: TargetBox[] = run && info ? boxesInFrame(run, frame.source_frame_index) : []
  const editor: EditorBox[] = boxes.map(item => ({ id: item.id, box: item.box, color: boxColor(item.slot), label: `${item.slot + 1}. ${slotName(run?.slots.find(slot => slot.slot === item.slot))}` }))
  const mirrored = !!info?.mirror
  const room = !!run && canDraw(run, frame.source_frame_index)
  return <figure className="overflow-hidden rounded-lg border hairline bg-black">
    <div className="relative w-full" style={{ aspectRatio: info ? `${info.image_size[0]} / ${info.image_size[1]}` : '16 / 9' }}>
      <img loading="lazy" className="absolute inset-0 h-full w-full object-contain" src={src} alt={`Target frame ${index + 1} at ${frame.relative_timestamp_seconds.toFixed(3)} seconds`} />
      {run && info && <BoxEditor size={info.image_size} boxes={editor} editable={!locked && !mirrored} drawing={drawing && !locked && !mirrored} label={`Frame ${index + 1} character boxes`}
        onDrawn={onDraw} onChange={onMove} onRemove={onRemove} />}
    </div>
    <figcaption className="space-y-1.5 p-2 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2"><span>Frame {index + 1} · {frame.relative_timestamp_seconds.toFixed(3)}s</span>
        <span className="flex items-center gap-2"><span className="muted">{frame.reason.replaceAll('_', ' ')}</span>
          {run && info && <>
            <span className="muted tabular-nums" title={mirrored ? 'Every box of the first frame' : 'A frame holds at most three boxes'}>{mirrored ? boxes.length : `${boxes.length}/3`}</span>
            {!mirrored && <button type="button" className="inline-flex items-center gap-1 text-[#c3aeff] hover:underline disabled:opacity-40" disabled={locked || (!room && !drawing)} title={room ? 'Draw a box for someone the detector missed' : 'A frame holds at most three boxes'} onClick={() => onDrawing(!drawing)}>{drawing ? <XCircle size={12} aria-hidden /> : <PencilLine size={12} aria-hidden />}{drawing ? 'Cancel drawing' : 'Draw box'}</button>}
          </>}</span></div>
      {mirrored && <p className="muted text-[11px]">The first frame, as you left it in Characters in this cut. Change its boxes there, then detect again.</p>}
      {described && described.characters.length > 0 && <ul className="space-y-0.5 text-[11px]" aria-label={`What ${described.label} shows`}>{described.characters.map(person => {
        const slot = slotOf(person.name)
        return <li key={person.name} className="pl-2" style={slot !== null ? { borderLeft: `3px solid ${boxColor(slot)}` } : undefined}>
          <span className="font-semibold text-[#d5dbe5]">{described.label} · {person.name}</span>{person.state && <span>: {person.state}</span>}
          {person.pose && <span className="muted block" title="Measured from DWPose body keypoints">Keypoints: {person.pose}</span>}
        </li>
      })}</ul>}
      {!mirrored && boxes.map(item => <div key={item.id} className="flex items-center gap-1.5 rounded bg-[#1e2430] p-1" style={{ borderLeft: `4px solid ${boxColor(item.slot)}` }}>
        <select aria-label={`Person for the box in frame ${index + 1}`} className="field !w-auto min-w-28 flex-1 !px-2 !py-1 !text-xs" value={item.slot} disabled={locked} title="Say this box is someone else. If they already have a box in this frame the two swap."
          onChange={event => onReslot(item.id, Number(event.target.value))}>{run!.slots.map(slot => <option key={slot.slot} value={slot.slot}>{slot.slot + 1}. {slotName(slot)}</option>)}</select>
        {item.source === 'manual' && <StatusChip tone="green" title="You drew or changed this box; detecting again keeps it">Yours</StatusChip>}
        <ActionButton compact variant="danger" icon={<Trash2 size={12} />} disabled={locked} onClick={() => onRemove(item.id)} aria-label={`Remove box in frame ${index + 1}`} title="Remove this box" />
      </div>)}
    </figcaption>
  </figure>
}
