import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { Bookmark, BookmarkMinus, CheckCircle2, ChevronDown, Info, Loader2, PencilLine, ScanFace, Trash2, UserPlus, UserX, XCircle } from 'lucide-react'
import { api, type ContextItem, type Job } from './api'
import { cachedCharacters, cachedReferences, invalidateReferences, loadCharacters, loadReferences, type CharacterReference } from './runCache'
import { ActionButton, Elapsed, Skeleton, StatusChip } from './RunUi'
import type { CharacterSummary } from './runState'
import { PoseDetailCard } from './PoseDetailCard'
import { NewCharacterDialog, type NewCharacter, type NewCharacterResult } from './NewCharacterDialog'
import { BOX_COLORS } from './boxColors'

type Frame = { position: 'first' | 'last'; time: number; image_size: [number, number] }
type Candidate = { character_id: string; character_name: string; score: number }
type Detection = { id: string; position: 'first' | 'last'; box: [number, number, number, number]; character_context_id: string | null; character_name: string | null; label_source: string | null; match_reason: string | null; is_reference: boolean; candidates?: Candidate[]; recognition_use?: number | null; gallery_status?: string | null; gallery_note?: string | null }
type DetectionRun = { id: string; job_id: string | null; cut_start: number; cut_end: number; stale: boolean; frames: Frame[]; detections: Detection[] }
const NEW_CHARACTER = '__new'
type Box = [number, number, number, number]
type Resize = { id: string; handle: string; box: Box; from?: [number, number]; orig?: Box }
const HANDLES: [string, number, number][] = [['nw', 0, 0], ['n', .5, 0], ['ne', 1, 0], ['e', 1, .5], ['se', 1, 1], ['s', .5, 1], ['sw', 0, 1], ['w', 0, .5]]
const CURSORS: Record<string, string> = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize' }

// A save continues after navigating away. A reopened cut waits for it before loading its boxes.
const inFlightEdits = new Map<string, Promise<void>>()
const editErrors = new Map<string, string>()
let nextTemporaryBoxId = 0

export const CharacterDetectionPanel = memo(function CharacterDetectionPanel({ cutId, cutStart, cutEnd, projectId, jobs, onStarted, onSummary, noCharacters = false, boundaryKey = '' }: { cutId: string; cutStart: number; cutEnd: number; projectId: string; jobs: Job[]; onStarted: () => Promise<void>; onSummary?: (summary: CharacterSummary | null) => void; noCharacters?: boolean; boundaryKey?: string }) {
  const [run, setRun] = useState<DetectionRun | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [characters, setCharacters] = useState<ContextItem[]>(() => cachedCharacters(projectId) ?? [])
  const [references, setReferences] = useState<CharacterReference[]>(() => cachedReferences(projectId) ?? [])
  const [error, setError] = useState(() => editErrors.get(cutId) || '')
  const [notice, setNotice] = useState('')
  const [noChars, setNoChars] = useState(noCharacters)
  useEffect(() => { if (!pendingSaves.current) setNoChars(noCharacters) }, [noCharacters, cutId])
  const [working, setWorking] = useState(false)
  const [saving, setSaving] = useState(0)
  const [priorSaving, setPriorSaving] = useState(() => inFlightEdits.has(cutId))
  const pendingSaves = useRef(0)
  const failedSaves = useRef(false)
  const editVersion = useRef(0)
  const createdIds = useRef(new Map<string, string>())
  // Serialize edits so a later drag, label, or deletion cannot overtake an earlier one.
  // The local model stays interactive while requests wait in the background.
  function enqueueEdit(task: () => Promise<void>) {
    editErrors.delete(cutId); setError('')
    pendingSaves.current++
    editVersion.current++
    setSaving(pendingSaves.current)
    const queued = (inFlightEdits.get(cutId) || Promise.resolve()).then(async () => {
      try { await task() } catch (reason) {
        failedSaves.current = true
        const message = `Could not save a character edit: ${(reason as Error).message}. Reloaded saved boxes.`
        editErrors.set(cutId, message); setError(message)
      } finally {
        pendingSaves.current--
        setSaving(pendingSaves.current)
        if (!pendingSaves.current && failedSaves.current) {
          const version = editVersion.current
          try {
            const latest = await api<DetectionRun | null>(`/cuts/${cutId}/character-detections`)
            if (version === editVersion.current && !pendingSaves.current) {
              failedSaves.current = false
              setRun(latest); setLoaded(true); void onStarted()
            }
          } catch (reason) { setError(`Could not reload boxes: ${(reason as Error).message}`) }
        }
      }
    })
    inFlightEdits.set(cutId, queued)
    void queued.then(() => { if (inFlightEdits.get(cutId) === queued) inFlightEdits.delete(cutId) })
  }
  const savedId = (id: string) => createdIds.current.get(id) || id
  const [detecting, setDetecting] = useState(false)
  const [drawing, setDrawing] = useState<'first' | 'last' | null>(null)
  const [start, setStart] = useState<[number, number] | null>(null)
  const [current, setCurrent] = useState<[number, number] | null>(null)
  const [resize, setResize] = useState<Resize | null>(null)
  // Last pointer position over a frame, in image pixels, for the "@" shortcut.
  const hover = useRef<{ position: 'first' | 'last'; point: [number, number] } | null>(null)
  const job = jobs.find(item => item.pipeline === 'character_detect' && item.cuts?.some(cut => cut.id === cutId))
  const active = job?.status === 'queued' || job?.status === 'running'
  const episodeBusy = jobs.some(item => item.status === 'queued' || item.status === 'running')
  // Detections are per cut; characters and references are per project and come from the shared cache first.
  const loadRun = useCallback(async () => { setRun(await api<DetectionRun | null>(`/cuts/${cutId}/character-detections`)); setLoaded(true) }, [cutId])
  const loadProject = useCallback(async () => {
    const [items, refs] = await Promise.all([loadCharacters(projectId), loadReferences(projectId)])
    setCharacters(items); setReferences(refs)
  }, [projectId])
  useEffect(() => {
    let alive = true
    if (pendingSaves.current) return
    setLoaded(false)
    const pending = inFlightEdits.get(cutId)
    if (pending) setPriorSaving(true)
    void (pending || Promise.resolve()).then(async () => {
      if (!alive) return
      if (editErrors.has(cutId)) setError(editErrors.get(cutId)!)
      setPriorSaving(false)
      await loadRun()
    }).catch(reason => { if (alive) { setPriorSaving(false); setError((reason as Error).message); setLoaded(true) } })
    return () => { alive = false }
  }, [loadRun, job?.status, job?.id, boundaryKey, cutId])
  useEffect(() => { void loadProject().catch(reason => setError((reason as Error).message)) }, [loadProject])
  useEffect(() => {
    if (!run) { onSummary?.((saving || priorSaving) ? { boxes: 0, autoUnverified: 0, unknown: 0, stale: false, saving: true } : null); return }
    const firstFrameDetections = run.detections.filter(item => item.position === 'first')
    const named = firstFrameDetections.filter(item => item.character_context_id)
    // A cut marked "no characters" is settled: its boxes are ignored, so they raise no warnings.
    if (noChars) { onSummary?.({ boxes: 0, autoUnverified: 0, unknown: 0, stale: false, saving: saving > 0 || priorSaving }); return }
    onSummary?.({ boxes: firstFrameDetections.length, autoUnverified: named.filter(item => item.label_source === 'auto').length, unknown: firstFrameDetections.length - named.length, stale: run.stale, saving: saving > 0 || priorSaving })
  }, [run, noChars, saving, priorSaving, onSummary])
  useEffect(() => () => onSummary?.(null), [onSummary])
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => { void api<{ id: string; status: string }>(`/jobs/${job.id}/character-detection-status`).then(value => { if (value.status !== 'queued' && value.status !== 'running') void onStarted() }).catch(() => {}) }, 3000)
    return () => clearInterval(timer)
  }, [active, job?.id, onStarted])
  /** Runs a change, then reloads only this cut's detections (and the references when `refs` is set). */
  async function action(fn: () => Promise<unknown>, refs = false) {
    setWorking(true); setError('')
    try {
      await fn()
      if (refs) { invalidateReferences(projectId); await Promise.all([loadRun(), loadReferences(projectId).then(setReferences)]) } else await loadRun()
    } catch (reason) { setError((reason as Error).message) } finally { setWorking(false) }
  }
  /** Changes paint immediately; the queue preserves their order on the server. */
  function label(detectionId: string, characterId: string | null) {
    const name = characters.find(item => item.id === characterId)?.title ?? null
    setRun(value => value && { ...value, detections: value.detections.map(item => item.id === detectionId ? { ...item, character_context_id: characterId, character_name: name, label_source: characterId ? 'manual' : null, match_reason: null } : item) })
    enqueueEdit(async () => { await api(`/character-detections/${savedId(detectionId)}`, { method: 'PUT', body: JSON.stringify({ character_context_id: characterId }) }) })
  }
  /** Creates a character from a box in the edit queue, so it waits for earlier edits and a just-drawn box's id. A refusal goes back to the dialog. */
  function createCharacter(detectionId: string, input: NewCharacter) {
    return new Promise<NewCharacterResult>((resolve, reject) => {
      enqueueEdit(async () => {
        let result: NewCharacterResult
        try { result = await api<NewCharacterResult>(`/character-detections/${savedId(detectionId)}/new-character`, { method: 'POST', body: JSON.stringify(input) }) } catch (reason) { reject(reason); return }
        const character = result.character
        setRun(value => value && { ...value, detections: value.detections.map(item => item.id === detectionId ? { ...item, character_context_id: character.id, character_name: character.title, label_source: 'manual', match_reason: null, gallery_status: null, gallery_note: null } : item) })
        setCharacters(items => [...items, character])
        resolve(result)
        void loadCharacters(projectId).then(setCharacters).catch(() => {})
        void onStarted()
      })
    })
  }
  const [adding, setAdding] = useState<string | null>(null)
  function setMatchingUse(detectionId: string, use: boolean | null) {
    setRun(value => value && { ...value, detections: value.detections.map(item => item.id === detectionId ? { ...item, recognition_use: use === null ? null : Number(use), gallery_status: null, gallery_note: null } : item) })
    enqueueEdit(async () => { await api(`/character-detections/${savedId(detectionId)}/recognition-use`, { method: 'PUT', body: JSON.stringify({ use }) }) })
  }
  function removeBox(detectionId: string) {
    setRun(value => value && { ...value, detections: value.detections.filter(item => item.id !== detectionId) })
    enqueueEdit(async () => { await api(`/character-detections/${savedId(detectionId)}`, { method: 'DELETE' }) })
  }
  async function removeStaleFrame(position: 'first' | 'last') {
    if (!run?.stale) return
    setWorking(true); setError(''); setNotice('')
    try {
      const result = await api<{ removed_detections: number }>(`/character-detection-runs/${run.id}/frame/${position}`, { method: 'DELETE' })
      const count = result.removed_detections
      setNotice(`Removed the outdated ${position} frame and ${count} ${count === 1 ? 'character box' : 'character boxes'}. Detect characters again to analyze the current boundary.`)
      await loadRun()
      void onStarted()
    } catch (reason) { setError((reason as Error).message) } finally { setWorking(false) }
  }
  async function detect() {
    setDetecting(true); setNotice(''); setError('')
    try {
      const result = await api<{ queued: number; detail?: string }>(`/cuts/${cutId}/character-detection`, { method: 'POST' })
      if (!result.queued) setNotice(result.detail || 'Nothing to detect in this cut.')
      void onStarted()
    } catch (reason) { setError((reason as Error).message) } finally { setDetecting(false) }
  }
  function toggleNoCharacters(value: boolean) {
    setNoChars(value); setError('')
    enqueueEdit(async () => { try { await api(`/cuts/${cutId}/no-characters`, { method: 'PUT', body: JSON.stringify({ no_characters: value }) }); void onStarted() } catch (reason) { setNoChars(current => current === value ? !value : current); throw reason } })
  }
  function coords(event: React.PointerEvent<SVGSVGElement>, frame: Frame): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect()
    return [Math.max(0, Math.min(frame.image_size[0], (event.clientX - rect.left) * frame.image_size[0] / rect.width)), Math.max(0, Math.min(frame.image_size[1], (event.clientY - rect.top) * frame.image_size[1] / rect.height))]
  }
  function svgPoint(event: React.PointerEvent<SVGElement>, frame: Frame): [number, number] {
    const rect = event.currentTarget.ownerSVGElement!.getBoundingClientRect()
    return [Math.max(0, Math.min(frame.image_size[0], (event.clientX - rect.left) * frame.image_size[0] / rect.width)), Math.max(0, Math.min(frame.image_size[1], (event.clientY - rect.top) * frame.image_size[1] / rect.height))]
  }
  function moveHandle(event: React.PointerEvent<SVGElement>, frame: Frame) {
    if (!resize) return
    const [x, y] = svgPoint(event, frame)
    if (resize.handle === 'move' && resize.from && resize.orig) {
      const [x1, y1, x2, y2] = resize.orig
      const dx = Math.max(-x1, Math.min(frame.image_size[0] - x2, x - resize.from[0]))
      const dy = Math.max(-y1, Math.min(frame.image_size[1] - y2, y - resize.from[1]))
      setResize({ ...resize, box: [x1 + dx, y1 + dy, x2 + dx, y2 + dy] })
      return
    }
    const box: Box = [...resize.box]
    const min = 8
    if (resize.handle.includes('w')) box[0] = Math.min(x, box[2] - min)
    if (resize.handle.includes('e')) box[2] = Math.max(x, box[0] + min)
    if (resize.handle.includes('n')) box[1] = Math.min(y, box[3] - min)
    if (resize.handle.includes('s')) box[3] = Math.max(y, box[1] + min)
    setResize({ ...resize, box })
  }
  function finishResize() {
    const value = resize
    setResize(null)
    if (!value) return
    const box = value.box.map(Math.round) as Box
    setRun(previous => previous && { ...previous, detections: previous.detections.map(item => item.id === value.id ? { ...item, box, gallery_status: null, gallery_note: null } : item) })
    enqueueEdit(async () => { await api(`/character-detections/${savedId(value.id)}/box`, { method: 'PUT', body: JSON.stringify({ box }) }) })
  }
  function finish(frame: Frame, end: [number, number]) {
    if (!run || !start) return
    const box: Box = [Math.round(Math.min(start[0], end[0])), Math.round(Math.min(start[1], end[1])), Math.round(Math.max(start[0], end[0])), Math.round(Math.max(start[1], end[1]))]
    setStart(null); setCurrent(null); setDrawing(null)
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) return
    const id = `pending-${++nextTemporaryBoxId}`
    setRun(previous => previous && { ...previous, detections: [...previous.detections, { id, position: frame.position, box, character_context_id: null, character_name: null, label_source: null, match_reason: null, is_reference: false }] })
    enqueueEdit(async () => {
      const result = await api<{ id: string }>(`/character-detection-runs/${run.id}/boxes`, { method: 'POST', body: JSON.stringify({ position: frame.position, box }) })
      createdIds.current.set(id, result.id)
    })
  }
  // The pose block belongs to the first-frame result; it is read again whenever a job of this cut changes.
  const poseKey = jobs.filter(item => item.cuts?.some(entry => entry.id === cutId)).map(item => `${item.id}:${item.status}`).join(',')
  const locked = working || priorSaving || !!active || !!run?.stale || noChars
  const frameBoundaryChanged = (position: 'first' | 'last') => !!run && (position === 'first' ? run.cut_start !== cutStart : run.cut_end !== cutEnd)
  const hasOutdatedFrame = !!run?.frames.some(frame => frameBoundaryChanged(frame.position))
  // "@" starts a box at the pointer on the hovered frame and a second "@" finishes it there (Escape cancels).
  const keyState = useRef({ run, start, drawing, locked, finish })
  keyState.current = { run, start, drawing, locked, finish }
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && keyState.current.start) { setStart(null); setCurrent(null); setDrawing(null); return }
      if (event.key !== '@' || event.repeat || event.ctrlKey || event.metaKey || event.altKey || event.isComposing) return
      const target = event.target as HTMLElement | null
      if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return
      if (document.querySelectorAll('[aria-modal="true"]').length > 1) return
      const { run: currentRun, start: from, drawing: where, locked: isLocked, finish: complete } = keyState.current
      const over = hover.current
      if (!currentRun || isLocked) return
      event.preventDefault()
      if (from && where) {
        const frame = currentRun.frames.find(item => item.position === where)
        if (frame) void complete(frame, over && over.position === where ? over.point : from)
        return
      }
      if (!over) return
      setDrawing(over.position); setStart(over.point); setCurrent(over.point)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  return <section aria-label="Characters in this cut" className="space-y-3 rounded-xl border hairline bg-[#171c25] p-3">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div className="min-w-0"><h3 className="flex items-center gap-2 text-sm font-semibold"><ScanFace size={15} className="text-[#e2709c]" aria-hidden /> Characters in this cut{(saving > 0 || priorSaving) && <StatusChip tone="amber">{priorSaving ? 'Saving earlier edits…' : `Saving ${saving} ${saving === 1 ? 'edit' : 'edits'}…`}</StatusChip>}{run && <StatusChip tone={run.detections.length ? 'blue' : 'grey'}>{run.detections.length} {run.detections.length === 1 ? 'box' : 'boxes'}</StatusChip>}
        <label className={`inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-semibold leading-4 transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-[#c3aeff] ${noChars ? 'border-[#6b5a2f] bg-[#2a2314] text-[#e6c58b]' : 'border-[#353c48] bg-[#1b202a] text-[#929aa9] hover:border-[#6b5a2f] hover:text-[#e6c58b]'} ${working ? 'opacity-60' : ''}`} title={noChars ? 'Marked as having no characters. Batch detection and cut analysis ignore this shot’s boxes. Untick to use them again.' : 'Tick if nobody appears in this shot. Batch detection skips it and cut analysis ignores any boxes.'}>
          <input type="checkbox" className="m-0 h-3.5 w-3.5 cursor-pointer accent-[#e6c58b]" checked={noChars} disabled={working || priorSaving} onChange={event => void toggleNoCharacters(event.target.checked)} />
          {noChars && <UserX size={12} aria-hidden />}{noChars ? 'No characters in this shot · skipped' : 'No characters in this shot'}
        </label></h3><p className="muted mt-1 text-[11px]">Finds people in the first frame and names them from your labelled boxes. Your labels and drawn boxes are kept.</p></div>
      <div className="flex items-center gap-2"><ActionButton icon={<ScanFace size={14} />} pending={detecting || !!active} disabled={working || !!saving || priorSaving || episodeBusy} onClick={() => void detect()} title="Find and name the people in this cut's first frame">{active ? 'Detecting' : 'Detect characters'}</ActionButton></div>
    </div>
    {noChars && <p role="status" className="flex items-start gap-2 rounded-lg border border-[#6b5a2f] bg-[#2a2314] px-3 py-2 text-xs text-[#e6c58b]"><UserX size={14} className="mt-0.5 shrink-0" aria-hidden />Marked as no characters. These frames are dimmed, their boxes are disabled, and they are ignored by batch detection and cut analysis. Untick the chip to edit them again.</p>}
    {active && <p role="status" className="flex items-center gap-2 rounded-lg border border-[#4a4363] bg-[#221c33] px-3 py-2 text-xs text-[#c3aeff]"><Loader2 size={14} className="animate-spin" aria-hidden />{job?.stage || 'Detection queued'}… <span className="ml-auto"><Elapsed since={job?.started_at} /></span></p>}
    {job?.status === 'failed' && <p role="alert" className="flex items-start gap-2 rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-xs text-[#ffb6bc]"><XCircle size={14} className="mt-0.5 shrink-0" aria-hidden />{job.error || 'Character detection failed.'}</p>}
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
    {notice && !active && <p role="status" className="muted text-xs">{notice}</p>}
    {run?.stale && <p className="rounded-lg border border-[#6b5a2f] bg-[#2a2314] px-3 py-2 text-xs text-[#e6c58b]">Cut timing changed. {hasOutdatedFrame ? 'Remove an outdated frame below or detect again' : 'Detect again'} before using these labels in analysis.</p>}
    {!loaded && !run && <div className="grid gap-3" aria-busy="true" aria-label="Loading character detections"><Skeleton className="aspect-video" /></div>}
    {loaded && !run && !active && <p className="muted rounded-lg border border-dashed hairline px-3 py-4 text-center text-xs">No characters detected yet. Press Detect characters.</p>}
    {run && <div className="grid gap-3">{run.frames.filter(frame => frame.position === 'first').map(frame => {
      const people = run.detections.filter(item => item.position === frame.position)
      const img = `/api/character-detection-runs/${run.id}/frame/${frame.position}`
      const boundaryChanged = frameBoundaryChanged(frame.position)
      const hasReference = people.some(person => person.is_reference)
      // Keep remove targets apart even when boxes have the same centre.
      const removeTargets: { x: number; y: number; radius: number }[] = []
      for (const person of people) {
        const box = resize?.id === person.id ? resize.box : person.box
        const radius = Math.min(frame.image_size[0] / 45, Math.max(frame.image_size[0] / 160, Math.min(box[2] - box[0], box[3] - box[1]) * .3))
        const centerX = (box[0] + box[2]) / 2, centerY = (box[1] + box[3]) / 2
        let x = centerX, y = centerY
        outer: for (let ring = 0; ring < 20; ring++) {
          const count = ring ? ring * 8 : 1
          for (let i = 0; i < count; i++) {
            const angle = i * Math.PI * 2 / count
            const candidateX = Math.max(radius, Math.min(frame.image_size[0] - radius, centerX + Math.cos(angle) * ring * radius * 2.2))
            const candidateY = Math.max(radius, Math.min(frame.image_size[1] - radius, centerY + Math.sin(angle) * ring * radius * 2.2))
            if (removeTargets.every(other => Math.hypot(other.x - candidateX, other.y - candidateY) >= other.radius + radius + 2)) { x = candidateX; y = candidateY; break outer }
          }
        }
        removeTargets.push({ x, y, radius })
      }
      return <div key={frame.position} className="space-y-2"><div className="flex flex-wrap items-center justify-between gap-2"><strong className="text-xs text-[#d5dbe5]">{frame.position === 'first' ? 'First' : 'Last'} frame · {frame.time.toFixed(2)}s</strong><div className="flex items-center gap-2">{run.stale && boundaryChanged && <ActionButton compact variant="danger" icon={<Trash2 size={12} />} disabled={working || !!saving || episodeBusy || !!active || hasReference} onClick={() => void removeStaleFrame(frame.position)} title={hasReference ? 'Remove the saved project reference first' : `Remove this outdated ${frame.position} frame and its character boxes from the stale run`}>Remove frame</ActionButton>}<button type="button" className="inline-flex items-center gap-1 text-xs text-[#c3aeff] hover:underline disabled:opacity-40" disabled={locked} onClick={() => { setDrawing(drawing === frame.position ? null : frame.position); setStart(null); setCurrent(null) }}>{drawing === frame.position ? <XCircle size={12} aria-hidden /> : <PencilLine size={12} aria-hidden />}{drawing === frame.position ? 'Cancel drawing' : 'Draw box'}<kbd className="ml-1 rounded border border-[#3b4352] px-1 font-mono text-[10px] leading-4 text-[#929aa9]" title="Hover a frame and press @ to start a box, press @ again to finish it">@</kbd></button></div></div>
        <div onPointerMove={event => { const rect = event.currentTarget.getBoundingClientRect(); hover.current = { position: frame.position, point: [Math.max(0, Math.min(frame.image_size[0], (event.clientX - rect.left) * frame.image_size[0] / rect.width)), Math.max(0, Math.min(frame.image_size[1], (event.clientY - rect.top) * frame.image_size[1] / rect.height))] } }} onPointerLeave={() => { hover.current = null }} className={`relative overflow-hidden bg-black ${noChars ? 'pointer-events-none select-none' : ''}`} aria-disabled={noChars || undefined} style={{ aspectRatio: `${frame.image_size[0]} / ${frame.image_size[1]}` }}><img src={img} alt={`${frame.position} detected frame`} loading="lazy" decoding="async" className={`absolute inset-0 h-full w-full transition-[filter,opacity] ${noChars ? 'opacity-35 grayscale' : ''}`} /><svg viewBox={`0 0 ${frame.image_size[0]} ${frame.image_size[1]}`} className={`absolute inset-0 h-full w-full ${noChars ? 'opacity-40 grayscale' : ''} ${drawing === frame.position ? 'touch-none cursor-crosshair' : 'pointer-events-none'}`} aria-label={`${frame.position} frame character boxes`} onPointerDown={event => { if (drawing !== frame.position) return; event.currentTarget.setPointerCapture(event.pointerId); const p = coords(event, frame); setStart(p); setCurrent(p) }} onPointerMove={event => { if (drawing === frame.position && start) setCurrent(coords(event, frame)) }} onPointerUp={event => { if (drawing === frame.position && start) void finish(frame, coords(event, frame)) }}>
          {people.map((person, index) => {
            const color = BOX_COLORS[index % BOX_COLORS.length]
            const box = resize?.id === person.id ? resize.box : person.box
            const editable = !noChars && drawing !== frame.position && !working && !active && !run.stale && !person.is_reference
            const size = Math.max(10, frame.image_size[0] / 80)
            // Same rules as the "Remove box" button below the frame: not while drawing, busy, or for a saved reference.
            return <g key={person.id}><rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill={color} fillOpacity=".12" stroke={color} strokeWidth="3" style={editable ? { pointerEvents: 'all', cursor: 'move', touchAction: 'none' } : undefined} onPointerDown={editable ? event => { event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); setResize({ id: person.id, handle: 'move', box: [...person.box] as Box, from: svgPoint(event, frame), orig: [...person.box] as Box }) } : undefined} onPointerMove={editable ? event => moveHandle(event, frame) : undefined} onPointerUp={editable ? () => void finishResize() : undefined} onPointerCancel={editable ? () => setResize(null) : undefined} /><text x={box[0] + 4} y={Math.max(20, box[1] - 4)} fill={color} stroke="black" strokeWidth="3" paintOrder="stroke" fontSize="20" fontWeight="700">{index + 1}. {person.character_name || 'Unknown'}</text>
              {editable && HANDLES.map(([handle, fx, fy]) => <rect key={handle} x={box[0] + (box[2] - box[0]) * fx - size / 2} y={box[1] + (box[3] - box[1]) * fy - size / 2} width={size} height={size} fill={color} stroke="black" strokeWidth="1.5" style={{ pointerEvents: 'all', cursor: CURSORS[handle], touchAction: 'none' }} onPointerDown={event => { event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); setResize({ id: person.id, handle, box: [...person.box] as Box }) }} onPointerMove={event => moveHandle(event, frame)} onPointerUp={() => void finishResize()} onPointerCancel={() => setResize(null)} />)}
              </g>
          })}
          {people.map((person, index) => {
            const color = BOX_COLORS[index % BOX_COLORS.length]
            const removable = !noChars && drawing !== frame.position && !working && !active && !person.is_reference && resize?.id !== person.id
            const { radius, x: cx, y: cy } = removeTargets[index]
            const remove = () => void removeBox(person.id)
            return <g key={person.id} className="group">
              {removable && <g role="button" tabIndex={0} aria-label={`Remove box ${index + 1}`} style={{ pointerEvents: 'all', cursor: 'pointer', touchAction: 'none' }} onPointerDown={event => event.stopPropagation()} onClick={event => { event.stopPropagation(); remove() }} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); remove() } }}>
                <title>Remove this box</title>
                <circle cx={cx} cy={cy} r={radius} className="group-hover:fill-[#b4233b] group-focus:fill-[#b4233b]" fill="#111" fillOpacity=".8" stroke={color} strokeWidth="2" />
                <path d={`M${cx - radius * .42} ${cy - radius * .42} L${cx + radius * .42} ${cy + radius * .42} M${cx + radius * .42} ${cy - radius * .42} L${cx - radius * .42} ${cy + radius * .42}`} stroke="#fff" strokeWidth={Math.max(2, radius / 6)} strokeLinecap="round" fill="none" />
              </g>}</g>
          })}
          {drawing === frame.position && start && current && <rect x={Math.min(start[0], current[0])} y={Math.min(start[1], current[1])} width={Math.abs(start[0] - current[0])} height={Math.abs(start[1] - current[1])} fill="none" stroke="white" strokeWidth="3" />}</svg></div>
        {!people.length && <p className="muted text-xs">No people detected. Use Draw box to add one.</p>}
        {people.map((person, index) => {
          const color = BOX_COLORS[index % BOX_COLORS.length]
          return <div key={person.id} className={`rounded-lg border hairline bg-[#1e2430] p-2 text-xs transition-opacity ${noChars ? 'opacity-45' : ''}`} aria-disabled={noChars || undefined} style={{ borderLeft: `4px solid ${noChars ? '#4a5262' : color}` }}>
            <div className="flex flex-wrap items-center gap-1.5"><span className="w-5 shrink-0 text-center font-semibold" style={{ color }}>{index + 1}</span>
              <select aria-label={`Character for box ${index + 1}`} className="field !w-auto min-w-32 flex-1 !px-2 !py-1.5 !text-xs" value={person.character_context_id || ''} disabled={locked || person.is_reference} onChange={event => { if (event.target.value === NEW_CHARACTER) setAdding(person.id); else void label(person.id, event.target.value || null) }}><option value="">Unknown</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}<option value={NEW_CHARACTER}>+ Add new character…</option></select>
              {person.label_source !== 'manual' && !person.is_reference && <ActionButton compact icon={<UserPlus size={13} />} disabled={locked} onClick={() => setAdding(person.id)} title="Create a new project character from this box and describe its features" aria-label={`Add new character from box ${index + 1}`} />}
              {person.is_reference ? <StatusChip tone="blue" icon={<Bookmark size={11} />}>Reference</StatusChip> : person.label_source === 'auto' ? <StatusChip tone="amber" title={person.match_reason || 'Automatic match, not yet confirmed by you'}>Auto · unverified</StatusChip> : person.label_source === 'manual' ? <StatusChip tone="green" icon={<CheckCircle2 size={11} />}>Confirmed</StatusChip> : <StatusChip tone="grey">No name</StatusChip>}
              {person.label_source === 'auto' && <ActionButton compact icon={<CheckCircle2 size={13} />} disabled={locked} onClick={() => void label(person.id, person.character_context_id)} title="Mark this label as yours; it then helps recognize this character elsewhere">Confirm</ActionButton>}
              {person.character_context_id && <ActionButton compact icon={person.is_reference ? <BookmarkMinus size={13} /> : <Bookmark size={13} />} disabled={locked || !!saving} onClick={() => void action(() => api(`/character-detections/${savedId(person.id)}/reference`, { method: person.is_reference ? 'DELETE' : 'POST' }), true)} title={person.is_reference ? 'Stop using this box as a saved reference' : 'Keep this box as a reference image for the whole project'} aria-label={person.is_reference ? 'Remove reference' : 'Save as reference'} />}
              <ActionButton compact variant="danger" icon={<Trash2 size={13} />} disabled={locked || person.is_reference} onClick={() => void removeBox(person.id)} title={person.is_reference ? 'Remove the reference first' : 'Remove this box'} aria-label={`Remove box ${index + 1}`} />
            </div>
            {person.label_source === 'auto' && person.match_reason && <p className="mt-1 text-[11px] text-[#e6c58b]">{person.match_reason}</p>}
            <Suggestions person={person} disabled={locked} onPick={id => void label(person.id, id)} />
            {person.label_source === 'manual' && <MatchingUse person={person} disabled={locked} onChange={use => setMatchingUse(person.id, use)} />}
            {frame.position === 'first' && person.character_context_id && <PoseDetailCard cutId={cutId} person={person} frameBoxes={people} refreshKey={poseKey} disabled={locked} />}
          </div>
        })}
      </div>
    })}</div>}
    {!!references.length && <details className="group rounded-lg border hairline"><summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-xs font-semibold text-[#c3aeff]"><Bookmark size={13} aria-hidden />Project character references <StatusChip tone="blue">{references.length}</StatusChip><ChevronDown size={14} className="ml-auto transition-transform group-open:rotate-180" aria-hidden /></summary><div className="grid grid-cols-2 gap-2 p-2 sm:grid-cols-3 xl:grid-cols-4">{references.map(ref => <div key={ref.id} className="rounded-lg border hairline bg-[#1e2430] p-2"><svg viewBox={`${ref.box[0]} ${ref.box[1]} ${ref.box[2]-ref.box[0]} ${ref.box[3]-ref.box[1]}`} className="h-24 w-full rounded bg-black"><image href={`/api/character-detection-runs/${ref.run_id}/frame/${ref.position}`} width={ref.image_size[0]} height={ref.image_size[1]} /></svg><div className="mt-1 flex items-center justify-between gap-1 text-xs"><span className="truncate" title={ref.character_name}>{ref.character_name}</span><ActionButton compact variant="danger" icon={<Trash2 size={12} />} disabled={working || !!saving} onClick={() => void action(() => api(`/character-references/${ref.id}`, { method: 'DELETE' }), true)} title="Remove this reference" aria-label={`Remove reference for ${ref.character_name}`} /></div></div>)}</div></details>}
    {adding && run && (() => {
      const index = run.detections.filter(item => item.position === 'first').findIndex(item => item.id === adding)
      const person = run.detections.find(item => item.id === adding)
      const frame = person && run.frames.find(item => item.position === person.position)
      if (!person || !frame) return null
      return <NewCharacterDialog projectId={projectId} box={person.box} frameUrl={`/api/character-detection-runs/${run.id}/frame/${frame.position}`} imageSize={frame.image_size} boxNumber={index + 1} onCreate={input => createCharacter(person.id, input)} onClose={() => setAdding(null)} />
    })()}
    <details className="group text-[11px] text-[#929aa9]"><summary className="inline-flex cursor-pointer items-center gap-1.5 hover:text-[#c3aeff]"><Info size={13} aria-hidden />How matching works</summary>
      <ul className="mt-2 list-disc space-y-1 pl-5 leading-5">
        <li>Pick a character for each box, or choose Add new character to create one from the box. Qwen then writes the character's look and outfit from that box into its Details.</li>
        <li>Every box you label or confirm becomes a matching example for the whole project. Tiny or unidentifiable boxes (a hand, a strand of hair) are left out of matching but keep their label for analysis.</li>
        <li>Automatic matches stay unverified until you queue this cut. Queuing treats named boxes as verified, so correct or clear them first.</li>
        <li>Single-cut frame analysis receives up to four reference images; keyframe analysis receives the names and boxes.</li>
      </ul></details>
  </section>
})

function Suggestions({ person, disabled, onPick }: { person: Detection; disabled: boolean; onPick: (characterId: string) => void }) {
  const options = (person.candidates || []).filter(item => person.label_source !== 'auto' || item.character_id !== person.character_context_id)
  if (person.label_source === 'manual' || !options.length) return null
  return <div className="mt-1 flex flex-wrap items-center gap-1"><span className="muted text-[11px]">{person.label_source === 'auto' ? 'Or:' : 'Suggestions:'}</span>{options.map(item => <button key={item.character_id} type="button" disabled={disabled} onClick={() => onPick(item.character_id)} className="rounded border border-[#4a4363] px-2 py-0.5 text-[11px] hover:border-[#c3aeff] hover:text-[#c3aeff]" title={`Label as ${item.character_name} (similarity ${item.score.toFixed(2)})`}>{item.character_name} <span className="muted">{item.score.toFixed(2)}</span></button>)}</div>
}

function MatchingUse({ person, disabled, onChange }: { person: Detection; disabled: boolean; onChange: (use: boolean | null) => void }) {
  const status = person.gallery_status
  const forced = person.recognition_use === 1
  const excluded = person.recognition_use === 0
  const text = excluded ? 'Not used as a matching example (your choice).'
    : forced ? 'Always used as a matching example (your choice).'
    : status === 'used' ? 'Used as a matching example.'
    : status === 'small' || status === 'atypical' ? `Not used as a matching example: ${person.gallery_note || status}.`
    : 'Will be checked as a matching example on the next recognition run.'
  const link = 'ml-1 text-[#c3aeff] hover:underline disabled:opacity-40'
  const tone = excluded || status === 'small' || status === 'atypical' ? 'text-[#e6c58b]' : forced || status === 'used' ? 'text-[#7bdcb2]' : 'text-[#929aa9]'
  return <p className={`mt-1 text-[11px] ${tone}`}>{text}
    {(forced || excluded) && <button type="button" disabled={disabled} className={link} onClick={() => onChange(null)}>Decide automatically</button>}
    {!forced && !excluded && (status === 'small' || status === 'atypical') && <button type="button" disabled={disabled} className={link} onClick={() => onChange(true)}>Use anyway</button>}
    {!forced && !excluded && status !== 'small' && status !== 'atypical' && <button type="button" disabled={disabled} className={link} onClick={() => onChange(false)}>Don’t use for matching</button>}
  </p>
}
