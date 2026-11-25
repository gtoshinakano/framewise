import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { CheckCircle2, ChevronDown, ChevronRight, ChevronUp, GripVertical, Hourglass, LoaderCircle, X } from 'lucide-react'
import { api } from './api'
import type { DialogTab, RunSection } from './cutRoute'
import { useEpisodeEvents } from './episodeEvents'
import { usePref } from './layoutPrefs'
import { CutChip } from './RunMultipleDialog'
import { clampFloat, groupHeld, plural, type FloatPlace, type Preview } from './runMultipleView'

const WIDTH = 300
const NUDGE = 16

/** The Evaluate preview's held cuts, read again when the person settles something (and, slower, when jobs move on). */
function useEvaluatePreview(episodeId: string) {
  const [preview, setPreview] = useState<Preview | null>(null)
  const [error, setError] = useState('')
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const load = useCallback(() => {
    api<Preview>(`/episodes/${episodeId}/batch-runs/preview?stage=evaluate`)
      .then(value => { setPreview(value); setError('') })
      .catch(err => setError((err as Error).message))
  }, [episodeId])
  useEffect(() => { setPreview(null); load() }, [load])
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  useEpisodeEvents(change => {
    if (!(change.data || change.stale || change.jobs) || timer.current) return
    timer.current = setTimeout(() => { timer.current = null; load() }, change.data || change.stale ? 400 : 5000)
  }, episodeId)
  return { preview, error }
}

/**
 * Run Multiple's "Waiting for you" (the Evaluate stage's held cuts) kept on screen while the person works through it: it
 * floats above the editor and the Analysis dialog, can be dragged by its header, collapsed to that header, and remembers
 * where it was. A cut number opens the cut where its hold is settled; the list reads itself again as holds are settled.
 */
export function WaitingForYou({ episodeId, currentCutId, onOpenCut, onClose }: {
  episodeId: string
  currentCutId: string | null
  onOpenCut: (cutId: string, tab: DialogTab, section?: RunSection) => void
  onClose: () => void
}) {
  const { preview, error } = useEvaluatePreview(episodeId)
  const [collapsed, setCollapsed] = usePref('waitingForYou.collapsed', false)
  const [stored, setStored] = usePref<FloatPlace | null>('waitingForYou.place', null)
  const [dragged, setDragged] = useState<FloatPlace | null>(null)
  const [folded, setFolded] = useState<ReadonlySet<string>>(() => new Set())
  const [viewport, setViewport] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }))
  const [size, setSize] = useState({ width: WIDTH, height: 40 })
  const panel = useRef<HTMLElement>(null)
  // The drag lives in a ref too: a quick drag can end before the render that saw its last move.
  const drag = useRef<{ pointerId: number; dx: number; dy: number; at: FloatPlace | null } | null>(null)

  useEffect(() => {
    const resize = () => setViewport({ width: window.innerWidth, height: window.innerHeight })
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])
  useLayoutEffect(() => {
    const element = panel.current
    if (!element) return
    const observer = new ResizeObserver(() => setSize({ width: element.offsetWidth, height: element.offsetHeight }))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  // The stored place is where the person dropped it; it is only kept inside the window when drawn, so expanding near the
  // bottom lifts the panel and collapsing puts it back.
  const place = clampFloat(dragged ?? stored ?? { x: viewport.width - WIDTH - 64, y: 64 }, size, viewport)

  function startDrag(event: PointerEvent<HTMLElement>) {
    if (event.button !== 0 || (event.target as HTMLElement).closest('button:not([data-grip])')) return
    event.preventDefault()
    drag.current = { pointerId: event.pointerId, dx: event.clientX - place.x, dy: event.clientY - place.y, at: null }
    try { event.currentTarget.setPointerCapture(event.pointerId) } catch { /* a pointer that is already gone: moves still reach the header */ }
  }
  function moveDrag(event: PointerEvent<HTMLElement>) {
    const current = drag.current
    if (current?.pointerId !== event.pointerId) return
    current.at = { x: event.clientX - current.dx, y: event.clientY - current.dy }
    setDragged(current.at)
  }
  function endDrag() {
    const at = drag.current?.at
    drag.current = null
    if (at) setStored(clampFloat(at, size, viewport))
    setDragged(null)
  }
  function nudge(event: KeyboardEvent<HTMLButtonElement>) {
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key]
    if (!step) return
    event.preventDefault()
    const distance = event.shiftKey ? NUDGE * 4 : NUDGE
    setStored(clampFloat({ x: place.x + step[0] * distance, y: place.y + step[1] * distance }, size, viewport))
  }
  const toggleGroup = (key: string) => setFolded(current => {
    const next = new Set(current)
    if (!next.delete(key)) next.add(key)
    return next
  })

  const held = preview?.held ?? []
  const groups = groupHeld(held)
  return <aside ref={panel} aria-label="Cuts waiting for you" style={{ left: place.x, top: place.y, width: WIDTH }}
    className={`fixed z-[60] flex max-h-[min(70vh,560px)] flex-col overflow-hidden rounded-xl border border-[#4a4030] bg-[#1b202a]/95 text-xs shadow-2xl backdrop-blur ${dragged ? 'ring-1 ring-[#f5c678]/60' : ''}`}>
    <header onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={endDrag} onPointerCancel={endDrag} onDoubleClick={event => { if (!(event.target as HTMLElement).closest('button')) setCollapsed(!collapsed) }}
      className={`flex shrink-0 touch-none select-none items-center gap-1.5 px-1.5 py-1.5 ${dragged ? 'cursor-grabbing' : 'cursor-grab'} ${collapsed ? '' : 'border-b hairline'}`}>
      <button type="button" data-grip onKeyDown={nudge} aria-label="Move the list (drag, or arrow keys)" title="Drag to move (arrow keys move it too)"
        className="cursor-[inherit] rounded p-0.5 text-[#8f97a6] hover:text-white"><GripVertical size={14} aria-hidden /></button>
      <Hourglass size={13} className="shrink-0 text-[#f5c678]" aria-hidden />
      <h2 className="min-w-0 flex-1 truncate font-semibold">Waiting for you</h2>
      {!preview && !error ? <LoaderCircle size={13} className="animate-spin text-[#8f97a6] motion-reduce:animate-none" aria-label="Loading" />
        : <span className={`rounded-full px-1.5 font-mono text-[11px] tabular-nums ${held.length ? 'bg-[#3a3020] text-[#f5c678]' : 'bg-[#1f3329] text-[#7bdcb2]'}`}
          title={plural(held.length, 'cut')}>{held.length}</span>}
      <button type="button" onClick={() => setCollapsed(!collapsed)} aria-expanded={!collapsed} aria-label={collapsed ? 'Expand the list' : 'Collapse the list'}
        title={collapsed ? 'Expand' : 'Collapse'} className="rounded p-1 text-[#aeb5c1] hover:bg-[#303746] hover:text-white">{collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}</button>
      <button type="button" onClick={onClose} aria-label="Close the list" title="Close (Run Multiple → Evaluate shows it again)"
        className="rounded p-1 text-[#aeb5c1] hover:bg-[#303746] hover:text-white"><X size={14} /></button>
    </header>
    {!collapsed && <div className="scrollbar min-h-0 flex-1 overflow-y-auto px-3 py-2">
      {error ? <p role="alert" className="text-[#ff9fa7]">{error}</p>
        : !preview ? <p className="muted">Loading…</p>
          : <>
            {!preview.available && preview.reason && <p className="muted mb-2">{preview.reason}</p>}
            {!held.length ? <p className="flex items-center gap-1.5 text-[#9ca5b3]"><CheckCircle2 size={13} className="text-[#7bdcb2]" aria-hidden /> No cut waits for you: Evaluate can run on every cut that needs it.</p>
              : <>
                <p className="muted mb-2">Evaluate skips these until you settle them. Click a cut to open it where the fix is.</p>
                <div className="space-y-1.5">{groups.map(group => {
                  const open = !folded.has(group.key)
                  return <section key={group.key} className="rounded-lg border hairline bg-[#171c25]">
                    <button type="button" onClick={() => toggleGroup(group.key)} aria-expanded={open}
                      className="flex w-full items-start gap-1 px-2 py-1.5 text-left font-semibold hover:text-white">
                      {open ? <ChevronDown size={13} className="mt-px shrink-0 text-[#8f97a6]" aria-hidden /> : <ChevronRight size={13} className="mt-px shrink-0 text-[#8f97a6]" aria-hidden />}
                      <span className="min-w-0 flex-1">{group.label}</span>
                      <span className="muted shrink-0 font-normal tabular-nums">{group.cuts.length}</span>
                    </button>
                    {open && <div className="flex flex-wrap gap-1 px-2 pb-2">{group.cuts.map(item => <CutChip key={item.cut_id} number={item.cut} title={item.reasons.join('; ')}
                      current={item.cut_id === currentCutId} onClick={() => onOpenCut(item.cut_id, group.tab, group.section)} />)}</div>}
                  </section>
                })}</div>
              </>}
          </>}
    </div>}
  </aside>
}
