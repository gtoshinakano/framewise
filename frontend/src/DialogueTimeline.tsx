import { memo, useMemo, useRef, useState, useEffect, type PointerEvent as ReactPointerEvent, type RefObject } from 'react'
import { GripHorizontal, ZoomIn, ZoomOut } from 'lucide-react'
import { clamp, lineColour, lineTracks, limits, placeLabels, snap, snapTargets, type Marker } from './dialogueTimeline'
import { usePref } from './layoutPrefs'
import { usePlayer, usePlayerValue } from './SourcePlayer'
import type { CompiledLine, SequenceCut } from './sequenceApi'
import { clock } from './sequenceView'

const TOP = 12                 // px above second 0
const ZOOMS = [8, 12, 18, 24, 36, 48, 72]
const AXIS = 46                // x of the time axis: hard cuts to its left, lines to its right
const BARS = 50                // first line track
const LABELS = 72              // line labels
const SHOTS = 170              // the shot lane
const LABEL_HEIGHT = 13
const SNAP_PX = 6

type Props = {
  duration: number; origin: number; cuts: SequenceCut[]; lines: CompiledLine[]; markers: Marker[]
  selectedLine: string | null; selectedMarker: string | null; disabled: boolean
  onSelectLine: (line: CompiledLine) => void; onSelectMarker: (marker: Marker) => void
  onPlace: (marker: Marker, t: number) => void; onRelease: (marker: Marker) => void
}

/**
 * The sequence as a thin vertical timeline: the hard cuts on the left, every line in its own colour in the middle
 * (the line rows use the same colours), and one shot boundary per hard cut on the right. A boundary you drag is
 * placed by you and is final for the director; the others stay where the plan (or the hard cut) puts them.
 */
export default function DialogueTimeline({ duration, origin, cuts, lines, markers, selectedLine, selectedMarker, disabled, onSelectLine, onSelectMarker, onPlace, onRelease }: Props) {
  const player = usePlayer()
  const [zoom, setZoom] = usePref('dialogue-timeline-zoom', 3)
  const pps = ZOOMS[Math.max(0, Math.min(ZOOMS.length - 1, zoom))]
  const scroller = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)
  const drag = useRef<{ id: string; t: number; moved: boolean } | null>(null)
  const [dragging, setDragging] = useState<{ id: string; t: number } | null>(null)
  const y = (t: number) => TOP + t * pps
  const timeAt = (clientY: number) => (clientY - (content.current?.getBoundingClientRect().top ?? 0) - TOP) / pps
  const targets = useMemo(() => snapTargets(lines, cuts), [lines, cuts])
  const tracks = useMemo(() => lineTracks(lines), [lines])
  const labels = useMemo(() => placeLabels(lines.map(line => line.t0 * pps - LABEL_HEIGHT / 2), LABEL_HEIGHT), [lines, pps])

  function moveTo(marker: Marker, clientY: number, free: boolean) {
    let t = timeAt(clientY)
    if (!free) t = snap(t, targets, SNAP_PX / pps)
    const element = scroller.current
    if (element) {
      // Dragging past the edge scrolls the timeline.
      const box = element.getBoundingClientRect()
      if (clientY < box.top + 24) element.scrollTop -= 12
      else if (clientY > box.bottom - 24) element.scrollTop += 12
    }
    return clamp(t, limits(marker.index, markers, cuts))
  }
  function down(event: ReactPointerEvent<HTMLButtonElement>, marker: Marker) {
    if (event.button !== 0) return
    event.stopPropagation()
    onSelectMarker(marker)
    if (disabled) return
    event.preventDefault()
    event.currentTarget.focus()
    event.currentTarget.setPointerCapture(event.pointerId)
    drag.current = { id: marker.id, t: marker.t, moved: false }
  }
  function move(event: ReactPointerEvent<HTMLButtonElement>, marker: Marker) {
    if (drag.current?.id !== marker.id) return
    const t = moveTo(marker, event.clientY, event.altKey)
    drag.current = { id: marker.id, t, moved: drag.current.moved || Math.abs(t - marker.t) > 0.0005 }
    setDragging({ id: marker.id, t })
  }
  function up(marker: Marker) {
    const done = drag.current
    drag.current = null
    setDragging(null)
    if (done?.id === marker.id && done.moved && Math.abs(done.t - marker.t) > 0.0005) onPlace(marker, done.t)
  }
  function keys(event: React.KeyboardEvent<HTMLButtonElement>, marker: Marker) {
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault(); event.stopPropagation()
      if (disabled) return
      const step = (event.shiftKey ? 0.5 : 0.04) * (event.key === 'ArrowUp' ? -1 : 1)
      onPlace(marker, clamp(marker.t + step, limits(marker.index, markers, cuts)))
    } else if ((event.key === 'Delete' || event.key === 'Backspace') && marker.source === 'you') {
      event.preventDefault(); event.stopPropagation()
      if (!disabled) onRelease(marker)
    } else if (event.key === ' ' || event.key === 'Enter') {
      event.stopPropagation()  // activates the handle, not the selected line
    }
  }

  const height = TOP + duration * pps + 28
  return <section aria-label="Timeline of cuts, lines and shots" className="flex min-h-0 flex-col rounded-xl border hairline bg-[#161b24] max-lg:max-h-[28rem]">
    <header className="flex shrink-0 items-center gap-1 border-b hairline px-3 py-2">
      <h3 className="mr-auto text-xs font-semibold">Timeline</h3>
      <button type="button" className="rounded p-1 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white disabled:opacity-40" disabled={zoom <= 0} onClick={() => setZoom(Math.max(0, zoom - 1))} aria-label="Zoom out" title="Zoom out"><ZoomOut size={13} /></button>
      <button type="button" className="rounded p-1 text-[#aeb5c1] hover:bg-[#272e3a] hover:text-white disabled:opacity-40" disabled={zoom >= ZOOMS.length - 1} onClick={() => setZoom(Math.min(ZOOMS.length - 1, zoom + 1))} aria-label="Zoom in" title="Zoom in"><ZoomIn size={13} /></button>
    </header>
    <div className="grid shrink-0 grid-cols-[46px_1fr_70px] border-b hairline px-1 py-1 text-[9px] font-semibold uppercase tracking-[.08em] text-[#8b93a3]">
      <span className="pl-1">Cuts</span><span className="pl-6">Lines</span><span className="pl-1">Shots</span>
    </div>
    <div ref={scroller} className="scrollbar relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
      <div ref={content} className="relative w-full cursor-crosshair select-none" style={{ height }} onClick={event => player.seek(origin + Math.max(0, Math.min(duration, timeAt(event.clientY))))}>
        {cuts.map((cut, index) => <div key={cut.id} aria-hidden className={`absolute inset-x-0 ${index % 2 ? 'bg-[#1b212b]' : ''}`} style={{ top: y(cut.t0), height: (cut.t1 - cut.t0) * pps }}>
          {index > 0 && <div className="absolute left-0 h-px bg-[#8b93a3]" style={{ width: AXIS + 4, top: 0 }} />}
          <span className="absolute left-1 top-0.5 whitespace-nowrap font-mono text-[10px] text-[#c9cfda]">cut {cut.number}</span>
          {index > 0 && <span className="absolute -top-[7px] text-[11px] leading-none text-[#e8edf5]" style={{ left: AXIS - 4 }}>ϕ</span>}
        </div>)}
        <div aria-hidden className="absolute bottom-0 top-0 w-px bg-[#3a4150]" style={{ left: AXIS }} />
        <div aria-hidden className="absolute bottom-0 top-0 w-px bg-[#3a4150]" style={{ left: SHOTS }} />

        {lines.map((line, index) => {
          const colour = lineColour(index)
          const selected = selectedLine === line.id
          const top = TOP + labels[index]
          const anchor = y(line.t0)
          return <div key={line.id}>
            {/* The bar repeats what the label button next to it does, so it stays out of the tab order. */}
            <div aria-hidden className="absolute cursor-pointer rounded-sm" style={{ left: BARS + tracks[index] * 6, width: 4, top: anchor, height: Math.max(2, (line.t1 - line.t0) * pps), background: colour, opacity: selected ? 1 : 0.85 }}
              onClick={event => { event.stopPropagation(); onSelectLine(line) }} />
            {top - anchor > 3 && <div aria-hidden className="absolute w-px" style={{ left: LABELS - 3, top: anchor, height: top - anchor + LABEL_HEIGHT / 2, background: colour, opacity: 0.6 }} />}
            <button type="button" className={`absolute flex items-center gap-1 overflow-hidden rounded px-1 text-left text-[10px] leading-[13px] ${selected ? 'bg-[#2a2319] ring-1 ring-[#7d6542]' : 'hover:bg-[#222833]'}`}
              style={{ left: LABELS, width: SHOTS - LABELS - 4, top, height: LABEL_HEIGHT, color: colour }} title={`${line.id} ${clock(line.t0)}–${clock(line.t1)} · ${line.character || line.speaker_label || 'Unknown speaker'}: ${line.text}`}
              onClick={event => { event.stopPropagation(); onSelectLine(line) }}>
              <span className="shrink-0 font-semibold">{(line.character || line.speaker_label || '?').split(' ')[0]}:</span><span className="truncate text-[#c9cfda]">{line.text}</span>
            </button>
          </div>
        })}

        <span className="absolute rounded bg-[#262b35] px-1.5 text-[10px] font-semibold leading-4 text-[#aeb5c1]" style={{ left: SHOTS + 6, top: TOP - 8 }} title="The first shot starts with the sequence">shot 1</span>
        {markers.map(marker => {
          const t = dragging?.id === marker.id ? dragging.t : marker.t
          const mine = marker.source === 'you' || dragging?.id === marker.id
          const selected = selectedMarker === marker.id
          const offset = Math.abs(t - marker.hard) > 0.002
          const tone = mine ? '#e0a96d' : '#8b93a3'
          return <div key={marker.id}>
            <div aria-hidden className={`pointer-events-none absolute h-0 border-t ${mine ? 'border-solid' : 'border-dashed'}`} style={{ left: AXIS, right: 0, top: y(t), borderColor: tone, opacity: selected ? 0.9 : 0.45 }} />
            {offset && <div aria-hidden className="pointer-events-none absolute w-px" style={{ left: SHOTS + 2, top: y(Math.min(t, marker.hard)), height: Math.abs(t - marker.hard) * pps, background: tone, opacity: 0.6 }} />}
            {offset && <div aria-hidden className="pointer-events-none absolute h-px w-2" style={{ left: SHOTS - 3, top: y(marker.hard), background: '#8b93a3' }} />}
            <button type="button" aria-label={`Shot ${marker.shot} starts at ${clock(t)}${marker.source === 'you' ? ', placed by you' : ''}. Drag or use the arrow keys to move it.`}
              title={`${marker.source === 'you' ? 'Placed by you' : marker.source === 'plan' ? `Where the plan has it${marker.by === 'director' ? ' (director)' : marker.by === 'app' ? ' (app)' : ''}` : marker.speech ? 'On the hard cut until the director decides' : 'On the hard cut'} · ${clock(t)}\nDrag to place it (Alt: no snapping), ↑/↓ nudge, Delete hands it back to the director`}
              className={`absolute z-10 flex touch-none items-center gap-1 rounded-full border px-1.5 text-[10px] font-semibold leading-4 ${disabled ? 'cursor-default' : 'cursor-ns-resize'} ${mine ? 'border-[#e0a96d] bg-[#e0a96d] text-[#1b1206]' : 'border-[#5b6372] bg-[#161b24] text-[#c9cfda]'} ${selected ? 'ring-2 ring-white/70' : ''}`}
              style={{ left: SHOTS + 6, top: y(t) - 9, height: 18 }}
              onPointerDown={event => down(event, marker)} onPointerMove={event => move(event, marker)} onPointerUp={() => up(marker)} onPointerCancel={() => { drag.current = null; setDragging(null) }}
              onClick={event => event.stopPropagation()} onKeyDown={event => keys(event, marker)}>
              <GripHorizontal size={11} aria-hidden /> shot {marker.shot}
            </button>
            {dragging?.id === marker.id && <span className="pointer-events-none absolute z-20 rounded bg-black/85 px-1.5 font-mono text-[10px] text-white" style={{ left: LABELS, top: y(t) - 20 }}>{clock(t)} · {(t - marker.hard >= 0 ? '+' : '−') + Math.abs(t - marker.hard).toFixed(2)} s</span>}
          </div>
        })}
        <Playhead origin={origin} pps={pps} duration={duration} scroller={scroller} />
      </div>
    </div>
    <footer className="shrink-0 border-t hairline px-3 py-1.5 text-[10px] leading-4 text-[#8b93a3]">
      <span className="mr-2 inline-flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-full bg-[#e0a96d]" /> placed by you</span>
      <span className="inline-flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-full border border-[#5b6372]" /> director</span>
      <p>Drag a shot to place it; Alt turns off snapping.</p>
    </footer>
  </section>
}

/** The playhead line; only it re-renders while the video plays, and it keeps itself in view. */
const Playhead = memo(function Playhead({ origin, pps, duration, scroller }: { origin: number; pps: number; duration: number; scroller: RefObject<HTMLDivElement | null> }) {
  const px = usePlayerValue(time => Math.round(Math.min(duration, Math.max(0, time - origin)) * pps))
  const playing = usePlayerValue((_, paused) => !paused)
  useEffect(() => {
    const element = scroller.current
    if (!playing || !element) return
    const top = TOP + px
    if (top < element.scrollTop + 16 || top > element.scrollTop + element.clientHeight - 48) element.scrollTop = Math.max(0, top - element.clientHeight / 3)
  }, [px, playing, scroller])
  return <div aria-hidden className="pointer-events-none absolute inset-x-0 z-20" style={{ top: TOP + px }}>
    <div className="h-px bg-[#ff7a7a]" />
    <span className="absolute -top-2 rounded bg-[#ff7a7a] px-1 font-mono text-[9px] leading-4 text-[#1b0b0b]" style={{ left: 2 }}>{clock(px / pps)}</span>
  </div>
})
