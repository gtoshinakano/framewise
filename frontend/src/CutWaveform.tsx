import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { clampEdge, percentIn, stamp, timeAt, type Span } from './lineEditView'

/** One line drawn over the waveform. */
export type Band = { id: string; span: Span; color: string; selected: boolean; label: string }
/** The two draggable edges of the line being timed, and where they may go. */
export type Handles = { span: Span; bounds: Span; onCommit: (edge: 'start' | 'end', value: number) => void | Promise<void> }

type Props = {
  /** The stretch of audio shown, and the cut inside it. */
  view: Span
  cut: Span
  /** Peaks from 0 to 1, evenly spread over the view. */
  bars: number[]
  bands: Band[]
  playhead: number
  handles: Handles | null
  /** Shown instead of the graph (no audio, still preparing). */
  message?: string
  onSeek: (time: number) => void
  onSelectBand: (id: string, additive: boolean) => void
}

const NUDGE = 0.01
const NUDGE_FAST = 0.1

/** The waveform of the cut and of the lines that cross its edges, with the cut's boundaries and the timing handles. */
export function CutWaveform({ view, cut, bars, bands, playhead, handles, message, onSeek, onSelectBand }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState<{ edge: 'start' | 'end'; value: number } | null>(null)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])
  const fraction = (event: { clientX: number }) => {
    const rect = box.current?.getBoundingClientRect()
    return rect && rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0
  }
  const span = handles && drag ? { ...handles.span, [drag.edge]: drag.value } as Span : handles?.span
  const scale = 1 / Math.max(0.08, ...bars)
  const startPercent = percentIn(view, cut.start), endPercent = percentIn(view, cut.end)

  async function finish(edge: 'start' | 'end', value: number) {
    window.clearTimeout(timer.current)
    try { if (handles && value !== handles.span[edge]) await handles.onCommit(edge, value) }
    finally { setDrag(null) }
  }
  function nudge(edge: 'start' | 'end', delta: number) {
    if (!handles) return
    const base = drag?.edge === edge ? drag.value : handles.span[edge]
    const value = clampEdge(edge, base + delta, handles.span, handles.bounds)
    setDrag({ edge, value })
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => void finish(edge, value), 500)
  }
  const handleProps = (edge: 'start' | 'end') => ({
    onPointerDown: (event: PointerEvent<HTMLDivElement>) => {
      if (!handles) return
      event.preventDefault(); event.stopPropagation()
      event.currentTarget.setPointerCapture(event.pointerId)
      setDrag({ edge, value: handles.span[edge] })
    },
    onPointerMove: (event: PointerEvent<HTMLDivElement>) => {
      if (!handles || drag?.edge !== edge) return
      setDrag({ edge, value: clampEdge(edge, timeAt(view, fraction(event)), handles.span, handles.bounds) })
    },
    onPointerUp: () => { if (drag?.edge === edge) void finish(edge, drag.value) },
    onPointerCancel: () => setDrag(null),
    onBlur: () => { if (drag?.edge === edge && timer.current !== undefined) void finish(edge, drag.value) },
    onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
      event.preventDefault()
      nudge(edge, (event.key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? NUDGE_FAST : NUDGE))
    },
  })

  if (message) return <div className="muted flex h-24 items-center justify-center rounded-md bg-[#11151d] text-xs">{message}</div>
  return <div ref={box} className="relative h-24 touch-none select-none overflow-hidden rounded-md bg-[#11151d]" role="group" aria-label="Waveform of this cut and the lines that cross it" onPointerDown={event => { if (event.button === 0) onSeek(timeAt(view, fraction(event))) }}>
    {bars.length ? <svg viewBox={`0 0 ${bars.length} 100`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden>{bars.map((peak, index) => {
      const height = Math.max(2, Math.min(94, peak * scale * 94))
      return <rect key={index} x={index + 0.1} y={(100 - height) / 2} width="0.8" height={height} rx="0.3" fill="#a58ad9" />
    })}</svg> : <p className="muted pointer-events-none absolute inset-0 flex items-center justify-center text-xs">Loading waveform…</p>}
    {/* Outside the cut the sound is dimmed: it belongs to the neighbouring cut, but a line may run into it. */}
    {startPercent > 0 && <div className="pointer-events-none absolute inset-y-0 left-0 bg-[#0b0e14]/60" style={{ width: `${startPercent}%` }} />}
    {endPercent < 100 && <div className="pointer-events-none absolute inset-y-0 right-0 bg-[#0b0e14]/60" style={{ width: `${100 - endPercent}%` }} />}
    {bands.map(band => {
      const left = percentIn(view, band.span.start), right = percentIn(view, band.span.end)
      return <div key={band.id} title={band.label} onPointerDown={event => { if (event.button !== 0) return; event.stopPropagation(); onSelectBand(band.id, event.shiftKey); onSeek(timeAt(view, fraction(event))) }}
        className={`absolute inset-y-0 cursor-pointer ${band.selected ? 'z-[2]' : 'z-[1]'}`}
        style={{ left: `${left}%`, width: `${Math.max(0.3, right - left)}%`, background: `${band.color}${band.selected ? '55' : '26'}`, borderBottom: `3px solid ${band.color}`, boxShadow: band.selected ? `inset 0 0 0 1px ${band.color}` : undefined }} />
    })}
    {[{ at: startPercent, text: 'Cut start' }, { at: endPercent, text: 'Cut end' }].map(mark => <div key={mark.text} className="pointer-events-none absolute inset-y-0 z-[3] border-l border-dashed border-[#f2c879]" style={{ left: `${mark.at}%` }}>
      <span className={`absolute top-0 whitespace-nowrap bg-[#11151d]/80 px-1 text-[9px] font-semibold uppercase tracking-wide text-[#f2c879] ${mark.at > 85 ? 'right-0' : 'left-0'}`}>{mark.text}</span></div>)}
    <span className="pointer-events-none absolute inset-y-0 z-[4] w-px bg-white" style={{ left: `${percentIn(view, playhead)}%` }} />
    {handles && span && (['start', 'end'] as const).map(edge => {
      const value = span[edge]
      const active = drag?.edge === edge
      return <div key={edge} role="slider" tabIndex={0} aria-label={edge === 'start' ? 'Line start' : 'Line end'} aria-valuemin={handles.bounds.start} aria-valuemax={handles.bounds.end} aria-valuenow={value} aria-valuetext={stamp(value)}
        title={`${edge === 'start' ? 'Start' : 'End'} ${stamp(value)}. Drag, or use the arrow keys (Shift for bigger steps).`}
        className="absolute inset-y-0 z-[5] flex w-3 -translate-x-1/2 cursor-ew-resize items-center justify-center focus:outline-none" style={{ left: `${percentIn(view, value)}%` }} {...handleProps(edge)}>
        <span className={`h-full w-1 rounded-full ${active ? 'bg-white' : 'bg-[#e9deff]'} shadow-[0_0_0_1px_#11151d]`} />
        <span className="absolute top-1/2 h-6 w-2.5 -translate-y-1/2 rounded bg-[#e9deff] shadow-[0_0_0_1px_#11151d]" />
        {active && <span className="pointer-events-none absolute bottom-1 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-black px-1 text-[10px] tabular-nums text-white">{stamp(value)}</span>}
      </div>
    })}
  </div>
}
