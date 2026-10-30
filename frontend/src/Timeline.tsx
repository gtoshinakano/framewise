import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { AudioLines, Layers3, Scissors } from 'lucide-react'
import type { Chunk, Cut, TranscriptSegment, Waveform } from './api'

type Props = {
  duration: number
  waveform: Waveform | null
  hasAudio: boolean
  preparing: boolean
  chunks: Chunk[]
  cuts: Cut[]
  transcript: TranscriptSegment[]
  currentTime: number
  seekRevision: number
  selectedCutId: string | null
  selectedChunkId: string | null
  selectedTranscriptId: string | null
  onSeek: (time: number) => void
  onSelectCut: (cut: Cut) => void
  onOpenCutAnalysis: (cut: Cut) => void
  onSelectChunk: (chunk: Chunk) => void
  onSelectTranscript: (segment: TranscriptSegment) => void
}

function stamp(seconds: number) {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

function preciseStamp(seconds: number) {
  const milliseconds = Math.round(Math.max(0, Number.isFinite(seconds) ? seconds : 0) * 1000)
  return `${stamp(Math.floor(milliseconds / 1000))}.${String(milliseconds % 1000).padStart(3, '0')}`
}

function WaveformCanvas({ waveform, duration, start, span, currentTime, onSeek, hasAudio, preparing }: Pick<Props, 'waveform' | 'duration' | 'currentTime' | 'onSeek' | 'hasAudio' | 'preparing'> & { start: number; span: number }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const element = canvas.current
    if (!element) return
    const draw = () => {
      const width = element.clientWidth, height = element.clientHeight, dpr = window.devicePixelRatio || 1
      element.width = width * dpr
      element.height = height * dpr
      const ctx = element.getContext('2d')
      if (!ctx) return
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, width, height)
      ctx.fillStyle = '#a58ad9'
      const peaks = waveform?.peaks || []
      for (let x = 0; peaks.length && x < width; x += 3) {
        const from = Math.max(0, Math.floor((start + x / width * span) / Math.max(duration, 1) * peaks.length))
        const to = Math.min(peaks.length, Math.ceil((start + (x + 3) / width * span) / Math.max(duration, 1) * peaks.length))
        let peak = 0
        for (let index = from; index < Math.max(from + 1, to) && index < peaks.length; index++) peak = Math.max(peak, peaks[index])
        const bar = Math.max(2, Math.min(height - 8, peak * (height - 8) * 2.2))
        ctx.fillRect(x, (height - bar) / 2, 2, bar)
      }
      if (!peaks.length) { ctx.fillStyle = '#4e5665'; ctx.fillRect(0, height / 2, width, 1) }
    }
    draw()
    const observer = new ResizeObserver(draw)
    observer.observe(element)
    return () => observer.disconnect()
  }, [waveform, duration, start, span])
  const playhead = (currentTime - start) / span * 100
  return <div className="relative h-16 cursor-crosshair touch-none overflow-hidden rounded-md bg-[#1b202b]" onPointerDown={event => { if (event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); const rect = event.currentTarget.getBoundingClientRect(); onSeek(start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span) }} onPointerMove={event => { if (!event.currentTarget.hasPointerCapture(event.pointerId)) return; const rect = event.currentTarget.getBoundingClientRect(); onSeek(start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span) }}>
    <canvas ref={canvas} className="h-full w-full" />
    {!waveform?.peaks.length && <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-[#1b202b] px-2 text-center text-[11px] text-[#a9b1c0]">{preparing ? 'Preparing waveform…' : !waveform ? 'Loading waveform…' : hasAudio ? 'Waveform unavailable. Run Prepare to rebuild it.' : 'No audio track in this video.'}</div>}
    {playhead >= 0 && playhead <= 100 && <div className="pointer-events-none absolute inset-y-0 w-px bg-[#eee7ff]" style={{ left: `${playhead}%` }} />}
  </div>
}

function blockStyle(item: { start_time: number; end_time: number }, start: number, span: number) {
  const left = Math.max(0, (Math.max(item.start_time, start) - start) / span * 100)
  const right = Math.min(100, (Math.min(item.end_time, start + span) - start) / span * 100)
  return { left: `${left}%`, width: `${Math.min(100 - left, Math.max(0.8, right - left))}%` }
}

export function Timeline(props: Props) {
  const { duration, waveform, hasAudio, preparing, chunks, cuts, transcript, currentTime, seekRevision, selectedCutId, selectedChunkId, selectedTranscriptId, onSeek, onSelectCut, onOpenCutAnalysis, onSelectChunk, onSelectTranscript } = props
  const [windowSeconds, setWindowSeconds] = useState<number | null>(() => duration > 1800 ? 600 : null)
  const [requestedStart, setRequestedStart] = useState(0)
  const followPlayhead = useRef(true)
  const overviewDrag = useRef<{ pointerId: number; x: number; start: number; mode: 'pan' | 'seek'; moved: boolean } | null>(null)
  const middleDrag = useRef<{ pointerId: number; x: number; start: number; span: number; width: number; maxStart: number } | null>(null)
  const [panning, setPanning] = useState(false)
  const zoomSurface = useRef<HTMLDivElement>(null)
  const overview = useRef<HTMLDivElement>(null)
  const span = Math.min(windowSeconds || duration || 1, duration || 1)
  const maxStart = Math.max(0, duration - span)
  const start = Math.min(maxStart, Math.max(0, requestedStart))
  const end = start + span
  const visible = (item: { start_time: number; end_time: number }) => item.start_time < end && item.end_time > start
  const center = (value: number) => setRequestedStart(Math.max(0, Math.min(maxStart, value - span / 2)))
  useEffect(() => {
    const selected = cuts.find(cut => cut.id === selectedCutId)
    if (selected) {
      followPlayhead.current = true
      if (selected.start_time < start || selected.start_time >= end) center(selected.start_time)
    }
  }, [selectedCutId])
  useEffect(() => {
    followPlayhead.current = true
  }, [seekRevision])
  useEffect(() => {
    if (followPlayhead.current && (currentTime < start || currentTime > end)) center(currentTime)
  }, [currentTime, seekRevision])
  const minSpan = Math.min(duration || 1, 1)
  const zoomWindow = useRef({ duration, start, span })
  zoomWindow.current = { duration, start, span }
  useEffect(() => {
    const surface = zoomSurface.current
    if (!surface) return
    const onWheel = (event: WheelEvent) => {
      if (middleDrag.current) { event.preventDefault(); return }
      const view = zoomWindow.current
      if (view.duration <= 1 || !event.deltaY || Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return
      const rect = overview.current?.getBoundingClientRect()
      if (!rect?.width) return
      event.preventDefault()
      const delta = event.deltaY * (event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? 800 : 1)
      const nextSpan = Math.max(1, Math.min(view.duration, view.span * Math.exp(Math.max(-250, Math.min(250, delta)) * 0.002)))
      if (nextSpan === view.span) return
      const position = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
      const anchor = view.start + position * view.span
      const nextStart = Math.max(0, Math.min(view.duration - nextSpan, anchor - position * nextSpan))
      followPlayhead.current = false
      zoomWindow.current = { ...view, span: nextSpan, start: nextStart }
      setWindowSeconds(nextSpan === view.duration ? null : nextSpan)
      setRequestedStart(nextStart)
    }
    surface.addEventListener('wheel', onWheel, { passive: false })
    return () => surface.removeEventListener('wheel', onWheel)
  }, [])
  const zoomValue = duration > minSpan ? Math.log(duration / span) / Math.log(duration / minSpan) * 100 : 0
  function setZoom(value: number) {
    const nextSpan = duration / Math.pow(duration / minSpan, value / 100)
    setWindowSeconds(value === 0 ? null : nextSpan)
    setRequestedStart(Math.max(0, Math.min(duration - nextSpan, currentTime - nextSpan / 2)))
  }
  function overviewTime(event: PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return rect.width ? Math.max(0, Math.min(duration, (event.clientX - rect.left) / rect.width * duration)) : 0
  }
  function seekOverview(target: number) {
    followPlayhead.current = true
    center(target)
    onSeek(target)
  }
  function startOverviewDrag(event: PointerEvent<HTMLDivElement>) {
    if (!duration || event.button !== 0) return
    const target = overviewTime(event)
    const mode = maxStart > 0 && target >= start && target <= end ? 'pan' : 'seek'
    overviewDrag.current = { pointerId: event.pointerId, x: event.clientX, start, mode, moved: false }
    event.currentTarget.setPointerCapture(event.pointerId)
    if (mode === 'seek') seekOverview(target)
  }
  function moveOverviewDrag(event: PointerEvent<HTMLDivElement>) {
    const drag = overviewDrag.current
    if (!drag || drag.pointerId !== event.pointerId) return
    if (drag.mode === 'seek') { seekOverview(overviewTime(event)); return }
    const delta = event.clientX - drag.x
    if (Math.abs(delta) >= 3) drag.moved = true
    if (drag.moved) {
      followPlayhead.current = false
      const width = event.currentTarget.getBoundingClientRect().width
      if (width) setRequestedStart(Math.max(0, Math.min(maxStart, drag.start + delta / width * duration)))
    }
  }
  function finishOverviewDrag(event: PointerEvent<HTMLDivElement>) {
    const drag = overviewDrag.current
    if (!drag || drag.pointerId !== event.pointerId) return
    if (drag.mode === 'pan' && !drag.moved) seekOverview(overviewTime(event))
    overviewDrag.current = null
  }
  function startMiddlePan(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerType !== 'mouse' || event.button !== 1) return
    event.preventDefault()
    event.stopPropagation()
    const width = overview.current?.getBoundingClientRect().width || 0
    if (!width || maxStart <= 0) return
    middleDrag.current = { pointerId: event.pointerId, x: event.clientX, start, span, width, maxStart }
    event.currentTarget.setPointerCapture(event.pointerId)
    setPanning(true)
  }
  function moveMiddlePan(event: PointerEvent<HTMLDivElement>) {
    const drag = middleDrag.current
    if (!drag || drag.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopPropagation()
    const delta = event.clientX - drag.x
    if (delta) {
      followPlayhead.current = false
      setRequestedStart(Math.max(0, Math.min(drag.maxStart, drag.start - delta / drag.width * drag.span)))
    }
  }
  function endMiddlePan(event: PointerEvent<HTMLDivElement>) {
    if (middleDrag.current?.pointerId !== event.pointerId) return
    middleDrag.current = null
    setPanning(false)
    event.stopPropagation()
  }
  const playhead = (currentTime - start) / span * 100
  return <section className="border-t hairline bg-[#11151d] px-4 py-3">
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-sm font-semibold">Timeline</h2><p className="muted mt-1 text-xs">Scroll to zoom, middle-drag to move the timeline, or drag the waveform to seek</p></div><div className="flex items-center gap-4"><span className="text-xs text-[#e8dfff]">Playhead <output aria-label="Playhead time" className="font-mono tabular-nums">{preciseStamp(currentTime)}</output></span><span className="muted text-xs">{cuts.length} {cuts.length === 1 ? 'cut' : 'cuts'}</span></div></div>
    <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
      <label htmlFor="timeline-zoom" className="muted">Zoom</label>
      <span className="muted" aria-hidden="true">−</span><input id="timeline-zoom" type="range" min="0" max="100" step="1" value={zoomValue} disabled={duration <= 1} onChange={event => setZoom(Number(event.target.value))} aria-valuetext={`${span.toFixed(1)} seconds visible`} className="w-32 accent-[#a58ae8]" /><span className="muted" aria-hidden="true">+</span>
      <span className="muted">{span >= duration ? 'Full video' : `${span.toFixed(1)}s visible`}</span>
      <button type="button" disabled={start <= 0} onClick={() => { followPlayhead.current = false; setRequestedStart(Math.max(0, start - span * 0.75)) }} className="btn btn-quiet px-2 py-1.5 text-xs">← Earlier</button>
      <button type="button" disabled={start >= maxStart} onClick={() => { followPlayhead.current = false; setRequestedStart(Math.min(maxStart, start + span * 0.75)) }} className="btn btn-quiet px-2 py-1.5 text-xs">Later →</button>
      <button type="button" onClick={() => { followPlayhead.current = true; center(currentTime) }} className="btn btn-quiet px-2 py-1.5 text-xs">Center playhead</button>
      <span className="muted ml-auto font-mono text-[11px]">{stamp(start)} – {stamp(Math.min(duration, end))}</span>
    </div>
    <div ref={zoomSurface} className={panning ? 'timeline-panning' : undefined} onPointerDownCapture={startMiddlePan} onPointerMoveCapture={moveMiddlePan} onPointerUpCapture={endMiddlePan} onPointerCancelCapture={endMiddlePan} onLostPointerCapture={endMiddlePan} onMouseDownCapture={event => { if (event.button === 1) event.preventDefault() }} onAuxClickCapture={event => { if (event.button === 1) event.preventDefault() }}>
    <div className="mb-5 grid grid-cols-[86px_minmax(0,1fr)] items-center gap-3"><span className="muted text-[11px]">Overview</span><div ref={overview} role="slider" aria-label="Video overview" aria-valuemin={0} aria-valuemax={Math.max(duration, 1)} aria-valuenow={Math.min(currentTime, duration)} aria-valuetext={preciseStamp(currentTime)} tabIndex={0} onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); seekOverview(Math.max(0, Math.min(duration, currentTime + (event.key === 'ArrowRight' ? 1 : -1) * Math.max(1, span / 10)))) } }} onPointerDown={startOverviewDrag} onPointerMove={moveOverviewDrag} onPointerUp={finishOverviewDrag} onPointerCancel={() => { overviewDrag.current = null }} onLostPointerCapture={() => { overviewDrag.current = null }} className="relative h-5 cursor-crosshair touch-none overflow-hidden rounded bg-[#262c39]"><div className={`absolute inset-y-0 rounded border border-[#ab8df2] bg-[#8d70c266] ${maxStart > 0 ? 'cursor-grab active:cursor-grabbing' : ''}`} style={{ left: `${duration ? start / duration * 100 : 0}%`, width: `${duration ? Math.min(100, span / duration * 100) : 100}%` }} /><div className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${duration ? currentTime / duration * 100 : 0}%` }} /></div></div>
    <div className="timeline-grid relative mb-2 flex justify-between border-b hairline pb-2 font-mono text-[10px] text-[#7e8797]">{Array.from({ length: 6 }, (_, index) => <span key={index}>{stamp(start + span * index / 5)}</span>)}</div>
    <div className="grid grid-cols-[86px_minmax(0,1fr)] gap-3">
      <div className="flex h-15 items-center gap-2 text-xs text-[#98a1af]"><Scissors size={16} /> Cuts</div>
      <div className="relative h-15 overflow-hidden rounded-md bg-[#1b202b]">{cuts.filter(visible).map(cut => <button key={cut.id} title={`Cut ${cut.cut_index + 1} · ${stamp(cut.start_time)}–${stamp(cut.end_time)}`} onClick={() => onSelectCut(cut)} onDoubleClick={() => onOpenCutAnalysis(cut)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedCutId === cut.id ? 'bg-[#9476d555] text-white' : 'bg-[#5c497d66] text-[#d4c4ed]'}`} style={blockStyle(cut, start, span)}><span className="block truncate">{cut.title || `Cut ${cut.cut_index + 1}`}</span><span className="block truncate text-[9px] opacity-70">{stamp(cut.start_time)}</span></button>)}{!cuts.length && <div className="muted flex h-full items-center justify-center text-xs">Run hard-cut detection to create timeline segments</div>}{playhead >= 0 && playhead <= 100 && <div className="pointer-events-none absolute inset-y-0 w-px bg-[#eee7ff]" style={{ left: `${playhead}%` }} />}</div>
      <div className="flex h-16 items-center gap-2 text-xs text-[#98a1af]"><AudioLines size={16} /> Audio</div>
      <WaveformCanvas waveform={waveform} duration={duration} start={start} span={span} currentTime={currentTime} onSeek={onSeek} hasAudio={hasAudio} preparing={preparing} />
      <div className="flex h-12 items-center gap-2 text-xs text-[#98a1af]"><AudioLines size={16} /> Speech</div>
      <div className="relative h-12 overflow-hidden rounded-md bg-[#1b202b]">{transcript.some(visible) ? transcript.filter(visible).map(segment => <button key={segment.id} title={`${segment.speaker_label || 'Speaker'}: ${segment.text}`} onClick={() => onSelectTranscript(segment)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedTranscriptId === segment.id ? 'bg-[#c2828e88] text-white' : 'bg-[#8d506566] text-[#f0c8d0]'}`} style={blockStyle(segment, start, span)}><span className="block truncate">{segment.speaker_label || 'Speaker'}</span><span className="block truncate text-[9px] opacity-70">{segment.text}</span></button>) : <div className="muted flex h-full items-center justify-center text-xs">No speech annotations in this window</div>}</div>
      <div className="flex h-12 items-center gap-2 text-xs text-[#98a1af]"><Layers3 size={16} /> Chunks</div>
      <div className="relative h-12 overflow-hidden rounded-md bg-[#1b202b]">{chunks.filter(visible).map(chunk => <button key={chunk.id} title={`Chunk ${chunk.chunk_index + 1} · ${stamp(chunk.start_time)}–${stamp(chunk.end_time)}`} onClick={() => onSelectChunk(chunk)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedChunkId === chunk.id ? 'bg-[#5d9ca777] text-white' : 'bg-[#3e717e66] text-[#bce7e8]'}`} style={blockStyle(chunk, start, span)}><span className="block truncate">{chunk.title || `Chunk ${chunk.chunk_index + 1}`}</span><span className="block truncate text-[9px] opacity-70">{stamp(chunk.start_time)}</span></button>)}</div>

    </div></div>
  </section>
}
