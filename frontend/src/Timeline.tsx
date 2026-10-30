import { useEffect, useRef, useState } from 'react'
import { AudioLines, Layers3, Scissors } from 'lucide-react'
import type { Chunk, Cut, TranscriptSegment, Waveform } from './api'

type Props = {
  duration: number
  waveform: Waveform | null
  chunks: Chunk[]
  cuts: Cut[]
  transcript: TranscriptSegment[]
  currentTime: number
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

function WaveformCanvas({ waveform, duration, start, span, currentTime, onSeek }: Pick<Props, 'waveform' | 'duration' | 'currentTime' | 'onSeek'> & { start: number; span: number }) {
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
      for (let x = 0; x < width; x += 3) {
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
  return <div className="relative h-16 cursor-crosshair touch-none overflow-hidden rounded-md bg-[#1b202b]" onPointerDown={event => { event.currentTarget.setPointerCapture(event.pointerId); const rect = event.currentTarget.getBoundingClientRect(); onSeek(start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span) }} onPointerMove={event => { if (!event.currentTarget.hasPointerCapture(event.pointerId)) return; const rect = event.currentTarget.getBoundingClientRect(); onSeek(start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span) }}>
    <canvas ref={canvas} className="h-full w-full" />
    {playhead >= 0 && playhead <= 100 && <div className="pointer-events-none absolute inset-y-0 w-px bg-[#eee7ff]" style={{ left: `${playhead}%` }} />}
  </div>
}

function blockStyle(item: { start_time: number; end_time: number }, start: number, span: number) {
  const left = Math.max(0, (Math.max(item.start_time, start) - start) / span * 100)
  const right = Math.min(100, (Math.min(item.end_time, start + span) - start) / span * 100)
  return { left: `${left}%`, width: `${Math.min(100 - left, Math.max(0.8, right - left))}%` }
}

export function Timeline(props: Props) {
  const { duration, waveform, chunks, cuts, transcript, currentTime, selectedCutId, selectedChunkId, selectedTranscriptId, onSeek, onSelectCut, onOpenCutAnalysis, onSelectChunk, onSelectTranscript } = props
  const [windowSeconds, setWindowSeconds] = useState<number | null>(() => duration > 1800 ? 600 : null)
  const [requestedStart, setRequestedStart] = useState(0)
  const span = Math.min(windowSeconds || duration || 1, duration || 1)
  const maxStart = Math.max(0, duration - span)
  const start = Math.min(maxStart, Math.max(0, requestedStart))
  const end = start + span
  const visible = (item: { start_time: number; end_time: number }) => item.start_time < end && item.end_time > start
  const center = (value: number) => setRequestedStart(Math.max(0, Math.min(maxStart, value - span / 2)))
  useEffect(() => {
    const selected = cuts.find(cut => cut.id === selectedCutId)
    if (selected && (selected.start_time < start || selected.start_time >= end)) center(selected.start_time)
  }, [selectedCutId])
  useEffect(() => {
    if (currentTime < start || currentTime > end) center(currentTime)
  }, [currentTime])
  const minSpan = Math.min(duration || 1, 1)
  const zoomValue = duration > minSpan ? Math.log(duration / span) / Math.log(duration / minSpan) * 100 : 0
  function setZoom(value: number) {
    const nextSpan = duration / Math.pow(duration / minSpan, value / 100)
    setWindowSeconds(value === 0 ? null : nextSpan)
    setRequestedStart(Math.max(0, Math.min(duration - nextSpan, currentTime - nextSpan / 2)))
  }
  const playhead = (currentTime - start) / span * 100
  return <section className="border-t hairline bg-[#11151d] px-4 py-3">
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-sm font-semibold">Timeline</h2><p className="muted mt-1 text-xs">Drag the waveform playhead or select a cut to seek</p></div><span className="muted text-xs">{cuts.length} {cuts.length === 1 ? 'cut' : 'cuts'}</span></div>
    <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
      <label htmlFor="timeline-zoom" className="muted">Zoom</label>
      <span className="muted" aria-hidden="true">−</span><input id="timeline-zoom" type="range" min="0" max="100" step="1" value={zoomValue} disabled={duration <= 1} onChange={event => setZoom(Number(event.target.value))} aria-valuetext={`${span.toFixed(1)} seconds visible`} className="w-32 accent-[#a58ae8]" /><span className="muted" aria-hidden="true">+</span>
      <span className="muted">{span >= duration ? 'Full video' : `${span.toFixed(1)}s visible`}</span>
      <button type="button" disabled={start <= 0} onClick={() => setRequestedStart(Math.max(0, start - span * 0.75))} className="btn btn-quiet px-2 py-1.5 text-xs">← Earlier</button>
      <button type="button" disabled={start >= maxStart} onClick={() => setRequestedStart(Math.min(maxStart, start + span * 0.75))} className="btn btn-quiet px-2 py-1.5 text-xs">Later →</button>
      <button type="button" onClick={() => center(currentTime)} className="btn btn-quiet px-2 py-1.5 text-xs">Center playhead</button>
      <span className="muted ml-auto font-mono text-[11px]">{stamp(start)} – {stamp(Math.min(duration, end))}</span>
    </div>
    <div className="mb-5 grid grid-cols-[86px_minmax(0,1fr)] items-center gap-3"><span className="muted text-[11px]">Overview</span><div role="slider" aria-label="Video overview" aria-valuemin={0} aria-valuemax={Math.max(duration, 1)} aria-valuenow={Math.min(currentTime, duration)} tabIndex={0} onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); const target = Math.max(0, Math.min(duration, currentTime + (event.key === 'ArrowRight' ? 1 : -1) * Math.max(1, span / 10))); center(target); onSeek(target) } }} onClick={event => { const rect = event.currentTarget.getBoundingClientRect(); const target = Math.max(0, Math.min(duration, (event.clientX - rect.left) / rect.width * duration)); center(target); onSeek(target) }} className="relative h-4 cursor-crosshair overflow-hidden rounded bg-[#262c39]"><div className="absolute inset-y-0 rounded border border-[#ab8df2] bg-[#8d70c266]" style={{ left: `${duration ? start / duration * 100 : 0}%`, width: `${duration ? Math.min(100, span / duration * 100) : 100}%` }} /><div className="absolute inset-y-0 w-px bg-white" style={{ left: `${duration ? currentTime / duration * 100 : 0}%` }} /></div></div>
    <div className="timeline-grid relative mb-2 flex justify-between border-b hairline pb-2 font-mono text-[10px] text-[#7e8797]">{Array.from({ length: 6 }, (_, index) => <span key={index}>{stamp(start + span * index / 5)}</span>)}</div>
    <div className="grid grid-cols-[86px_minmax(0,1fr)] gap-3">
      <div className="flex h-15 items-center gap-2 text-xs text-[#98a1af]"><Scissors size={16} /> Cuts</div>
      <div className="relative h-15 overflow-hidden rounded-md bg-[#1b202b]">{cuts.filter(visible).map(cut => <button key={cut.id} title={`Cut ${cut.cut_index + 1} · ${stamp(cut.start_time)}–${stamp(cut.end_time)}`} onClick={() => onSelectCut(cut)} onDoubleClick={() => onOpenCutAnalysis(cut)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedCutId === cut.id ? 'bg-[#9476d555] text-white' : 'bg-[#5c497d66] text-[#d4c4ed]'}`} style={blockStyle(cut, start, span)}><span className="block truncate">{cut.title || `Cut ${cut.cut_index + 1}`}</span><span className="block truncate text-[9px] opacity-70">{stamp(cut.start_time)}</span></button>)}{!cuts.length && <div className="muted flex h-full items-center justify-center text-xs">Run hard-cut detection to create timeline segments</div>}{playhead >= 0 && playhead <= 100 && <div className="pointer-events-none absolute inset-y-0 w-px bg-[#eee7ff]" style={{ left: `${playhead}%` }} />}</div>
      <div className="flex h-16 items-center gap-2 text-xs text-[#98a1af]"><AudioLines size={16} /> Audio</div>
      <WaveformCanvas waveform={waveform} duration={duration} start={start} span={span} currentTime={currentTime} onSeek={onSeek} />
      <div className="flex h-12 items-center gap-2 text-xs text-[#98a1af]"><AudioLines size={16} /> Speech</div>
      <div className="relative h-12 overflow-hidden rounded-md bg-[#1b202b]">{transcript.some(visible) ? transcript.filter(visible).map(segment => <button key={segment.id} title={`${segment.speaker_label || 'Speaker'}: ${segment.text}`} onClick={() => onSelectTranscript(segment)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedTranscriptId === segment.id ? 'bg-[#c2828e88] text-white' : 'bg-[#8d506566] text-[#f0c8d0]'}`} style={blockStyle(segment, start, span)}><span className="block truncate">{segment.speaker_label || 'Speaker'}</span><span className="block truncate text-[9px] opacity-70">{segment.text}</span></button>) : <div className="muted flex h-full items-center justify-center text-xs">No speech annotations in this window</div>}</div>
      <div className="flex h-12 items-center gap-2 text-xs text-[#98a1af]"><Layers3 size={16} /> Chunks</div>
      <div className="relative h-12 overflow-hidden rounded-md bg-[#1b202b]">{chunks.filter(visible).map(chunk => <button key={chunk.id} title={`Chunk ${chunk.chunk_index + 1} · ${stamp(chunk.start_time)}–${stamp(chunk.end_time)}`} onClick={() => onSelectChunk(chunk)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedChunkId === chunk.id ? 'bg-[#5d9ca777] text-white' : 'bg-[#3e717e66] text-[#bce7e8]'}`} style={blockStyle(chunk, start, span)}><span className="block truncate">{chunk.title || `Chunk ${chunk.chunk_index + 1}`}</span><span className="block truncate text-[9px] opacity-70">{stamp(chunk.start_time)}</span></button>)}</div>

    </div>
  </section>
}
