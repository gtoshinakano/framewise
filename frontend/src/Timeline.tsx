import { CUT_FILTERS, cutsNeedingAction, matchesAnyFilter, nextNeedingAction, type CutFilterKey, type FilterDisplay } from './cutFilters'
import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { AudioLines, ChevronLeft, ChevronRight, Clapperboard, CircleHelp, Crosshair, Layers3, MessageSquareText, Minus, Plus, Scissors, SkipForward } from 'lucide-react'
import type { AudioSelection, Chunk, Cut, TranscriptSegment, Waveform } from './api'
import type { AnalysisQueueState } from './analysisQueueStates'
import type { SequenceSummary } from './sequenceApi'
import { ActiveFilterChips, CutFilterPopover } from './CutFilterPopover'
import { CUT_STATUS_LEGEND, CutStatusIcons } from './CutStatus'
import { IconButton } from './EditorUi'
import { Popover } from './Popover'
import { formatTimecode } from './timecode'

type Props = {
  activeFilters: ReadonlySet<CutFilterKey>
  filterDisplay: FilterDisplay
  onToggleFilter: (key: CutFilterKey) => void
  onClearFilters: () => void
  onFilterDisplay: (display: FilterDisplay) => void
  filtersOpen: boolean
  onFiltersOpenChange: (open: boolean) => void
  /** Undo, redo, split, merge and delete, built by the editor and shown first in the toolbar. */
  editTools: ReactNode
  duration: number
  waveform: Waveform | null
  hasAudio: boolean
  preparing: boolean
  chunks: Chunk[]
  cuts: Cut[]
  cutQueueStates: Map<string, AnalysisQueueState>
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
  audioSelection: AudioSelection | null
  onSelectAudio: (selection: AudioSelection | null) => void
  sequences: SequenceSummary[]
  onOpenSequence: (sequenceId: string) => void
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

function SelectionBand({ selection, start, span }: { selection: AudioSelection | null; start: number; span: number }) {
  if (!selection || selection.end <= start || selection.start >= start + span) return null
  const { left, width } = blockStyle({ start_time: selection.start, end_time: selection.end }, start, span)
  return <div aria-hidden="true" className="pointer-events-none absolute inset-y-0 border-x border-[#7bc5cb] bg-[#7bc5cb33]" style={{ left, width }} />
}

function WaveformCanvas({ waveform, duration, start, span, currentTime, onSeek, hasAudio, preparing, selection, onSelect }: Pick<Props, 'waveform' | 'duration' | 'currentTime' | 'onSeek' | 'hasAudio' | 'preparing'> & { start: number; span: number; selection: AudioSelection | null; onSelect: (selection: AudioSelection) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const selecting = useRef<{ pointerId: number; anchor: number } | null>(null)
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
  const timeAt = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return start + Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * span
  }
  const tidy = (seconds: number) => Math.round(seconds * 100) / 100
  const select = (anchor: number, time: number) => onSelect({ start: tidy(Math.min(anchor, time)), end: tidy(Math.max(anchor, time)) })
  return <div className="relative h-16 cursor-crosshair touch-none overflow-hidden rounded-md bg-[#1b202b]" title="Drag to seek. Shift-drag to select audio to transcribe again." onPointerDown={event => {
    if (event.button !== 0) return
    event.currentTarget.setPointerCapture(event.pointerId)
    const time = timeAt(event)
    if (event.shiftKey) { selecting.current = { pointerId: event.pointerId, anchor: time }; return }
    onSeek(time)
  }} onPointerMove={event => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return
    const drag = selecting.current
    if (drag?.pointerId === event.pointerId) select(drag.anchor, timeAt(event))
    else if (!drag) onSeek(timeAt(event))
  }} onPointerUp={event => { if (selecting.current?.pointerId === event.pointerId) selecting.current = null }} onPointerCancel={() => { selecting.current = null }} onLostPointerCapture={() => { selecting.current = null }}>
    <canvas ref={canvas} className="h-full w-full" />
    {!waveform?.peaks.length && <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-[#1b202b] px-2 text-center text-[11px] text-[#a9b1c0]">{preparing ? 'Preparing waveform…' : !waveform ? 'Loading waveform…' : hasAudio ? 'Waveform unavailable. Run Prepare to rebuild it.' : 'No audio track in this video.'}</div>}
    <SelectionBand selection={selection} start={start} span={span} />
    {playhead >= 0 && playhead <= 100 && <div className="pointer-events-none absolute inset-y-0 w-px bg-[#eee7ff]" style={{ left: `${playhead}%` }} />}
  </div>
}

function blockStyle(item: { start_time: number; end_time: number }, start: number, span: number) {
  const left = Math.max(0, (Math.max(item.start_time, start) - start) / span * 100)
  const right = Math.min(100, (Math.min(item.end_time, start + span) - start) / span * 100)
  return { left: `${left}%`, width: `${Math.min(100 - left, Math.max(0.8, right - left))}%` }
}

export function Timeline(props: Props) {
  const { activeFilters, filterDisplay, onToggleFilter, onClearFilters, onFilterDisplay, filtersOpen, onFiltersOpenChange, editTools, duration, waveform, hasAudio, preparing, chunks, cuts, cutQueueStates, transcript, currentTime, seekRevision, selectedCutId, selectedChunkId, selectedTranscriptId, onSeek, onSelectCut, onOpenCutAnalysis, onSelectChunk, onSelectTranscript, audioSelection, onSelectAudio, sequences, onOpenSequence } = props
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
  const actionCuts = cutsNeedingAction(cuts, activeFilters)
  function selectNextNeedingAction() {
    const next = nextNeedingAction(cuts, selectedCutId, activeFilters)
    if (next) onSelectCut(next)
  }
  const step = (direction: 1 | -1) => { followPlayhead.current = false; setRequestedStart(Math.max(0, Math.min(maxStart, start + direction * span * 0.75))) }
  const visibleSpan = span >= duration ? 'Full video' : `${span < 60 ? span.toFixed(1) + 's' : formatTimecode(span, { precision: 0 })} visible`
  return <section aria-label="Timeline" className="bg-[#11151d] px-3 pb-3 pt-2.5">
    <div role="toolbar" aria-label="Timeline tools" className="mb-2.5 flex flex-wrap items-center gap-x-1.5 gap-y-2 text-xs">
      {editTools}
      <span className="mx-1 h-5 w-px bg-[#2c323d]" aria-hidden />
      <IconButton icon={Minus} label="Zoom out" size="sm" className="ghost" disabled={duration <= 1 || zoomValue <= 0} onClick={() => setZoom(Math.max(0, zoomValue - 10))} />
      <input id="timeline-zoom" type="range" min="0" max="100" step="1" value={zoomValue} disabled={duration <= 1} onChange={event => setZoom(Number(event.target.value))} aria-label="Zoom" aria-valuetext={`${span.toFixed(1)} seconds visible`} className="w-24 accent-[#a58ae8]" title="Zoom (or scroll over the timeline)" />
      <IconButton icon={Plus} label="Zoom in" size="sm" className="ghost" disabled={duration <= 1 || zoomValue >= 100} onClick={() => setZoom(Math.min(100, zoomValue + 10))} />
      <IconButton icon={ChevronLeft} label="Earlier" size="sm" className="ghost" disabled={start <= 0} onClick={() => step(-1)} />
      <IconButton icon={ChevronRight} label="Later" size="sm" className="ghost" disabled={start >= maxStart} onClick={() => step(1)} />
      <IconButton icon={Crosshair} label="Center the playhead" size="sm" className="ghost" onClick={() => { followPlayhead.current = true; center(currentTime) }} />
      <span className="muted hidden whitespace-nowrap xl:inline">{visibleSpan}</span>
      <span className="mx-1 h-5 w-px bg-[#2c323d]" aria-hidden />
      <output aria-label="Playhead time" className="whitespace-nowrap font-mono tabular-nums text-[#e8dfff]" title="Playhead">{preciseStamp(currentTime)}</output>
      <span className="muted whitespace-nowrap">{cuts.length} {cuts.length === 1 ? 'cut' : 'cuts'}</span>
      <span className="ml-auto flex min-w-0 flex-wrap items-center justify-end gap-1.5">
        <ActiveFilterChips active={activeFilters} onToggle={onToggleFilter} />
        <CutFilterPopover cuts={cuts} active={activeFilters} display={filterDisplay} onToggle={onToggleFilter} onClear={onClearFilters} onDisplay={onFilterDisplay} open={filtersOpen} onOpenChange={onFiltersOpenChange} />
        <IconButton icon={SkipForward} label={activeFilters.size ? 'Next cut matching the filters' : 'Next cut needing action'} shortcut="N" size="sm" text={<span className="font-mono tabular-nums">{actionCuts.length}</span>} disabled={!actionCuts.length} onClick={selectNextNeedingAction} />
        <TimelineHelp />
      </span>
    </div>
    <div ref={zoomSurface} className={panning ? 'timeline-panning' : undefined} onPointerDownCapture={startMiddlePan} onPointerMoveCapture={moveMiddlePan} onPointerUpCapture={endMiddlePan} onPointerCancelCapture={endMiddlePan} onLostPointerCapture={endMiddlePan} onMouseDownCapture={event => { if (event.button === 1) event.preventDefault() }} onAuxClickCapture={event => { if (event.button === 1) event.preventDefault() }}>
    <div className="mb-2 grid grid-cols-[28px_minmax(0,1fr)] items-center gap-2"><span className="muted text-center text-[10px]" title="Whole video: drag the window to move, click to seek">All</span><div ref={overview} role="slider" aria-label="Video overview" aria-valuemin={0} aria-valuemax={Math.max(duration, 1)} aria-valuenow={Math.min(currentTime, duration)} aria-valuetext={preciseStamp(currentTime)} tabIndex={0} onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { event.preventDefault(); seekOverview(Math.max(0, Math.min(duration, currentTime + (event.key === 'ArrowRight' ? 1 : -1) * Math.max(1, span / 10)))) } }} onPointerDown={startOverviewDrag} onPointerMove={moveOverviewDrag} onPointerUp={finishOverviewDrag} onPointerCancel={() => { overviewDrag.current = null }} onLostPointerCapture={() => { overviewDrag.current = null }} className="relative h-5 cursor-crosshair touch-none overflow-hidden rounded bg-[#262c39]"><div className={`absolute inset-y-0 rounded border border-[#ab8df2] bg-[#8d70c266] ${maxStart > 0 ? 'cursor-grab active:cursor-grabbing' : ''}`} style={{ left: `${duration ? start / duration * 100 : 0}%`, width: `${duration ? Math.min(100, span / duration * 100) : 100}%` }} /><div className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${duration ? currentTime / duration * 100 : 0}%` }} /></div></div>
    <div className="timeline-grid relative mb-1.5 ml-[36px] flex justify-between border-b hairline pb-1 font-mono text-[10px] text-[#7e8797]">{Array.from({ length: 6 }, (_, index) => <span key={index}>{stamp(start + span * index / 5)}</span>)}</div>
    <div className="grid grid-cols-[28px_minmax(0,1fr)] gap-x-2 gap-y-1.5">
      <LaneLabel icon={<Scissors size={15} />} label="Cuts" />
      <div className="relative h-15 overflow-hidden rounded-md bg-[#1b202b]">{cuts.filter(cut => visible(cut) && (!activeFilters.size || filterDisplay === 'dim' || matchesAnyFilter(cut, activeFilters))).map(cut => {
        const queueState = cutQueueStates.get(cut.id)
        const queueLabel = queueState === 'running' ? 'Analysis running' : queueState === 'queued' ? 'Analysis queued' : ''
        const matched = activeFilters.size ? CUT_FILTERS.filter(filter => activeFilters.has(filter.key) && filter.matches(cut)) : []
        const dimmed = activeFilters.size > 0 && !matched.length
        return <button key={cut.id} title={`Cut ${cut.cut_index + 1} · ${stamp(cut.start_time)}–${stamp(cut.end_time)}${cut.analysis_enabled ? '' : ' · Analysis disabled'} · ${cut.has_analysis_results ? 'Has analysis results' : 'No analysis results'} · ${cut.has_boxes ? 'Boxes detected' : 'No boxes'} · ${cut.no_characters ? 'No characters' : cut.has_characters ? 'Character detected' : 'No character labels'} · ${cut.has_summary ? 'Has cut summary' : 'No cut summary'}${queueLabel ? ` · ${queueLabel}` : ''}${matched.length ? ` · Needs: ${matched.map(filter => filter.label).join(', ')}` : ''}`} aria-label={`Cut ${cut.cut_index + 1}, ${cut.has_analysis_results ? 'has analysis results' : 'no analysis results'}${cut.analysis_enabled ? '' : ', analysis disabled'}${queueLabel ? `, ${queueLabel}` : ''}${dimmed ? ', matches no filter' : ''}`} onClick={() => onSelectCut(cut)} onDoubleClick={() => onOpenCutAnalysis(cut)} className={`cut-block ${cut.analysis_enabled ? '' : 'cut-block-disabled'} absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 pt-[3px] text-left text-[10px] transition-opacity ${dimmed && selectedCutId !== cut.id ? 'opacity-25 hover:opacity-60' : ''} ${!cut.analysis_enabled ? selectedCutId === cut.id ? 'bg-[#a05b36] text-white' : 'bg-[#75462f] text-[#ffe1c4]' : selectedCutId === cut.id ? 'bg-[#9476d555] text-white ring-1 ring-inset ring-[#c3aeff]' : 'bg-[#5c497d66] text-[#d4c4ed]'}`} style={blockStyle(cut, start, span)}>
          {matched.length > 0 && <span aria-hidden className="absolute inset-x-0 top-0 flex h-[3px]">{matched.map(filter => <span key={filter.key} className="flex-1" style={{ background: filter.color }} />)}</span>}
          <span className="flex items-center gap-1"><CutStatusIcons cut={cut} queueState={queueState} /><span className="truncate">{cut.title}{!cut.analysis_enabled && ' · Disabled'}</span></span><span className="block truncate text-[9px] opacity-70"><b className="text-[10px] font-extrabold opacity-100">{cut.cut_index + 1}</b> · {stamp(cut.start_time)}</span>{queueState && <span aria-hidden className={`absolute inset-x-0 bottom-0 h-1 ${queueState === 'running' ? 'bg-[#f5c678]' : 'bg-[#86d1e1]'}`} />}</button>
      })}{!cuts.length && <div className="muted flex h-full items-center justify-center text-xs">Run hard-cut detection to create timeline segments</div>}{playhead >= 0 && playhead <= 100 && <div className="pointer-events-none absolute inset-y-0 w-px bg-[#eee7ff]" style={{ left: `${playhead}%` }} />}</div>
      <LaneLabel icon={<AudioLines size={15} />} label="Audio waveform: drag to seek, Shift-drag to select audio to transcribe again" />
      <WaveformCanvas waveform={waveform} duration={duration} start={start} span={span} currentTime={currentTime} onSeek={onSeek} hasAudio={hasAudio} preparing={preparing} selection={audioSelection} onSelect={onSelectAudio} />
      <LaneLabel icon={<MessageSquareText size={15} />} label="Speech: click a line to edit it" />
      <div className="relative h-10 overflow-hidden rounded-md bg-[#1b202b]">{transcript.some(visible) ? transcript.filter(visible).map(segment => <button key={segment.id} title={`${segment.speaker_label || 'Speaker'}: ${segment.text}`} onClick={() => onSelectTranscript(segment)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedTranscriptId === segment.id ? 'bg-[#c2828e88] text-white' : 'bg-[#8d506566] text-[#f0c8d0]'}`} style={blockStyle(segment, start, span)}><span className="block truncate">{segment.speaker_label || 'Speaker'}</span><span className="block truncate text-[9px] opacity-70">{segment.text}</span></button>) : <div className="muted flex h-full items-center justify-center text-xs">No speech annotations in this window</div>}<SelectionBand selection={audioSelection} start={start} span={span} /></div>
      <LaneLabel icon={<Layers3 size={15} />} label="Chunks" />
      <div className="relative h-8 overflow-hidden rounded-md bg-[#1b202b]">{chunks.filter(visible).map(chunk => <button key={chunk.id} title={`Chunk ${chunk.chunk_index + 1} · ${stamp(chunk.start_time)}–${stamp(chunk.end_time)}`} onClick={() => onSelectChunk(chunk)} className={`cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] px-1 text-left text-[10px] ${selectedChunkId === chunk.id ? 'bg-[#5d9ca777] text-white' : 'bg-[#3e717e66] text-[#bce7e8]'}`} style={blockStyle(chunk, start, span)}><span className="block truncate">{chunk.title || `Chunk ${chunk.chunk_index + 1}`}</span><span className="block truncate text-[9px] opacity-70">{stamp(chunk.start_time)}</span></button>)}</div>
      <LaneLabel icon={<Clapperboard size={15} />} label="Sequences" />
      <div className="relative h-8 overflow-hidden rounded-md bg-[#1b202b]">{sequences.some(visible) ? sequences.filter(visible).map(sequence => <button key={sequence.id} title={`${sequence.title} · cuts ${sequence.first_cut ?? '?'}–${sequence.last_cut ?? '?'} · ${sequence.shots_written}/${sequence.shots_planned} shot prompts${sequence.cuts_changed ? ' · cuts changed' : ''}`} onClick={() => onOpenSequence(sequence.id)} className="cut-block absolute inset-y-0 overflow-hidden border-r border-[#10131a] bg-[#7d5a3066] px-1 text-left text-[10px] text-[#f1d9b8]" style={blockStyle(sequence, start, span)}><span className="block truncate">{sequence.title}</span><span className="block truncate text-[9px] opacity-70">{sequence.shots_planned ? `${sequence.shots_written}/${sequence.shots_planned} prompts` : 'not directed'}</span></button>) : <div className="muted flex h-full items-center justify-center text-[11px]">{sequences.length ? 'No sequence in this window' : 'Create a sequence in the Sequences tab to direct a range of cuts as one short video'}</div>}</div>

    </div></div>
  </section>
}

/** A lane's icon with its name as tooltip and accessible label, so the lanes get the width. */
function LaneLabel({ icon, label }: { icon: ReactNode; label: string }) {
  return <div className="flex items-center justify-center text-[#98a1af]" title={label} aria-label={label} role="img">{icon}</div>
}

const SHORTCUTS: [string, string][] = [
  ['Space', 'Play or pause'], ['S', 'Split the cut under the playhead'], ['Ctrl+Z', 'Undo the last cut edit'], ['Ctrl+Shift+Z', 'Redo'],
  ['[ / ]', 'Previous / next cut'], ['N', 'Next cut needing action'], ['F', 'Filters'], ['Ctrl+S', 'Save the cut notes'],
]

/** Icon legend, mouse gestures and keyboard shortcuts, out of the way until asked for. */
function TimelineHelp() {
  return <Popover label="Timeline help" side="top" align="end" panelClassName="w-[320px] p-3 text-xs"
    trigger={props => <button type="button" {...props} className="icon-btn icon-btn-sm ghost" aria-label="Timeline help" title="Legend and shortcuts"><CircleHelp size={15} aria-hidden /></button>}>
    <h3 className="mb-2 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">Cut icons</h3>
    <ul className="grid grid-cols-2 gap-x-3 gap-y-1.5">{CUT_STATUS_LEGEND.map(item => <li key={item.label} className="flex items-center gap-1.5"><item.icon size={13} style={{ color: item.color }} aria-hidden />{item.label}</li>)}</ul>
    <h3 className="mb-2 mt-4 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">Mouse</h3>
    <ul className="muted space-y-1"><li>Scroll over the timeline to zoom; middle-drag to move it.</li><li>Drag the waveform to seek; Shift-drag it to choose audio to transcribe again.</li><li>Double-click a cut to open its analysis.</li><li>Drag the top edge of the timeline to resize it.</li></ul>
    <h3 className="mb-2 mt-4 text-[10px] font-bold uppercase tracking-[.12em] text-[#7e8797]">Keyboard</h3>
    <dl className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1.5">{SHORTCUTS.map(([key, text]) => <div key={key} className="contents"><dt><kbd className="kbd">{key}</kbd></dt><dd className="muted">{text}</dd></div>)}</dl>
  </Popover>
}
