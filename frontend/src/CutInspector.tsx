import { useEffect, useState, type KeyboardEvent } from 'react'
import { ArrowLeft, ArrowRight, Copy, Frame, NotebookPen, Play, RotateCcw, Save, ScanSearch, Scissors, Timer, WandSparkles } from 'lucide-react'
import type { Cut, Job } from './api'
import type { AnalysisQueueState } from './analysisQueueStates'
import { CutFramePair } from './CutFramePair'
import { CutStatusIcons } from './CutStatus'
import { IconButton, Section, Switch } from './EditorUi'
import { PANEL_ACCENT } from './InspectorRail'
import { TimeRangeInput } from './TimecodeInput'
import { formatDuration, formatTimecode } from './timecode'

const MIN_LENGTH = 0.1
type Notes = Pick<Cut, 'title' | 'description' | 'notes'>

/** The Cut panel: what the selected cut is, its notes and frames, and precise timing. Split, merge and delete live in the timeline toolbar. */
export function CutInspector({ cut, cuts, queueState, time, duration, mediaBusy, busy, frameVersion, sceneThreshold, detectionJob, onSelect, onPlay, onSeek, onOpenAnalysis, onToggleAnalysis, onSaveNotes, onApplyTiming, onDetectWithin, onCopy }: {
  cut: Cut
  cuts: Cut[]
  queueState?: AnalysisQueueState
  time: number
  duration: number
  mediaBusy: boolean
  busy: boolean
  frameVersion: string
  sceneThreshold: number
  detectionJob?: Job
  onSelect: (cut: Cut) => void
  onPlay: (start: number, end: number) => void
  onSeek: (time: number) => void
  onOpenAnalysis: () => void
  onToggleAnalysis: () => void
  onSaveNotes: (notes: Notes) => Promise<void>
  onApplyTiming: (start: number, end: number) => void
  onDetectWithin: (threshold: number) => void
  onCopy: (text: string) => void
}) {
  const [notes, setNotes] = useState<Notes>(() => ({ title: cut.title, description: cut.description, notes: cut.notes }))
  const [saving, setSaving] = useState(false)
  const [range, setRange] = useState<{ start: number | null; end: number | null }>({ start: cut.start_time, end: cut.end_time })
  const [threshold, setThreshold] = useState(String(sceneThreshold))
  // A different cut, or the same cut changed elsewhere, starts from what is saved.
  useEffect(() => { setNotes({ title: cut.title, description: cut.description, notes: cut.notes }) }, [cut.id, cut.title, cut.description, cut.notes])
  useEffect(() => { setRange({ start: cut.start_time, end: cut.end_time }) }, [cut.id, cut.start_time, cut.end_time])
  useEffect(() => { setThreshold(String(sceneThreshold)) }, [cut.id, sceneThreshold])
  const dirty = notes.title !== cut.title || notes.description !== cut.description || notes.notes !== cut.notes
  const index = cuts.findIndex(item => item.id === cut.id)
  const previous = cut.active ? cuts[index - 1] : undefined
  const next = cut.active ? cuts[index + 1] : undefined
  async function save() {
    if (!dirty || saving) return
    setSaving(true)
    try { await onSaveNotes(notes) } finally { setSaving(false) }
  }
  function onKey(event: KeyboardEvent<HTMLDivElement>) {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); void save() }
  }
  // What the typed times would do to the neighbours, checked like the server checks them.
  const start = range.start ?? cut.start_time, end = range.end ?? cut.end_time
  const timingChanged = Math.abs(start - cut.start_time) > 1e-6 || Math.abs(end - cut.end_time) > 1e-6
  const timingError = end - start < MIN_LENGTH ? 'The cut must be at least 0.1 s long.'
    : previous && start - previous.start_time < MIN_LENGTH ? `Cut ${index} would become shorter than 0.1 s.`
    : next && next.end_time - end < MIN_LENGTH ? `Cut ${index + 2} would become shorter than 0.1 s.` : ''
  const thresholdValue = Number(threshold)
  const thresholdValid = threshold.trim() !== '' && Number.isFinite(thresholdValue) && thresholdValue >= 0.05 && thresholdValue <= 0.9
  const rangeText = `${formatTimecode(cut.start_time)} – ${formatTimecode(cut.end_time)}`
  return <div className="flex h-full min-h-0 flex-col" onKeyDown={onKey}>
    <div className="border-b hairline px-4 pb-3 pt-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-[.14em]" style={{ color: PANEL_ACCENT.cut }}>{cut.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut'}<span className="muted ml-2 normal-case tracking-normal">{cut.active ? `of ${cuts.length}` : ''}</span></p>
        <div className="flex items-center gap-1">
          <IconButton icon={ArrowLeft} label="Previous cut" shortcut="[" size="sm" className="ghost" disabled={!previous} onClick={() => previous && onSelect(previous)} />
          <IconButton icon={ArrowRight} label="Next cut" shortcut="]" size="sm" className="ghost" disabled={!next} onClick={() => next && onSelect(next)} />
          <IconButton icon={Play} label="Play this cut" size="sm" onClick={() => onPlay(cut.start_time, cut.end_time)} />
        </div>
      </div>
      <div className="mt-1 flex items-center gap-2">
        <button type="button" onClick={() => onCopy(rangeText)} title="Copy the time range" className="group flex min-w-0 items-center gap-1.5 font-mono text-[15px] font-semibold tabular-nums hover:text-white">{rangeText}<Copy size={12} className="muted opacity-0 group-hover:opacity-100" aria-hidden /></button>
        <span className="shrink-0 rounded-md bg-[#282438] px-1.5 py-0.5 text-[10px] text-[#c3aeff]">{formatDuration(cut.end_time - cut.start_time)}</span>
      </div>
      <div className="mt-2.5 flex items-center justify-between gap-2">
        {cut.active ? <Switch checked={cut.analysis_enabled} onChange={onToggleAnalysis} label={cut.analysis_enabled ? 'Analysis on' : 'Analysis off'} hint="Cuts with analysis off are skipped by batch runs and filters" /> : <span className="muted text-xs">Kept after a redetection, merge or delete</span>}
        <CutStatusIcons cut={cut} queueState={queueState} size={14} />
      </div>
      <button onClick={onOpenAnalysis} className="btn btn-primary mt-3 w-full !py-2"><WandSparkles size={15} /> Analysis &amp; results</button>
    </div>
    <div className="scrollbar min-h-0 flex-1 overflow-y-auto pb-4">
      <Section id="cut.notes" title="Notes" icon={NotebookPen} accent={PANEL_ACCENT.cut} badge={dirty ? <span className="ml-1 h-1.5 w-1.5 rounded-full bg-[#f2c879]" title="Unsaved changes" /> : null}>
        <div className="space-y-3">
          <label className="block"><span className="label">Title</span><input className="field !py-2" value={notes.title} onChange={event => setNotes({ ...notes, title: event.target.value })} placeholder="Name this cut" /></label>
          <label className="block"><span className="label">Cut description</span><textarea className="field min-h-20 resize-y !py-2" value={notes.description} onChange={event => setNotes({ ...notes, description: event.target.value })} placeholder="What happens in this shot?" /></label>
          <label className="block"><span className="label">Working notes</span><textarea className="field min-h-20 resize-y !py-2" value={notes.notes} onChange={event => setNotes({ ...notes, notes: event.target.value })} placeholder="Visual details, scene context, follow-up ideas…" /></label>
        </div>
      </Section>
      <Section id="cut.frames" title="Frames" icon={Frame} accent={PANEL_ACCENT.cut}>
        <CutFramePair key={`${cut.id}-${frameVersion}`} cut={cut} onSeek={onSeek} frameVersion={frameVersion} />
      </Section>
      {cut.active && <Section id="cut.timing" title="Timing & detection" icon={Timer} accent={PANEL_ACCENT.cut} defaultOpen={false} badge={timingChanged ? <span className="ml-1 h-1.5 w-1.5 rounded-full bg-[#f2c879]" title="Not applied yet" /> : null}>
        <TimeRangeInput start={range.start} end={range.end} onChange={setRange} min={0} max={duration} playhead={time} onPlay={onPlay}
          lockStart={mediaBusy || index === 0} lockEnd={mediaBusy || index === cuts.length - 1} />
        <ul className="muted mt-2 space-y-0.5 text-[11px]">
          {index === 0 ? <li>The first cut always starts at 0:00.000.</li> : timingChanged && previous && Math.abs(start - cut.start_time) > 1e-6 && <li>Cut {index} will end at <span className="font-mono text-[#e4e5ea]">{formatTimecode(start)}</span>.</li>}
          {index === cuts.length - 1 ? <li>The last cut always ends with the video.</li> : timingChanged && next && Math.abs(end - cut.end_time) > 1e-6 && <li>Cut {index + 2} will start at <span className="font-mono text-[#e4e5ea]">{formatTimecode(end)}</span>.</li>}
        </ul>
        {timingChanged && timingError && <p role="alert" className="mt-2 text-[11px] text-[#ff9fa7]">{timingError}</p>}
        <div className="mt-3 flex gap-2">
          <button className="btn btn-quiet flex-1 !py-2" disabled={!timingChanged} onClick={() => setRange({ start: cut.start_time, end: cut.end_time })}><RotateCcw size={14} /> Reset</button>
          <button className="btn btn-primary flex-1 !py-2" disabled={mediaBusy || !timingChanged || !!timingError} onClick={() => onApplyTiming(start, end)} title={mediaBusy ? 'Wait for the current media job to finish' : 'Neighbouring cuts adjust so the whole video stays covered. Undo with Ctrl+Z.'}>Apply timing</button>
        </div>
        <div className="mt-4 rounded-lg border hairline p-3">
          <p className="flex items-center gap-1.5 text-xs font-semibold"><ScanSearch size={14} className="text-[#bca7ff]" aria-hidden /> Detect cuts inside this cut</p>
          <p className="muted mt-1 text-[11px] leading-4">Lower thresholds find more changes. The first part keeps this cut’s notes. Undo with Ctrl+Z.</p>
          <div className="mt-2 flex items-end gap-2">
            <label className="w-24"><span className="label !mb-1 !text-[10px]">Threshold</span><input className="field !py-1.5" type="number" min="0.05" max="0.9" step="0.01" value={threshold} onChange={event => setThreshold(event.target.value)} aria-invalid={!thresholdValid} /></label>
            <button className="btn btn-quiet flex-1 !py-2" disabled={busy || !thresholdValid} onClick={() => onDetectWithin(thresholdValue)} title={busy ? 'Wait for the running jobs to finish' : undefined}><Scissors size={14} /> Detect</button>
          </div>
          {detectionJob && <p role="status" className="muted mt-2 text-[11px]">{detectionJob.error || detectionJob.stage}</p>}
        </div>
      </Section>}
    </div>
    {dirty && <div className="flex items-center justify-between gap-2 border-t hairline bg-[#171b24] px-4 py-2.5">
      <span className="text-[11px] text-[#f2c879]">Unsaved notes</span>
      <div className="flex gap-2">
        <button className="btn btn-quiet !px-3 !py-1.5 !text-xs" onClick={() => setNotes({ title: cut.title, description: cut.description, notes: cut.notes })}>Discard</button>
        <button className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={saving} onClick={() => void save()} title="Save notes (Ctrl+S)"><Save size={14} /> Save</button>
      </div>
    </div>}
  </div>
}
