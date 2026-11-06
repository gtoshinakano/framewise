import { Fragment, useMemo } from 'react'
import { AudioLines } from 'lucide-react'
import type { Cut, TranscriptSegment, Waveform } from './api'

function stamp(seconds: number) {
  const value = Math.max(0, seconds)
  const minutes = Math.floor(value / 60)
  return `${Math.floor(minutes / 60).toString().padStart(2, '0')}:${(minutes % 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`
}

export function CutAudioPreview({ cut, waveform, transcript, hasAudio, preparing, time, onPlaySegment, canPlay }: { cut: Cut; waveform: Waveform | null; transcript: TranscriptSegment[]; hasAudio: boolean; preparing: boolean; time: number; onPlaySegment: (time: number) => void; canPlay: boolean }) {
  const duration = Math.max(0, cut.end_time - cut.start_time)
  const segments = useMemo(() => transcript.filter(segment => !segment.deleted && !segment.superseded_by && segment.start_time < cut.end_time && segment.end_time > cut.start_time).sort((a, b) => a.start_time - b.start_time), [transcript, cut.start_time, cut.end_time])
  const bars = useMemo(() => {
    if (!waveform?.peaks.length || waveform.duration <= 0 || duration <= 0) return []
    const peaks = waveform.peaks
    return Array.from({ length: 100 }, (_, index) => {
      const from = Math.max(0, Math.min(peaks.length - 1, Math.floor((cut.start_time + duration * index / 100) / waveform.duration * peaks.length)))
      const to = Math.max(from + 1, Math.min(peaks.length, Math.ceil((cut.start_time + duration * (index + 1) / 100) / waveform.duration * peaks.length)))
      let peak = 0
      for (let position = from; position < to; position++) peak = Math.max(peak, peaks[position])
      return Math.max(2, Math.min(90, peak * 90))
    })
  }, [waveform, cut.start_time, duration])
  return <section className="space-y-3 rounded-xl border hairline bg-[#1b202a] p-4" aria-label="Cut audio and transcript">
    <div><h3 className="flex items-center gap-2 text-xs font-semibold"><AudioLines size={15} className="text-[#bca7ff]" /> Cut audio</h3><p className="muted mt-1 text-[11px]">{stamp(cut.start_time)} → {stamp(cut.end_time)} · Episode time</p></div>
    <div className="relative h-16 overflow-hidden rounded-md bg-[#11151d]">
      {preparing ? <p className="muted flex h-full items-center justify-center text-xs">Preparing audio and waveform…</p> : !hasAudio ? <p className="muted flex h-full items-center justify-center text-xs">This episode has no audio track.</p> : bars.length ? <><svg viewBox="0 0 1000 100" preserveAspectRatio="none" className="h-full w-full" role="img" aria-label="Waveform for this cut">{bars.map((height, index) => <rect key={index} x={index * 10 + 2} y={(100 - height) / 2} width="6" height={height} rx="1" fill="#a58ad9" />)}</svg><span className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${duration ? Math.min(100, Math.max(0, time / duration * 100)) : 0}%` }} /></> : <p className="muted flex h-full items-center justify-center text-xs">{waveform ? 'Waveform unavailable. Run Prepare to rebuild it.' : 'Loading waveform…'}</p>}
    </div>
    {bars.length > 0 && !preparing && hasAudio && <p className="muted text-[10px]">Waveform uses the prepared episode overview and may be coarse for short cuts.</p>}
    <div><p className="label !mb-2">Spoken words</p>{segments.length ? <><p className="muted mb-2 text-[11px]">Transcript segments overlapping this cut. {canPlay ? 'Select a chip to play from that point. ' : ''}Segments may extend beyond the cut.</p><div className="scrollbar flex max-h-36 flex-wrap gap-1.5 overflow-y-auto">{segments.map((segment, index) => {
      const active = time + cut.start_time >= segment.start_time && time + cut.start_time < segment.end_time
      const label = `${segment.speaker_label || 'Unknown speaker'} · ${stamp(segment.start_time)}–${stamp(segment.end_time)} · ${segment.text}`
      const content = segment.text
      const speaker = segment.speaker_label || 'Unknown speaker'
      const previousSpeaker = index ? segments[index - 1].speaker_label || 'Unknown speaker' : null
      const chip = canPlay ? <button type="button" title={`${label} · Play from this segment`} aria-label={`Play ${label}`} onClick={() => onPlaySegment(Math.max(cut.start_time, segment.start_time))} className={`rounded-md border px-2 py-1 text-left text-[11px] ${active ? 'border-[#bca7ff] bg-[#322a48]' : 'border-[#38404c] bg-[#232a35] hover:border-[#a58ae8]'}`}>{content}</button> : <span title={label} className="rounded-md border border-[#38404c] bg-[#232a35] px-2 py-1 text-[11px]">{content}</span>
      return <Fragment key={segment.id}>{speaker !== previousSpeaker && <span className="self-center px-1 text-[10px] text-[#aeb5c1]">{speaker}</span>}{chip}</Fragment>
    })}</div></> : <p className="muted text-[11px]">No transcript segments overlap this cut.</p>}</div>
  </section>
}
