import { useEffect, useMemo, useState, type RefObject } from 'react'
import { AudioLines, Download } from 'lucide-react'
import { api, type ContextItem, type Cut, type TranscriptSegment, type Waveform } from './api'
import { loadCharacters } from './runCache'
import { useToast } from './Toasts'
import { unknownVoices } from './voiceGroups'

function stamp(seconds: number) {
  const value = Math.max(0, seconds)
  const minutes = Math.floor(value / 60)
  return `${Math.floor(minutes / 60).toString().padStart(2, '0')}:${(minutes % 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`
}

const SPEAKER_COLORS = ['#6aa8ff', '#e2709c', '#4cc9c0', '#f0a64a', '#b496ff', '#7bd88f', '#e5d25a', '#ff8a94']
/** The same speaker label always gets the same colour, so a change of speaker is visible at a glance. */
function speakerColor(label: string) {
  let hash = 0
  for (const char of label) hash = (hash * 31 + char.charCodeAt(0)) >>> 0
  return SPEAKER_COLORS[hash % SPEAKER_COLORS.length]
}

/** Seconds from the start of the cut. It lives here, not in the dialog, so a playing video does not re-render the whole Results tab. */
function useCutTime(video: RefObject<HTMLVideoElement | null>, start: number, end: number, enabled: boolean) {
  const [time, setTime] = useState(0)
  useEffect(() => {
    const player = video.current
    if (!player || !enabled) return
    const sync = () => setTime(Math.max(0, Math.min(end - start, player.currentTime - start)))
    const events = ['timeupdate', 'seeked', 'loadedmetadata'] as const
    for (const name of events) player.addEventListener(name, sync)
    sync()
    return () => { for (const name of events) player.removeEventListener(name, sync) }
  }, [video, start, end, enabled])
  return time
}

export function CutAudioPreview({ cut, waveform, transcript, hasAudio, preparing, video, onPlaySegment, canPlay, projectId, episodeId, onChanged }: { cut: Cut; waveform: Waveform | null; transcript: TranscriptSegment[]; hasAudio: boolean; preparing: boolean; video: RefObject<HTMLVideoElement | null>; onPlaySegment: (time: number) => void; canPlay: boolean; projectId: string; episodeId: string; onChanged: () => void }) {
  const duration = Math.max(0, cut.end_time - cut.start_time)
  const time = useCutTime(video, cut.start_time, cut.end_time, canPlay)
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
  const toast = useToast()
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [busy, setBusy] = useState(false)
  useEffect(() => { loadCharacters(projectId).then(setCharacters).catch(() => { /* names only label the choices */ }) }, [projectId])
  const names = useMemo(() => new Map(characters.map(item => [item.id, item.title])), [characters])
  const voices = useMemo(() => unknownVoices(transcript, cut), [transcript, cut.start_time, cut.end_time])
  const assignLine = async (segment: TranscriptSegment, character_context_id: string | null) => {
    setBusy(true)
    try { await api(`/transcript-segments/${encodeURIComponent(segment.id)}`, { method: 'PUT', body: JSON.stringify({ start_time: segment.start_time, end_time: segment.end_time, text: segment.text, speaker_label: segment.speaker_label, character_context_id }) }) }
    catch (err) { toast({ message: (err as Error).message, tone: 'error' }) }
    finally { setBusy(false); onChanged() }
  }
  const assignVoice = async (label: string, character_context_id: string) => {
    const name = names.get(character_context_id) || 'the character'
    setBusy(true)
    try {
      const { updated } = await api<{ updated: number }>(`/episodes/${encodeURIComponent(episodeId)}/transcript/assign-voice`, { method: 'POST', body: JSON.stringify({ speaker_label: label, character_context_id }) })
      toast({ tone: updated ? 'success' : 'info', message: updated ? `${updated} line${updated === 1 ? '' : 's'} of ${label || 'the unlabelled voice'} assigned to ${name}.` : `${name} already spoke every line of that voice.` })
    } catch (err) { toast({ message: (err as Error).message, tone: 'error' }) }
    finally { setBusy(false); onChanged() }
  }
  return <section className="flex flex-col gap-3 rounded-xl border hairline bg-[#1b202a] p-3 lg:min-h-[16rem] lg:flex-1" aria-label="Cut audio and transcript">
    <div className="flex items-start justify-between gap-2"><div className="min-w-0"><h3 className="flex items-center gap-2 text-xs font-semibold"><AudioLines size={15} className="text-[#bca7ff]" aria-hidden /> Cut audio</h3><p className="muted mt-1 truncate text-[10px] tabular-nums" title="Episode time">{stamp(cut.start_time)} → {stamp(cut.end_time)}</p></div>{hasAudio && !preparing ? <a href={`/api/v1/cuts/${encodeURIComponent(cut.id)}/audio?min_duration=2`} download target="_blank" rel="noreferrer" className="btn btn-quiet shrink-0 !px-2.5 !py-1.5 !text-xs" aria-label="Download audio" title="Download audio. Short cuts include nearby audio or silence to reach two seconds."><Download size={14} aria-hidden /> Audio</a> : <button type="button" disabled className="btn btn-quiet shrink-0 !px-2.5 !py-1.5 !text-xs" aria-label="Download audio" title={preparing ? 'Audio is being prepared' : 'This episode has no audio track'}><Download size={14} aria-hidden /> Audio</button>}</div>
    <div className="relative h-12 overflow-hidden rounded-md bg-[#11151d]" title="Waveform uses the prepared episode overview and may be coarse for short cuts.">
      {preparing ? <p className="muted flex h-full items-center justify-center text-xs">Preparing audio and waveform…</p> : !hasAudio ? <p className="muted flex h-full items-center justify-center text-xs">This episode has no audio track.</p> : bars.length ? <><svg viewBox="0 0 1000 100" preserveAspectRatio="none" className="h-full w-full" role="img" aria-label="Waveform for this cut">{bars.map((height, index) => <rect key={index} x={index * 10 + 2} y={(100 - height) / 2} width="6" height={height} rx="1" fill="#a58ad9" />)}</svg><span className="pointer-events-none absolute inset-y-0 w-px bg-white" style={{ left: `${duration ? Math.min(100, Math.max(0, time / duration * 100)) : 0}%` }} /></> : <p className="muted flex h-full items-center justify-center text-xs">{waveform ? 'Waveform unavailable. Run Prepare to rebuild it.' : 'Loading waveform…'}</p>}
    </div>
    <div className="flex min-h-0 flex-1 flex-col"><p className="label !mb-1.5 shrink-0">Spoken words</p>{segments.length ? <><ul className="scrollbar min-h-0 max-h-80 flex-1 space-y-1 overflow-y-auto pr-0.5 lg:max-h-none">{segments.map(segment => {
      const active = time + cut.start_time >= segment.start_time && time + cut.start_time < segment.end_time
      const assigned = segment.character_context_id ? names.get(segment.character_context_id) : null
      const speaker = assigned || segment.speaker_label || 'Unknown speaker'
      const color = speakerColor(speaker)
      const label = `${speaker} · ${stamp(segment.start_time)}–${stamp(segment.end_time)} · ${segment.text}`
      const row = <><span style={{ color }} className="block truncate text-[10px] font-bold uppercase tracking-wide">{speaker}</span><span className="block break-words text-[12px] leading-[1.45]">{segment.text}</span></>
      const rowClass = `flex-1 min-w-0 rounded-md border-l-[3px] px-2 py-1 text-left ${active ? 'bg-[#322a48]' : 'bg-[#232a35]'}`
      const speakerSelect = <select aria-label={assigned ? `Character speaking "${segment.text.slice(0, 48)}"` : 'Who speaks this line?'} disabled={busy} value={segment.character_context_id || ''} onChange={event => void assignLine(segment, event.target.value || null)} className={`field shrink-0 !w-[6.2rem] !rounded-md !py-0.5 !text-[10px] ${segment.character_context_id ? '' : '!border-[#6d5526] !text-[#f0c979]'}`}><option value="">Unassigned</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select>
      return <li key={segment.id} className="flex items-start gap-1.5">{canPlay
        ? <button type="button" title={`${label} · Play from this segment`} aria-label={`Play ${label}`} aria-current={active || undefined} onClick={() => onPlaySegment(Math.max(cut.start_time, segment.start_time))} style={{ borderLeftColor: color }} className={`${rowClass} hover:bg-[#303746]`}>{row}</button>
        : <div title={label} style={{ borderLeftColor: color }} className={rowClass}>{row}</div>}{speakerSelect}</li>
    })}</ul><p className="muted mt-1.5 shrink-0 text-[10px] leading-4">{canPlay ? 'Select a line to play from there. ' : ''}Lines may extend beyond the cut.</p></> : <p className="muted text-[11px]">No transcript segments overlap this cut.</p>}{voices.length ? <div className="mt-2 shrink-0 rounded-md border hairline bg-[#11151d] p-2"><p className="label !mb-1">Unknown voices</p><p className="muted !mb-1.5 text-[10px] leading-4">Pick a character for each voice: every line of that voice in the episode is updated.</p><ul className="space-y-1">{voices.map(voice => <li key={voice.label || '(none)'} className="flex items-center gap-1.5"><span className="min-w-0 flex-1 truncate text-[10px] tabular-nums" title={`${voice.lines} lines${voice.inCut ? `, ${voice.inCut} in this cut` : ''}`}><span className="font-bold uppercase tracking-wide" style={{ color: speakerColor(voice.label || 'Unknown speaker') }}>{voice.label || 'No label'}</span><span className="muted"> · {voice.lines}{voice.inCut ? ` · ${voice.inCut} here` : ''}</span></span><select aria-label={`Character speaking ${voice.label || 'the unlabelled voice'}`} disabled={busy} value="" onChange={event => event.target.value && void assignVoice(voice.label, event.target.value)} className="field shrink-0 !w-[6.2rem] !rounded-md !py-0.5 !text-[10px] !border-[#6d5526] !text-[#f0c979]"><option value="">Assign…</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select></li>)}</ul></div> : null}</div>
  </section>
}
