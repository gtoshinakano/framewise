import { useEffect, useMemo, useState, type RefObject } from 'react'
import { AudioLines, Download } from 'lucide-react'
import { api, type ContextItem, type Cut, type TranscriptSegment, type Waveform } from './api'
import { CutLineEditor } from './CutLineEditor'
import { speakerColor, stamp } from './lineEditView'
import { loadCharacters } from './runCache'
import { useToast } from './Toasts'
import { unknownVoices } from './voiceGroups'

export function CutAudioPreview({ cut, waveform, transcript, hasAudio, preparing, video, canPlay, projectId, episodeId, onChanged, marks }: { cut: Cut; waveform: Waveform | null; transcript: TranscriptSegment[]; hasAudio: boolean; preparing: boolean; video: RefObject<HTMLVideoElement | null>; canPlay: boolean; projectId: string; episodeId: string; onChanged: () => void; marks?: { voice: boolean; lines: boolean } }) {
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
  // data-need marks where the analysis dialog points the person (cutNeedsView.placeForNeed); the amber edge stays while the need is open.
  return <section data-need="lines" className={`flex flex-col gap-3 rounded-xl border bg-[#1b202a] p-3 lg:min-h-[16rem] lg:flex-1 ${marks?.lines ? 'border-[#6b5a2f]' : 'hairline'}`} aria-label="Cut audio and transcript">
    <div className="flex items-start justify-between gap-2"><div className="min-w-0"><h3 className="flex items-center gap-2 text-xs font-semibold"><AudioLines size={15} className="text-[#bca7ff]" aria-hidden /> Cut audio</h3><p className="muted mt-1 truncate text-[10px] tabular-nums" title="Episode time">{stamp(cut.start_time)} → {stamp(cut.end_time)}</p></div>{hasAudio && !preparing ? <a href={`/api/v1/cuts/${encodeURIComponent(cut.id)}/audio?min_duration=2`} download target="_blank" rel="noreferrer" className="btn btn-quiet shrink-0 !px-2.5 !py-1.5 !text-xs" aria-label="Download audio" title="Download audio. Short cuts include nearby audio or silence to reach two seconds."><Download size={14} aria-hidden /> Audio</a> : <button type="button" disabled className="btn btn-quiet shrink-0 !px-2.5 !py-1.5 !text-xs" aria-label="Download audio" title={preparing ? 'Audio is being prepared' : 'This episode has no audio track'}><Download size={14} aria-hidden /> Audio</button>}</div>
    <div className="flex min-h-0 flex-1 flex-col"><CutLineEditor cut={cut} waveform={waveform} transcript={transcript} hasAudio={hasAudio} preparing={preparing} video={video} canPlay={canPlay} episodeId={episodeId} characters={characters} names={names} busy={busy} onAssign={(segment, id) => void assignLine(segment, id)} onChanged={onChanged} />{voices.length ? <div data-need="voices" className={`mt-2 shrink-0 rounded-md border bg-[#11151d] p-2 ${marks?.voice ? 'border-[#f0a64a]' : 'hairline'}`}><p className="label !mb-1">Unknown voices</p><p className="muted !mb-1.5 text-[10px] leading-4">Pick a character for each voice: every line of that voice in the episode is updated.</p><ul className="space-y-1">{voices.map(voice => <li key={voice.label || '(none)'} className="flex items-center gap-1.5"><span className="min-w-0 flex-1 truncate text-[10px] tabular-nums" title={`${voice.lines} lines${voice.inCut ? `, ${voice.inCut} in this cut` : ''}`}><span className="font-bold uppercase tracking-wide" style={{ color: speakerColor(voice.label || 'Unknown speaker') }}>{voice.label || 'No label'}</span><span className="muted"> · {voice.lines}{voice.inCut ? ` · ${voice.inCut} here` : ''}</span></span><select aria-label={`Character speaking ${voice.label || 'the unlabelled voice'}`} disabled={busy} value="" onChange={event => event.target.value && void assignVoice(voice.label, event.target.value)} className="field shrink-0 !w-[6.2rem] !rounded-md !py-0.5 !text-[10px] !border-[#6d5526] !text-[#f0c979]"><option value="">Assign…</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select></li>)}</ul></div> : null}</div>
  </section>
}
