import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { Pause, Play } from 'lucide-react'
import { api, mediaUrl, type ContextItem, type Cut, type TranscriptSegment, type Waveform } from './api'
import { CutWaveform, type Band, type Handles } from './CutWaveform'
import { IconButton } from './EditorUi'
import { binsFor, contiguousRange, mergeText, newLineSpan, overviewBars, speakerColor, splitTime, stamp, windowFor, type Span } from './lineEditView'
import { useToast } from './Toasts'
import { TranscriptLines, draftDirty, type Draft } from './TranscriptLines'

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

type Props = {
  cut: Cut
  waveform: Waveform | null
  transcript: TranscriptSegment[]
  hasAudio: boolean
  preparing: boolean
  video: RefObject<HTMLVideoElement | null>
  canPlay: boolean
  episodeId: string
  characters: ContextItem[]
  names: Map<string, string>
  busy: boolean
  onAssign: (line: TranscriptSegment, characterId: string | null) => void
  onChanged: () => void
}

type LineBody = { start_time: number; end_time: number; text: string; speaker_label: string; character_context_id: string | null }
const bodyOf = (line: TranscriptSegment): LineBody => ({ start_time: line.start_time, end_time: line.end_time, text: line.text, speaker_label: line.speaker_label, character_context_id: line.character_context_id })

/**
 * The waveform of a cut with its lines, and the list to correct them: rewrite, split, merge, delete, add and retime.
 * Lines are whole sentences, so one may start before the cut or end after it: the graph then starts and ends where the lines do,
 * shows the cut's boundaries, and a line added here cannot be dragged out of the cut.
 */
export function CutLineEditor({ cut, waveform, transcript, hasAudio, preparing, video, canPlay, episodeId, characters, names, busy, onAssign, onChanged }: Props) {
  const toast = useToast()
  const lines = useMemo(() => transcript.filter(line => !line.deleted && !line.superseded_by && line.start_time < cut.end_time && line.end_time > cut.start_time).sort((a, b) => a.start_time - b.start_time || a.id.localeCompare(b.id)), [transcript, cut.start_time, cut.end_time])
  const view = useMemo(() => windowFor(cut, lines), [cut.start_time, cut.end_time, lines])
  const cutSpan = useMemo<Span>(() => ({ start: cut.start_time, end: cut.end_time }), [cut.start_time, cut.end_time])
  const [selected, setSelected] = useState<string[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [working, setWorking] = useState(false)
  // A new line's id reaches the list a moment after it was saved, so ids that are not there (yet) are simply not shown.
  const live = useMemo(() => selected.filter(id => lines.some(line => line.id === id)), [selected, lines])

  // The detailed waveform of exactly the window; the episode overview is drawn until it arrives.
  const audioAvailable = hasAudio && !preparing
  const key = `${episodeId}:${view.start}:${view.end}`
  const [fine, setFine] = useState<{ key: string; peaks: number[] } | null>(null)
  useEffect(() => {
    if (!audioAvailable || view.end <= view.start) return
    let current = true
    api<{ peaks: number[] }>(`/episodes/${encodeURIComponent(episodeId)}/waveform-range?start=${view.start}&end=${view.end}&bins=${binsFor(view)}`)
      .then(result => { if (current) setFine({ key, peaks: result.peaks }) })
      .catch(() => { if (current) setFine({ key, peaks: [] }) })
    return () => { current = false }
  }, [audioAvailable, episodeId, key])
  const fineHere = fine?.key === key ? fine.peaks : null
  const bars = useMemo(() => fineHere?.length ? fineHere : overviewBars(waveform, view, 100), [fineHere, waveform, view])

  // Playback: the episode audio over the whole window. The video stays inside the cut, so only one of them plays.
  const audio = useRef<HTMLAudioElement>(null)
  const stopAt = useRef<number | null>(null)
  const [playing, setPlaying] = useState(false)
  const [audioTime, setAudioTime] = useState(cut.start_time)
  const [source, setSource] = useState<'audio' | 'video'>('video')
  const videoTime = useCutTime(video, cut.start_time, cut.end_time, canPlay)
  const playhead = source === 'audio' ? audioTime : cut.start_time + videoTime
  useEffect(() => {
    const element = audio.current
    if (!element) return
    let frame = 0
    const tick = () => {
      setAudioTime(element.currentTime)
      if (stopAt.current !== null && element.currentTime >= stopAt.current) { element.pause(); return }
      frame = requestAnimationFrame(tick)
    }
    const onPlay = () => { setPlaying(true); setSource('audio'); video.current?.pause(); cancelAnimationFrame(frame); frame = requestAnimationFrame(tick) }
    const onPause = () => { setPlaying(false); cancelAnimationFrame(frame); setAudioTime(element.currentTime); stopAt.current = null }
    element.addEventListener('play', onPlay); element.addEventListener('pause', onPause); element.addEventListener('ended', onPause)
    return () => { cancelAnimationFrame(frame); element.removeEventListener('play', onPlay); element.removeEventListener('pause', onPause); element.removeEventListener('ended', onPause) }
  }, [audioAvailable, video])
  useEffect(() => {
    const player = video.current
    if (!player || !canPlay) return
    const onPlay = () => { audio.current?.pause(); setSource('video') }
    player.addEventListener('play', onPlay)
    return () => player.removeEventListener('play', onPlay)
  }, [video, canPlay])
  // Moving to another cut leaves nothing half-done behind.
  useEffect(() => {
    audio.current?.pause(); stopAt.current = null; setSource('video'); setDraft(null); setSelected([])
  }, [cut.id])

  const playFrom = useCallback((start: number, end: number) => {
    const element = audio.current
    if (!element) return
    element.currentTime = start
    stopAt.current = end
    setAudioTime(start); setSource('audio')
    void element.play().catch(err => toast({ message: (err as Error).message || 'The audio could not be played', tone: 'error' }))
  }, [toast])
  function togglePlay() {
    const element = audio.current
    if (!element) return
    if (playing) { element.pause(); return }
    playFrom(playhead >= view.start && playhead < view.end - 0.05 ? playhead : view.start, view.end)
  }
  function seek(time: number) {
    const element = audio.current
    if (element) { element.currentTime = time; if (!element.paused) stopAt.current = view.end }
    setAudioTime(time); setSource('audio')
    if (video.current && time >= cut.start_time && time <= cut.end_time) video.current.currentTime = time
  }

  // What is being timed: the open draft, else the one selected line.
  const single = !draft && live.length === 1 ? lines.find(line => line.id === live[0]) || null : null
  async function request<T>(path: string, method: string, body?: unknown): Promise<T | null> {
    setWorking(true)
    try { return await api<T>(path, { method, ...(body === undefined ? {} : { body: JSON.stringify(body) }) }) }
    catch (err) { toast({ message: (err as Error).message, tone: 'error' }); return null }
    finally { setWorking(false) }
  }
  async function commitTiming(line: TranscriptSegment, edge: 'start' | 'end', value: number) {
    const result = await request<TranscriptSegment>(`/transcript-segments/${encodeURIComponent(line.id)}`, 'PUT', { ...bodyOf(line), [edge === 'start' ? 'start_time' : 'end_time']: value })
    if (result) { setSelected([result.id]); onChanged() }
  }
  const handles: Handles | null = draft
    ? draft.kind === 'merge' ? null : { span: { start: draft.start, end: draft.end }, bounds: draft.kind === 'new' ? cutSpan : view, onCommit: (edge, value) => setDraft({ ...draft, [edge]: value }) }
    : single ? { span: { start: single.start_time, end: single.end_time }, bounds: view, onCommit: (edge, value) => commitTiming(single, edge, value) } : null

  const bands = useMemo<Band[]>(() => {
    const result: Band[] = []
    for (const line of lines) {
      if (draft?.kind === 'merge' && draft.ids.includes(line.id)) continue
      const editing = draft?.kind === 'edit' && draft.id === line.id
      const name = (line.character_context_id && names.get(line.character_context_id)) || line.speaker_label || 'Unknown speaker'
      result.push({ id: line.id, span: editing ? { start: draft.start, end: draft.end } : { start: line.start_time, end: line.end_time }, color: speakerColor(name), selected: editing || live.includes(line.id), label: `${name}: ${line.text}` })
    }
    if (draft && draft.kind !== 'edit') {
      const name = (draft.character_context_id && names.get(draft.character_context_id)) || draft.speaker_label || 'Unknown speaker'
      result.push({ id: '__draft', span: { start: draft.start, end: draft.end }, color: speakerColor(name), selected: true, label: draft.kind === 'new' ? 'New line' : 'Merged line' })
    }
    return result
  }, [lines, live, draft, names])
  const order = useMemo(() => lines.map(line => line.id), [lines])
  const playable = draft ? { start: draft.start, end: draft.end } : live.length ? (() => { const picked = lines.filter(line => live.includes(line.id)); return { start: Math.min(...picked.map(line => line.start_time)), end: Math.max(...picked.map(line => line.end_time)) } })() : null

  function edit(line: TranscriptSegment) {
    if (draft && draftDirty(draft)) { toast({ message: 'Save or cancel the line you are editing first.' }); return }
    const base = { start: line.start_time, end: line.end_time, text: line.text, speaker_label: line.speaker_label, character_context_id: line.character_context_id }
    setDraft({ kind: 'edit', id: line.id, ids: [], ...base, base })
    setSelected([line.id])
  }
  function add() {
    const span = newLineSpan(cut, playhead)
    const base = { start: span.start, end: span.end, text: '', speaker_label: '', character_context_id: null }
    setDraft({ kind: 'new', id: null, ids: [], ...base, base })
    setSelected([])
  }
  function merge() {
    const picked = lines.filter(line => live.includes(line.id))
    if (picked.length < 2) return
    const base = { start: Math.min(...picked.map(line => line.start_time)), end: Math.max(...picked.map(line => line.end_time)), text: mergeText(picked.map(line => line.text)), speaker_label: picked[0].speaker_label, character_context_id: picked[0].character_context_id }
    setDraft({ kind: 'merge', id: null, ids: picked.map(line => line.id), ...base, base })
  }
  async function save() {
    if (!draft || !draft.text.trim() || draft.end <= draft.start) return
    const body: LineBody = { start_time: draft.start, end_time: draft.end, text: draft.text.trim(), speaker_label: draft.speaker_label, character_context_id: draft.character_context_id }
    const result = draft.kind === 'edit' ? await request<TranscriptSegment>(`/transcript-segments/${encodeURIComponent(draft.id!)}`, 'PUT', body)
      : draft.kind === 'new' ? await request<TranscriptSegment>(`/episodes/${encodeURIComponent(episodeId)}/transcript`, 'POST', body)
      : await request<TranscriptSegment>(`/episodes/${encodeURIComponent(episodeId)}/transcript/merge`, 'POST', { segment_ids: draft.ids, text: body.text, speaker_label: body.speaker_label, character_context_id: body.character_context_id })
    if (result) { setDraft(null); setSelected([result.id]); onChanged() }
  }
  async function split(caret: number) {
    const line = draft?.kind === 'edit' ? lines.find(item => item.id === draft.id) : null
    if (!draft || !line) return
    const at = splitTime(line, draft.text, caret)
    if (at === null) { toast({ message: 'Put the text cursor inside the sentence to split it there.' }); return }
    const result = await request<TranscriptSegment[]>(`/transcript-segments/${encodeURIComponent(line.id)}/split`, 'POST', { at, left: { text: draft.text.slice(0, caret).trim() }, right: { text: draft.text.slice(caret).trim() } })
    if (result) { setDraft(null); setSelected([result[0].id]); onChanged() }
  }
  async function remove() {
    const line = draft?.kind === 'edit' ? lines.find(item => item.id === draft.id) : null
    if (!line) return
    if (await request(`/transcript-segments/${encodeURIComponent(line.id)}`, 'DELETE') === null) return
    setDraft(null); setSelected([]); onChanged()
    const undo = async () => { if (await request<TranscriptSegment>(`/episodes/${encodeURIComponent(episodeId)}/transcript`, 'POST', bodyOf(line))) onChanged() }
    toast({ message: 'Line deleted.', tone: 'success', action: { label: 'Undo', run: () => void undo() } })
  }

  const line = draft?.kind === 'edit' ? lines.find(item => item.id === draft.id) : null
  const canSplit = !!line && draft!.start === line.start_time && draft!.end === line.end_time
  const picked = draft?.kind === 'merge' ? lines.filter(item => draft.ids.includes(item.id)) : []
  const mixedSpeakers = picked.some(item => item.character_context_id !== picked[0].character_context_id || item.speaker_label !== picked[0].speaker_label)
  const message = preparing ? 'Preparing audio and waveform…' : !hasAudio ? 'This episode has no audio track.' : fineHere && !fineHere.length && !bars.length ? 'Waveform unavailable. Run Prepare to rebuild it.' : undefined

  return <div className="flex min-h-0 flex-1 flex-col gap-2">
    <CutWaveform view={view} cut={cutSpan} bars={bars} bands={bands} playhead={playhead} handles={handles} message={message}
      onSeek={seek} onSelectBand={(id, additive) => { if (id !== '__draft') setSelected(additive && live.length ? contiguousRange(order, live[0], id) : [id]) }} />
    {audioAvailable && <audio ref={audio} preload="none" src={mediaUrl(episodeId, 'audio')} className="hidden" />}
    <div className="flex shrink-0 items-center gap-1.5">
      <IconButton icon={playing ? Pause : Play} label={playing ? 'Pause' : 'Play'} size="sm" tone="primary" disabled={!audioAvailable} onClick={togglePlay} />
      <IconButton icon={Play} text="Line" label={playable && live.length > 1 ? 'Play the selected lines' : 'Play the selected line'} size="sm" disabled={!audioAvailable || !playable} onClick={() => playable && playFrom(playable.start, playable.end)} />
      <span className="muted ml-1 text-[10px] tabular-nums" title="Playhead (episode time)">{stamp(playhead)}</span>
      <span className="flex-1" />
      <span className="muted truncate text-[10px] tabular-nums" title="The part of the audio shown above">{stamp(view.start)} → {stamp(view.end)}</span>
    </div>
    <TranscriptLines cut={cut} lines={lines} characters={characters} names={names} selected={live} onSelect={setSelected} draft={draft} onDraft={setDraft} busy={busy || working} time={playhead}
      onEdit={edit} onSave={() => void save()} onCancel={() => setDraft(null)} onDelete={() => void remove()} onSplit={caret => void split(caret)} canSplit={canSplit} onMerge={merge} onAdd={add} onAssign={onAssign}
      hint={mixedSpeakers ? 'These lines have different speakers: pick who speaks the merged line.' : 'Click a line to see its range above; double-click to edit; drag across lines to select several. Lines may extend beyond the cut.'} />
  </div>
}
