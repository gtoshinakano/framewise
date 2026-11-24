import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Play, Plus, Trash2 } from 'lucide-react'
import { api, type Episode, type EpisodeSummary, type TranscriptSegment } from './api'
import { CutWaveform } from './CutWaveform'
import { IconButton } from './EditorUi'
import { PlayheadStore, usePlayheadValue } from './playhead'
import { TimeRangeInput } from './TimecodeInput'
import { formatTimecode } from './timecode'
import type { Span } from './lineEditView'

export type VoiceRange = { start_time: number; end_time: number }
export type VoiceSample = { id: string; episode_title: string | null; ranges: VoiceRange[]; source_filename: string | null; preview_available: boolean }
export type VoiceDraft = Record<string, VoiceRange[]>
const empty: VoiceRange[] = []
const stamp = (value: number) => formatTimecode(value, { precision: 2 })

export function RegistrationVoice({ episodes, initialEpisode, time, draft, onDraft, samples, removed, onRemove }: {
  episodes: EpisodeSummary[]; initialEpisode?: string; time?: number; draft: VoiceDraft; onDraft: (draft: VoiceDraft) => void
  samples: VoiceSample[]; removed: string[]; onRemove: (ids: string[]) => void
}) {
  const [episodeId, setEpisodeId] = useState(initialEpisode || episodes[0]?.id || '')
  const [episode, setEpisode] = useState<Episode | null>(null)
  const [lines, setLines] = useState<TranscriptSegment[]>([])
  const [beats, setBeats] = useState<number[]>([])
  const [zoom, setZoom] = useState(20)
  const [center, setCenter] = useState(time ?? 0)
  const [selection, setSelection] = useState<Span>({ start: time ?? 0, end: (time ?? 0) + 3 })
  const [error, setError] = useState('')
  const [preview, setPreview] = useState('')
  const [previewing, setPreviewing] = useState(false)
  const request = useRef<AbortController | null>(null)
  const previewUrl = useRef('')
  const playhead = useMemo(() => new PlayheadStore(), [])
  const cursor = usePlayheadValue(playhead, value => value)
  const pending = draft[episodeId] ?? empty
  const duration = episode?.duration || 0
  const span = Math.min(zoom, duration)
  const start = Math.max(0, Math.min(duration - span, center - span / 2))
  const end = start + span
  const view = { start, end }
  const bandDuration = pending.reduce((sum, range) => sum + range.end_time - range.start_time, 0)
  const selectable = selection.start >= 0 && selection.end > selection.start && selection.end <= duration
  const canAdd = selectable && pending.length < 50 && bandDuration + selection.end - selection.start <= 60
    && !pending.some(range => selection.start < range.end_time && range.start_time < selection.end)
  const nearLines = lines.filter(line => line.end_time > start && line.start_time < end)

  useEffect(() => {
    if (!episodeId) return
    const signal = new AbortController()
    setEpisode(null); setLines([]); setError('')
    Promise.all([
      api<Episode>(`/episodes/${episodeId}`, { signal: signal.signal }),
      api<TranscriptSegment[]>(`/episodes/${episodeId}/transcript`, { signal: signal.signal }),
    ]).then(([info, transcript]) => {
      setEpisode(info); setLines(transcript)
      const length = info.duration || 0
      const anchor = episodeId === initialEpisode ? Math.min(time ?? 0, length) : 0
      playhead.set(anchor)
      setCenter(anchor); setSelection({ start: Math.min(anchor, Math.max(0, length - 3)), end: Math.min(anchor + 3, length) })
    }).catch(reason => { if (!signal.signal.aborted) setError(reason.message) })
    return () => signal.abort()
  }, [episodeId, initialEpisode, playhead, time])

  useEffect(() => {
    const signal = new AbortController()
    setBeats([])
    if (episode?.has_audio && start < end) api<{ peaks: number[] }>(`/episodes/${episodeId}/waveform-range?start=${start}&end=${end}`, { signal: signal.signal })
      .then(result => setBeats(result.peaks)).catch(reason => { if (!signal.signal.aborted) setError(reason.message) })
    return () => signal.abort()
  }, [episode?.has_audio, episodeId, start, end])

  useEffect(() => () => { request.current?.abort(); URL.revokeObjectURL(previewUrl.current) }, [episodeId])

  function changeEpisode(value: string) {
    setPreview(''); setPreviewing(false); setEpisodeId(value)
  }

  function pick(line: Span) {
    setSelection(line); setCenter((line.start + line.end) / 2); playhead.seek(line.start)
  }

  async function listen(ranges: VoiceRange[]) {
    request.current?.abort()
    const current = new AbortController()
    request.current = current
    setPreviewing(true); setError('')
    try {
      const response = await fetch(`/api/episodes/${episodeId}/voice-samples/preview`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ranges }), signal: current.signal,
      })
      if (!response.ok) throw new Error((await response.json()).detail || 'Audio preview is unavailable.')
      const blob = await response.blob()
      if (current.signal.aborted) return
      URL.revokeObjectURL(previewUrl.current)
      previewUrl.current = URL.createObjectURL(blob)
      setPreview(previewUrl.current)
    } catch (reason) { if (!current.signal.aborted) setError((reason as Error).message) }
    finally { if (!current.signal.aborted) setPreviewing(false) }
  }

  return <div className="space-y-4">
    {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
    {episodes.length > 0 && <label className="block text-xs">Audio episode <select className="field mt-1" value={episodeId} onChange={event => changeEpisode(event.target.value)}>
      {episodes.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>}
    {!episode || !episode.has_audio || duration <= 0 ? <p className="muted text-sm">{episodeId && !episode && !error ? 'Loading episode audio…' : 'Choose an episode with prepared audio to add a sample.'}</p> : <>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <IconButton icon={ChevronLeft} label="Previous waveform window" size="sm" disabled={start <= 0} onClick={() => setCenter(Math.max(span / 2, start - span / 2))} />
        <IconButton icon={ChevronRight} label="Next waveform window" size="sm" disabled={end >= duration} onClick={() => setCenter(Math.min(duration - span / 2, end + span / 2))} />
        <input type="range" aria-label="Waveform position" min={0} max={duration} step={0.1} value={center} onChange={event => setCenter(Number(event.target.value))} className="min-w-24 flex-1 accent-[#bca7ff]" />
        <label>Window <select className="field ml-1 !w-auto !py-1" value={zoom} onChange={event => setZoom(Number(event.target.value))}>
          {[5, 10, 20, 45, 90].map(size => <option key={size} value={size}>{size}s</option>)}</select></label>
      </div>
      <CutWaveform view={view} cut={view} bars={beats} bands={nearLines.map(line => ({ id: line.id, span: { start: line.start_time, end: line.end_time },
        color: '#9a7de1', selected: selection.start === line.start_time && selection.end === line.end_time, label: line.text }))}
        playhead={cursor} handles={{ span: selection, bounds: { start: 0, end: duration }, onCommit: (edge, value) => setSelection(item => ({ ...item, [edge]: value })) }}
        message={beats.length ? undefined : 'Waveform unavailable or loading…'} onSeek={value => playhead.seek(value)} onSelectBand={id => {
          const line = nearLines.find(item => item.id === id)
          if (line) pick({ start: line.start_time, end: line.end_time })
        }} />
      {lines.length > 0 && <label className="block text-xs">Jump to transcript line <select className="field mt-1" value="" onChange={event => {
        const line = lines.find(item => item.id === event.target.value)
        if (line) pick({ start: line.start_time, end: line.end_time })
      }}><option value="">Select a line…</option>{lines.map(line => <option key={line.id} value={line.id}>{stamp(line.start_time)} · {line.text.slice(0, 90)}</option>)}</select></label>}
      <TimeRangeInput compact start={selection.start} end={selection.end} max={duration} playhead={playhead}
        onChange={range => setSelection(value => ({ start: range.start ?? value.start, end: range.end ?? value.end }))} />
      <div className="flex flex-wrap gap-2">
        <IconButton icon={Play} label="Preview selection" text="Preview selection" disabled={!selectable || selection.end - selection.start > 60 || previewing}
          onClick={() => void listen([{ start_time: selection.start, end_time: selection.end }])} />
        <IconButton icon={Plus} label="Add excerpt" text="Add excerpt" disabled={!canAdd}
          onClick={() => onDraft({ ...draft, [episodeId]: [...pending, { start_time: selection.start, end_time: selection.end }].sort((a, b) => a.start_time - b.start_time) })} />
      </div>
      {pending.length > 0 && <section className="space-y-1 text-xs">
        <div className="flex items-center justify-between"><strong>New excerpts · {bandDuration.toFixed(1)} s</strong><IconButton icon={Play} label="Preview all excerpts" text="Preview all" size="sm" disabled={previewing} onClick={() => void listen(pending)} /></div>
        {bandDuration < 2 && <p className="text-[#f5c678]">Add at least 2 seconds of audio for this sample.</p>}
        {pending.map((range, index) => <div className="flex items-center gap-2" key={range.start_time}>
          <IconButton icon={Play} label={`Preview excerpt ${index + 1}`} size="sm" disabled={previewing} onClick={() => void listen([range])} />
          <button type="button" className="text-[#cbb9ff]" onClick={() => pick({ start: range.start_time, end: range.end_time })}>{stamp(range.start_time)}–{stamp(range.end_time)}</button>
          <IconButton icon={Trash2} label={`Remove excerpt ${index + 1}`} size="sm" onClick={() => onDraft({ ...draft, [episodeId]: pending.filter(item => item !== range) })} />
        </div>)}
      </section>}
      {preview && <audio aria-label="Excerpt playback" className="w-full" src={preview} controls autoPlay />}
    </>}
    {samples.length > 0 && <section className="space-y-2">
      <h3 className="text-sm font-semibold">Saved voice samples</h3>
      {samples.map(sample => <div className="rounded-lg border hairline p-2 text-xs" key={sample.id}>
        <div className="flex items-center justify-between gap-2"><span>{sample.source_filename || sample.episode_title || 'Voice reference'}{sample.ranges.length > 0 && ` · ${sample.ranges.map(range => stamp(range.start_time)).join(', ')}`}</span>
          <button type="button" className="btn btn-quiet !py-1 !text-xs" onClick={() => onRemove(removed.includes(sample.id) ? removed.filter(id => id !== sample.id) : [...removed, sample.id])}>
            {removed.includes(sample.id) ? 'Undo removal' : 'Remove on save'}</button></div>
        {sample.preview_available && !removed.includes(sample.id) && <audio aria-label="Saved voice sample" controls preload="none" className="mt-1 h-8 w-full" src={`/api/voice-samples/${sample.id}/audio`} />}
        {removed.includes(sample.id) && <span className="text-[#ff9fa7]">Marked for removal</span>}
      </div>)}
    </section>}
  </div>
}
