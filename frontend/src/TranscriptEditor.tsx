import { useEpisodeEvents } from './episodeEvents'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { AudioLines, Captions, History, ListMusic, Mic, Play, Plus, RefreshCw, Save, Trash2, Wrench, X } from 'lucide-react'
import { api, type AudioSelection, type ContextItem, type Job, type SubtitleRun, type TranscriptSegment } from './api'
import { IconButton, Section, SegmentedControl } from './EditorUi'
import { PANEL_ACCENT } from './InspectorRail'
import SubtitlePanel from './SubtitlePanel'
import { TimeRangeInput } from './TimecodeInput'
import TranscribeRangePanel from './TranscribeRangePanel'
import { formatDuration, formatTimecode } from './timecode'

export type AudioView = 'lines' | 'tools' | 'voices'

type Draft = Pick<TranscriptSegment, 'start_time' | 'end_time' | 'text' | 'speaker_label' | 'character_context_id'> & { id?: string; deleted?: boolean; superseded_by?: string | null }
type VoiceRange = { start_time: number; end_time: number }
type VoiceSample = { id: string; character_context_id: string; episode_title: string | null; ranges: VoiceRange[]; source_filename: string | null; preview_available: boolean }

const PAGE_SIZE = 50
const accent = PANEL_ACCENT.audio

function ExcerptPreview({ episodeId, ranges }: { episodeId: string; ranges: VoiceRange[] }) {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const objectUrl = useRef('')
  useEffect(() => () => { controller.current?.abort(); if (objectUrl.current) URL.revokeObjectURL(objectUrl.current) }, [])
  async function prepare() {
    setLoading(true); setError('')
    const request = new AbortController()
    controller.current = request
    try {
      const response = await fetch(`/api/episodes/${episodeId}/voice-samples/preview`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ranges }), signal: request.signal })
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail.detail || 'Could not prepare the excerpt preview')
      }
      const blob = await response.blob()
      if (request.signal.aborted) return
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = URL.createObjectURL(blob)
      setUrl(objectUrl.current)
    } catch (err) { if (!request.signal.aborted) setError((err as Error).message) }
    finally { if (!request.signal.aborted) setLoading(false) }
  }
  return <div className="space-y-2"><button type="button" className="btn btn-quiet w-full !py-1.5 !text-xs" disabled={loading} onClick={prepare}><ListMusic size={14} /> {loading ? 'Preparing preview…' : 'Preview all excerpts together'}</button>
    {url && <audio className="w-full" controls autoPlay src={url} aria-label="Combined excerpt preview" />}
    {error && <p className="text-xs text-[#ff9fa7]">{error}</p>}
  </div>
}

function UploadedPreview({ file }: { file: File }) {
  const [url, setUrl] = useState('')
  useEffect(() => { const next = URL.createObjectURL(file); setUrl(next); return () => URL.revokeObjectURL(next) }, [file])
  return url ? <audio className="w-full" controls src={url} aria-label="Uploaded voice preview" /> : null
}

/** One transcript line opened for editing, right where it sits in the list. */
function SegmentEditor({ draft, setDraft, characters, duration, currentTime, busy, onSave, onDelete, onCancel, playRange }: {
  draft: Draft; setDraft: (draft: Draft) => void; characters: ContextItem[]; duration: number; currentTime: number; busy: boolean
  onSave: () => void; onDelete: () => void; onCancel: () => void; playRange: (start: number, end: number) => void
}) {
  const historical = !!draft.superseded_by || !!draft.deleted
  const valid = !!draft.text.trim() && draft.end_time > draft.start_time
  const onKey = (event: KeyboardEvent<HTMLDivElement>) => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); if (valid && !historical) onSave() } }
  return <div onKeyDown={onKey} className="space-y-3 rounded-b-lg border border-t-0 border-[#7a4a57] bg-[#1d171c] p-3">
    {historical && <p className="text-[11px] text-[#f2c879]">An earlier revision; it cannot be edited.</p>}
    <TimeRangeInput compact start={draft.start_time} end={draft.end_time} min={0} max={duration} playhead={currentTime} lockStart={historical} lockEnd={historical}
      onChange={range => setDraft({ ...draft, start_time: range.start ?? draft.start_time, end_time: range.end ?? draft.end_time })} />
    <label className="block"><span className="label !mb-1 !text-[10px]">Speech or audio note</span><textarea className="field min-h-20 resize-y !py-2 text-sm" maxLength={20000} disabled={historical} value={draft.text} onChange={event => setDraft({ ...draft, text: event.target.value })} placeholder="What is heard?" autoFocus={!draft.id} /></label>
    <div className="grid grid-cols-2 gap-2">
      <label className="block"><span className="label !mb-1 !text-[10px]">Speaker label</span><input className="field !py-1.5 text-sm" maxLength={200} disabled={historical} value={draft.speaker_label} onChange={event => setDraft({ ...draft, speaker_label: event.target.value })} placeholder="e.g. Narrator" /></label>
      <label className="block"><span className="label !mb-1 !text-[10px]">Character</span><select className="field !py-1.5 text-sm" disabled={historical} value={draft.character_context_id || ''} onChange={event => setDraft({ ...draft, character_context_id: event.target.value || null })}>
        <option value="">Unassigned</option>{characters.map(character => <option key={character.id} value={character.id}>{character.title}</option>)}
      </select></label>
    </div>
    <div className="flex items-center gap-1.5">
      <IconButton icon={Play} label="Play this line" size="sm" disabled={draft.end_time <= draft.start_time} onClick={() => playRange(draft.start_time, draft.end_time)} />
      {draft.id && !historical && <IconButton icon={Trash2} label="Delete this line" tone="danger" size="sm" disabled={busy} onClick={onDelete} />}
      <span className="flex-1" />
      <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" onClick={onCancel}>Close</button>
      {!historical && <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={busy || !valid} onClick={onSave} title="Save (Ctrl+S)"><Save size={13} /> Save</button>}
    </div>
  </div>
}

export default function TranscriptEditor({ episodeId, projectId, duration, currentTime, segments, selectedId, onSelect, onSaved, playRange, hasAudio, transcriptStale, attributionStale, subtitleLanguage, subtitleRun, jobs, audioSelection, onAudioSelection, view, onViewChange }: {
  episodeId: string
  projectId: string
  duration: number
  currentTime: number
  segments: TranscriptSegment[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onSaved: () => Promise<void>
  playRange: (start: number, end: number) => void
  hasAudio: boolean
  transcriptStale: string | null
  attributionStale: string | null
  subtitleLanguage: string | null
  subtitleRun: SubtitleRun | null
  jobs: Job[]
  audioSelection: AudioSelection | null
  onAudioSelection: (selection: AudioSelection | null) => void
  view: AudioView
  onViewChange: (view: AudioView) => void
}) {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [history, setHistory] = useState(false)
  const [checkOnly, setCheckOnly] = useState(false)
  const [page, setPage] = useState(0)
  const [historySegments, setHistorySegments] = useState<TranscriptSegment[]>([])
  const [busy, setBusy] = useState(false)
  const preparing = jobs.some(job => job.pipeline === 'prepare' && (job.status === 'queued' || job.status === 'running'))
  const [error, setError] = useState('')
  const [voiceSamples, setVoiceSamples] = useState<VoiceSample[]>([])
  const [voiceCharacter, setVoiceCharacter] = useState('')
  const [voiceRange, setVoiceRange] = useState<{ start: number | null; end: number | null }>({ start: null, end: null })
  const [voiceRanges, setVoiceRanges] = useState<VoiceRange[]>([])
  const [voiceFile, setVoiceFile] = useState<File | null>(null)
  const [voiceFileKey, setVoiceFileKey] = useState(0)
  useEffect(() => { setVoiceRanges([]); setVoiceFile(null) }, [episodeId])
  async function loadVoices() { setVoiceSamples(await api<VoiceSample[]>('/projects/' + projectId + '/voice-samples')) }
  useEffect(() => {
    api<ContextItem[]>('/projects/' + projectId + '/context')
      .then(items => setCharacters(items.filter(item => item.kind === 'character')))
      .catch(err => setError((err as Error).message))
  }, [projectId])
  useEffect(() => { loadVoices().catch(err => setError((err as Error).message)) }, [projectId])
  useEpisodeEvents(change => { if (change.data && view === 'voices') void loadVoices().catch(err => setError((err as Error).message)) }, episodeId)
  useEffect(() => {
    const selected = segments.find(segment => segment.id === selectedId)
    if (selected) setDraft(current => current?.id === selected.id ? current : selected)
  }, [segments, selectedId])
  async function loadHistory() {
    try { setHistorySegments(await api<TranscriptSegment[]>('/episodes/' + episodeId + '/transcript?history=true')) }
    catch (err) { setError((err as Error).message) }
  }
  async function toggleHistory() {
    if (history) { setHistory(false); return }
    setHistory(true); await loadHistory()
  }
  function addAtPlayhead() {
    const start = Math.max(0, Math.min(duration - 0.1, Math.round(currentTime * 10) / 10))
    setDraft({ start_time: start, end_time: Math.min(duration, start + 5), text: '', speaker_label: '', character_context_id: null })
    onSelect(null); setHistory(false); onViewChange('lines')
  }
  function close() { setDraft(null); onSelect(null) }
  async function save() {
    if (!draft || !draft.text.trim() || draft.end_time <= draft.start_time) return
    setBusy(true); setError('')
    try {
      const body = { start_time: draft.start_time, end_time: draft.end_time, text: draft.text, speaker_label: draft.speaker_label, character_context_id: draft.character_context_id }
      const value = draft.id
        ? await api<TranscriptSegment>('/transcript-segments/' + draft.id, { method: 'PUT', body: JSON.stringify(body) })
        : await api<TranscriptSegment>('/episodes/' + episodeId + '/transcript', { method: 'POST', body: JSON.stringify(body) })
      onSelect(value.id); setDraft(value); await onSaved()
      if (history) await loadHistory()
    } catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function remove() {
    if (!draft?.id || !window.confirm('Delete this transcript line? It stays in the history.')) return
    setBusy(true); setError('')
    try {
      await api('/transcript-segments/' + draft.id, { method: 'DELETE' })
      onSelect(null); setDraft(null); await onSaved()
      if (history) await loadHistory()
    } catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function queueAudio(path: string, body?: object) {
    setBusy(true); setError('')
    try { await api('/episodes/' + episodeId + path, { method: 'POST', ...(body ? { body: JSON.stringify(body) } : {}) }); await onSaved() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function removeVoice(id: string) {
    setBusy(true); setError('')
    try { await api('/voice-samples/' + id, { method: 'DELETE' }); await loadVoices() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function saveVoiceRanges() {
    setBusy(true); setError('')
    try {
      await api('/episodes/' + episodeId + '/voice-samples', { method: 'POST', body: JSON.stringify({ character_context_id: voiceCharacter, ranges: voiceRanges }) })
      setVoiceRanges([]); await onSaved()
    } catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function uploadVoice() {
    if (!voiceFile) return
    setBusy(true); setError('')
    const form = new FormData()
    form.append('character_context_id', voiceCharacter); form.append('file', voiceFile)
    try { await api('/episodes/' + episodeId + '/voice-samples/upload', { method: 'POST', body: form }); setVoiceFile(null); setVoiceFileKey(key => key + 1); await onSaved() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  const needsCheck = (segment: TranscriptSegment) => segment.confidence === 'medium' || segment.confidence === 'low'
  const checkCount = segments.filter(needsCheck).length
  const visibleSegments = history ? historySegments : checkOnly ? segments.filter(needsCheck) : segments
  const pageCount = Math.max(1, Math.ceil(visibleSegments.length / PAGE_SIZE))
  const currentPage = Math.min(page, pageCount - 1)
  useEffect(() => {
    const index = visibleSegments.findIndex(segment => segment.id === selectedId)
    if (index >= 0) setPage(Math.floor(index / PAGE_SIZE))
  }, [selectedId, history, checkOnly])
  useEffect(() => { if (view === 'lines' && selectedId) document.getElementById(`transcript-row-${selectedId}`)?.scrollIntoView({ block: 'nearest' }) }, [selectedId, view, currentPage])
  const voiceStart = voiceRange.start, voiceEnd = voiceRange.end
  const voiceRangeValid = voiceStart !== null && voiceEnd !== null && voiceEnd > voiceStart && voiceStart >= 0 && voiceEnd <= duration && !voiceRanges.some(range => voiceStart < range.end_time && voiceEnd > range.start_time)
  const totalVoiceDuration = voiceRanges.reduce((sum, range) => sum + range.end_time - range.start_time, 0)
  const transcribing = jobs.find(job => job.pipeline === 'transcribe' && ['queued', 'running'].includes(job.status))
  const attributing = jobs.some(job => job.pipeline === 'speaker_attribution' && ['queued', 'running'].includes(job.status))
  const subtitling = jobs.some(job => job.pipeline === 'subtitles' && ['queued', 'running'].includes(job.status))
  const characterName = (id: string | null) => characters.find(character => character.id === id)?.title
  return <div className="flex h-full min-h-0 flex-col">
    <div className="border-b hairline px-4 pb-3 pt-3">
      <p className="text-xs font-semibold uppercase tracking-[.14em]" style={{ color: accent }}>Audio</p>
      <SegmentedControl className="mt-2 w-full" label="Audio views" value={view} onChange={onViewChange} options={[
        { value: 'lines', label: 'Lines', icon: AudioLines, badge: <span className="muted font-mono text-[10px]">{segments.length}</span> },
        { value: 'tools', label: 'Tools', icon: Wrench, badge: transcribing || subtitling || attributing ? <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#f5c678]" aria-label="Working" /> : undefined },
        { value: 'voices', label: 'Voices', icon: Mic },
      ]} />
    </div>
    {error && <p role="alert" className="mx-4 mt-3 flex items-start justify-between gap-2 text-xs text-[#ff9fa7]">{error}<button onClick={() => setError('')} aria-label="Dismiss"><X size={13} /></button></p>}
    {(transcriptStale || attributionStale) && <div className="space-y-1 border-b hairline px-4 py-2 text-xs text-[#f5c678]">
      {transcriptStale && <div className="flex flex-wrap items-center justify-between gap-1"><span>{transcriptStale}</span><button type="button" className="btn btn-quiet !py-1 !text-xs" disabled={busy || !!transcribing} onClick={() => void queueAudio('/transcribe')}>Transcribe again</button></div>}
      {attributionStale && <div className="flex flex-wrap items-center justify-between gap-1"><span>{attributionStale}</span><button type="button" className="btn btn-quiet !py-1 !text-xs" disabled={busy || attributing} onClick={() => void queueAudio('/speaker-attribution')}>Attribute speakers again</button></div>}
    </div>}
    <div className="scrollbar min-h-0 flex-1 overflow-y-auto">
      {view === 'lines' && <div className="p-4">
        <div className="mb-3 flex items-center justify-between gap-2">
          <p className="muted text-[11px]">{history ? 'All revisions, newest state last' : `${segments.length} current lines · used by cut analysis`}</p>
          <div className="flex gap-1">
            <IconButton icon={History} label={history ? 'Hide earlier revisions' : 'Show earlier revisions'} size="sm" aria-pressed={history} onClick={() => void toggleHistory()} />
            <IconButton icon={Plus} label="Add a line at the playhead" text="Add" size="sm" disabled={!duration || busy} onClick={addAtPlayhead} />
          </div>
        </div>
        {!history && <label className="mb-3 flex items-center gap-2 text-xs">
          <input type="checkbox" className="accent-[#d18c9e]" checked={checkOnly} onChange={event => { setCheckOnly(event.target.checked); setPage(0); close() }} />
          Lines to check <span className="muted tabular-nums">{checkCount}</span>
        </label>}
        {draft && !draft.id && <div className="mb-2"><div className="rounded-t-lg border border-[#7a4a57] bg-[#2a1d23] px-3 py-2 text-xs font-semibold">New line</div>
          <SegmentEditor draft={draft} setDraft={setDraft} characters={characters} duration={duration} currentTime={currentTime} busy={busy} onSave={() => void save()} onDelete={() => void remove()} onCancel={close} playRange={playRange} /></div>}
        <div className="space-y-1.5">
          {visibleSegments.length ? visibleSegments.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE).map(segment => {
            const open = draft?.id === segment.id && selectedId === segment.id
            return <div key={segment.id} id={`transcript-row-${segment.id}`}>
              <button aria-expanded={open} onClick={() => { if (open) close(); else { onSelect(segment.id); setDraft(segment) } }}
                className={`w-full border p-2.5 text-left ${open ? 'rounded-t-lg border-[#7a4a57] bg-[#2a1d23]' : selectedId === segment.id ? 'rounded-lg border-[#7a4a57] bg-[#241a1f]' : 'rounded-lg border-[#2c323d] bg-[#181d26] hover:border-[#56464d]'}`}>
                <span className="flex justify-between gap-2 font-mono text-[10px] tabular-nums text-[#c7aab2]"><span>{formatTimecode(segment.start_time, { precision: 1 })} – {formatTimecode(segment.end_time, { precision: 1 })}</span><span className="font-sans">{segment.deleted ? 'Deleted' : segment.superseded_by ? 'Earlier revision' : formatDuration(segment.end_time - segment.start_time)}</span></span>
                <span className="mt-0.5 line-clamp-2 text-xs"><strong className="font-semibold" style={{ color: accent }}>{characterName(segment.character_context_id) || segment.speaker_label || 'Speaker'}</strong>
                  {needsCheck(segment) && <span className="mx-1 rounded border border-[#735a35] bg-[#342818] px-1 py-0.5 text-[9px] text-[#f5c678]" title={`Speaker confidence: ${segment.confidence}`}>check</span>} {segment.text}</span>
              </button>
              {open && draft && <SegmentEditor draft={draft} setDraft={setDraft} characters={characters} duration={duration} currentTime={currentTime} busy={busy} onSave={() => void save()} onDelete={() => void remove()} onCancel={close} playRange={playRange} />}
            </div>
          }) : <div className="muted rounded-lg border border-dashed hairline p-5 text-center text-xs">{checkOnly && !history && segments.length ? 'No lines with medium or low speaker confidence.' : 'No transcript lines yet. Transcribe the episode under Tools.'}</div>}
        </div>
        {pageCount > 1 && <div className="mt-3 flex items-center justify-between text-xs"><button disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)} className="btn btn-quiet !px-2.5 !py-1">Previous</button><span className="muted">Page {currentPage + 1} of {pageCount}</span><button disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)} className="btn btn-quiet !px-2.5 !py-1">Next</button></div>}
      </div>}
      {view === 'tools' && <>
        <Section id="audio.transcribe" title="Transcribe whole episode" icon={AudioLines} accent={accent} badge={transcribing ? <span className="ml-1 text-[10px] normal-case tracking-normal text-[#f5c678]">{transcribing.status === 'queued' ? 'queued' : 'running'}</span> : null}>
          <p className="muted text-[11px] leading-4">Transcribes speech outside the excluded audio ranges and identifies speakers. Hand-edited lines are kept.</p>
          <button disabled={busy || (!hasAudio && !preparing)} onClick={() => queueAudio('/transcribe')} className="btn btn-primary mt-3 w-full !py-2"><AudioLines size={14} /> Transcribe whole episode</button>
          {!hasAudio && !preparing && <p className="muted mt-2 text-[11px]">{duration > 0 ? 'This video has no audio track.' : 'Prepare the video first.'}</p>}
        </Section>
        <Section id="audio.redo" title="Redo part of the transcript" icon={RefreshCw} accent={accent} defaultOpen={false} openSignal={audioSelection ? `${audioSelection.start}-${audioSelection.end}` : null}>
          <TranscribeRangePanel episodeId={episodeId} duration={duration} currentTime={currentTime} segments={segments} selection={audioSelection} onSelectionChange={onAudioSelection} playRange={playRange} hasAudio={hasAudio} jobs={jobs} onQueued={onSaved} />
        </Section>
        <Section id="audio.subtitles" title="Burned-in subtitles" icon={Captions} accent={accent} defaultOpen={false} badge={subtitling ? <span className="ml-1 text-[10px] normal-case tracking-normal text-[#f5c678]">running</span> : null}>
          <SubtitlePanel episodeId={episodeId} ready={duration > 0 || preparing} savedLanguage={subtitleLanguage} run={subtitleRun} jobs={jobs} onChanged={onSaved} />
        </Section>
      </>}
      {view === 'voices' && <div className="space-y-4 p-4">
        <p className="muted text-[11px] leading-4">A voice reference helps match speakers to characters across this project. Add excerpts of one speaker totalling 2–60 seconds, or upload a recording.</p>
        <label className="block"><span className="label">Character</span><select className="field !py-2" value={voiceCharacter} onChange={event => setVoiceCharacter(event.target.value)}><option value="">Choose character</option>{characters.map(character => <option key={character.id} value={character.id}>{character.title}</option>)}</select></label>
        <div className="rounded-lg border hairline p-3">
          <p className="mb-2 text-xs font-semibold">From this episode</p>
          <TimeRangeInput compact start={voiceStart} end={voiceEnd} min={0} max={duration} playhead={currentTime} onChange={setVoiceRange} onPlay={hasAudio ? playRange : undefined} />
          <button className="btn btn-quiet mt-2 w-full !py-1.5 !text-xs" disabled={!hasAudio || !voiceRangeValid || voiceRanges.length >= 50 || totalVoiceDuration + (voiceEnd ?? 0) - (voiceStart ?? 0) > 60} onClick={() => { if (voiceStart !== null && voiceEnd !== null) { setVoiceRanges([...voiceRanges, { start_time: voiceStart, end_time: voiceEnd }]); setVoiceRange({ start: null, end: null }) } }}><Plus size={13} /> Add excerpt</button>
          {voiceRanges.length > 0 && <div className="mt-3 flex flex-wrap gap-1.5">{voiceRanges.map((range, index) => <span key={index} className="chip font-mono tabular-nums">
            <button onClick={() => playRange(range.start_time, range.end_time)} aria-label={`Play excerpt ${index + 1}`} className="!text-[#e89aab]"><Play size={11} /></button>{formatTimecode(range.start_time, { precision: 1 })}–{formatTimecode(range.end_time, { precision: 1 })}
            <button onClick={() => setVoiceRanges(voiceRanges.filter((_, i) => i !== index))} aria-label={`Remove excerpt ${index + 1}`}><X size={11} /></button></span>)}</div>}
          {voiceRanges.length > 0 && <div className="mt-3 space-y-2"><ExcerptPreview key={episodeId + JSON.stringify(voiceRanges)} episodeId={episodeId} ranges={voiceRanges} />
            <button className="btn btn-primary w-full !py-1.5 !text-xs" disabled={busy || !hasAudio || !voiceCharacter || totalVoiceDuration < 2 || totalVoiceDuration > 60} onClick={saveVoiceRanges} title={!voiceCharacter ? 'Choose a character first' : undefined}><Save size={13} /> Save {voiceRanges.length} excerpt{voiceRanges.length === 1 ? '' : 's'} ({totalVoiceDuration.toFixed(1)}s)</button></div>}
        </div>
        <div className="rounded-lg border hairline p-3">
          <label className="block"><span className="mb-2 block text-xs font-semibold">Or upload a recording <span className="muted font-normal">· 2–60 s, up to 25 MB</span></span><input key={voiceFileKey} type="file" accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm" className="field !py-1.5 text-xs" onChange={event => setVoiceFile(event.target.files?.[0] || null)} /></label>
          {voiceFile && <div className="mt-2"><UploadedPreview file={voiceFile} /></div>}
          <button className="btn btn-quiet mt-2 w-full !py-1.5 !text-xs" disabled={busy || !voiceCharacter || !voiceFile} onClick={uploadVoice}>Upload voice</button>
        </div>
        <div>
          <p className="mb-2 text-xs font-semibold">Saved references <span className="muted font-normal">{voiceSamples.length}</span></p>
          <div className="space-y-2">{voiceSamples.map(sample => <div key={sample.id} className="space-y-2 rounded-lg border hairline p-2 text-[11px]"><div className="flex items-center justify-between gap-2"><span className="min-w-0"><strong>{characterName(sample.character_context_id) || 'Character'}</strong> <span className="muted">· {sample.source_filename ? 'Upload: ' + sample.source_filename : (sample.episode_title || 'Deleted episode') + ' · ' + sample.ranges.map(range => formatTimecode(range.start_time, { precision: 1 }) + '–' + formatTimecode(range.end_time, { precision: 1 })).join(', ')}</span></span><IconButton icon={Trash2} label="Delete voice reference" tone="danger" size="sm" className="ghost" disabled={busy} onClick={() => removeVoice(sample.id)} /></div>{sample.preview_available ? <audio className="h-8 w-full" controls preload="none" src={'/api/voice-samples/' + sample.id + '/audio'} aria-label={'Voice reference for ' + (characterName(sample.character_context_id) || 'character')} /> : <p className="muted">Audio preview unavailable for this older reference.</p>}</div>)}
            {!voiceSamples.length && <p className="muted text-[11px]">No voice references yet.</p>}</div>
        </div>
      </div>}
    </div>
  </div>
}
