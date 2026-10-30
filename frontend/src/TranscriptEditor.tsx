import { useEffect, useState } from 'react'
import { AudioLines, History, Plus, Save, Trash2 } from 'lucide-react'
import { api, type ContextItem, type ResultStatus, type TranscriptSegment } from './api'

type Draft = Pick<TranscriptSegment, 'start_time' | 'end_time' | 'text' | 'speaker_label' | 'character_context_id' | 'status'> & { id?: string }
type VoiceSample = { id: string; character_context_id: string; episode_title: string | null; start_time: number; end_time: number }

export default function TranscriptEditor({ episodeId, projectId, duration, currentTime, segments, selectedId, onSelect, onSaved, playRange, processing, hasAudio }: {
  episodeId: string
  projectId: string
  duration: number
  currentTime: number
  segments: TranscriptSegment[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onSaved: () => Promise<void>
  playRange: (start: number, end: number) => void
  processing: boolean
  hasAudio: boolean
}) {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [history, setHistory] = useState(false)
  const [page, setPage] = useState(0)
  const [historySegments, setHistorySegments] = useState<TranscriptSegment[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [voiceSamples, setVoiceSamples] = useState<VoiceSample[]>([])
  const [voiceCharacter, setVoiceCharacter] = useState('')
  const [voiceStart, setVoiceStart] = useState(0)
  const [voiceEnd, setVoiceEnd] = useState(5)
  async function loadVoices() { setVoiceSamples(await api<VoiceSample[]>('/projects/' + projectId + '/voice-samples')) }
  useEffect(() => {
    api<ContextItem[]>('/projects/' + projectId + '/context')
      .then(items => setCharacters(items.filter(item => item.kind === 'character')))
      .catch(err => setError((err as Error).message))
  }, [projectId])
  useEffect(() => { loadVoices().catch(err => setError((err as Error).message)) }, [projectId, processing])
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
    setDraft({ start_time: start, end_time: Math.min(duration, start + 5), text: '', speaker_label: '', character_context_id: null, status: 'draft' })
    onSelect(null)
  }
  async function save() {
    if (!draft || !draft.text.trim() || draft.end_time <= draft.start_time) return
    setBusy(true); setError('')
    try {
      const value = draft.id
        ? await api<TranscriptSegment>('/transcript-segments/' + draft.id, { method: 'PUT', body: JSON.stringify(draft) })
        : await api<TranscriptSegment>('/episodes/' + episodeId + '/transcript', { method: 'POST', body: JSON.stringify(draft) })
      onSelect(value.id); setDraft(value); await onSaved()
      if (history) await loadHistory()
    } catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  async function remove() {
    if (!draft?.id || !window.confirm('Delete this transcript segment?')) return
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
  async function approveDrafts() {
    setBusy(true); setError('')
    try { await api('/episodes/' + episodeId + '/transcript/approve-drafts', { method: 'POST' }); await onSaved() }
    catch (err) { setError((err as Error).message) }
    finally { setBusy(false) }
  }
  const visibleSegments = history ? historySegments : segments
  const pageSize = 50
  const pageCount = Math.max(1, Math.ceil(visibleSegments.length / pageSize))
  const currentPage = Math.min(page, pageCount - 1)
  useEffect(() => {
    const index = visibleSegments.findIndex(segment => segment.id === selectedId)
    if (index >= 0) setPage(Math.floor(index / pageSize))
  }, [selectedId, history])
  const historicalDraft = !!draft && (!!('superseded_by' in draft && draft.superseded_by) || draft.status === 'deleted')
  return <div className="scrollbar max-h-[calc(100vh-115px)] overflow-y-auto p-5">
    <div className="flex items-center justify-between gap-3">
      <div><p className="text-xs font-semibold uppercase tracking-[.14em] text-[#7bc5cb]">Audio workspace</p><h2 className="mt-1 text-lg font-semibold">Transcript</h2></div>
      <button onClick={addAtPlayhead} disabled={!duration || busy} className="btn btn-quiet px-2.5 py-2"><Plus size={14} /> Add</button>
    </div>
    <p className="muted mt-2 text-xs leading-5">Review speech against playback and adjust its timing independently of cuts. Approved segments are included in cut analysis.</p>
    <button disabled={busy || processing || !hasAudio} onClick={() => queueAudio('/transcribe')} className="btn btn-primary mt-4 w-full"><AudioLines size={14} /> Extract audio &amp; transcribe</button>
    <div className="mt-5 space-y-3 rounded-lg border hairline p-3">
      <h3 className="text-xs font-semibold">Character voice reference</h3>
      <p className="muted text-[11px]">Select a clear 2–60 second excerpt with one speaker. References can be used across this project.</p>
      <select className="field" value={voiceCharacter} onChange={event => setVoiceCharacter(event.target.value)}><option value="">Choose character</option>{characters.map(character => <option key={character.id} value={character.id}>{character.title}</option>)}</select>
      <div className="grid grid-cols-2 gap-2"><label><span className="label">Start · seconds</span><input type="number" className="field" step="0.01" min="0" max={duration} value={voiceStart} onChange={event => setVoiceStart(Number(event.target.value))} /><button className="muted mt-1 text-[11px]" onClick={() => setVoiceStart(Math.round(currentTime * 100) / 100)}>Set to playhead</button></label><label><span className="label">End · seconds</span><input type="number" className="field" step="0.01" min="0" max={duration} value={voiceEnd} onChange={event => setVoiceEnd(Number(event.target.value))} /><button className="muted mt-1 text-[11px]" onClick={() => setVoiceEnd(Math.round(currentTime * 100) / 100)}>Set to playhead</button></label></div>
      <div className="flex gap-2"><button className="btn btn-quiet flex-1" disabled={voiceEnd <= voiceStart} onClick={() => playRange(voiceStart, voiceEnd)}>Play excerpt</button><button className="btn btn-quiet flex-1" disabled={busy || processing || !hasAudio || !voiceCharacter || voiceEnd - voiceStart < 2 || voiceEnd - voiceStart > 60 || voiceEnd > duration} onClick={() => queueAudio('/voice-samples', { character_context_id: voiceCharacter, start_time: voiceStart, end_time: voiceEnd })}>Save voice</button></div>
      {voiceSamples.map(sample => <div key={sample.id} className="flex items-center justify-between gap-2 text-[11px]"><span>{characters.find(c => c.id === sample.character_context_id)?.title || 'Character'} · {sample.episode_title || 'Deleted episode'} · {sample.start_time.toFixed(1)}–{sample.end_time.toFixed(1)}s</span><button disabled={busy} onClick={() => removeVoice(sample.id)} aria-label="Delete voice reference"><Trash2 size={13} /></button></div>)}
    </div>
    {error && <p className="mt-4 text-xs text-[#ff9fa7]">{error}</p>}
    {segments.some(segment => segment.status === 'draft') && <button disabled={busy || processing} onClick={approveDrafts} className="btn btn-quiet mt-4 w-full">Approve all draft speech</button>}
    <div className="mt-5 space-y-2">
      {visibleSegments.length ? visibleSegments.slice(currentPage * pageSize, (currentPage + 1) * pageSize).map(segment => <button key={segment.id} onClick={() => { onSelect(segment.id); setDraft(segment) }}
        className={'w-full rounded-lg border p-3 text-left ' + (selectedId === segment.id ? 'border-[#7bc5cb] bg-[#20363c]' : 'border-[#343b47] bg-[#1b222c] hover:border-[#56777e]')}>
        <span className="flex justify-between gap-2 text-[10px] text-[#a9bbc0]"><span>{segment.start_time.toFixed(1)}s – {segment.end_time.toFixed(1)}s</span><span>{segment.status}</span></span>
        <strong className="mt-1 block truncate text-xs">{segment.speaker_label || 'Speaker'}: {segment.text}</strong>{segment.superseded_by && <span className="muted block text-[10px]">Earlier revision</span>}
      </button>) : <div className="muted rounded-lg border border-dashed hairline p-5 text-center text-xs">No transcript segments yet.</div>}
    </div>
    {pageCount > 1 && <div className="mt-3 flex items-center justify-between text-xs"><button disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)} className="btn btn-quiet">Previous</button><span className="muted">Page {currentPage + 1} of {pageCount}</span><button disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)} className="btn btn-quiet">Next</button></div>}
    <button onClick={toggleHistory} className="muted mt-3 flex items-center gap-1 text-[11px] hover:text-white"><History size={12} /> {history ? 'Hide history' : 'Show history'}</button>
    {draft && <div className="mt-6 space-y-4 border-t hairline pt-5">
      <h3 className="text-sm font-semibold">{historicalDraft ? 'Earlier speech revision' : draft.id ? 'Edit speech segment' : 'New speech segment'}</h3>
      <div className="grid grid-cols-2 gap-3">
        <label><span className="label">Start · seconds</span><input className="field" type="number" min="0" max={duration} step="0.01" disabled={historicalDraft} value={draft.start_time} onChange={event => setDraft({ ...draft, start_time: Number(event.target.value) })} /><button type="button" disabled={historicalDraft} onClick={() => setDraft({ ...draft, start_time: Math.round(currentTime * 100) / 100 })} className="muted mt-1 text-[11px] hover:text-white">Set to playhead</button></label>
        <label><span className="label">End · seconds</span><input className="field" type="number" min="0" max={duration} step="0.01" disabled={historicalDraft} value={draft.end_time} onChange={event => setDraft({ ...draft, end_time: Number(event.target.value) })} /><button type="button" disabled={historicalDraft} onClick={() => setDraft({ ...draft, end_time: Math.round(currentTime * 100) / 100 })} className="muted mt-1 text-[11px] hover:text-white">Set to playhead</button></label>
      </div>
      <label className="block"><span className="label">Speech or audio note</span><textarea className="field min-h-24 resize-y" maxLength={20000} disabled={historicalDraft} value={draft.text} onChange={event => setDraft({ ...draft, text: event.target.value })} placeholder="What is heard?" /></label>
      <label className="block"><span className="label">Speaker label</span><input className="field" maxLength={200} disabled={historicalDraft} value={draft.speaker_label} onChange={event => setDraft({ ...draft, speaker_label: event.target.value })} placeholder="e.g. Narrator, Voice A" /></label>
      <label className="block"><span className="label">Linked character</span><select className="field" disabled={historicalDraft} value={draft.character_context_id || ''} onChange={event => setDraft({ ...draft, character_context_id: event.target.value || null })}>
        <option value="">Unassigned</option>{characters.map(character => <option key={character.id} value={character.id}>{character.title}</option>)}
      </select></label>
      <label className="block"><span className="label">Status</span><select className="field" disabled={historicalDraft} value={draft.status} onChange={event => setDraft({ ...draft, status: event.target.value as ResultStatus })}>
        <option value="draft">Draft</option><option value="approved">Approved</option><option value="rejected">Rejected</option><option value="deleted" disabled>Deleted</option>
      </select></label>
      {draft.end_time <= draft.start_time && <p className="text-xs text-[#ff9fa7]">End must be after start.</p>}
      <div className="flex gap-2"><button onClick={() => playRange(draft.start_time, draft.end_time)} disabled={draft.end_time <= draft.start_time} className="btn btn-quiet flex-1"><AudioLines size={14} /> Play</button>
        <button onClick={save} disabled={busy || historicalDraft || !draft.text.trim() || draft.end_time <= draft.start_time} className="btn btn-primary flex-1"><Save size={14} /> Save</button></div>
      {draft.id && !historicalDraft && <button onClick={remove} disabled={busy} className="text-xs text-[#e9919a]"><Trash2 size={13} className="mr-1 inline" /> Delete segment</button>}
    </div>}
  </div>
}
