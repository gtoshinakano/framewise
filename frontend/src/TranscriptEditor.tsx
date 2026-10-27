import { useEffect, useState } from 'react'
import { AudioLines, History, Plus, Save, Trash2 } from 'lucide-react'
import { api, type ContextItem, type ResultStatus, type TranscriptSegment } from './api'

type Draft = Pick<TranscriptSegment, 'start_time' | 'end_time' | 'text' | 'speaker_label' | 'character_context_id' | 'status'> & { id?: string }

export default function TranscriptEditor({ episodeId, projectId, duration, currentTime, segments, selectedId, onSelect, onSaved, playRange }: {
  episodeId: string
  projectId: string
  duration: number
  currentTime: number
  segments: TranscriptSegment[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onSaved: () => Promise<void>
  playRange: (start: number, end: number) => void
}) {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [history, setHistory] = useState(false)
  const [historySegments, setHistorySegments] = useState<TranscriptSegment[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    api<ContextItem[]>('/projects/' + projectId + '/context')
      .then(items => setCharacters(items.filter(item => item.kind === 'character')))
      .catch(err => setError((err as Error).message))
  }, [projectId])
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
  const visibleSegments = history ? historySegments : segments
  const historicalDraft = !!draft && (!!('superseded_by' in draft && draft.superseded_by) || draft.status === 'deleted')
  return <div className="scrollbar max-h-[calc(100vh-115px)] overflow-y-auto p-5">
    <div className="flex items-center justify-between gap-3">
      <div><p className="text-xs font-semibold uppercase tracking-[.14em] text-[#7bc5cb]">Audio workspace</p><h2 className="mt-1 text-lg font-semibold">Transcript</h2></div>
      <button onClick={addAtPlayhead} disabled={!duration || busy} className="btn btn-quiet px-2.5 py-2"><Plus size={14} /> Add</button>
    </div>
    <p className="muted mt-2 text-xs leading-5">Add timed speech notes now. A later transcription pipeline can populate this lane automatically.</p>
    {error && <p className="mt-4 text-xs text-[#ff9fa7]">{error}</p>}
    <div className="mt-5 space-y-2">
      {visibleSegments.length ? visibleSegments.map(segment => <button key={segment.id} onClick={() => { onSelect(segment.id); setDraft(segment) }}
        className={'w-full rounded-lg border p-3 text-left ' + (selectedId === segment.id ? 'border-[#7bc5cb] bg-[#20363c]' : 'border-[#343b47] bg-[#1b222c] hover:border-[#56777e]')}>
        <span className="flex justify-between gap-2 text-[10px] text-[#a9bbc0]"><span>{segment.start_time.toFixed(1)}s – {segment.end_time.toFixed(1)}s</span><span>{segment.status}</span></span>
        <strong className="mt-1 block truncate text-xs">{segment.speaker_label || 'Speaker'}: {segment.text}</strong>{segment.superseded_by && <span className="muted block text-[10px]">Earlier revision</span>}
      </button>) : <div className="muted rounded-lg border border-dashed hairline p-5 text-center text-xs">No transcript segments yet.</div>}
    </div>
    <button onClick={toggleHistory} className="muted mt-3 flex items-center gap-1 text-[11px] hover:text-white"><History size={12} /> {history ? 'Hide history' : 'Show history'}</button>
    {draft && <div className="mt-6 space-y-4 border-t hairline pt-5">
      <h3 className="text-sm font-semibold">{historicalDraft ? 'Earlier speech revision' : draft.id ? 'Edit speech segment' : 'New speech segment'}</h3>
      <div className="grid grid-cols-2 gap-3">
        <label><span className="label">Start · seconds</span><input className="field" type="number" min="0" max={duration} step="0.1" disabled={historicalDraft} value={draft.start_time} onChange={event => setDraft({ ...draft, start_time: Number(event.target.value) })} /></label>
        <label><span className="label">End · seconds</span><input className="field" type="number" min="0" max={duration} step="0.1" disabled={historicalDraft} value={draft.end_time} onChange={event => setDraft({ ...draft, end_time: Number(event.target.value) })} /></label>
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
