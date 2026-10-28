import { useCallback, useEffect, useState } from 'react'
import { History, Layers3, Plus, Save, WandSparkles, X } from 'lucide-react'
import { api, type AnalysisContext, type AnalysisResult, type ResultKind, type ResultStatus } from './api'

const kinds: { value: ResultKind; label: string }[] = [
  { value: 'first_frame', label: 'First frame' },
  { value: 'camera_movement', label: 'Camera movement' },
  { value: 'composition', label: 'Composition' },
  { value: 'last_frame', label: 'Last frame' },
  { value: 'cut_description', label: 'Cut description' },
  { value: 'audio_visual_match', label: 'Audio and video match' },
  { value: 'generation_prompt', label: 'Generation prompt' },
  { value: 'other', label: 'Other' },
]

type Draft = Pick<AnalysisResult, 'kind' | 'title' | 'content' | 'status'> & { id?: string; superseded_by?: string | null; origin?: string; model_id?: string | null }

export default function ResultEditor({ cutId, refreshKey }: { cutId: string; refreshKey: string }) {
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [history, setHistory] = useState(false)
  const [contextOpen, setContextOpen] = useState(false)
  const [context, setContext] = useState<AnalysisContext | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = useCallback(async (showHistory: boolean) => {
    try { setResults(await api<AnalysisResult[]>(`/cuts/${cutId}/analysis-results?include_context=false${showHistory ? '&history=true' : ''}`)); setError('') }
    catch (err) { setError((err as Error).message) }
  }, [cutId])
  useEffect(() => { setDraft(null); setHistory(false); setContextOpen(false); setContext(null); void load(false) }, [cutId, load])
  useEffect(() => { void load(history) }, [refreshKey])
  async function toggleHistory() { const next = !history; setHistory(next); await load(next) }
  async function toggleContext() {
    if (contextOpen) { setContextOpen(false); return }
    setContextOpen(true)
    try { setContext(await api<AnalysisContext>(`/cuts/${cutId}/analysis-context`)) }
    catch (err) { setError((err as Error).message) }
  }
  async function save() {
    if (!draft?.content.trim()) return
    setBusy(true)
    try {
      const value = draft.id
        ? await api<AnalysisResult>(`/analysis-results/${draft.id}`, { method: 'PUT', body: JSON.stringify({ title: draft.title, content: draft.content, status: draft.status }) })
        : await api<AnalysisResult>(`/cuts/${cutId}/analysis-results`, { method: 'POST', body: JSON.stringify({ kind: draft.kind, title: draft.title, content: draft.content, status: draft.status }) })
      setDraft(value); await load(history)
    } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  return <div className="rounded-xl border hairline bg-[#1b202a] p-4">
    <div className="flex items-center justify-between gap-2"><p className="flex items-center gap-2 text-xs font-semibold"><WandSparkles size={14} className="text-[#bca7ff]" /> Analysis fields</p><button onClick={() => setDraft({ kind: 'first_frame', title: '', content: '', status: 'draft' })} className="rounded-md p-1 text-[#c3aeff] hover:bg-[#343047]" title="Add analysis field" aria-label="Add analysis field"><Plus size={16} /></button></div>
    <p className="muted mt-2 text-[11px] leading-5">Review and edit findings. Model runs add drafts without replacing your existing work.</p>
    {error && <p className="mt-3 text-xs text-[#ff9fa7]">{error}</p>}
    {results.length > 0 ? <div className="mt-4 space-y-1">{results.map(result => <button key={result.id} onClick={() => setDraft(result)} className={`w-full rounded-lg border px-3 py-2 text-left ${draft?.id === result.id ? 'border-[#a58ae8] bg-[#322a48]' : 'border-[#333a45] bg-[#232a35] hover:border-[#716585]'}`}><span className="flex items-center justify-between gap-2"><strong className="truncate text-[11px]">{result.title || kinds.find(kind => kind.value === result.kind)?.label}</strong><span className={`text-[10px] ${result.status === 'approved' ? 'text-[#7bdcb2]' : 'text-[#aeb5c1]'}`}>{result.status}</span></span><span className="muted mt-1 block truncate text-[10px]">{result.content}</span></button>)}</div> : <p className="muted mt-4 text-[11px]">No analysis fields yet.</p>}
    <button onClick={toggleHistory} className="muted mt-3 flex items-center gap-1 text-[11px] hover:text-white"><History size={12} /> {history ? 'Hide history' : 'Show history'}</button>
    <button onClick={toggleContext} className="muted mt-3 flex items-center gap-1 text-[11px] hover:text-white"><Layers3 size={12} /> {contextOpen ? 'Hide input context' : 'Review input context'}</button>
    {contextOpen && <div className="mt-3 space-y-3 rounded-lg border border-[#3c4654] bg-[#151a22] p-3 text-[11px] leading-5">{context ? <><div><strong className="text-[#c3aeff]">Project synopsis</strong><p className="muted">{context.project.synopsis || 'None entered'}</p></div><div><strong className="text-[#c3aeff]">Working context</strong><p className="muted">{context.context_items.length ? context.context_items.map(item => `${item.title} (${item.kind.replaceAll('_', ' ')})`).join(', ') : 'No context items'}</p></div><div><strong className="text-[#c3aeff]">Adjacent cuts</strong><p className="muted">Previous: {context.previous_cut?.title || context.previous_cut?.description || 'None'} · Next: {context.next_cut?.title || context.next_cut?.description || 'None'}</p><p className="muted">Approved neighbor results: {context.neighbor_results.previous.length + context.neighbor_results.next.length}</p></div><div><strong className="text-[#c3aeff]">Audio in this cut</strong><p className="muted">{context.transcript_segments.length ? context.transcript_segments.map(segment => `${segment.speaker_label || 'Speaker'}: ${segment.text}`).join(' · ') : 'No approved transcript segments yet'}</p></div></> : <p className="muted">Loading context…</p>}</div>}
    {draft && <div className="mt-4 space-y-3 border-t hairline pt-4"><div className="flex items-center justify-between"><strong className="text-xs">{draft.id ? 'Edit result' : 'New result'}</strong><button onClick={() => setDraft(null)} aria-label="Close result editor" className="muted"><X size={14} /></button></div><label className="block"><span className="label">Field</span><select className="field" disabled={!!draft.id} value={draft.kind} onChange={event => setDraft({ ...draft, kind: event.target.value as ResultKind })}>{kinds.map(kind => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></label><label className="block"><span className="label">Title</span><input className="field" maxLength={200} value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="Optional label" /></label><label className="block"><span className="label">Content</span><textarea className="field min-h-32 resize-y" maxLength={50000} value={draft.content} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Describe what you see or want to generate…" /></label><label className="block"><span className="label">Status</span><select className="field" value={draft.status} onChange={event => setDraft({ ...draft, status: event.target.value as ResultStatus })}><option value="draft">Draft</option><option value="approved">Approved</option><option value="rejected">Rejected</option></select></label>{draft.id && <p className="muted text-[10px]">Source: {draft.origin}{draft.model_id ? ` · ${draft.model_id}` : ''}. Saving creates a new revision.</p>}<button onClick={save} disabled={busy || !draft.content.trim() || !!draft.superseded_by} className="btn btn-primary w-full"><Save size={14} /> {draft.id ? 'Save revision' : 'Add result'}</button>{draft.superseded_by && <p className="muted text-[10px]">Historical revision. Select the current version to edit.</p>}</div>}
  </div>
}
