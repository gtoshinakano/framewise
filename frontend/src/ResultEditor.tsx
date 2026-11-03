import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { History, Layers3, Plus, Save, Trash2, WandSparkles, X } from 'lucide-react'
import { api, type AnalysisContext, type AnalysisResult, type ResultKind } from './api'

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

type Draft = Pick<AnalysisResult, 'kind' | 'title' | 'content'> & { id?: string; superseded_by?: string | null; origin?: string; model_id?: string | null }

export default function ResultEditor({ cutId, refreshKey, onLoaded, onChanged, editorTarget }: { cutId: string; refreshKey: string; onLoaded?: (results: AnalysisResult[]) => void; onChanged?: () => Promise<void>; editorTarget: HTMLElement | null }) {
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [history, setHistory] = useState(false)
  const [contextOpen, setContextOpen] = useState(false)
  const [context, setContext] = useState<AnalysisContext | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const load = useCallback(async (showHistory: boolean) => {
    try { const value = await api<AnalysisResult[]>(`/cuts/${cutId}/analysis-results?include_context=false${showHistory ? '&history=true' : ''}`); setResults(value); onLoaded?.(value); setError('') }
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
  async function remove(result: Draft) {
    if (!result.id) return
    setBusy(true)
    try {
      await api(`/analysis-results/${result.id}`, { method: 'DELETE' })
      if (draft?.id === result.id) setDraft(null)
      await load(history); await onChanged?.()
    } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  async function save() {
    if (!draft?.content.trim()) return
    setBusy(true)
    try {
      const value = draft.id
        ? await api<AnalysisResult>(`/analysis-results/${draft.id}`, { method: 'PUT', body: JSON.stringify({ title: draft.title, content: draft.content }) })
        : await api<AnalysisResult>(`/cuts/${cutId}/analysis-results`, { method: 'POST', body: JSON.stringify({ kind: draft.kind, title: draft.title, content: draft.content }) })
      setDraft(value); await load(history); await onChanged?.()
    } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  return <div className="rounded-xl border hairline bg-[#1b202a] p-4">
    <div className="flex items-center justify-between gap-2"><p className="flex items-center gap-2 text-xs font-semibold"><WandSparkles size={14} className="text-[#bca7ff]" /> Analysis fields</p><button onClick={() => setDraft({ kind: 'first_frame', title: '', content: '' })} className="rounded-md p-1 text-[#c3aeff] hover:bg-[#343047]" title="Add analysis field" aria-label="Add analysis field"><Plus size={16} /></button></div>
    <p className="muted mt-2 text-[11px] leading-5">Review and edit findings. Model runs add findings without replacing your existing work.</p>
    {error && <p className="mt-3 text-xs text-[#ff9fa7]">{error}</p>}
    {results.length > 0 ? <div className="mt-4 space-y-1">{results.map(result => <div key={result.id} className="flex items-center gap-2 rounded-lg border border-[#333a45] bg-[#232a35] p-2">
      <button onClick={() => setDraft(result)} className={`min-w-0 flex-1 rounded-md px-2 py-1 text-left ${draft?.id === result.id ? 'bg-[#322a48] ring-1 ring-[#a58ae8]' : 'hover:bg-[#303746]'}`}><strong className="block truncate text-[11px]">{result.title || kinds.find(kind => kind.value === result.kind)?.label}</strong><span className="muted mt-1 block truncate text-[10px]">{result.content}</span>{result.superseded_by && <span className="muted text-[10px]">Historical revision</span>}</button>
      {!result.superseded_by && <button onClick={() => remove(result)} disabled={busy} className="shrink-0 rounded-md p-1.5 text-[#98a1af] hover:bg-[#3a2229] hover:text-[#ff9fa7]" title="Delete this result" aria-label={`Delete ${result.title || result.kind}`}><Trash2 size={14} /></button>}</div>)}</div> : <p className="muted mt-4 text-[11px]">No analysis fields yet.</p>}
    <button onClick={toggleHistory} className="muted mt-3 flex items-center gap-1 text-[11px] hover:text-white"><History size={12} /> {history ? 'Hide history' : 'Show history'}</button>
    <button onClick={toggleContext} className="muted mt-3 flex items-center gap-1 text-[11px] hover:text-white"><Layers3 size={12} /> {contextOpen ? 'Hide input context' : 'Review input context'}</button>
    {contextOpen && <div className="mt-3 space-y-3 rounded-lg border border-[#3c4654] bg-[#151a22] p-3 text-[11px] leading-5">{context ? <><div><strong className="text-[#c3aeff]">Project synopsis</strong><p className="muted">{context.project.synopsis || 'None entered'}</p></div><div><strong className="text-[#c3aeff]">Working context</strong><p className="muted">{context.context_items.length ? context.context_items.map(item => `${item.title} (${item.kind.replaceAll('_', ' ')})`).join(', ') : 'No context items'}</p></div><div><strong className="text-[#c3aeff]">Adjacent cuts</strong><p className="muted">Previous: {context.previous_cut?.title || context.previous_cut?.description || 'None'} · Next: {context.next_cut?.title || context.next_cut?.description || 'None'}</p><p className="muted">Neighbor results: {context.neighbor_results.previous.length + context.neighbor_results.next.length}</p></div><div><strong className="text-[#c3aeff]">Audio in this cut</strong><p className="muted">{context.transcript_segments.length ? context.transcript_segments.map(segment => `${segment.speaker_label || 'Speaker'}: ${segment.text}`).join(' · ') : 'No transcript segments in this cut'}</p></div></> : <p className="muted">Loading context…</p>}</div>}
    {draft && editorTarget && createPortal(<div className="flex flex-1 flex-col gap-3 rounded-xl border hairline bg-[#1b202a] p-4"><div className="flex items-center justify-between"><strong className="text-xs">{draft.id ? 'Edit result' : 'New result'}</strong><button onClick={() => setDraft(null)} aria-label="Close result editor" className="muted"><X size={14} /></button></div><label className="block"><span className="label">Field</span><select className="field" disabled={!!draft.id} value={draft.kind} onChange={event => setDraft({ ...draft, kind: event.target.value as ResultKind })}>{kinds.map(kind => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></label><label className="block"><span className="label">Title</span><input className="field" maxLength={200} value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="Optional label" /></label><label className="flex flex-1 flex-col"><span className="label">Content</span><textarea className="field min-h-32 flex-1 resize-y" maxLength={50000} value={draft.content} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Describe what you see or want to generate…" /></label>{draft.id && <p className="muted text-[10px]">Source: {draft.origin}{draft.model_id ? ` · ${draft.model_id}` : ''}. Saving creates a new revision.</p>}<button onClick={save} disabled={busy || !draft.content.trim() || !!draft.superseded_by} className="btn btn-primary w-full"><Save size={14} /> {draft.id ? 'Save revision' : 'Add result'}</button>{draft.id && !draft.superseded_by && <button onClick={() => remove(draft)} disabled={busy} className="btn btn-danger w-full"><Trash2 size={14} /> Delete result</button>}{draft.superseded_by && <p className="muted text-[10px]">Historical revision. Select the current version to edit.</p>}</div>, editorTarget)}
  </div>
}
