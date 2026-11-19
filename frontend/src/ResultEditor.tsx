import { ResultMetrics } from './ResultMetrics'
import { useCallback, useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { AlignLeft, AudioLines, BoxSelect, Captions, Drama, FileText, GalleryHorizontal, History, Image as ImageIcon, Layers3, LayoutTemplate, ListTree, Plus, RotateCcw, Save, Sparkles, StickyNote, Trash2, Video, WandSparkles, X, type LucideIcon } from 'lucide-react'
import { api, type AnalysisContext, type AnalysisResult, type ResultKind } from './api'
import { resultAccent, StaleChip } from './RunUi'

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

const KIND_ICON: Record<ResultKind, LucideIcon> = {
  first_frame: ImageIcon, last_frame: ImageIcon, camera_movement: Video, composition: LayoutTemplate,
  cut_description: AlignLeft, audio_visual_match: AudioLines, generation_prompt: WandSparkles, other: StickyNote,
}

/** Icon, colour and name for the kind of content a result holds, so the list reads without opening anything. */
function lookOf(result: Pick<AnalysisResult, 'kind' | 'title' | 'settings'>): { icon: LucideIcon; color: string; label: string } {
  const type = String(result.settings?.structured_type ?? '')
  const accent = resultAccent(result.settings, result.title)
  if (type === 'consistency_summary') return { icon: Sparkles, ...accent }
  if (result.title === 'Cut summary') return { icon: FileText, ...accent }
  if (type === 'initial_frame' || type === 'last_frame') return { icon: ImageIcon, ...accent }
  if (type === 'keyframe_analysis') return { icon: GalleryHorizontal, ...accent }
  if (type === 'performance_beats') return { icon: Drama, ...accent }
  if (type === 'video_analysis') return { icon: Video, ...accent }
  if (type === 'subtitle_ocr' || /^Raw OCR at /.test(result.title)) return { icon: Captions, color: '#e2709c', label: 'Subtitle text' }
  if (/^Object detections at /.test(result.title)) return { icon: BoxSelect, color: '#f0a64a', label: 'Object detections' }
  if (/^Frame at /.test(result.title)) return { icon: ImageIcon, color: '#6aa8ff', label: 'Frame' }
  return { icon: KIND_ICON[result.kind] ?? StickyNote, color: '#8b95a7', label: kinds.find(kind => kind.value === result.kind)?.label ?? 'Result' }
}

type Draft = Pick<AnalysisResult, 'kind' | 'title' | 'content'> & { id?: string; superseded_by?: string | null; origin?: string; model_id?: string | null; settings?: Record<string, unknown> }

export default function ResultEditor({ cutId, refreshKey, onLoaded, onChanged, editorTarget }: { cutId: string; refreshKey: string; onLoaded?: (results: AnalysisResult[]) => void; onChanged?: () => Promise<void>; editorTarget: HTMLElement | null }) {
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [history, setHistory] = useState(false)
  const [contextOpen, setContextOpen] = useState(false)
  const [context, setContext] = useState<AnalysisContext | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const structured = typeof draft?.settings?.structured_type === 'string'
  let invalidJson = false
  if (structured && draft) {
    try { const value: unknown = JSON.parse(draft.content); invalidJson = !value || typeof value !== 'object' || Array.isArray(value) }
    catch { invalidJson = true }
  }
  const load = useCallback(async (showHistory: boolean) => {
    try { const value = await api<AnalysisResult[]>(`/cuts/${cutId}/analysis-results?include_context=false${showHistory ? '&history=true' : ''}`); setResults(value); onLoaded?.(value); setError('') }
    catch (err) { setError((err as Error).message) }
  }, [cutId])
  useEffect(() => { setDraft(null); setHistory(false); setContextOpen(false); setContext(null); void load(false) }, [cutId, load])
  useEffect(() => { void load(history) }, [refreshKey])
  useEffect(() => {
    // Stacked layout: the editor opens below the lists, so scroll to it. From lg up it is already in view.
    if (draft && editorTarget && window.matchMedia('(max-width: 1023.98px)').matches) editorTarget.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [draft === null, editorTarget])
  async function rerun(result: AnalysisResult) {
    const type = result.settings.structured_type
    const path = result.title === 'Cut summary' ? 'regenerate-summary'
      : type === 'consistency_summary' ? 'consistency-summary'
      : type === 'performance_beats' ? 'performance-beats' : 'analysis-stages/run'
    const body = path === 'regenerate-summary' ? { mode: 'regenerate' }
      : path === 'analysis-stages/run' ? { stage: type } : undefined
    setBusy(true); setError('')
    try { await api(`/cuts/${cutId}/${path}`, { method: 'POST', body: body ? JSON.stringify(body) : undefined }); await onChanged?.() }
    catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }

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
    if (!draft?.content.trim() || invalidJson) return
    setBusy(true)
    try {
      const value = draft.id
        ? await api<AnalysisResult>(`/analysis-results/${draft.id}`, { method: 'PUT', body: JSON.stringify({ title: draft.title, content: draft.content }) })
        : await api<AnalysisResult>(`/cuts/${cutId}/analysis-results`, { method: 'POST', body: JSON.stringify({ kind: draft.kind, title: draft.title, content: draft.content }) })
      setDraft(value); await load(history); await onChanged?.()
    } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  return <section aria-label="Analysis fields" className="shrink-0 border-t hairline pt-3">
    <div className="flex items-center justify-between gap-2"><h4 className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-[.1em] text-[#aeb5c1]" title="Review and edit findings. Model runs add findings without replacing your existing work."><ListTree size={13} className="text-[#bca7ff]" aria-hidden /> Analysis fields <span className="font-normal normal-case tracking-normal text-[#929aa9]">{results.length}</span></h4><button onClick={() => setDraft({ kind: 'first_frame', title: '', content: '' })} className="rounded-md p-1 text-[#c3aeff] hover:bg-[#343047]" title="Add analysis field" aria-label="Add analysis field"><Plus size={15} /></button></div>
    {error && <p className="mt-2 text-xs text-[#ff9fa7]">{error}</p>}
    {results.length > 0 ? <ul className="mt-1.5 space-y-px">{results.map(result => {
      const look = lookOf(result)
      const Icon = look.icon
      const name = result.title || look.label
      const selected = draft?.id === result.id
      return <li key={result.id} className="group flex items-center gap-0.5">
        <button type="button" onClick={() => setDraft(result)} title={`${name} · ${look.label}${result.superseded_by ? ' · earlier revision' : ''}`} aria-current={selected || undefined} className={`flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1 text-left text-xs ${selected ? 'bg-[#322a48] ring-1 ring-[#a58ae8]' : 'hover:bg-[#262d3a]'} ${result.superseded_by ? 'opacity-60' : ''}`}>
          <Icon size={14} style={{ color: look.color }} className="shrink-0" aria-hidden />
          <span className="min-w-0 flex-1 truncate">{name}</span>
          {!result.superseded_by && result.stale_reason && <StaleChip reason={result.stale_reason} />}
          {result.superseded_by && <span className="shrink-0 rounded bg-white/5 px-1 text-[9px] uppercase tracking-wide text-[#929aa9]">older</span>}
        </button>
        {!result.superseded_by && result.stale_reason && <button type="button" onClick={() => void rerun(result)} disabled={busy} className="btn btn-quiet !px-2 !py-1 text-[11px]" title={result.stale_reason} aria-label={`Re-run ${name}`}><RotateCcw size={12} /> Re-run</button>}
        {!result.superseded_by && <button onClick={() => remove(result)} disabled={busy} className="shrink-0 rounded-md p-1 text-[#98a1af] opacity-40 hover:bg-[#3a2229] hover:text-[#ff9fa7] hover:opacity-100 focus-visible:opacity-100 group-hover:opacity-100" title="Delete this result" aria-label={`Delete ${result.title || result.kind}`}><Trash2 size={13} /></button>}
      </li>
    })}</ul> : <p className="muted mt-2 text-[11px]">No analysis fields yet.</p>}
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
      <button onClick={toggleHistory} className="muted flex items-center gap-1 text-[10px] hover:text-white"><History size={11} /> {history ? 'Hide history' : 'Show history'}</button>
      <button onClick={toggleContext} className="muted flex items-center gap-1 text-[10px] hover:text-white"><Layers3 size={11} /> {contextOpen ? 'Hide input context' : 'Review input context'}</button>
    </div>
    {contextOpen && <div className="mt-3 space-y-3 rounded-lg border border-[#3c4654] bg-[#151a22] p-3 text-[11px] leading-5">{context ? <><div><strong className="text-[#c3aeff]">Project synopsis</strong><p className="muted">{context.project.synopsis || 'None entered'}</p></div><div><strong className="text-[#c3aeff]">Working context</strong><p className="muted">{context.context_items.length ? context.context_items.map(item => `${item.title} (${item.kind.replaceAll('_', ' ')})`).join(', ') : 'No context items'}</p></div><div><strong className="text-[#c3aeff]">Adjacent cuts</strong><p className="muted">Previous: {context.previous_cut?.title || context.previous_cut?.description || 'None'} · Next: {context.next_cut?.title || context.next_cut?.description || 'None'}</p><p className="muted">Neighbor results: {context.neighbor_results.previous.length + context.neighbor_results.next.length}</p></div><div><strong className="text-[#c3aeff]">Audio in this cut</strong><p className="muted">{context.transcript_segments.length ? context.transcript_segments.map(segment => `${segment.speaker_label || 'Speaker'}: ${segment.text}`).join(' · ') : 'No transcript segments in this cut'}</p></div></> : <p className="muted">Loading context…</p>}</div>}
    {draft && editorTarget && createPortal(<div className="flex flex-1 flex-col gap-3 rounded-xl border hairline bg-[#1b202a] p-4"><div className="flex items-center justify-between"><strong className="text-xs">{draft.id ? 'Edit result' : 'New result'}</strong><button onClick={() => setDraft(null)} aria-label="Close result editor" className="muted"><X size={14} /></button></div><label className="block"><span className="label">Field</span><select className="field" disabled={!!draft.id} value={draft.kind} onChange={event => setDraft({ ...draft, kind: event.target.value as ResultKind })}>{kinds.map(kind => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></label><label className="block"><span className="label">Title</span><input className="field" maxLength={200} value={draft.title} onChange={event => setDraft({ ...draft, title: event.target.value })} placeholder="Optional label" /></label><label className="flex flex-1 flex-col"><span className="label">Content</span><textarea className="field min-h-32 flex-1 resize-y" maxLength={structured ? 5000000 : 50000} value={draft.content} onChange={event => setDraft({ ...draft, content: event.target.value })} placeholder="Describe what you see or want to generate…" /></label>{structured && <p className={invalidJson ? "text-[10px] text-[#ff9fa7]" : "muted text-[10px]"}>{invalidJson ? "Enter a valid JSON object before saving." : "Structured result · JSON object"}</p>}{draft.id && 'settings' in draft && <ResultMetrics result={draft as AnalysisResult} />}{draft.id && <p className="muted text-[10px]">Source: {draft.origin}{draft.model_id ? ` · ${draft.model_id}` : ''}. Saving creates a new revision.</p>}<button onClick={save} disabled={busy || !draft.content.trim() || invalidJson || !!draft.superseded_by} className="btn btn-primary w-full"><Save size={14} /> {draft.id ? 'Save revision' : 'Add result'}</button>{draft.id && !draft.superseded_by && <button onClick={() => remove(draft)} disabled={busy} className="btn btn-danger w-full"><Trash2 size={14} /> Delete result</button>}{draft.superseded_by && <p className="muted text-[10px]">Historical revision. Select the current version to edit.</p>}</div>, editorTarget)}
  </section>
}
