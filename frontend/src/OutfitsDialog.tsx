import { useEffect, useRef, useState } from 'react'
import { Pencil, Plus, Trash2, X } from 'lucide-react'
import { api, type ContextItem } from './api'
import { IconButton } from './EditorUi'
import { groupOutfits, originLabel, preview, type Outfit } from './outfitView'

type Draft = { id?: string; character_context_id: string; name: string; summary: string; description: string }
const blank: Draft = { character_context_id: '', name: '', summary: '', description: '' }

/** The project's outfit library: what each character wears, shared by every episode. The Sequence director adds outfits here when it writes a bible. */
export function OutfitsDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [rows, setRows] = useState<Outfit[]>([])
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const url = `/projects/${projectId}/outfits`
  useEffect(() => {
    dialog.current?.showModal()
    const controller = new AbortController()
    Promise.all([api<Outfit[]>(url, { signal: controller.signal }), api<ContextItem[]>(`/projects/${projectId}/context`, { signal: controller.signal })])
      .then(([outfits, context]) => { setRows(outfits); setCharacters(context.filter(item => item.kind === 'character' && !item.unregistered)) })
      .catch(err => { if (!controller.signal.aborted) setError((err as Error).message) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => { controller.abort(); dialog.current?.close() }
  }, [projectId, url])
  async function save(remove = false) {
    if (!draft) return
    setBusy(true)
    setError('')
    try {
      const path = draft.id ? `${url}/${draft.id}` : url
      const body = { character_context_id: draft.character_context_id, name: draft.name, summary: draft.summary, description: draft.description }
      setRows(await api<Outfit[]>(path, remove ? { method: 'DELETE' } : { method: draft.id ? 'PUT' : 'POST', body: JSON.stringify(body) }))
      setDraft(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const groups = groupOutfits(rows)
  return <dialog ref={dialog} aria-modal="true" aria-labelledby="outfits-title"
    onCancel={event => { event.preventDefault(); if (!busy) onClose() }}
    className="m-auto max-h-[88vh] w-[min(94vw,46rem)] overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 text-[#ece9f3] shadow-2xl backdrop:bg-black/70">
    <div className="mb-1 flex items-center justify-between gap-2">
      <h2 id="outfits-title" className="font-semibold">Outfits</h2>
      <IconButton icon={X} label="Close outfits" size="sm" disabled={busy} onClick={onClose} />
    </div>
    <p className="muted mb-4 text-xs leading-5">Shared by every episode of this project. The Sequence director picks from this list and adds an outfit only when none matches; the full description goes into the sequence download so a character sheet can be made from it.</p>
    {error && <p role="alert" className="mb-3 text-xs text-[#ff9fa7]">{error}</p>}
    {draft ? <form className="mb-4 space-y-3 rounded-lg border hairline p-3" onSubmit={event => { event.preventDefault(); void save() }}>
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="text-xs">Character<select className="field mt-1" required value={draft.character_context_id}
          onChange={event => setDraft({ ...draft, character_context_id: event.target.value })}>
          <option value="" disabled>Choose a character</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
        </select></label>
        <label className="text-xs">Outfit name<input className="field mt-1" required maxLength={120} value={draft.name}
          onChange={event => setDraft({ ...draft, name: event.target.value })} /></label>
      </div>
      <label className="block text-xs">Summary <span className="muted">(one sentence; left empty it is taken from the description)</span>
        <input className="field mt-1" maxLength={400} value={draft.summary} onChange={event => setDraft({ ...draft, summary: event.target.value })} /></label>
      <label className="block text-xs">Full description <span className="muted">(clothes only: every garment, colours, materials, details)</span>
        <textarea className="field mt-1 min-h-[9rem]" required maxLength={6000} value={draft.description}
          onChange={event => setDraft({ ...draft, description: event.target.value })} /></label>
      <div className="flex justify-end gap-2">
        {draft.id && <button type="button" className="btn btn-quiet mr-auto text-[#ff9fa7]" disabled={busy} onClick={() => void save(true)}><Trash2 size={14} /> Delete</button>}
        <button type="button" className="btn btn-quiet" onClick={() => setDraft(null)} disabled={busy}>Cancel</button>
        <button type="submit" className="btn btn-primary" disabled={busy}>Save outfit</button>
      </div>
    </form> : <IconButton icon={Plus} label="Add an outfit" text="Add outfit" size="sm" className="mb-3" disabled={loading || !characters.length} onClick={() => setDraft(blank)} />}
    {loading ? <p className="muted text-sm">Loading outfits…</p> : !groups.length ? <p className="muted text-sm">No outfits yet. The Sequence director adds them when it writes a bible, or add one here.</p> :
      <div className="space-y-4">{groups.map(group => <section key={group.character_context_id}>
        <h3 className="mb-1 text-sm font-semibold">{group.character_name}</h3>
        <ul className="divide-y divide-[#333945] rounded-lg border hairline">{group.outfits.map(row => <li key={row.id} className="flex items-start justify-between gap-3 px-3 py-2 text-xs">
          <span className="min-w-0">
            <strong>{row.name}</strong>
            <span className="muted"> · {originLabel(row.origin)}</span>
            {row.needs_review ? <span className="ml-1.5 rounded bg-[#4a3a17] px-1.5 py-0.5 text-[10px] font-semibold text-[#ffd59a]" title="Imported from text that may be cut off">check text</span> : null}
            <span className="mt-0.5 block leading-5 text-[#c9cfda]">{row.summary || preview(row.description)}</span>
          </span>
          <IconButton icon={Pencil} label={`Edit ${row.name}`} size="sm" disabled={busy}
            onClick={() => { setDraft({ id: row.id, character_context_id: row.character_context_id, name: row.name, summary: row.summary, description: row.description }); dialog.current?.scrollTo(0, 0) }} />
        </li>)}</ul>
      </section>)}</div>}
  </dialog>
}
