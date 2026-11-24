import { CharacterSelect } from './CharacterSelect'
import { useEffect, useRef, useState } from 'react'
import { Pencil, Plus, Trash2, X } from 'lucide-react'
import { api, type ContextItem } from './api'
import { IconButton } from './EditorUi'

type Entry = { id: string; spelling: string; reading: string; character_context_id: string | null; character_name: string | null; part: 'family' | 'given' | null }
type Draft = Omit<Entry, 'id' | 'character_name'> & { id?: string }
const blank: Draft = { spelling: '', reading: '', character_context_id: null, part: null }

export function NameGlossaryDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [rows, setRows] = useState<Entry[]>([])
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [draft, setDraft] = useState<Draft | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const url = `/projects/${projectId}/name-glossary`
  useEffect(() => {
    dialog.current?.showModal()
    const controller = new AbortController()
    Promise.all([api<Entry[]>(url, { signal: controller.signal }), api<ContextItem[]>(`/projects/${projectId}/context`, { signal: controller.signal })])
      .then(([entries, context]) => { setRows(entries); setCharacters(context.filter(item => item.kind === 'character' && !item.unregistered)) })
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
      setRows(await api<Entry[]>(path, remove ? { method: 'DELETE' } : { method: draft.id ? 'PUT' : 'POST', body: JSON.stringify(draft) }))
      setDraft(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return <dialog ref={dialog} aria-modal="true" aria-labelledby="name-glossary-title"
    onCancel={event => { event.preventDefault(); if (!busy) onClose() }}
    className="m-auto max-h-[85vh] w-[min(92vw,38rem)] overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 text-[#ece9f3] shadow-2xl backdrop:bg-black/70">
    <div className="mb-4 flex items-center justify-between gap-2">
      <h2 id="name-glossary-title" className="font-semibold">Name glossary</h2>
      <IconButton icon={X} label="Close name glossary" size="sm" disabled={busy} onClick={onClose} />
    </div>
    {error && <p role="alert" className="mb-3 text-xs text-[#ff9fa7]">{error}</p>}
    {draft ? <form className="mb-4 space-y-3 rounded-lg border hairline p-3" onSubmit={event => { event.preventDefault(); void save() }}>
      <div className="grid grid-cols-2 gap-2">
        <label className="text-xs">Spelling<input className="field mt-1" required maxLength={120} value={draft.spelling}
          onChange={event => setDraft({ ...draft, spelling: event.target.value })} /></label>
        <label className="text-xs">Kana reading<input className="field mt-1" required maxLength={120} value={draft.reading}
          onChange={event => setDraft({ ...draft, reading: event.target.value })} /></label>
        <label className="text-xs">Character<CharacterSelect aria-label="Character" projectId={characters[0]?.project_id} className="field mt-1" value={draft.character_context_id || ''}
          onChange={event => setDraft({ ...draft, character_context_id: event.target.value || null, part: event.target.value ? draft.part || 'family' : null })}>
          <option value="">Other proper noun</option>{characters.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
        </CharacterSelect></label>
        {draft.character_context_id && <label className="text-xs">Name part<select className="field mt-1" value={draft.part || 'family'}
          onChange={event => setDraft({ ...draft, part: event.target.value as 'family' | 'given' })}>
          <option value="family">Family name</option><option value="given">Given name</option>
        </select></label>}
      </div>
      <div className="flex justify-end gap-2">
        {draft.id && <button type="button" className="btn btn-quiet mr-auto text-[#ff9fa7]" disabled={busy} onClick={() => void save(true)}><Trash2 size={14} /> Delete</button>}
        <button type="button" className="btn btn-quiet" onClick={() => setDraft(null)} disabled={busy}>Cancel</button>
        <button type="submit" className="btn btn-primary" disabled={busy}>Save entry</button>
      </div>
    </form> : <IconButton icon={Plus} label="Add a glossary entry" text="Add entry" size="sm" className="mb-3" disabled={loading} onClick={() => setDraft(blank)} />}
    {loading ? <p className="muted text-sm">Loading glossary…</p> :
      <ul className="divide-y divide-[#333945]">{rows.map(row => <li key={row.id} className="flex items-center justify-between gap-3 py-2 text-xs">
        <span><strong>{row.spelling}</strong> <span className="muted">{row.reading}</span><span className="muted block">{row.character_name ? `${row.character_name} · ${row.part} name` : 'Other proper noun'}</span></span>
        <IconButton icon={Pencil} label={`Edit ${row.spelling}`} size="sm" disabled={busy} onClick={() => { setDraft(row); dialog.current?.scrollTo(0, 0) }} />
      </li>)}</ul>}
  </dialog>
}
