import { useEffect, useRef, useState } from 'react'
import { api, type ContextItem } from './api'

type Item = { id: string; character_context_id: string; episode_id: string; episode_title: string; cut_index: number; state: 'active' | 'disabled' | 'automatic'; gallery_note: string | null }

export function CharacterMatcherDialog({ projectId, episodeId, onClose }: { projectId: string; episodeId: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const start = useRef<{ x: number; y: number } | null>(null)
  const dragged = useRef(false)
  const [rectangle, setRectangle] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [items, setItems] = useState<Item[]>([])
  const [characters, setCharacters] = useState<ContextItem[]>([])
  const [character, setCharacter] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [preview, setPreview] = useState<Item | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const url = `/projects/${projectId}/character-matcher`
  async function reload() { setItems(await api<Item[]>(url)) }
  useEffect(() => {
    dialog.current?.showModal()
    Promise.all([api<Item[]>(url), api<ContextItem[]>(`/projects/${projectId}/context`)]).then(([rows, context]) => {
      const chars = context.filter(c => c.kind === 'character' && !c.unregistered)
      setItems(rows); setCharacters(chars); setCharacter(chars[0]?.id || '')
    }).catch(e => setError((e as Error).message))
    return () => { dialog.current?.close() }
  }, [projectId, url])
  async function update(ids: string[], use: boolean | null) {
    if (!ids.length) return
    setBusy(true); setError('')
    try { await api(url, { method: 'PUT', body: JSON.stringify({ ids, use }) }); await reload(); setSelected(new Set()) }
    catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  function toggle(id: string) { setSelected(old => { const next = new Set(old); if (next.has(id)) next.delete(id); else next.add(id); return next }) }
  function begin(e: React.PointerEvent<HTMLDivElement>) {
    if (e.button !== 0) return
    start.current = { x: e.clientX, y: e.clientY }; dragged.current = false
  }
  function move(e: React.PointerEvent<HTMLDivElement>) {
    if (!start.current) return
    const x = Math.min(start.current.x, e.clientX), y = Math.min(start.current.y, e.clientY)
    const w = Math.abs(e.clientX - start.current.x), h = Math.abs(e.clientY - start.current.y)
    if (w + h > 5 && !dragged.current) { dragged.current = true; e.currentTarget.setPointerCapture(e.pointerId) }
    setRectangle({ x, y, w, h })
  }
  function end(e: React.PointerEvent<HTMLDivElement>) {
    if (start.current && dragged.current) {
      const x1 = Math.min(start.current.x, e.clientX), x2 = Math.max(start.current.x, e.clientX), y1 = Math.min(start.current.y, e.clientY), y2 = Math.max(start.current.y, e.clientY)
      const ids = Array.from(e.currentTarget.querySelectorAll<HTMLElement>('[data-matcher-id]')).filter(el => { const r = el.getBoundingClientRect(); return r.left < x2 && r.right > x1 && r.top < y2 && r.bottom > y1 }).map(el => el.dataset.matcherId!)
      setSelected(old => new Set([...old, ...ids]))
    }
    start.current = null; setRectangle(null)
  }
  const visible = items.filter(i => i.character_context_id === character)
  const episodeItems = items.filter(i => i.episode_id === episodeId && i.state !== 'disabled')
  function grid(title: string, rows: Item[], small: boolean) {
    return <section className="min-w-0 flex-1"><h3 className="mb-2 text-sm font-semibold">{title} ({rows.length})</h3><div onPointerDown={begin} onPointerMove={move} onPointerUp={end} onPointerCancel={() => { start.current = null; setRectangle(null) }} className="min-h-36 rounded border hairline p-2 select-none"><div className={`grid gap-2 ${small ? 'grid-cols-3 sm:grid-cols-4 lg:grid-cols-5' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-4'}`}>
      {rows.map(item => <button type="button" key={item.id} data-matcher-id={item.id} aria-pressed={selected.has(item.id)} title={`${item.episode_title} · Cut ${item.cut_index}${item.gallery_note ? ' · ' + item.gallery_note : ''}`} onClick={() => toggle(item.id)} onDoubleClick={() => setPreview(item)} className={`overflow-hidden rounded border text-left ${selected.has(item.id) ? 'border-[#bca7ff] ring-2 ring-[#bca7ff]' : 'hairline'}`}><img src={`/api/character-matcher/${item.id}/image`} loading="lazy" draggable={false} alt={`${item.episode_title}, cut ${item.cut_index}`} className={`${small ? 'h-16' : 'h-28'} w-full bg-black object-contain`} /><span className="block truncate p-1 text-[10px]">{item.episode_title} · {item.cut_index}{item.state === 'automatic' ? ' · auto excluded' : ''}</span></button>)}
    </div>{!rows.length && <p className="muted p-3 text-xs">No images</p>}</div></section>
  }
  const chosen = items.filter(i => selected.has(i.id))
  return <dialog ref={dialog} onCancel={e => { e.preventDefault(); if (preview) setPreview(null); else onClose() }} className="m-auto max-h-[90vh] w-[min(1200px,95vw)] max-w-none rounded-xl border hairline bg-[#171b23] p-5 text-white backdrop:bg-black/70"><div className="flex items-center justify-between gap-3"><div><h2 className="text-lg font-semibold">Character matcher store</h2><p className="muted text-xs">Select images or drag across empty grid space. Double-click an image to preview it.</p></div><button onClick={onClose} aria-label="Close" className="icon-btn">✕</button></div>
    {error && <p role="alert" className="my-2 text-sm text-red-300">{error}</p>}
    <div className="my-3 flex flex-wrap gap-2"><button className="btn" disabled={busy || !chosen.some(i => i.state !== 'disabled')} onClick={() => void update(chosen.filter(i => i.state !== 'disabled').map(i => i.id), false)}>Disable selected</button><button className="btn" disabled={busy || !chosen.some(i => i.state === 'disabled')} onClick={() => void update(chosen.filter(i => i.state === 'disabled').map(i => i.id), null)}>Restore selected</button><span className="self-center text-xs muted">{selected.size} selected</span><button className="btn ml-auto" disabled={busy || !episodeItems.length} onClick={() => { if (window.confirm(`Disable ${episodeItems.length} matcher images from this episode across all characters?`)) void update(episodeItems.map(i => i.id), false) }}>Disable this episode ({episodeItems.length})</button></div>
    <div className="flex max-h-[65vh] gap-4 overflow-auto"><nav className="w-36 shrink-0 space-y-1" aria-label="Characters">{characters.map(c => <button key={c.id} onClick={() => { setCharacter(c.id); setSelected(new Set()) }} className={`block w-full rounded px-2 py-1 text-left text-sm ${character === c.id ? 'bg-[#514576]' : 'hover:bg-[#303540]'}`}>{c.title} <span className="muted text-xs">{items.filter(i => i.character_context_id === c.id).length}</span></button>)}</nav><div className="flex min-w-0 flex-1 gap-4">{grid('Active', visible.filter(i => i.state === 'active'), false)}{grid('Disabled and automatically excluded', visible.filter(i => i.state !== 'active'), true)}</div></div>
    {rectangle && <div className="pointer-events-none fixed z-50 border border-[#bca7ff] bg-[#bca7ff]/20" style={{ left: rectangle.x, top: rectangle.y, width: rectangle.w, height: rectangle.h }} />}
    {preview && <div className="absolute inset-0 z-40 flex items-center justify-center rounded-xl bg-black/90 p-6" onClick={() => setPreview(null)}><div className="text-center"><img src={`/api/character-matcher/${preview.id}/image`} alt={`Preview: ${preview.episode_title}, cut ${preview.cut_index}`} className="max-h-[65vh] max-w-full object-contain"/><p className="mt-2 text-sm">{preview.episode_title} · Cut {preview.cut_index}</p><button className="btn mt-2" onClick={() => setPreview(null)}>Close preview</button></div></div>}
  </dialog>
}
