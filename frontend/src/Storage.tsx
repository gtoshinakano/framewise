import { useCallback, useEffect, useState } from 'react'
import { ArrowLeft, RefreshCw, Trash2 } from 'lucide-react'
import { api } from './api'

type Link = { job_id?: string | null; cut_id?: string; result_id?: string; sample_id?: string }
type StoredFile = { path: string; size: number; modified_ns: string; project_title: string | null; episode_id: string | null; episode_title: string | null; kind: string; reason: string; deletable: boolean; cleanup_candidate: boolean; links: Link[] }
type Orphan = { id: string; cut_id?: string; episode_id?: string; title?: string; kind?: string; content?: string; text?: string; job_id: string | null; created_at: string }
type Inventory = { orphan_evidence: { job_id: string; cut_id: string; frame_count: number | null }[]; orphan_results: Orphan[]; orphan_transcripts: Orphan[]; files: StoredFile[]; missing: { episode_id: string; episode_title: string; file: string; description: string; links?: Link[] }[]; warnings: string[]; total_bytes: number; reclaimable_bytes: number }
function bytes(n: number) { if (n < 1024) return `${n} B`; const unit = Math.min(Math.floor(Math.log(n) / Math.log(1024)), 4); return `${(n / 1024 ** unit).toFixed(1)} ${['B', 'KiB', 'MiB', 'GiB', 'TiB'][unit]}` }

export default function Storage() {
  const [inventory, setInventory] = useState<Inventory | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState<'all' | 'cleanup'>('all')
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(100)
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const [message, setMessage] = useState('')
  const load = useCallback(async () => { setWorking(true); try { setInventory(await api<Inventory>('/storage')); setSelected(new Set()); setError('') } catch (err) { setError((err as Error).message) } finally { setWorking(false) } }, [])
  useEffect(() => { void load() }, [load])
  const visible = (inventory?.files || []).filter(file => (filter === 'all' || file.cleanup_candidate) && `${file.path} ${file.kind} ${file.project_title || ''} ${file.episode_title || ''}`.toLowerCase().includes(query.toLowerCase()))
  const chosen = (inventory?.files || []).filter(file => selected.has(file.path))
  async function remove() {
    if (chosen.length > 1000) { setError('Select at most 1,000 files per cleanup.'); return }
    if (!chosen.length || !window.confirm(`Permanently delete ${chosen.length} selected file(s), totaling ${bytes(chosen.reduce((n, file) => n + file.size, 0))}?${chosen.some(file => !file.cleanup_candidate) ? " Some are used by the app and must be rebuilt before those features work again." : ""}`)) return
    setWorking(true); setError(''); setMessage('')
    try {
      const result = await api<{ removed: string[]; errors: { path: string; error: string }[] }>('/storage/cleanup', { method: 'POST', body: JSON.stringify({ files: chosen.map(({ path, size, modified_ns }) => ({ path, size, modified_ns })) }) })
      await load(); setMessage(`Deleted ${result.removed.length} file(s).${result.errors.length ? ` Could not delete: ${result.errors.map(item => `${item.path}: ${item.error}`).join('; ')}` : ''}`)
    } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function removeOrphan(kind: 'result' | 'transcript', item: Orphan) {
    if (!window.confirm(`Permanently delete orphan ${kind} ${item.id} and any linked orphan revisions? This cannot be undone.`)) return
    setWorking(true); setError(''); setMessage('')
    try { await api(`/storage/orphans/${kind}/${item.id}`, { method: 'DELETE' }); await load(); setMessage(`Deleted orphan ${kind}.`) } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function removeOrphanEvidence(item: { job_id: string; cut_id: string }) {
    if (!window.confirm('Delete saved evidence whose run is missing? Unreferenced files can then be removed.')) return
    setWorking(true); setError(''); setMessage('')
    try { await api(`/storage/orphan-evidence/${item.job_id}/${item.cut_id}`, { method: 'DELETE' }); await load(); setMessage('Deleted orphan evidence.') } catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  return <div className="min-h-screen"><header className="border-b hairline bg-[#10131a]"><div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4"><h1 className="text-lg font-semibold">Storage &amp; cleanup</h1><a href="#/" className="btn btn-quiet"><ArrowLeft size={15} /> Projects</a></div></header>
    <main className="mx-auto max-w-7xl px-6 py-8"><p className="muted max-w-3xl text-sm">Inspect files stored with projects and episodes, their links, and missing media. Originals are protected. Video frames can be recreated on demand; playback media can be rebuilt with Prepare. The model cache and database are not included.</p>
      {error && <p role="alert" className="mt-4 text-sm text-[#ff9fa7]">{error}</p>}{message && <p role="status" className="mt-4 text-sm text-[#7bdcb2]">{message}</p>}
      {inventory && <><div className="my-6 flex flex-wrap items-center gap-4 text-sm"><strong>{bytes(inventory.total_bytes)} in project files</strong><span className="muted">{bytes(inventory.reclaimable_bytes)} in cleanup candidates</span><button onClick={load} disabled={working} className="btn btn-quiet"><RefreshCw size={14} /> Refresh</button></div>
        {inventory.warnings.map(warning => <p key={warning} className="mb-2 text-sm text-[#ffbf85]">{warning}</p>)}
        <section className="panel mb-6 p-4"><h2 className="font-semibold">Missing files ({inventory.missing.length})</h2><p className="muted mt-1 text-xs">Frames are recreated automatically from their source. A missing original requires restoring the original file.</p>{inventory.missing.length > 0 && <ul className="mt-3 max-h-56 space-y-2 overflow-y-auto text-xs">{inventory.missing.map((item, index) => <li key={`${item.episode_id}-${item.file}-${index}`}><a className="text-[#c3aeff] hover:underline" href={`#/episode/${item.episode_id}`}>{item.episode_title}</a> · {item.description} <span className="muted break-all">{item.file}</span>{item.links?.map(link => <span key={`${link.job_id}-${link.cut_id}`} className="muted block">{link.job_id && `Run ${link.job_id} `}{link.cut_id && `· Cut ${link.cut_id}`}</span>)}</li>)}</ul>}</section>
        {(inventory.orphan_results.length > 0 || inventory.orphan_transcripts.length > 0 || inventory.orphan_evidence.length > 0) && <section className="panel mb-6 p-4"><h2 className="font-semibold">Orphan data ({inventory.orphan_results.length + inventory.orphan_transcripts.length + inventory.orphan_evidence.length})</h2><p className="muted mt-1 text-xs">These records point to a missing cut or episode. Review their content before deleting them.</p><div className="mt-3 max-h-80 space-y-3 overflow-y-auto">{([['result', inventory.orphan_results], ['transcript', inventory.orphan_transcripts]] as const).flatMap(([kind, records]) => records.map(item => <div className="rounded border hairline p-3 text-xs" key={`${kind}-${item.id}`}><strong>{kind === 'result' ? item.title || item.kind || 'Analysis result' : 'Transcript segment'}</strong><p className="muted mt-1 break-all">{item.id} · Missing {kind === 'result' ? `cut ${item.cut_id}` : `episode ${item.episode_id}`}{item.job_id ? ` · Run ${item.job_id}` : ''} · {item.created_at}</p><p className="mt-2 whitespace-pre-wrap break-words">{item.content || item.text}</p><button className="btn btn-danger mt-2" disabled={working} onClick={() => removeOrphan(kind, item)}>Delete orphan {kind}</button></div>))}{inventory.orphan_evidence.map(item => <div className="rounded border hairline p-3 text-xs" key={`${item.job_id}-${item.cut_id}`}><strong>Saved run evidence</strong><p className="muted mt-1 break-all">Missing run {item.job_id} · Cut {item.cut_id} · {item.frame_count === null ? 'Unreadable evidence' : `${item.frame_count} frames`}</p><button className="btn btn-danger mt-2" disabled={working} onClick={() => removeOrphanEvidence(item)}>Delete orphan evidence</button></div>)}</div></section>}
        <div className="mb-3 flex flex-wrap items-center gap-2"><h2 className="mr-auto font-semibold">Project files ({visible.length})</h2><input aria-label="Search files" className="field w-64" placeholder="Search files, projects, episodes" value={query} onChange={event => setQuery(event.target.value)} /><select aria-label="Filter files" className="field w-auto" value={filter} onChange={event => setFilter(event.target.value as 'all' | 'cleanup')}><option value="all">All files</option><option value="cleanup">Cleanup candidates</option></select></div>
        <div className="mb-3 flex flex-wrap items-center gap-3"><button className="btn btn-quiet" onClick={() => setSelected(new Set([...selected, ...visible.slice(0, limit).filter(file => file.cleanup_candidate).map(file => file.path)]))}>Select cleanup candidates on page</button><button className="btn btn-quiet" onClick={() => setSelected(new Set())}>Clear selection</button><button className="btn btn-danger" disabled={working || !chosen.length} onClick={remove}><Trash2 size={14} /> Delete selected ({chosen.length})</button></div>
        <div className="panel max-h-[65vh] overflow-auto">{visible.length ? visible.slice(0, limit).map(file => <div key={file.path} className="flex gap-3 border-b hairline p-3 text-xs last:border-0"><input type="checkbox" aria-label={`Select ${file.path}`} disabled={!file.deletable || working} checked={selected.has(file.path)} onChange={event => setSelected(previous => { const next = new Set(previous); if (event.target.checked) next.add(file.path); else next.delete(file.path); return next })} /><div className="min-w-0 flex-1"><div className="flex flex-wrap gap-x-3 gap-y-1"><strong>{file.kind}</strong><span>{bytes(file.size)}</span><span className="muted">{file.project_title || 'Unlinked project'}{file.episode_title ? ` → ${file.episode_title}` : ''}</span>{file.episode_id && <a className="text-[#c3aeff] hover:underline" href={`#/episode/${file.episode_id}`}>Open episode</a>}</div><p className="mt-1 break-all text-[#c8cfda]">{file.path}</p><p className="muted mt-1">{file.reason}</p>{file.links.map((link, i) => <p key={i} className="muted mt-1 break-all">{link.job_id && `Run ${link.job_id} `}{link.cut_id && `· Cut ${link.cut_id} `}{link.result_id && `· Result ${link.result_id}`}{link.sample_id && `· Voice sample ${link.sample_id}`}</p>)}</div></div>) : <p className="muted p-6 text-sm">No files match.</p>}</div>{visible.length > limit && <button className="btn btn-quiet mt-3" onClick={() => setLimit(value => value + 100)}>Show more files</button>}
      </>}
    </main></div>
}
