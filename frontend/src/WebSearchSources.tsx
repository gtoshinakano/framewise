import { useEffect, useRef, useState } from 'react'
import { ArrowDown, ArrowUp, Globe, Plus, Save, Trash2, X } from 'lucide-react'
import { api } from './api'
import { IconButton } from './EditorUi'

type Source = { kind: 'wiki' | 'page' | 'directory'; url: string }
const path = '/web-search/sources'

export function WebSearchSources({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false)
  return <>
    <IconButton icon={Globe} label="Web search sources" text={compact ? undefined : 'Web search sources'} size="sm" onClick={() => setOpen(true)} />
    {open && <WebSearchSourcesDialog onClose={() => setOpen(false)} />}
  </>
}

function WebSearchSourcesDialog({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [sources, setSources] = useState<Source[] | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    const panel = dialog.current
    panel?.showModal()
    const controller = new AbortController()
    api<{ sources: Source[] }>(path, { signal: controller.signal }).then(result => setSources(result.sources))
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message) })
    return () => { controller.abort(); panel?.close() }
  }, [])

  function edit(index: number, value: Partial<Source>) {
    setSources(items => items?.map((item, position) => position === index ? { ...item, ...value } : item) ?? null)
  }
  function move(index: number, offset: number) {
    if (!sources) return
    const next = [...sources]
    const [item] = next.splice(index, 1)
    next.splice(index + offset, 0, item)
    setSources(next)
  }
  async function save() {
    setSaving(true); setError('')
    try {
      await api(path, { method: 'PUT', body: JSON.stringify({ sources }) })
      onClose()
    } catch (reason) { setError((reason as Error).message) }
    finally { setSaving(false) }
  }

  return <dialog ref={dialog} aria-modal="true" aria-labelledby="web-sources-heading" onKeyDown={event => { if (event.key === 'Escape') event.stopPropagation() }} onCancel={event => { event.preventDefault(); if (!saving) onClose() }}
    className="m-auto max-h-[85vh] w-[min(94vw,46rem)] overflow-auto rounded-xl border hairline bg-[#1b202a] p-5 text-[#ece9f3] shadow-2xl backdrop:bg-black/70">
    <div className="mb-3 flex items-center justify-between"><h2 id="web-sources-heading" className="font-semibold">Web search sources</h2><IconButton icon={X} size="sm" label="Close web search sources" onClick={onClose} disabled={saving} /></div>
    <p className="muted mb-4 text-xs">Framewise searches these sites in order across all projects. Add a MediaWiki site, a character directory, or the exact URL of a character page.</p>
    {error && <p role="alert" className="mb-3 text-sm text-[#ff9fa7]">{error}</p>}
    {sources && <form onSubmit={event => { event.preventDefault(); void save() }} className="space-y-3">
      {sources.map((source, index) => <div key={index} className="flex flex-wrap items-center gap-2">
        <select className="field !w-auto" aria-label={`Source ${index + 1} type`} value={source.kind} onChange={event => edit(index, { kind: event.target.value as Source['kind'] })}>
          <option value="wiki">MediaWiki site</option><option value="directory">Directory</option><option value="page">Exact page</option>
        </select>
        <input className="field min-w-40 flex-1" aria-label={`Source ${index + 1} URL`} type="url" placeholder="https://" value={source.url} required maxLength={1000} onChange={event => edit(index, { url: event.target.value })} />
        <IconButton icon={ArrowUp} size="sm" label={`Move source ${index + 1} up`} disabled={index === 0} onClick={() => move(index, -1)} />
        <IconButton icon={ArrowDown} size="sm" label={`Move source ${index + 1} down`} disabled={index === sources.length - 1} onClick={() => move(index, 1)} />
        <IconButton icon={Trash2} size="sm" label={`Remove source ${index + 1}`} onClick={() => setSources(sources.filter((_, position) => position !== index))} />
      </div>)}
      <div className="flex justify-between gap-2"><IconButton icon={Plus} label="Add source" text="Add source" onClick={() => setSources([...sources, { kind: 'page', url: '' }])} disabled={sources.length >= 20} />
        <button className="btn btn-primary" type="submit" disabled={saving}><Save size={15} /> Save sources</button></div>
    </form>}
  </dialog>
}
