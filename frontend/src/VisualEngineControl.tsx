import { useEffect, useRef, useState } from 'react'
import { Eye, X } from 'lucide-react'
import { api } from './api'
import { engineBadge, installHints, roleOptions, type VisualModels } from './visualEngine'

/** The visual engine button and dialog: which model runs each visual analysis role. */
export function VisualEngineControl({ compact = false }: { compact?: boolean } = {}) {
  const [data, setData] = useState<VisualModels | null>(null)
  const [open, setOpen] = useState(false)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState('')
  const panel = useRef<HTMLDivElement>(null)
  async function refresh() {
    try { setData(await api<VisualModels>('/visual-models')) } catch (err) { setError((err as Error).message) }
  }
  useEffect(() => { void refresh() }, [])
  useEffect(() => {
    if (!open) return
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    void refresh()
    const timer = setInterval(() => void refresh(), 5000)  // shows download progress while open
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', key)
    return () => { clearInterval(timer); window.removeEventListener('keydown', key); opener?.focus() }
  }, [open])
  async function choose(role: string, model: string) {
    setSaving(role); setError('')
    try { setData(await api<VisualModels>('/visual-models', { method: 'PUT', body: JSON.stringify({ role, model }) })) }
    catch (err) { setError((err as Error).message) } finally { setSaving('') }
  }
  const hints = data ? installHints(data) : []
  return <>
    <button type="button" className={compact ? 'icon-btn icon-btn-md icon-btn-text whitespace-nowrap' : 'btn btn-quiet whitespace-nowrap'} onClick={() => { setError(''); setOpen(true) }} title="Visual engine: models for frame, video and character analysis">
      <Eye size={15} aria-hidden /><span className="hidden sm:inline">Visual · </span><span>{engineBadge(data)}</span>
    </button>
    {open && <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) setOpen(false) }}>
      <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="visual-engine-title" className="max-h-[92vh] w-full max-w-xl space-y-4 overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 shadow-2xl outline-none">
        <div className="flex items-center justify-between"><h2 id="visual-engine-title" className="text-base font-semibold">Visual inference engine</h2><button type="button" onClick={() => setOpen(false)} aria-label="Close visual engine settings" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button></div>
        <p className="muted text-xs">Applies to analysis queued from now on; queued jobs keep the model they were queued with and existing results stay. To re-analyse cuts with another model, use “Re-run all visual analysis” or a cut's own analysis buttons; earlier results stay in history.</p>
        {data && !data.worker_reported && <p role="status" className="text-xs text-[#f5c678]">The inference worker has not reported its models recently, so only the defaults can be chosen.</p>}
        {data?.roles.map(role => {
          const options = roleOptions(data, role)
          const current = data.models.find(model => model.key === data.selected[role.role])
          return <label key={role.role} className="block"><span className="label">{role.label}</span>
            <select className="field" value={data.selected[role.role]} disabled={!!saving || options.length < 2} onChange={event => void choose(role.role, event.target.value)}>
              {options.map(option => <option key={option.key} value={option.key} disabled={option.disabled}>{option.label}</option>)}
            </select>
            {current && <span className="muted mt-1 block text-[11px]">{current.note}</span>}
          </label>
        })}
        {data?.models.filter(model => model.helper).map(model => <p key={model.key} className="text-xs"><strong>{model.label}</strong> <span className={model.available ? 'text-[#7bd88f]' : 'text-[#f5c678]'}>· {model.available ? 'Installed' : model.detail}</span><span className="muted mt-0.5 block text-[11px]">{model.note}</span></p>)}
        {hints.length > 0 && <div className="rounded-lg border hairline p-3 text-xs"><strong>Not installed</strong>
          {hints.map(hint => <p key={hint.key} className="muted mt-1">{hint.label} · {hint.detail}. Install on the host with <code className="rounded bg-[#11151c] px-1">{hint.command}</code>, then recreate the inference worker.</p>)}
        </div>}
        {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
        <div className="flex justify-end"><button className="btn btn-quiet" onClick={() => setOpen(false)}>Close</button></div>
      </div>
    </div>}
  </>
}
