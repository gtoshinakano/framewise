import { useEffect, useRef, useState } from 'react'
import { Cpu, X } from 'lucide-react'
import { api } from './api'

type ModelStatus = { desired_loaded: boolean; state: string; detail: string }
type TextSettings = { text_provider: 'local' | 'openrouter'; local_thinking: boolean; openrouter_model: string }
const labels: Record<string, string> = { unloaded: 'Unloaded', downloading: 'Downloading', loading: 'Loading', loaded: 'Loaded', generating: 'Generating', unloading: 'Unloading', worker_offline: 'Worker offline', error: 'Error' }

export function LocalModelControl() {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [settings, setSettings] = useState<TextSettings | null>(null)
  const [draft, setDraft] = useState<TextSettings | null>(null)
  const [open, setOpen] = useState(false)
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const panel = useRef<HTMLDivElement>(null)
  async function refresh() { try { setStatus(await api<ModelStatus>('/local-text-model')) } catch (err) { setError((err as Error).message) } }
  useEffect(() => { void refresh(); void api<TextSettings>('/text-generation-settings').then(setSettings).catch(err => setError((err as Error).message)); const timer = setInterval(() => void refresh(), 2500); return () => clearInterval(timer) }, [])
  useEffect(() => {
    if (!open) return
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', key)
    return () => { window.removeEventListener('keydown', key); opener?.focus() }
  }, [open])
  async function toggle() {
    if (!status) return
    setWorking(true); setError('')
    try { setStatus(await api<ModelStatus>('/local-text-model', { method: 'PUT', body: JSON.stringify({ loaded: !status.desired_loaded }) })) }
    catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function save() {
    if (!draft) return
    setWorking(true); setError('')
    try { const value = await api<TextSettings>('/text-generation-settings', { method: 'PUT', body: JSON.stringify(draft) }); setSettings(value); setOpen(false) }
    catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  return <>
    <button type="button" className="btn btn-quiet whitespace-nowrap" onClick={() => { setDraft(settings); setError(''); setOpen(true); void api<TextSettings>('/text-generation-settings').then(value => { setSettings(value); setDraft(value) }).catch(err => setError((err as Error).message)) }} title="Text engine settings"><Cpu size={15} /><span className="hidden sm:inline">{settings?.text_provider === 'local' ? 'Local' : 'OpenRouter'} · </span><span>Local {status ? labels[status.state] || status.state : 'checking…'}</span></button>
    {open && <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) setOpen(false) }}>
      <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="text-engine-title" className="w-full max-w-lg space-y-4 rounded-xl border hairline bg-[#1b202a] p-5 shadow-2xl outline-none">
        <div className="flex items-center justify-between"><h2 id="text-engine-title" className="text-base font-semibold">Text inference engine</h2><button type="button" onClick={() => setOpen(false)} aria-label="Close text engine settings" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button></div>
        <p className="muted text-xs">The selected engine is used for new text jobs. Jobs already queued keep their settings.</p>
        {draft && <><label className="block"><span className="label">Provider</span><select className="field" value={draft.text_provider} onChange={event => setDraft({ ...draft, text_provider: event.target.value as TextSettings['text_provider'] })}><option value="local">Local Qwen 3.6 35B Q4_K_M</option><option value="openrouter">OpenRouter</option></select></label>
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={draft.local_thinking} disabled={draft.text_provider !== 'local'} onChange={event => setDraft({ ...draft, local_thinking: event.target.checked })} /> Enable local model thinking</label>
        <label className="block"><span className="label">OpenRouter model</span><input className="field" maxLength={200} value={draft.openrouter_model} onChange={event => setDraft({ ...draft, openrouter_model: event.target.value })} /><span className="muted mt-1 block text-[11px]">OpenRouter requests send project and cut context, transcripts, and analysis evidence to the selected provider and may incur charges.</span></label></>}
        <div className="rounded-lg border hairline p-3 text-xs"><div className="flex flex-wrap items-center justify-between gap-2"><div><strong>Local model</strong><p role="status" className="muted mt-1">{status ? labels[status.state] || status.state : 'Checking status…'}{status?.desired_loaded && status.state === 'unloaded' ? ' · Waiting for visual work to finish' : ''}{status?.detail ? ` · ${status.detail}` : ''}</p></div><button type="button" className="btn btn-quiet" disabled={!status || working || status.state === 'worker_offline'} onClick={() => void toggle()}>{status?.desired_loaded ? 'Unload when idle' : 'Keep model loaded'}</button></div><p className="muted mt-2 text-[11px]">The model is released for visual analysis. A queued local text job loads it when needed.</p></div>
        {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
        <div className="flex justify-end gap-2"><button className="btn btn-quiet" onClick={() => setOpen(false)}>Cancel</button><button className="btn btn-primary" disabled={!draft || working || !draft.openrouter_model.trim()} onClick={() => void save()}>Save engine</button></div>
      </div>
    </div>}
  </>
}
