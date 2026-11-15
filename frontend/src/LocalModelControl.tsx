import { useEffect, useRef, useState } from 'react'
import { Cpu, X } from 'lucide-react'
import { api } from './api'
import { RemoteTextServer } from './RemoteTextServer'
import { remoteBadge, variantLabel, type RemoteServer } from './remoteText'

type Variant = 'standard' | 'mtp'
type VariantInfo = { available: boolean; detail: string; label: string }
type ModelStatus = { desired_loaded: boolean; state: string; detail: string; variant?: string; variants?: Partial<Record<Variant, VariantInfo>> }
type TextSettings = { text_provider: 'local' | 'openrouter'; local_thinking: boolean; openrouter_model: string; local_model?: Variant }
type Comparison = { variant: string; results: number; avg_completion_tokens: number; avg_generation_seconds: number | null; overall_tokens_per_second: number | null; decode_tokens_per_second: number | null; draft_acceptance: number | null }
const variantNames: Record<string, string> = { standard: 'Standard Q4_K_M', mtp: 'MTP (speculative)' }
const fixed = (value: number | null, suffix = '') => value === null ? '–' : `${value}${suffix}`
const labels: Record<string, string> = { unloaded: 'Unloaded', downloading: 'Downloading', loading: 'Loading', loaded: 'Loaded', generating: 'Generating', unloading: 'Unloading', worker_offline: 'Worker offline', error: 'Error' }

/** The text engine button and its settings dialog. `compact` matches the episode editor's smaller top bar. */
export function LocalModelControl({ compact = false }: { compact?: boolean } = {}) {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [settings, setSettings] = useState<TextSettings | null>(null)
  const [draft, setDraft] = useState<TextSettings | null>(null)
  const [open, setOpen] = useState(false)
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const [comparison, setComparison] = useState<Comparison[]>([])
  const [remote, setRemote] = useState<RemoteServer | null>(null)
  const panel = useRef<HTMLDivElement>(null)
  async function refresh() {
    try { setStatus(await api<ModelStatus>('/local-text-model')) } catch (err) { setError((err as Error).message) }
    try { setRemote(await api<RemoteServer>('/remote-text-server')) } catch { /* an older API has no remote server */ }
  }
  useEffect(() => { void refresh(); void api<TextSettings>('/text-generation-settings').then(setSettings).catch(err => setError((err as Error).message)); const timer = setInterval(() => void refresh(), 2500); return () => clearInterval(timer) }, [])
  useEffect(() => {
    if (!open) return
    const opener = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', key)
    void api<{ variants: Comparison[] }>('/text-model-comparison').then(value => setComparison(value.variants)).catch(() => setComparison([]))
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
    <button type="button" className={compact ? 'icon-btn icon-btn-md icon-btn-text whitespace-nowrap' : 'btn btn-quiet whitespace-nowrap'} onClick={() => { setDraft(settings); setError(''); setOpen(true); void api<TextSettings>('/text-generation-settings').then(value => { setSettings(value); setDraft(value) }).catch(err => setError((err as Error).message)) }} title="Text engine settings"><Cpu size={15} /><span className="hidden sm:inline">{settings?.text_provider === 'local' ? 'Local' : 'OpenRouter'} · </span><span>Local {status ? labels[status.state] || status.state : 'checking…'}{status?.variant === 'mtp' ? ' · MTP' : ''}</span><span className="hidden md:inline">{remoteBadge(remote)}</span></button>
    {open && <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4" onMouseDown={event => { if (event.target === event.currentTarget) setOpen(false) }}>
      <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="text-engine-title" className="max-h-[92vh] w-full max-w-2xl space-y-4 overflow-y-auto rounded-xl border hairline bg-[#1b202a] p-5 shadow-2xl outline-none">
        <div className="flex items-center justify-between"><h2 id="text-engine-title" className="text-base font-semibold">Text inference engine</h2><button type="button" onClick={() => setOpen(false)} aria-label="Close text engine settings" className="rounded-md p-1.5 text-[#aeb5c1] hover:bg-[#303746]"><X size={18} /></button></div>
        <p className="muted text-xs">The selected engine is used for new text jobs. Jobs already queued keep their settings.</p>
        {draft && <><label className="block"><span className="label">Provider</span><select className="field" value={draft.text_provider} onChange={event => setDraft({ ...draft, text_provider: event.target.value as TextSettings['text_provider'] })}><option value="local">Local Qwen 3.6 35B Q4_K_M</option><option value="openrouter">OpenRouter</option></select></label>
        <label className="block"><span className="label">Local model</span><select className="field" disabled={draft.text_provider !== 'local'} value={draft.local_model ?? 'standard'} onChange={event => setDraft({ ...draft, local_model: event.target.value as Variant })}>
          {(['standard', 'mtp'] as Variant[]).map(name => { const info = status?.variants?.[name]; const blocked = name !== 'standard' && !info?.available && draft.local_model !== name
            return <option key={name} value={name} disabled={blocked}>{variantNames[name]}{info && name !== 'standard' ? ` · ${info.detail}` : ''}</option> })}
        </select><span className="muted mt-1 block text-[11px]">{status?.variant ? `Loaded now: ${variantNames[status.variant] || status.variant}. ` : ''}Switching restarts the model when no text job or chat is running; a running job finishes on the model it started with. Every result records the model, decode speed and, for MTP, how many drafted tokens were accepted.</span></label>
        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={draft.local_thinking} disabled={draft.text_provider !== 'local'} onChange={event => setDraft({ ...draft, local_thinking: event.target.checked })} /> Enable local model thinking</label>
        <label className="block"><span className="label">OpenRouter model</span><input className="field" maxLength={200} value={draft.openrouter_model} onChange={event => setDraft({ ...draft, openrouter_model: event.target.value })} /><span className="muted mt-1 block text-[11px]">OpenRouter requests send project and cut context, transcripts, and analysis evidence to the selected provider and may incur charges.</span></label></>}
        <div className="rounded-lg border hairline p-3 text-xs"><div className="flex flex-wrap items-center justify-between gap-2"><div><strong>Local model</strong><p role="status" className="muted mt-1">{status ? labels[status.state] || status.state : 'Checking status…'}{status?.desired_loaded && status.state === 'unloaded' ? ' · Waiting for visual work to finish' : ''}{status?.detail ? ` · ${status.detail}` : ''}</p></div><button type="button" className="btn btn-quiet" disabled={!status || working || status.state === 'worker_offline'} onClick={() => void toggle()}>{status?.desired_loaded ? 'Unload when idle' : 'Keep model loaded'}</button></div><p className="muted mt-2 text-[11px]">The model is released for visual analysis. A queued local text job loads it when needed.</p></div>
        <RemoteTextServer server={remote} onSaved={setRemote} />
        {comparison.length > 0 && <div className="rounded-lg border hairline p-3 text-xs"><strong>Speed by model</strong>
          <table className="mt-2 w-full text-left"><thead className="muted"><tr><th className="font-normal">Model</th><th className="font-normal">Results</th><th className="font-normal" title="Completion tokens over the whole generation time">tok/s overall</th><th className="font-normal" title="llama-server decode speed, recorded since this option exists">tok/s decode</th><th className="font-normal" title="Share of drafted tokens the model accepted">Draft hit</th><th className="font-normal">Avg tokens</th></tr></thead>
            <tbody>{comparison.map(row => <tr key={row.variant}><td>{variantLabel(row.variant, variantNames)}</td><td>{row.results}</td><td>{fixed(row.overall_tokens_per_second)}</td><td>{fixed(row.decode_tokens_per_second)}</td><td>{row.draft_acceptance === null ? '–' : `${Math.round(row.draft_acceptance * 100)}%`}</td><td>{row.avg_completion_tokens}</td></tr>)}</tbody></table>
          <p className="muted mt-2 text-[11px]">Regenerate the same Cut summary under each model to compare the text; both revisions stay in the result history.</p></div>}
        {error && <p role="alert" className="text-xs text-[#ff9fa7]">{error}</p>}
        <div className="flex justify-end gap-2"><button className="btn btn-quiet" onClick={() => setOpen(false)}>Cancel</button><button className="btn btn-primary" disabled={!draft || working || !draft.openrouter_model.trim()} onClick={() => void save()}>Save engine</button></div>
      </div>
    </div>}
  </>
}
