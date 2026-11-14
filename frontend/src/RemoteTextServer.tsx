import { useEffect, useState } from 'react'
import { api } from './api'
import { REMOTE_STATE_LABELS, TEXT_JOB_TYPES, defaultModel, jobSlots, modelChoices, shownRoute, parseRouteValue, routeOptions, routeValue, type RemoteModel, type RemoteServer, type TextRoute } from './remoteText'

type Draft = Omit<RemoteServer, 'has_token' | 'state' | 'effective_routes' | 'local_slots'> & { token?: string }
type Probe = { state: string; detail: string; models: RemoteModel[] }

const toDraft = (server: RemoteServer): Draft => ({
  enabled: server.enabled, base_url: server.base_url, parallel: server.parallel,
  routes: { ...server.routes }, assistant_fallback: server.assistant_fallback, prefer: server.prefer, chat_reserve: server.chat_reserve,
})

const contextLabel = (value?: number | null) => value ? `${Math.round(value / 1024)}k context per slot` : 'context unknown'

/** Remote llama.cpp settings and job routing, saved separately from the provider settings. */
export function RemoteTextServer({ server, onSaved }: { server: RemoteServer | null; onSaved: (value: RemoteServer) => void }) {
  const [draft, setDraft] = useState<Draft | null>(server ? toDraft(server) : null)
  const [probe, setProbe] = useState<Probe | null>(null)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [clearToken, setClearToken] = useState(false)
  useEffect(() => { if (server && !draft) setDraft(toDraft(server)) }, [server, draft])
  if (!server || !draft) return <div className="rounded-lg border hairline p-3 text-xs muted">Loading remote server settings…</div>

  const models = probe?.models.length ? probe.models : server.state.models
  const body = () => JSON.stringify({ ...draft, token: clearToken ? '' : draft.token || undefined })  // omitted keeps the saved token
  const update = (change: Partial<Draft>) => { setDraft({ ...draft, ...change }); setSaved(false) }
  const setRoute = (pipeline: string, route: TextRoute) => update({ routes: { ...draft.routes, [pipeline]: route } })

  async function test() {
    setWorking(true); setError(''); setProbe(null)
    try { setProbe(await api<Probe>('/remote-text-server/test', { method: 'POST', body: body() })) }
    catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }
  async function save() {
    setWorking(true); setError('')
    try { const value = await api<RemoteServer>('/remote-text-server', { method: 'PUT', body: body() }); onSaved(value); setDraft(toDraft(value)); setClearToken(false); setSaved(true) }
    catch (err) { setError((err as Error).message) } finally { setWorking(false) }
  }

  const state = server.state
  const status = server.enabled ? `${REMOTE_STATE_LABELS[state.state] || state.state}${state.detail ? ` · ${state.detail}` : ''}${state.state === 'online' ? ` · ${state.active_jobs} running` : ''}` : 'Off'
  const fallbackChoices = modelChoices(models, [draft.assistant_fallback])
  const fallbackModel = defaultModel(draft.assistant_fallback, models)
  const sameForAll = () => { if (fallbackModel) update({ routes: Object.fromEntries(TEXT_JOB_TYPES.map(({ pipeline }) => [pipeline, { mode: 'any', model: fallbackModel }])) }) }
  return <div className="space-y-3 rounded-lg border hairline p-3 text-xs">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div><strong>Remote llama.cpp server</strong><p role="status" className="muted mt-1">{status}</p></div>
      <label className="flex items-center gap-2"><input type="checkbox" checked={draft.enabled} onChange={event => update({ enabled: event.target.checked })} /> Use for text jobs</label>
    </div>
    <div className="grid gap-2 sm:grid-cols-[1fr_7rem]">
      <label className="block"><span className="label">Address</span><input className="field" placeholder="http://192.168.0.20:8080" maxLength={500} value={draft.base_url} onChange={event => update({ base_url: event.target.value })} /></label>
      <label className="block"><span className="label">Parallel jobs</span><input className="field" type="number" min={1} max={8} value={draft.parallel} onChange={event => update({ parallel: Math.max(1, Math.min(8, Number(event.target.value) || 1)) })} /></label>
      <label className="block sm:col-span-2"><span className="label">Access token (Bearer)</span><input className="field" type="password" autoComplete="new-password" maxLength={500} placeholder={server.has_token ? 'Saved · type to replace' : 'Optional · the server\'s --api-key'} value={draft.token ?? ''} onChange={event => { setClearToken(false); update({ token: event.target.value }) }} /></label>
      {server.has_token && <label className="flex items-center gap-2 sm:col-span-2"><input type="checkbox" checked={clearToken} onChange={event => { setClearToken(event.target.checked); update({ token: '' }) }} /> Remove the saved token</label>}
    </div>
    <p className="muted text-[11px]">The PC has no host name, so enter its current IP here when it changes. Parallel jobs should match the remote <code>--parallel</code>. The token is sent as a Bearer header when set; leave it empty to keep the saved one.</p>
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className="btn btn-quiet" disabled={working || !draft.base_url.trim()} onClick={() => void test()}>Test connection</button>
      {probe && <span className={probe.state === 'online' ? 'text-[#9fe0b4]' : 'text-[#ff9fa7]'}>{REMOTE_STATE_LABELS[probe.state] || probe.state}{probe.detail ? ` · ${probe.detail}` : ''}</span>}
    </div>
    {models.length > 0 && <ul className="muted space-y-0.5 text-[11px]">{models.map(model => <li key={model.id}><span className="text-[#dfe3ea]">{model.id}</span> · {contextLabel(model.n_ctx)}</li>)}</ul>}
    <div>
      <span className="label">Job routing</span>
      <div className="mt-1 space-y-1.5">{TEXT_JOB_TYPES.map(({ pipeline, label }) => <label key={pipeline} className="grid items-center gap-2 sm:grid-cols-[10rem_1fr]">
        <span>{label}</span>
        <select className="field !py-1.5 !text-xs" value={routeValue(shownRoute(draft.routes[pipeline], fallbackModel))} onChange={event => setRoute(pipeline, parseRouteValue(event.target.value))}>
          {routeOptions(models, shownRoute(draft.routes[pipeline], fallbackModel)).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select></label>)}</div>
      <button type="button" className="btn btn-quiet mt-2" disabled={!fallbackModel} onClick={sameForAll}>Share every job type with the remote ({fallbackModel || 'no model yet'})</button>
      <p className="muted mt-1 text-[11px]">A job type without a choice is shared with the remote automatically. Applies to jobs queued with the Local provider. Routing is read when a job starts, so queued jobs follow changes. Remote-only jobs wait while the remote is offline; a job whose server stops answering goes back to the queue.</p>
    </div>
    <div className="grid gap-2 sm:grid-cols-2">
      <label className="block"><span className="label">When both servers are free, prefer</span>
        <select className="field !py-1.5 !text-xs" value={draft.prefer} onChange={event => update({ prefer: event.target.value as 'local' | 'remote' })}>
          <option value="local">This host</option><option value="remote">The remote PC</option>
        </select></label>
      <label className="block"><span className="label">Slots kept free for chat on each server</span>
        <input className="field" type="number" min={0} max={8} value={draft.chat_reserve} onChange={event => update({ chat_reserve: Math.max(0, Math.min(8, Number(event.target.value) || 0)) })} /></label>
    </div>
    <p className="muted text-[11px]">Chat comes first. Text jobs stop starting while a chat message waits and never use the reserved slots, so a reply never queues behind a running job. This host has {server.local_slots} slot{server.local_slots === 1 ? '' : 's'} (<code>TEXT_JOB_CONCURRENCY</code>), so jobs use {jobSlots(server.local_slots, draft.chat_reserve)} here and {jobSlots(draft.parallel, draft.chat_reserve)} on the remote. Set <code>TEXT_JOB_CONCURRENCY</code> one higher than before to keep job speed.</p>
    <label className="block"><span className="label">Remote model for chat and shared jobs</span>
      <select className="field !py-1.5 !text-xs" value={draft.assistant_fallback} onChange={event => update({ assistant_fallback: event.target.value })}>
        <option value="">First model the server lists</option>
        {fallbackChoices.map(({ id, listed }) => <option key={id} value={id}>Remote · {id}{listed ? '' : ' (not reported)'}</option>)}
      </select>
      <span className="muted mt-1 block text-[11px]">Cut Assistance's Auto provider answers on whichever server is free, using this model on the remote. It also answers a chat that allows a GPU-busy fallback; OpenRouter free does only when the remote is offline.</span></label>
    {error && <p role="alert" className="text-[#ff9fa7]">{error}</p>}
    <div className="flex items-center justify-end gap-2">{saved && <span className="muted">Saved</span>}<button type="button" className="btn btn-quiet" disabled={working} onClick={() => void save()}>Save remote server</button></div>
  </div>
}
