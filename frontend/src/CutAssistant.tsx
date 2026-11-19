import { useEpisodeEvents, useEpisodeRevision } from './episodeEvents'
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, ArrowDown, ArrowUp, Brain, Check, ChevronRight, Cloud, Cpu, Eye, Gauge, Layers, Loader2, MessageSquarePlus, Pencil, RefreshCw, Send, Sparkles, Square, Timer, Trash2, Wrench, X, Zap } from 'lucide-react'
import type { Cut } from './api'
import { assistantApi, pictureUrl, streamUrl, type AssistantConfig, type AssistantScope, type AssistantProposal, type AssistantSession, type AssistantTurn, type AssistantUsage, type LocalModelStatus, type ProviderInfo, type SessionDetail, type SkillInfo } from './assistantApi'
import { ACTIVE_STATUSES, activeTurn, applyStreamEvent, contextShare, formatSeconds, formatTokens, groupTurns, prettyArguments, prettyResult, reasoningTokens, starterText, turnElapsed, type Step, type ToolStep, type TurnView } from './assistantFormat'
import { DiffText } from './DiffText'
import { CopyButton, Markdown } from './Markdown'

const STREAM_EVENTS = ['snapshot', 'delta', 'message', 'proposal', 'turn', 'done', 'gone'] as const
const MODEL_STATE: Record<string, string> = { unloaded: 'Unloaded', downloading: 'Downloading', loading: 'Loading', loaded: 'Loaded', generating: 'Generating', unloading: 'Unloading', worker_offline: 'Worker offline', error: 'Error' }
// Cut chats keep their historic key (the bare cut id); sequence chats are prefixed.
const storageKey = (owner: string) => `framewise:assistant:${owner}`
const readStored = (cutId: string) => { try { return localStorage.getItem(storageKey(cutId)) } catch { return null } }
const writeStored = (cutId: string, value: string | null) => { try { if (value) localStorage.setItem(storageKey(cutId), value); else localStorage.removeItem(storageKey(cutId)) } catch { /* storage may be unavailable */ } }
// On by default: a busy GPU should not stop the conversation. Remembered per browser, not per cut.
const FALLBACK_KEY = 'framewise:assistant:gpu-fallback'
const readFallback = () => { try { return localStorage.getItem(FALLBACK_KEY) !== '0' } catch { return true } }
const writeFallback = (value: boolean) => { try { localStorage.setItem(FALLBACK_KEY, value ? '1' : '0') } catch { /* storage may be unavailable */ } }

function relativeTime(value: string) {
  const time = Date.parse(value.includes('T') ? value : value.replace(' ', 'T') + 'Z')
  if (!Number.isFinite(time)) return ''
  const seconds = Math.max(0, (Date.now() - time) / 1000)
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return new Date(time).toLocaleDateString()
}

type Choice = { provider: string; model: string; thinking: boolean }

function defaultModel(provider: ProviderInfo | undefined, config: AssistantConfig | null) {
  if (!provider) return ''
  if (provider.models.length && !provider.free_model) return provider.models[0].id
  if (provider.id === 'openrouter') return config?.defaults.openrouter_model || 'openrouter/free'
  return provider.models[0]?.id || ''
}

/** A sequence the chat is about, instead of a cut: its id and the line shown under the chat title. */
export type SequenceChat = { id: string; label: string }

export default function CutAssistant({ cut, sequence, active, onResultsChanged }: { cut?: Cut; sequence?: SequenceChat; active: boolean; onResultsChanged: () => void }) {
  const scopeKind = sequence ? 'sequence' as const : 'cut' as const
  const scopeId = sequence ? sequence.id : cut?.id || ''
  const scope = useMemo<AssistantScope>(() => ({ kind: scopeKind, id: scopeId }), [scopeKind, scopeId])
  const owner = sequence ? `sequence:${sequence.id}` : scopeId
  const [config, setConfig] = useState<AssistantConfig | null>(null)
  const [sessions, setSessions] = useState<AssistantSession[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [choice, setChoice] = useState<Choice | null>(null)
  const [gpuFallback, setGpuFallback] = useState(readFallback)
  const [draft, setDraft] = useState('')
  const [skill, setSkill] = useState<SkillInfo | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [localStatus, setLocalStatus] = useState<LocalModelStatus | null>(null)
  const [modelWorking, setModelWorking] = useState(false)
  const [now, setNow] = useState(() => Date.now())
  const scroller = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const composer = useRef<HTMLTextAreaElement>(null)

  const refreshSessions = useCallback(async () => {
    const list = await assistantApi.sessions(scope)
    setSessions(list)
    return list
  }, [scope])

  // Initial load: config, sessions, and the last opened session for this cut or sequence.
  useEffect(() => {
    let alive = true
    setLoading(true)
    Promise.all([assistantApi.config(), assistantApi.sessions(scope)]).then(([value, list]) => {
      if (!alive) return
      setConfig(value); setSessions(list)
      const remembered = readStored(owner)
      setSessionId(list.find(item => item.id === remembered)?.id || list[0]?.id || null)
    }).catch(err => alive && setError((err as Error).message)).finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [scope, owner])

  useEpisodeEvents(change => { if (active && change.queue) void assistantApi.config().then(setConfig).catch(() => {}) })

  useEffect(() => {
    writeStored(owner, sessionId)
    if (!sessionId) { setDetail(null); return }
    let alive = true
    assistantApi.session(sessionId).then(value => { if (alive) setDetail(value) }).catch(err => alive && setError((err as Error).message))
    return () => { alive = false }
  }, [sessionId, owner])

  // Model choice: the session's last choice, else the configured defaults.
  useEffect(() => {
    if (!config) return
    const providers = config.providers
    const session = detail?.session
    const fromSession = session && providers.some(item => item.id === session.provider) ? { provider: session.provider, model: session.model, thinking: session.thinking } : null
    if (fromSession?.model) { setChoice(fromSession); return }
    setChoice(current => {
      if (current && providers.some(item => item.id === current.provider)) return current
      const preferred = providers.find(item => item.id === config.defaults.provider && item.available) || providers.find(item => item.available) || providers[0]
      return preferred ? { provider: preferred.id, model: defaultModel(preferred, config), thinking: config.defaults.thinking } : null
    })
  }, [config, detail?.session])

  const provider = config?.providers.find(item => item.id === choice?.provider)
  const openrouter = config?.providers.find(item => item.id === 'openrouter')
  const sees = provider?.vision ?? provider?.models.find(item => item.id === choice?.model)?.vision  // a provider says it, or its model does (the remote PC)
  // The worker publishes which provider answers while the GPU is busy (remote llama.cpp or OpenRouter free).
  const fallbackTarget = provider?.gpu_fallback
  const remoteFallback = fallbackTarget?.provider === 'remote'
  const fallbackReady = remoteFallback || (!!openrouter?.available && !!openrouter.fallback_model)
  const useFallback = !!provider?.managed && gpuFallback
  const running = activeTurn(detail)
  const runningId = running?.id

  // Stream the active turn; the server sends a snapshot first, so reconnects resynchronise.
  useEffect(() => {
    if (!runningId) return
    const source = new EventSource(streamUrl(runningId))
    for (const name of STREAM_EVENTS) {
      source.addEventListener(name, event => {
        const data = JSON.parse((event as MessageEvent).data)
        if (name === 'gone') { source.close(); if (sessionId) void assistantApi.session(sessionId).then(setDetail).catch(() => {}); return }
        setDetail(current => current ? applyStreamEvent(current, name, data) : current)
        if (name === 'done') { source.close(); void refreshSessions().catch(() => {}) }
      })
    }
    return () => source.close()
  }, [runningId, sessionId, refreshSessions])

  useEffect(() => {
    if (!runningId) return
    const timer = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(timer)
  }, [runningId])

  // Local model state for the managed provider.
  const showLocal = active && !!provider?.managed
  const localRevision = useEpisodeRevision(['queue'], showLocal)
  useEffect(() => {
    if (!showLocal) return
    let alive = true
    const load = () => assistantApi.localModel().then(value => alive && setLocalStatus(value)).catch(() => {})
    void load()
    return () => { alive = false }
  }, [showLocal, localRevision])

  useEffect(() => {
    const element = scroller.current
    if (element && stick.current) element.scrollTop = element.scrollHeight
  }, [detail, active])

  useEffect(() => { if (!error) return; const timer = setTimeout(() => setError(''), 6000); return () => clearTimeout(timer) }, [error])

  const views = useMemo(() => detail ? groupTurns(detail) : [], [detail])
  const lastTurn = detail?.turns[detail.turns.length - 1]
  const lastFinished = [...(detail?.turns || [])].reverse().find(turn => typeof turn.usage.context_tokens === 'number')
  const sessionContext = contextShare(lastFinished?.usage, provider?.context_limit ?? provider?.models.find(item => item.id === choice?.model)?.context)
  const workerOffline = config ? !config.worker_online : false
  const localOffline = provider?.managed && localStatus?.state === 'worker_offline'
  const sendBlocker = !config ? 'Loading…' : workerOffline ? 'The inference worker is offline; start it to chat.' : !provider ? 'Choose a provider.' : !provider.available ? (provider.detail || `${provider.label} is unavailable.`) : !choice?.model.trim() ? 'Choose a model.' : localOffline ? 'The local model worker is not responding.' : ''
  const canSend = !sendBlocker && !running && !sending && !!draft.trim()

  async function newChat() {
    setSessionId(null); setDetail(null); setDraft(''); setSkill(null)
    composer.current?.focus()
  }

  async function send(text = draft, skillName = skill?.name ?? null) {
    const content = text.trim()
    if (!content || !choice || sendBlocker || running) return
    setSending(true); setError('')
    try {
      let id = sessionId
      let base = detail
      if (!id || !base) {
        base = await assistantApi.createSession(scope, choice)
        id = base.session.id
      }
      const result = await assistantApi.send(id, { content, provider: choice.provider, model: choice.model.trim(), thinking: choice.thinking, skill: skillName, gpu_fallback: useFallback })
      stick.current = true
      setDetail({ ...base, messages: [...base.messages, result.message], turns: [...base.turns, result.turn] })
      if (id !== sessionId) setSessionId(id)
      setDraft(''); setSkill(null)
      void refreshSessions().catch(() => {})
    } catch (err) { setError((err as Error).message) } finally { setSending(false) }
  }

  async function stop() {
    if (!running) return
    try { const turn = await assistantApi.cancel(running.id); setDetail(current => current ? applyStreamEvent(current, 'turn', turn) : current) }
    catch (err) { setError((err as Error).message) }
  }

  async function retry(turn: AssistantTurn) {
    if (!choice) return
    try { setDetail(await assistantApi.retry(turn.id, { provider: choice.provider, model: choice.model.trim(), thinking: choice.thinking, gpu_fallback: useFallback })); stick.current = true }
    catch (err) { setError((err as Error).message) }
  }

  async function removeSession(item: AssistantSession) {
    if (!window.confirm(`Delete the chat “${item.title || 'Untitled chat'}”? Its messages and pending proposals are removed; applied changes stay.`)) return
    try {
      await assistantApi.remove(item.id)
      const list = await refreshSessions()
      if (item.id === sessionId) setSessionId(list[0]?.id || null)
    } catch (err) { setError((err as Error).message) }
  }

  async function renameSession(item: AssistantSession) {
    const title = window.prompt('Rename chat', item.title)
    if (title === null) return
    try { await assistantApi.rename(item.id, title); await refreshSessions(); if (detail?.session.id === item.id) setDetail({ ...detail, session: { ...detail.session, title } }) }
    catch (err) { setError((err as Error).message) }
  }

  async function resolveProposal(proposal: AssistantProposal, action: 'apply' | 'discard') {
    try {
      const updated = action === 'apply' ? await assistantApi.apply(proposal.id) : await assistantApi.discard(proposal.id)
      setDetail(current => current ? applyStreamEvent(current, 'proposal', updated) : current)
      if (action === 'apply') {
        onResultsChanged()
        // The transcript lives with the episode, not the cut's results, so the editor reloads it separately.
        if (proposal.target === 'transcript') window.dispatchEvent(new Event('framewise:transcript-changed'))
      }
      void refreshSessions().catch(() => {})
    } catch (err) {
      setError((err as Error).message)
      if (sessionId) void assistantApi.session(sessionId).then(setDetail).catch(() => {})
    }
  }

  async function toggleLocalModel() {
    if (!localStatus) return
    setModelWorking(true)
    try { setLocalStatus(await assistantApi.setLocalModel(!localStatus.desired_loaded)) } catch (err) { setError((err as Error).message) } finally { setModelWorking(false) }
  }

  function pickStarter(item: SkillInfo, example: string) {
    setSkill(item)
    setDraft(starterText(item.name, example))
    requestAnimationFrame(() => { const field = composer.current; if (field) { field.focus(); field.setSelectionRange(field.value.length, field.value.length) } })
  }

  const activeTitle = sessions.find(item => item.id === sessionId)?.title || detail?.session.title || 'New chat'
  // Each chat offers the skills of its own kind: prompt fixing for a cut, direction editing for a sequence.
  const starters = (config?.skills || []).filter(item => (item.scope || 'cut') === scopeKind || item.scope === 'any').flatMap(item => item.examples.map(example => ({ item, example })))

  return <div className={active ? 'grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[250px_minmax(0,1fr)]' : 'hidden'}>
    <aside className="hidden min-h-0 flex-col border-r hairline bg-[#11151d] lg:flex" aria-label="Chat history">
      <div className="p-3"><button type="button" className="btn btn-quiet w-full" onClick={() => void newChat()}><MessageSquarePlus size={15} /> New chat</button></div>
      <p className="label px-3">History · {sessions.length}</p>
      <div className="scrollbar min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-3">
        {loading && <p className="muted px-2 text-xs">Loading chats…</p>}
        {!loading && !sessions.length && <p className="muted px-2 text-xs leading-5">No chats for this {scopeKind} yet. Ask a question to start one.</p>}
        {sessions.map(item => <SessionItem key={item.id} item={item} selected={item.id === sessionId} onSelect={() => { stick.current = true; setSessionId(item.id) }} onRename={() => void renameSession(item)} onDelete={() => void removeSession(item)} />)}
      </div>
      {cut && !sequence && <div className="border-t hairline p-3"><p className="label">Cut frames</p><div className="grid grid-cols-2 gap-2">{(['first', 'last'] as const).map(position => <a key={position} href={`/api/cuts/${cut.id}/frame/${position}`} target="_blank" rel="noreferrer" className="overflow-hidden rounded-md border border-[#343b49] hover:border-[#a58ae8]"><img src={`/api/cuts/${cut.id}/frame/${position}`} alt={`${position} frame`} className="aspect-video w-full object-cover" loading="lazy" /></a>)}</div></div>}
    </aside>
    <section className="flex min-h-0 flex-col" aria-label="Cut Assistance chat">
      <header className="flex flex-wrap items-center gap-2 border-b hairline px-4 py-2.5">
        <div className="mr-auto min-w-0"><p className="truncate text-sm font-semibold">{activeTitle}</p><p className="muted text-[11px]">{sequence ? `${sequence.label} · tools read this sequence's direction and the analysis of its cuts; edits need your approval` : `${cut?.active ? `Cut ${cut.cut_index + 1}` : 'Archived cut'} · tools read this cut's saved analysis; edits need your approval`}</p></div>
        <select className="field !w-auto !py-1.5 !text-xs" aria-label="Provider" value={choice?.provider || ''} disabled={!config} onChange={event => { const next = config?.providers.find(item => item.id === event.target.value); if (next && choice) setChoice({ ...choice, provider: next.id, model: defaultModel(next, config) }) }}>
          {(config?.providers || []).map(item => <option key={item.id} value={item.id} disabled={!item.available}>{item.label}{item.available ? '' : ' (unavailable)'}</option>)}
        </select>
        {provider && (provider.models.length && !provider.free_model
          ? <select className="field !w-auto max-w-56 !py-1.5 !text-xs" aria-label="Model" value={choice?.model || ''} onChange={event => choice && setChoice({ ...choice, model: event.target.value })}>{provider.models.map(model => <option key={model.id} value={model.id}>{model.label || model.id}</option>)}</select>
          : <><input className="field !w-56 !py-1.5 !text-xs" aria-label="Model" list={`assistant-models-${owner}`} maxLength={200} value={choice?.model || ''} onChange={event => choice && setChoice({ ...choice, model: event.target.value })} placeholder="provider/model" /><datalist id={`assistant-models-${owner}`}>{[config?.defaults.openrouter_model, ...(config?.recent_models || [])].filter((value, index, list): value is string => !!value && list.indexOf(value) === index).map(value => <option key={value} value={value} />)}</datalist></>)}
        <label className="flex items-center gap-1.5 rounded-lg border border-[#303640] px-2.5 py-1.5 text-xs" title="Show the model's reasoning. Uses more tokens and time."><input type="checkbox" checked={!!choice?.thinking} disabled={!provider?.thinking} onChange={event => choice && setChoice({ ...choice, thinking: event.target.checked })} /><Brain size={13} /> Thinking</label>
        {sees !== undefined && <span className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs ${sees ? 'border-[#2f5a49] text-[#9ff0c8]' : 'border-[#303640] text-[#8b93a3]'}`} title={sees ? 'This model can look at the video: it can open frames, keyframes and zoomed details while it works (view_frames).' : provider?.vision_detail || 'This model cannot look at pictures.'}><Eye size={13} /> {sees ? 'Vision' : 'No vision'}</span>}
        {provider?.managed && <LocalModelBadge status={localStatus} working={modelWorking} busy={!!provider.busy} onToggle={() => void toggleLocalModel()} />}
        {provider?.managed && <label className={`flex items-center gap-1.5 rounded-lg border border-[#303640] px-2.5 py-1.5 text-xs ${fallbackReady ? '' : 'opacity-60'}`} title={remoteFallback ? `When visual analysis is using the GPU, answer with ${fallbackTarget?.label} instead of waiting.` : fallbackReady ? 'When visual analysis is using the GPU, answer with OpenRouter free instead of waiting. OpenRouter then receives this cut\'s analysis.' : openrouter?.available ? 'Restart the inference worker to enable the OpenRouter fallback. Until then replies wait for the GPU.' : openrouter?.detail || 'OpenRouter is not available in the inference worker. Replies wait for the GPU. Set OPENROUTER_API_KEY and restart the worker.'}><input type="checkbox" checked={gpuFallback && fallbackReady} disabled={!fallbackReady} onChange={event => { setGpuFallback(event.target.checked); writeFallback(event.target.checked) }} /><Cloud size={13} /> {remoteFallback ? 'Remote' : 'OpenRouter'} if GPU busy</label>}
        {sessionContext && <ContextMeter used={sessionContext.used} limit={sessionContext.limit} percent={sessionContext.percent} />}
      </header>
      {(workerOffline || (provider && !provider.available)) && <div role="alert" className="flex items-center gap-2 border-b border-[#5b4a2a] bg-[#2a2216] px-4 py-2 text-xs text-[#ffd59a]"><AlertTriangle size={14} /> {workerOffline ? 'The inference worker is offline. Chat replies run there; start it to continue.' : provider?.detail || `${provider?.label} is unavailable.`}</div>}
      <div ref={scroller} onScroll={event => { const element = event.currentTarget; stick.current = element.scrollHeight - element.scrollTop - element.clientHeight < 140 }} className="scrollbar min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="mx-auto flex max-w-4xl flex-col gap-5">
          {!views.length && <EmptyState starters={starters} sequence={!!sequence} disabled={!!sendBlocker} onPick={pickStarter} />}
          {views.map(view => <TurnBlock key={view.id} view={view} now={now} latest={view.turn?.id === lastTurn?.id} onRetry={retry} onResolve={resolveProposal} />)}
        </div>
      </div>
      <footer className="border-t hairline bg-[#11151d] px-4 py-3">
        <div className="mx-auto max-w-4xl">
          {views.length > 0 && !running && !draft && <div className="scrollbar mb-2 flex gap-1.5 overflow-x-auto pb-1">{starters.map(({ item, example }) => <button key={example} type="button" className="shrink-0 rounded-full border border-[#353c48] px-2.5 py-1 !text-[11px] text-[#c9cfda] hover:border-[#a58ae8] hover:text-white" onClick={() => pickStarter(item, example)} title={item.description}>{example}</button>)}</div>}
          {skill && <div className="mb-2 inline-flex items-center gap-1.5 rounded-full bg-[#2b2442] px-2.5 py-1 text-[11px] text-[#d6c8ff]"><Sparkles size={12} /> Skill: {skill.title}<button type="button" aria-label="Remove skill" onClick={() => setSkill(null)} className="rounded-full p-0.5 hover:bg-[#3b3260]"><X size={11} /></button></div>}
          <div className="flex items-end gap-2">
            <textarea ref={composer} aria-label="Message Cut Assistance" className="field scrollbar max-h-56 min-h-[44px] flex-1 resize-none !text-sm leading-6" rows={Math.min(8, Math.max(1, draft.split('\n').length))} maxLength={20000} value={draft}
              placeholder={running ? 'Wait for the reply, or stop it…' : sequence ? 'Ask about this sequence, or request a change to a shot, the plan or the bible…' : 'Ask about this cut, paste a prompt to fix, or request a correction…'}
              onChange={event => setDraft(event.target.value)}
              onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (canSend) void send() } }} />
            {running
              ? <button type="button" className="btn btn-danger h-[44px]" onClick={() => void stop()} disabled={running.cancel_requested} title="Stop generating"><Square size={14} /> {running.cancel_requested ? 'Stopping…' : 'Stop'}</button>
              : <button type="button" className="btn btn-primary h-[44px]" onClick={() => void send()} disabled={!canSend} title={sendBlocker || 'Send (Enter)'}>{sending ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />} Send</button>}
          </div>
          <p className="muted mt-1.5 text-[11px]">{sendBlocker && !running ? sendBlocker : 'Enter to send · Shift+Enter for a new line · proposals change saved results only when you apply them'}{provider?.id === 'openrouter' ? ' · OpenRouter receives this cut\'s analysis and may charge for usage' : ''}{useFallback && fallbackReady && !remoteFallback ? ' · if the GPU is busy, OpenRouter free receives this cut\'s analysis' : ''}</p>
        </div>
      </footer>
      {error && <div role="alert" className="fixed bottom-6 left-1/2 z-[60] flex max-w-xl -translate-x-1/2 items-start gap-3 rounded-xl border border-[#63363e] bg-[#2b191f] px-4 py-3 text-sm text-[#ffb6bc] shadow-lg">{error}<button onClick={() => setError('')} aria-label="Dismiss"><X size={16} /></button></div>}
    </section>
  </div>
}

function SessionItem({ item, selected, onSelect, onRename, onDelete }: { item: AssistantSession; selected: boolean; onSelect: () => void; onRename: () => void; onDelete: () => void }) {
  const live = item.last_status === 'running' || item.last_status === 'queued'
  return <div className={`group flex items-start gap-1 rounded-lg px-2 py-2 ${selected ? 'bg-[#2a2440]' : 'hover:bg-[#1b202a]'}`}>
    <button type="button" className="min-w-0 flex-1 text-left" onClick={onSelect} aria-current={selected}>
      <span className="flex items-center gap-1.5"><span className={`status-dot shrink-0 ${live ? 'status-processing animate-pulse' : item.last_status === 'failed' ? 'status-failed' : item.last_status === 'done' ? 'status-ready' : ''}`} /><span className="truncate text-xs font-medium">{item.title || 'Untitled chat'}</span></span>
      <span className="muted mt-0.5 block truncate pl-3 text-[10px]">{item.provider === 'local' ? 'Local' : item.model || item.provider} · {relativeTime(item.updated_at)}{item.message_count ? ` · ${item.message_count} msg` : ''}</span>
      {!!item.pending_proposals && <span className="ml-3 mt-1 inline-block rounded bg-[#4a3a17] px-1.5 py-0.5 text-[10px] text-[#ffd59a]">{item.pending_proposals} pending</span>}
    </button>
    <span className="flex shrink-0 opacity-0 transition group-hover:opacity-100 group-focus-within:opacity-100">
      <button type="button" className="rounded p-1 text-[#aeb5c1] hover:bg-[#303746] hover:text-white" onClick={onRename} aria-label={`Rename ${item.title || 'chat'}`}><Pencil size={12} /></button>
      <button type="button" className="rounded p-1 text-[#aeb5c1] hover:bg-[#3d2025] hover:text-[#ffaaa7]" onClick={onDelete} aria-label={`Delete ${item.title || 'chat'}`}><Trash2 size={12} /></button>
    </span>
  </div>
}

function LocalModelBadge({ status, working, busy, onToggle }: { status: LocalModelStatus | null; working: boolean; busy: boolean; onToggle: () => void }) {
  const state = status?.state || 'checking'
  const tone = state === 'loaded' || state === 'generating' ? 'status-ready' : state === 'loading' || state === 'downloading' || state === 'unloading' ? 'status-processing' : state === 'error' || state === 'worker_offline' ? 'status-failed' : ''
  return <div className="flex items-center gap-1.5 rounded-lg border border-[#303640] py-1 pl-2.5 pr-1 text-xs" title={status?.detail || ''}>
    <Cpu size={13} /><span className={`status-dot ${tone}`} /><span>{status ? MODEL_STATE[state] || state : 'Checking…'}{busy ? ' · GPU busy' : status?.desired_loaded && state === 'unloaded' ? ' · waiting for GPU' : ''}</span>
    <button type="button" className="rounded-md px-2 py-0.5 !text-[11px] text-[#c3aeff] hover:bg-[#2b2442] disabled:opacity-40" disabled={!status || working || state === 'worker_offline'} onClick={onToggle}>{status?.desired_loaded ? 'Unload' : 'Load'}</button>
  </div>
}

function ContextMeter({ used, limit, percent }: { used: number; limit: number | null; percent: number | null }) {
  return <div className="flex items-center gap-2 text-[11px] text-[#aeb5c1]" title="Tokens in the model's context for the latest reply">
    <Layers size={13} /><span>{formatTokens(used)}{limit ? ` / ${formatTokens(limit)}` : ''}</span>
    {percent !== null && <span className="h-1.5 w-16 overflow-hidden rounded-full bg-[#262d39]"><span className={`block h-full ${percent > 80 ? 'bg-[#ff898c]' : percent > 55 ? 'bg-[#e8bd72]' : 'bg-[#a58ae8]'}`} style={{ width: `${Math.max(2, percent)}%` }} /></span>}
  </div>
}

function EmptyState({ starters, sequence, disabled, onPick }: { starters: { item: SkillInfo; example: string }[]; sequence: boolean; disabled: boolean; onPick: (item: SkillInfo, example: string) => void }) {
  return <div className="mx-auto mt-6 max-w-2xl text-center">
    <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-[#282438] text-[#bca7ff]"><Sparkles size={21} /></div>
    <h3 className="text-base font-semibold">Cut Assistance</h3>
    <p className="muted mx-auto mt-2 max-w-lg text-xs leading-5">{sequence ? "Chat about this sequence's direction. The assistant reads the bible, the shot plan, the shots and the analysis of their cuts, and proposes changes that you review before they are saved. A shot's prompt is rendered and checked again for every proposal." : "Chat about this cut's analysis. The assistant reads saved results through tools, loads skills for prompt work, and proposes corrections that you review before they are saved."}</p>
    <div className="mt-5 grid gap-2 text-left sm:grid-cols-2">
      {starters.map(({ item, example }) => <button key={example} type="button" disabled={disabled} onClick={() => onPick(item, example)} className="rounded-xl border border-[#2f3643] bg-[#161b24] p-3 hover:border-[#a58ae8] disabled:opacity-50">
        <span className="block text-xs font-semibold text-[#e8e2ff]">{example}</span><span className="muted mt-1 block text-[11px] leading-4">{item.description}</span>
      </button>)}
    </div>
  </div>
}

function TurnBlock({ view, now, latest, onRetry, onResolve }: { view: TurnView; now: number; latest: boolean; onRetry: (turn: AssistantTurn) => void; onResolve: (proposal: AssistantProposal, action: 'apply' | 'discard') => Promise<void> }) {
  const turn = view.turn
  const live = !!turn && ACTIVE_STATUSES.has(turn.status)
  const elapsed = turnElapsed(turn, now)
  return <div className="flex flex-col gap-3">
    {view.user && <div className="flex justify-end"><div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-[#2d2645] px-4 py-2.5 text-sm leading-6 text-[#f1edff]">{view.user.content}</div></div>}
    <div className="flex flex-col gap-2.5 rounded-2xl border border-[#262d39] bg-[#151a23] px-4 py-3">
      {view.steps.map((step, index) => <StepBlock key={step.message.id} step={step} live={live && index === view.steps.length - 1} />)}
      {view.proposals.map(proposal => <ProposalCard key={proposal.id} proposal={proposal} onResolve={onResolve} />)}
      {live && <p role="status" className="flex items-center gap-2 text-xs text-[#c3aeff]"><Loader2 size={13} className="animate-spin" /> {turn?.status === 'queued' ? 'Queued for the inference worker' : turn?.stage || 'Working'}{turn?.usage.fallback ? ` · via ${turn.usage.fallback.provider === 'remote' ? 'remote llama.cpp' : 'OpenRouter'} (GPU busy)` : ''}{elapsed !== null ? ` · ${formatSeconds(elapsed)}` : ''}</p>}
      {turn?.status === 'failed' && <div role="alert" className="flex flex-wrap items-start justify-between gap-2 rounded-lg border border-[#63363e] bg-[#2b191f] px-3 py-2 text-xs text-[#ffb6bc]"><span className="flex gap-2"><AlertTriangle size={14} className="mt-0.5 shrink-0" /> {turn.error || 'The reply failed.'}</span>{latest && <button type="button" className="btn btn-quiet !px-2.5 !py-1 !text-[11px]" onClick={() => onRetry(turn)}><RefreshCw size={12} /> Retry</button>}</div>}
      {turn?.status === 'cancelled' && <div className="flex items-center justify-between gap-2 text-xs text-[#aeb5c1]"><span>Stopped.</span>{latest && <button type="button" className="btn btn-quiet !px-2.5 !py-1 !text-[11px]" onClick={() => onRetry(turn)}><RefreshCw size={12} /> Regenerate</button>}</div>}
      {turn && !live && <TurnStats turn={turn} latest={latest} onRetry={onRetry} />}
    </div>
  </div>
}

function StepBlock({ step, live }: { step: Step; live: boolean }) {
  const { message } = step
  const thinkingNow = live && message.streaming && !message.content && !!message.reasoning
  const preloaded = step.tools.length > 0 && step.tools.every(tool => tool.result?.usage?.preloaded)
  return <div className="flex flex-col gap-2">
    {message.reasoning && <Reasoning text={message.reasoning} active={thinkingNow} streaming={live && message.streaming} />}
    {step.tools.length > 0 && <div className="flex flex-col gap-1.5">{step.tools.map(tool => <ToolItem key={tool.call.id} tool={tool} preloaded={preloaded} />)}</div>}
    {message.content && <Markdown text={message.content} className={live && message.streaming ? 'assistant-caret' : ''} />}
    {!message.content && !message.reasoning && !step.tools.length && live && message.streaming && <span className="assistant-caret text-sm" aria-hidden="true" />}
    {!live && !message.streaming && message.content && <div className="flex justify-end"><CopyButton text={message.content} label="Copy answer" /></div>}
  </div>
}

function Reasoning({ text, active, streaming }: { text: string; active: boolean; streaming: boolean }) {
  const [open, setOpen] = useState<boolean | null>(null)
  const expanded = open ?? active
  const body = useRef<HTMLDivElement>(null)
  useEffect(() => { if (expanded && streaming && body.current) body.current.scrollTop = body.current.scrollHeight }, [text, expanded, streaming])
  return <div className="rounded-lg border border-[#2b3240] bg-[#11151d]">
    <button type="button" className="flex w-full items-center gap-2 px-3 py-1.5 text-left !text-[11px] text-[#aeb5c1] hover:text-white" aria-expanded={expanded} onClick={() => setOpen(!expanded)}>
      <ChevronRight size={13} className={`transition ${expanded ? 'rotate-90' : ''}`} /><Brain size={13} className={active ? 'animate-pulse text-[#c3aeff]' : ''} />
      <span>{active ? 'Thinking…' : 'Reasoning'}</span><span className="muted ml-auto">{formatTokens(Math.round(text.length / 3.6))} tokens (approx.)</span>
    </button>
    {expanded && <div ref={body} className="scrollbar max-h-72 overflow-y-auto whitespace-pre-wrap border-t border-[#2b3240] px-3 py-2 text-[12px] leading-5 text-[#9ea6b5]">{text}</div>}
  </div>
}

function ToolItem({ tool, preloaded }: { tool: ToolStep; preloaded: boolean }) {
  const [open, setOpen] = useState(false)
  const result = tool.result
  const failed = result?.usage?.ok === false
  const args = prettyArguments(tool.call.arguments)
  const pictures = !failed && result ? result.usage?.images || [] : []
  return <div className={`rounded-lg border text-[11px] ${failed ? 'border-[#5b3439] bg-[#21161a]' : 'border-[#2b3240] bg-[#11151d]'}`}>
    <button type="button" className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[#c9cfda] hover:text-white" aria-expanded={open} onClick={() => setOpen(!open)}>
      <ChevronRight size={13} className={`shrink-0 transition ${open ? 'rotate-90' : ''}`} />
      {result ? failed ? <AlertTriangle size={13} className="shrink-0 text-[#ff9fa7]" /> : preloaded ? <Sparkles size={13} className="shrink-0 text-[#c3aeff]" /> : <Check size={13} className="shrink-0 text-[#7bdcb2]" /> : <Loader2 size={13} className="shrink-0 animate-spin text-[#c3aeff]" />}
      <span className="truncate">{result?.tool_summary || `Running ${tool.call.name}…`}</span>
      <span className="muted ml-auto flex shrink-0 items-center gap-1 font-mono"><Wrench size={11} /> {tool.call.name}{result ? ` · ${formatTokens(Math.round(result.content.length / 3.6))} tok` : ''}</span>
    </button>
    {pictures.length > 0 && result && <div className="flex flex-wrap gap-2 border-t border-[#2b3240] p-2" aria-label="Pictures the assistant looked at">
      {pictures.map((picture, index) => <a key={index} href={pictureUrl(result.id, index)} target="_blank" rel="noreferrer" className="block" title={`${picture.label} (opens full size)`}>
        <img src={pictureUrl(result.id, index)} alt={picture.label} loading="lazy" className="h-20 rounded border border-[#2b3240] bg-black object-contain" />
        <span className="mt-0.5 block max-w-[9rem] truncate text-[10px] text-[#9ea6b5]">{picture.label}</span>
      </a>)}
    </div>}
    {open && <div className="space-y-2 border-t border-[#2b3240] px-3 py-2">
      {args && <div><p className="label !mb-1 !text-[10px]">Arguments</p><pre className="scrollbar max-h-40 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-[#b8bfcc]">{args}</pre></div>}
      {result && <div><p className="label !mb-1 !text-[10px]">Result sent to the model</p><pre className="scrollbar max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-[#b8bfcc]">{prettyResult(result.content)}</pre></div>}
    </div>}
  </div>
}

const PROMPT_TARGETS = new Set(['h3_prompt', 'image_prompt'])

function ProposalCard({ proposal, onResolve }: { proposal: AssistantProposal; onResolve: (proposal: AssistantProposal, action: 'apply' | 'discard') => Promise<void> }) {
  const prompt = PROMPT_TARGETS.has(proposal.target)
  const [working, setWorking] = useState(false)
  // Prompts are mainly copied, so they open on the full text; result corrections open on the diff.
  const [full, setFull] = useState(prompt)
  const pending = proposal.status === 'pending'
  const badge = proposal.status === 'applied' ? [prompt ? 'Saved to cut' : 'Applied', 'bg-[#173a2b] text-[#9ff0c8]'] : proposal.status === 'discarded' ? [prompt ? 'Replaced or discarded' : 'Discarded', 'bg-[#262b35] text-[#aeb5c1]'] : proposal.stale ? ['Outdated', 'bg-[#4a3a17] text-[#ffd59a]'] : [prompt ? 'Not saved' : 'Pending approval', 'bg-[#2b2442] text-[#d6c8ff]']
  const changed = proposal.diff.some(segment => segment.op !== 'equal')
  async function act(action: 'apply' | 'discard') { setWorking(true); try { await onResolve(proposal, action) } finally { setWorking(false) } }
  return <div className={`rounded-xl border p-3 ${pending && !proposal.stale ? 'border-[#5a4a8c] bg-[#1a1730]' : 'border-[#2b3240] bg-[#11151d]'}`}>
    <div className="flex flex-wrap items-center gap-2">{prompt ? <Sparkles size={13} className="text-[#c3aeff]" /> : <Pencil size={13} className="text-[#c3aeff]" />}<span className="text-xs font-semibold">{prompt ? `Corrected ${proposal.label}` : `Proposed change · ${proposal.label}`}</span><span className={`rounded px-1.5 py-0.5 text-[10px] ${badge[1]}`}>{badge[0]}</span><span className="ml-auto"><CopyButton text={proposal.content} label={prompt ? 'Copy prompt' : 'Copy text'} /></span></div>
    {proposal.summary && <p className="mt-1.5 text-xs text-[#c9cfda]">{proposal.summary}</p>}
    <div className={`scrollbar mt-2 overflow-y-auto whitespace-pre-wrap rounded-lg border border-[#2b3240] bg-[#0f131a] p-2.5 font-mono text-[11px] leading-5 ${full ? 'max-h-[28rem] text-[#c9cfda]' : 'max-h-72 text-[#9ea6b5]'}`}>
      {full ? proposal.content : <DiffText segments={proposal.diff} />}
    </div>
    <div className="mt-2 flex flex-wrap items-center gap-2">
      {(changed || !full) && <button type="button" className="!text-[11px] text-[#c3aeff] underline" onClick={() => setFull(!full)}>{full ? `Show changes${proposal.diff_label ? ` vs ${proposal.diff_label}` : ''}` : prompt ? 'Show full prompt' : 'Show full text'}</button>}
      {proposal.stale && pending && <span className="text-[11px] text-[#ffd59a]">{proposal.target === 'transcript' ? 'The transcript' : 'The saved result'} changed since this was proposed. Ask again to rebase it.</span>}
      {pending && <span className="ml-auto flex gap-2">
        <button type="button" className="btn btn-quiet !px-3 !py-1.5 !text-xs" disabled={working} onClick={() => void act('discard')}><X size={13} /> Discard</button>
        <button type="button" className="btn btn-primary !px-3 !py-1.5 !text-xs" disabled={working || proposal.stale} onClick={() => void act('apply')} title={prompt ? `Save as this cut's ${proposal.label}; the previous version stays in history` : 'Save as a new revision'}>{working ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} />} {prompt ? 'Save to cut' : 'Apply'}</button>
      </span>}
      {proposal.status === 'applied' && <span className="ml-auto text-[11px] text-[#9ff0c8]">Saved as a new revision · previous version kept in history</span>}
    </div>
  </div>
}

function Stat({ icon, label, value, title }: { icon: ReactNode; label?: string; value: string; title?: string }) {
  return <span className="inline-flex items-center gap-1" title={title}>{icon}{label && <span className="muted">{label}</span>}<span className="text-[#d7dbe3]">{value}</span></span>
}

function TurnStats({ turn, latest, onRetry }: { turn: AssistantTurn; latest: boolean; onRetry: (turn: AssistantTurn) => void }) {
  const usage: AssistantUsage = turn.usage
  const thinking = reasoningTokens(usage)
  const context = contextShare(usage)
  if (!usage || (usage.seconds === undefined && usage.prompt_tokens === undefined)) return null
  return <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 border-t border-[#232a35] pt-2 text-[11px] text-[#8f97a6]">
    <Stat icon={<Timer size={12} />} value={formatSeconds(usage.seconds)} title="Total time for this reply" />
    {typeof usage.first_token_seconds === 'number' && <Stat icon={<Zap size={12} />} label="first token" value={formatSeconds(usage.first_token_seconds)} title="Time until the first token streamed" />}
    {!!usage.load_seconds && <Stat icon={<Cpu size={12} />} label="model load" value={formatSeconds(usage.load_seconds)} />}
    {usage.server && !usage.fallback && <Stat icon={<Cloud size={12} />} value={usage.server === 'remote' ? 'Remote PC' : 'This host'} title={`The reply was generated on ${usage.server === 'remote' ? 'the remote llama.cpp server' : 'this host\'s llama.cpp server'}.`} />}
    {usage.fallback && <Stat icon={<Cloud size={12} />} value={`${usage.fallback.provider === 'remote' ? 'Remote' : 'OpenRouter'} · GPU busy`} title={`The local model was in use for visual analysis, so this reply came from ${usage.fallback.provider === 'remote' ? `the remote llama.cpp server (${usage.fallback.model})` : 'OpenRouter free'}. Regenerate to try the local model again.`} />}
    <Stat icon={<ArrowUp size={12} />} label="in" value={formatTokens(usage.prompt_tokens)} title={`Prompt tokens across ${usage.steps ?? 0} request(s)${usage.cached_tokens ? `, ${usage.cached_tokens.toLocaleString()} cached` : ''}`} />
    <Stat icon={<ArrowDown size={12} />} label="out" value={formatTokens(usage.completion_tokens)} title="Generated tokens, including reasoning" />
    {thinking && <Stat icon={<Brain size={12} />} label="thinking" value={`${thinking.estimated ? '~' : ''}${formatTokens(thinking.value)}`} title={thinking.estimated ? 'Estimated from the reasoning text' : 'Reported by the provider'} />}
    {typeof usage.tokens_per_second === 'number' && <Stat icon={<Gauge size={12} />} value={`${usage.tokens_per_second} tok/s`} />}
    {context && <Stat icon={<Layers size={12} />} label="context" value={`${formatTokens(context.used)}${context.limit ? ` / ${formatTokens(context.limit)} (${context.percent}%)` : ''}`} title="Context window used by the last request" />}
    {!!usage.tool_calls && <Stat icon={<Wrench size={12} />} value={`${usage.tool_calls} tool call${usage.tool_calls === 1 ? '' : 's'}`} />}
    {typeof usage.cost === 'number' && usage.cost > 0 && <Stat icon={<span className="text-[10px]">$</span>} value={usage.cost.toFixed(4)} />}
    <span className="muted truncate" title={usage.model || turn.model}>{(usage.model || turn.model).split('/').pop()}{turn.thinking ? ' · thinking' : ''}</span>
    {latest && turn.status === 'done' && <button type="button" className="ml-auto inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[#aeb5c1] hover:bg-[#262d39] hover:text-white" onClick={() => onRetry(turn)} title="Regenerate this reply with the selected model"><RefreshCw size={11} /> Regenerate</button>}
  </div>
}
