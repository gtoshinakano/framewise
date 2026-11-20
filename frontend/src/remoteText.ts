// Pure helpers for the remote llama.cpp section of the Text engine dialog (see backend/app/remote_text.py).

export type RouteMode = 'local' | 'remote' | 'any'
export type TextRoute = { mode: RouteMode; model: string }
export type RemoteModel = { id: string; n_ctx?: number | null }
export type RemoteState = { state: string; detail: string; models: RemoteModel[]; active_jobs: number; heartbeat: number }
export type RemoteServer = {
  enabled: boolean; base_url: string; has_token: boolean; parallel: number
  routes: Partial<Record<string, TextRoute>>; assistant_fallback: string; state: RemoteState
  prefer: 'local' | 'remote'; chat_reserve: number; local_slots: number; effective_routes: Partial<Record<string, TextRoute>>
}

// Text pipelines the remote lanes can take, in the order the cut workflow runs them.
export const TEXT_JOB_TYPES: { pipeline: string; label: string }[] = [
  { pipeline: 'performance_beats', label: 'Performance beats' },
  { pipeline: 'speaker_attribution', label: 'Speaker attribution' },
  { pipeline: 'consistency_summary', label: 'Consistency summary' },
  { pipeline: 'review_answers', label: 'Production decisions' },
  { pipeline: 'regen', label: 'Cut summary' },
  { pipeline: 'sequence', label: 'Sequence steps' },
]

export const REMOTE_STATE_LABELS: Record<string, string> = {
  disabled: 'Off', online: 'Online', offline: 'Offline', auth_failed: 'Login refused', error: 'Error', worker_offline: 'Worker offline',
}

/** The remote model that unrouted jobs and Auto chat use: the chosen assistant model, else the first one the server lists. */
export function defaultModel(assistantModel: string, models: RemoteModel[]): string {
  return assistantModel || models[0]?.id || ''
}

/** What a pipeline does when it has no saved route: share the work with the remote's default model (this server when none is known). */
export function shownRoute(saved: TextRoute | undefined, fallbackModel: string): TextRoute {
  return saved ?? (fallbackModel ? { mode: 'any', model: fallbackModel } : { mode: 'local', model: '' })
}

/** Slots a server leaves to text jobs once the chat reserve is taken (at least one, as the backend does). */
export function jobSlots(capacity: number, reserve: number): number {
  return Math.max(1, capacity - Math.min(Math.max(0, reserve), capacity - 1))
}

export function routeValue(route?: TextRoute | null): string {
  if (!route || route.mode === 'local' || !route.model) return 'local'
  return `${route.mode}|${route.model}`
}

export function parseRouteValue(value: string): TextRoute {
  const split = value.indexOf('|')
  if (split < 0) return { mode: 'local', model: '' }
  const mode = value.slice(0, split)
  const model = value.slice(split + 1)
  return (mode === 'remote' || mode === 'any') && model ? { mode, model } : { mode: 'local', model: '' }
}

/** Models offered in the selects: the reported ones, plus a saved model the server no longer lists. */
export function modelChoices(models: RemoteModel[], saved: string[]): { id: string; listed: boolean }[] {
  const out = models.map(model => ({ id: model.id, listed: true }))
  for (const id of saved) if (id && !out.some(item => item.id === id)) out.push({ id, listed: false })
  return out
}

export function routeOptions(models: RemoteModel[], current?: TextRoute | null): { value: string; label: string }[] {
  const choices = modelChoices(models, current && current.mode !== 'local' ? [current.model] : [])
  const options = [{ value: 'local', label: 'This server' }]
  for (const { id, listed } of choices) {
    const note = listed ? '' : ' (not reported)'
    options.push({ value: `remote|${id}`, label: `Remote · ${id}${note}` })
    options.push({ value: `any|${id}`, label: `Remote · ${id} or this server (first free)${note}` })
  }
  return options
}

/** Label for a ``model_variant`` in the speed table: local variants by name, remote ones by model id. */
export function variantLabel(variant: string, names: Record<string, string>): string {
  if (variant.startsWith('remote:')) return `Remote · ${variant.slice('remote:'.length)}`
  return names[variant] || variant
}

export function remoteBadge(server: RemoteServer | null): string {
  if (!server?.enabled) return ''
  const state = server.state.state
  if (state === 'online') return server.state.active_jobs ? ` · Remote ${server.state.active_jobs} running` : ' · Remote online'
  return ` · Remote ${(REMOTE_STATE_LABELS[state] || state).toLowerCase()}`
}
