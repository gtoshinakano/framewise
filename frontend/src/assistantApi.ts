import { api, jsonBody } from './api'

export type TurnStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
export type AssistantUsage = {
  prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; reasoning_tokens?: number; reasoning_tokens_estimated?: number
  cached_tokens?: number; cost?: number; context_tokens?: number; context_limit?: number | null; steps?: number; tool_calls?: number
  seconds?: number; generation_seconds?: number; first_token_seconds?: number | null; load_seconds?: number; tokens_per_second?: number; model?: string
  fallback?: { reason: 'gpu_busy'; from: string; provider: string; model: string }
}
export type ToolCall = { id: string; name: string; arguments: string }
export type AssistantMessage = {
  id: string; session_id: string; turn_id: string | null; seq: number; role: 'user' | 'assistant' | 'tool'; content: string; reasoning: string
  tool_calls: ToolCall[]; tool_call_id: string | null; tool_name: string | null; tool_summary: string | null; streaming: boolean
  usage: AssistantUsage & { ok?: boolean; preloaded?: boolean; seconds?: number; finish_reason?: string | null }; created_at: string
}
export type AssistantTurn = {
  id: string; session_id: string; status: TurnStatus; stage: string; provider: string; model: string; thinking: boolean; skill: string | null
  cancel_requested: boolean; gpu_fallback: boolean; attempt_count: number; error: string | null; usage: AssistantUsage; version: number; created_at: string
  started_at: number | null; finished_at: number | null
}
export type DiffSegment = { op: 'equal' | 'insert' | 'delete'; text: string }
export type AssistantProposal = {
  id: string; session_id: string; turn_id: string | null; cut_id: string; target: string; label: string; title: string; kind: string
  base_result_id: string | null; content: string; summary: string; status: 'pending' | 'applied' | 'discarded'; stale: boolean
  applied_result_id: string | null; model_id: string | null; provider: string | null; diff: DiffSegment[]; diff_label: string | null; created_at: string; resolved_at: string | null
}
export type AssistantSession = {
  id: string; cut_id: string; title: string; provider: string; model: string; thinking: boolean; created_at: string; updated_at: string
  message_count?: number; last_status?: TurnStatus | null; pending_proposals?: number
}
export type SessionDetail = { session: AssistantSession; messages: AssistantMessage[]; turns: AssistantTurn[]; proposals: AssistantProposal[] }
export type ProviderModel = { id: string; label?: string; context?: number | null }
export type ProviderInfo = { id: string; label: string; available: boolean; detail?: string; models: ProviderModel[]; thinking: boolean; managed: boolean; free_model: boolean; context_limit?: number | null; busy?: boolean; fallback_model?: string }
export type SkillInfo = { name: string; title: string; description: string; examples: string[] }
export type AssistantConfig = { worker_online: boolean; providers: ProviderInfo[]; skills: SkillInfo[]; defaults: { provider: string; openrouter_model: string; thinking: boolean }; recent_models: string[] }
export type LocalModelStatus = { desired_loaded: boolean; state: string; detail: string }

export const assistantApi = {
  config: () => api<AssistantConfig>('/assistant/config'),
  sessions: (cutId: string) => api<AssistantSession[]>(`/cuts/${cutId}/assistant/sessions`),
  createSession: (cutId: string, body: { provider: string; model: string; thinking: boolean }) => api<SessionDetail>(`/cuts/${cutId}/assistant/sessions`, jsonBody(body)),
  session: (sessionId: string) => api<SessionDetail>(`/assistant/sessions/${sessionId}`),
  rename: (sessionId: string, title: string) => api<AssistantSession>(`/assistant/sessions/${sessionId}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  remove: (sessionId: string) => api<{ ok: boolean }>(`/assistant/sessions/${sessionId}`, { method: 'DELETE' }),
  send: (sessionId: string, body: { content: string; provider: string; model: string; thinking: boolean; skill?: string | null; gpu_fallback?: boolean }) =>
    api<{ turn: AssistantTurn; message: AssistantMessage }>(`/assistant/sessions/${sessionId}/messages`, jsonBody(body)),
  cancel: (turnId: string) => api<AssistantTurn>(`/assistant/turns/${turnId}/cancel`, { method: 'POST' }),
  retry: (turnId: string, body: { provider?: string; model?: string; thinking?: boolean; gpu_fallback?: boolean }) => api<SessionDetail>(`/assistant/turns/${turnId}/retry`, jsonBody(body)),
  apply: (proposalId: string) => api<AssistantProposal>(`/assistant/proposals/${proposalId}/apply`, { method: 'POST' }),
  discard: (proposalId: string) => api<AssistantProposal>(`/assistant/proposals/${proposalId}/discard`, { method: 'POST' }),
  localModel: () => api<LocalModelStatus>('/local-text-model'),
  setLocalModel: (loaded: boolean) => api<LocalModelStatus>('/local-text-model', { method: 'PUT', body: JSON.stringify({ loaded }) }),
}

export const streamUrl = (turnId: string) => `/api/assistant/turns/${turnId}/stream`
