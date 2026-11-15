import { api, jsonBody } from './api'
import type { HistoryFilter, QueueState, RunPage } from './queueView'

/** Calls of the episode queue drawer (backend `queue_api.py`). */
export const queueApi = {
  state: (episodeId: string) => api<QueueState>(`/episodes/${episodeId}/queue`),
  history: (episodeId: string, filter: HistoryFilter, before?: number | null) => {
    const query = new URLSearchParams()
    if (filter !== 'all') query.set('status', filter)
    if (before != null) query.set('before', String(before))
    const text = query.toString()
    return api<RunPage>(`/episodes/${episodeId}/runs${text ? `?${text}` : ''}`)
  },
  cancel: (jobId: string) => api<{ ok: boolean }>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  cancelEpisode: (episodeId: string, options: { include_running?: boolean; group?: string } = {}) =>
    api<{ cancelled: number }>(`/episodes/${episodeId}/queue/cancel`, jsonBody(options)),
  retry: (jobId: string) => api<{ job_id: string }>(`/jobs/${jobId}/retry`, { method: 'POST' }),
  dismiss: (jobId: string) => api<{ ok: boolean }>(`/jobs/${jobId}/dismiss`, { method: 'POST' }),
  dismissAll: (episodeId: string) => api<{ dismissed: number }>(`/episodes/${episodeId}/queue/dismiss-failures`, { method: 'POST' }),
  remove: (jobId: string) => api<{ ok: boolean }>(`/jobs/${jobId}`, { method: 'DELETE' }),
  stopChat: (turnId: string) => api(`/assistant/turns/${turnId}/cancel`, { method: 'POST' }),
  stopChatApi: (requestId: string) => api(`/chat-gateway/requests/${requestId}/cancel`, { method: 'POST' }),
}
