import { api } from './api'
import { queueApi } from './queueApi'
import type { Choice, Escalation } from './inboxView'

/**
 * Send one of a question's one-click answers (inboxView.choicesFor) as the person. A choice that names a new character asks
 * for the name first; returns false when the person cancelled that. Used by the Inbox and the analysis dialog's Your answers.
 */
export async function answerEscalation(item: Escalation, choice: Choice): Promise<boolean> {
  let request = choice.request
  if (choice.input === 'title') {
    const title = window.prompt('Name of the new character (created from the clearest box; every box of this person follows):')
    if (!title?.trim()) return false
    request = { path: `/escalations/${item.id}/person`, body: { choice: 'new', title: title.trim() } }
  }
  if (request) await api(request.path, { method: 'POST', body: JSON.stringify({ tier: 'human', ...(request.body as object) }) })
  if (choice.retry) { for (const job of choice.retry) await queueApi.retry(job).catch(() => undefined) }
  if (choice.dismiss) { for (const job of choice.dismiss) await queueApi.dismiss(job).catch(() => undefined) }
  if (choice.retry || choice.dismiss || choice.close) {
    await api(`/escalations/${item.id}/close`, { method: 'POST', body: JSON.stringify({ status: choice.close ?? (choice.retry ? 'resolved' : 'dismissed'), resolution: { by: 'inbox', choice: choice.key } }) })
  }
  return true
}
