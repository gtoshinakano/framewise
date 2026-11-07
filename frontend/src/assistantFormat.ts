import type { AssistantMessage, AssistantProposal, AssistantTurn, AssistantUsage, SessionDetail, ToolCall } from './assistantApi'

export const ACTIVE_STATUSES = new Set(['queued', 'running'])

function upsert<T extends { id: string }>(list: T[], item: T): T[] {
  const index = list.findIndex(entry => entry.id === item.id)
  if (index < 0) return [...list, item]
  const next = list.slice()
  next[index] = item
  return next
}

const bySeq = (a: AssistantMessage, b: AssistantMessage) => a.seq - b.seq

/** Apply one server-sent event from a turn stream to the session state. */
export function applyStreamEvent(detail: SessionDetail, event: string, data: unknown): SessionDetail {
  switch (event) {
    case 'snapshot': {
      const value = data as { turn: AssistantTurn; messages: AssistantMessage[]; proposals: AssistantProposal[] }
      return {
        ...detail,
        turns: upsert(detail.turns, value.turn),
        messages: [...detail.messages.filter(message => message.turn_id !== value.turn.id), ...value.messages].sort(bySeq),
        proposals: [...detail.proposals.filter(proposal => proposal.turn_id !== value.turn.id), ...value.proposals],
      }
    }
    case 'delta': {
      const value = data as { id: string; content: string; reasoning: string }
      return { ...detail, messages: detail.messages.map(message => message.id === value.id ? { ...message, content: message.content + value.content, reasoning: message.reasoning + value.reasoning } : message) }
    }
    case 'message':
      return { ...detail, messages: upsert(detail.messages, data as AssistantMessage).sort(bySeq) }
    case 'proposal':
      return { ...detail, proposals: upsert(detail.proposals, data as AssistantProposal) }
    case 'turn':
    case 'done':
      return { ...detail, turns: upsert(detail.turns, data as AssistantTurn) }
    default:
      return detail
  }
}

export type ToolStep = { call: ToolCall; result?: AssistantMessage }
export type Step = { message: AssistantMessage; tools: ToolStep[] }
export type TurnView = { id: string; turn?: AssistantTurn; user?: AssistantMessage; steps: Step[]; proposals: AssistantProposal[] }

/** Group flat messages into user turns: steps of reasoning, tool calls with results, and answers. */
export function groupTurns(detail: SessionDetail): TurnView[] {
  const views: TurnView[] = []
  const byId = new Map<string, TurnView>()
  const results = new Map(detail.messages.filter(message => message.role === 'tool' && message.tool_call_id).map(message => [message.tool_call_id as string, message]))
  for (const message of [...detail.messages].sort(bySeq)) {
    const key = message.turn_id || message.id
    let view = byId.get(key)
    if (!view) {
      view = { id: key, turn: detail.turns.find(turn => turn.id === message.turn_id), steps: [], proposals: detail.proposals.filter(proposal => proposal.turn_id && proposal.turn_id === message.turn_id) }
      byId.set(key, view)
      views.push(view)
    }
    if (message.role === 'user') view.user = message
    else if (message.role === 'assistant') view.steps.push({ message, tools: message.tool_calls.map(call => ({ call, result: results.get(call.id) })) })
  }
  return views
}

export const activeTurn = (detail: SessionDetail | null) => detail?.turns.find(turn => ACTIVE_STATUSES.has(turn.status))

export function formatTokens(value?: number | null) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (value >= 100000) return `${Math.round(value / 1000)}k`
  if (value >= 10000) return `${(value / 1000).toFixed(1)}k`
  return value.toLocaleString('en-US')
}

export function formatSeconds(value?: number | null) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—'
  if (value < 10) return `${value.toFixed(1)}s`
  if (value < 60) return `${Math.round(value)}s`
  const minutes = Math.floor(value / 60)
  return `${minutes}m ${String(Math.round(value - minutes * 60)).padStart(2, '0')}s`
}

export function contextShare(usage?: AssistantUsage | null, fallbackLimit?: number | null) {
  const used = usage?.context_tokens
  const limit = usage?.context_limit ?? fallbackLimit ?? null
  if (typeof used !== 'number') return null
  return { used, limit, percent: limit ? Math.min(100, Math.round((used / limit) * 1000) / 10) : null }
}

export function reasoningTokens(usage?: AssistantUsage | null) {
  if (!usage) return null
  if (typeof usage.reasoning_tokens === 'number') return { value: usage.reasoning_tokens, estimated: false }
  if (typeof usage.reasoning_tokens_estimated === 'number') return { value: usage.reasoning_tokens_estimated, estimated: true }
  return null
}

/** Pretty JSON arguments for display; falls back to the raw text. */
export function prettyArguments(raw: string) {
  try {
    const value = JSON.parse(raw || '{}')
    if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 0) return ''
    return JSON.stringify(value, null, 2)
  } catch {
    return raw
  }
}

export function prettyResult(raw: string) {
  try { return JSON.stringify(JSON.parse(raw), null, 2) } catch { return raw }
}

const PASTE_SKILLS = new Set(['h3_animation_prompt', 'still_image_prompt'])

/** Text placed in the composer for a starter prompt; prompt-fixing skills leave room to paste. */
export function starterText(skill: string, example: string) {
  return PASTE_SKILLS.has(skill) ? `${example}:\n\n` : example
}

/** Collapse long unchanged runs of a diff so the card shows the edits. */
export function compactDiff(segments: { op: string; text: string }[], context = 90) {
  return segments.map((segment, index) => {
    if (segment.op !== 'equal' || segment.text.length <= context * 2 + 20) return segment
    const first = index === 0, last = index === segments.length - 1
    const head = first ? '' : segment.text.slice(0, context)
    const tail = last ? '' : segment.text.slice(-context)
    return { op: 'equal', text: `${head}${head && tail ? ' … ' : '… '}${tail}` }
  })
}

export function turnElapsed(turn: AssistantTurn | undefined, now: number) {
  if (!turn) return null
  if (typeof turn.usage.seconds === 'number' && !ACTIVE_STATUSES.has(turn.status)) return turn.usage.seconds
  if (turn.started_at) return Math.max(0, now / 1000 - turn.started_at)
  return null
}
