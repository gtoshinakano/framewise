export type BatchQueueResult = { queued: number; skipped?: number; blocked?: Record<string, number[]> }

function cutList(numbers: number[]) {
  return numbers.slice(0, 8).join(', ') + (numbers.length > 8 ? ` and ${numbers.length - 8} more` : '')
}

/**
 * Says what a "from this cut" batch queued and which cuts were held back, with the reason for each.
 * `reasons` maps a `blocked` key from the server to the words that follow "Not queued,"; its order is the order shown.
 */
export function batchQueueMessage(what: string, firstCut: number, result: BatchQueueResult, reasons: Record<string, string>, skippedReason?: string) {
  const parts = [`${what} queued for ${result.queued} ${result.queued === 1 ? 'cut' : 'cuts'} from cut ${firstCut}.`]
  if (result.skipped && skippedReason) parts.push(`${result.skipped} skipped, ${skippedReason}.`)
  for (const [key, reason] of Object.entries(reasons)) {
    const numbers = result.blocked?.[key] ?? []
    if (numbers.length) parts.push(`Not queued, ${reason}: cut ${cutList(numbers)}.`)
  }
  return parts.join(' ')
}

export const ANSWERS_BATCH_REASONS = {
  no_consistency: 'no consistency summary yet',
  nothing_to_review: 'nothing to decide and no saved answers',
  busy: 'analysis, consistency summary or answers already queued or running',
}

export const SUMMARY_BATCH_REASONS = {
  no_consistency: 'no consistency summary yet',
  answers_pending: 'production decisions still being drafted, queue again when they finish',
  busy: 'analysis, summary or answers already queued or running',
}

export const CHAIN_BATCH_REASONS = {
  missing_results: 'initial frame, video or last frame result missing',
  busy: 'analysis, consistency summary, answers or summary already queued or running',
}
