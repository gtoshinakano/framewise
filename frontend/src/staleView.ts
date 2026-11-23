import { parseQuestions, selection } from './consistencyView.ts'

/** What `answersStale` needs of the current consistency review. */
export type ReviewRow = { content: string; created_at: string; stale_reason?: string | null }

/**
 * Why the saved answers are out of date, or null. They are when the review they answer is itself out of date, or when
 * the review was written again after they were saved and has a question they do not answer (an answer that still
 * matches its question stays valid: the person's answer is their last word).
 */
export function answersStale(answers: string, savedAt: string | undefined, review: ReviewRow | undefined): string | null {
  if (!review || !answers.trim()) return null
  if (review.stale_reason) return `the review they answer is out of date (${review.stale_reason})`
  if (!savedAt || savedAt >= review.created_at) return null
  let questions: ReturnType<typeof parseQuestions> = []
  try { questions = parseQuestions((JSON.parse(review.content) as { questions?: unknown }).questions) } catch { return null }
  const open = questions.filter(question => !selection(answers, question, questions)).length
  return open ? `the review was written again after these answers; ${open} question${open === 1 ? ' has' : 's have'} no answer` : null
}
