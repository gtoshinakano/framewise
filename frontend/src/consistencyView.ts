// Pure helpers for the Review column of the Results tab. They only reshape the consistency summary's text for
// reading; nothing here is saved back and the stored JSON is never changed.

export const CONSISTENCY_KEYS = ['human_review', 'recommended_checks', 'confusing', 'clear', 'summary'] as const
export type ConsistencyKey = (typeof CONSISTENCY_KEYS)[number]

export type ConsistencySection = {
  key: ConsistencyKey
  /** The field exactly as stored (trimmed); empty when the key is missing or not a string. */
  text: string
  present: boolean
  /** "None required." / "None identified." with nothing left to act on. */
  none: boolean
  /** Text after a leading "None required." (the model's explanation), or ''. */
  note: string
  /** Human review and Suggested checks as separate things to do. */
  items: string[]
  /** Every other field as paragraphs. */
  paragraphs: string[]
}

export type ConsistencyView = {
  /** False when the stored content is not a JSON object (older results that the Analysis fields editor can repair). */
  valid: boolean
  sections: Record<ConsistencyKey, ConsistencySection>
  decisions: number
  checks: number
  conflicts: boolean
}

const NONE_PREFIX = /^(?:none required|none identified|none needed|not required)\b[\s.!:;,-]*/i
const NONE_BARE = /^(?:none|n\/a|nil)[\s.!]*$/i
const CONTRADICTS = /^(?:but|however|except|although|though)\b/i

/** Whether a field says there is nothing to do. "None of the characters are confirmed…" is real content, not "none". */
export function noneNote(text: string): { none: boolean; note: string } {
  const value = text.trim()
  if (NONE_BARE.test(value)) return { none: true, note: '' }
  const match = NONE_PREFIX.exec(value)
  if (!match) return { none: false, note: '' }
  const note = value.slice(match[0].length).trim()
  if (CONTRADICTS.test(note)) return { none: false, note: '' }
  return { none: true, note }
}

const BULLET = /^\s*(?:[-*•–]|\d+[.)]|\(\d+\))\s+/
const ABBREVIATIONS = new Set(['approx', 'e.g', 'i.e', 'vs', 'etc', 'cf', 'no', 'fig', 'dr', 'mr', 'mrs', 'ms', 'st', 'sec', 'min', 'ca', 'al'])
const MIN_PIECE = 12

/**
 * Splits one paragraph into sentences. A boundary is a . ! or ? followed by whitespace and a capital letter, so
 * decimals ("1269.19s"), numbers after a period ("approx. 1227.89s") and lowercase continuations stay together.
 */
export function splitSentences(text: string): string[] {
  const value = text.trim()
  const boundary = /[.!?]["'”’)\]]*\s+(?=["“'‘(\[]?[A-Z])/g
  const pieces: string[] = []
  let start = 0
  for (let match = boundary.exec(value); match; match = boundary.exec(value)) {
    if (match[0][0] === '.') {
      const word = /([A-Za-z.]+)$/.exec(value.slice(start, match.index))?.[1]?.toLowerCase()
      if (word && (word.length === 1 || ABBREVIATIONS.has(word))) continue
    }
    pieces.push(value.slice(start, match.index + match[0].trimEnd().length))
    start = match.index + match[0].length
  }
  pieces.push(value.slice(start))
  // A fragment that short is a stray abbreviation or label, not a thing to do: keep it with its neighbour.
  const merged: string[] = []
  for (const piece of pieces.map(item => item.trim()).filter(Boolean)) {
    if (merged.length && (piece.length < MIN_PIECE || merged[merged.length - 1].length < MIN_PIECE)) merged[merged.length - 1] += ` ${piece}`
    else merged.push(piece)
  }
  return merged
}

/** Separate things to do: one per line or bullet, otherwise one per sentence. */
export function splitItems(text: string): string[] {
  const lines = text.replace(/\r\n?/g, '\n').split('\n').map(line => line.trim()).filter(Boolean)
  if (!lines.length) return []
  if (lines.length > 1 || BULLET.test(lines[0])) return lines.map(line => line.replace(BULLET, '').trim()).filter(Boolean)
  return splitSentences(lines[0])
}

export function paragraphsOf(text: string): string[] {
  return text.replace(/\r\n?/g, '\n').split('\n').map(line => line.replace(BULLET, '').trim()).filter(Boolean)
}

/**
 * Some results keep a field as a JSON array written inside the string (`["first", "second"]`). Reading it as prose
 * would show brackets, quotes and commas, so it becomes one line per entry.
 */
export function unwrapList(text: string): string {
  const value = text.trim()
  if (!/^\[[\s\S]*\]$/.test(value)) return value
  try {
    const parsed: unknown = JSON.parse(value)
    if (Array.isArray(parsed)) return parsed.map(entry => (typeof entry === 'string' ? entry : JSON.stringify(entry)).trim()).filter(Boolean).join('\n')
  } catch { /* Invalid JSON (for example unescaped quotes): strip the brackets and quotes by hand. */ }
  return value.slice(1, -1).split('\n').map(line => line.trim().replace(/,$/, '').replace(/^"([\s\S]*)"$/, '$1').trim()).filter(Boolean).join('\n')
}

const TIME = /\b\d{1,5}(?:\.\d+)?(?:\s?[–-]\s?\d{1,5}(?:\.\d+)?)?\s?s\b/g

/** Cuts text into runs, marking the timestamps ("1269.19s", "0.0–0.4s") so they can be styled. */
export function timeParts(text: string): { text: string; time: boolean }[] {
  const parts: { text: string; time: boolean }[] = []
  let last = 0
  for (const match of text.matchAll(TIME)) {
    if (match.index > last) parts.push({ text: text.slice(last, match.index), time: false })
    parts.push({ text: match[0], time: true })
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push({ text: text.slice(last), time: false })
  return parts
}

export function parseConsistency(content: string): ConsistencyView {
  let data: Record<string, unknown> | null = null
  try {
    const parsed: unknown = JSON.parse(content)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) data = parsed as Record<string, unknown>
  } catch { /* Shown as an invalid result; the Analysis fields editor can repair it. */ }
  const sections = {} as Record<ConsistencyKey, ConsistencySection>
  for (const key of CONSISTENCY_KEYS) {
    const value = data?.[key]
    const text = typeof value === 'string' ? unwrapList(value) : ''
    const { none, note } = text ? noneNote(text) : { none: false, note: '' }
    const actionable = key === 'human_review' || key === 'recommended_checks'
    sections[key] = {
      key, text, present: text !== '', none, note,
      items: actionable && text && !none ? splitItems(text) : [],
      paragraphs: !actionable && text && !none ? paragraphsOf(text) : [],
    }
  }
  return {
    valid: data !== null, sections,
    decisions: sections.human_review.items.length,
    checks: sections.recommended_checks.items.length,
    conflicts: sections.confusing.present && !sections.confusing.none,
  }
}
