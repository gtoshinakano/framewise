// Pure helpers of the dialogue editor: no React and no fetch, so the node tests can import them.
// How a line is played (on screen, off screen…) is not part of the dialogue: each shot decides it.
import type { Compilation, CompiledLine, CompiledVoice } from './sequenceApi'

export const NARRATOR = 'Narrator'
const MIN_LINE = 0.2

const round = (value: number) => Math.round(value * 1000) / 1000
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

/** Words of a line as parts, spread over its time in proportion to their characters (the server does the same when the text changed). */
export function spread(text: string, t0: number, t1: number) {
  const units = text.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9']+/g)?.flatMap(token => /[぀-ヿ㐀-鿿]/.test(token) ? token.match(/.{1,3}/gu) || [] : [token]) || []
  const total = units.reduce((sum, unit) => sum + unit.length, 0)
  let cursor = t0
  return units.map((unit, index) => {
    const end = index === units.length - 1 ? t1 : cursor + (t1 - t0) * unit.length / total
    const part = { t0: round(cursor), t1: round(end), text: unit }
    cursor = end
    return part
  })
}

/** Move one edge of a line, keeping it at least 0.2 s long and inside the sequence. */
export function setEdge(line: CompiledLine, edge: 't0' | 't1', value: number, duration: number): CompiledLine {
  if (!Number.isFinite(value)) return line
  const next = edge === 't0' ? clamp(value, 0, line.t1 - MIN_LINE) : clamp(value, line.t0 + MIN_LINE, duration)
  return { ...line, [edge]: round(next) }
}

export function nudge(line: CompiledLine, edge: 't0' | 't1', delta: number, duration: number) {
  return setEdge(line, edge, line[edge] + delta, duration)
}

export function setText(line: CompiledLine, text: string): CompiledLine {
  return { ...line, text, parts: spread(text, line.t0, line.t1) }
}

/** Join a line with the next one into one utterance. */
export function mergeNext(lines: CompiledLine[], index: number): CompiledLine[] {
  const a = lines[index], b = lines[index + 1]
  if (!a || !b) return lines
  const joined = /[぀-ヿ㐀-鿿]$/.test(a.text) || /^[぀-ヿ㐀-鿿]/.test(b.text) ? a.text + b.text : `${a.text} ${b.text}`
  const merged: CompiledLine = { ...a, t1: Math.max(a.t1, b.t1), text: joined, original_text: `${a.original_text} ${b.original_text}`.trim(), parts: [...a.parts, ...b.parts],
    source_ids: [...a.source_ids, ...b.source_ids], subtitle_ids: [...new Set([...a.subtitle_ids, ...b.subtitle_ids])], continues_next: b.continues_next,
    subtitle_text: [a.subtitle_text, b.subtitle_text].filter(Boolean).join(' ') || null, character: a.character || b.character, character_context_id: a.character_context_id || b.character_context_id }
  return [...lines.slice(0, index), merged, ...lines.slice(index + 2)]
}

/** Split a line at a time: the text is cut where the time falls in proportion, and can then be corrected by hand. */
export function splitAt(lines: CompiledLine[], index: number, time: number): CompiledLine[] {
  const line = lines[index]
  if (!line || time < line.t0 + MIN_LINE || time > line.t1 - MIN_LINE) return lines
  const characters = [...line.text]
  const cut = clamp(Math.round(characters.length * (time - line.t0) / (line.t1 - line.t0)), 1, Math.max(1, characters.length - 1))
  const first = characters.slice(0, cut).join(''), second = characters.slice(cut).join('')
  const make = (part: Partial<CompiledLine>): CompiledLine => ({ ...line, continues_next: false, ...part, original_text: line.original_text, subtitle_text: line.subtitle_text })
  return [...lines.slice(0, index),
    make({ id: `${line.id}a`, t1: round(time), text: first, parts: spread(first, line.t0, round(time)), continues_next: true }),
    make({ id: `${line.id}b`, t0: round(time), text: second, parts: spread(second, round(time), line.t1) }),
    ...lines.slice(index + 1)]
}

/** Give every line of one voice label the same character, and record the decision in the voices table. */
export function applyVoice(content: Compilation, label: string, name: string | null, characters: { id: string; name: string }[]): Compilation {
  const person = name && name !== NARRATOR ? characters.find(item => item.name === name) : undefined
  const lines = content.lines.map(line => line.speaker_label === label ? { ...line, character: name, character_context_id: person?.id ?? null } : line)
  const voice: CompiledVoice = { speaker_label: label, character: name, character_context_id: person?.id ?? null, kind: name === NARRATOR ? 'narrator' : name ? 'character' : 'unknown', confidence: 'high', reason: 'Set by you' }
  const voices = content.voices.some(item => item.speaker_label === label) ? content.voices.map(item => item.speaker_label === label ? voice : item) : [...content.voices, voice]
  return { ...content, lines, voices }
}

export function setSpeaker(line: CompiledLine, name: string | null, characters: { id: string; name: string }[]): CompiledLine {
  const person = name && name !== NARRATOR ? characters.find(item => item.name === name) : undefined
  return { ...line, character: name, character_context_id: person?.id ?? null }
}

/** Problems the server would refuse, found on the client so a row can show them before saving. */
export function lineProblems(line: CompiledLine, duration: number): string[] {
  const problems: string[] = []
  if (!line.text.trim()) problems.push('Write the words, or delete the line')
  if (!(line.t0 >= 0 && line.t0 < line.t1 && line.t1 <= duration + 0.001)) problems.push(`Times must satisfy 0 ≤ start < end ≤ ${duration.toFixed(1)}`)
  return problems
}

export function problemsById(lines: CompiledLine[], duration: number) {
  const out: Record<string, string[]> = {}
  for (const line of lines) {
    const problems = lineProblems(line, duration)
    if (problems.length) out[line.id] = problems
  }
  return out
}

/** A line that a person should still look at. */
export function needsReview(line: CompiledLine) {
  return line.confidence === 'low' || !line.character || line.continues_next
}

export type Filter = 'all' | 'review' | 'changed'

export function changed(line: CompiledLine) {
  return line.text.trim() !== line.original_text.trim()
}

export function visibleLines(lines: CompiledLine[], filter: Filter) {
  return lines.filter(line => filter === 'all' || (filter === 'review' ? needsReview(line) : changed(line)))
}

export function counts(lines: CompiledLine[]) {
  return { total: lines.length, review: lines.filter(needsReview).length, changed: lines.filter(changed).length }
}

export type DiffPart = { text: string; kind: 'same' | 'del' | 'add' }

/** A character-level difference between two Japanese or Latin texts, as runs of unchanged, removed and added text. */
export function diff(before: string, after: string): DiffPart[] {
  const a = [...before], b = [...after]
  if (a.length * b.length > 250_000) return before === after ? [{ text: after, kind: 'same' }] : [{ text: before, kind: 'del' }, { text: after, kind: 'add' }]
  const table = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--) table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1])
  const parts: DiffPart[] = []
  const push = (kind: DiffPart['kind'], text: string) => { const last = parts[parts.length - 1]; if (last && last.kind === kind) last.text += text; else parts.push({ text, kind }) }
  let i = 0, j = 0
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { push('same', a[i]); i++; j++ }
    else if (table[i + 1][j] >= table[i][j + 1]) { push('del', a[i]); i++ }
    else { push('add', b[j]); j++ }
  }
  while (i < a.length) push('del', a[i++])
  while (j < b.length) push('add', b[j++])
  return parts
}

/** The speaker choices of a line: no one, the project's characters and the narrator. */
export function speakerOptions(characters: { id: string; name: string }[]) {
  return [{ value: '', label: 'Unknown speaker' }, ...characters.map(person => ({ value: person.name, label: person.name })), { value: NARRATOR, label: NARRATOR }]
}

/** The lines and voices as the server stores them: nothing but the compilation fields. */
export function payload(content: Compilation): Compilation {
  return { lines: content.lines, voices: content.voices, open_questions: content.open_questions }
}

export function sameContent(a: Compilation, b: Compilation) {
  return JSON.stringify(payload(a)) === JSON.stringify(payload(b))
}

/** The line being spoken at a sequence time, if any. */
export function lineAt(lines: CompiledLine[], time: number) {
  return lines.find(line => time >= line.t0 && time < line.t1)
}
