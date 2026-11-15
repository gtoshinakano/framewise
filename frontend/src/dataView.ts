// Pure helpers behind the run-details viewer: decide how a saved value should be shown.

import { formatRunTime } from './queueView.ts'

export type Parsed = { kind: 'json'; value: unknown; fenced: boolean } | { kind: 'text'; text: string }

const FENCE = /^```(?:json|JSON)?\s*\n([\s\S]*?)\n?```\s*$/

function tryParse(text: string): { ok: true; value: unknown } | { ok: false } {
  try { return { ok: true, value: JSON.parse(text) } } catch { return { ok: false } }
}

/** Objects and arrays are JSON; a string is JSON if it parses (also when wrapped in one ```json fence or double-encoded). */
export function parseValue(value: unknown): Parsed {
  if (typeof value !== 'string') return value !== null && typeof value === 'object' ? { kind: 'json', value, fenced: false } : { kind: 'text', text: value === undefined || value === null ? '' : String(value) }
  const trimmed = value.trim()
  const fence = FENCE.exec(trimmed)
  const body = fence ? fence[1].trim() : trimmed
  if (/^[[{"]/.test(body)) {
    const first = tryParse(body)
    if (first.ok) {
      // A JSON string that itself holds JSON (a response saved after being encoded twice).
      if (typeof first.value === 'string') { const inner = parseValue(first.value); if (inner.kind === 'json') return inner }
      if (first.value !== null && typeof first.value === 'object') return { kind: 'json', value: first.value, fenced: !!fence }
    }
  }
  return { kind: 'text', text: value }
}

export type TextStyle = 'markdown' | 'plain'

/** Markdown is used only when the text clearly uses it; model prompts and error text stay literal. */
export function looksLikeMarkdown(text: string): boolean {
  return /^#{1,4}\s+\S/m.test(text) || /^\s*(?:[-*]|\d+\.)\s+\S/m.test(text) || /\*\*[^*\n]+\*\*/.test(text) || /^```/m.test(text) || /^\|.+\|\s*$/m.test(text)
}

export type OutlineItem = { line: number; label: string; level: 1 | 2 | 3 }

const TAG_LINE = /^\s*<(Picture|Audio|Video|d|shot|scene)(?:\s+\d+)?[^>]*>/i
const HEADING = /^(#{1,4})\s+(.+?)\s*#*$/
const LABEL = /^([A-Z][A-Za-z0-9 /&'()-]{1,48}):\s*$/
const NUMBERED = /^(\d+)[.)]\s+([A-Z][^.]{2,60})$/

/** Section markers in a long prompt, so the reader can jump to them. Lines are zero-based. */
export function promptOutline(text: string): OutlineItem[] {
  const items: OutlineItem[] = []
  text.split('\n').forEach((raw, line) => {
    const row = raw.trimEnd()
    const heading = HEADING.exec(row)
    if (heading) { items.push({ line, label: heading[2], level: Math.min(3, heading[1].length) as 1 | 2 | 3 }); return }
    const label = LABEL.exec(row)
    if (label) { items.push({ line, label: label[1], level: 1 }); return }
    const numbered = NUMBERED.exec(row)
    if (numbered) { items.push({ line, label: `${numbered[1]}. ${numbered[2]}`, level: 2 }); return }
    const tag = TAG_LINE.exec(row)
    if (tag) items.push({ line, label: row.trim().slice(0, 60), level: 3 })
  })
  return items
}

export type Node = { kind: 'object' | 'array'; size: number } | { kind: 'string' | 'number' | 'boolean' | 'null' }

export function nodeOf(value: unknown): Node {
  if (value === null || value === undefined) return { kind: 'null' }
  if (Array.isArray(value)) return { kind: 'array', size: value.length }
  if (typeof value === 'object') return { kind: 'object', size: Object.keys(value as object).length }
  if (typeof value === 'number') return { kind: 'number' }
  if (typeof value === 'boolean') return { kind: 'boolean' }
  return { kind: 'string' }
}

/** Big containers start closed so a long window or context list does not push everything else off screen. */
export function startsCollapsed(value: unknown, depth: number): boolean {
  const node = nodeOf(value)
  if (node.kind === 'array') return node.size > 20 || (depth > 1 && node.size > 0)
  if (node.kind === 'object') return node.size > 8 || depth > 2
  return false
}

/** A string deserves its own wrapped block instead of one long line. */
export const isLongString = (value: string) => value.length > 140 || value.includes('\n')

export const MAX_RENDER_CHARS = 200_000

export function stringify(value: unknown): string {
  if (typeof value === 'string') return value
  return JSON.stringify(value, null, 2) ?? ''
}

export type Fact = { label: string; value: string }

const FACT_KEYS: [label: string, keys: string[]][] = [
  ['Model', ['model_id', 'model', 'llm_model', 'openrouter_model']],
  ['Provider', ['text_provider', 'provider']],
  ['Thinking', ['local_thinking', 'thinking']],
  ['Text model variant', ['model_variant']], ['Decode speed (tok/s)', ['decode_tokens_per_second']], ['Draft acceptance', ['draft_acceptance']],
  ['Width', ['video_width']], ['Height', ['video_height']], ['Frames per second', ['video_fps']], ['Frame budget', ['video_frame_budget']],
  ['Stages', ['stages']], ['Workflow', ['workflow']],
]

function find(source: unknown, keys: string[], depth = 0): unknown {
  if (!source || typeof source !== 'object' || depth > 3) return undefined
  const record = source as Record<string, unknown>
  for (const key of keys) if (key in record && record[key] !== null && record[key] !== undefined && record[key] !== '') return record[key]
  for (const child of Object.values(record)) {
    if (child && typeof child === 'object' && !Array.isArray(child)) { const found = find(child, keys, depth + 1); if (found !== undefined) return found }
  }
  return undefined
}

const show = (value: unknown) => Array.isArray(value) ? value.join(', ') : typeof value === 'boolean' ? (value ? 'on' : 'off') : String(value)

/** Short facts about how a run was configured, read from whichever of the saved snapshots has them. */
export function runFacts(sources: unknown[]): Fact[] {
  const facts: Fact[] = []
  for (const [label, keys] of FACT_KEYS) {
    for (const source of sources) {
      const value = find(source, keys)
      if (value !== undefined) { facts.push({ label, value: show(value) }); break }
    }
  }
  const context = sources.map(source => find(source, ['additional_context'])).find(value => typeof value === 'string' && value.trim())
  if (typeof context === 'string') facts.push({ label: 'Additional context', value: context.length > 400 ? `${context.slice(0, 400)}…` : context })
  return facts
}

export function durationText(start: string | null, end: string | null): string {
  if (!start) return ''
  const from = Date.parse(`${start.replace(' ', 'T')}Z`)
  const to = end ? Date.parse(`${end.replace(' ', 'T')}Z`) : Date.now()
  if (!Number.isFinite(from) || !Number.isFinite(to) || to < from) return ''
  return formatRunTime(Math.round((to - from) / 1000))
}
