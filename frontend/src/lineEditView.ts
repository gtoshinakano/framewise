// Pure helpers of the line editor on the Results tab: no React and no fetch, so the node tests can import them.
// Times are episode seconds unless a name says otherwise.
import type { Cut, TranscriptSegment, Waveform } from './api'

export type Span = { start: number; end: number }
/** A line cannot be made shorter than this by dragging an edge. */
export const MIN_LINE = 0.1
/** Length of a line added at the playhead. */
export const NEW_LINE_SECONDS = 1
/** Bars per second asked of the detailed waveform, and the most it is worth drawing. */
export const BINS_PER_SECOND = 50
export const MAX_BINS = 1500

const round = (value: number) => Math.round(value * 1000) / 1000

/**
 * The stretch of audio the graph shows: the cut, widened to the whole of every line that overlaps it.
 * Lines now hold whole sentences, so one can start before the cut or end after it.
 */
export function windowFor(cut: Pick<Cut, 'start_time' | 'end_time'>, lines: Pick<TranscriptSegment, 'start_time' | 'end_time'>[]): Span {
  let start = cut.start_time, end = cut.end_time
  for (const line of lines) {
    if (line.end_time <= cut.start_time || line.start_time >= cut.end_time) continue
    start = Math.min(start, line.start_time)
    end = Math.max(end, line.end_time)
  }
  return { start: Math.max(0, start), end }
}

/** How many bins to ask for a window. */
export function binsFor(window: Span) {
  return Math.max(50, Math.min(MAX_BINS, Math.round((window.end - window.start) * BINS_PER_SECOND)))
}

/**
 * Where a dragged edge may go: inside `bounds`, at least MIN_LINE from the other edge, rounded to the millisecond.
 * The `line` is the span as it is now, so the edge that is not dragged is the fixed one.
 */
export function clampEdge(edge: 'start' | 'end', value: number, line: Span, bounds: Span): number {
  if (edge === 'start') return round(Math.max(bounds.start, Math.min(value, line.end - MIN_LINE)))
  return round(Math.min(bounds.end, Math.max(value, line.start + MIN_LINE)))
}

/** A new line at the playhead, one second long and always inside the cut. */
export function newLineSpan(cut: Pick<Cut, 'start_time' | 'end_time'>, playhead: number): Span {
  const length = Math.min(NEW_LINE_SECONDS, cut.end_time - cut.start_time)
  const start = Math.max(cut.start_time, Math.min(playhead, cut.end_time - length))
  return { start: round(start), end: round(start + length) }
}

/** Ids from `from` to `to` inclusive in list order, whichever comes first. A missing id gives an empty list. */
export function contiguousRange(order: string[], from: string, to: string): string[] {
  const a = order.indexOf(from), b = order.indexOf(to)
  if (a < 0 || b < 0) return []
  return order.slice(Math.min(a, b), Math.max(a, b) + 1)
}

const ASCII = /^[\x00-\x7f]$/
/** The text of two lines run together: a space between words, none where a language is written without them. */
export function mergeText(texts: string[]): string {
  let joined = ''
  for (const raw of texts) {
    const text = raw.trim()
    if (!text) continue
    if (joined && ASCII.test(joined.slice(-1)) && ASCII.test(text[0])) joined += ' '
    joined += text
  }
  return joined
}

type Word = [number, number, string]
function wordsOf(line: Pick<TranscriptSegment, 'words_json'>): Word[] | null {
  if (!line.words_json) return null
  try {
    const words = JSON.parse(line.words_json)
    return Array.isArray(words) && words.length && words.every(word => Array.isArray(word) && word.length >= 3) ? words as Word[] : null
  } catch { return null }
}

/**
 * Where to divide a line when the text is cut at `caret` (a character offset into `text`).
 * With the aligner's word times and the text they were made from, the cut falls between two words (a word the caret is
 * inside goes to the side where most of it is). Otherwise it is the caret's share of the line's length.
 * `null` when the caret leaves nothing on one side.
 */
export function splitTime(line: Pick<TranscriptSegment, 'start_time' | 'end_time' | 'words_json'>, text: string, caret: number): number | null {
  const left = text.slice(0, caret).trim(), right = text.slice(caret).trim()
  if (!left || !right) return null
  const words = wordsOf(line)
  const squash = (value: string) => value.replace(/\s+/g, '')
  if (words && squash(words.map(word => String(word[2])).join('')) === squash(text)) {
    // Place each word in the text, in order. Words only count while they still spell the text (an edit drops back to the proportion).
    const ranges: Span[] = []
    let cursor = 0
    for (const [, , word] of words) {
      const token = String(word).trim()
      const at = token ? text.indexOf(token, cursor) : -1
      if (at < 0) { ranges.length = 0; break }
      ranges.push({ start: at, end: at + token.length })
      cursor = at + token.length
    }
    if (ranges.length === words.length) {
      const before = ranges.filter(range => (range.start + range.end) / 2 < caret).length
      if (before > 0 && before < words.length) return round(words[before - 1][1])
    }
  }
  const share = Math.max(0, Math.min(1, caret / text.length))
  const at = line.start_time + (line.end_time - line.start_time) * share
  return round(Math.max(line.start_time + 0.01, Math.min(line.end_time - 0.01, at)))
}

/** `count` bars for [start, end) taken from the episode overview, used until (or without) the detailed waveform. */
export function overviewBars(waveform: Waveform | null, window: Span, count: number): number[] {
  const length = window.end - window.start
  if (!waveform?.peaks.length || waveform.duration <= 0 || length <= 0) return []
  const peaks = waveform.peaks
  return Array.from({ length: count }, (_, index) => {
    const from = Math.max(0, Math.min(peaks.length - 1, Math.floor((window.start + length * index / count) / waveform.duration * peaks.length)))
    const to = Math.max(from + 1, Math.min(peaks.length, Math.ceil((window.start + length * (index + 1) / count) / waveform.duration * peaks.length)))
    let peak = 0
    for (let position = from; position < to; position++) peak = Math.max(peak, peaks[position])
    return peak
  })
}

/** How a line sits against the cut, for the chip beside it. */
export function crossing(line: Pick<TranscriptSegment, 'start_time' | 'end_time'>, cut: Pick<Cut, 'start_time' | 'end_time'>): 'before' | 'after' | 'both' | null {
  const before = line.start_time < cut.start_time - 0.001, after = line.end_time > cut.end_time + 0.001
  return before && after ? 'both' : before ? 'before' : after ? 'after' : null
}

/** Position of an episode time on the graph, in percent of the window. */
export function percentIn(window: Span, time: number) {
  const length = window.end - window.start
  return length > 0 ? Math.max(0, Math.min(100, (time - window.start) / length * 100)) : 0
}

/** The time under a pointer: `fraction` is 0..1 across the graph. */
export function timeAt(window: Span, fraction: number) {
  return round(window.start + (window.end - window.start) * Math.max(0, Math.min(1, fraction)))
}

/** `HH:MM:SS.cc` of a time in seconds. */
export function stamp(seconds: number) {
  const value = Math.max(0, seconds)
  const minutes = Math.floor(value / 60)
  return `${Math.floor(minutes / 60).toString().padStart(2, '0')}:${(minutes % 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`
}

const SPEAKER_COLORS = ['#6aa8ff', '#e2709c', '#4cc9c0', '#f0a64a', '#b496ff', '#7bd88f', '#e5d25a', '#ff8a94']
/** The same speaker label always gets the same colour, so a change of speaker is visible at a glance. */
export function speakerColor(label: string) {
  let hash = 0
  for (const char of label) hash = (hash * 31 + char.charCodeAt(0)) >>> 0
  return SPEAKER_COLORS[hash % SPEAKER_COLORS.length]
}
