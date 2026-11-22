import type { Episode } from './api'

/** Structural equality of parsed JSON. */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((item, index) => deepEqual(item, b[index]))
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>
  const keys = Object.keys(left)
  return keys.length === Object.keys(right).length && keys.every(key => key in right && deepEqual(left[key], right[key]))
}

/** The new list, reusing every item that did not change and the previous list itself when nothing did. */
export function shareList<T extends { id: string }>(previous: T[] | undefined, next: T[]): T[] {
  if (!previous) return next
  const before = new Map(previous.map(item => [item.id, item]))
  let same = previous.length === next.length
  const merged = next.map((item, index) => {
    const old = before.get(item.id)
    const keep = old !== undefined && deepEqual(old, item) ? old : item
    if (keep !== previous[index]) same = false
    return keep
  })
  return same ? previous : merged
}

function shareValue<T>(previous: T | undefined, next: T): T {
  return previous !== undefined && deepEqual(previous, next) ? previous : next
}

/**
 * A reloaded episode that keeps the identity of everything that is unchanged: the cuts, jobs, chunks, settings and the
 * episode itself. Memoized children then skip their render, and `setEpisode` with the same object does nothing.
 */
export function shareEpisode(previous: Episode | null | undefined, next: Episode): Episode {
  if (!previous || previous.id !== next.id) return next
  const shared: Episode = {
    ...next,
    cuts: shareList(previous.cuts, next.cuts),
    archived_cuts: shareList(previous.archived_cuts, next.archived_cuts),
    chunks: shareList(previous.chunks, next.chunks),
    jobs: shareList(previous.jobs, next.jobs),
    audio_ranges: shareList(previous.audio_ranges, next.audio_ranges),
    settings: shareValue(previous.settings, next.settings),
    subtitles: shareValue(previous.subtitles, next.subtitles),
    cut_history: shareValue(previous.cut_history, next.cut_history),
  }
  return deepEqual(previous, shared) ? previous : shared
}
