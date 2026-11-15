/** Timecodes for the episode editor: one display format and a forgiving parser for typed times. */

const pad = (value: number, width = 2) => String(value).padStart(width, '0')

/** Whole milliseconds, so 1.0004 and 1.0 show and compare the same. */
export function roundMs(seconds: number) {
  return Math.round(seconds * 1000) / 1000
}

/** `M:SS.mmm`, or `H:MM:SS.mmm` from one hour up. Invalid input shows as an em dash. */
export function formatTimecode(seconds: number | null | undefined, { precision = 3 }: { precision?: 0 | 1 | 2 | 3 } = {}) {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const sign = seconds < 0 ? '-' : ''
  const total = Math.round(Math.abs(seconds) * 1000)
  const factor = 10 ** (3 - precision)
  const fraction = Math.floor((total % 1000) / factor)
  const whole = Math.floor(total / 1000)
  const hours = Math.floor(whole / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  const secs = whole % 60
  const tail = precision ? `.${pad(fraction, precision)}` : ''
  return hours ? `${sign}${hours}:${pad(minutes)}:${pad(secs)}${tail}` : `${sign}${minutes}:${pad(secs)}${tail}`
}

/** A length such as `8.14s` or `1:05.2`, for durations next to a range. */
export function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return '—'
  return seconds < 60 ? `${seconds.toFixed(seconds < 10 ? 2 : 1)}s` : formatTimecode(seconds, { precision: 1 })
}

function parseAbsolute(text: string): number | null {
  const value = text.replace(/s$/i, '').trim()
  if (!value) return null
  const parts = value.split(':')
  if (parts.length > 3) return null
  const last = parts[parts.length - 1]
  if (!/^\d+(\.\d+)?$|^\.\d+$/.test(last)) return null
  if (parts.slice(0, -1).some(part => !/^\d+$/.test(part))) return null
  const seconds = Number(last)
  if (parts.length === 1) return seconds
  // With colons, a field that overflows is almost always a typo: refuse it instead of guessing.
  if (seconds >= 60) return null
  if (parts.length === 2) return Number(parts[0]) * 60 + seconds
  const minutes = Number(parts[1])
  if (minutes >= 60) return null
  return Number(parts[0]) * 3600 + minutes * 60 + seconds
}

/**
 * Reads `83.4`, `1:23.4`, `01:23.456`, `1:02:03.4`, `83,4` and `12s`. A leading `+` or `-` is relative
 * to `base` (for example `+0.25` moves a boundary a quarter second later). Returns seconds rounded to
 * milliseconds, or null when the text is not a time.
 */
export function parseTimecode(text: string, base?: number | null): number | null {
  const value = text.trim().replace(/,/g, '.')
  const relative = /^([+-])\s*(.+)$/.exec(value)
  if (relative) {
    if (base == null || !Number.isFinite(base)) return null
    const offset = parseAbsolute(relative[2])
    return offset === null ? null : roundMs(base + (relative[1] === '-' ? -offset : offset))
  }
  const absolute = parseAbsolute(value)
  return absolute === null ? null : roundMs(absolute)
}

/** Clamp to [min, max] and round to milliseconds. */
export function clampTime(seconds: number, min = 0, max = Number.POSITIVE_INFINITY) {
  return roundMs(Math.min(max, Math.max(min, seconds)))
}

/** The nudge size for an arrow key: 0.1 s, Shift 1 s, Alt 0.01 s. */
export function nudgeStep(event: { shiftKey: boolean; altKey: boolean }) {
  return event.shiftKey ? 1 : event.altKey ? 0.01 : 0.1
}
