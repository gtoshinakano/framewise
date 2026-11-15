import { useCallback, useState } from 'react'

/** Per-viewer layout preferences (panel sizes, collapsed sections). Everything still works without storage. */
const PREFIX = 'framewise.layout.'

export function readPref<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : JSON.parse(raw) as T
  } catch { return fallback }
}

export function writePref(key: string, value: unknown) {
  try { localStorage.setItem(PREFIX + key, JSON.stringify(value)) } catch { /* private window or blocked storage */ }
}

/** useState that remembers its value between visits. */
export function usePref<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(() => readPref(key, fallback))
  const update = useCallback((next: T | ((current: T) => T)) => {
    setValue(current => {
      const resolved = typeof next === 'function' ? (next as (current: T) => T)(current) : next
      writePref(key, resolved)
      return resolved
    })
  }, [key])
  return [value, update] as const
}
