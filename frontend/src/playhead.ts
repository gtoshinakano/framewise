import { useRef, useSyncExternalStore } from 'react'

/**
 * The editor's playhead, in episode seconds, kept outside React state.
 *
 * A playing video reports its time several times a second. As state of the editor that re-rendered the
 * timeline (hundreds of buttons), the cut list, the inspector and the open dialog every time. Here only
 * the components that show or use the time subscribe (`usePlayheadValue`), and everything else reads
 * `time` when it needs it (a click, a save).
 */
export class PlayheadStore {
  private current = 0
  private seeks = 0
  private readonly listeners = new Set<() => void>()

  get time() { return this.current }
  /** How many times the playhead was moved on purpose (a seek, a scrub), as opposed to playing on. */
  get seekCount() { return this.seeks }

  /** The video reports its time (playing, or after a seek). */
  set(time: number) {
    if (time === this.current) return
    this.current = time
    this.notify()
  }

  /** Move the playhead to `time` on purpose. */
  seek(time: number) {
    this.current = time
    this.seeks += 1
    this.notify()
  }

  /** The video started a seek of its own (the user scrubbed its controls); its new time follows with `set`. */
  markSeek() {
    this.seeks += 1
    this.notify()
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private notify() { for (const listener of [...this.listeners]) listener() }
}

/**
 * A value derived from the playhead. The component re-renders only when the value changes, so `select`
 * must return a primitive (a rounded number, a string, an id), never a new object.
 */
export function usePlayheadValue<T>(store: PlayheadStore, select: (time: number) => T): T {
  const latest = useRef(select)
  latest.current = select
  return useSyncExternalStore(store.subscribe, () => latest.current(store.time))
}
