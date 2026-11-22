import { useCallback, useRef } from 'react'

/**
 * A function with a fixed identity that always runs the latest `handler`. Memoized children can take it as a prop
 * without re-rendering whenever the parent does, and it still sees the parent's current state.
 */
export function useEvent<Args extends unknown[], Result>(handler: (...args: Args) => Result): (...args: Args) => Result {
  const latest = useRef(handler)
  latest.current = handler
  return useCallback((...args: Args) => latest.current(...args), [])
}
