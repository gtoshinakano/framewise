import { useEffect, useRef } from 'react'

export type ShortcutAction = 'playPause' | 'split' | 'undo' | 'redo' | 'previous' | 'next' | 'nextNeeding' | 'filters' | 'clearSelection'

const TYPING = new Set(['INPUT', 'TEXTAREA', 'SELECT'])

/**
 * Episode editor keyboard shortcuts. They stay quiet while the user types in a field, while a
 * dialog is open (anything with aria-modal), and on buttons and the video for Space, which those
 * already handle themselves.
 */
export function useEditorShortcuts(enabled: boolean, actions: Partial<Record<ShortcutAction, () => void>>) {
  const latest = useRef(actions)
  latest.current = actions
  useEffect(() => {
    if (!enabled) return
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat) return
      const target = event.target as HTMLElement | null
      if (target && (target.isContentEditable || TYPING.has(target.tagName))) return
      if (document.querySelector('[aria-modal="true"]')) return
      const mod = event.ctrlKey || event.metaKey
      const key = event.key.toLowerCase()
      let action: ShortcutAction | null = null
      if (mod && !event.altKey && key === 'z') action = event.shiftKey ? 'redo' : 'undo'
      else if (mod && !event.altKey && key === 'y') action = 'redo'
      else if (mod || event.altKey) return
      else if (event.key === ' ') { if (target && (target.tagName === 'BUTTON' || target.tagName === 'VIDEO' || target.getAttribute('role') === 'slider')) return; action = 'playPause' }
      else if (key === 's') action = 'split'
      else if (event.key === '[') action = 'previous'
      else if (event.key === ']') action = 'next'
      else if (key === 'n') action = 'nextNeeding'
      else if (key === 'f') action = 'filters'
      else if (event.key === 'Escape') action = 'clearSelection'
      const run = action && latest.current[action]
      if (run) { event.preventDefault(); run() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled])
}
