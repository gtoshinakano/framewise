import { useCallback, useState } from 'react'
import type { Cut } from './api'

export type CutFilterKey = 'needs_analysis' | 'no_boxes' | 'unknown_characters' | 'needs_consistency' | 'needs_summary' | 'job_failed' | 'long_cut'

export type CutFilter = { key: CutFilterKey; label: string; hint: string; color: string; matches: (cut: Cut) => boolean }

export const LONG_CUT_SECONDS = 15

// Cuts with analysis disabled are left out on purpose: nothing needs doing for them.
const enabled = (cut: Cut) => cut.active && cut.analysis_enabled
const expectsCharacters = (cut: Cut) => enabled(cut) && !cut.no_characters

/** Each filter answers one question: does this cut still need a person or a queued run? */
export const CUT_FILTERS: CutFilter[] = [
  { key: 'needs_analysis', label: 'No visual analysis', color: '#f2a65a', hint: 'Frames and video have not been analyzed yet',
    matches: cut => enabled(cut) && !cut.has_visual_results },
  { key: 'no_boxes', label: 'No boxes', color: '#e5646c', hint: 'No first-frame character boxes detected yet (cuts marked as having no characters are skipped)',
    matches: cut => expectsCharacters(cut) && !cut.has_boxes },
  { key: 'unknown_characters', label: 'Unknown characters', color: '#4cc9f0', hint: 'A first-frame box has no character assigned',
    matches: cut => expectsCharacters(cut) && !!cut.has_boxes && !!cut.has_unlabeled_boxes },
  { key: 'needs_consistency', label: 'Needs consistency summary', color: '#a58ae8', hint: 'Visual analysis is done but there is no consistency summary',
    matches: cut => enabled(cut) && !!cut.has_visual_results && !cut.has_consistency_summary },
  { key: 'needs_summary', label: 'Needs cut summary', color: '#7bdcb2', hint: 'There is a consistency summary but no final cut summary',
    matches: cut => enabled(cut) && !!cut.has_consistency_summary && !cut.has_summary },
  { key: 'job_failed', label: 'Last job failed', color: '#ff7a7a', hint: 'The most recent job that included this cut failed',
    matches: cut => cut.active && !!cut.last_job_failed },
  { key: 'long_cut', label: `Over ${LONG_CUT_SECONDS}s`, color: '#e6d36b', hint: `Cuts longer than ${LONG_CUT_SECONDS} seconds`,
    matches: cut => cut.active && cut.end_time - cut.start_time > LONG_CUT_SECONDS },
]

export function matchesAnyFilter(cut: Cut, active: ReadonlySet<CutFilterKey>) {
  return CUT_FILTERS.some(filter => active.has(filter.key) && filter.matches(cut))
}

const STORAGE_KEY = 'timeline-cut-filters'

/** Toggle state shared by the timeline and the analysis dialog, remembered between visits. */
export function useCutFilters() {
  const [active, setActive] = useState<Set<CutFilterKey>>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
      return new Set(CUT_FILTERS.map(filter => filter.key).filter(key => Array.isArray(saved) && saved.includes(key)))
    } catch { return new Set() }
  })
  const update = useCallback((change: (next: Set<CutFilterKey>) => void) => {
    setActive(current => {
      const next = new Set(current)
      change(next)
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify([...next])) } catch { /* the filters still work without storage */ }
      return next
    })
  }, [])
  const toggle = useCallback((key: CutFilterKey) => update(next => { if (!next.delete(key)) next.add(key) }), [update])
  const clear = useCallback(() => update(next => next.clear()), [update])
  return { active, toggle, clear }
}

/**
 * The cut before (-1) or after (+1) the current one. With no filter on this is the plain neighbour;
 * with filters on it is the nearest cut in that direction that still matches one of them.
 */
export function adjacentCut(cuts: Cut[], current: Cut, direction: 1 | -1, active: ReadonlySet<CutFilterKey>) {
  const start = cuts.findIndex(cut => cut.id === current.id)
  if (start < 0) return undefined
  if (!active.size) return cuts[start + direction]
  for (let index = start + direction; index >= 0 && index < cuts.length; index += direction) {
    if (matchesAnyFilter(cuts[index], active)) return cuts[index]
  }
  return undefined
}
