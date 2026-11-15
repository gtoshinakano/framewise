import { useCallback, useState } from 'react'
import type { Cut } from './api'

export type CutFilterKey = 'needs_analysis' | 'no_boxes' | 'unknown_characters' | 'needs_consistency' | 'needs_summary' | 'job_failed' | 'long_cut'

export type CutFilterGroup = 'Analysis' | 'Characters' | 'Text steps' | 'Health' | 'Length'

export type CutFilter = { key: CutFilterKey; group: CutFilterGroup; label: string; hint: string; color: string; matches: (cut: Cut) => boolean }

/** How the timeline shows cuts that match none of the active filters. */
export type FilterDisplay = 'dim' | 'hide'

/** The order groups appear in the filters popover. */
export const FILTER_GROUPS: CutFilterGroup[] = ['Analysis', 'Characters', 'Text steps', 'Health', 'Length']

export const LONG_CUT_SECONDS = 15

// Cuts with analysis disabled are left out on purpose: nothing needs doing for them.
const enabled = (cut: Cut) => cut.active && cut.analysis_enabled
const expectsCharacters = (cut: Cut) => enabled(cut) && !cut.no_characters

/** Each filter answers one question: does this cut still need a person or a queued run? */
export const CUT_FILTERS: CutFilter[] = [
  { key: 'needs_analysis', group: 'Analysis', label: 'No visual analysis', color: '#f2a65a', hint: 'Frames and video have not been analyzed yet',
    matches: cut => enabled(cut) && !cut.has_visual_results },
  { key: 'no_boxes', group: 'Characters', label: 'No boxes', color: '#e5646c', hint: 'No first-frame character boxes detected yet (cuts marked as having no characters are skipped)',
    matches: cut => expectsCharacters(cut) && !cut.has_boxes },
  { key: 'unknown_characters', group: 'Characters', label: 'Unknown characters', color: '#4cc9f0', hint: 'A first-frame box has no character assigned',
    matches: cut => expectsCharacters(cut) && !!cut.has_boxes && !!cut.has_unlabeled_boxes },
  { key: 'needs_consistency', group: 'Text steps', label: 'Needs consistency summary', color: '#a58ae8', hint: 'Visual analysis is done but there is no consistency summary',
    matches: cut => enabled(cut) && !!cut.has_visual_results && !cut.has_consistency_summary },
  { key: 'needs_summary', group: 'Text steps', label: 'Needs cut summary', color: '#7bdcb2', hint: 'There is a consistency summary but no final cut summary',
    matches: cut => enabled(cut) && !!cut.has_consistency_summary && !cut.has_summary },
  { key: 'job_failed', group: 'Health', label: 'Last job failed', color: '#ff7a7a', hint: 'The most recent job that included this cut failed',
    matches: cut => cut.active && !!cut.last_job_failed },
  { key: 'long_cut', group: 'Length', label: `Over ${LONG_CUT_SECONDS}s`, color: '#e6d36b', hint: `Cuts longer than ${LONG_CUT_SECONDS} seconds`,
    matches: cut => cut.active && cut.end_time - cut.start_time > LONG_CUT_SECONDS },
]

export function matchesAnyFilter(cut: Cut, active: ReadonlySet<CutFilterKey>) {
  return CUT_FILTERS.some(filter => active.has(filter.key) && filter.matches(cut))
}

const STORAGE_KEY = 'timeline-cut-filters'

/** Reads what was saved: the current `{keys, display}` object or the older plain list of keys. */
export function parseStoredFilters(raw: string | null): { keys: Set<CutFilterKey>; display: FilterDisplay } {
  let saved: unknown = null
  try { saved = JSON.parse(raw || 'null') } catch { /* unreadable: start empty */ }
  const object = saved && typeof saved === 'object' && !Array.isArray(saved) ? saved as { keys?: unknown; display?: unknown } : null
  const list = Array.isArray(saved) ? saved : Array.isArray(object?.keys) ? object.keys : []
  return {
    keys: new Set(CUT_FILTERS.map(filter => filter.key).filter(key => list.includes(key))),
    display: object?.display === 'hide' ? 'hide' : 'dim',
  }
}

/** The filters sorted into their groups, in FILTER_GROUPS order; empty groups are left out. */
export function groupFilters(filters: CutFilter[] = CUT_FILTERS) {
  return FILTER_GROUPS.map(group => ({ group, filters: filters.filter(filter => filter.group === group) })).filter(entry => entry.filters.length)
}

/** Toggle state shared by the timeline and the analysis dialog, remembered between visits. */
export function useCutFilters() {
  const [state, setState] = useState(() => {
    try { return parseStoredFilters(localStorage.getItem(STORAGE_KEY)) } catch { return parseStoredFilters(null) }
  })
  const update = useCallback((change: (next: { keys: Set<CutFilterKey>; display: FilterDisplay }) => void) => {
    setState(current => {
      const next = { keys: new Set(current.keys), display: current.display }
      change(next)
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ keys: [...next.keys], display: next.display })) } catch { /* the filters still work without storage */ }
      return next
    })
  }, [])
  const toggle = useCallback((key: CutFilterKey) => update(next => { if (!next.keys.delete(key)) next.keys.add(key) }), [update])
  const clear = useCallback(() => update(next => next.keys.clear()), [update])
  const setDisplay = useCallback((display: FilterDisplay) => update(next => { next.display = display }), [update])
  return { active: state.keys as ReadonlySet<CutFilterKey>, display: state.display, toggle, clear, setDisplay }
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

/** Cuts that still need something: with filters on, the cuts they show; otherwise any cut some filter would show. */
export function cutsNeedingAction(cuts: Cut[], active: ReadonlySet<CutFilterKey>) {
  return cuts.filter(cut => active.size ? matchesAnyFilter(cut, active) : CUT_FILTERS.some(filter => filter.matches(cut)))
}

/** The next cut after the selected one that needs action, wrapping around to the first. */
export function nextNeedingAction(cuts: Cut[], selectedId: string | null, active: ReadonlySet<CutFilterKey>) {
  const pending = cutsNeedingAction(cuts, active)
  const selectedIndex = cuts.findIndex(cut => cut.id === selectedId)
  return pending.find(cut => cuts.indexOf(cut) > selectedIndex) || pending[0]
}
