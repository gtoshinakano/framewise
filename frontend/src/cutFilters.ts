import { useCallback, useState } from 'react'
import type { Cut } from './api'
import { EMPTY_NEEDS, hasNeed, NEED_LABEL, NEED_ORDER, type CutNeeds, type NeedKey, type NeedsMap } from './cutNeedsView.ts'

export type CutFilterKey = NeedKey | 'claude_pending' | 'agent_answered' | 'needs_analysis' | 'no_boxes' | 'needs_consistency' | 'needs_summary' | 'job_failed' | 'long_cut'

export type CutFilterGroup = 'Needs you' | 'Claude' | 'Analysis' | 'Characters' | 'Text steps' | 'Health' | 'Length'

export type CutFilter = { key: CutFilterKey; group: CutFilterGroup; label: string; hint: string; color: string; matches: (cut: Cut, needs?: CutNeeds) => boolean }

/** How the timeline shows cuts that match none of the active filters. */
export type FilterDisplay = 'dim' | 'hide'

/** The order groups appear in the filters popover. */
export const FILTER_GROUPS: CutFilterGroup[] = ['Needs you', 'Claude', 'Analysis', 'Characters', 'Text steps', 'Health', 'Length']

export const LONG_CUT_SECONDS = 15

// Cuts with analysis disabled are left out on purpose: nothing needs doing for them.
const enabled = (cut: Cut) => cut.active && cut.analysis_enabled
const expectsCharacters = (cut: Cut) => enabled(cut) && !cut.no_characters

const NEED_LOOK: Record<NeedKey, { color: string; hint: string; fallback?: (cut: Cut) => boolean }> = {
  missed_cut: { color: '#ff9f6e', hint: 'The agent found a possible hard cut inside this cut and asks you whether to split it' },
  line_split: { color: '#f7b2d9', hint: 'A transcript line crosses this cut\'s hard cut and the agent asks how to divide it' },
  voice: { color: '#f0c979', hint: 'Spoken lines in this cut have no character, or the agent asks who a voice is' },
  box_names: { color: '#4cc9f0', hint: 'A first-frame box has no character, or the agent asks who a person is',
    fallback: cut => expectsCharacters(cut) && !!cut.has_boxes && !!cut.has_unlabeled_boxes },
  target_frames: { color: '#e2709c', hint: 'Target frames are flagged for review or show an unnamed person',
    fallback: cut => expectsCharacters(cut) && !!cut.has_unlabeled_target_boxes },
  decisions: { color: '#f0a64a', hint: 'The consistency review asks questions that have no answer saved since it was written' },
  picture_check: { color: '#ff7a8a', hint: 'A picture of the cut contradicts what the analysis says' },
  lines_to_check: { color: '#d9c27a', hint: 'The speaker of a line was attributed with medium or low confidence' },
  failed_run: { color: '#ff6b6b', hint: 'A failed run on this cut waits for you to retry or dismiss it' },
}

/** Each filter answers one question: does this cut still need a person or a queued run? */
export const CUT_FILTERS: CutFilter[] = [
  ...NEED_ORDER.map((key): CutFilter => ({ key, group: 'Needs you', label: NEED_LABEL[key], color: NEED_LOOK[key].color, hint: NEED_LOOK[key].hint,
    // Before the per-cut needs have loaded, the cut's own flags stand in where they can.
    matches: (cut, needs) => cut.active && (needs ? hasNeed(needs, key) : !!NEED_LOOK[key].fallback?.(cut)) })),
  { key: 'claude_pending', group: 'Claude', label: 'Waiting for Claude', color: '#b496ff', hint: 'Claude (the second reviewer) has a question about this cut to answer',
    matches: (cut, needs) => cut.active && !!needs?.claude.length },
  { key: 'agent_answered', group: 'Claude', label: 'Answered by an agent', color: '#8fb8ff', hint: 'Claude, the local model or a rule applied an answer on this cut: spot-check it',
    matches: (cut, needs) => cut.active && !!needs?.answered_by.length },
  { key: 'needs_analysis', group: 'Analysis', label: 'No visual analysis', color: '#f2a65a', hint: 'The first frame and the keyframes (with their performance beats) have not been analyzed yet',
    matches: cut => enabled(cut) && !cut.has_visual_results },
  { key: 'no_boxes', group: 'Characters', label: 'No boxes', color: '#e5646c', hint: 'No first-frame character boxes detected yet (cuts marked as having no characters are skipped)',
    matches: cut => expectsCharacters(cut) && !cut.has_boxes },
  { key: 'needs_consistency', group: 'Text steps', label: 'Needs consistency summary', color: '#a58ae8', hint: 'Visual analysis and beats are done but there is no consistency summary',
    matches: cut => enabled(cut) && !!cut.has_visual_results && !cut.has_consistency_summary },
  { key: 'needs_summary', group: 'Text steps', label: 'Needs cut summary', color: '#7bdcb2', hint: 'There is a consistency summary but no final cut summary',
    matches: cut => enabled(cut) && !!cut.has_consistency_summary && !cut.has_summary },
  { key: 'job_failed', group: 'Health', label: 'Last job failed', color: '#ff7a7a', hint: 'The most recent job that included this cut failed',
    matches: cut => cut.active && !!cut.last_job_failed },
  { key: 'long_cut', group: 'Length', label: `Over ${LONG_CUT_SECONDS}s`, color: '#e6d36b', hint: `Cuts longer than ${LONG_CUT_SECONDS} seconds`,
    matches: cut => cut.active && cut.end_time - cut.start_time > LONG_CUT_SECONDS },
]

const PERSON_FILTERS = CUT_FILTERS.filter(filter => filter.group === 'Needs you')
/** The active filters that are a person's action (the analysis dialog jumps to these when a cut opens). */
export const isNeedKey = (key: CutFilterKey): key is NeedKey => (NEED_ORDER as string[]).includes(key)

/** Filters that match a cut, given the episode's per-cut needs. */
export function matchingFilters(cut: Cut, active: ReadonlySet<CutFilterKey>, needs: NeedsMap = EMPTY_NEEDS) {
  const own = needs.get(cut.id)
  return CUT_FILTERS.filter(filter => active.has(filter.key) && filter.matches(cut, own))
}

/** Whether the cut waits for the person in any way (what "next" means with no filter on). */
export function needsPerson(cut: Cut, needs: NeedsMap = EMPTY_NEEDS) {
  const own = needs.get(cut.id)
  return PERSON_FILTERS.some(filter => filter.matches(cut, own))
}

export function matchesAnyFilter(cut: Cut, active: ReadonlySet<CutFilterKey>, needs: NeedsMap = EMPTY_NEEDS) {
  const own = needs.get(cut.id)
  return CUT_FILTERS.some(filter => active.has(filter.key) && filter.matches(cut, own))
}

const STORAGE_KEY = 'timeline-cut-filters'

// Filters that were folded into the "Needs you" group keep a saved selection.
const RENAMED: Record<string, CutFilterKey> = { unknown_characters: 'box_names', unknown_target_characters: 'target_frames' }

/** Reads what was saved: the current `{keys, display}` object or the older plain list of keys. */
export function parseStoredFilters(raw: string | null): { keys: Set<CutFilterKey>; display: FilterDisplay } {
  let saved: unknown = null
  try { saved = JSON.parse(raw || 'null') } catch { /* unreadable: start empty */ }
  const object = saved && typeof saved === 'object' && !Array.isArray(saved) ? saved as { keys?: unknown; display?: unknown } : null
  const list = (Array.isArray(saved) ? saved : Array.isArray(object?.keys) ? object.keys : []).map(key => RENAMED[key as string] ?? key)
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
export function adjacentCut(cuts: Cut[], current: Cut, direction: 1 | -1, active: ReadonlySet<CutFilterKey>, needs: NeedsMap = EMPTY_NEEDS) {
  const start = cuts.findIndex(cut => cut.id === current.id)
  if (start < 0) return undefined
  if (!active.size) return cuts[start + direction]
  for (let index = start + direction; index >= 0 && index < cuts.length; index += direction) {
    if (matchesAnyFilter(cuts[index], active, needs)) return cuts[index]
  }
  return undefined
}

/** Cuts that still need something: with filters on, the cuts they show; otherwise the cuts that wait for the person. */
export function cutsNeedingAction(cuts: Cut[], active: ReadonlySet<CutFilterKey>, needs: NeedsMap = EMPTY_NEEDS) {
  return cuts.filter(cut => active.size ? matchesAnyFilter(cut, active, needs) : needsPerson(cut, needs))
}

/** The next cut after the selected one that needs action, wrapping around to the first. */
export function nextNeedingAction(cuts: Cut[], selectedId: string | null, active: ReadonlySet<CutFilterKey>, needs: NeedsMap = EMPTY_NEEDS) {
  const pending = cutsNeedingAction(cuts, active, needs)
  const selectedIndex = cuts.findIndex(cut => cut.id === selectedId)
  return pending.find(cut => cuts.indexOf(cut) > selectedIndex) || pending[0]
}
