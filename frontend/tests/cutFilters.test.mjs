import { test } from 'node:test'
import assert from 'node:assert/strict'
import { CUT_FILTERS, adjacentCut, cutsNeedingAction, groupFilters, matchesAnyFilter, nextNeedingAction, parseStoredFilters } from '../src/cutFilters.ts'

const base = { active: true, analysis_enabled: true, no_characters: false, has_boxes: true, has_unlabeled_boxes: false,
  has_visual_results: true, has_consistency_summary: true, has_summary: true }
const cut = (id, overrides = {}) => ({ id, ...base, ...overrides })
const cuts = [
  cut('done'),
  cut('no-boxes', { has_boxes: false }),
  cut('plain-ok'),
  cut('unknown', { has_unlabeled_boxes: true }),
  cut('needs-consistency', { has_consistency_summary: false }),
  cut('needs-summary', { has_summary: false }),
  cut('disabled', { analysis_enabled: false, has_visual_results: false, has_boxes: false }),
  cut('no-analysis', { has_visual_results: false, has_consistency_summary: false }),
]

test('each filter picks the cuts that still need that step, and ignores disabled ones', () => {
  const ids = key => cuts.filter(item => CUT_FILTERS.find(filter => filter.key === key).matches(item)).map(c => c.id)
  assert.deepEqual(ids('no_boxes'), ['no-boxes'])
  // Before the per-cut needs load, the box flag stands in for "Name the boxes".
  assert.deepEqual(ids('box_names'), ['unknown'])
  assert.deepEqual(ids('needs_consistency'), ['needs-consistency'])
  assert.deepEqual(ids('needs_summary'), ['needs-summary'])
  assert.deepEqual(ids('needs_analysis'), ['no-analysis'])
})

test('cuts marked as having no characters never need boxes or recognition', () => {
  const quiet = cut('quiet', { no_characters: true, has_boxes: false, has_unlabeled_boxes: true })
  assert.equal(matchesAnyFilter(quiet, new Set(['no_boxes', 'box_names'])), false)
})

test('without filters next and previous are the plain neighbours', () => {
  assert.equal(adjacentCut(cuts, cuts[2], 1, new Set()).id, 'unknown')
  assert.equal(adjacentCut(cuts, cuts[2], -1, new Set()).id, 'no-boxes')
  assert.equal(adjacentCut(cuts, cuts[0], -1, new Set()), undefined)
})

test('with filters on, next and previous jump to the nearest cut that matches', () => {
  const filters = new Set(['needs_consistency', 'needs_summary'])
  assert.equal(adjacentCut(cuts, cuts[0], 1, filters).id, 'needs-consistency')
  assert.equal(adjacentCut(cuts, cuts[4], 1, filters).id, 'needs-summary')
  assert.equal(adjacentCut(cuts, cuts[5], 1, filters), undefined)           // nothing further: no wrap-around
  assert.equal(adjacentCut(cuts, cuts[5], -1, filters).id, 'needs-consistency')
  assert.equal(adjacentCut(cuts, cuts[2], 1, new Set(['box_names'])).id, 'unknown')
})

test('a cut that is not in the list has no neighbours', () => {
  assert.equal(adjacentCut(cuts, cut('elsewhere'), 1, new Set(['no_boxes'])), undefined)
})

test('saved filters are read from the old list and the new object, and default to dimming', () => {
  assert.deepEqual([...parseStoredFilters('["no_boxes","gone"]').keys], ['no_boxes'])
  assert.equal(parseStoredFilters('["no_boxes"]').display, 'dim')
  const saved = parseStoredFilters(JSON.stringify({ keys: ['long_cut', 'needs_summary'], display: 'hide' }))
  assert.deepEqual([...saved.keys].sort(), ['long_cut', 'needs_summary'])
  assert.equal(saved.display, 'hide')
  assert.equal(parseStoredFilters('not json').keys.size, 0)
  assert.equal(parseStoredFilters(null).display, 'dim')
})

test('every filter belongs to a group, listed in group order', () => {
  const groups = groupFilters()
  assert.deepEqual(groups.map(entry => entry.group), ['Needs you', 'Claude', 'Analysis', 'Characters', 'Text steps', 'Health', 'Length'])
  assert.equal(groups.flatMap(entry => entry.filters).length, CUT_FILTERS.length)
  assert.deepEqual(groups[0].filters.map(filter => filter.key),
    ['missed_cut', 'line_split', 'voice', 'speaker_unsure', 'box_names', 'target_frames', 'decisions', 'picture_check', 'lines_to_check', 'failed_run'])
  assert.deepEqual(groups[1].filters.map(filter => filter.key), ['claude_pending', 'agent_answered'])
})

const needs = new Map([
  ['done', { steps: [], people: [], claude: [], answered_by: ['claude'], analysis_off: false }],
  ['no-boxes', { steps: ['boxes'], people: [{ key: 'missed_cut', count: 1, escalation_ids: ['m'] }], claude: [], answered_by: [], analysis_off: false }],
  ['unknown', { steps: [], people: [{ key: 'box_names', count: 1, escalation_ids: [] }], claude: [], answered_by: [], analysis_off: false }],
  ['needs-summary', { steps: ['summary'], people: [{ key: 'decisions', count: 2, escalation_ids: [] }], claude: [{ kind: 'review_question', escalation_id: 'q' }], answered_by: [], analysis_off: false }],
])

test('the Needs you and Claude filters read the per-cut needs', () => {
  const ids = key => cuts.filter(item => matchesAnyFilter(item, new Set([key]), needs)).map(c => c.id)
  assert.deepEqual(ids('missed_cut'), ['no-boxes'])
  assert.deepEqual(ids('decisions'), ['needs-summary'])
  assert.deepEqual(ids('claude_pending'), ['needs-summary'])
  assert.deepEqual(ids('agent_answered'), ['done'])
  // Once loaded, the needs decide: a cut with no needs entry has no box names to settle, whatever its flags say.
  assert.deepEqual(ids('box_names'), ['unknown'])
})

test('with no filter on, next needing action means the next cut that waits for the person', () => {
  assert.deepEqual(cutsNeedingAction(cuts, new Set(), needs).map(c => c.id), ['no-boxes', 'unknown', 'needs-summary'])
  assert.equal(nextNeedingAction(cuts, 'unknown', new Set(), needs).id, 'needs-summary')
  assert.equal(nextNeedingAction(cuts, 'needs-summary', new Set(), needs).id, 'no-boxes')
  assert.equal(nextNeedingAction(cuts, 'needs-consistency', new Set(['no_boxes'])).id, 'no-boxes')
  assert.equal(nextNeedingAction([cut('done')], null, new Set()), undefined)
})

test('the target-frame filter falls back to its flag, and leaves quiet or disabled cuts alone', () => {
  const target = CUT_FILTERS.find(filter => filter.key === 'target_frames')
  const unknown = cut('unknown-target', { has_target_boxes: true, has_unlabeled_target_boxes: true })
  assert.equal(target.matches(unknown), true)
  assert.equal(target.matches(cut('named-target', { has_target_boxes: true, has_unlabeled_target_boxes: false })), false)
  assert.equal(CUT_FILTERS.find(filter => filter.key === 'box_names').matches(unknown), false)
  assert.equal(matchesAnyFilter(cut('quiet', { no_characters: true, has_unlabeled_target_boxes: true }), new Set(['target_frames'])), false)
  assert.equal(matchesAnyFilter(cut('off', { analysis_enabled: false, has_unlabeled_target_boxes: true }), new Set(['target_frames'])), false)
  assert.equal(adjacentCut([cut('a'), unknown], cut('a'), 1, new Set(['target_frames'])).id, 'unknown-target')
})

test('filters saved under their old names read back as the Needs you filters', () => {
  assert.deepEqual([...parseStoredFilters(JSON.stringify({ keys: ['no_boxes', 'unknown_target_characters'], display: 'hide' })).keys], ['target_frames', 'no_boxes'])
  assert.deepEqual([...parseStoredFilters(JSON.stringify(['unknown_characters'])).keys], ['box_names'])
  assert.deepEqual([...parseStoredFilters(JSON.stringify(['no_boxes'])).keys], ['no_boxes'])
})
