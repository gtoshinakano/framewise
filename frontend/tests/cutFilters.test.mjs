import { test } from 'node:test'
import assert from 'node:assert/strict'
import { CUT_FILTERS, adjacentCut, groupFilters, matchesAnyFilter, nextNeedingAction, parseStoredFilters } from '../src/cutFilters.ts'

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
  const ids = key => cuts.filter(CUT_FILTERS.find(filter => filter.key === key).matches).map(c => c.id)
  assert.deepEqual(ids('no_boxes'), ['no-boxes'])
  assert.deepEqual(ids('unknown_characters'), ['unknown'])
  assert.deepEqual(ids('needs_consistency'), ['needs-consistency'])
  assert.deepEqual(ids('needs_summary'), ['needs-summary'])
  assert.deepEqual(ids('needs_analysis'), ['no-analysis'])
})

test('cuts marked as having no characters never need boxes or recognition', () => {
  const quiet = cut('quiet', { no_characters: true, has_boxes: false, has_unlabeled_boxes: true })
  assert.equal(matchesAnyFilter(quiet, new Set(['no_boxes', 'unknown_characters'])), false)
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
  assert.equal(adjacentCut(cuts, cuts[2], 1, new Set(['unknown_characters'])).id, 'unknown')
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
  assert.deepEqual(groups.map(entry => entry.group), ['Analysis', 'Characters', 'Text steps', 'Health', 'Length'])
  assert.equal(groups.flatMap(entry => entry.filters).length, CUT_FILTERS.length)
  assert.deepEqual(groups[1].filters.map(filter => filter.key), ['no_boxes', 'unknown_characters', 'unknown_target_characters'])
})

test('next needing action follows the filters and wraps around', () => {
  assert.equal(nextNeedingAction(cuts, 'done', new Set()).id, 'no-boxes')
  assert.equal(nextNeedingAction(cuts, 'needs-summary', new Set()).id, 'no-analysis')
  assert.equal(nextNeedingAction(cuts, 'no-analysis', new Set()).id, 'no-boxes')
  assert.equal(nextNeedingAction(cuts, 'needs-consistency', new Set(['no_boxes'])).id, 'no-boxes')
  assert.equal(nextNeedingAction([cut('done')], null, new Set()), undefined)
})

test('unknown people on the evaluated frames have their own filter, apart from the first frame', () => {
  const target = CUT_FILTERS.find(filter => filter.key === 'unknown_target_characters')
  assert.equal(target.group, 'Characters')
  const unknown = cut('unknown-target', { has_target_boxes: true, has_unlabeled_target_boxes: true })
  assert.equal(target.matches(unknown), true)
  assert.equal(target.matches(cut('named-target', { has_target_boxes: true, has_unlabeled_target_boxes: false })), false)
  // The first-frame filter keeps meaning "ready for visual analysis": it ignores target frames.
  assert.equal(CUT_FILTERS.find(filter => filter.key === 'unknown_characters').matches(unknown), false)
  // Cuts the user marked as having no characters, or turned analysis off for, are left alone.
  assert.equal(matchesAnyFilter(cut('quiet', { no_characters: true, has_unlabeled_target_boxes: true }), new Set(['unknown_target_characters'])), false)
  assert.equal(matchesAnyFilter(cut('off', { analysis_enabled: false, has_unlabeled_target_boxes: true }), new Set(['unknown_target_characters'])), false)
  assert.equal(adjacentCut([cut('a'), unknown], cut('a'), 1, new Set(['unknown_target_characters'])).id, 'unknown-target')
})

test('a filter saved before the target-frame filter existed still reads back', () => {
  assert.deepEqual([...parseStoredFilters(JSON.stringify({ keys: ['no_boxes', 'unknown_target_characters'], display: 'hide' })).keys], ['no_boxes', 'unknown_target_characters'])
  assert.deepEqual([...parseStoredFilters(JSON.stringify(['no_boxes'])).keys], ['no_boxes'])
})
