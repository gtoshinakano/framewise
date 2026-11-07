import { test } from 'node:test'
import assert from 'node:assert/strict'
import { CUT_FILTERS, adjacentCut, matchesAnyFilter } from '../src/cutFilters.ts'

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
