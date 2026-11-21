import test from 'node:test'
import assert from 'node:assert/strict'
import { parseEditorRoute, editorPath, defaultTab, routeFromLocation } from '../src/cutRoute.ts'

const E = 'a1b2c3d4-0000-4000-8000-000000000001'
const C = 'c0ffee00-0000-4000-8000-000000000002'

test('an episode address has no cut', () => {
  assert.deepEqual(parseEditorRoute(`/episode/${E}`), { episodeId: E, cutId: null, analysis: false, tab: 'results', section: 'characters' })
})

test('a cut address selects the cut without opening the dialog', () => {
  const route = parseEditorRoute(`/episode/${E}/cut/${C}`)
  assert.equal(route.cutId, C)
  assert.equal(route.analysis, false)
})

test('the analysis places', () => {
  const place = suffix => { const r = parseEditorRoute(`/episode/${E}/cut/${C}/analysis${suffix}`); return [r.analysis, r.tab, r.section] }
  assert.deepEqual(place(''), [true, 'results', 'characters'])
  assert.deepEqual(place('/run'), [true, 'run', 'characters'])
  assert.deepEqual(place('/characters'), [true, 'run', 'characters'])
  assert.deepEqual(place('/target-frames'), [true, 'run', 'targets'])
  assert.deepEqual(place('/assistant'), [true, 'assistant', 'characters'])
  assert.deepEqual(place('/nonsense'), [true, 'results', 'characters'])
})

test('paths that are not episode addresses, or carry odd ids, are not routes', () => {
  assert.equal(parseEditorRoute('/'), null)
  assert.equal(parseEditorRoute('/storage'), null)
  assert.equal(parseEditorRoute('/project/abc'), null)
  assert.equal(parseEditorRoute('/episode/'), null)
  assert.equal(parseEditorRoute('/episode/a%2Fb'), null)
  assert.equal(parseEditorRoute(`/episode/${E}/cut/x y`).cutId, null)
})

test('editorPath builds the canonical address and round-trips', () => {
  const cases = [
    { episodeId: E },
    { episodeId: E, cutId: C },
    { episodeId: E, cutId: C, analysis: true, tab: 'results' },
    { episodeId: E, cutId: C, analysis: true, tab: 'run', section: 'characters' },
    { episodeId: E, cutId: C, analysis: true, tab: 'run', section: 'targets' },
    { episodeId: E, cutId: C, analysis: true, tab: 'assistant' },
  ]
  for (const state of cases) {
    const path = editorPath(state)
    const back = parseEditorRoute(path)
    assert.equal(editorPath(back), path)
    assert.equal(back.episodeId, E)
  }
  assert.equal(editorPath({ episodeId: E, cutId: C, analysis: true, tab: 'run', section: 'targets' }), `/episode/${E}/cut/${C}/analysis/target-frames`)
  // the dialog is only part of the address while it is open, and a cut is needed for it
  assert.equal(editorPath({ episodeId: E, analysis: true, tab: 'results' }), `/episode/${E}`)
  assert.equal(editorPath({ episodeId: E, cutId: C, analysis: false, tab: 'run' }), `/episode/${E}/cut/${C}`)
})

test('a new dialog starts on Run only for an active cut without results', () => {
  assert.equal(defaultTab({ active: true, has_analysis_results: false }), 'run')
  assert.equal(defaultTab({ active: true, has_analysis_results: true }), 'results')
  assert.equal(defaultTab({ active: true }), 'results')
  assert.equal(defaultTab({ active: false, has_analysis_results: false }), 'results')
})

test('routeFromLocation prefers the hash and accepts a pasted path address', () => {
  assert.equal(routeFromLocation('#/episode/x', '/'), '/episode/x')
  assert.equal(routeFromLocation('', `/episode/${E}/cut/${C}`), `/episode/${E}/cut/${C}`)
  assert.equal(routeFromLocation('', '/'), '/')
  assert.equal(routeFromLocation('#', '/'), '/')
})
