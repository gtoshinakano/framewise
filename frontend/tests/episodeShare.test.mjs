import { test } from 'node:test'
import assert from 'node:assert/strict'
import { deepEqual, shareEpisode, shareList } from '../src/episodeShare.ts'

const cut = (id, over = {}) => ({ id, cut_index: 0, title: '', start_time: 0, end_time: 1, results: {}, ...over })
const episode = (over = {}) => ({
  id: 'e', title: 'Episode', status: 'ready', stage: 'Ready', cuts: [cut('a'), cut('b')], archived_cuts: [], chunks: [{ id: 'k', title: '' }],
  jobs: [{ id: 'j1', status: 'done', cuts: [{ id: 'a', completed: true }] }], audio_ranges: [], settings: { proxy_width: 960 }, subtitles: { language: null, run: null }, cut_history: { undo: null, redo: null }, ...over,
})
// What a reload returns: equal values, new objects.
const reload = value => JSON.parse(JSON.stringify(value))

test('deepEqual compares parsed JSON structurally', () => {
  assert.ok(deepEqual({ a: [1, { b: 2 }] }, { a: [1, { b: 2 }] }))
  assert.ok(!deepEqual({ a: 1 }, { a: 1, b: undefined === 1 }))
  assert.ok(!deepEqual([1, 2], [1, 2, 3]))
  assert.ok(!deepEqual({ a: null }, { a: {} }))
  assert.ok(!deepEqual([], {}))
})

test('a reload with nothing new is the previous episode itself', () => {
  const before = episode()
  assert.equal(shareEpisode(before, reload(before)), before)
})

test('unchanged cuts keep their identity when another one changes', () => {
  const before = episode()
  const next = reload(before); next.cuts[1].title = 'Renamed'
  const shared = shareEpisode(before, next)
  assert.notEqual(shared, before)
  assert.equal(shared.cuts[0], before.cuts[0])
  assert.notEqual(shared.cuts[1], before.cuts[1])
  assert.equal(shared.cuts[1].title, 'Renamed')
  assert.equal(shared.chunks, before.chunks)
  assert.equal(shared.jobs, before.jobs)
  assert.equal(shared.settings, before.settings)
})

test('a changed job list is new but keeps the unchanged jobs', () => {
  const before = episode()
  const next = reload(before)
  next.jobs.unshift({ id: 'j2', status: 'queued', cuts: [] })
  const shared = shareEpisode(before, next)
  assert.equal(shared.jobs.length, 2)
  assert.equal(shared.jobs[1], before.jobs[0])
  assert.equal(shared.cuts, before.cuts)
})

test('scalar changes, added cuts and reordering are honoured', () => {
  const before = episode()
  assert.equal(shareEpisode(before, { ...reload(before), status: 'processing' }).status, 'processing')
  const added = reload(before); added.cuts.push(cut('c'))
  assert.deepEqual(shareEpisode(before, added).cuts.map(item => item.id), ['a', 'b', 'c'])
  const swapped = reload(before); swapped.cuts.reverse()
  const result = shareEpisode(before, swapped)
  assert.deepEqual(result.cuts.map(item => item.id), ['b', 'a'])
  assert.equal(result.cuts[0], before.cuts[1])
})

test('another episode is never merged with the previous one', () => {
  const before = episode()
  const other = episode({ id: 'other' })
  assert.equal(shareEpisode(before, other), other)
  assert.equal(shareEpisode(null, other), other)
})

test('shareList returns the previous array when every item is equal', () => {
  const before = [cut('a'), cut('b')]
  assert.equal(shareList(before, reload(before)), before)
  assert.equal(shareList(undefined, before), before)
})
