import test from 'node:test'
import assert from 'node:assert/strict'
import { addPair, orderCharacters, relevanceOf, removePair, sameProfile, statText, TOPIC_HELP, withRelevance } from '../src/autopilotView.ts'

const profile = () => ({ version: 1, modes: { voices: 'apply' }, claude: { enabled: true, max_items_per_run: 40 }, topics: {}, characters: {}, look_alikes: [], notes: '' })

test('stat text says only what is known', () => {
  assert.equal(statText(undefined), 'no questions yet')
  assert.equal(statText({ asked: 0, answered: 0, recommended: 0 }), 'no questions yet')
  assert.equal(statText({ asked: 181, answered: 0, recommended: 0 }), '181 asked · 0 answered')
  assert.equal(statText({ asked: 181, answered: 57, recommended: 39 }), '181 asked · 57 answered · you took the recommended option 68% of the time')
})

test('auto relevance is the default and is not stored', () => {
  const always = withRelevance(profile(), 'mai', 'always')
  assert.deepEqual(always.characters, { mai: { relevance: 'always' } })
  assert.equal(relevanceOf(always, 'mai'), 'always')
  assert.equal(relevanceOf(always, 'kaede'), 'auto')
  assert.deepEqual(withRelevance(always, 'mai', 'auto').characters, {})
  assert.ok(sameProfile(withRelevance(always, 'mai', 'auto'), profile()))
})

test('look-alike pairs are unordered, unique and never one character twice', () => {
  let pairs = addPair([], 'nodoka', 'mai')
  assert.deepEqual(pairs, [['mai', 'nodoka']])
  assert.equal(addPair(pairs, 'mai', 'nodoka'), pairs)
  assert.equal(addPair(pairs, 'mai', 'mai'), pairs)
  assert.equal(addPair(pairs, '', 'mai'), pairs)
  pairs = addPair(pairs, 'kaede', 'mai')
  assert.deepEqual(removePair(pairs, 0), [['kaede', 'mai']])
})

test('main characters come first, then by name', () => {
  const ordered = orderCharacters([{ title: 'zed' }, { title: 'Mai', is_main: 1 }, { title: 'anna' }])
  assert.deepEqual(ordered.map(item => item.title), ['Mai', 'anna', 'zed'])
})

test('every consistency topic has help text', () => {
  for (const topic of ['pose', 'gaze', 'state', 'speaking', 'emotion', 'position', 'identity', 'camera', 'objects', 'other']) assert.ok(TOPIC_HELP[topic]?.label, topic)
})
