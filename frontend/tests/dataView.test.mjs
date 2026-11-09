import { test } from 'node:test'
import assert from 'node:assert/strict'
import { durationText, isLongString, looksLikeMarkdown, parseValue, promptOutline, runFacts, startsCollapsed } from '../src/dataView.ts'

test('objects, JSON strings, fenced blocks and double-encoded JSON are all json', () => {
  assert.equal(parseValue({ a: 1 }).kind, 'json')
  assert.deepEqual(parseValue('{"a": 1}').value, { a: 1 })
  const fenced = parseValue('```json\n{"a": [1, 2]}\n```')
  assert.equal(fenced.kind, 'json'); assert.equal(fenced.fenced, true)
  assert.deepEqual(parseValue(JSON.stringify(JSON.stringify({ a: 1 }))).value, { a: 1 })
})

test('prose and broken JSON stay text; scalars become their text', () => {
  assert.deepEqual(parseValue('A man walks in.'), { kind: 'text', text: 'A man walks in.' })
  assert.equal(parseValue('{"a": ').kind, 'text')
  assert.equal(parseValue('"just a string"').kind, 'text')
  assert.deepEqual(parseValue(null), { kind: 'text', text: '' })
  assert.deepEqual(parseValue(3), { kind: 'text', text: '3' })
})

test('markdown is only assumed when the text uses it', () => {
  assert.equal(looksLikeMarkdown('## Scene\n- one\n- two'), true)
  assert.equal(looksLikeMarkdown('He said **no** twice'), true)
  assert.equal(looksLikeMarkdown('Plain sentence, with commas. And more.'), false)
})

test('prompt outline finds headings, labels, numbered sections and tags', () => {
  const text = ['Shot overview:', 'text', '## Audio', '<Picture 1> hero', '2. Camera movement', 'noise line'].join('\n')
  assert.deepEqual(promptOutline(text).map(item => [item.line, item.label, item.level]), [[0, 'Shot overview', 1], [2, 'Audio', 2], [3, '<Picture 1> hero', 3], [4, '2. Camera movement', 2]])
})

test('large containers and deep nesting start collapsed; scalars never do', () => {
  assert.equal(startsCollapsed(Array.from({ length: 21 }), 1), true)
  assert.equal(startsCollapsed({ a: 1 }, 0), false)
  assert.equal(startsCollapsed({ a: 1 }, 3), true)
  assert.equal(startsCollapsed([1], 2), true)
  assert.equal(startsCollapsed('x', 9), false)
  assert.equal(isLongString('short'), false); assert.equal(isLongString('a\nb'), true)
})

test('run facts come from nested snapshots in source order', () => {
  const facts = runFacts([{ config: { video_width: 896, video_height: 504, video_fps: 2, additional_context: 'Night scene' } }, { model_id: 'qwen3-vl', local_thinking: false }])
  const map = Object.fromEntries(facts.map(item => [item.label, item.value]))
  assert.equal(map.Width, '896'); assert.equal(map['Frames per second'], '2'); assert.equal(map.Model, 'qwen3-vl'); assert.equal(map.Thinking, 'off'); assert.equal(map['Additional context'], 'Night scene')
})

test('duration is formatted from UTC timestamps', () => {
  assert.equal(durationText('2026-11-04 12:00:00', '2026-11-04 12:02:05'), '2m 5s')
  assert.equal(durationText('2026-11-04 12:00:00', '2026-11-04 13:05:00'), '1h 5m')
  assert.equal(durationText(null, null), '')
})
