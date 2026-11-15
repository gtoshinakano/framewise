import { test } from 'node:test'
import assert from 'node:assert/strict'
import { clampTime, formatDuration, formatTimecode, nudgeStep, parseTimecode } from '../src/timecode.ts'

test('formats minutes, hours and invalid values', () => {
  assert.equal(formatTimecode(0), '0:00.000')
  assert.equal(formatTimecode(83.4567), '1:23.457')
  assert.equal(formatTimecode(3723.5), '1:02:03.500')
  assert.equal(formatTimecode(59.9996), '1:00.000')
  assert.equal(formatTimecode(83.456, { precision: 1 }), '1:23.4')
  assert.equal(formatTimecode(83.456, { precision: 0 }), '1:23')
  assert.equal(formatTimecode(null), '—')
  assert.equal(formatTimecode(Number.NaN), '—')
})

test('durations are short for short spans', () => {
  assert.equal(formatDuration(8.144), '8.14s')
  assert.equal(formatDuration(12.34), '12.3s')
  assert.equal(formatDuration(65.25), '1:05.2')
})

test('reads seconds, timecodes, decimal commas and a seconds suffix', () => {
  assert.equal(parseTimecode('83.4'), 83.4)
  assert.equal(parseTimecode('1:23.4'), 83.4)
  assert.equal(parseTimecode('01:23.456'), 83.456)
  assert.equal(parseTimecode('1:02:03.4'), 3723.4)
  assert.equal(parseTimecode('83,4'), 83.4)
  assert.equal(parseTimecode(' 12s '), 12)
  assert.equal(parseTimecode('.5'), 0.5)
  assert.equal(parseTimecode('1.23456'), 1.235)
})

test('a leading sign is relative to the current value', () => {
  assert.equal(parseTimecode('+0.25', 10), 10.25)
  assert.equal(parseTimecode('-1', 10), 9)
  assert.equal(parseTimecode('+1:00', 10), 70)
  assert.equal(parseTimecode('+0.25'), null)
})

test('refuses text that is not a time', () => {
  for (const text of ['', 'abc', '1:75', '1:60:00', '1::2', '1:2:3:4', '1.2.3', '-', '1:a']) assert.equal(parseTimecode(text, 5), null, text)
})

test('clamping and nudge sizes', () => {
  assert.equal(clampTime(-1, 0, 10), 0)
  assert.equal(clampTime(12, 0, 10), 10)
  assert.equal(clampTime(1.23456), 1.235)
  assert.equal(nudgeStep({ shiftKey: false, altKey: false }), 0.1)
  assert.equal(nudgeStep({ shiftKey: true, altKey: false }), 1)
  assert.equal(nudgeStep({ shiftKey: false, altKey: true }), 0.01)
})
