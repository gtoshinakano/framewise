import { test } from 'node:test'
import assert from 'node:assert/strict'
import { judgeFrame, lastFrameTime, nextFrameEstimate } from '../src/playbackLimit.ts'

const frame = 1 / 24

test('keeps playing until the frame before the limit is shown', () => {
  assert.equal(judgeFrame(1.0, frame, 2.0), 'continue')
  assert.equal(judgeFrame(2.0 - 2 * frame, frame, 2.0), 'continue')
})

test('stops on the last frame of a cut whose end is the next cut start', () => {
  assert.equal(judgeFrame(2.0 - frame, frame, 2.0), 'last')
  assert.equal(judgeFrame(2.0 - frame + 0.002, frame, 2.0), 'last')
})

test('reports a frame that already starts at the limit', () => {
  assert.equal(judgeFrame(2.0, frame, 2.0), 'over')
  assert.equal(judgeFrame(2.1, frame, 2.0), 'over')
})

test('a limit between two frames ends on the frame that is showing it', () => {
  assert.equal(judgeFrame(1.0, frame, 1.03), 'last')
  assert.equal(judgeFrame(1.0 + frame, frame, 1.03), 'over')
})

test('rests inside the last frame, never before the start', () => {
  assert.ok(Math.abs(lastFrameTime(2.0, frame) - (2.0 - frame / 2)) < 1e-9)
  assert.equal(lastFrameTime(0.01, frame), 0)
})

test('frame estimate keeps the smallest plausible gap', () => {
  assert.equal(nextFrameEstimate(null, null, 1), null)
  assert.ok(Math.abs(nextFrameEstimate(null, 1, 1 + frame) - frame) < 1e-9)
  assert.equal(nextFrameEstimate(frame, 1, 1 + 2 * frame), frame)
  assert.equal(nextFrameEstimate(frame, 1, 5), frame)
  assert.equal(nextFrameEstimate(frame, 1, 1), frame)
})
