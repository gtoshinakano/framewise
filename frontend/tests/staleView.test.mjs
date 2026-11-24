import test from 'node:test'
import assert from 'node:assert/strict'
import { answersStale } from '../src/staleView.ts'

const review = (questions, extra = {}) => ({ content: JSON.stringify({ questions }), created_at: '2026-11-24 10:00:00', ...extra })
const pose = { id: 'q1', topic: 'pose', question: 'Standing?', options: [{ id: 'a', text: 'Standing' }, { id: 'b', text: 'Sitting' }] }
const gaze = { id: 'q2', topic: 'gaze', character: 'Mai', question: 'Looks?', options: [{ id: 'a', text: 'At Rio' }, { id: 'b', text: 'Down' }] }

test('answers are stale when their review is, or when a newer review asks what they do not answer', () => {
  assert.equal(answersStale('', '2026-11-24 09:00:00', review([pose])), null)
  assert.match(answersStale('- Pose: Standing', '2026-11-24 11:00:00', review([pose], { stale_reason: 'Keyframes changed' })), /out of date \(Keyframes changed\)/)
  assert.equal(answersStale('- Pose: Standing', '2026-11-24 11:00:00', review([pose, gaze])), null)  // saved after the review
  assert.equal(answersStale('- Pose: Standing\n- Gaze (Mai): Down', '2026-11-24 09:00:00', review([pose, gaze])), null)  // older but covering
  assert.equal(answersStale('- Pose: Standing', '2026-11-24 09:00:00', review([pose, gaze])), 'the review was written again after these answers; 1 question has no answer')
})

test('a correction the reviewer settled itself never makes the answers stale', () => {
  const settled = { ...gaze, settled: 'gaze_rule', recommended: 'a' }
  assert.equal(answersStale('- Pose: Standing', '2026-11-24 09:00:00', review([pose, settled])), null)
})
