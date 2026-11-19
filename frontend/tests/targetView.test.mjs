import { test } from 'node:test'
import assert from 'node:assert/strict'
import { MAX_TARGET_BOXES, boxesInFrame, canDraw, frameNeedsReview, framesPerSlot, isJobActive, mergeTargets, slotChip, slotName, targetQueueMessage, unknownSlots } from '../src/targetView.ts'

const slot = (number, overrides = {}) => ({ slot: number, origin: 'first_frame', character_context_id: null, character_name: null, label_source: null, ...overrides })
const box = (id, frame, slotNumber, source = 'auto') => ({ id, frame_ordinal: frame, slot: slotNumber, box: [0, 0, 10, 10], source })
const run = {
  id: 'run', selection_key: 'k', stale: false, frames: [],
  slots: [slot(0, { character_context_id: 'ctx', character_name: 'Hero', label_source: 'inherited' }), slot(1), slot(2, { origin: 'appeared' })],
  detections: [box('a', 10, 0), box('b', 10, 1), box('c', 15, 0), box('d', 15, 1), box('e', 15, 2)],
}

test('a frame holds three boxes, so drawing needs a free place', () => {
  assert.equal(MAX_TARGET_BOXES, 3)
  assert.equal(canDraw(run, 10), true)
  assert.equal(canDraw(run, 15), false)
  assert.equal(canDraw(run, 99), true)
  assert.deepEqual(boxesInFrame(run, 10).map(item => item.id), ['a', 'b'])
})

test('unknown slots are the unnamed ones that have a box', () => {
  assert.deepEqual(unknownSlots(run).map(item => item.slot), [1, 2])
  assert.deepEqual(unknownSlots({ ...run, detections: [box('a', 10, 0)] }), [])
  assert.deepEqual(unknownSlots(null), [])
  assert.deepEqual(unknownSlots({ ...run, slots: [run.slots[0], { ...run.slots[1], ignored: true }, run.slots[2]] }).map(item => item.slot), [2])
})

test('the exception filter shows flagged frames until the user reviews them', () => {
  assert.equal(frameNeedsReview(undefined), false)
  assert.equal(frameNeedsReview({ review: { score: 1, flags: {} } }), false)
  assert.equal(frameNeedsReview({ review: { score: .72, flags: { suspected_merge: ['m'] } } }), true)
  assert.equal(frameNeedsReview({ review: { reviewed: true, score: .72, flags: { suspected_merge: ['m'] } } }), false)
})

test('each slot knows in how many frames it appears', () => {
  assert.deepEqual([...framesPerSlot(run)].sort(), [[0, 2], [1, 2], [2, 1]])
})

test('a slot says where its name came from', () => {
  assert.equal(slotChip(run.slots[0]).text, 'From first frame')
  assert.equal(slotChip(slot(3, { character_context_id: 'x', label_source: 'manual' })).tone, 'green')
  assert.equal(slotChip(slot(3, { character_context_id: 'x', label_source: 'auto' })).tone, 'amber')
  assert.equal(slotChip(run.slots[1]).text, 'No name')
  assert.equal(slotName(run.slots[0]), 'Hero')
  assert.equal(slotName(run.slots[1]), 'Unknown')
  assert.equal(slotName(undefined), 'Unknown')
})

test('job status stays visible while the controls are available', () => {
  assert.equal(isJobActive({ id: 'j', status: 'running', stage: '', error: null }), true)
  assert.equal(isJobActive({ id: 'j', status: 'done', stage: '', error: null }), false)
  assert.equal(isJobActive(null), false)
})

test('a batch queue says what it did and which cuts were left out', () => {
  const none = { not_evaluated: [], analysis_disabled: [], no_characters: [], up_to_date: [], already_queued: [] }
  assert.deepEqual(targetQueueMessage({ queued: 3, skipped: { ...none, not_evaluated: [2, 5], up_to_date: [1] } }, 1), {
    tone: 'success', lines: ['Target-frame characters queued for 3 cuts.', 'Not evaluated yet, evaluate them first: cut 2, 5.', 'Already detected: cut 1.'] })
  const later = targetQueueMessage({ queued: 1, skipped: none }, 4)
  assert.equal(later.lines[0], 'Target-frame characters queued for 1 cut from cut 4.')
  const nothing = targetQueueMessage({ queued: 0, skipped: { ...none, no_characters: Array.from({ length: 10 }, (_, i) => i + 1) } }, 1)
  assert.equal(nothing.tone, 'warning')
  assert.equal(nothing.lines[1], 'Marked as having no characters: cut 1, 2, 3, 4, 5, 6, 7, 8 and 2 more.')
})

test('a colour can be merged into any other, except when it is a separate person in the first frame', () => {
  const frames = [{ ordinal: 10, time: 0, relative_time: 0, image_size: [100, 80], mirror: true }, { ordinal: 15, time: 1, relative_time: 1, image_size: [100, 80] }]
  const shown = { ...run, frames, detections: [box('a', 10, 0), box('b', 10, 1), box('c', 15, 0), box('d', 15, 2)] }
  assert.deepEqual(mergeTargets(shown, 2).map(item => item.slot), [0, 1])   // appeared later: can join either
  assert.deepEqual(mergeTargets(shown, 0), [])                              // two people in the first frame stay two
  assert.deepEqual(mergeTargets(shown, 1), [])
  assert.deepEqual(mergeTargets({ ...shown, frames: frames.map(frame => ({ ...frame, mirror: false })) }, 1).map(item => item.slot), [0, 2])
})
