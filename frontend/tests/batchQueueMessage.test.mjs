import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ANSWERS_BATCH_REASONS, SUMMARY_BATCH_REASONS, batchQueueMessage } from '../src/batchQueueMessage.ts'

test('says how many cuts were queued and why the others were not', () => {
  const result = { queued: 1, blocked: { no_consistency: [7, 9], nothing_to_review: [5], busy: [] } }
  assert.equal(batchQueueMessage('Production decisions draft', 4, result, ANSWERS_BATCH_REASONS),
    'Production decisions draft queued for 1 cut from cut 4. Not queued, no consistency summary yet: cut 7, 9. Not queued, nothing to decide and no saved answers: cut 5.')
})

test('cuts that already have a summary are counted, and long lists are shortened', () => {
  const numbers = Array.from({ length: 11 }, (_, index) => index + 2)
  const result = { queued: 0, skipped: 3, blocked: { answers_pending: numbers } }
  const message = batchQueueMessage('Cut summary', 2, result, SUMMARY_BATCH_REASONS, 'they already have a Cut summary')
  assert.match(message, /^Cut summary queued for 0 cuts from cut 2\. 3 skipped, they already have a Cut summary\./)
  assert.match(message, /still being drafted, queue again when they finish: cut 2, 3, 4, 5, 6, 7, 8, 9 and 3 more\.$/)
})

test('a clean run mentions nothing held back', () => {
  assert.equal(batchQueueMessage('Cut summary', 1, { queued: 5, skipped: 0, blocked: { no_consistency: [], answers_pending: [], busy: [] } }, SUMMARY_BATCH_REASONS, 'x'),
    'Cut summary queued for 5 cuts from cut 1.')
})
