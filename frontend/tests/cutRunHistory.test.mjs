import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cutRunOutcome } from '../src/cutRunHistory.ts'
import { analysisQueueStates } from '../src/analysisQueueStates.ts'

const cut = (id, index, completed = false) => ({ id, cut_index: index, completed })
const job = (overrides = {}) => ({ pipeline: 'analyze', status: 'failed', stage: 'Failed · Analyzing cut 2, frame 1/2', error: 'Frame extraction failed', started_at: '2026-11-04 12:00:00', cuts: [cut('a', 0, true), cut('b', 1), cut('c', 2)], ...overrides })

test('a batch failure is attributed to its actual cut, not completed or unreached cuts', () => {
  assert.equal(cutRunOutcome(job(), 'a').status, 'Succeeded')
  assert.deepEqual(cutRunOutcome(job(), 'b'), { status: 'Failed', step: 'Analyzing cut 2, frame 1/2', error: 'Frame extraction failed' })
  assert.deepEqual(cutRunOutcome(job(), 'c'), { status: 'Not completed', step: 'The run stopped on another cut', error: null })
  assert.equal(cutRunOutcome(job(), 'other'), null)
})

test('a consistency summary appears as queued or running for its cut', () => {
  const summary = job({ pipeline: 'consistency_summary', status: 'running', stage: 'Summarizing consistency', cuts: [cut('a', 0)] })
  assert.equal(analysisQueueStates([summary]).get('a'), 'running')
  assert.equal(cutRunOutcome({ ...summary, status: 'done' }, 'a').step, 'Summary generation completed')
})

test('older failures, queued cancellations, and summary runs use recorded facts', () => {
  assert.deepEqual(cutRunOutcome(job({ cuts: [cut('a', 0)], stage: 'Failed' }), 'a'), { status: 'Failed', step: 'Failure step was not recorded', error: 'Frame extraction failed' })
  assert.equal(cutRunOutcome(job({ cuts: [cut('a', 0)], stage: 'Cancelled', started_at: null }), 'a').step, 'Cancelled before processing')
  assert.equal(cutRunOutcome(job({ pipeline: 'regen', status: 'done', stage: 'Complete', cuts: [cut('a', 0)] }), 'a').step, 'Summary generation completed')
  assert.equal(cutRunOutcome(job({ stage: 'Failed' }), 'b').step, 'The cut that failed was not recorded')
})
