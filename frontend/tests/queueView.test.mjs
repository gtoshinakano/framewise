import { test } from 'node:test'
import assert from 'node:assert/strict'
import { attentionText, attentionTitle, counterText, counterTitle, cutText, formatRunTime, mergeFirstPage, runSeconds, stageText, targetText, unitsText } from '../src/queueView.ts'

const summary = (overrides = {}) => ({ busy: true, since: null, done: 35, failed: 2, running: 1, waiting: 202, stopping: 0, cancelled: 0, total: 240, finished: 37, attention: 0, other_active: 0, ...overrides })
const cut = (number, overrides = {}) => ({ id: `c${number}`, number, title: '', archived: false, missing: false, start_time: 0, end_time: 1, completed: false, ...overrides })
const run = (overrides = {}) => ({ cuts: [], cut_count: 0, cut_span: null, sequence: null, range: null, ...overrides })

test('durations are minutes and seconds, never hours', () => {
  assert.equal(formatRunTime(42.9), '42s')
  assert.equal(formatRunTime(303), '5m 03s')
  assert.equal(formatRunTime(4510), '75m 10s')
  assert.equal(formatRunTime(null), '')
  assert.equal(runSeconds({ status: 'running', started_at: '2026-11-15T10:00:00Z', duration_seconds: null }, Date.parse('2026-11-15T10:01:30Z')), 90)
  assert.equal(runSeconds({ status: 'done', started_at: '2026-11-15T10:00:00Z', duration_seconds: 12 }, Date.now()), 12)
})

test('every run names what it works on', () => {
  assert.equal(targetText(run({ cuts: [cut(12, { title: 'Rooftop' })], cut_count: 1 })), 'Cut 12 · Rooftop')
  assert.equal(targetText(run({ cuts: [cut(7, { archived: true })], cut_count: 1 })), 'Archived cut (was 7)')
  assert.equal(targetText(run({ cuts: [cut(3), cut(4)], cut_count: 40, cut_span: { first: 3, last: 42 } })), 'Cuts 3–42 (40)')
  assert.equal(targetText(run({ cuts: [cut(5)], cut_count: 1, sequence: { id: 'q', title: 'Chase', step: 'shot', shot_key: '1-2', shot_number: 3 } })), 'Sequence “Chase” · Shot 3 · Cut 5')
  assert.equal(targetText(run({ range: { start: 12.4, end: 45 } })), '0:12–0:45 of the audio')
  assert.equal(targetText(run()), 'Whole episode')
  assert.equal(cutText(cut(2, { missing: true })), 'Deleted cut')
})

test('the counter shows finished of total GPU steps only while busy', () => {
  assert.equal(counterText(summary()), '37/240')
  assert.equal(counterText(summary({ busy: false, total: 0 })), '')
  assert.equal(counterTitle(summary()), '37 of 240 GPU steps finished in this episode: 35 done, 2 failed · 1 running · 202 waiting.')
  assert.equal(counterTitle(summary({ stopping: 1, waiting: 0 })), '37 of 240 GPU steps finished in this episode: 35 done, 2 failed · 1 running · 1 cancelled run still stopping.')
  assert.match(counterTitle(summary({ busy: false, total: 0 })), /No GPU work is queued/)
})

test('the failure chip says what it counts and what to do', () => {
  assert.equal(attentionText(3), '3 failed')
  assert.equal(attentionTitle(3), '3 runs failed and were not run again. Open the queue to retry or dismiss them.')
  assert.equal(attentionTitle(1), '1 run failed and was not run again. Open the queue to retry or dismiss it.')
})

test('stages and steps read as words', () => {
  assert.equal(stageText({ stage: 'Failed · Shot 13 · attempt 3/3', status: 'failed' }), 'Shot 13 · attempt 3/3')
  assert.equal(stageText({ stage: 'Complete', status: 'done' }), '')
  assert.equal(stageText({ stage: 'Cancelled', status: 'stopping' }), 'Stopping — finishing the current step')
  assert.equal(stageText({ stage: 'Cancelled', status: 'cancelled' }), '')
  assert.equal(stageText({ stage: 'queued', status: 'queued', waiting_for: ['Character boxes'] }), 'Queued after Character boxes')
  assert.equal(stageText({ stage: 'Inputs changed, restarting', status: 'queued' }), 'Restarting with your changes')
  assert.equal(unitsText({ units: { done: 3, total: 12 }, status: 'running' }), '3/12 steps')
  assert.equal(unitsText({ units: { done: 0, total: 1 }, status: 'queued' }), '')
})

test('a refreshed first page keeps the older pages already loaded', () => {
  const item = seq => ({ id: `r${seq}`, seq })
  const shown = [item(10), item(9), item(8), item(5), item(4)]
  assert.deepEqual(mergeFirstPage(shown, [item(12), item(11), item(10)]).map(run => run.seq), [12, 11, 10, 9, 8, 5, 4])
  assert.deepEqual(mergeFirstPage(shown, []), [])
})
