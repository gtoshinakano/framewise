import { test } from 'node:test'
import assert from 'node:assert/strict'
import { activeRunFor, batchCounts, cutStepStates, preflight } from '../src/runState.ts'

const job = (over = {}) => ({ id: 'j1', pipeline: 'analyze', status: 'queued', stage: 'Queued', stages: null, cuts: [{ id: 'c1', completed: false }], ...over })
const result = (type, extra = {}) => ({ title: 'x', superseded_by: null, settings: { structured_type: type }, ...extra })
const NONE = { initial_frame: 'none', keyframe_analysis: 'none', beats: 'none', consistency: 'none', summary: 'none' }

test('finished results are done; superseded ones are ignored', () => {
  const state = cutStepStates([result('initial_frame'), result('keyframe_analysis', { superseded_by: 'n' }), result('performance_beats'), { title: 'Cut summary', superseded_by: null, settings: {} }], [], [], 'c1')
  assert.deepEqual(state, { ...NONE, initial_frame: 'done', beats: 'done', summary: 'done' })
})

test('an active run shows queued, running, and checkpointed stages for this cut only', () => {
  const stages = [{ id: 's1', job_id: 'j1', stage_key: 'initial_frame', status: 'done' }, { id: 's2', job_id: 'j1', stage_key: 'keyframe_analysis', status: 'running' }]
  const state = cutStepStates([], stages, [job({ status: 'running' })], 'c1')
  assert.deepEqual([state.initial_frame, state.keyframe_analysis], ['done', 'running'])
  assert.deepEqual(cutStepStates([], [], [job()], 'c1'), { ...NONE, initial_frame: 'queued', keyframe_analysis: 'queued' })
  assert.equal(cutStepStates([], [], [job({ cuts: [{ id: 'other', completed: false }] })], 'c1').keyframe_analysis, 'none')
  assert.equal(cutStepStates([], [], [job({ stages: ['keyframe_analysis'] })], 'c1').initial_frame, 'none')
  // Jobs queued before the last frame and the video stage were retired add no step, and old results show nowhere.
  const old = cutStepStates([result('last_frame'), result('video_analysis')], [], [job({ stages: ['last_frame', 'video_analysis'] })], 'c1')
  assert.deepEqual(old, NONE)
})

test('text pipelines map to their own steps and failed attempts show until a result exists', () => {
  const state = cutStepStates([], [{ id: 's', job_id: 'old', stage_key: 'keyframe_analysis', status: 'failed' }],
    [job({ pipeline: 'consistency_summary', status: 'running' }), job({ id: 'j2', pipeline: 'regen' }), job({ id: 'j3', pipeline: 'performance_beats' })], 'c1')
  assert.deepEqual([state.consistency, state.summary, state.keyframe_analysis, state.beats], ['running', 'queued', 'failed', 'queued'])
  assert.equal(cutStepStates([result('keyframe_analysis')], [{ id: 's', job_id: 'old', stage_key: 'keyframe_analysis', status: 'failed' }], [], 'c1').keyframe_analysis, 'done')
})

test('the running job is preferred as the active run', () => {
  assert.equal(activeRunFor([job({ id: 'a' }), job({ id: 'b', status: 'running' })], 'c1').id, 'b')
  assert.equal(activeRunFor([job({ status: 'done' })], 'c1'), undefined)
  assert.equal(activeRunFor([job({ id: 'b', pipeline: 'performance_beats', status: 'running' })], 'c1').id, 'b')
})

test('preflight explains what is blocking or risky', () => {
  assert.deepEqual(preflight({ mediaBusy: false, enabled: true, characters: null }), [])
  const texts = preflight({ mediaBusy: true, enabled: false, characters: { boxes: 3, autoUnverified: 2, unknown: 1, stale: true } })
  assert.equal(texts.length, 5)
  assert.equal(texts[0].level, 'error')
  assert.match(texts.find(item => /unverified/.test(item.text)).text, /^2 automatic labels are unverified/)
})

test('batch counts include only active, enabled cuts from here', () => {
  const cuts = [0, 1, 2, 3].map(index => ({ active: true, cut_index: index, analysis_enabled: index !== 2 })).concat([{ active: false, cut_index: -1, analysis_enabled: true }])
  assert.deepEqual(batchCounts(cuts, { cut_index: 1 }), { from: 2, all: 3 })
})

test('a batch message becomes lines and warns when cuts were held back', async () => {
  const { queueNotice } = await import('../src/runState.ts')
  assert.deepEqual(queueNotice('Cut summary queued for 5 cuts from cut 1.'), { tone: 'success', lines: ['Cut summary queued for 5 cuts from cut 1.'] })
  const held = queueNotice('Queued 3 cuts from cut 2; 1 already complete were skipped. Not queued, unknown characters: cut 7, 9. Not queued, characters not detected yet: cut 11.')
  assert.equal(held.tone, 'warning'); assert.equal(held.lines.length, 3); assert.match(held.lines[1], /^Not queued, unknown/)
  assert.equal(queueNotice('Production decisions draft queued for 0 cuts from cut 4.').tone, 'warning')
})
