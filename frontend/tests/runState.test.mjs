import { test } from 'node:test'
import assert from 'node:assert/strict'
import { activeRunFor, batchCounts, cutStepStates, preflight, videoSettingsErrors, videoSettingsSummary } from '../src/runState.ts'

const config = { frame_mode: 'per_second', frame_count: 8, video_width: 896, video_height: 504, video_fps: 2, video_frame_budget: null, context_ids: [], detection_queries: [], additional_context: '', text_provider: 'local', local_thinking: false, openrouter_model: '' }
const cut = { start_time: 0, end_time: 10 }
const job = (over = {}) => ({ id: 'j1', pipeline: 'analyze', status: 'queued', stage: 'Queued', stages: null, cuts: [{ id: 'c1', completed: false }], ...over })
const result = (type, extra = {}) => ({ title: 'x', superseded_by: null, settings: { structured_type: type }, ...extra })

test('validation reports each field separately', () => {
  assert.deepEqual(videoSettingsErrors(config), {})
  assert.equal(videoSettingsErrors({ ...config, video_width: 897 }).width, 'Must be even')
  assert.match(videoSettingsErrors({ ...config, video_height: 100 }).height, /128 to 1080/)
  assert.ok(videoSettingsErrors({ ...config, video_fps: 0 }).fps)
  assert.ok(videoSettingsErrors({ ...config, video_fps: Number.NaN }).fps)
  assert.ok(videoSettingsErrors({ ...config, video_fps: 31 }).fps)
})

test('summary line shows canvas, rate and estimated frames, or flags invalid input', () => {
  assert.match(videoSettingsSummary(config, cut), /^896×504 · 2 fps · ≈\d+ frames$/)
  assert.equal(videoSettingsSummary({ ...config, video_width: 3 }, cut), 'Invalid settings')
})

test('finished results are done; superseded ones are ignored', () => {
  const state = cutStepStates([result('initial_frame'), result('video_analysis', { superseded_by: 'n' }), { title: 'Cut summary', superseded_by: null, settings: {} }], [], [], 'c1')
  assert.deepEqual(state, { initial_frame: 'done', video_analysis: 'none', last_frame: 'none', consistency: 'none', summary: 'done' })
})

test('an active run shows queued, running, and checkpointed stages for this cut only', () => {
  const stages = [{ id: 's1', job_id: 'j1', stage_key: 'initial_frame', status: 'done' }, { id: 's2', job_id: 'j1', stage_key: 'video_analysis', status: 'running' }]
  const state = cutStepStates([], stages, [job({ status: 'running' })], 'c1')
  assert.deepEqual([state.initial_frame, state.video_analysis, state.last_frame], ['done', 'running', 'queued'])
  assert.equal(cutStepStates([], [], [job({ cuts: [{ id: 'other', completed: false }] })], 'c1').video_analysis, 'none')
  assert.equal(cutStepStates([], [], [job({ stages: ['last_frame'] })], 'c1').initial_frame, 'none')
})

test('summary pipelines map to their own steps and failed attempts show until a result exists', () => {
  const state = cutStepStates([], [{ id: 's', job_id: 'old', stage_key: 'last_frame', status: 'failed' }], [job({ pipeline: 'consistency_summary', status: 'running' }), job({ id: 'j2', pipeline: 'regen' })], 'c1')
  assert.deepEqual([state.consistency, state.summary, state.last_frame], ['running', 'queued', 'failed'])
  assert.equal(cutStepStates([result('last_frame')], [{ id: 's', job_id: 'old', stage_key: 'last_frame', status: 'failed' }], [], 'c1').last_frame, 'done')
})

test('the running job is preferred as the active run', () => {
  assert.equal(activeRunFor([job({ id: 'a' }), job({ id: 'b', status: 'running' })], 'c1').id, 'b')
  assert.equal(activeRunFor([job({ status: 'done' })], 'c1'), undefined)
})

test('preflight explains what is blocking or risky', () => {
  assert.deepEqual(preflight({ mediaBusy: false, enabled: true, errors: {}, characters: null }), [])
  const texts = preflight({ mediaBusy: true, enabled: false, errors: { fps: 'x' }, characters: { boxes: 3, autoUnverified: 2, unknown: 1, stale: true } })
  assert.equal(texts.length, 6)
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
