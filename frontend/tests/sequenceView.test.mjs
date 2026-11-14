import { test } from 'node:test'
import assert from 'node:assert/strict'
import { canMerge, clock, issueInstruction, issuesFor, progress, promptSections, rangeSeconds, shotStatus, splitChoices, stage, staleText } from '../src/sequenceView.ts'

const shot = (index, extra = {}) => ({ index, key: `k${index}`, t0: (index - 1) * 4, t1: index * 4, duration: 4, stale: [], job: null, result: { check: { ok: true } }, ...extra })
const detail = (extra = {}) => ({ sequence: { cuts_changed: false, cuts_missing: false }, missing: [], plan_problems: [], bible: {}, plan: {}, shots: [shot(1), shot(2)], jobs: [], ...extra })

test('clock shows minutes, seconds and tenths', () => {
  assert.equal(clock(0), '0:00.0')
  assert.equal(clock(75.26), '1:15.3')
  assert.equal(clock(59.96), '1:00.0')
  assert.equal(clock(NaN), '0:00.0')
})

test('a shot reports its most urgent need first', () => {
  assert.equal(shotStatus(shot(1)).key, 'ready')
  assert.equal(shotStatus(shot(1, { stale: ['bible'] })).key, 'stale')
  assert.equal(shotStatus(shot(1, { stale: ['bible'], result: { check: { ok: false } } })).key, 'errors')
  assert.equal(shotStatus(shot(1, { result: null })).key, 'missing')
  assert.equal(shotStatus(shot(1, { result: null, job: 'queued' })).key, 'queued')
  assert.equal(shotStatus(shot(1, { stale: ['plan'], job: 'running' })).label, 'Writing')
})

test('stale reasons read as one sentence', () => {
  assert.equal(staleText([]), '')
  assert.match(staleText(['bible']), /^Written before the bible changed/)
  assert.match(staleText(['plan', 'evidence']), /spoken lines changed and the analysis or your answers/)
})

test('the stage follows the two-click flow', () => {
  assert.equal(stage(detail({ bible: null, plan: null, shots: [] })).key, 'direct')
  assert.equal(stage(detail({ missing: [3, 4], bible: null, plan: null, shots: [] })).key, 'analyse')
  assert.match(stage(detail({ missing: [3, 4], bible: null, plan: null, shots: [] })).hint, /Cuts 3, 4 need/)
  assert.equal(stage(detail({ jobs: [{ status: 'running', step: 'bible' }] })).key, 'directing')
  assert.equal(stage(detail({ shots: [shot(1, { result: null }), shot(2, { result: null })] })).key, 'write')
  assert.match(stage(detail({ shots: [shot(1, { result: null }), shot(2, { result: null })] })).hint, /Review the bible and the plan/)
  assert.match(stage(detail({ shots: [shot(1), shot(2, { stale: ['plan'] })] })).hint, /^1 shot need writing/)
  const writing = stage(detail({ shots: [shot(1), shot(2, { result: null, job: 'running' })], jobs: [{ status: 'running', step: 'shot' }, { status: 'queued', step: 'shot' }] }))
  assert.deepEqual([writing.key, writing.hint], ['writing', 'Writing shot prompts: 1/2 ready.'])
  assert.equal(stage(detail({ sequence: { cuts_changed: true, cuts_missing: false } })).key, 'redirect')
  assert.equal(stage(detail({ sequence: { cuts_changed: false, cuts_missing: true } })).key, 'recreate')
  assert.equal(stage(detail()).key, 'ready')
  assert.deepEqual(progress(detail({ shots: [shot(1), shot(2, { result: null })], jobs: [{ status: 'queued', step: 'shot' }] })),
    { shots: 2, written: 1, ready: 1, needWriting: 1, running: 0, queued: 1, step: 'shot' })
})

test('split choices lie inside the shot, and a merge must fit one take', () => {
  const candidates = [{ id: 'B0', t: 0, time: 100, kind: 'start' }, { id: 'B1', t: 0.3, time: 100.3, kind: 'quiet' }, { id: 'B2', t: 2, time: 102, kind: 'pause', pause: 0.6 }, { id: 'B3', t: 3.8, time: 103.8, kind: 'event' }, { id: 'B4', t: 4, time: 104, kind: 'cut' }]
  assert.deepEqual(splitChoices(shot(1), candidates), [{ time: 102, label: '0:02.0 · pause 0.6 s' }])
  const shots = [shot(1), shot(2), shot(3, { t1: 30 })]
  assert.equal(canMerge(shots, 0, 15), true)
  assert.equal(canMerge(shots, 1, 15), false)
  assert.equal(canMerge(shots, 2, 15), false)
})

test('review issues attach to shots by key and become a rewrite instruction', () => {
  const issues = [{ shot: 1, shot_key: 'k2', severity: 'fix', problem: 'Coat changes colour.', suggestion: 'Keep the green coat.' }, { shot: 2, shot_key: 'k2', severity: 'check', problem: 'Hand position unclear.', suggestion: '' }]
  assert.equal(issuesFor(issues, shot(1)).length, 0)
  assert.equal(issueInstruction(issuesFor(issues, shot(2))), 'Keep the green coat. Hand position unclear.')
  assert.deepEqual(issuesFor(undefined, shot(1)), [])
})

test('an H3 prompt is split into its sections', () => {
  const sections = promptSections('subject_definitions:\n<Picture 1> is the first frame.\n\nsummary:\n[keyframe completion + reference generation] She waves.\n\ndetailed_description:\nThe target video is anime.\n00:00.000-00:02.000: She waves.\n\nnon_diegetic_music:\nN/A')
  assert.deepEqual(sections.map(section => [section.name, section.lines.length]), [['subject_definitions', 1], ['summary', 1], ['detailed_description', 2], ['non_diegetic_music', 1]])
  assert.equal(sections[2].title, 'Shot and beats')
  assert.deepEqual(promptSections(''), [])
})

test('a range of cuts has a length only when both ends exist in order', () => {
  const cuts = [{ cut_index: 0, start_time: 0, end_time: 2 }, { cut_index: 1, start_time: 2, end_time: 5.5 }]
  assert.equal(rangeSeconds(cuts, 1, 2), 5.5)
  assert.equal(rangeSeconds(cuts, 2, 2), 3.5)
  assert.equal(rangeSeconds(cuts, 2, 1), null)
  assert.equal(rangeSeconds(cuts, 1, 9), null)
})
