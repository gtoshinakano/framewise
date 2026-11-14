import { test } from 'node:test'
import assert from 'node:assert/strict'
import { boundaryChoices, changeAt, clock, deciderLabel, deliveryLabel, edgeTags, issueInstruction, issuesFor, progress, promptSections, rangeSeconds, shotStatus, speakerRanges, stage, staleText } from '../src/sequenceView.ts'

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
  assert.match(staleText(['plan', 'evidence']), /your voice settings changed and the analysis or your answers/)
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

test('a hard cut inside speech offers only the positions that are possible, with signed moves', () => {
  const boundary = { id: 'K2', speech: { lines: ['L3'], keeps: ['L3', 'L4'] }, options: { keep_cut: { t: 5, time: 105, shift: 0 }, after_line: { t: 6.21, time: 106.21, shift: 1.21 } }, unavailable: { before_line: 'too far' } }
  assert.deepEqual(boundaryChoices(boundary), [{ choice: 'keep_cut', label: 'On the hard cut · divides L3' }, { choice: 'after_line', label: 'After L3–L4 (+1.21 s)' }])
  assert.deepEqual(boundaryChoices({ ...boundary, speech: { lines: ['L1'], keeps: ['L1'] }, options: { keep_cut: { shift: 0 }, before_line: { shift: -0.4 } } }).map(item => item.label), ['On the hard cut · divides L1', 'Before L1 (−0.40 s)'])
  assert.deepEqual([deciderLabel('director'), deciderLabel('app'), deciderLabel('user'), deciderLabel(null)], ['director', 'app', 'you', null])
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

test('the dialogue comes before direction', () => {
  const compilation = state => ({ compilation: { state, has_speech: true }, bible: null, plan: null })
  assert.equal(stage(detail(compilation('review'))).key, 'review')
  assert.equal(stage(detail(compilation('none'))).key, 'review')
  assert.equal(stage(detail(compilation('confirmed'))).key, 'direct')
  assert.equal(stage(detail({ ...compilation('review'), compilation: { state: 'review', has_speech: false } })).key, 'direct')
  assert.equal(stage(detail({ ...compilation('running'), jobs: [{ status: 'running', step: 'compile' }] })).key, 'compiling')
})

test('edge tags show borrowed audio, divided speech and the parts of a long cut', () => {
  const plain = { how: 'keep_cut' }
  assert.deepEqual(edgeTags({ opens: { how: 'start' }, closes: plain, part: null }), [])
  const runsOn = edgeTags({ opens: plain, closes: { how: 'after_line', shift: 0.8, keeps: ['L5'] }, part: null })
  assert.equal(runsOn[0].label, 'ends +0.80 s')
  assert.match(runsOn[0].title, /Runs past the end of its cut to keep L5 whole/)
  const later = edgeTags({ opens: { how: 'after_line', shift: 0.8, keeps: ['L5'] }, closes: { how: 'end' }, part: null })
  assert.match(later[0].title, /the neighbouring shot carries L5 whole/)
  const divided = edgeTags({ opens: { how: 'start' }, closes: { how: 'keep_cut', split: { lines: ['L7'], before: 'ねえ', after: '聞いてる？' }, reason: 'Mai cuts him off.' }, part: [1, 2] })
  assert.deepEqual(divided.map(tag => tag.label), ['part 1/2', 'divides L7'])
  assert.match(divided[1].title, /“ねえ” \| “聞いてる？” — Mai cuts him off\./)
})

test('voice ranges: your ranges win over the director, and a change from the playhead keeps what came before', () => {
  const written = [{ speaker: 'Mai', from: 0, to: 4, delivery: 'narrative', subject: 'Mai' }, { speaker: 'Mai', from: 4, to: 10, delivery: 'on_screen', subject: 'Mai' }, { speaker: 'Ken', from: 0, to: 10, delivery: 'off_screen', subject: null }]
  const fixed = [{ speaker: 'Mai', from: 6, to: 10, delivery: 'inner_voice' }]
  const ranges = speakerRanges(written, fixed, 'Mai', 10)
  assert.deepEqual(ranges.map(range => [range.from, range.to, range.delivery, range.mine]), [[0, 4, 'narrative', false], [4, 6, 'on_screen', false], [6, 10, 'inner_voice', true]])
  assert.deepEqual(speakerRanges(undefined, [], 'Mai', 10), [])
  assert.deepEqual(changeAt(ranges, 0, 'off_screen'), [{ from: 0, delivery: 'off_screen' }, { from: 6, delivery: 'inner_voice' }])
  assert.deepEqual(changeAt(ranges, 2.04, 'on_screen'), [{ from: 0, delivery: 'narrative' }, { from: 2, delivery: 'on_screen' }, { from: 6, delivery: 'inner_voice' }])
  assert.deepEqual(changeAt(speakerRanges(written, [], 'Mai', 10), 5, 'on_screen'), [{ from: 0, delivery: 'narrative' }, { from: 4, delivery: 'on_screen' }])
  assert.equal(deliveryLabel('narrative'), 'Narrative')
  assert.equal(deliveryLabel(null), 'Director decides')
})
