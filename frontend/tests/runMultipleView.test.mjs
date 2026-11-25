import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  clampFloat, cutNumbers, defaultStage, evaluateCounts, groupHeld, humanInbox, inboxText, progressText, reasonGroup, shownRun, startButton, stepLinks, stepText, stepTone,
} from '../src/runMultipleView.ts'
import { modeOf, modesFor } from '../src/autopilotView.ts'

const step = (name, state, extra = {}) => ({ step: name, label: name, state, ...extra })
const run = (id, stage, status) => ({ id, stage, status, episode_id: 'e', options: {}, steps: [], summary: {}, error: null, created_at: '', updated_at: '', finished_at: null })

test('steps are marked by state and read what they found, else what they will do', () => {
  assert.deepEqual(['done', 'running', 'starting', 'will_run', 'pending', 'off', 'skipped', 'failed', 'cancelled'].map(stepTone),
    ['done', 'active', 'active', 'waiting', 'waiting', 'quiet', 'quiet', 'failed', 'failed'])
  assert.equal(stepText(step('hardcuts', 'done', { detail: 'finds the hard cuts', summary: '352 cuts detected' })), '352 cuts detected')
  assert.equal(stepText(step('subtitles', 'off', { detail: 'off: subtitles are optional' })), 'off: subtitles are optional')
  assert.equal(stepText(step('visual', 'cancelled', { detail: '3 cuts' })), 'Cancelled')
  assert.equal(progressText({ total: 4, finished: 1, failed: 1 }), '1/4 jobs, 1 failed')
  assert.equal(progressText({ total: 0, finished: 0, failed: 0 }), '')
  assert.equal(cutNumbers([9, 3, 3, 5]), '3, 5, 9')
  assert.equal(cutNumbers(Array.from({ length: 14 }, (_, index) => index + 1)), '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 and 2 more')
})

test('held cuts are grouped by what the person does about them, in the order they settle it', () => {
  assert.equal(reasonGroup(['box 1 label not confirmed (Ann 0.85, lead 0.03)']).key, 'names')
  assert.equal(reasonGroup(['first-frame boxes not detected for the current cut']).section, 'characters')
  assert.equal(reasonGroup(['inbox: an unknown voice']).tab, 'results')
  assert.equal(reasonGroup(['2 unnamed person(s) on the target frames']).section, 'targets')
  assert.equal(reasonGroup(['keyframe analysis stale (Cut timing changed)']).key, 'earlier')
  const groups = groupHeld([
    { cut: 7, cut_id: 'c7', step: 'beats', reasons: ['inbox: an unknown voice'] },
    { cut: 2, cut_id: 'c2', step: 'visual', reasons: ['target frame F2 flagged (low_match)'] },
    { cut: 3, cut_id: 'c3', step: 'visual', reasons: ['box 2 has no character'] },
    { cut: 4, cut_id: 'c4', step: 'visual', reasons: ['inbox: box names'] },
  ])
  assert.deepEqual(groups.map(group => [group.key, group.cuts.map(item => item.cut)]), [['names', [3, 4]], ['targets', [2]], ['voice', [7]]])
})

test('the start button says what it will run, or why it cannot', () => {
  assert.deepEqual(startButton('evaluate', null), { label: 'Run Evaluate', disabled: true, why: 'Loading…' })
  assert.deepEqual(startButton('evaluate', { available: false, reason: 'Run Prepare first', steps: [] }), { label: 'Run Evaluate', disabled: true, why: 'Run Prepare first' })
  assert.deepEqual(startButton('evaluate', { available: true, reason: '', steps: [], cuts: 12, held: [{}] }), { label: 'Evaluate 12 cuts', disabled: false, why: '' })
  assert.equal(startButton('evaluate', { available: true, reason: '', steps: [], cuts: 0, held: [{}] }).why, 'Every cut that needs it waits for you')
  assert.equal(startButton('evaluate', { available: true, reason: '', steps: [], cuts: 0, held: [] }).why, 'Every cut is evaluated and current')
  const changed = { available: true, reason: '', steps: [step('boxes', 'will_run', { cuts: [{ cut: 4 }, { cut: 5 }] }), step('target_frames', 'will_run', { cuts: [{ cut: 5 }] }), step('cast', 'will_run')] }
  assert.equal(startButton('prepare_cuts', changed).label, 'Update 2 cuts')
  assert.equal(startButton('prepare_cuts', { available: true, reason: '', steps: [step('boxes', 'skipped')] }).disabled, true)
  assert.deepEqual(startButton('prepare', { available: true, reason: '', steps: [step('media', 'done')] }), { label: 'Run Prepare', disabled: false, why: '' })
})

test('an Evaluate preview counts the cuts per step and in all', () => {
  const preview = { steps: [step('visual', 'will_run', { cuts: [{ cut: 1 }, { cut: 2 }] }), step('beats', 'will_run', { cuts: [{ cut: 5 }] }), step('consistency', 'skipped')], cuts: 3, held: [{}, {}] }
  assert.deepEqual(evaluateCounts(preview), { visual: 2, beats: 1, consistency: 0, cuts: 3, held: 2 })
})

test('a finished step links the cuts it names, where each is fixed', () => {
  assert.deepEqual(stepLinks(step('target_frames', 'done', { stats: { to_check: [4, 9], not_detected: [] } })), [{ label: 'To check', cuts: [4, 9], tab: 'run', section: 'targets' }])
  assert.deepEqual(stepLinks(step('consistency', 'done', { stats: { not_done: [3], questions_for_you: [5] } })).map(link => link.label), ['Not done', 'Questions for you'])
  assert.deepEqual(stepLinks(step('missed_cuts', 'done', { stats: { cuts: [12] } }))[0].cuts, [12])
  assert.deepEqual(stepLinks(step('voices', 'done', { stats: { lines_given: 4 } })), [])
})

test('the inbox after a run, in words', () => {
  const inbox = { voice: { human: 2, claude: 0 }, box_label: { human: 0, claude: 1 } }
  assert.equal(inboxText(inbox), '1 box label for Claude · 2 voices for you')
  assert.equal(humanInbox(inbox), 2)
  assert.equal(inboxText(undefined), '')
})

test('the dialog opens on the running stage, else Prepare until it ran, else Evaluate', () => {
  assert.equal(defaultStage([run('a', 'evaluate', 'running')], true, 'prepare'), 'evaluate')
  assert.equal(defaultStage([], false), 'prepare')
  assert.equal(defaultStage([], true), 'evaluate')
  assert.equal(defaultStage([], true, 'prepare'), 'prepare')
  const runs = [run('b', 'prepare', 'finished'), run('a', 'prepare', 'running')]
  assert.equal(shownRun(runs, 'prepare', null).id, 'a')
  assert.equal(shownRun([run('b', 'prepare', 'finished')], 'prepare', 'b').id, 'b')   // the run followed in this dialog stays shown
  assert.equal(shownRun([run('b', 'prepare', 'finished')], 'prepare', null), undefined)
})

test('missed cuts and line splits only apply or are off: a stored "ask" reads as apply', () => {
  assert.deepEqual(modesFor('missed_cuts').map(option => option.value), ['apply', 'off'])
  assert.deepEqual(modesFor('voices').map(option => option.value), ['apply', 'ask', 'off'])
  assert.equal(modeOf('line_splits', 'ask'), 'apply')
  assert.equal(modeOf('voices', 'ask'), 'ask')
  assert.equal(modeOf('voices', undefined), 'apply')
})

test('the floating list stays inside the window', () => {
  const viewport = { width: 1000, height: 800 }, size = { width: 300, height: 200 }
  assert.deepEqual(clampFloat({ x: 100.4, y: 50 }, size, viewport), { x: 100, y: 50 })
  assert.deepEqual(clampFloat({ x: 900, y: 700 }, size, viewport), { x: 692, y: 592 })
  assert.deepEqual(clampFloat({ x: -40, y: -10 }, size, viewport), { x: 8, y: 8 })
  assert.deepEqual(clampFloat({ x: 500, y: 500 }, { width: 300, height: 900 }, viewport), { x: 500, y: 8 })   // taller than the window: the header stays
})
