import { test } from 'node:test'
import assert from 'node:assert/strict'
import { answerFor, firstNeed, minutesSeconds, placeForNeed, rangeSummary, samePlace, staysAt, toNeedsMap } from '../src/cutNeedsView.ts'

const entry = (people = [], steps = [], claude = []) => ({ steps, people: people.map(key => ({ key, count: 1, escalation_ids: [] })), claude, answered_by: [], analysis_off: false })
const cut = (index, start, end, overrides = {}) => ({ id: `c${index + 1}`, cut_index: index, active: true, analysis_enabled: true, start_time: start, end_time: end, ...overrides })

test('the first need follows the flow order and the focus keys', () => {
  const needs = entry(['decisions', 'failed_run', 'voice'])
  assert.equal(firstNeed(needs).key, 'voice')
  assert.equal(firstNeed(needs, new Set(['decisions', 'failed_run'])).key, 'decisions')
  assert.equal(firstNeed(needs, new Set(['target_frames'])), undefined)
  assert.equal(firstNeed(undefined), undefined)
})

test('each need has a place in the analysis dialog', () => {
  assert.deepEqual(placeForNeed('target_frames'), { tab: 'run', section: 'targets', anchor: 'targets' })
  assert.deepEqual(placeForNeed('box_names'), { tab: 'run', section: 'characters', anchor: 'characters' })
  assert.equal(placeForNeed('decisions').anchor, 'decisions')
  assert.equal(placeForNeed('failed_run').anchor, 'other-decisions')
  assert.equal(placeForNeed('voice').anchor, 'voices')
  assert.equal(placeForNeed('failed_run').tab, 'results')
})

test('a range sums times and lists the cuts per need, the person first', () => {
  const cuts = [cut(0, 0, 4), cut(1, 4, 70.4), cut(2, 70.4, 80, { analysis_enabled: false }), cut(3, 80, 90)]
  const needs = toNeedsMap({ cuts: { c2: entry(['decisions'], ['summary']), c3: entry([], [], [{ kind: 'voice', escalation_id: 'v' }]), c4: entry(['decisions'], ['beats', 'summary']) } })
  const summary = rangeSummary(cuts, needs, 3, 1)
  assert.equal(summary.first.id, 'c2')
  assert.equal(summary.last.id, 'c4')
  assert.equal(summary.count, 3)
  assert.equal(summary.analysisOff, 1)
  assert.equal(summary.start, 4)
  assert.equal(summary.end, 90)
  assert.equal(summary.seconds, 86)
  assert.deepEqual(summary.totals.map(total => [total.key, total.kind, total.cuts]),
    [['decisions', 'person', [2, 4]], ['claude', 'claude', [3]], ['beats', 'step', [4]], ['summary', 'step', [2, 4]]])
  assert.equal(rangeSummary([], needs, 0, 1), null)
})

test('minutes and seconds', () => {
  assert.equal(minutesSeconds(86), '1:26')
  assert.equal(minutesSeconds(9.6), '0:10')
  assert.equal(minutesSeconds(-1), '0:00')
})

test('a review question shows the latest agent answer that wrote it', () => {
  const decision = (id, created_at, lines, extra = {}) => ({ id, kind: 'review_answers', tier: 'claude', created_at, undone_at: null, overridden_at: null, summary: '', after: { lines }, evidence: {}, ...extra })
  const decisions = [
    decision('old', '2026-11-20 10:00:00', [{ question: 'q1', option: 'A', text: 'Smiles' }]),
    decision('new', '2026-11-21 10:00:00', [{ question: 'q1', option: 'B', text: 'Frowns', why: 'F2 shows it' }], { evidence: { note: 'checked F2' } }),
    decision('undone', '2026-11-22 10:00:00', [{ question: 'q1', option: 'C', text: 'Cries' }], { undone_at: '2026-11-22 11:00:00' }),
    { ...decision('voice', '2026-11-23 10:00:00', [{ question: 'q1' }]), kind: 'voices' },
  ]
  const answer = answerFor(decisions, 'q1')
  assert.equal(answer.decision.id, 'new')
  assert.equal(answer.option, 'B')
  assert.equal(answer.note, 'checked F2')
  assert.equal(answerFor(decisions.slice(0, 1), 'q1').note, '')
  assert.equal(answerFor(decisions, 'q2'), null)
})

test('an open dialog stays where a link points only when the cut\'s first need is settled there', () => {
  const results = { tab: 'results', section: 'targets' }, characters = { tab: 'run', section: 'characters' }, targets = { tab: 'run', section: 'targets' }
  assert.equal(samePlace(results, { tab: 'results' }), true)   // the section counts only on the Run tab
  assert.equal(samePlace(characters, targets), false)
  assert.equal(staysAt(entry(['box_names']), null, characters, characters), true)
  assert.equal(staysAt(entry(['voice', 'box_names']), null, characters, characters), false)   // arriving would move it to the voice
  assert.equal(staysAt(entry(['voice', 'box_names']), new Set(['box_names']), characters, characters), true)
  assert.equal(staysAt(entry(), null, characters, characters), true)   // nothing to move to
  assert.equal(staysAt(undefined, null, { tab: 'results' }, results), true)
  assert.equal(staysAt(entry(['target_frames']), null, targets, characters), false)   // another place: open it again
})
