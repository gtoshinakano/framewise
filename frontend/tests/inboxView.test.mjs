import test from 'node:test'
import assert from 'node:assert/strict'
import { choicesFor, decisionCounts, elsewhereText, flatOrder, groupByKind, mineCount, placeFor, sampleDecisions, sheetPath } from '../src/inboxView.ts'

const item = (kind, extra = {}) => ({ id: `${kind}-1`, episode_id: 'e', run_id: null, kind, group_key: 'g', question: 'Q', status: 'open', reviewer: 'human',
  created_at: '', updated_at: '', cut_ids: ['c1'], options: [], proposal: null, evidence: {}, ...extra })

test('questions are grouped in the order of the episode flow', () => {
  const items = [item('target_frame'), item('mystery'), item('missed_cut'), item('voice'), item('box_label')]
  assert.deepEqual(groupByKind(items).map(group => group.kind), ['missed_cut', 'voice', 'box_label', 'target_frame', 'mystery'])
  assert.deepEqual(flatOrder(items).map(entry => entry.kind), ['missed_cut', 'voice', 'box_label', 'target_frame', 'mystery'])
  assert.equal(groupByKind([item('mystery_kind')])[0].label, 'Mystery kind')
})

test('one-click answers per kind carry the request the backend expects', () => {
  const split = choicesFor(item('missed_cut', { proposal: { answer: 'split' } }))
  assert.equal(split[0].proposed, true)
  assert.deepEqual(split[0].request, { path: '/escalations/missed_cut-1/missed-cut', body: { answer: 'split' } })
  const voice = choicesFor(item('voice', { proposal: { character_context_id: 'mai' }, evidence: { candidates: [{ id: 'mai', name: 'Mai' }, { id: 'rio', name: 'Rio' }] } }))
  assert.deepEqual(voice.map(choice => choice.label), ['Mai', 'Rio', 'New Unregistered', 'New character', 'Leave unknown'])
  assert.equal(voice[0].proposed, true)
  // Unregistered cards boxed in those cuts come before a new number; an unsure guess is kept, never asked again.
  const unsure = choicesFor(item('voice', { evidence: { candidates: [{ id: 'mai', name: 'Mai' }], cards: [{ id: 'u2', name: 'Unregistered 2' }], unsure: true } }))
  assert.deepEqual(unsure.map(choice => choice.label), ['Mai', 'Unregistered 2', 'New Unregistered', 'New character', 'Keep as it is'])
  assert.deepEqual(unsure[1].request.body, { choice: 'unregistered', character_context_id: 'u2' })
  assert.deepEqual(unsure[2].request.body, { choice: 'unregistered' })
  const labels = choicesFor(item('box_label', { proposal: { character_context_id: 'mai', name: 'Mai' }, evidence: { boxes: [{ id: 'b1' }, { id: 'b2' }] } }))
  assert.deepEqual(labels[0].request.body, { confirm: ['b1', 'b2'] })
  assert.deepEqual(labels[1].request.body, { reject: ['b1', 'b2'] })
  const person = choicesFor(item('new_person', { proposal: { candidates: [['rio', 'Rio', 0.41]] } }))
  assert.deepEqual(person.map(choice => choice.key), ['rio', 'new', 'unregistered', 'background', 'leave'])
  assert.equal(person[1].input, 'title')
  const frames = choicesFor(item('target_frame', { evidence: { frames: [{ ordinal: 4 }, { ordinal: 9 }] } }))
  assert.deepEqual(frames[0].request.body, { frames: { 4: 'ok', 9: 'ok' } })
  const failed = choicesFor(item('failed_run', { evidence: { jobs: [{ job_id: 'j1' }, { job_id: 'j2' }] } }))
  assert.deepEqual(failed.map(choice => [choice.label, choice.retry ?? choice.dismiss]), [['Retry 2', ['j1', 'j2']], ['Dismiss', ['j1', 'j2']]])
  assert.deepEqual(choicesFor(item('something_else')).map(choice => choice.close), ['resolved', 'dismissed'])
})

test('sheets and places', () => {
  assert.equal(sheetPath(item('missed_cut', { evidence: { strip: '/api/cuts/c1/missed-cut-strip?time=1.5' } })), '/cuts/c1/missed-cut-strip?time=1.5')
  assert.equal(sheetPath(item('box_label')), '/escalations/box_label-1/box-sheet')
  assert.equal(sheetPath(item('failed_run')), null)
  assert.deepEqual(placeFor('target_frame'), { tab: 'run', section: 'targets' })
  assert.deepEqual(placeFor('voice'), { tab: 'results' })
})

test('the spot-check sample is stable and per kind; counts skip undone decisions', () => {
  const decisions = Array.from({ length: 12 }, (_, index) => ({ id: `d${index}`, kind: index < 8 ? 'voices' : 'box_names', tier: index % 2 ? 'rules' : 'claude',
    undone_at: index === 0 ? 'x' : null, overridden_at: index === 3 ? 'x' : null }))
  const sample = sampleDecisions(decisions, 3)
  assert.equal(sample.length, 6)
  assert.deepEqual(sampleDecisions([...decisions].reverse(), 3).map(entry => entry.id).sort(), sample.map(entry => entry.id).sort())
  assert.ok(!sample.some(entry => entry.id === 'd0'))
  assert.deepEqual(decisionCounts(decisions).voices, { total: 7, overridden: 1, tiers: { rules: 4, claude: 3 } })
})

test('header counts', () => {
  const counts = { episodes: [{ episode_id: 'e', title: 'Ep 2', project_title: 'P', for_person: 4, for_claude: 1 }, { episode_id: 'f', title: 'Ep 3', project_title: 'P', for_person: 2, for_claude: 0 }] }
  assert.equal(mineCount(counts, 'e'), 4)
  assert.equal(mineCount(null, 'e'), 0)
  assert.equal(elsewhereText(counts, 'e'), 'Also waiting for you: Ep 3 (2)')
})
