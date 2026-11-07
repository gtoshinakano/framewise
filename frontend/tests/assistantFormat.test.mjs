import { test } from 'node:test'
import assert from 'node:assert/strict'
import { applyStreamEvent, compactDiff, contextShare, formatSeconds, formatTokens, groupTurns, prettyArguments, starterText } from '../src/assistantFormat.ts'

const message = (id, seq, role, extra = {}) => ({ id, session_id: 's', turn_id: 't1', seq, role, content: '', reasoning: '', tool_calls: [], tool_call_id: null, tool_name: null, tool_summary: null, streaming: false, usage: {}, created_at: '', ...extra })
const turn = (status, extra = {}) => ({ id: 't1', session_id: 's', status, stage: '', provider: 'local', model: 'm', thinking: true, skill: null, cancel_requested: false, attempt_count: 1, error: null, usage: {}, version: 1, created_at: '', started_at: null, finished_at: null, ...extra })
const base = { session: { id: 's' }, messages: [message('u', 1, 'user', { content: 'Hi' })], turns: [turn('queued')], proposals: [] }

test('stream events build up a turn and deltas append text', () => {
  let detail = applyStreamEvent(base, 'snapshot', { turn: turn('running'), messages: [message('u', 1, 'user', { content: 'Hi' }), message('a', 2, 'assistant', { streaming: true })], proposals: [] })
  assert.equal(detail.turns[0].status, 'running')
  detail = applyStreamEvent(detail, 'delta', { id: 'a', content: 'Hel', reasoning: 'think' })
  detail = applyStreamEvent(detail, 'delta', { id: 'a', content: 'lo', reasoning: '' })
  assert.equal(detail.messages[1].content, 'Hello')
  assert.equal(detail.messages[1].reasoning, 'think')
  detail = applyStreamEvent(detail, 'message', message('a', 2, 'assistant', { content: 'Hello!', streaming: false }))
  assert.equal(detail.messages.length, 2)
  assert.equal(detail.messages[1].content, 'Hello!')
  detail = applyStreamEvent(detail, 'proposal', { id: 'p', turn_id: 't1', status: 'pending' })
  detail = applyStreamEvent(detail, 'done', turn('done'))
  assert.equal(detail.turns[0].status, 'done')
  assert.equal(detail.proposals.length, 1)
})

test('turns group tool calls with their results', () => {
  const detail = { ...base, turns: [turn('done')], messages: [
    message('u', 1, 'user', { content: 'Fix name' }),
    message('a1', 2, 'assistant', { reasoning: 'r', tool_calls: [{ id: 'c1', name: 'get_result', arguments: '{"type":"video_analysis"}' }] }),
    message('r1', 3, 'tool', { tool_call_id: 'c1', tool_name: 'get_result', content: '{}' }),
    message('a2', 4, 'assistant', { content: 'Done' }),
  ], proposals: [{ id: 'p', turn_id: 't1' }] }
  const [view] = groupTurns(detail)
  assert.equal(view.user.content, 'Fix name')
  assert.equal(view.steps.length, 2)
  assert.equal(view.steps[0].tools[0].result.id, 'r1')
  assert.equal(view.steps[1].message.content, 'Done')
  assert.equal(view.proposals.length, 1)
})

test('formatting helpers', () => {
  assert.equal(formatTokens(1234), '1,234')
  assert.equal(formatTokens(65536), '65.5k')
  assert.equal(formatTokens(undefined), '—')
  assert.equal(formatSeconds(2.345), '2.3s')
  assert.equal(formatSeconds(75), '1m 15s')
  assert.deepEqual(contextShare({ context_tokens: 6554, context_limit: 65536 }), { used: 6554, limit: 65536, percent: 10 })
  assert.equal(contextShare({}), null)
  assert.equal(prettyArguments('{}'), '')
  assert.equal(prettyArguments('{"a":1}'), '{\n  "a": 1\n}')
  assert.equal(prettyArguments('not json'), 'not json')
  assert.equal(starterText('h3_animation_prompt', 'Fix this H3 prompt based on results'), 'Fix this H3 prompt based on results:\n\n')
  assert.equal(starterText('cut_status_summary', 'Summarize'), 'Summarize')
})

test('diffs keep edits and shorten long unchanged text', () => {
  const long = 'x'.repeat(500)
  const segments = compactDiff([{ op: 'equal', text: long }, { op: 'delete', text: 'a person' }, { op: 'insert', text: 'Hero' }, { op: 'equal', text: long }])
  assert.ok(segments[0].text.length < 200)
  assert.equal(segments[1].text, 'a person')
  assert.equal(segments[2].text, 'Hero')
  assert.ok(segments[3].text.length < 200)
})
