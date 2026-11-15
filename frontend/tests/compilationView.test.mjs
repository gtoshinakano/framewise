import { test } from 'node:test'
import assert from 'node:assert/strict'
import { applyVoice, changed, counts, diff, isNew, lineProblems, mergeNext, needsReview, nudge, setSpeaker, setText, splitAt, visibleLines } from '../src/compilationView.ts'

const line = (id, t0, t1, text, extra = {}) => ({ id, t0, t1, text, original_text: text, subtitle_text: null, speaker_label: 'SPEAKER_01', character: 'Hero', character_context_id: 'h',
  continues_next: false, confidence: 'high', note: '', source_ids: [`s${id}`], subtitle_ids: [], parts: [{ t0, t1, text }], ...extra })
const people = [{ id: 'h', name: 'Hero' }, { id: 'f', name: 'Friend' }]

test('nudging keeps a line inside the sequence and at least 0.2 s long', () => {
  const base = line('C1', 1, 2, 'はい')
  assert.equal(nudge(base, 't0', -5, 10).t0, 0)
  assert.equal(nudge(base, 't0', 5, 10).t0, 1.8)
  assert.equal(nudge(base, 't1', 20, 10).t1, 10)
  assert.equal(nudge(base, 't1', -5, 10).t1, 1.2)
})

test('changing the text spreads its words over the line', () => {
  const edited = setText(line('C1', 2, 5, 'はい'), 'おはようございます')
  assert.equal(edited.parts[0].t0, 2)
  assert.equal(edited.parts.at(-1).t1, 5)
  assert.equal(edited.parts.map(part => part.text).join(''), 'おはようございます')
})

test('merging joins Japanese without a space and Latin words with one', () => {
  const merged = mergeNext([line('C1', 0, 1, '私は昨日', { continues_next: true }), line('C2', 1.2, 2, 'に行きました。')], 0)
  assert.equal(merged.length, 1)
  assert.deepEqual([merged[0].text, merged[0].t0, merged[0].t1, merged[0].continues_next, merged[0].source_ids], ['私は昨日に行きました。', 0, 2, false, ['sC1', 'sC2']])
  assert.equal(mergeNext([line('C1', 0, 1, 'Good'), line('C2', 1, 2, 'morning')], 0)[0].text, 'Good morning')
  assert.equal(mergeNext([line('C1', 0, 1, 'x')], 0).length, 1)
})

test('splitting cuts the text in proportion to the time and refuses edges', () => {
  const lines = splitAt([line('C1', 0, 4, '今日はいい天気ですね')], 0, 2)
  assert.deepEqual(lines.map(item => [item.t0, item.t1, item.text]), [[0, 2, '今日はいい'], [2, 4, '天気ですね']])
  assert.equal(lines[0].continues_next, true)
  assert.equal(splitAt([line('C1', 0, 4, 'あいうえお')], 0, 0.1).length, 1)
})

test('a voice choice applies to every line of the label and to the voices table', () => {
  const content = { lines: [line('C1', 0, 1, 'a'), line('C2', 1, 2, 'b', { speaker_label: 'SPEAKER_02', character: null, character_context_id: null }), line('C3', 2, 3, 'c', { speaker_label: 'SPEAKER_02', character: null, character_context_id: null })], voices: [], open_questions: [] }
  const out = applyVoice(content, 'SPEAKER_02', 'Friend', people)
  assert.deepEqual(out.lines.map(item => item.character), ['Hero', 'Friend', 'Friend'])
  assert.equal(out.lines[1].character_context_id, 'f')
  assert.deepEqual(out.voices.map(voice => [voice.speaker_label, voice.character, voice.kind]), [['SPEAKER_02', 'Friend', 'character']])
  assert.equal(applyVoice(out, 'SPEAKER_02', 'Narrator', people).voices[0].kind, 'narrator')
})

test('a line carries words, speaker and timing only: how it is played is decided per shot', () => {
  const spoken = setSpeaker(line('C1', 0, 1, 'a'), 'Friend', people)
  assert.deepEqual([spoken.character, spoken.character_context_id], ['Friend', 'f'])
  assert.equal('delivery' in spoken, false)
  assert.deepEqual(lineProblems(line('C1', 0, 1, 'a', { character: null }), 10), [])
  assert.deepEqual(lineProblems(line('C1', 0, 11, ''), 10).length, 2)
})

test('review flags and filters', () => {
  const lines = [line('C1', 0, 1, 'a'), line('C2', 1, 2, 'b', { confidence: 'low' }), line('C3', 2, 3, 'c', { character: null }), line('C4', 3, 4, 'd', { text: 'D' })]
  assert.deepEqual([needsReview(lines[0]), needsReview(lines[1]), needsReview(lines[2])], [false, true, true])
  assert.equal(changed(lines[3]), true)
  assert.deepEqual(visibleLines(lines, 'review').map(item => item.id), ['C2', 'C3'])
  assert.deepEqual(visibleLines(lines, 'changed').map(item => item.id), ['C4'])
  assert.deepEqual(counts(lines), { total: 4, review: 2, changed: 1, added: 0 })
})

test('the difference highlights corrected kanji', () => {
  const parts = diff('今日は意見いい天気', '今日はいい天気')
  assert.equal(parts.filter(part => part.kind === 'del').map(part => part.text).join(''), '意見')
  assert.equal(parts.map(part => part.text).filter((_, index) => parts[index].kind !== 'del').join(''), '今日はいい天気')
  assert.deepEqual(diff('同じ', '同じ'), [{ text: '同じ', kind: 'same' }])
})

test('lines of an added cut are new and count as lines to review', () => {
  const lines = [line('C1', 1, 2, 'はい'), line('C2', 3.7, 4, 'ちょっと', { continues_next: true }), line('C3', 3.95, 4.6, '待って'), line('C4', 5, 6, 'じゃあね')]
  assert.deepEqual(lines.map(item => isNew(item, 4)), [false, false, true, true])  // a new line may start just before the old end, its middle may not
  assert.equal(isNew(lines[3], null), false)
  assert.deepEqual(counts(lines, 4), { total: 4, review: 3, changed: 0, added: 2 })
  assert.deepEqual(counts(lines), { total: 4, review: 1, changed: 0, added: 0 })
  assert.deepEqual(visibleLines(lines, 'review', 4).map(item => item.id), ['C2', 'C3', 'C4'])
})
