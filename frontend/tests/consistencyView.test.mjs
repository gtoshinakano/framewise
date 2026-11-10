import { test } from 'node:test'
import assert from 'node:assert/strict'
import { noneNote, parseConsistency, splitItems, splitSentences, timeParts, unwrapList } from '../src/consistencyView.ts'

test('a JSON array written inside a string becomes one line per entry, even when it is not valid JSON', () => {
  assert.equal(unwrapList('[\n  "Determine if A and B merge.",\n  "Confirm the spelling (\\"Mai-san\\")."\n]'), 'Determine if A and B merge.\nConfirm the spelling ("Mai-san").')
  assert.equal(unwrapList('["None required."]'), 'None required.')
  assert.equal(unwrapList('[\n  "He said "hi" twice.",\n  "Second."\n]'), 'He said "hi" twice.\nSecond.')
  assert.equal(unwrapList('[1] is a footnote marker and not a list'), '[1] is a footnote marker and not a list')
  const view = parseConsistency(JSON.stringify({ human_review: '[\n  "Determine if A and B merge.",\n  "Confirm the spelling."\n]', recommended_checks: '["None required."]' }))
  assert.deepEqual(view.sections.human_review.items, ['Determine if A and B merge.', 'Confirm the spelling.'])
  assert.equal(view.sections.recommended_checks.none, true)
})

test('bullets and numbers are removed from paragraph fields too', () => {
  const view = parseConsistency(JSON.stringify({ confusing: '1. Gender differs: boy vs girl.\n2. Camera framing differs.', clear: '- Camera is static.\n- Poses match.' }))
  assert.deepEqual(view.sections.confusing.paragraphs, ['Gender differs: boy vs girl.', 'Camera framing differs.'])
  assert.deepEqual(view.sections.clear.paragraphs, ['Camera is static.', 'Poses match.'])
})

test('"None required" is recognised with or without an explanation, but real content starting with "None" is not', () => {
  assert.deepEqual(noneNote('None required.'), { none: true, note: '' })
  assert.deepEqual(noneNote('None identified.'), { none: true, note: '' })
  assert.deepEqual(noneNote('none'), { none: true, note: '' })
  assert.deepEqual(noneNote('None required. The evidence is consistent.'), { none: true, note: 'The evidence is consistent.' })
  assert.equal(noneNote('None of the characters are confirmed by a detection box.').none, false)
  assert.equal(noneNote('None required, but confirm the blazer colour.').none, false)
  assert.equal(noneNote('Verify the framing.').none, false)
})

test('sentences split on a capital letter but not inside decimals, timestamps or abbreviations', () => {
  assert.deepEqual(
    splitSentences('Compare the opening and last frames. Cross-reference the events (1269.19s and 1269.69s) with the dialogue at 1268.80s.'),
    ['Compare the opening and last frames.', 'Cross-reference the events (1269.19s and 1269.69s) with the dialogue at 1268.80s.'])
  assert.deepEqual(splitSentences('Check the spans around approx. 1227.89s to confirm the timing.'), ['Check the spans around approx. 1227.89s to confirm the timing.'])
  assert.deepEqual(splitSentences('Confirm tan vs. brown blazer, e.g. under the lamp. Then update the costume notes.'), ['Confirm tan vs. brown blazer, e.g. under the lamp.', 'Then update the costume notes.'])
  assert.deepEqual(splitSentences('Ask J. Smith about the line. Then record it.'), ['Ask J. Smith about the line.', 'Then record it.'])
})

test('quoted dialogue and closing quotes stay attached to their sentence', () => {
  assert.deepEqual(
    splitSentences('Confirm that Sakuta says "なんで？" at the end. Mai answers afterwards.'),
    ['Confirm that Sakuta says "なんで？" at the end.', 'Mai answers afterwards.'])
  assert.deepEqual(splitSentences('Is the zoom real? Check the last frame.'), ['Is the zoom real?', 'Check the last frame.'])
})

test('a one-sentence field stays one item and a stray short fragment is merged', () => {
  assert.deepEqual(splitItems('Verify the framing transition between both frames.'), ['Verify the framing transition between both frames.'])
  assert.deepEqual(splitSentences('Fix it. Confirm the long pan between the two frames.'), ['Fix it. Confirm the long pan between the two frames.'])
  assert.deepEqual(splitItems(''), [])
})

test('lines and bullets win over sentence splitting', () => {
  assert.deepEqual(splitItems('- Check A. Then B.\n- Check C'), ['Check A. Then B.', 'Check C'])
  assert.deepEqual(splitItems('1. First thing\n2) Second thing'), ['First thing', 'Second thing'])
  assert.deepEqual(splitItems('1. Only one numbered thing'), ['Only one numbered thing'])
})

test('timestamps and ranges are marked for styling', () => {
  assert.deepEqual(timeParts('Events at 1269.19s and 0.0–0.4s done'), [
    { text: 'Events at ', time: false }, { text: '1269.19s', time: true }, { text: ' and ', time: false },
    { text: '0.0–0.4s', time: true }, { text: ' done', time: false }])
  assert.deepEqual(timeParts('1251.600–1252.080s'), [{ text: '1251.600–1252.080s', time: true }])
  assert.deepEqual(timeParts('Among 10 sides and H2s'), [{ text: 'Among 10 sides and H2s', time: false }])
  assert.deepEqual(timeParts(''), [])
})

test('parseConsistency separates items, paragraphs and "none" and counts what needs attention', () => {
  const view = parseConsistency(JSON.stringify({
    summary: 'A static two-shot.', clear: 'Poses match.\nTiming matches.', confusing: 'The framing changes.',
    human_review: 'Verify the zoom. Confirm the blazer colour.', recommended_checks: 'None required.',
  }))
  assert.equal(view.valid, true)
  assert.deepEqual(view.sections.human_review.items, ['Verify the zoom.', 'Confirm the blazer colour.'])
  assert.equal(view.sections.recommended_checks.none, true)
  assert.deepEqual(view.sections.recommended_checks.items, [])
  assert.deepEqual(view.sections.clear.paragraphs, ['Poses match.', 'Timing matches.'])
  assert.equal(view.decisions, 2)
  assert.equal(view.checks, 0)
  assert.equal(view.conflicts, true)
})

test('all-clear results report nothing to do and invalid or partial content is flagged', () => {
  const clear = parseConsistency(JSON.stringify({ summary: 'x', clear: 'y', confusing: 'None identified.', human_review: 'None required. Known model limitation.', recommended_checks: 'None required.' }))
  assert.equal(clear.conflicts, false)
  assert.equal(clear.decisions + clear.checks, 0)
  assert.equal(clear.sections.human_review.note, 'Known model limitation.')
  assert.equal(parseConsistency('not json').valid, false)
  assert.equal(parseConsistency('[1,2]').valid, false)
  const partial = parseConsistency(JSON.stringify({ summary: 5 }))
  assert.equal(partial.valid, true)
  assert.equal(partial.sections.summary.present, false)
  assert.equal(partial.conflicts, false)
})
