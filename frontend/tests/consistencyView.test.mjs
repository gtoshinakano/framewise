import { test } from 'node:test'
import assert from 'node:assert/strict'
import { answerLine, answerLabel, noneNote, parseConsistency, parseQuestions, parseVisualChecks, selection, setAnswer, splitItems, splitSentences, timeParts, topicLabel, unwrapList } from '../src/consistencyView.ts'

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

const QUESTIONS = parseQuestions([
  { id: 'q1', topic: 'gaze', character: 'Mai', question: 'Where does Mai look?', recommended: 'a', why: 'first frame', options: [{ id: 'a', text: 'At Rio' }, { id: 'b', text: 'Down at the cup' }] },
  { id: 'q2', topic: 'state', character: '', question: 'What is Rio doing?', options: [{ id: 'a', text: 'Sitting' }, { id: 'b', text: 'Standing' }, 'Walking'] },
])

test('questions are read leniently and a result with them has no text sections to decide', () => {
  assert.equal(QUESTIONS.length, 2)
  assert.deepEqual([QUESTIONS[0].recommended, QUESTIONS[1].recommended], ['a', null])
  assert.deepEqual(QUESTIONS[1].options.map(option => option.id), ['a', 'b', 'c'])  // an option written as plain text gets an id
  assert.deepEqual(parseQuestions([{ question: 'one option', options: ['x'] }, { question: '', options: ['a', 'b'] }, 'junk', null]), [])
  assert.deepEqual(parseQuestions('nothing'), [])
  assert.equal(parseQuestions([{ question: 'q', recommended: 'zz', options: ['a', 'b'] }])[0].recommended, null)
  const view = parseConsistency(JSON.stringify({ summary: 's', clear: 'c', confusing: 'None identified.', questions: QUESTIONS }))
  assert.equal(view.format, 'questions')
  assert.equal(view.decisions, 2)
  assert.equal(view.questions.length, 2)
  const old = parseConsistency(JSON.stringify({ summary: 's', human_review: 'Confirm the coat.', recommended_checks: 'None required.' }))
  assert.equal(old.format, 'text')
  assert.equal(old.decisions, 1)
  assert.equal(parseConsistency(JSON.stringify({ summary: 's', questions: [] })).decisions, 0)
})

test('visual checks are read leniently, contradictions first, and missing on reviews that did not look', () => {
  const checks = parseVisualChecks([
    { character: 'Mai', picture: 'F2', source: 'keyframes', claim: 'smiles', seen: 'brows raised, eyes wide, mouth open: surprise', verdict: 'agrees' },
    { character: 'Mai', picture: 'First frame', source: 'first_frame', claim: 'big smile', seen: 'raised brows, wide eyes, open mouth: surprised', verdict: 'Contradicts' },
    { picture: 'Line 1.00-2.00s', claim: 'on_screen', seen: 'face hidden', verdict: 'unclear' },
    { picture: '', claim: 'x', verdict: 'agrees' }, { picture: 'F1', verdict: 'maybe', claim: 'x' }, { picture: 'F1', verdict: 'agrees' }, 'junk', null,
  ])
  assert.deepEqual(checks.map(check => [check.picture, check.verdict]), [['First frame', 'contradicts'], ['Line 1.00-2.00s', 'unclear'], ['F2', 'agrees']])
  assert.equal(parseVisualChecks(undefined), null)
  assert.equal(parseConsistency(JSON.stringify({ summary: 's', questions: [] })).visualChecks, null)
  assert.deepEqual(parseConsistency(JSON.stringify({ visual_check: [], summary: 's', questions: [] })).visualChecks, [])
})

test('an answer is one bullet line named by topic and character', () => {
  assert.equal(topicLabel('gaze'), 'Gaze')
  assert.equal(answerLabel(QUESTIONS[0]), 'Gaze (Mai)')
  assert.equal(answerLabel(QUESTIONS[1]), 'State')
  assert.equal(answerLine(QUESTIONS[0], 'At  Rio\n'), '- Gaze (Mai): At Rio')
})

test('choosing writes, replaces and clears the line of that question only', () => {
  let text = ''
  text = setAnswer(text, QUESTIONS[0], 'At Rio', QUESTIONS)
  assert.equal(text, '- Gaze (Mai): At Rio')
  text = setAnswer(text, QUESTIONS[1], 'Sitting', QUESTIONS)
  assert.equal(text, '- Gaze (Mai): At Rio\n- State: Sitting')
  text = setAnswer(text, QUESTIONS[0], 'Down at the cup', QUESTIONS)  // a second choice replaces the first
  assert.equal(text, '- Gaze (Mai): Down at the cup\n- State: Sitting')
  assert.deepEqual(selection(text, QUESTIONS[0], QUESTIONS), { option: 'b', custom: '' })
  text = setAnswer(text, QUESTIONS[0], null, QUESTIONS)
  assert.equal(text, '- State: Sitting')
  assert.equal(selection(text, QUESTIONS[0], QUESTIONS), null)
})

test('text the user wrote around the choices is kept, and an own answer shows as custom', () => {
  const notes = 'Keep the coat green\n\n- Camera: static\n  but slightly handheld'
  const withOwn = setAnswer(notes, QUESTIONS[0], 'She looks at the window', QUESTIONS)
  assert.equal(withOwn, `${notes}\n- Gaze (Mai): She looks at the window`)
  assert.deepEqual(selection(withOwn, QUESTIONS[0], QUESTIONS), { option: null, custom: 'She looks at the window' })
  assert.equal(setAnswer(withOwn, QUESTIONS[0], null, QUESTIONS), notes)
  // A wrapped answer (the AI draft may use two lines) is replaced as one.
  const wrapped = '- Gaze (Mai): At Rio and then\n  down at the cup\n- State: Standing'
  assert.deepEqual(selection(wrapped, QUESTIONS[1], QUESTIONS), { option: 'b', custom: '' })
  assert.equal(setAnswer(wrapped, QUESTIONS[0], 'Down at the cup', QUESTIONS), '- Gaze (Mai): Down at the cup\n- State: Standing')
})

test('a line that copies an option word for word is found whatever its label says, and labels survive case and a trailing full stop', () => {
  assert.deepEqual(selection('- Eyes: at rio.', QUESTIONS[0], QUESTIONS), { option: 'a', custom: '' })
  assert.deepEqual(selection('- gaze (mai): Down at the cup', QUESTIONS[0], QUESTIONS), { option: 'b', custom: '' })
  assert.equal(selection('- Camera: static', QUESTIONS[0], QUESTIONS), null)
})

test('settled corrections are parsed, kept apart from what the person decides and not counted', async () => {
  const { asked, settledBy } = await import('../src/consistencyView.ts')
  const questions = parseQuestions([
    { id: 'q1', topic: 'emotion', character: 'Mai', question: 'How does Mai feel?', options: ['calm', 'irritated'], recommended: 'a' },
    { id: 'q2', topic: 'gaze', character: 'Mai', question: 'Gaze of Mai, settled by the gaze rule.', options: ['Looks past the camera at Rio', 'At the camera'], recommended: 'a', settled: 'gaze_rule' },
  ])
  assert.equal(questions[1].settled, 'gaze_rule')
  assert.equal('settled' in questions[0], false)
  assert.deepEqual(asked(questions).map(question => question.id), ['q1'])
  assert.equal(settledBy('gaze_rule'), 'the gaze rule')
  assert.equal(parseConsistency(JSON.stringify({ summary: 's', questions })).decisions, 1)
})
