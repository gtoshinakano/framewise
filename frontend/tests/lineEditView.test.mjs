import { test } from 'node:test'
import assert from 'node:assert/strict'
import { binsFor, clampEdge, contiguousRange, crossing, mergeText, newLineSpan, overviewBars, percentIn, splitTime, timeAt, windowFor } from '../src/lineEditView.ts'

const cut = { start_time: 10, end_time: 20 }
const line = (start_time, end_time, extra = {}) => ({ start_time, end_time, words_json: null, ...extra })

test('the window widens to every line that overlaps the cut and ignores the others', () => {
  assert.deepEqual(windowFor(cut, []), { start: 10, end: 20 })
  assert.deepEqual(windowFor(cut, [line(8, 12), line(18, 23.5), line(12, 14)]), { start: 8, end: 23.5 })
  assert.deepEqual(windowFor(cut, [line(2, 10), line(20, 30)]), { start: 10, end: 20 })
  assert.deepEqual(windowFor({ start_time: 1, end_time: 4 }, [line(-0, 2)]), { start: 0, end: 4 })
})

test('bins follow the window length within limits', () => {
  assert.equal(binsFor({ start: 0, end: 1 }), 50)
  assert.equal(binsFor({ start: 0, end: 10 }), 500)
  assert.equal(binsFor({ start: 0, end: 200 }), 1500)
})

test('an edge stays inside the bounds and leaves the line its minimum length', () => {
  const span = { start: 12, end: 15 }, bounds = { start: 8, end: 23 }
  assert.equal(clampEdge('start', 2, span, bounds), 8)
  assert.equal(clampEdge('start', 14.99, span, bounds), 14.9)
  assert.equal(clampEdge('end', 30, span, bounds), 23)
  assert.equal(clampEdge('end', 12.01, span, bounds), 12.1)
  assert.equal(clampEdge('end', 16.12349, span, bounds), 16.123)
  // A new line is bounded by the cut.
  assert.equal(clampEdge('start', 4, { start: 11, end: 12 }, { start: 10, end: 20 }), 10)
})

test('a new line is one second long at the playhead and never leaves the cut', () => {
  assert.deepEqual(newLineSpan(cut, 12), { start: 12, end: 13 })
  assert.deepEqual(newLineSpan(cut, 3), { start: 10, end: 11 })
  assert.deepEqual(newLineSpan(cut, 19.8), { start: 19, end: 20 })
  assert.deepEqual(newLineSpan({ start_time: 5, end_time: 5.4 }, 9), { start: 5, end: 5.4 })
})

test('a drag across rows selects the run between them in either direction', () => {
  const order = ['a', 'b', 'c', 'd']
  assert.deepEqual(contiguousRange(order, 'b', 'd'), ['b', 'c', 'd'])
  assert.deepEqual(contiguousRange(order, 'c', 'a'), ['a', 'b', 'c'])
  assert.deepEqual(contiguousRange(order, 'c', 'c'), ['c'])
  assert.deepEqual(contiguousRange(order, 'c', 'zz'), [])
})

test('merged text puts a space between words and none between Japanese phrases', () => {
  assert.equal(mergeText(['Hello there', ' how are you ']), 'Hello there how are you')
  assert.equal(mergeText(['Fine.', 'Thanks']), 'Fine. Thanks')
  assert.equal(mergeText(['こんにちは', '元気ですか']), 'こんにちは元気ですか')
  assert.equal(mergeText(['Hi', '']), 'Hi')
})

const spoken = { start_time: 4, end_time: 6, words_json: JSON.stringify([[4, 4.5, 'Hello'], [4.5, 5.2, 'there'], [5.2, 6, 'friend']]) }

test('a split between words takes the time of the word before it', () => {
  assert.equal(splitTime(spoken, 'Hello there friend', 11), 5.2)
  assert.equal(splitTime(spoken, 'Hello there friend', 5), 4.5)
})

test('a caret inside a word goes to the side that holds most of it', () => {
  assert.equal(splitTime(spoken, 'Hello there friend', 8), 4.5)   // "Hello th|ere": more of the word is before
  assert.equal(splitTime(spoken, 'Hello there friend', 14), 5.2)  // "Hello there fr|iend"... the word `friend` is mostly after
})

test('without usable word times the split follows the caret share of the line', () => {
  assert.equal(splitTime(line(0, 10), 'abcd efgh', 4), 4.444)
  assert.equal(splitTime({ ...spoken, words_json: null }, 'Hello there friend', 9), 5)
  // An edited text no longer matches the words.
  assert.equal(splitTime(spoken, 'Hello there my dear friend', 9), 4.692)
  assert.equal(splitTime(spoken, 'garbled', 3) > 4, true)
})

test('a caret that leaves one side empty has no split time', () => {
  assert.equal(splitTime(spoken, 'Hello there friend', 0), null)
  assert.equal(splitTime(spoken, 'Hello there friend', 18), null)
  assert.equal(splitTime(spoken, 'Hello there friend ', 18), null)
})

test('overview bars resample the episode waveform for the window', () => {
  const waveform = { duration: 10, peaks: [0, 0, 0.5, 0.5, 0, 0, 0, 0, 1, 0] }
  assert.deepEqual(overviewBars(waveform, { start: 2, end: 4 }, 2), [0.5, 0.5])
  assert.deepEqual(overviewBars(null, { start: 0, end: 1 }, 4), [])
  assert.deepEqual(overviewBars(waveform, { start: 3, end: 3 }, 4), [])
})

test('the chip says which side of the cut a line crosses', () => {
  assert.equal(crossing(line(11, 15), cut), null)
  assert.equal(crossing(line(8, 15), cut), 'before')
  assert.equal(crossing(line(15, 22), cut), 'after')
  assert.equal(crossing(line(8, 22), cut), 'both')
})

test('positions and times convert between the graph and the episode clock', () => {
  const window = { start: 8, end: 24 }
  assert.equal(percentIn(window, 16), 50)
  assert.equal(percentIn(window, 30), 100)
  assert.equal(timeAt(window, 0.25), 12)
  assert.equal(timeAt(window, 2), 24)
})
