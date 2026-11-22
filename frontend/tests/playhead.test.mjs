import { test } from 'node:test'
import assert from 'node:assert/strict'
import { PlayheadStore } from '../src/playhead.ts'

test('time updates notify subscribers once and only when the time changed', () => {
  const store = new PlayheadStore(); let calls = 0
  const stop = store.subscribe(() => { calls += 1 })
  store.set(1.5); store.set(1.5); store.set(2)
  assert.equal(calls, 2); assert.equal(store.time, 2); assert.equal(store.seekCount, 0)
  stop(); store.set(3)
  assert.equal(calls, 2)
})

test('a seek is counted and notifies even when the time is the same', () => {
  const store = new PlayheadStore(); const seen = []
  store.subscribe(() => seen.push([store.time, store.seekCount]))
  store.seek(4); store.seek(4); store.markSeek()
  assert.deepEqual(seen, [[4, 1], [4, 2], [4, 3]])
})

test('a subscriber may unsubscribe while being notified', () => {
  const store = new PlayheadStore(); let first = 0, second = 0
  const stop = store.subscribe(() => { first += 1; stop() })
  store.subscribe(() => { second += 1 })
  store.set(1); store.set(2)
  assert.deepEqual([first, second], [1, 2])
})
