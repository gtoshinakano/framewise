import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cutJobsKey, mergeJobs } from '../src/jobsView.ts'

const job = (id, created_at, over = {}) => ({ id, pipeline: 'regen', status: 'done', created_at, cuts: [{ id: 'c1', completed: true }], ...over })

test('the cut history is added behind the episode list, newest first', () => {
  const episode = [job('new', '2026-11-22 10:00:03'), job('mid', '2026-11-22 10:00:02')]
  const history = [job('mid', '2026-11-22 10:00:02', { status: 'running' }), job('old', '2026-11-22 10:00:01'), job('older', '2026-11-22 09:00:00')]
  const merged = mergeJobs(episode, history)
  assert.deepEqual(merged.map(item => item.id), ['new', 'mid', 'old', 'older'])
  // The episode's copy is the fresher one.
  assert.equal(merged[1].status, 'done')
})

test('nothing to add returns the episode list itself', () => {
  const episode = [job('a', '2026-11-22 10:00:00')]
  assert.equal(mergeJobs(episode, null), episode)
  assert.equal(mergeJobs(episode, []), episode)
  assert.equal(mergeJobs(episode, [job('a', '2026-11-22 10:00:00')]), episode)
})

test('jobs created in the same second keep the episode list order', () => {
  const episode = [job('x', '2026-11-22 10:00:00'), job('y', '2026-11-22 10:00:00')]
  const merged = mergeJobs(episode, [job('z', '2026-11-22 10:00:00')])
  assert.deepEqual(merged.map(item => item.id), ['x', 'y', 'z'])
})

test('the history key follows the jobs of the cut and their status only', () => {
  const jobs = [job('a', 't1'), job('b', 't2', { cuts: [{ id: 'c2' }] }), job('c', 't3', { status: 'running' }), { id: 'd', pipeline: 'prepare', status: 'done', created_at: 't4', cuts: [] }]
  assert.equal(cutJobsKey(jobs, 'c1'), 'a:done,c:running')
  assert.equal(cutJobsKey(jobs, 'none'), '')
})
