import test from 'node:test'
import assert from 'node:assert/strict'
import { modelChoices, parseRouteValue, remoteBadge, routeOptions, routeValue, variantLabel } from '../src/remoteText.ts'

test('route values round-trip and fall back to this server', () => {
  for (const route of [{ mode: 'remote', model: 'bonsai2' }, { mode: 'any', model: 'org/qwen|odd' }]) assert.deepEqual(parseRouteValue(routeValue(route)), route)
  assert.equal(routeValue(undefined), 'local')
  assert.equal(routeValue({ mode: 'any', model: '' }), 'local')
  assert.deepEqual(parseRouteValue('local'), { mode: 'local', model: '' })
  assert.deepEqual(parseRouteValue('bogus|m'), { mode: 'local', model: '' })
})

test('options keep a saved model the server no longer reports', () => {
  const models = [{ id: 'qwen3.6-mtp', n_ctx: 32768 }]
  assert.deepEqual(modelChoices(models, ['bonsai2', '']), [{ id: 'qwen3.6-mtp', listed: true }, { id: 'bonsai2', listed: false }])
  const options = routeOptions(models, { mode: 'remote', model: 'bonsai2' })
  assert.deepEqual(options.map(option => option.value), ['local', 'remote|qwen3.6-mtp', 'any|qwen3.6-mtp', 'remote|bonsai2', 'any|bonsai2'])
  assert.match(options[3].label, /not reported/)
})

test('labels for the speed table and the header badge', () => {
  assert.equal(variantLabel('remote:bonsai2', { mtp: 'MTP' }), 'Remote · bonsai2')
  assert.equal(variantLabel('mtp', { mtp: 'MTP' }), 'MTP')
  const state = { state: 'online', detail: '', models: [], active_jobs: 0, heartbeat: 0 }
  const server = { enabled: true, base_url: '', has_token: false, parallel: 1, routes: {}, assistant_fallback: '', state }
  assert.equal(remoteBadge(server), ' · Remote online')
  assert.equal(remoteBadge({ ...server, state: { ...state, active_jobs: 2 } }), ' · Remote 2 running')
  assert.equal(remoteBadge({ ...server, state: { ...state, state: 'auth_failed' } }), ' · Remote login refused')
  assert.equal(remoteBadge({ ...server, enabled: false }), '')
})

test('unrouted jobs share the work with the default remote model', async () => {
  const { shownRoute, defaultModel, jobSlots } = await import('../src/remoteText.ts')
  assert.deepEqual(shownRoute(undefined, 'qwen'), { mode: 'any', model: 'qwen' })
  assert.deepEqual(shownRoute(undefined, ''), { mode: 'local', model: '' })
  assert.deepEqual(shownRoute({ mode: 'local', model: '' }, 'qwen'), { mode: 'local', model: '' })
  assert.equal(defaultModel('', [{ id: 'first' }]), 'first')
  assert.equal(defaultModel('chosen', [{ id: 'first' }]), 'chosen')
  assert.equal(jobSlots(2, 1), 1)
  assert.equal(jobSlots(3, 1), 2)
  assert.equal(jobSlots(1, 1), 1)
  assert.equal(jobSlots(4, 9), 1)
})
