import test from 'node:test'
import assert from 'node:assert/strict'
import { engineBadge, installHints, roleOptions } from '../src/visualEngine.ts'

const model = (key, label, roles, available, extra = {}) => ({ key, label, model_id: key, roles, note: '', hears_audio: false, install: null, available, detail: available ? 'Installed' : 'Not installed', ...extra })
const data = (selected, lfmAvailable = true) => ({
  roles: [{ role: 'initial_frame', label: 'First frame', default: 'qwen35-27b' }, { role: 'video_analysis', label: 'Video motion', default: 'qwen3-vl-8b' }],
  models: [model('lfm3-omni-2b', 'LFM3-Omni 2B · BF16', ['initial_frame', 'video_analysis'], lfmAvailable, { install: 'python3 scripts/install-model.py lfm3-omni-2b', detail: lfmAvailable ? 'Installed' : 'Downloading 40%' }),
           model('qwen35-27b', 'Qwen3.5 27B · Q4_K_M', ['initial_frame'], true), model('qwen3-vl-8b', 'Qwen3-VL 8B · BF16', ['video_analysis'], true)],
  selected, worker_reported: true,
})

test('a role lists only the models that run it, the default first', () => {
  const value = data({ initial_frame: 'qwen35-27b', video_analysis: 'qwen3-vl-8b' })
  assert.deepEqual(roleOptions(value, value.roles[0]).map(option => option.key), ['qwen35-27b', 'lfm3-omni-2b'])
  assert.equal(roleOptions(value, value.roles[1])[0].label, 'Qwen3-VL 8B · BF16 (default)')
})

test('a model that is not installed cannot be chosen but shows its progress', () => {
  const value = data({ initial_frame: 'qwen35-27b', video_analysis: 'qwen3-vl-8b' }, false)
  const lfm = roleOptions(value, value.roles[0]).find(option => option.key === 'lfm3-omni-2b')
  assert.equal(lfm.disabled, true)
  assert.match(lfm.label, /Downloading 40%/)
  assert.deepEqual(installHints(value).map(hint => hint.command), ['python3 scripts/install-model.py lfm3-omni-2b'])
  // The current choice stays selectable even while its files are missing, so the select is never blank.
  const chosen = data({ initial_frame: 'lfm3-omni-2b', video_analysis: 'qwen3-vl-8b' }, false)
  assert.equal(roleOptions(chosen, chosen.roles[0]).find(option => option.key === 'lfm3-omni-2b').disabled, false)
})

test('the header badge names the non-default models once', () => {
  assert.equal(engineBadge(null), 'checking…')
  assert.equal(engineBadge(data({ initial_frame: 'qwen35-27b', video_analysis: 'qwen3-vl-8b' })), 'Default')
  assert.equal(engineBadge(data({ initial_frame: 'lfm3-omni-2b', video_analysis: 'lfm3-omni-2b' })), 'LFM3-Omni 2B')
})
