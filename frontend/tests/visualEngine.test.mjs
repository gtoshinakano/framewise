import test from 'node:test'
import assert from 'node:assert/strict'
import { engineBadge, installHints, roleOptions } from '../src/visualEngine.ts'

const model = (key, label, roles, available, extra = {}) => ({ key, label, model_id: key, roles, note: '', helper: !roles.length, install: null, available, detail: available ? 'Installed' : 'Not installed', ...extra })
const data = (selected, lfmAvailable = true, dwposeAvailable = true) => ({
  roles: [{ role: 'initial_frame', label: 'First frame', default: 'qwen35-27b' }, { role: 'keyframe_analysis', label: 'Keyframes (motion)', default: 'qwen35-27b' }],
  models: [model('lfm3-omni-2b', 'LFM3-Omni 2B · BF16', ['initial_frame'], lfmAvailable, { install: 'python3 scripts/install-model.py lfm3-omni-2b', detail: lfmAvailable ? 'Installed' : 'Downloading 40%' }),
           model('qwen35-27b', 'Qwen3.5 27B · Q4_K_M', ['initial_frame', 'keyframe_analysis'], true),
           model('dwpose', 'DWPose · ONNX (CPU)', [], dwposeAvailable, { install: 'python3 scripts/install-model.py dwpose' })],
  selected, worker_reported: true,
})

test('a role lists only the models that run it, the default first', () => {
  const value = data({ initial_frame: 'qwen35-27b', keyframe_analysis: 'qwen35-27b' })
  assert.deepEqual(roleOptions(value, value.roles[0]).map(option => option.key), ['qwen35-27b', 'lfm3-omni-2b'])
  assert.deepEqual(roleOptions(value, value.roles[1]).map(option => option.label), ['Qwen3.5 27B · Q4_K_M (default)'])  // the DWPose helper is never a choice
})

test('a model that is not installed cannot be chosen but shows its progress', () => {
  const value = data({ initial_frame: 'qwen35-27b', keyframe_analysis: 'qwen35-27b' }, false, false)
  const lfm = roleOptions(value, value.roles[0]).find(option => option.key === 'lfm3-omni-2b')
  assert.equal(lfm.disabled, true)
  assert.match(lfm.label, /Downloading 40%/)
  assert.deepEqual(installHints(value).map(hint => hint.command), ['python3 scripts/install-model.py lfm3-omni-2b', 'python3 scripts/install-model.py dwpose'])
  // The current choice stays selectable even while its files are missing, so the select is never blank.
  const chosen = data({ initial_frame: 'lfm3-omni-2b', keyframe_analysis: 'qwen35-27b' }, false)
  assert.equal(roleOptions(chosen, chosen.roles[0]).find(option => option.key === 'lfm3-omni-2b').disabled, false)
})

test('the header badge names the non-default models once', () => {
  assert.equal(engineBadge(null), 'checking…')
  assert.equal(engineBadge(data({ initial_frame: 'qwen35-27b', keyframe_analysis: 'qwen35-27b' })), 'Default')
  assert.equal(engineBadge(data({ initial_frame: 'lfm3-omni-2b', keyframe_analysis: 'qwen35-27b' })), 'LFM3-Omni 2B')
})
