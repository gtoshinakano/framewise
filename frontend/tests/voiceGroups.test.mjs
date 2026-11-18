import { test } from 'node:test'
import assert from 'node:assert/strict'
import { unknownVoices } from '../src/voiceGroups.ts'

const line = (id, t0, t1, label, character = null, extra = {}) => ({ id, episode_id: 'e', start_time: t0, end_time: t1,
  text: `line ${id}`, speaker_label: label, character_context_id: character, deleted: false, origin: 'model',
  model_id: null, parent_id: null, superseded_by: null, created_at: '', updated_at: '', ...extra })
const cut = { start_time: 10, end_time: 20 }

test('groups every unassigned line of the episode by voice', () => {
  assert.deepEqual(unknownVoices([
    line('a', 0, 2, 'SPEAKER_01'),
    line('b', 11, 12, 'SPEAKER_01'),
    line('c', 12, 13, 'SPEAKER_02'),
    line('d', 19, 21, ''),
    line('e', 25, 26, 'SPEAKER_02', 'k'),
    line('f', 13, 14, 'SPEAKER_02', null, { deleted: true }),
    line('g', 14, 15, 'SPEAKER_02', null, { superseded_by: 'x' }),
  ], cut), [
    { label: 'SPEAKER_01', lines: 2, inCut: 1 },
    { label: '', lines: 1, inCut: 1 },
    { label: 'SPEAKER_02', lines: 1, inCut: 1 },
  ])
})

test('says nothing when every voice already has a character', () => {
  assert.deepEqual(unknownVoices([line('a', 0, 2, 'SPEAKER_01', 'k'), line('b', 3, 4, '', 'k')], cut), [])
})
