import test from 'node:test'
import assert from 'node:assert/strict'
import { groupOutfits, originLabel, outfitsOf, preview } from '../src/outfitView.ts'

const row = (character, name, extra = {}) => ({ id: `${character}-${name}`, project_id: 'p', character_context_id: character.toLowerCase(), character_name: character, name, summary: '', description: 'd', origin: 'user', source_sequence_id: null, needs_review: 0, created_at: '', updated_at: '', ...extra })

test('outfits are grouped by character and sorted by name', () => {
  const groups = groupOutfits([row('Mai', 'School Uniform'), row('Fumika', 'Pale Blue Blouse'), row('Mai', 'black Bunny Costume')])
  assert.deepEqual(groups.map(group => group.character_name), ['Fumika', 'Mai'])
  assert.deepEqual(groups[1].outfits.map(item => item.name), ['black Bunny Costume', 'School Uniform'])
})

test('origin labels say who made the outfit', () => {
  assert.equal(originLabel('director'), 'Sequence director')
  assert.equal(originLabel('seed'), 'Imported')
  assert.equal(originLabel('user'), 'You')
})

test('the bible lists outfits per character and tolerates bibles without any', () => {
  const outfits = [{ character: 'Mai', name: 'A', status: 'new' }, { character: 'Rio', name: 'B', status: 'existing' }]
  assert.deepEqual(outfitsOf(outfits, 'Mai').map(item => item.name), ['A'])
  assert.deepEqual(outfitsOf(undefined, 'Mai'), [])
})

test('preview cuts a long description at a word', () => {
  assert.equal(preview('short text'), 'short text')
  const text = preview('word '.repeat(60), 30)
  assert.ok(text.endsWith('…') && text.length <= 31)
})
