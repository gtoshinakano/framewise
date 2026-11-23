import { test } from 'node:test'
import assert from 'node:assert/strict'
import { cardNumber, cards, isCard, nextCardTitle, registered } from '../src/unregisteredView.ts'

const items = [
  { id: 'mai', title: 'Mai' },
  { id: 'u3', title: 'Unregistered 3', unregistered: 1 },
  { id: 'u1', title: 'Unregistered 1', unregistered: 1 },
]

test('cards are told apart from registered characters and ordered by number', () => {
  assert.equal(isCard(items[1]), true)
  assert.equal(isCard(items[0]), false)
  assert.deepEqual(registered(items).map(item => item.id), ['mai'])
  assert.deepEqual(cards(items).map(item => item.title), ['Unregistered 1', 'Unregistered 3'])
  assert.equal(cardNumber('Unregistered 12'), 12)
  assert.equal(cardNumber('Mai'), null)
})

test('the next card takes the lowest free number', () => {
  assert.equal(nextCardTitle(items), 'Unregistered 2')
  assert.equal(nextCardTitle([]), 'Unregistered 1')
})
