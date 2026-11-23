import type { ContextItem } from './api'

/** Select value that asks for a new Unregistered card (the lowest free number). */
export const NEW_CARD = '__new_unregistered'

type Item = Pick<ContextItem, 'title'> & { unregistered?: number | boolean }

/** A numbered Unregistered card: a minor person with a line or a box but no character record (backend/app/unregistered.py). */
export const isCard = (item: Item | null | undefined) => !!item?.unregistered

/** The number of an `Unregistered N` title, else null. */
export function cardNumber(title: string): number | null {
  const match = /^Unregistered (\d+)$/.exec(title.trim())
  return match ? Number(match[1]) : null
}

/** Characters that are real records: what recognition, voice samples, outfits and profiles list. */
export const registered = <T extends Item>(items: T[]): T[] => items.filter(item => !isCard(item))

/** The project's cards in number order. */
export const cards = <T extends Item>(items: T[]): T[] =>
  items.filter(isCard).sort((a, b) => (cardNumber(a.title) ?? 0) - (cardNumber(b.title) ?? 0))

/** The title the next card gets: the lowest number no character uses. */
export function nextCardTitle(items: Item[]): string {
  const taken = new Set(items.map(item => cardNumber(item.title)).filter((value): value is number => value !== null))
  let number = 1
  while (taken.has(number)) number += 1
  return `Unregistered ${number}`
}
