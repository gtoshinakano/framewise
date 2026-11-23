import type { ContextItem } from './api'
import { cards, NEW_CARD, nextCardTitle, registered } from './unregisteredView'

/** A character select's options: the registered characters, then the Unregistered cards and "+ Unregistered N" (value `NEW_CARD`). */
export function CharacterOptions({ characters, allowNew = true }: { characters: ContextItem[]; allowNew?: boolean }) {
  const numbered = cards(characters)
  return <>
    {registered(characters).map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
    {(numbered.length > 0 || allowNew) && <optgroup label="Unregistered (minor people, no record)">
      {numbered.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}
      {allowNew && <option value={NEW_CARD}>+ {nextCardTitle(characters)}</option>}
    </optgroup>}
  </>
}
