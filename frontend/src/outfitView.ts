/** Pure helpers for the outfit library dialog and the bible's outfit list. */

export type Outfit = {
  id: string
  project_id: string
  character_context_id: string
  character_name: string
  name: string
  summary: string
  description: string
  origin: 'seed' | 'user' | 'director'
  source_sequence_id: string | null
  needs_review: number
  created_at: string
  updated_at: string
}

export type BibleOutfit = {
  character: string
  character_context_id: string
  outfit_id?: string
  name: string
  status: 'existing' | 'new'
  summary: string
  scenes?: string[]
}

export type OutfitGroup = { character_context_id: string; character_name: string; outfits: Outfit[] }

/** Outfits grouped by character, characters and outfits in name order (the API already sorts, but the dialog edits locally). */
export function groupOutfits(rows: Outfit[]): OutfitGroup[] {
  const groups = new Map<string, OutfitGroup>()
  for (const row of rows) {
    const group = groups.get(row.character_context_id) ?? { character_context_id: row.character_context_id, character_name: row.character_name, outfits: [] }
    group.outfits.push(row)
    groups.set(row.character_context_id, group)
  }
  const byName = (a: string, b: string) => a.localeCompare(b, undefined, { sensitivity: 'base' })
  return [...groups.values()]
    .map(group => ({ ...group, outfits: [...group.outfits].sort((a, b) => byName(a.name, b.name)) }))
    .sort((a, b) => byName(a.character_name, b.character_name))
}

export function originLabel(origin: Outfit['origin']): string {
  return origin === 'director' ? 'Sequence director' : origin === 'seed' ? 'Imported' : 'You'
}

/** The bible's outfit entries of one character, in the order the bible lists them. */
export function outfitsOf(outfits: BibleOutfit[] | undefined, character: string): BibleOutfit[] {
  return (outfits ?? []).filter(item => item.character === character)
}

/** A short line for a long description, for lists. */
export function preview(text: string, limit = 120): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length <= limit ? flat : `${flat.slice(0, limit).replace(/\s+\S*$/, '')}…`
}
