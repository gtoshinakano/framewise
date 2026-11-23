import { api, type AnalysisConfig, type AnalysisContext, type ContextItem } from './api'
import { NEW_CARD } from './unregisteredView'

// Small stale-while-revalidate caches for the Run settings tab. The dialog remounts its content for every
// cut, so without these each Next/Previous click would wait for the same requests again.

export type CharacterReference = { id: string; character_name: string; run_id: string; position: string; box: [number, number, number, number]; image_size: [number, number] }

const configs = new Map<string, AnalysisConfig>()
const configRequests = new Map<string, Promise<AnalysisConfig>>()
const projectIds = new Map<string, string>()
const characters = new Map<string, ContextItem[]>()
const references = new Map<string, CharacterReference[]>()

export const cachedConfig = (cutId: string) => configs.get(cutId)
export const storeConfig = (cutId: string, config: AnalysisConfig) => { configs.set(cutId, config) }

/** One request per cut at a time, shared by the prefetch and the tab that opens a moment later. */
export function loadConfig(cutId: string): Promise<AnalysisConfig> {
  const pending = configRequests.get(cutId)
  if (pending) return pending
  const request = api<AnalysisConfig>(`/cuts/${cutId}/analysis-config`).then(value => { configs.set(cutId, value); return value }).finally(() => configRequests.delete(cutId))
  configRequests.set(cutId, request)
  return request
}

export function prefetchConfigs(cutIds: (string | undefined)[]) {
  for (const id of cutIds) if (id && !configs.has(id) && !configRequests.has(id)) void loadConfig(id).catch(() => {})
}

export const cachedProjectId = (episodeId: string) => projectIds.get(episodeId)

/** The project id comes from the heavy analysis-context response, so ask once per episode. */
export async function loadProjectId(episodeId: string, cutId: string): Promise<string> {
  const known = projectIds.get(episodeId)
  if (known) return known
  const context = await api<AnalysisContext>(`/cuts/${cutId}/analysis-context`)
  projectIds.set(episodeId, context.project.id)
  return context.project.id
}

export const cachedCharacters = (projectId: string) => characters.get(projectId)
export const cachedReferences = (projectId: string) => references.get(projectId)

export async function loadCharacters(projectId: string): Promise<ContextItem[]> {
  const items = (await api<ContextItem[]>(`/projects/${projectId}/context`)).filter(item => item.kind === 'character')
  characters.set(projectId, items)
  return items
}

/** A new Unregistered card (the lowest free number); refreshes the cached characters. */
export async function createCard(projectId: string): Promise<{ card: ContextItem; characters: ContextItem[] }> {
  const card = await api<ContextItem>(`/projects/${projectId}/unregistered-characters`, { method: 'POST' })
  return { card, characters: await loadCharacters(projectId) }
}

/** The character id a select's value stands for: a new card is created first. */
export async function chosenCharacter(projectId: string, value: string, onCharacters?: (items: ContextItem[]) => void): Promise<string | null> {
  if (value !== NEW_CARD) return value || null
  const { card, characters } = await createCard(projectId)
  onCharacters?.(characters)
  return card.id
}

export async function loadReferences(projectId: string): Promise<CharacterReference[]> {
  const items = await api<CharacterReference[]>(`/projects/${projectId}/character-references`)
  references.set(projectId, items)
  return items
}

export const invalidateReferences = (projectId: string) => { references.delete(projectId) }
