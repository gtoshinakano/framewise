import { useCallback, useEffect, useSyncExternalStore } from 'react'
import { api, type AnalysisConfig, type AnalysisContext, type ContextItem } from './api'
import { NEW_CARD } from './unregisteredView'

// Small stale-while-revalidate caches for the Run settings tab. The dialog remounts its content for every
// cut, so without these each Next/Previous click would wait for the same requests again.

export type CharacterReference = { id: string; character_name: string; run_id: string; position: string; box: [number, number, number, number]; image_size: [number, number] }

const configs = new Map<string, AnalysisConfig>()
const configRequests = new Map<string, Promise<AnalysisConfig>>()
const projectIds = new Map<string, string>()
const characters = new Map<string, ContextItem[]>()
const characterRequests = new Map<string, Promise<ContextItem[]>>()
const characterListeners = new Map<string, Set<() => void>>()
const emptyCharacters: ContextItem[] = []
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

export function loadCharacters(projectId: string, refresh = false): Promise<ContextItem[]> {
  const pending = characterRequests.get(projectId)
  if (pending && !refresh) return pending
  const request = api<ContextItem[]>(`/projects/${projectId}/context`).then(rows => {
    const items = rows.filter(item => item.kind === 'character')
    if (characterRequests.get(projectId) === request) {
      updateCharacters(projectId, items)
    }
    return items
  }).finally(() => { if (characterRequests.get(projectId) === request) characterRequests.delete(projectId) })
  characterRequests.set(projectId, request)
  return request
}

function updateCharacters(projectId: string, items: ContextItem[]) {
  characters.set(projectId, items)
  characterListeners.get(projectId)?.forEach(listener => listener())
}

export function rememberCharacter(item: ContextItem) {
  const rows = characters.get(item.project_id) ?? emptyCharacters
  updateCharacters(item.project_id, [...rows.filter(row => row.id !== item.id), item])
}

export function useCharacters(projectId: string) {
  const subscribe = useCallback((listener: () => void) => {
    const listeners = characterListeners.get(projectId) ?? new Set()
    listeners.add(listener)
    characterListeners.set(projectId, listeners)
    return () => {
      listeners.delete(listener)
      if (!listeners.size) characterListeners.delete(projectId)
    }
  }, [projectId])
  const snapshot = useCallback(() => characters.get(projectId) ?? emptyCharacters, [projectId])
  useEffect(() => { if (projectId) void loadCharacters(projectId).catch(() => {}) }, [projectId])
  return useSyncExternalStore(subscribe, snapshot)
}

/** A new Unregistered card (the lowest free number); refreshes the cached characters. */
export async function createCard(projectId: string): Promise<{ card: ContextItem; characters: ContextItem[] }> {
  const card = await api<ContextItem>(`/projects/${projectId}/unregistered-characters`, { method: 'POST' })
  return { card, characters: await loadCharacters(projectId, true) }
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
