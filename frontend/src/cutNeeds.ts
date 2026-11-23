import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { useEpisodeEvents } from './episodeEvents'
import { EMPTY_NEEDS, toNeedsMap, type CutNeeds, type NeedsMap } from './cutNeedsView'

/** The episode's per-cut needs, reloaded when data or stale results change and (at most every few seconds) when jobs move. */
export function useCutNeeds(episodeId: string | null): NeedsMap {
  const [needs, setNeeds] = useState<NeedsMap>(EMPTY_NEEDS)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const load = useCallback(() => {
    if (!episodeId) return
    api<{ cuts: Record<string, CutNeeds> }>(`/episodes/${episodeId}/cut-needs`).then(payload => setNeeds(toNeedsMap(payload))).catch(() => { /* filters fall back to the cut flags */ })
  }, [episodeId])
  useEffect(() => { setNeeds(EMPTY_NEEDS); load() }, [load])
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  useEpisodeEvents(change => {
    if (!(change.data || change.stale || change.jobs) || timer.current) return
    timer.current = setTimeout(() => { timer.current = null; load() }, change.data || change.stale ? 400 : 5000)
  }, episodeId)
  return needs
}
