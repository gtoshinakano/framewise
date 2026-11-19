import { useEpisodeRevision } from './episodeEvents'
import { useEffect, useState } from 'react'
import { api } from './api'
import type { StageRow } from './runState'

/**
 * Per-attempt stage rows for one cut. Fetched when the Run tab is visible and whenever the jobs change.
 */
export function useCutStages(cutId: string, visible: boolean, refreshKey: string) {
  const [stages, setStages] = useState<StageRow[]>([])
  const revision = useEpisodeRevision(['jobs'], visible)
  useEffect(() => { setStages([]) }, [cutId])
  useEffect(() => {
    if (!visible) return
    let active = true
    const load = () => { void api<StageRow[]>(`/cuts/${cutId}/analysis-stages`).then(data => { if (active) setStages(previous => JSON.stringify(previous) === JSON.stringify(data) ? previous : data) }).catch(() => {}) }
    load()
    return () => { active = false }
  }, [cutId, visible, refreshKey, revision])
  return stages
}
