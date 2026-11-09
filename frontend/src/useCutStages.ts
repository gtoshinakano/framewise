import { useEffect, useState } from 'react'
import { api } from './api'
import type { StageRow } from './runState'

/**
 * Per-attempt stage rows for one cut. Fetched when the Run tab is visible and whenever the jobs change;
 * it polls only while a run is queued or running, so an idle tab costs nothing.
 */
export function useCutStages(cutId: string, polling: boolean, visible: boolean, refreshKey: string) {
  const [stages, setStages] = useState<StageRow[]>([])
  useEffect(() => { setStages([]) }, [cutId])
  useEffect(() => {
    if (!visible) return
    let active = true
    const load = () => { void api<StageRow[]>(`/cuts/${cutId}/analysis-stages`).then(data => { if (active) setStages(previous => JSON.stringify(previous) === JSON.stringify(data) ? previous : data) }).catch(() => {}) }
    load()
    const timer = polling ? setInterval(load, 4000) : undefined
    return () => { active = false; if (timer) clearInterval(timer) }
  }, [cutId, polling, visible, refreshKey])
  return stages
}
