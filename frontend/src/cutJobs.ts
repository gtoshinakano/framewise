import { useEffect, useMemo, useState } from 'react'
import { api, type Job } from './api'
import { cutJobsKey, mergeJobs } from './jobsView'

// A cut's job history, kept after the dialog moves on so coming back (or stepping to a neighbour that was read ahead)
// shows it at once while a fresh copy is requested. Finished jobs never change, so a stale copy only lacks newer jobs,
// and those are in the episode's own list.
const cache = new Map<string, Job[]>()
const pending = new Set<string>()

async function fetchCutJobs(cutId: string): Promise<Job[]> {
  const jobs = await api<Job[]>(`/cuts/${cutId}/jobs`)
  cache.set(cutId, jobs)
  return jobs
}

export function prefetchCutJobs(cutIds: (string | undefined)[]) {
  for (const id of cutIds) {
    if (!id || cache.has(id) || pending.has(id)) continue
    pending.add(id)
    void fetchCutJobs(id).catch(() => {}).finally(() => pending.delete(id))
  }
}

/**
 * The jobs for one cut's panels: the episode's list with the cut's older jobs added. `loaded` is false until the
 * history of this cut has been read once, so a panel can tell "no runs" from "not read yet".
 */
export function useCutJobs(cutId: string, episodeJobs: Job[]) {
  const [fetched, setFetched] = useState<Job[] | null>(() => cache.get(cutId) ?? null)
  const key = cutJobsKey(episodeJobs, cutId)
  useEffect(() => {
    let active = true
    void fetchCutJobs(cutId).then(jobs => { if (active) setFetched(jobs) }).catch(() => {})
    return () => { active = false }
  }, [cutId, key])
  const jobs = useMemo(() => mergeJobs(episodeJobs, fetched), [episodeJobs, fetched])
  return { jobs, loaded: fetched !== null }
}
