import type { Job } from './api'

/**
 * The jobs the Analysis dialog shows for one cut: the episode's own (bounded, always current) list plus the cut's
 * full history read from `/api/cuts/{id}/jobs`. A job in both is the episode's copy, because it is the fresher one.
 * The result is newest first; jobs created in the same second keep the episode list's order.
 */
export function mergeJobs(episodeJobs: Job[], cutJobs: Job[] | null | undefined): Job[] {
  if (!cutJobs?.length) return episodeJobs
  const known = new Set(episodeJobs.map(job => job.id))
  const older = cutJobs.filter(job => !known.has(job.id))
  if (!older.length) return episodeJobs
  return [...episodeJobs, ...older].sort((a, b) => (a.created_at < b.created_at ? 1 : a.created_at > b.created_at ? -1 : 0))
}

/** What decides whether the cut's job history must be read again: which jobs of the episode list touch it, and how they stand. */
export function cutJobsKey(episodeJobs: Job[], cutId: string): string {
  return episodeJobs.filter(job => job.cuts?.some(item => item.id === cutId)).map(job => `${job.id}:${job.status}`).join(',')
}
