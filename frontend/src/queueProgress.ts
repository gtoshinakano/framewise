import type { Job } from './api'

// Keep finished jobs in the counter until the active queue empties. This also
// lets individual cut jobs contribute to the same progress as a batch job.
export function analysisQueueProgress(jobs: Job[], session: Set<string>) {
  const active = jobs.filter(job => ['analyze', 'regen', 'consistency_summary', 'character_detect'].includes(job.pipeline) && ['queued', 'running'].includes(job.status))
  if (!active.length) {
    session.clear()
    return { completed: 0, total: 0 }
  }
  for (const job of active) session.add(job.id)
  return jobs.filter(job => session.has(job.id)).reduce((counts, job) => {
    const completed = job.cuts.filter(cut => cut.completed).length
    const total = Math.max(1, job.cuts.length)
    if (job.status === 'failed') {
      // A failed batch stops at the first incomplete cut; later cuts leave the queue.
      if (job.stage !== 'Cancelled') {
        const finished = Math.min(total, completed + 1)
        counts.completed += finished; counts.total += finished
      }
    } else {
      counts.total += total
      counts.completed += job.status === 'done' ? total : completed
    }
    return counts
  }, { completed: 0, total: 0 })
}
