import type { Job } from './api'

export type AnalysisQueueState = 'queued' | 'running'

// Analysis jobs process their cuts in snapshot order and checkpoint each one.
export function analysisQueueStates(jobs: Job[]): Map<string, AnalysisQueueState> {
  const states = new Map<string, AnalysisQueueState>()
  for (const job of jobs) {
    if (!['analyze', 'regen'].includes(job.pipeline) || !['queued', 'running'].includes(job.status)) continue
    const pending = job.cuts.filter(cut => !cut.completed)
    pending.forEach((cut, index) => {
      const state = job.status === 'running' && index === 0 ? 'running' : 'queued'
      if (states.get(cut.id) !== 'running') states.set(cut.id, state)
    })
  }
  return states
}
