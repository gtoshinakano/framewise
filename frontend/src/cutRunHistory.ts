import type { Job } from './api'

// A batch can finish some cuts before it fails on another. The stage contains
// the cut number from the run snapshot, not necessarily the current timeline.
export function cutRunOutcome(job: Job, cutId: string) {
  const cut = job.cuts.find(item => item.id === cutId)
  if (!cut) return null
  const stageCut = /\bcut\s+(\d+)/i.exec(job.stage)
  const isThisCut = job.cuts.length === 1 || (stageCut !== null && cut.cut_index !== null && Number(stageCut[1]) === cut.cut_index + 1)
  const stageIsSpecific = !!job.stage && !/^(failed|complete|queued|starting|resuming|cancelled)$/i.test(job.stage)
  if (cut.completed || job.status === 'done') return { status: 'Succeeded', step: job.pipeline === 'regen' ? 'Summary generation completed' : 'Analysis completed', error: null }
  if (job.stage === 'Cancelled') return { status: 'Cancelled', step: job.started_at ? 'Cancelled during processing' : 'Cancelled before processing', error: null }
  if (job.status === 'queued') return { status: 'Queued', step: 'Waiting to start', error: null }
  if (job.status === 'running') return { status: isThisCut ? 'Running' : 'Waiting', step: isThisCut ? job.stage : stageCut ? 'Another cut is being processed' : 'The active cut has not been recorded', error: null }
  if (job.status === 'failed') {
    if (isThisCut) return { status: 'Failed', step: stageIsSpecific ? job.stage.replace(/^Failed\s*·\s*/i, '') : 'Failure step was not recorded', error: job.error }
    return { status: 'Not completed', step: stageCut ? 'The run stopped on another cut' : 'The cut that failed was not recorded', error: null }
  }
  return { status: job.status, step: job.stage || 'Step not recorded', error: null }
}
