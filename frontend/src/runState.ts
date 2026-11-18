import type { AnalysisResult, Cut, Job } from './api'

export type StepKey = 'initial_frame' | 'keyframe_analysis' | 'beats' | 'consistency' | 'summary'
export type StepState = 'none' | 'queued' | 'running' | 'done' | 'failed'
export type StageRow = { id: string; job_id: string; stage_key: string; label: string; status: string; elapsed_seconds: number; error: string | null }

export const STEP_LABELS: Record<StepKey, string> = { initial_frame: 'Initial frame', keyframe_analysis: 'Keyframes', beats: 'Beats', consistency: 'Consistency', summary: 'Cut summary' }
// Keyframes replaced the video step on 2026-11-18 and the last frame is no longer analyzed. Older video and last-frame results stay in the result list.
export const VISUAL_STEPS = ['initial_frame', 'keyframe_analysis'] as const
const SUMMARY_TITLE = 'Cut summary'

type ResultLike = Pick<AnalysisResult, 'title' | 'superseded_by' | 'settings'>

function resultKey(result: ResultLike): StepKey | null {
  if (result.superseded_by) return null
  const type = result.settings?.structured_type
  if (type === 'initial_frame' || type === 'keyframe_analysis') return type
  if (type === 'performance_beats') return 'beats'
  if (type === 'consistency_summary') return 'consistency'
  return result.title === SUMMARY_TITLE ? 'summary' : null
}

const activeFor = (job: Job, cutId: string) => ['queued', 'running'].includes(job.status) && job.cuts?.some(item => item.id === cutId && !item.completed)

/**
 * State of each step for one cut: finished results win unless the step is being redone now;
 * a run in progress shows queued/running; otherwise the latest failed attempt shows as failed.
 */
export function cutStepStates(results: ResultLike[], stages: StageRow[], jobs: Job[], cutId: string): Record<StepKey, StepState> {
  const state: Record<StepKey, StepState> = { initial_frame: 'none', keyframe_analysis: 'none', beats: 'none', consistency: 'none', summary: 'none' }
  for (const result of results) { const key = resultKey(result); if (key) state[key] = 'done' }
  // Stage attempts are returned oldest first, so the last row per stage is the newest attempt.
  const latest = new Map<string, StageRow>()
  for (const row of stages) latest.set(row.stage_key, row)
  for (const key of VISUAL_STEPS) {
    const row = latest.get(key)
    if (state[key] !== 'done' && row && (row.status === 'failed' || row.status === 'interrupted')) state[key] = 'failed'
  }
  for (const job of jobs) {
    if (!activeFor(job, cutId)) continue
    if (job.pipeline === 'analyze') {
      for (const key of job.stages?.length ? job.stages.filter((item): item is typeof VISUAL_STEPS[number] => (VISUAL_STEPS as readonly string[]).includes(item)) : VISUAL_STEPS) {
        const rows = stages.filter(row => row.job_id === job.id && row.stage_key === key)
        const row = rows[rows.length - 1]
        if (row?.status === 'running') state[key] = 'running'
        else if (row?.status === 'done' || row?.status === 'complete') state[key] = 'done'
        else if (state[key] !== 'running') state[key] = 'queued'
      }
    } else if (job.pipeline === 'performance_beats') state.beats = job.status === 'running' ? 'running' : 'queued'
    else if (job.pipeline === 'consistency_summary') state.consistency = job.status === 'running' ? 'running' : 'queued'
    else if (job.pipeline === 'regen') state.summary = job.status === 'running' ? 'running' : 'queued'
  }
  return state
}

/** Active run that touches this cut, if any, to show its current stage and elapsed time. */
export function activeRunFor(jobs: Job[], cutId: string): Job | undefined {
  const active = jobs.filter(job => ['analyze', 'performance_beats', 'regen', 'consistency_summary'].includes(job.pipeline) && activeFor(job, cutId))
  return active.find(job => job.status === 'running') ?? active[0]
}

export type Warning = { level: 'error' | 'warn' | 'info'; text: string }
export type CharacterSummary = { boxes: number; autoUnverified: number; unknown: number; stale: boolean; saving?: boolean }

export function preflight(input: { mediaBusy: boolean; enabled: boolean; characters: CharacterSummary | null }): Warning[] {
  const out: Warning[] = []
  if (!input.enabled) out.push({ level: 'error', text: 'Analysis is disabled for this cut. Enable it in the cut editor to queue it.' })
  if (input.mediaBusy) out.push({ level: 'warn', text: 'Media preparation or detection is running. Analysis can be queued when it finishes.' })
  const people = input.characters
  if (people?.stale) out.push({ level: 'warn', text: 'Cut timing changed since characters were detected. Detect again before analysis.' })
  if (people && people.autoUnverified > 0) out.push({ level: 'warn', text: `${people.autoUnverified} automatic ${people.autoUnverified === 1 ? 'label is' : 'labels are'} unverified. Queuing treats named boxes as verified, so confirm or correct them first.` })
  if (people && people.unknown > 0) out.push({ level: 'info', text: `${people.unknown} ${people.unknown === 1 ? 'box has' : 'boxes have'} no character name.` })
  return out
}

/** How many cuts a "from this cut" and an "all cuts" batch can reach. */
export function batchCounts(cuts: Pick<Cut, 'active' | 'cut_index' | 'analysis_enabled'>[], cut: Pick<Cut, 'cut_index'>) {
  const active = cuts.filter(item => item.active)
  return { from: active.filter(item => item.cut_index >= cut.cut_index && item.analysis_enabled).length, all: active.filter(item => item.analysis_enabled).length }
}

export function stateTone(state: StepState): 'green' | 'purple' | 'amber' | 'red' | 'grey' {
  return state === 'done' ? 'green' : state === 'running' ? 'purple' : state === 'queued' ? 'amber' : state === 'failed' ? 'red' : 'grey'
}
export const STATE_TEXT: Record<StepState, string> = { none: 'Not run', queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed' }

export type QueueNotice = { tone: 'success' | 'warning'; lines: string[] }

/** Turns the one-sentence-after-another batch message into separate lines, and flags it when something was held back. */
export function queueNotice(text: string): QueueNotice {
  const lines = text.split(/(?<=\.)\s+(?=[A-Z0-9])/).map(line => line.trim()).filter(Boolean)
  const heldBack = lines.some(line => /^Not queued/i.test(line)) || /\bqueued for 0 /i.test(text) || /^Queued 0 /i.test(text)
  return { tone: heldBack ? 'warning' : 'success', lines }
}
