// Run Multiple (backend/app/batch_runs.py, docs/autopilot/HARNESS.md D37): the shapes of a stage preview and a run, and
// pure helpers for the dialog. Tested in frontend/tests/runMultipleView.test.mjs.
import type { DialogTab, RunSection } from './cutRoute'

export type StageKey = 'prepare' | 'prepare_cuts' | 'evaluate'
export type StepState = 'will_run' | 'done' | 'off' | 'skipped' | 'pending' | 'starting' | 'running' | 'failed' | 'cancelled'
export type CutRef = { cut: number; cut_id: string }
export type HeldCut = CutRef & { step: string; reasons: string[] }
export type StepProgress = { total: number; finished: number; failed: number }
export type PlanStep = {
  step: string; label: string; state: StepState; detail?: string; summary?: string; skip?: string
  cuts?: CutRef[]; held?: HeldCut[]; stats?: Record<string, unknown>; progress?: StepProgress
  errors?: (CutRef & { error: string })[]; start_error?: string
}
export type Preview = {
  stage: StageKey; available: boolean; reason: string; steps: PlanStep[]
  prepared: { id: string; started_at: string; finished_at: string | null } | null
  cuts?: number; held?: HeldCut[]
}
export type InboxCount = Record<string, { human: number; claude: number }>
export type BatchRun = {
  id: string; episode_id: string; stage: StageKey; status: 'running' | 'finished' | 'failed' | 'cancelled'
  options: Record<string, unknown>; steps: PlanStep[]; summary: { inbox?: InboxCount; failures?: number; held?: HeldCut[] }
  error: string | null; created_at: string; updated_at: string; finished_at: string | null
}

export const STAGES: StageKey[] = ['prepare', 'prepare_cuts', 'evaluate']
export const STAGE_LABEL: Record<StageKey, string> = { prepare: 'Prepare', prepare_cuts: 'Update changed cuts', evaluate: 'Evaluate' }
export const STAGE_INTRO: Record<StageKey, string> = {
  prepare: 'Runs once per episode: hard cuts, the missed-cut check, transcription and speakers, line splits, voices, character boxes and target frames. It asks you only what it cannot settle, once, in the Inbox.',
  prepare_cuts: 'For cuts you split, merged or re-boxed after Prepare: their character boxes, recognition and target frames.',
  evaluate: 'Visual analysis, performance beats and the consistency review, only for cuts that are missing them or are out of date. Cuts already evaluated and current are left alone.',
}
export const STATUS_LABEL: Record<BatchRun['status'], string> = { running: 'Running', finished: 'Finished', failed: 'Failed', cancelled: 'Cancelled' }

/** How the dialog marks a step: done, running, waiting, left out or failed. */
export function stepTone(state: StepState): 'done' | 'active' | 'waiting' | 'quiet' | 'failed' {
  if (state === 'done') return 'done'
  if (state === 'running' || state === 'starting') return 'active'
  if (state === 'will_run' || state === 'pending') return 'waiting'
  if (state === 'failed' || state === 'cancelled') return 'failed'
  return 'quiet'
}

/** One line under a step's label: what it found once over, else what it will do (or why it does nothing). */
export function stepText(step: PlanStep): string {
  if (step.summary) return step.summary
  if (step.state === 'cancelled') return 'Cancelled'
  return step.detail ?? ''
}

/** "3, 5, 9 and 4 more" */
export function cutNumbers(numbers: number[], limit = 12): string {
  const sorted = [...new Set(numbers)].sort((a, b) => a - b)
  return sorted.slice(0, limit).join(', ') + (sorted.length > limit ? ` and ${sorted.length - limit} more` : '')
}

export function progressText(progress: StepProgress | undefined): string {
  if (!progress?.total) return ''
  return `${progress.finished}/${progress.total} jobs` + (progress.failed ? `, ${progress.failed} failed` : '')
}

export function plural(count: number, word: string, many = `${word}s`) {
  return `${count} ${count === 1 ? word : many}`
}

export type ReasonGroup = { key: string; label: string; tab: DialogTab; section?: RunSection }
const GROUPS: (ReasonGroup & { test: (reason: string) => boolean })[] = [
  { key: 'boxes', label: 'Character boxes not detected (run Update changed cuts)', tab: 'run', section: 'characters', test: reason => reason.startsWith('first-frame boxes not detected') },
  { key: 'person', label: 'Unknown person to name (Inbox)', tab: 'run', section: 'characters', test: reason => reason === 'inbox: an unknown person' },
  { key: 'names', label: 'Box names to confirm', tab: 'run', section: 'characters', test: reason => reason === 'inbox: box names' || /^box \d+ /.test(reason) },
  { key: 'targets', label: 'Target frames to check', tab: 'run', section: 'targets', test: reason => reason === 'inbox: target frames' || reason.startsWith('target frame') || reason.includes('unnamed person') },
  { key: 'voice', label: 'Voice question (Inbox)', tab: 'results', test: reason => reason === 'inbox: an unknown voice' },
]
const EARLIER: ReasonGroup = { key: 'earlier', label: 'Waits for an earlier step', tab: 'results' }

/** The kind of person's action a held cut waits for, from its first reason (backend cut_readiness reasons). */
export function reasonGroup(reasons: string[]): ReasonGroup {
  for (const reason of reasons) {
    const found = GROUPS.find(group => group.test(reason))
    if (found) return { key: found.key, label: found.label, tab: found.tab, ...(found.section ? { section: found.section } : {}) }
  }
  return EARLIER
}

/** The held cuts grouped by what they wait for, in the order a person settles them. */
export function groupHeld(held: HeldCut[]): (ReasonGroup & { cuts: HeldCut[] })[] {
  const order = [...GROUPS.map(group => group.key), EARLIER.key]
  const groups = new Map<string, ReasonGroup & { cuts: HeldCut[] }>()
  for (const item of held) {
    const group = reasonGroup(item.reasons)
    if (!groups.has(group.key)) groups.set(group.key, { ...group, cuts: [] })
    groups.get(group.key)!.cuts.push(item)
  }
  return [...groups.values()].sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))
}

/** What an Evaluate preview will do: cuts per step and in all, and how many wait for the person. */
export function evaluateCounts(preview: Pick<Preview, 'steps' | 'cuts' | 'held'>) {
  const per = Object.fromEntries(preview.steps.map(step => [step.step, step.state === 'will_run' ? step.cuts?.length ?? 0 : 0]))
  return { visual: per.visual ?? 0, beats: per.beats ?? 0, consistency: per.consistency ?? 0, cuts: preview.cuts ?? 0, held: preview.held?.length ?? 0 }
}

/** The label and state of a stage's start button, from its preview. */
export function startButton(stage: StageKey, preview: Preview | null): { label: string; disabled: boolean; why: string } {
  if (!preview) return { label: `Run ${STAGE_LABEL[stage]}`, disabled: true, why: 'Loading…' }
  if (!preview.available) return { label: `Run ${STAGE_LABEL[stage]}`, disabled: true, why: preview.reason }
  if (stage === 'evaluate') {
    const cuts = preview.cuts ?? 0
    return cuts ? { label: `Evaluate ${plural(cuts, 'cut')}`, disabled: false, why: '' }
      : { label: 'Evaluate', disabled: true, why: (preview.held?.length ? 'Every cut that needs it waits for you' : 'Every cut is evaluated and current') }
  }
  const work = preview.steps.filter(step => step.state === 'will_run')
  if (!work.length) return { label: `Run ${STAGE_LABEL[stage]}`, disabled: stage !== 'prepare', why: stage === 'prepare' ? '' : 'No cut changed since Prepare' }
  if (stage === 'prepare_cuts') {
    const cuts = new Set(work.flatMap(step => (step.cuts ?? []).map(item => item.cut)))
    return { label: `Update ${plural(cuts.size, 'cut')}`, disabled: false, why: '' }
  }
  return { label: 'Run Prepare', disabled: false, why: '' }
}

/** Cut numbers a finished step points the person to, each with where it opens: what the stats name as split, not
 *  done, to check or with questions. */
export function stepLinks(step: PlanStep): { label: string; cuts: number[]; tab: DialogTab; section?: RunSection }[] {
  const stats = step.stats ?? {}
  const list = (key: string) => (Array.isArray(stats[key]) ? (stats[key] as unknown[]).filter((value): value is number => typeof value === 'number') : [])
  const links: { label: string; cuts: number[]; tab: DialogTab; section?: RunSection }[] = []
  const add = (label: string, cuts: number[], tab: DialogTab, section?: RunSection) => { if (cuts.length) links.push({ label, cuts, tab, ...(section ? { section } : {}) }) }
  if (step.step === 'missed_cuts') add('Split', list('cuts'), 'run')
  if (step.step === 'boxes') add('Not detected', list('not_detected'), 'run', 'characters')
  if (step.step === 'target_frames') { add('To check', list('to_check'), 'run', 'targets'); add('Not detected', list('not_detected'), 'run', 'targets') }
  if (['visual', 'beats', 'consistency'].includes(step.step)) add('Not done', list('not_done'), 'results')
  if (step.step === 'consistency') add('Questions for you', list('questions_for_you'), 'results')
  return links
}

/** "2 voices for you · 1 box label for Claude" from a run's inbox counts. */
export function inboxText(inbox: InboxCount | undefined): string {
  const words: Record<string, [string, string]> = {
    voice: ['voice', 'voices'], box_label: ['box label', 'box labels'], new_person: ['unknown person', 'unknown people'],
    target_frame: ['target-frame cut', 'target-frame cuts'], failed_run: ['failed run', 'failed runs'],
  }
  const parts: string[] = []
  for (const [kind, counts] of Object.entries(inbox ?? {}).sort()) {
    const [one, many] = words[kind] ?? [kind.replace(/_/g, ' '), `${kind.replace(/_/g, ' ')}s`]
    if (counts.human) parts.push(`${plural(counts.human, one, many)} for you`)
    if (counts.claude) parts.push(`${plural(counts.claude, one, many)} for Claude`)
  }
  return parts.join(' · ')
}

export function humanInbox(inbox: InboxCount | undefined) {
  return Object.values(inbox ?? {}).reduce((total, counts) => total + (counts.human || 0), 0)
}

// The floating Waiting-for-you list (WaitingForYou.tsx): where it sits on the screen.
export type FloatPlace = { x: number; y: number }
type Size = { width: number; height: number }

/** The panel's top-left kept inside the window by `margin`; a panel taller or wider than the window keeps its header on screen. */
export function clampFloat(place: FloatPlace, size: Size, viewport: Size, margin = 8): FloatPlace {
  const x = Math.max(margin, Math.min(place.x, viewport.width - size.width - margin))
  const y = Math.max(margin, Math.min(place.y, viewport.height - size.height - margin))
  return { x: Math.round(x), y: Math.round(y) }
}

/** The stage the dialog opens on: a running run's, else Prepare until it ran, else Evaluate. */
export function defaultStage(runs: BatchRun[], prepared: boolean, wanted?: StageKey): StageKey {
  const running = runs.find(run => run.status === 'running')
  if (running) return running.stage
  if (wanted) return wanted
  return prepared ? 'evaluate' : 'prepare'
}

/** The run the stage panel shows: the stage's running one, else the one that just finished while the dialog was open. */
export function shownRun(runs: BatchRun[], stage: StageKey, followed: string | null): BatchRun | undefined {
  return runs.find(run => run.stage === stage && run.status === 'running') ?? (followed ? runs.find(run => run.id === followed && run.stage === stage) : undefined)
}
