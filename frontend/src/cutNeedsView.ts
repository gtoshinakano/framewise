// What each cut still needs (GET /api/episodes/{id}/cut-needs, backend/app/cut_needs.py): the person's actions, Claude's
// queue, the missing automatic steps and which agents applied something. Pure helpers, tested in frontend/tests/cutNeeds.test.mjs;
// the loading hook is cutNeeds.ts.
import type { Cut } from './api'
import type { DialogTab, RunSection } from './cutRoute'

// Missed cuts and lines across a hard cut are never a need, and a voice is one only while its question is open (HARNESS.md D37).
export type NeedKey = 'voice' | 'box_names' | 'target_frames' | 'decisions' | 'picture_check' | 'failed_run'
export type StepKey = 'boxes' | 'target' | 'first_frame' | 'keyframes' | 'beats' | 'consistency' | 'summary'
export type PersonNeed = { key: NeedKey; count: number; escalation_ids: string[] }
export type CutNeeds = { steps: StepKey[]; people: PersonNeed[]; claude: { kind: string; escalation_id: string }[]; answered_by: string[]; analysis_off: boolean; stale?: { title: string; reason: string }[] }
export type NeedsMap = ReadonlyMap<string, CutNeeds>

export const EMPTY_NEEDS: NeedsMap = new Map()

/** The order of an episode's flow (backend cut_needs.PEOPLE): a cut change first, the review answers late. */
export const NEED_ORDER: NeedKey[] = ['voice', 'box_names', 'target_frames', 'decisions', 'picture_check', 'failed_run']

export const NEED_LABEL: Record<NeedKey, string> = {
  voice: 'Unknown voice',
  box_names: 'Name the boxes',
  target_frames: 'Target frames to check',
  decisions: 'Review questions',
  picture_check: 'Picture disagrees',
  failed_run: 'Failed run',
}

export const STEP_LABEL: Record<StepKey, string> = {
  boxes: 'Character boxes',
  target: 'Target frames',
  first_frame: 'First frame',
  keyframes: 'Keyframes',
  beats: 'Performance beats',
  consistency: 'Consistency summary',
  summary: 'Cut summary',
}
const STEP_ORDER = Object.keys(STEP_LABEL) as StepKey[]

/** `data-need` anchors in the analysis dialog. */
export type NeedAnchor = 'decisions' | 'picture-check' | 'voices' | 'lines' | 'other-decisions' | 'characters' | 'targets'
export type NeedPlace = { tab: DialogTab; section?: RunSection; anchor: NeedAnchor }

/** Where in the analysis dialog a need is settled. Missed cuts, lines across cuts and failed runs have their card in Your answers. */
export function placeForNeed(key: NeedKey): NeedPlace {
  switch (key) {
    case 'box_names': return { tab: 'run', section: 'characters', anchor: 'characters' }
    case 'target_frames': return { tab: 'run', section: 'targets', anchor: 'targets' }
    case 'decisions': return { tab: 'results', anchor: 'decisions' }
    case 'picture_check': return { tab: 'results', anchor: 'picture-check' }
    case 'voice': return { tab: 'results', anchor: 'voices' }
    default: return { tab: 'results', anchor: 'other-decisions' }
  }
}

type Place = { tab: DialogTab; section?: RunSection }

/** The same place of the analysis dialog: the section counts only on the Run tab. */
export function samePlace(a: Place, b: Place) {
  return a.tab === b.tab && (a.tab !== 'run' || a.section === b.section)
}

/** Whether the analysis dialog, open at `current`, stays at `wanted` when it moves to a cut with these needs without being
 *  opened again: arriving moves it to the cut's first need, so only when that need is settled at `wanted` (or there is none). */
export function staysAt(needs: CutNeeds | undefined, focusKeys: ReadonlySet<NeedKey> | null, wanted: Place, current: Place) {
  if (!samePlace(wanted, current)) return false
  const need = firstNeed(needs, focusKeys)
  return !need || samePlace(placeForNeed(need.key), wanted)
}

export function hasNeed(needs: CutNeeds | undefined, key: NeedKey) {
  return !!needs?.people.some(item => item.key === key)
}

/** The first of the cut's needs (in flow order) among `keys`; with no keys, its first need of any kind. */
export function firstNeed(needs: CutNeeds | undefined, keys: ReadonlySet<NeedKey> | null = null): PersonNeed | undefined {
  if (!needs) return undefined
  return NEED_ORDER.map(key => needs.people.find(item => item.key === key)).find(item => item && (!keys?.size || keys.has(item.key)))
}

export function toNeedsMap(payload: { cuts: Record<string, CutNeeds> }): NeedsMap {
  return new Map(Object.entries(payload.cuts ?? {}))
}

const seconds = (cut: Cut) => Math.max(0, cut.end_time - cut.start_time)

/** `m:ss` of a length in seconds (rounded to the second). */
export function minutesSeconds(value: number) {
  const total = Math.round(Math.max(0, value))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

export type RangeTotal = { key: string; label: string; kind: 'person' | 'step' | 'claude'; cuts: number[] }
export type RangeRow = { cut: Cut; number: number; seconds: number; needs?: CutNeeds }
export type RangeSummary = { first: Cut; last: Cut; count: number; analysisOff: number; start: number; end: number; seconds: number; rows: RangeRow[]; totals: RangeTotal[] }

/** The cuts from index `from` to `to` (either order) with what each still needs, and per need the cut numbers that have it. */
export function rangeSummary(cuts: Cut[], needs: NeedsMap, from: number, to: number): RangeSummary | null {
  const low = Math.max(0, Math.min(from, to)), high = Math.min(cuts.length - 1, Math.max(from, to))
  if (high < low || !cuts.length) return null
  const rows = cuts.slice(low, high + 1).map(cut => ({ cut, number: cut.cut_index + 1, seconds: seconds(cut), needs: needs.get(cut.id) }))
  const totals: RangeTotal[] = []
  const add = (key: string, label: string, kind: RangeTotal['kind'], match: (row: RangeRow) => boolean) => {
    const numbers = rows.filter(match).map(row => row.number)
    if (numbers.length) totals.push({ key, label, kind, cuts: numbers })
  }
  for (const key of NEED_ORDER) add(key, NEED_LABEL[key], 'person', row => hasNeed(row.needs, key))
  add('claude', 'Waiting for Claude', 'claude', row => !!row.needs?.claude.length)
  for (const key of STEP_ORDER) add(key, `Needs ${STEP_LABEL[key].toLowerCase()}`, 'step', row => !!row.needs?.steps.includes(key))
  const first = rows[0].cut, last = rows[rows.length - 1].cut
  return { first, last, count: rows.length, analysisOff: rows.filter(row => !row.cut.analysis_enabled).length, start: first.start_time, end: last.end_time,
    seconds: last.end_time - first.start_time, rows, totals }
}

/** Who applied an answer, as the strips name them. */
export const TIER_NAME: Record<string, string> = { claude: 'Claude', local: 'The local model', rules: 'A rule', human: 'You' }

type DecisionLike = { id: string; kind: string; tier: string; created_at: string; undone_at: string | null; overridden_at: string | null; summary: string; after?: any; evidence?: any }
export type QuestionAnswer<T extends DecisionLike = DecisionLike> = { decision: T; option: string | null; text: string; note: string }

/** The latest agent answer (not undone) that wrote this review question's line, with the note it left. */
export function answerFor<T extends DecisionLike>(decisions: T[], questionId: string): QuestionAnswer<T> | null {
  const latest = decisions.filter(decision => decision.kind === 'review_answers' && !decision.undone_at)
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
  for (const decision of latest) {
    const line = (decision.after?.lines ?? []).find((item: { question?: string }) => item.question === questionId)
    if (line) return { decision, option: line.option ?? null, text: String(line.text ?? ''), note: String(decision.evidence?.note || line.why || '') }
  }
  return null
}
