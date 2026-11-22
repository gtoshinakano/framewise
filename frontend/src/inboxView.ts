// The autopilot inbox (docs/autopilot/HARNESS.md, milestone M7): what the agents asked, what they applied, what failed.
// Pure helpers only (tested in frontend/tests/inboxView.test.mjs); the drawer is InboxDrawer.tsx.

export type Reviewer = 'claude' | 'human'

export type Escalation = {
  id: string
  episode_id: string
  run_id: string | null
  kind: string
  group_key: string
  question: string
  status: 'open' | 'resolved' | 'dismissed'
  reviewer: Reviewer
  created_at: string
  updated_at: string
  cut_ids: string[]
  options: unknown[]
  proposal: any
  evidence: any
}

export type Decision = {
  id: string
  run_id: string | null
  cut_id: string | null
  kind: string
  tier: 'rules' | 'local' | 'claude' | 'human'
  target_table: string
  target_id: string
  confidence: number | null
  summary: string
  created_at: string
  undone_at: string | null
  overridden_at: string | null
}

export type AgentRun = {
  id: string
  agent: string
  purpose: string
  status: 'running' | 'finished' | 'undone'
  claude_items: number
  started_at: string
  finished_at: string | null
  summary: Record<string, unknown>
  decisions: { total: number; undone: number; overridden: number }
}

export type TriageItem = {
  job_id: string
  pipeline: string
  label: string
  cut_ids: string[]
  cuts: number[]
  error: string
  cause: string
  cause_text: string
  action: string
  fix: string
  attempts: number
  limit: number
}

export type InboxCounts = { episodes: { episode_id: string; title: string; project_title: string; for_person: number; for_claude: number }[] }

/** The order questions appear in: the order of an episode's flow (a cut change first, the review answers last). */
export const KIND_ORDER = ['failed_run', 'missed_cut', 'line_split', 'voice', 'new_person', 'box_label', 'target_frame', 'review_question']

export const KIND_LABEL: Record<string, string> = {
  failed_run: 'Failed runs',
  missed_cut: 'Missed hard cuts',
  line_split: 'Lines across a hard cut',
  voice: 'Unknown voices',
  new_person: 'Unknown people',
  box_label: 'Box names',
  target_frame: 'Target frames',
  review_question: 'Review questions',
}

export const DECISION_LABEL: Record<string, string> = {
  failed_runs: 'Failed runs fixed',
  missed_cuts: 'Cuts split',
  line_splits: 'Lines split or trimmed',
  voices: 'Voices assigned',
  box_names: 'Boxes named or unboxed',
  target_frames: 'Target frames settled',
  review_answers: 'Review answers',
  stale_reruns: 'Stale results re-run',
}

export const TIER_LABEL: Record<Decision['tier'], string> = { rules: 'Rule', local: 'Local model', claude: 'Claude', human: 'You' }

export function kindLabel(kind: string) {
  return KIND_LABEL[kind] ?? kind.replace(/_/g, ' ').replace(/^./, letter => letter.toUpperCase())
}

/** Open questions grouped by kind, in flow order (unknown kinds last, alphabetically). */
export function groupByKind(items: Escalation[]) {
  const groups = new Map<string, Escalation[]>()
  for (const item of items) groups.set(item.kind, [...(groups.get(item.kind) ?? []), item])
  const rank = (kind: string) => { const index = KIND_ORDER.indexOf(kind); return index < 0 ? KIND_ORDER.length : index }
  return [...groups.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
    .map(([kind, list]) => ({ kind, label: kindLabel(kind), items: list }))
}

/** The same items in the order the drawer shows them, for keyboard navigation. */
export function flatOrder(items: Escalation[]) {
  return groupByKind(items).flatMap(group => group.items)
}

export type Request = { path: string; body: unknown }
/** One answer the person can give with one click or one key. `input: 'title'` asks for a name first (a new character). */
export type Choice = {
  key: string
  label: string
  hint?: string
  tone?: 'primary' | 'danger'
  proposed?: boolean
  request?: Request
  close?: 'resolved' | 'dismissed'
  retry?: string[]
  dismiss?: string[]
  input?: 'title'
}

const post = (id: string, path: string, body: unknown): Request => ({ path: `/escalations/${id}/${path}`, body })

/** The answers to one question that need nothing but a click. Review questions are answered per question (reviewProposal). */
export function choicesFor(item: Escalation): Choice[] {
  const id = item.id
  const evidence = item.evidence ?? {}
  switch (item.kind) {
    case 'missed_cut':
      return [
        { key: 'split', label: 'Split here', tone: 'primary', proposed: item.proposal?.answer === 'split', request: post(id, 'missed-cut', { answer: 'split' }) },
        { key: 'no_cut', label: 'No cut', request: post(id, 'missed-cut', { answer: 'no_cut' }) },
      ]
    case 'line_split':
      return [
        { key: 'apply', label: 'Apply', tone: 'primary', proposed: true, request: post(id, 'line-split', { answer: 'apply' }) },
        { key: 'leave', label: 'Leave the line', request: post(id, 'line-split', { answer: 'leave' }) },
      ]
    case 'voice': {
      const proposed = item.proposal?.character_context_id
      const candidates: { id: string; name: string }[] = evidence.candidates ?? []
      return [
        ...candidates.map(candidate => ({ key: candidate.id, label: candidate.name, proposed: candidate.id === proposed, tone: candidate.id === proposed ? 'primary' as const : undefined,
          request: post(id, 'voice', { choice: 'character', character_context_id: candidate.id }) })),
        { key: 'new', label: 'New character', hint: 'Closes it: create the character and its voice sample, then run the resolver again', request: post(id, 'voice', { choice: 'new' }) },
        { key: 'leave', label: 'Leave unknown', request: post(id, 'voice', { choice: 'leave' }) },
      ]
    }
    case 'box_label': {
      const boxes: { id: string }[] = evidence.boxes ?? []
      const name = item.proposal?.name ?? 'this character'
      if (!item.proposal?.character_context_id) return [{ key: 'open', label: 'Name them in the cut', hint: 'No proposal: open the cut and pick a name' }]
      return [
        { key: 'confirm', label: `All are ${name}`, tone: 'primary', request: post(id, 'box-labels', { confirm: boxes.map(box => box.id) }) },
        { key: 'reject', label: `None is ${name}`, tone: 'danger', request: post(id, 'box-labels', { reject: boxes.map(box => box.id) }) },
      ]
    }
    case 'new_person': {
      const candidates: [string, string | null, number][] = item.proposal?.candidates ?? []
      return [
        ...candidates.filter(([, name]) => name).slice(0, 3).map(([characterId, name, score]) => ({ key: characterId, label: `${name}`, hint: `similarity ${score.toFixed(2)}`,
          request: post(id, 'person', { choice: 'character', character_context_id: characterId }) })),
        { key: 'new', label: 'New character…', tone: 'primary' as const, input: 'title' as const },
        { key: 'background', label: 'Background', hint: 'Unbox every box of this person', request: post(id, 'person', { choice: 'background' }) },
        { key: 'leave', label: 'Leave', hint: 'Keep the boxes unnamed and stop asking', request: post(id, 'person', { choice: 'leave' }) },
      ]
    }
    case 'target_frame': {
      const frames: { ordinal: number }[] = evidence.frames ?? []
      if (!frames.length) return [{ key: 'open', label: 'Name the colours', hint: 'Open the Target frames tab' }]
      return [{ key: 'ok', label: frames.length === 1 ? 'The frame is right' : `All ${frames.length} frames are right`, tone: 'primary',
        request: post(id, 'target-frames', { frames: Object.fromEntries(frames.map(frame => [String(frame.ordinal), 'ok'])) }) }]
    }
    case 'review_question': {
      const answers = reviewProposal(item)
      return answers.length ? [{ key: 'accept', label: `Take the ${answers.length} proposed answer${answers.length === 1 ? '' : 's'}`, tone: 'primary', proposed: true,
        request: post(id, 'review-answers', { answers }) }] : []
    }
    case 'failed_run': {
      const jobs: string[] = (evidence.jobs ?? []).map((job: { job_id: string }) => job.job_id)
      return [
        { key: 'retry', label: jobs.length > 1 ? `Retry ${jobs.length}` : 'Retry', tone: 'primary', retry: jobs },
        { key: 'dismiss', label: 'Dismiss', dismiss: jobs },
      ]
    }
    default:
      return [{ key: 'resolved', label: 'Done', close: 'resolved' }, { key: 'dismissed', label: 'Dismiss', close: 'dismissed' }]
  }
}

/** A review question as the drawer shows it. */
export type ReviewQuestion = { id: string; topic?: string; character?: string; question: string; options: { id: string; text: string }[]; recommended?: string; proposal?: string | null; why_asked?: string }

export function reviewQuestions(item: Escalation): ReviewQuestion[] {
  return (item.evidence?.questions ?? []) as ReviewQuestion[]
}

/** `[{question, option}]` for the questions that have a proposal (the agent's pick, else the reviewer's recommended option). */
export function reviewProposal(item: Escalation, picks: Record<string, string> = {}) {
  return reviewQuestions(item)
    .map(question => ({ question: question.id, option: picks[question.id] ?? (item.proposal?.[question.id] || question.proposal || question.recommended || null) }))
    .filter((pick): pick is { question: string; option: string } => !!pick.option)
}

/** A sheet image the question has, if any (relative to /api). */
export function sheetPath(item: Escalation): string | null {
  const evidence = item.evidence ?? {}
  if (item.kind === 'missed_cut' && evidence.strip) return String(evidence.strip).replace(/^\/api/, '')
  if (item.kind === 'review_question' && evidence.sheet) return String(evidence.sheet).replace(/^\/api/, '')
  if (item.kind === 'voice') return `/escalations/${item.id}/voice-sheet`
  if (item.kind === 'box_label' || item.kind === 'new_person') return `/escalations/${item.id}/box-sheet`
  if (item.kind === 'target_frame') return `/escalations/${item.id}/target-sheet`
  return null
}

/** Where in the editor a question is settled by hand. */
export function placeFor(kind: string): { tab: 'results' | 'run'; section?: 'characters' | 'targets' } {
  if (kind === 'box_label' || kind === 'new_person') return { tab: 'run', section: 'characters' }
  if (kind === 'target_frame') return { tab: 'run', section: 'targets' }
  return { tab: 'results' }
}

/** A few decisions per kind to spot-check: deterministic (by id) so the sample stays put while the list reloads. */
export function sampleDecisions(decisions: Decision[], perKind = 5) {
  const score = (id: string) => { let hash = 2166136261; for (const char of id) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619); return hash >>> 0 }
  const byKind = new Map<string, Decision[]>()
  for (const decision of decisions) if (!decision.undone_at) byKind.set(decision.kind, [...(byKind.get(decision.kind) ?? []), decision])
  return [...byKind.values()].flatMap(list => [...list].sort((a, b) => score(a.id) - score(b.id)).slice(0, perKind))
}

/** Applied decisions per kind and tier, not counting undone ones. */
export function decisionCounts(decisions: Decision[]) {
  const counts: Record<string, { total: number; overridden: number; tiers: Record<string, number> }> = {}
  for (const decision of decisions) {
    if (decision.undone_at) continue
    const entry = counts[decision.kind] ??= { total: 0, overridden: 0, tiers: {} }
    entry.total += 1
    if (decision.overridden_at) entry.overridden += 1
    entry.tiers[decision.tier] = (entry.tiers[decision.tier] ?? 0) + 1
  }
  return counts
}

/** Open questions in other episodes, for the header badge's title. */
export function elsewhereText(counts: InboxCounts | null, episodeId: string) {
  const others = (counts?.episodes ?? []).filter(item => item.episode_id !== episodeId && item.for_person > 0)
  if (!others.length) return ''
  return `Also waiting for you: ${others.map(item => `${item.title} (${item.for_person})`).join(', ')}`
}

export function mineCount(counts: InboxCounts | null, episodeId: string) {
  return counts?.episodes.find(item => item.episode_id === episodeId)?.for_person ?? 0
}
