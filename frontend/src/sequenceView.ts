// Pure helpers of the sequence workspace: no React and no fetch, so the node tests can import them.
import type { ReviewIssue, SequenceCandidate, SequenceDetail, SequenceShot, StaleReason } from './sequenceApi'

/** Seconds as m:ss.s, the clock every time in the workspace uses. */
export function clock(seconds: number) {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  const tenths = Math.round(safe * 10)
  return `${Math.floor(tenths / 600)}:${String(Math.floor(tenths % 600 / 10)).padStart(2, '0')}.${tenths % 10}`
}

export type Tone = 'good' | 'busy' | 'warn' | 'bad' | 'idle'
export type ShotStatus = { key: 'writing' | 'queued' | 'missing' | 'errors' | 'stale' | 'ready'; label: string; tone: Tone }

/** What a shot needs next, most urgent first: a running job, a missing prompt, format errors, stale inputs. */
export function shotStatus(shot: SequenceShot): ShotStatus {
  if (shot.job === 'running') return { key: 'writing', label: 'Writing', tone: 'busy' }
  if (shot.job === 'queued') return { key: 'queued', label: 'Queued', tone: 'busy' }
  if (!shot.result) return { key: 'missing', label: 'Not written', tone: 'idle' }
  if (!shot.result.check?.ok) return { key: 'errors', label: 'Format errors', tone: 'bad' }
  if (shot.stale.length) return { key: 'stale', label: 'Stale', tone: 'warn' }
  return { key: 'ready', label: 'Ready', tone: 'good' }
}

const STALE: Record<StaleReason, string> = {
  plan: 'its plan entry or its spoken lines changed',
  bible: 'the bible changed (style, characters, voices or its scene)',
  evidence: 'the analysis or your answers for its cuts changed',
}

/** Why a written shot no longer matches what it was written from, as one sentence. */
export function staleText(reasons: StaleReason[]) {
  if (!reasons.length) return ''
  const parts = reasons.map(reason => STALE[reason] || reason)
  return `Written before ${parts.length > 1 ? `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}` : parts[0]}. Write it again to bring it up to date.`
}

export type Progress = { shots: number; written: number; ready: number; needWriting: number; running: number; queued: number; step: string | null }

export function progress(detail: SequenceDetail): Progress {
  const statuses = detail.shots.map(shotStatus)
  const active = detail.jobs.filter(job => job.status === 'queued' || job.status === 'running')
  const current = active.find(job => job.status === 'running') || active[0]
  return {
    shots: detail.shots.length, written: detail.shots.filter(shot => shot.result).length, ready: statuses.filter(status => status.key === 'ready').length,
    needWriting: statuses.filter(status => status.key === 'missing' || status.key === 'errors' || status.key === 'stale').length,
    running: active.filter(job => job.status === 'running').length, queued: active.filter(job => job.status === 'queued').length,
    step: current ? current.step : null,
  }
}

export type Stage = 'recreate' | 'analyse' | 'direct' | 'directing' | 'redirect' | 'write' | 'writing' | 'ready'

/** Where the sequence stands in its two-click flow: direct (bible and plan), review the plan, write the shots. */
export function stage(detail: SequenceDetail): { key: Stage; hint: string } {
  const counts = progress(detail)
  if (detail.sequence.cuts_missing) return { key: 'recreate', hint: 'The first or last cut of this sequence was archived. Create the sequence again.' }
  if (counts.step === 'bible' || counts.step === 'plan') return { key: 'directing', hint: counts.step === 'bible' ? 'Writing the sequence bible…' : 'Planning the shots…' }
  if (counts.running + counts.queued) return { key: 'writing', hint: counts.step === 'review' ? 'Reviewing continuity…' : `Writing shot prompts: ${counts.ready}/${counts.shots} ready.` }
  if (detail.missing.length) return { key: 'analyse', hint: `Cut${detail.missing.length > 1 ? 's' : ''} ${detail.missing.join(', ')} need a first-frame and a video analysis before the sequence can be directed.` }
  if (detail.sequence.cuts_changed || detail.plan_problems.length) return { key: 'redirect', hint: 'The cuts changed since this sequence was directed. Direct it again.' }
  if (!detail.bible || !detail.plan) return { key: 'direct', hint: 'Direct the sequence: the director writes the bible and the shot plan, then waits for you.' }
  if (counts.needWriting) return { key: 'write', hint: counts.written ? `${counts.needWriting} shot${counts.needWriting > 1 ? 's' : ''} need writing.` : 'Review the bible and the plan, then write the shot prompts.' }
  return { key: 'ready', hint: 'Every shot has a current prompt. The sequence is ready to download.' }
}

/** Places where a planned shot can be split by hand: boundaries strictly inside it, at least half a second from its ends. */
export function splitChoices(shot: SequenceShot, candidates: SequenceCandidate[]) {
  return candidates.filter(candidate => candidate.t >= shot.t0 + 0.5 && candidate.t <= shot.t1 - 0.5)
    .map(candidate => ({ time: candidate.time, label: `${clock(candidate.t)} · ${candidate.kind === 'cut' ? 'hard cut' : candidate.kind === 'pause' ? (candidate.pause ? `pause ${candidate.pause} s` : 'pause') : candidate.kind === 'event' ? 'action change' : 'quiet point'}` }))
}

/** Whether a shot and the next one fit one take. */
export function canMerge(shots: SequenceShot[], index: number, limit: number) {
  const next = shots[index + 1]
  return !!next && next.t1 - shots[index].t0 <= limit + 0.001
}

export function issuesFor(issues: ReviewIssue[] | undefined, shot: SequenceShot) {
  return (issues || []).filter(issue => issue.shot_key === shot.key)
}

/** The review's suggestions for one shot as the instruction of a rewrite. */
export function issueInstruction(issues: ReviewIssue[]) {
  return issues.map(issue => (issue.suggestion || issue.problem).trim()).filter(Boolean).join(' ')
}

const SECTIONS = ['subject_definitions', 'summary', 'retention_analysis', 'detailed_description', 'overall_soundscape', 'non_diegetic_music']
const TITLES: Record<string, string> = { subject_definitions: 'Subjects and references', summary: 'Summary', retention_analysis: 'What stays', detailed_description: 'Shot and beats', overall_soundscape: 'Sound', non_diegetic_music: 'Music' }

/** An H3 prompt split into its six sections for the Read view. */
export function promptSections(prompt: string) {
  const sections: { name: string; title: string; lines: string[] }[] = []
  for (const line of prompt.split('\n')) {
    const header = /^([a-z_]+):\s*$/.exec(line)
    if (header && SECTIONS.includes(header[1])) sections.push({ name: header[1], title: TITLES[header[1]], lines: [] })
    else if (sections.length && line.trim()) sections[sections.length - 1].lines.push(line)
  }
  return sections
}

/** Length of an inclusive range of cuts, or null when the numbers do not name existing cuts in order. */
export function rangeSeconds(cuts: { cut_index: number; start_time: number; end_time: number }[], from: number, to: number) {
  const first = cuts.find(cut => cut.cut_index === from - 1), last = cuts.find(cut => cut.cut_index === to - 1)
  if (!first || !last || to < from) return null
  return last.end_time - first.start_time
}
