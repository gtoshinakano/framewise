// Pure helpers of the sequence workspace: no React and no fetch, so the node tests can import them.
import type { DiffSegment } from './assistantApi'
import type { BoundaryChoice, Decider, Delivery, FixedVoice, ReviewIssue, SequenceBoundary, SequenceDetail, SequenceShot, ShotEdge, ShotResult, ShotRevision, ShotVoice, StaleReason } from './sequenceApi'

/** Seconds as m:ss.s, the clock every time in the workspace uses. */
export function clock(seconds: number) {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  const tenths = Math.round(safe * 10)
  return `${Math.floor(tenths / 600)}:${String(Math.floor(tenths % 600 / 10)).padStart(2, '0')}.${tenths % 10}`
}

/** Seconds as MM:SS.mmm, exactly as the H3 prompt writes its timecodes (`h3_prompt.fmt`). */
export function timecode(seconds: number) {
  const millis = Math.round(Math.max(0, Number.isFinite(seconds) ? seconds : 0) * 1000)
  const minutes = Math.floor(millis / 60000), rest = millis % 60000
  return `${String(minutes).padStart(2, '0')}:${String(Math.floor(rest / 1000)).padStart(2, '0')}.${String(rest % 1000).padStart(3, '0')}`
}

// The shape `h3_prompt.fmt` writes: two-digit minutes, two-digit seconds, milliseconds.
// A sentence may end right after one ("from 00:00.076 to 00:02.239."), so only a digit or ".digit" around it disqualifies it.
const TIMECODE = /(?<![\d:]|\d\.)(\d{2}):(\d{2}\.\d{3})(?!\d|\.\d)/g
const SPOKEN_AT = /\bAt (\d{2}):(\d{2}\.\d{3})\b/g

/** A prompt line split around its timecodes; `t` is the shot second a timecode names, null for plain text. Joined, the parts give the line back. */
export function timecodeSegments(line: string): { text: string; t: number | null }[] {
  const parts: { text: string; t: number | null }[] = []
  let last = 0
  for (const match of line.matchAll(TIMECODE)) {
    if (match.index > last) parts.push({ text: line.slice(last, match.index), t: null })
    parts.push({ text: match[0], t: Number(match[1]) * 60 + Number(match[2]) })
    last = match.index + match[0].length
  }
  if (last < line.length || !parts.length) parts.push({ text: line.slice(last), t: null })
  return parts
}

/** When each spoken line starts in a shot, from the prompt's `At MM:SS.mmm` sentences. */
export function spokenTimes(prompt: string | null | undefined) {
  return [...(prompt || '').matchAll(SPOKEN_AT)].map(match => Number(match[1]) * 60 + Number(match[2]))
}

/** Seconds between labelled ticks on a timeline of this length. */
export function rulerStep(duration: number) {
  return duration <= 20 ? 1 : duration <= 75 ? 5 : duration <= 180 ? 10 : 30
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
  plan: 'its plan entry, its spoken lines or your voice settings changed',
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

export type Stage = 'recreate' | 'compiling' | 'review' | 'analyse' | 'direct' | 'directing' | 'redirect' | 'write' | 'writing' | 'ready'

/** Where the sequence stands in its two-click flow: direct (bible and plan), review the plan, write the shots. */
export function stage(detail: SequenceDetail): { key: Stage; hint: string } {
  const counts = progress(detail)
  if (detail.sequence.cuts_missing) return { key: 'recreate', hint: 'The first or last cut of this sequence was archived. Create the sequence again.' }
  if (counts.step === 'compile') return { key: 'compiling', hint: 'Qwen is compiling the dialogue…' }
  const dialogue = detail.compilation
  const added = detail.sequence.extension
  if ((!detail.bible || added) && dialogue?.has_speech && dialogue.state !== 'confirmed' && dialogue.state !== 'running') return { key: 'review', hint: dialogue.state === 'none' ? 'Compile the dialogue first: Qwen matches the transcript with the subtitles.' : added ? `${added.cuts.length > 1 ? `Cuts ${added.cuts.join(', ')} were` : `Cut ${added.cuts[0]} was`} added: review the new lines and save the dialogue. The direction is then redone.` : 'Review the dialogue, then save it. The director starts from it.' }
  if (counts.step === 'bible' || counts.step === 'plan') return { key: 'directing', hint: counts.step === 'bible' ? 'Writing the sequence bible…' : 'Planning the shots…' }
  if (counts.running + counts.queued) return { key: 'writing', hint: counts.step === 'review' ? 'Reviewing continuity…' : `Writing shot prompts: ${counts.ready}/${counts.shots} ready.` }
  if (detail.missing.length) return { key: 'analyse', hint: `Cut${detail.missing.length > 1 ? 's' : ''} ${detail.missing.join(', ')} need a first-frame and a video analysis before the sequence can be directed.` }
  if (detail.sequence.cuts_changed || detail.plan_problems.length) return { key: 'redirect', hint: 'The cuts changed since this sequence was directed. Direct it again.' }
  if (!detail.bible || !detail.plan) return { key: 'direct', hint: 'Direct the sequence: the director writes the bible and the shot plan, then waits for you.' }
  if (counts.needWriting) return { key: 'write', hint: counts.written ? `${counts.needWriting} shot${counts.needWriting > 1 ? 's' : ''} need writing.` : 'Review the bible and the plan, then write the shot prompts.' }
  return { key: 'ready', hint: 'Every shot has a current prompt. The sequence is ready to download.' }
}

function phrase(keeps: string[] | undefined) {
  if (!keeps?.length) return 'the phrase'
  return keeps.length === 1 ? keeps[0] : `${keeps[0]}–${keeps[keeps.length - 1]}`
}

const signed = (seconds: number) => `${seconds < 0 ? '−' : '+'}${Math.abs(seconds).toFixed(2)} s`

/** The choices at a hard cut inside speech, as the boundary picker lists them; options that are not possible are left out. */
export function boundaryChoices(boundary: SequenceBoundary) {
  const lines = boundary.speech?.lines.join(' and ') ?? ''
  const keeps = phrase(boundary.speech?.keeps)
  const labels: Record<BoundaryChoice, (shift: number) => string> = {
    keep_cut: () => `On the hard cut · divides ${lines}`,
    before_line: shift => `Before ${keeps} (${signed(shift)})`,
    after_line: shift => `After ${keeps} (${signed(shift)})`,
  }
  return (['keep_cut', 'before_line', 'after_line'] as BoundaryChoice[])
    .filter(choice => boundary.options[choice])
    .map(choice => ({ choice, label: labels[choice](boundary.options[choice]!.shift) }))
}

export function deciderLabel(by: Decider | null | undefined) {
  return by === 'director' ? 'director' : by === 'app' ? 'app' : by === 'user' ? 'you' : null
}

/** Short tags for how a shot meets its neighbours: borrowed audio before or after its cut, and speech divided at an edge. */
export function edgeTags(shot: Pick<SequenceShot, 'opens' | 'closes' | 'part'>) {
  const tags: { label: string; title: string }[] = []
  if (shot.part) tags.push({ label: `part ${shot.part[0]}/${shot.part[1]}`, title: 'Its cut is longer than one take, so the app divides it into consecutive parts of one take' })
  const describe = (edge: ShotEdge, side: 'opens' | 'closes') => {
    if (edge.how === 'before_line' || edge.how === 'after_line') {
      const shift = edge.shift ?? 0
      const borrows = (side === 'opens') === (edge.how === 'before_line')
      const placed = edge.placed ? ' (you)' : ''
      tags.push({ label: `${side === 'opens' ? 'starts' : 'ends'} ${signed(shift)}${placed}`, title: edge.placed
        ? `You placed this boundary on the Dialogue timeline: ${side === 'opens' ? 'it starts' : 'it ends'} ${Math.abs(shift).toFixed(2)} s ${edge.how === 'before_line' ? 'before' : 'after'} the hard cut${borrows ? `, playing that much of the neighbouring cut's audio over its own picture` : ''}`
        : borrows
          ? `${side === 'opens' ? 'Starts before' : 'Runs past the end of'} its cut to keep ${phrase(edge.keeps)} whole: it plays ${Math.abs(shift).toFixed(2)} s of the neighbouring cut's audio over its own picture`
          : `${side === 'opens' ? 'Starts after the start of' : 'Ends before the end of'} its cut: the neighbouring shot carries ${phrase(edge.keeps)} whole` })
    }
    if (edge.split) tags.push({ label: `${side === 'opens' ? 'opens inside' : 'divides'} ${edge.split.lines.join('+')}`, title: `The speech is divided here: “${edge.split.before}” | “${edge.split.after}”${edge.reason ? ` — ${edge.reason}` : ''}` })
  }
  describe(shot.opens, 'opens')
  describe(shot.closes, 'closes')
  return tags
}

export const DELIVERIES: { value: Delivery; label: string; hint: string }[] = [
  { value: 'on_screen', label: 'On screen', hint: 'In the picture with the mouth visible: the lips move in sync with the words' },
  { value: 'off_screen', label: 'Off screen', hint: 'Not in the picture: the words are heard and no visible lips move for them' },
  { value: 'inner_voice', label: 'Inner voice', hint: 'In the picture but the mouth does not move: a thought, or the face is turned away' },
  { value: 'narrative', label: 'Narrative', hint: 'Too far from the camera for lip sync (or not visible): the words simply play over the picture' },
]

export function deliveryLabel(delivery: Delivery | null | undefined) {
  return DELIVERIES.find(item => item.value === delivery)?.label ?? 'Director decides'
}

/** One range of a speaker's voice as the Voices panel draws it. */
export type VoiceRange = { from: number; to: number; delivery: Delivery; mine: boolean; subject: string | null }

/** A speaker's ranges in a shot: what the written direction says, with the ranges you set laid over it (they win). */
export function speakerRanges(written: ShotVoice[] | undefined, fixed: FixedVoice[], speaker: string, duration: number): VoiceRange[] {
  const own = (written || []).filter(item => item.speaker === speaker)
  const mine = fixed.filter(item => item.speaker === speaker)
  const points = [...new Set([0, duration, ...own.map(item => item.from), ...mine.flatMap(item => [item.from, Math.min(item.to, duration)])])].sort((a, b) => a - b)
  const ranges: VoiceRange[] = []
  for (let index = 0; index + 1 < points.length; index++) {
    const from = points[index], to = points[index + 1], middle = (from + to) / 2
    if (to - from < 0.001) continue
    const user = [...mine].reverse().find(item => item.from <= middle && middle < item.to)
    const base = [...own].reverse().find(item => item.from <= middle)
    const piece = user ? { delivery: user.delivery, mine: true, subject: base?.subject ?? null } : base ? { delivery: base.delivery, mine: !!(base.fixed_by_user || base.fixed), subject: base.subject } : null
    if (!piece) continue
    const last = ranges[ranges.length - 1]
    if (last && last.delivery === piece.delivery && last.mine === piece.mine && Math.abs(last.to - from) < 0.001) last.to = to
    else ranges.push({ from, to, ...piece })
  }
  return ranges
}

/** The segments to save when the speaker changes to ``delivery`` from second ``at`` (0 sets the whole shot). */
export function changeAt(ranges: VoiceRange[], at: number, delivery: Delivery): { from: number; delivery: Delivery }[] {
  const start = Math.max(0, Math.round(at * 10) / 10)
  const kept = ranges.filter(range => range.from < start - 0.05).map(range => ({ from: range.from, delivery: range.delivery }))
  const segments = [...kept, { from: start, delivery }]
  // Ranges after the new one are replaced by it until the next range you set yourself.
  const later = ranges.find(range => range.from > start + 0.05 && range.mine)
  if (later) segments.push(...ranges.filter(range => range.from >= later.from).map(range => ({ from: range.from, delivery: range.delivery })))
  const joined: { from: number; delivery: Delivery }[] = []
  for (const segment of segments) if (!joined.length || joined[joined.length - 1].delivery !== segment.delivery) joined.push(segment)
  if (joined.length) joined[0] = { ...joined[0], from: 0 }
  return joined
}

/** The speakers whose voice ranges the user set for a shot. */
export function hasOwnVoices(shot: SequenceShot, speaker: string) {
  return (shot.fixed_voices || []).some(item => item.speaker === speaker)
}

/** Whether a shot and the next one fit one take. */
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

/** Why a version of a shot's prompt was written, in a few words. */
export function revisionLabel(revision: Pick<ShotRevision, 'number' | 'origin' | 'instruction' | 'trigger' | 'instruction_source'>) {
  if (revision.origin === 'manual_edit') return 'Edited by you'
  if (revision.origin === 'assistant_edit') return 'Edited with the assistant'
  const verb = revision.number === 1 ? 'Written' : 'Rewritten'
  if (revision.instruction) return revision.instruction_source === 'review' ? `${verb} with the review's suggestion` : `${verb} with your instruction`
  if (revision.trigger === 'voices') return `${verb} after you changed the voices`
  return `${verb} by the director`
}

/** What caused the current version of a shot, for the line above its prompt; null when nothing but the director wrote it. */
export function versionNote(result: Pick<ShotResult, 'origin' | 'parent_id' | 'instruction' | 'instruction_source' | 'trigger' | 'assistant_summary'>, limit = 240) {
  const label = revisionLabel({ number: result.parent_id ? 2 : 1, origin: result.origin, instruction: result.instruction, instruction_source: result.instruction_source, trigger: result.trigger ?? null })
  const clip = (text: string) => text.length > limit ? `${text.slice(0, limit).trimEnd()}…` : text
  if (result.origin === 'assistant_edit') return result.assistant_summary ? `${label}: ${clip(result.assistant_summary)}` : label
  if (result.origin !== 'model') return null
  if (result.instruction) return `${label}: “${clip(result.instruction)}”`
  return result.trigger === 'voices' ? label : null
}

/** How many words a change adds and removes. */
export function changeSize(diff: DiffSegment[] | null | undefined) {
  const words = (text: string) => text.split(/\s+/).filter(Boolean).length
  const size = { added: 0, removed: 0 }
  for (const part of diff || []) {
    if (part.op === 'insert') size.added += words(part.text)
    else if (part.op === 'delete') size.removed += words(part.text)
  }
  return size
}

/** Length of an inclusive range of cuts, or null when the numbers do not name existing cuts in order. */
export function rangeSeconds(cuts: { cut_index: number; start_time: number; end_time: number }[], from: number, to: number) {
  const first = cuts.find(cut => cut.cut_index === from - 1), last = cuts.find(cut => cut.cut_index === to - 1)
  if (!first || !last || to < from) return null
  return last.end_time - first.start_time
}
