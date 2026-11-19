// Pure helpers of the Dialogue tab's vertical timeline: no React and no fetch, so the node tests can import them.
// Times are seconds from the start of the sequence unless a name says episode time.
import type { CompiledLine, Decider, DialogueView, SequenceBoundary, SequenceCut, SequenceShot, ShotPoint } from './sequenceApi'

/** A shot keeps at least this much of its own cut's picture (the whole cut when it is shorter); `sequence_timeline.MIN_OWN_SECONDS`. */
export const MIN_OWN_SECONDS = 0.2
/** How far the director may move a boundary; a placed one may go farther, with a warning. */
export const SHIFT_SECONDS = 3
/** Snapping to a line puts the boundary this far into the silence before or after it (`sequence_timeline.EDGE_PAD`). */
export const EDGE_PAD = 0.06

/** Colours for the lines, far enough apart that neighbours never look alike; the timeline and the line rows share them. */
export const LINE_COLOURS = ['#e0a96d', '#6fb3e0', '#9ad37a', '#e07a9a', '#b89ae8', '#5fd0c0', '#e0d36d', '#e88a5f', '#7a9ae0', '#c8e07a']

export function lineColour(index: number) {
  const count = LINE_COLOURS.length
  return LINE_COLOURS[((index % count) + count) % count]
}

/** Where a shot boundary sits: placed by you, where the current plan has it, or on the hard cut until the director decides. */
export type MarkerSource = 'you' | 'dialogue' | 'plan' | 'cut'
export type Marker = {
  /** The hard cut's id (K1…), the cut before it and the cut after it. */
  id: string; after_cut_id: string; before_cut_id: string; index: number
  /** The hard cut and the boundary, in sequence seconds. */
  hard: number; t: number
  source: MarkerSource; by: Decider | null
  /** The number of the shot that starts here: the plan's, else the cut's place in the sequence. */
  shot: number
  speech: boolean; reason?: string
}

/** One marker per hard cut, in order, for the draft `points` (episode seconds). */
export function boundaryMarkers(boundaries: SequenceBoundary[], shots: SequenceShot[], points: ShotPoint[], origin: number, suggestions: DialogueView['shot_starts'] = []): Marker[] {
  return boundaries.map((boundary, index) => {
    const point = points.find(item => item.after_cut_id === boundary.after_cut_id)
    const suggested = suggestions.find(item => item.after_cut_id === boundary.after_cut_id)
    const planned = shots.find(shot => shot.cut_id === boundary.before_cut_id && (!shot.part || shot.part[0] === 1))
    const base = { id: boundary.id, after_cut_id: boundary.after_cut_id, before_cut_id: boundary.before_cut_id, index, hard: boundary.t, shot: planned?.index ?? index + 2, speech: !!boundary.speech }
    if (point) return { ...base, t: round(point.time - origin), source: 'you', by: 'user' }
    // Handed back in the draft: the plan still has the old placement until it is saved, so show the hard cut instead.
    if (suggested) return { ...base, t: round(suggested.time - origin), source: 'dialogue', by: suggested.by, speech: suggested.speech, reason: suggested.reason }
    if (planned && !boundary.fixed) return { ...base, t: planned.t0, source: 'plan', by: boundary.by }
    return { ...base, t: boundary.t, source: 'cut', by: null }
  })
}

const round = (value: number) => Math.round(value * 1000) / 1000

/**
 * The range a boundary may be dragged in: the server's hard rules (`sequence_timeline.check_points`), with every
 * boundary you did not place on its hard cut. The shots stay in order, each keeps some of its own cut's picture,
 * and a shot of a cut that fits one take stays within it.
 */
export function limits(index: number, markers: Marker[], cuts: SequenceCut[], shotSeconds = 15) {
  const at = (marker: Marker | undefined, fallback: number) => !marker ? fallback : marker.source === 'you' ? marker.t : marker.hard
  const duration = cuts.length ? cuts[cuts.length - 1].t1 : 0
  const previous = at(markers[index - 1], 0), next = at(markers[index + 1], duration)
  const left = cuts[index], right = cuts[index + 1]
  if (!left || !right) return { min: 0, max: duration }
  const need = (cut: SequenceCut) => Math.min(MIN_OWN_SECONDS, cut.t1 - cut.t0)
  const fits = (cut: SequenceCut) => cut.t1 - cut.t0 <= shotSeconds + 1e-6
  let min = Math.max(previous, left.t0) + need(left)
  let max = Math.min(next, right.t1) - need(right)
  if (fits(right)) min = Math.max(min, next - shotSeconds)
  if (fits(left)) max = Math.min(max, previous + shotSeconds)
  return { min: round(Math.max(0, min)), max: round(Math.min(duration, max)) }
}

export function clamp(value: number, range: { min: number; max: number }) {
  return round(Math.min(range.max, Math.max(range.min, value)))
}

/** Times a dragged boundary snaps to: the hard cuts, just before and after each line, and the word edges inside lines. */
export function snapTargets(lines: Pick<CompiledLine, 't0' | 't1' | 'parts'>[], cuts: SequenceCut[]) {
  const targets = new Set<number>()
  for (const cut of cuts.slice(1)) targets.add(cut.t0)
  for (const line of lines) {
    targets.add(round(Math.max(0, line.t0 - EDGE_PAD)))
    targets.add(round(line.t1 + EDGE_PAD))
    for (const part of line.parts.slice(1)) targets.add(round(part.t0))
  }
  return [...targets].sort((a, b) => a - b)
}

/** The nearest target within `tolerance` seconds, else the time itself. */
export function snap(t: number, targets: number[], tolerance: number) {
  let best = t, distance = tolerance
  for (const target of targets) {
    const gap = Math.abs(target - t)
    if (gap <= distance) { best = target; distance = gap }
  }
  return best
}

/**
 * The frame the shot that starts at a boundary opens on, as the director will get it (`sequence_timeline.expand`):
 * its cut's first frame when the shot starts on or before its cut, else the picture at that moment of the cut.
 */
export function openingFrame(marker: Pick<Marker, 't'>, cut: SequenceCut, origin: number) {
  const into = round(marker.t - cut.t0)
  if (into <= 0.002) {
    const held = into < -0.002 ? `; it holds ${(-into).toFixed(2)} s before the cut's own motion starts` : ''
    return { url: `/api/cuts/${cut.id}/frame/first?v=${cut.start_time}`, caption: `First frame of cut ${cut.number}${held}`, derived: false }
  }
  return { url: `/api/cuts/${cut.id}/boundary-preview-frame?time=${(origin + marker.t).toFixed(3)}`,
    caption: `${into.toFixed(2)} s into cut ${cut.number}: the director describes this moment from the motion, it was never analysed`, derived: true }
}

/** The line being spoken at `t` (not at its very edges), if any. */
export function lineCrossing<T extends Pick<CompiledLine, 't0' | 't1'>>(lines: T[], t: number) {
  return lines.find(line => line.t0 + 0.05 < t && t < line.t1 - 0.05)
}

/** The words of a line on each side of `t`, by the midpoint of each word. */
export function wordsAround(line: Pick<CompiledLine, 'parts' | 'text'>, t: number) {
  // Like `sequence_timeline.join_text`: a space only between two Latin words.
  const join = (parts: CompiledLine['parts']) => parts.reduce((text, part) => text && /[A-Za-z0-9]$/.test(text) && /^[A-Za-z0-9]/.test(part.text) ? `${text} ${part.text}` : text + part.text, '')
  const before = line.parts.filter(part => (part.t0 + part.t1) / 2 < t), after = line.parts.filter(part => (part.t0 + part.t1) / 2 >= t)
  return { before: join(before), after: join(after) }
}

/** What to look at before keeping a boundary where it is: speech it divides, a long move, a cut it mostly hides. None of these is refused. */
export function pointWarnings(marker: Marker, cuts: SequenceCut[], lines: CompiledLine[]) {
  const warnings: string[] = []
  const line = lineCrossing(lines, marker.t)
  if (line) {
    const { before, after } = wordsAround(line, marker.t)
    warnings.push(`Divides ${line.id}: “${before}” ends the shot before, “${after}” opens shot ${marker.shot}.`)
  }
  const shift = marker.t - marker.hard
  if (Math.abs(shift) > SHIFT_SECONDS + 0.002) warnings.push(`${Math.abs(shift).toFixed(2)} s ${shift < 0 ? 'before' : 'after'} the hard cut; the director never moves a boundary more than ${SHIFT_SECONDS} s.`)
  const hidden = shift < 0 ? cuts[marker.index] : cuts[marker.index + 1]
  if (hidden && Math.abs(shift) > (hidden.t1 - hidden.t0) / 2 + 0.002) {
    warnings.push(`Hides ${Math.abs(shift).toFixed(2)} s of cut ${hidden.number}'s ${(hidden.t1 - hidden.t0).toFixed(2)} s picture.`)
  }
  return warnings
}

/** Overlapping lines go to separate sub-columns: the track of each line, in input order. */
export function lineTracks(lines: Pick<CompiledLine, 't0' | 't1'>[]) {
  const order = lines.map((_, index) => index).sort((a, b) => lines[a].t0 - lines[b].t0)
  const ends: number[] = [], tracks = new Array<number>(lines.length).fill(0)
  for (const index of order) {
    let track = ends.findIndex(end => end <= lines[index].t0 + 0.01)
    if (track < 0) { track = ends.length; ends.push(lines[index].t1) } else ends[track] = lines[index].t1
    tracks[index] = track
  }
  return tracks
}

/** Label positions that do not overlap: each label sits at its wanted y or just below the label above it. In input order. */
export function placeLabels(wanted: number[], height: number) {
  const order = wanted.map((_, index) => index).sort((a, b) => wanted[a] - wanted[b])
  const placed = new Array<number>(wanted.length).fill(0)
  let floor = -Infinity
  for (const index of order) {
    placed[index] = Math.max(wanted[index], floor)
    floor = placed[index] + height
  }
  return placed
}

/** Set one boundary to a time (placing it), or hand it back (`time` null). */
export function placePoint(points: ShotPoint[], afterCutId: string, time: number | null): ShotPoint[] {
  const rest = points.filter(item => item.after_cut_id !== afterCutId)
  return time === null ? rest : [...rest, { after_cut_id: afterCutId, time: Math.round(time * 1e6) / 1e6 }]
}

export function samePoints(a: ShotPoint[], b: ShotPoint[]) {
  const key = (points: ShotPoint[]) => JSON.stringify([...points].sort((x, y) => x.after_cut_id.localeCompare(y.after_cut_id)).map(item => [item.after_cut_id, Math.round(item.time * 1000)]))
  return key(a) === key(b)
}
