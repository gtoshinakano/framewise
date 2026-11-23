import type { Cut, TranscriptSegment } from './api'

/** A diarization voice of the episode that still has lines with no character on them (and not left as they are by the person). */
export type UnknownVoice = { label: string; lines: number; inCut: number }

/**
 * Voices that still have unassigned lines, most lines first.
 * `label` is the speaker label, empty for lines the diarizer left unlabelled;
 * `lines` counts them across the episode, `inCut` only the ones overlapping this cut.
 */
export function unknownVoices(segments: TranscriptSegment[], cut: Pick<Cut, 'start_time' | 'end_time'>): UnknownVoice[] {
  const groups = new Map<string, UnknownVoice>()
  for (const segment of segments) {
    // A line the person left unknown is never asked about again (backend cut_readiness.speaker_open).
    if (segment.deleted || segment.superseded_by || segment.character_context_id || segment.speaker_source === 'left_unknown') continue
    const label = segment.speaker_label || ''
    let group = groups.get(label)
    if (!group) { group = { label, lines: 0, inCut: 0 }; groups.set(label, group) }
    group.lines += 1
    if (segment.start_time < cut.end_time && segment.end_time > cut.start_time) group.inCut += 1
  }
  return [...groups.values()].sort((a, b) => b.lines - a.lines || a.label.localeCompare(b.label))
}
