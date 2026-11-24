/** What the Target frames tab knows about the people on a cut's evaluated frames, and the pure helpers that read it. */
export type Box = [number, number, number, number]
export type TargetSlot = { slot: number; origin: 'first_frame' | 'appeared'; character_context_id: string | null; character_name: string | null; label_source: 'inherited' | 'auto' | 'manual' | null; ignored?: boolean }
export type TargetBox = { id: string; frame_ordinal: number; slot: number; box: Box; source: 'auto' | 'manual' }
export type TargetCandidate = { id: string; frame_ordinal: number; tag: 'unmatched' | 'merge'; box: Box; source: 'tracker'; suggested_slot?: number | null; parts?: { box: Box; slot_hint: number | null }[] }
export type TargetReview = { score: number | null; flags: Record<string, unknown>; reviewed?: boolean }
/** `mirror`: the first frame as the user left it in Characters in this cut, copied rather than detected, so it is read only here. */
export type TargetFrameInfo = { ordinal: number; time: number; relative_time: number; image_size: [number, number]; mirror?: boolean; review?: TargetReview }
export type TargetRun = { id: string; selection_key: string; stale: boolean; stale_reason?: string | null; frames: TargetFrameInfo[]; slots: TargetSlot[]; detections: TargetBox[]; candidates?: TargetCandidate[]; unmatched_count?: number; cast_locked?: boolean; review_version?: number }
export type TargetJob = { id: string; status: string; stage: string; error: string | null }
export type TargetDetections = { evaluation: 'current' | 'missing'; selection_key: string | null; max_boxes: number; run: TargetRun | null; job: TargetJob | null; cast_locked?: boolean }

export const MAX_TARGET_BOXES = 3

export const boxesInFrame = (run: TargetRun, ordinal: number) => run.detections.filter(item => item.frame_ordinal === ordinal)
export const frameNeedsReview = (frame: TargetFrameInfo | undefined) => !!frame?.review && !frame.review.reviewed && Object.keys(frame.review.flags).length > 0

export const REVIEW_LABELS: Record<string, string> = {
  low_match: 'Low match', count_changed: 'Cast count changed', unmatched: 'Unmatched crop',
  suspected_merge: 'Possible merged box', jump: 'Position jump', duplicate_character: 'Duplicate character',
}

/** A frame holds at most three boxes: drawing another needs one removed first. */
export const canDraw = (run: TargetRun, ordinal: number, limit = MAX_TARGET_BOXES) => boxesInFrame(run, ordinal).length < limit

export const slotName = (slot: TargetSlot | undefined) => slot?.character_name || 'Unknown'

/** The slots that have a box somewhere and no character: what still needs a name. */
export function unknownSlots(run: TargetRun | null) {
  if (!run) return []
  const used = new Set(run.detections.map(item => item.slot))
  return run.slots.filter(slot => !slot.ignored && used.has(slot.slot) && !slot.character_context_id && slot.label_source !== 'manual')
}

/** In how many evaluated frames each slot has a box. */
export function framesPerSlot(run: TargetRun) {
  const counts = new Map<number, number>()
  for (const ordinal of new Set(run.detections.map(item => item.frame_ordinal))) {
    for (const slot of new Set(boxesInFrame(run, ordinal).map(item => item.slot))) counts.set(slot, (counts.get(slot) || 0) + 1)
  }
  return counts
}

export type SlotChip = { tone: 'blue' | 'amber' | 'green' | 'grey'; text: string; title: string }

/** Where a slot's name came from, in the words the first-frame panel already uses. */
export function slotChip(slot: TargetSlot): SlotChip {
  if (!slot.character_context_id && slot.label_source === 'manual') return { tone: 'grey', text: 'Left unnamed', title: 'You left this colour without a name; nothing asks about it again' }
  if (!slot.character_context_id) return { tone: 'grey', text: 'No name', title: 'Nobody is assigned to this colour yet' }
  if (slot.label_source === 'manual') return { tone: 'green', text: 'Confirmed', title: 'You set this name' }
  if (slot.label_source === 'auto') return { tone: 'amber', text: 'Auto · unverified', title: 'Matched with your labelled boxes, not yet confirmed by you' }
  return { tone: 'blue', text: 'From first frame', title: 'Copied from the first-frame box of the same colour' }
}

export const isJobActive = (job: TargetJob | null | undefined) => job?.status === 'queued' || job?.status === 'running'

/**
 * The colours a colour can be merged into when it is really another person: every other colour, unless it has a
 * box on the mirrored first frame, where the user already told the people apart.
 */
export function mergeTargets(run: TargetRun, slot: number) {
  const mirrored = new Set(run.frames.filter(frame => frame.mirror).map(frame => frame.ordinal))
  if (run.detections.some(item => item.slot === slot && mirrored.has(item.frame_ordinal))) return []
  return run.slots.filter(item => item.slot !== slot && !item.ignored)
}
