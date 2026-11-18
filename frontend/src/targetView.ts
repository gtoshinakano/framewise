/** What the Target frames tab knows about the people on a cut's evaluated frames, and the pure helpers that read it. */
export type Box = [number, number, number, number]
export type TargetSlot = { slot: number; origin: 'first_frame' | 'appeared'; character_context_id: string | null; character_name: string | null; label_source: 'inherited' | 'auto' | 'manual' | null }
export type TargetBox = { id: string; frame_ordinal: number; slot: number; box: Box; source: 'auto' | 'manual' }
/** `mirror`: the first frame as the user left it in Characters in this cut, copied rather than detected, so it is read only here. */
export type TargetFrameInfo = { ordinal: number; time: number; relative_time: number; image_size: [number, number]; mirror?: boolean }
export type TargetRun = { id: string; selection_key: string; stale: boolean; frames: TargetFrameInfo[]; slots: TargetSlot[]; detections: TargetBox[] }
export type TargetJob = { id: string; status: string; stage: string; error: string | null }
export type TargetDetections = { evaluation: 'current' | 'missing'; selection_key: string | null; max_boxes: number; run: TargetRun | null; job: TargetJob | null }

export const MAX_TARGET_BOXES = 3

export const boxesInFrame = (run: TargetRun, ordinal: number) => run.detections.filter(item => item.frame_ordinal === ordinal)

/** A frame holds at most three boxes: drawing another needs one removed first. */
export const canDraw = (run: TargetRun, ordinal: number, limit = MAX_TARGET_BOXES) => boxesInFrame(run, ordinal).length < limit

export const slotName = (slot: TargetSlot | undefined) => slot?.character_name || 'Unknown'

/** The slots that have a box somewhere and no character: what still needs a name. */
export function unknownSlots(run: TargetRun | null) {
  if (!run) return []
  const used = new Set(run.detections.map(item => item.slot))
  return run.slots.filter(slot => used.has(slot.slot) && !slot.character_context_id)
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
  if (!slot.character_context_id) return { tone: 'grey', text: 'No name', title: 'Nobody is assigned to this colour yet' }
  if (slot.label_source === 'manual') return { tone: 'green', text: 'Confirmed', title: 'You set this name' }
  if (slot.label_source === 'auto') return { tone: 'amber', text: 'Auto · unverified', title: 'Matched with your labelled boxes, not yet confirmed by you' }
  return { tone: 'blue', text: 'From first frame', title: 'Copied from the first-frame box of the same colour' }
}

/** Why detection cannot be queued now, or '' when it can. */
export function detectReason(input: { evaluated: boolean; episodeBusy: boolean; pending: boolean; saving: boolean }) {
  if (!input.evaluated) return 'Evaluate this cut first'
  if (input.pending) return 'Detection is already queued for this cut'
  if (input.saving) return 'Wait for your edit to save'
  if (input.episodeBusy) return 'Another episode job is running'
  return ''
}

export const isJobActive = (job: TargetJob | null | undefined) => job?.status === 'queued' || job?.status === 'running'

export type TargetQueueResult = { queued: number; skipped: { not_evaluated: number[]; analysis_disabled: number[]; no_characters: number[]; up_to_date: number[]; already_queued: number[] } }

const cutList = (numbers: number[]) => numbers.slice(0, 8).join(', ') + (numbers.length > 8 ? ` and ${numbers.length - 8} more` : '')

/** What a batch queue did, and which cuts were left out and why (cut numbers as the editor shows them). */
export function targetQueueMessage(result: TargetQueueResult, firstCut: number): { tone: 'success' | 'warning'; lines: string[] } {
  const skipped = result.skipped
  const lines = [result.queued
    ? `Target-frame characters queued for ${result.queued} ${result.queued === 1 ? 'cut' : 'cuts'}${firstCut > 1 ? ` from cut ${firstCut}` : ''}.`
    : 'Nothing was queued.']
  const reasons: [number[], string][] = [
    [skipped.not_evaluated, 'Not evaluated yet, evaluate them first'], [skipped.up_to_date, 'Already detected'], [skipped.already_queued, 'Already queued'],
    [skipped.no_characters, 'Marked as having no characters'], [skipped.analysis_disabled, 'Analysis disabled'],
  ]
  for (const [numbers, text] of reasons) if (numbers.length) lines.push(`${text}: cut ${cutList(numbers)}.`)
  return { tone: result.queued ? 'success' : 'warning', lines }
}

/**
 * The colours a colour can be merged into when it is really another person: every other colour, unless it has a
 * box on the mirrored first frame, where the user already told the people apart.
 */
export function mergeTargets(run: TargetRun, slot: number) {
  const mirrored = new Set(run.frames.filter(frame => frame.mirror).map(frame => frame.ordinal))
  if (run.detections.some(item => item.slot === slot && mirrored.has(item.frame_ordinal))) return []
  return run.slots.filter(item => item.slot !== slot)
}
