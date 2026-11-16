// Pure helpers for the pose block of one first-frame character (`visible_characters[].pose_detail`). The backend's
// pose_detail.py writes it; this file reads it for the box card and applies a person's corrections to it.

export const BODY_POSES = ['standing', 'walking', 'running', 'sitting', 'lying', 'crouching', 'kneeling', 'leaning', 'jumping', 'other', 'unclear'] as const
export const EYE_GAZES = ['up_left', 'up', 'up_right', 'left', 'center', 'right', 'down_left', 'down', 'down_right', 'closed', 'not_visible', 'unclear'] as const
export const TARGET_KINDS = ['camera', 'off_screen', 'object', 'none', 'unclear'] as const
export const ANGLE_STEP = 15

export type Pose = {
  status?: 'missing'
  body_pose?: string; body_yaw?: number | null; head_yaw?: number | null; head_pitch?: number | null
  eye_gaze?: string; gaze_target?: string; gaze_object?: string; gaze_target_id?: string
  body_view?: string | null; head_view?: string | null; frame_fill?: number
  flags?: { code: string; text: string }[]; coerced?: string[]; edited?: boolean
}
type Person = Record<string, unknown> & { character?: string; character_context_id?: string; pose_detail?: Pose }
export type FrameContent = Record<string, unknown> & { visible_characters?: Person[] }
type BoxLike = { character_context_id: string | null; box: [number, number, number, number] }

/** An angle as a whole number on the 15-degree step within the limit; null when it is not a number. */
export function snapAngle(value: unknown, limit: number): number | null {
  const number = typeof value === 'string' && value.trim() !== '' ? Number(value) : value
  if (typeof number !== 'number' || !Number.isFinite(number)) return null
  const snapped = Math.round(Math.max(-limit, Math.min(limit, number)) / ANGLE_STEP) * ANGLE_STEP
  const clamped = Math.max(-limit, Math.min(limit, snapped))
  return clamped === -180 ? 180 : clamped
}

/** The direction as a view of the camera with the image side it turns toward; same rule as the backend. */
export function viewOf(yaw: number | null | undefined): string | null {
  if (yaw === null || yaw === undefined) return null
  const side = yaw > 0 ? 'right' : 'left'
  const size = Math.abs(yaw)
  return size <= 15 ? 'front' : size <= 60 ? `three_quarter_${side}` : size <= 105 ? `profile_${side}` : size <= 150 ? `back_three_quarter_${side}` : 'back'
}

const VIEW_NAMES: Record<string, string> = { front: 'facing the camera', three_quarter: 'three-quarter view', profile: 'profile', back_three_quarter: 'mostly turned away', back: 'back to the camera' }

export function viewWords(view: string | null | undefined, yaw?: number | null): string {
  if (!view) return 'not visible'
  const side = /_(left|right)$/.exec(view)?.[1]
  const base = side ? view.slice(0, -side.length - 1) : view
  return `${VIEW_NAMES[base] ?? base.replace(/_/g, ' ')}${side ? ` toward the image ${side}` : ''}${typeof yaw === 'number' && view !== 'front' ? ` (${yaw > 0 ? '+' : ''}${yaw}°)` : ''}`
}

export const label = (value: string | null | undefined) => (value ? value.replace(/_/g, ' ') : 'unclear')

/**
 * The entry of `content.visible_characters` that belongs to a box. Entries are in the order of the boxes' left edges
 * (the handles the model saw), so the box's rank among the named boxes of its frame finds it; when boxes were edited
 * since, the first entry of the same character is used.
 */
export function entryIndex(content: FrameContent | null | undefined, box: BoxLike, frameBoxes: BoxLike[]): number {
  const people = Array.isArray(content?.visible_characters) ? content.visible_characters : []
  if (!box.character_context_id || !people.length) return -1
  const named = frameBoxes.filter(item => item.character_context_id).sort((a, b) => a.box[0] - b.box[0])
  const rank = named.indexOf(box)
  if (rank >= 0 && people[rank]?.character_context_id === box.character_context_id) return rank
  const sameBefore = named.slice(0, Math.max(rank, 0)).filter(item => item.character_context_id === box.character_context_id).length
  const matches = people.map((person, index) => (person.character_context_id === box.character_context_id ? index : -1)).filter(index => index >= 0)
  return matches[Math.min(sameBefore, matches.length - 1)] ?? -1
}

export type PoseSummary = { state: string; torso: string; head: string; eyes: string; target: string; flags: string[] }

/** The block as short lines for the card. A character without a block has no summary. */
export function summarize(pose: Pose | null | undefined): PoseSummary | null {
  if (!pose || pose.status === 'missing' || pose.body_pose === undefined) return null
  const pitch = typeof pose.head_pitch === 'number' && Math.abs(pose.head_pitch) >= 15 ? `, tilted ${pose.head_pitch < 0 ? 'down' : 'up'} (${pose.head_pitch}°)` : ''
  const target = pose.gaze_target && pose.gaze_target !== 'unclear' ? (pose.gaze_target === 'object' ? `an object${pose.gaze_object ? ` (${pose.gaze_object})` : ''}` : label(pose.gaze_target)) : 'unclear'
  return { state: label(pose.body_pose), torso: viewWords(pose.body_view, pose.body_yaw), head: viewWords(pose.head_view, pose.head_yaw) + pitch, eyes: label(pose.eye_gaze), target, flags: (pose.flags ?? []).map(flag => flag.text) }
}

export type PoseEdit = Partial<Pick<Pose, 'body_pose' | 'eye_gaze' | 'gaze_target' | 'gaze_object'>> & { body_yaw?: unknown; head_yaw?: unknown; head_pitch?: unknown }

/**
 * A copy of the frame content with one character's pose block corrected by the user. Angles are snapped, the views are
 * derived again, and the app's flags and coercion notes (which described the old values) are dropped.
 */
export function applyEdit(content: FrameContent, index: number, edit: PoseEdit): FrameContent {
  const people = Array.isArray(content.visible_characters) ? content.visible_characters : []
  const person = people[index]
  if (!person) return content
  const pose: Pose = { ...(person.pose_detail && person.pose_detail.status !== 'missing' ? person.pose_detail : { body_pose: 'unclear', eye_gaze: 'unclear', gaze_target: 'unclear', gaze_object: '', body_yaw: null, head_yaw: null, head_pitch: null }) }
  if (edit.body_pose !== undefined) pose.body_pose = edit.body_pose
  if (edit.eye_gaze !== undefined) pose.eye_gaze = edit.eye_gaze
  if ('body_yaw' in edit) pose.body_yaw = snapAngle(edit.body_yaw, 180)
  if ('head_yaw' in edit) pose.head_yaw = snapAngle(edit.head_yaw, 180)
  if ('head_pitch' in edit) pose.head_pitch = snapAngle(edit.head_pitch, 90)
  if (edit.gaze_target !== undefined) {
    const other = people.find(item => item.character === edit.gaze_target && item !== person)
    pose.gaze_target = edit.gaze_target
    if (other?.character_context_id) pose.gaze_target_id = other.character_context_id
    else delete pose.gaze_target_id
    if (edit.gaze_target !== 'object') pose.gaze_object = ''
  }
  if (edit.gaze_object !== undefined) pose.gaze_object = edit.gaze_object.slice(0, 60)
  pose.body_view = viewOf(pose.body_yaw)
  pose.head_view = viewOf(pose.head_yaw)
  delete pose.flags
  delete pose.coerced
  pose.edited = true
  return { ...content, visible_characters: people.map((item, position) => (position === index ? { ...item, pose_detail: pose } : item)) }
}
