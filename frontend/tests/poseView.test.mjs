import { test } from 'node:test'
import assert from 'node:assert/strict'
import { applyEdit, entryIndex, snapAngle, summarize, viewOf, viewWords } from '../src/poseView.ts'

const pose = (extra = {}) => ({ body_pose: 'sitting', body_yaw: 45, head_yaw: -60, head_pitch: -15, eye_gaze: 'left', gaze_target: 'Rio', gaze_object: '', gaze_target_id: 'rio', body_view: 'three_quarter_right', head_view: 'three_quarter_left',
  flags: [{ code: 'gaze_side', text: 'Mai looks at Rio, who is on the image right, but the head points left' }], coerced: ['body_pose'], ...extra })
const content = () => ({ composition: 'MS', visible_characters: [{ character: 'Mai', character_context_id: 'mai', pose_detail: pose() }, { character: 'Rio', character_context_id: 'rio', pose_detail: { status: 'missing' } }, { character: 'Mai', character_context_id: 'mai' }] })
const box = (left, id) => ({ character_context_id: id, box: [left, 0, left + 100, 500] })

test('angles are whole numbers on the 15-degree step, clamped, or null', () => {
  assert.equal(snapAngle(44, 180), 45)
  assert.equal(snapAngle('-100', 180), -105)
  assert.equal(snapAngle(-181, 180), 180)  // -180 and 180 are the same direction
  assert.equal(snapAngle(200, 90), 90)
  for (const value of [null, undefined, '', 'abc', NaN, Infinity, {}]) assert.equal(snapAngle(value, 180), null)
})

test('views follow the backend rule and read as words', () => {
  assert.deepEqual([0, 15, 45, 90, 135, 180, -45, -90, -170].map(viewOf), ['front', 'front', 'three_quarter_right', 'profile_right', 'back_three_quarter_right', 'back', 'three_quarter_left', 'profile_left', 'back'])
  assert.equal(viewOf(null), null)
  assert.equal(viewWords('three_quarter_right', 45), 'three-quarter view toward the image right (+45°)')
  assert.equal(viewWords('profile_left', -90), 'profile toward the image left (-90°)')
  assert.equal(viewWords('front', 0), 'facing the camera')
  assert.equal(viewWords(null), 'not visible')
})

test('a box finds its entry by rank among the named boxes, then by character', () => {
  const frame = content()
  const [mai, rio, mai2] = [box(100, 'mai'), box(400, 'rio'), box(700, 'mai')]
  const boxes = [rio, mai2, mai, box(50, null)]  // order of the list does not matter; unnamed boxes do not count
  assert.deepEqual([mai, rio, mai2].map(item => entryIndex(frame, item, boxes)), [0, 1, 2])
  assert.equal(entryIndex(frame, box(10, null), boxes), -1)
  assert.equal(entryIndex(null, mai, boxes), -1)
  // The boxes changed since the analysis (Rio's box was relabelled): the first entry of the same character is used.
  const moved = box(5, 'mai')
  assert.equal(entryIndex(frame, moved, [moved, rio]), 0)
  assert.equal(entryIndex(frame, box(900, 'kaede'), [box(900, 'kaede')]), -1)
})

test('the card shows short lines and the flags, and nothing for a character without a block', () => {
  const view = summarize(pose())
  assert.equal(view.state, 'sitting')
  assert.equal(view.torso, 'three-quarter view toward the image right (+45°)')
  assert.equal(view.head, 'three-quarter view toward the image left (-60°), tilted down (-15°)')
  assert.equal(view.target, 'Rio')
  assert.equal(view.flags.length, 1)
  assert.equal(summarize({ status: 'missing' }), null)
  assert.equal(summarize(undefined), null)
  assert.equal(summarize(pose({ gaze_target: 'object', gaze_object: 'cup' })).target, 'an object (cup)')
})

test('a correction snaps angles, derives the views and drops what described the old values', () => {
  const edited = applyEdit(content(), 0, { head_yaw: '44', body_pose: 'standing' })
  const next = edited.visible_characters[0].pose_detail
  assert.equal(next.body_pose, 'standing')
  assert.equal(next.head_yaw, 45)
  assert.equal(next.head_view, 'three_quarter_right')
  assert.equal(next.body_yaw, 45)  // untouched fields stay
  assert.equal(next.edited, true)
  assert.equal(next.flags, undefined)
  assert.equal(next.coerced, undefined)
  assert.equal(content().visible_characters[0].pose_detail.head_yaw, -60)  // the original is not changed
  assert.deepEqual(edited.visible_characters[1], content().visible_characters[1])
  assert.equal(edited.composition, 'MS')
})

test('looking at another character links it; an object keeps its noun and any other target clears it', () => {
  const toRio = applyEdit(content(), 0, { gaze_target: 'Rio' }).visible_characters[0].pose_detail
  assert.deepEqual([toRio.gaze_target, toRio.gaze_target_id], ['Rio', 'rio'])
  const object = applyEdit(content(), 0, { gaze_target: 'object' })
  const namedFrame = applyEdit(object, 0, { gaze_object: 'red cup' })
  const named = namedFrame.visible_characters[0].pose_detail
  assert.deepEqual([named.gaze_target, named.gaze_object, named.gaze_target_id], ['object', 'red cup', undefined])
  const camera = applyEdit(namedFrame, 0, { gaze_target: 'camera' }).visible_characters[0].pose_detail
  assert.deepEqual([camera.gaze_target, camera.gaze_object], ['camera', ''])
  assert.equal(applyEdit(content(), 0, { head_pitch: '' }).visible_characters[0].pose_detail.head_pitch, null)
})

test('a missing block can be filled in by hand and a missing entry changes nothing', () => {
  const filled = applyEdit(content(), 1, { body_pose: 'lying', head_yaw: 90 }).visible_characters[1].pose_detail
  assert.deepEqual([filled.body_pose, filled.head_yaw, filled.head_view, filled.eye_gaze, filled.gaze_target], ['lying', 90, 'profile_right', 'unclear', 'unclear'])
  const same = content()
  assert.equal(applyEdit(same, 7, { body_pose: 'lying' }), same)
})
