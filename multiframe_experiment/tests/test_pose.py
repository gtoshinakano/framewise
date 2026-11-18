"""DWPose rules (synthetic keypoints -> phrases and changes), tracking, swap detection, the C+A merge and the new scoring."""
import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import pose  # noqa: E402  (pure parts only: no onnxruntime needed)
import pose_rules as rules  # noqa: E402
import schema  # noqa: E402
import scoring  # noqa: E402
from tests.test_core import sample  # noqa: E402

SIZE = (1000, 500)
STATIC = {'move': 'static', 'dx': 0.0, 'dy': 0.0, 'scale': 1.0}
# A person facing the camera: their left shoulder (5) is on screen-right. Torso length 80, shoulders 60 wide.
BASE = {0: (100, 40), 1: (105, 35), 2: (95, 35), 3: (112, 38), 4: (88, 38), 5: (130, 70), 6: (70, 70), 7: (135, 110), 8: (65, 110),
        9: (137, 150), 10: (63, 150), 11: (112, 150), 12: (88, 150), 13: (113, 210), 14: (87, 210), 15: (113, 270), 16: (87, 270)}


def person(name='Tomoe Koga', dx=0, box=None, moves=None):
    points = [[0.0, 0.0, 0.0] for _ in range(133)]
    for index, (x, y) in {**BASE, **(moves or {})}.items():
        if (x, y) != (None, None):
            points[index] = [x + dx, y, 0.9]
    found = box or [60 + dx, 20, 140 + dx, 280]
    return {'name': name, 'box': found, 'points': points, 'usable': pose.body_usable(points)}


def state(p):
    return rules.person_state(p, SIZE)


class Phrases(unittest.TestCase):
    def test_standing_facing_camera(self):
        s = state(person())
        self.assertEqual(s['pose'], 'standing')
        self.assertEqual(s['torso_yaw'], 0)
        self.assertEqual(s['head_yaw'], 0)
        self.assertEqual(s['arms']['screen-left']['phrase'], 'hanging naturally at the side')
        self.assertEqual(s['arms']['screen-right']['elbow_phrase'], 'fully extended')
        text = rules.state_text(s)
        self.assertIn('standing upright', text)
        self.assertIn('head facing the camera', text)
        self.assertIn('far screen-left', text)

    def test_arm_raised_named_by_screen_side(self):
        a, b = state(person()), state(person(moves={7: (170, 70), 9: (210, 70)}))  # the person's left arm = screen-right
        self.assertEqual(b['arms']['screen-right']['phrase'], 'raised to shoulder height')
        walking, changes = rules.compare(a, b)
        self.assertEqual(walking, 'no')
        self.assertIn({'part': 'arms', 'description': 'screen-right arm: hanging naturally at the side → raised to shoulder height'}, changes)
        self.assertFalse(any('screen-left arm' in c['description'] for c in changes))

    def test_arm_above_head_and_hand_regions(self):
        up = state(person(moves={7: (135, 30), 9: (137, -10)}))
        self.assertEqual(up['arms']['screen-right']['phrase'], 'raised above the head')
        self.assertEqual(up['hands']['screen-right']['region'], 'above the head')
        face = state(person(moves={7: (125, 100), 9: (106, 45)}))
        self.assertEqual(face['hands']['screen-right']['region'], 'near the face')
        _, changes = rules.compare(state(person()), face)
        self.assertIn({'part': 'hands', 'description': 'screen-right hand: beside the body → near the face'}, changes)

    def test_body_poses(self):
        lying = state(person(moves={5: (200, 100), 6: (200, 120), 11: (300, 100), 12: (300, 120), 13: (360, 100), 14: (360, 120), 15: (420, 100), 16: (420, 120)}))
        self.assertEqual(lying['pose'], 'lying')
        crouch = state(person(moves={13: (140, 150), 14: (60, 150), 15: (115, 175), 16: (85, 175)}))
        self.assertEqual(crouch['pose'], 'crouching')
        sit = state(person(moves={13: (150, 150), 14: (50, 150), 15: (150, 210), 16: (50, 210)}))
        self.assertEqual(sit['pose'], 'sitting')
        _, changes = rules.compare(state(person()), sit)
        self.assertIn({'part': 'body_pose', 'description': 'standing upright → sitting'}, changes)

    def test_kick_is_not_sitting(self):
        # Tomoe's high kick (cut 143): one thigh horizontal, the foot above the hips, the other leg straight down.
        kick = state(person(moves={13: (160, 150), 15: (210, 140)}))
        self.assertEqual(kick['pose'], 'standing')

    def test_wide_body_is_lying_and_does_not_turn(self):
        # Seen from behind and above (cut 144): the torso looks upright in the image but the box is wide.
        dive = state(person(box=[0, 100, 400, 280], moves={5: (150, 70), 6: (50, 70), 7: (200, 40), 8: (0, 40)}))
        self.assertEqual(dive['pose'], 'lying')
        self.assertIsNone(dive['torso_yaw'])

    def test_hand_carried_by_the_body_is_not_a_hand_move(self):
        _, changes = rules.compare(state(person()), state(person(dx=150)))
        self.assertFalse(any(c['part'] == 'hands' for c in changes))
        self.assertTrue(any(c['part'] == 'position' for c in changes))

    def test_alternating_ankles_with_travel_is_walking(self):
        a = state(person())
        b = state(person(dx=70, moves={15: (95, 255), 16: (105, 270)}))  # the feet cross over: the other foot leads
        self.assertTrue(rules.alternation(a, b)['flip'])
        self.assertEqual(rules.compare(a, b, STATIC)[0], 'yes')

    def test_head_turn(self):
        turned = state(person(moves={0: (112, 40)}))
        self.assertEqual(turned['head_yaw'], 90)
        _, changes = rules.compare(state(person()), turned)
        self.assertIn({'part': 'head', 'description': 'head turns: facing the camera → side profile facing screen-right'}, changes)
        away = state(person(moves={0: (None, None), 1: (None, None), 2: (None, None)}))
        self.assertEqual(away['head_yaw'], 180)

    def test_travel_with_unchanged_legs_is_left_undecided(self):
        a = state(person())
        slide = state(person(dx=100))  # travels with the legs unchanged: a position change, and the keypoints cannot say walking
        walking, changes = rules.compare(a, slide, STATIC)
        self.assertEqual(walking, 'unclear')
        self.assertTrue(any(c['part'] == 'position' and 'screen-right' in c['description'] for c in changes))

    def test_legs_that_open_need_a_step_next_to_them(self):
        a = state(person())
        opened = state(person(dx=100, moves={15: (140, 270), 16: (60, 270)}))  # feet apart, none crossed: half a step
        self.assertFalse(rules.alternation(a, opened)['flip'])
        self.assertGreaterEqual(rules.alternation(a, opened)['swing'], rules.SWING_STEP)
        self.assertEqual(rules.compare(a, opened, STATIC)[0], 'unclear')
        walking, changes = rules.compare(a, opened, STATIC, near_flip=True)
        self.assertEqual(walking, 'yes')
        self.assertFalse(any(c['description'].startswith('moves toward') for c in changes))  # the walk says it

    def test_ankles_seen_toward_the_camera_alternate_in_height(self):
        a = state(person(moves={15: (113, 285), 16: (87, 255)}))   # the person's left foot is the nearer, lower one
        b = state(person(dx=70, moves={15: (113, 255), 16: (87, 285)}))
        self.assertEqual(rules.alternation(a, b)['flip'], True)
        self.assertEqual(rules.compare(a, b, STATIC)[0], 'yes')

    def test_steps_without_travel_are_not_walking(self):
        a = state(person())
        b = state(person(moves={15: (95, 255), 16: (105, 270)}))
        self.assertEqual(rules.compare(a, b, STATIC)[0], 'no')           # in place, camera static
        self.assertEqual(rules.compare(a, b, None)[0], 'unclear')        # no camera estimate: the camera may be following them
        tall = person(box=[60, 0, 140, 500])
        tall_b = person(box=[60, 0, 140, 500], moves={15: (95, 255), 16: (105, 270)})
        self.assertEqual(rules.compare(state(tall), state(tall_b), STATIC)[0], 'unclear')  # fills the picture: the estimate may follow them

    def test_a_pan_that_follows_the_walker(self):
        a = state(person())
        steps = state(person(moves={15: (95, 255), 16: (105, 270)}))    # same place in the picture, feet alternate
        follow = {'move': 'pan_right', 'dx': -0.1, 'dy': 0.0, 'scale': 1.0}     # the background slides left: the camera follows to the right
        self.assertEqual(rules.compare(a, steps, follow)[0], 'yes')
        still = {'move': 'pan_left', 'dx': 0.1, 'dy': 0.0, 'scale': 1.0}        # the picture moves with the person: a stationary person
        moved = state(person(dx=100, moves={15: (95, 255), 16: (105, 270)}))
        self.assertEqual(rules.compare(a, moved, still)[0], 'no')
        _, changes = rules.compare(a, moved, still)
        self.assertEqual([c for c in changes if c['part'] == 'position'], [])

    def test_without_a_camera_estimate_travel_must_be_larger(self):
        a = state(person())
        small = state(person(dx=70, moves={15: (95, 255), 16: (105, 270)}))   # 0.07 of the width
        large = state(person(dx=120, moves={15: (95, 255), 16: (105, 270)}))  # 0.12
        self.assertEqual(rules.compare(a, small, {'move': 'unclear'})[0], 'unclear')
        self.assertEqual(rules.compare(a, large, {'move': 'unclear'})[0], 'yes')

    def test_sitting_state_blocks_walking(self):
        sit = state(person(dx=100, moves={13: (150, 150), 14: (50, 150), 15: (150, 210), 16: (50, 210)}))
        self.assertEqual(sit['pose'], 'sitting')
        self.assertEqual(rules.compare(state(person()), sit, STATIC)[0], 'no')
        self.assertEqual(rules.compare(sit, state(person(dx=200)), STATIC)[0], 'no')  # standing up and moving off is not walking either
        kick = state(person(dx=100, moves={13: (160, 150), 15: (210, 140)}))
        self.assertEqual(rules.compare(state(person()), kick, STATIC)[0], 'no')

    def test_a_lower_body_outside_the_picture_is_never_sitting(self):
        bust = rules.person_state(person(), (1000, 120))  # hips, knees and ankles lie below the picture: DWPose invented them
        self.assertIsNone(bust['pose'])
        self.assertEqual(bust['legs'], {})
        self.assertFalse(bust['parts']['walking'])
        self.assertEqual(rules.compare(bust, rules.person_state(person(dx=100), (1000, 120)), STATIC)[0], 'unclear')

    def test_hips_at_the_bottom_of_the_picture_are_not_a_seen_body(self):
        # Cut 153: a close-up whose hips lie in the bottom strip, knees "above" them. DWPose invented the legs; it is no sitting or lying body.
        bust = person(box=[0, 0, 1000, 500], moves={5: (700, 250), 6: (300, 250), 11: (640, 470), 12: (360, 475), 13: (700, 200), 14: (300, 210), 15: (700, 150), 16: (300, 160)})
        s = state(bust)
        self.assertIsNone(s['pose'])
        self.assertEqual(s['legs'], {})
        self.assertIsNone(s.get('torso_tilt'))
        seen = state(person(moves={11: (112, 380), 12: (88, 380), 13: (150, 380), 14: (50, 380), 15: (150, 420), 16: (50, 420)}))  # hips at 76%: a real body
        self.assertIsNotNone(seen['pose'])

    def test_a_box_cut_by_the_picture_or_a_tilt_says_nothing_about_size(self):
        a = state(person())
        bigger = person(box=[40, 0, 160, 330])
        self.assertTrue(any('comes closer' in c['description'] for c in rules.compare(a, state(bigger), STATIC)[1]) is False)  # top edge cuts the box
        # not cut, same body in view, box 40% taller: a real approach
        no_torso = {5: (None, None), 6: (None, None)}  # no shoulders: the box is all there is to compare
        free_a, free_b = state(person(box=[60, 40, 140, 260], moves=no_torso)), state(person(box=[40, 20, 160, 340], moves=no_torso))
        self.assertTrue(any('comes closer' in c['description'] for c in rules.compare(free_a, free_b, STATIC)[1]))
        tilt = {'move': 'tilt_up', 'dx': 0.0, 'dy': 0.04, 'scale': 1.0}
        self.assertFalse(any('comes closer' in c['description'] for c in rules.compare(free_a, free_b, tilt)[1]))  # the camera moved the crop

    def test_size_comes_from_the_torso_when_both_frames_have_one(self):
        a = state(person())
        far = state(person(moves={5: (115, 90), 6: (85, 90), 11: (108, 130), 12: (92, 130), 7: (115, 110), 8: (85, 110), 9: (115, 128), 10: (85, 128)}))  # torso 40 instead of 80
        self.assertTrue(a['torso_real'] and far['torso_real'])
        self.assertTrue(any('moves away' in c['description'] for c in rules.compare(a, far, STATIC)[1]))

    def test_a_body_that_was_revealed_or_hidden_gives_no_size_claim(self):
        few = state(person(box=[60, 40, 140, 260], moves={i: (None, None) for i in range(11, 17)}))
        many = state(person(box=[40, 20, 160, 340]))
        self.assertGreaterEqual(many['n_seen'] - few['n_seen'], rules.VISIBLE_STEP)
        self.assertFalse(any('comes closer' in c['description'] for c in rules.compare(few, many, STATIC)[1]))

    def test_position_follows_the_torso_not_the_box(self):
        a = state(person())
        wide = state(person(box=[-200, 20, 140, 280]))  # an arm thrown out widens the box; the torso stays where it was
        self.assertFalse(any(c['part'] == 'position' for c in rules.compare(a, wide, STATIC)[1]))
        moved = state(person(dx=120))
        self.assertTrue(any('moves toward screen-right' in c['description'] for c in rules.compare(a, moved, STATIC)[1]))

    def test_people_enter_and_leave_only_through_the_border(self):
        sample3 = sample(3)
        states = rules.frame_states(pose_sample([[person()], [person()], [person(), person('unknown person 1', dx=400, box=[460, 100, 540, 400])]]))  # appears mid-picture: a detector miss
        answer = rules.answer_for_window(states, sample3['windows'][0], sample3['pairs'])
        self.assertEqual([c['changes'] for c in answer['segments'][1]['characters'] if c['name'] == 'unknown person 1'], [[]])
        edge = rules.frame_states(pose_sample([[person()], [person()], [person(), person('unknown person 1', dx=900, box=[960, 100, 1040, 400])]]))
        answer = rules.answer_for_window(edge, sample3['windows'][0], sample3['pairs'])
        self.assertTrue(any(c['description'].startswith('enters') for p in answer['segments'][1]['characters'] for c in p['changes']))

    def test_arms_on_the_lower_edge_of_a_close_up_are_not_seen(self):
        low = rules.person_state(person(moves={7: (135, 495), 8: (65, 495), 9: (137, 498), 10: (63, 498)}), SIZE)  # elbows and wrists on the bottom edge
        self.assertEqual((low['arms'], low['hands']), ({}, {}))
        raised = state(person(moves={7: (135, 30), 9: (137, -10)}))  # above the picture: a raised hand, kept
        self.assertIn('screen-right', raised['hands'])

    def test_alternating_steps_are_also_a_leg_change(self):
        a = state(person())
        b = state(person(dx=70, moves={15: (95, 255), 16: (105, 270)}))
        walking, changes = rules.compare(a, b, STATIC)
        self.assertEqual(walking, 'yes')
        self.assertIn({'part': 'legs', 'description': 'steps: the other foot takes the lead'}, changes)
        parts = [c['part'] for c in rules.answer_for_window(
            rules.frame_states(pose_sample([[person()], [person(dx=70, moves={15: (95, 255), 16: (105, 270)})]])), sample(2)['windows'][0],
            [{'from': 'F1', 'to': 'F2', 'camera': STATIC}])['segments'][0]['characters'][0]['changes']]
        self.assertIn('legs', parts)

    def test_qwen_keeps_a_body_claim_where_dwpose_reports_nothing(self):
        states = rules.frame_states(pose_sample([[person()], [person(moves={7: (170, 70), 9: (210, 70)})]]))  # DWPose sees an arm rise, no leg change
        window = sample(2)['windows'][0]
        pairs = [{'from': 'F1', 'to': 'F2', 'camera': STATIC}]
        c_answer = rules.answer_for_window(states, window, pairs)
        a_answer = {'frames': [], 'summary': '', 'segments': [{'from': 'F1', 'to': 'F2', 'start': 0, 'end': 0.4, 'camera': {'move': 'static', 'detail': ''},
                                                               'characters': [{'name': 'Tomoe Koga', 'walking': 'no', 'speaking': 'no', 'blinking': 'no',
                                                                               'changes': [{'part': 'arms', 'description': 'raises her left arm'}, {'part': 'legs', 'description': 'shifts her weight'}]}],
                                                               'other_changes': []}]}
        [tomoe] = rules.merge(a_answer, c_answer, states, pairs)['segments'][0]['characters']
        descriptions = [c['description'] for c in tomoe['changes']]
        self.assertNotIn('raises her left arm', descriptions)          # DWPose measured the arm: its claim replaces Qwen's
        self.assertIn('shifts her weight', descriptions)               # DWPose measured no leg change: Qwen's claim stays

    def test_a_turned_body_is_not_an_alternation(self):
        a = state(person())
        turned = state(person(dx=70, moves={11: (88, 150), 12: (112, 150), 15: (87, 270), 16: (113, 270)}))  # sides traded: a turn or a DWPose swap
        self.assertEqual((a['hip_order'], turned['hip_order']), (1, -1))
        self.assertFalse(rules.alternation(a, turned)['flip'])
        self.assertEqual(rules.compare(a, turned, STATIC)[0], 'unclear')

    def test_camera_pan_is_removed_from_position(self):
        a, b = state(person()), state(person(dx=100))
        _, changes = rules.compare(a, b, {'move': 'pan_left', 'dx': 0.1, 'scale': 1.0})
        self.assertEqual([c for c in changes if c['part'] == 'position'], [])

    def test_unusable_body_says_nothing(self):
        few = {i: (None, None) for i in range(5, 17)}
        a, b = state(person(moves=few)), state(person(moves={**few, 0: (112, 40)}))
        self.assertFalse(a['usable'])
        walking, changes = rules.compare(a, b)
        self.assertEqual((walking, changes), ('unclear', []))
        self.assertIn('not usable', rules.state_text(a))


class Tracking(unittest.TestCase):
    def test_names_follow_boxes_backward_and_forward(self):
        frames = [[{'box': [800, 50, 1000, 600], 'name': 'Police Officer'}],
                  [{'box': [790, 60, 990, 600], 'name': None}, {'box': [0, 200, 400, 700], 'name': None}],
                  [{'box': [10, 210, 420, 710], 'name': 'Tomoe Koga'}, {'box': [780, 60, 980, 600], 'name': None}]]
        pose.track(frames, 1280)
        self.assertEqual([p['name'] for p in frames[1]], ['Police Officer', 'Tomoe Koga'])
        self.assertEqual(frames[0][0]['track'], frames[2][1]['track'])

    def test_zoom_out_keeps_the_small_person_on_its_track(self):
        # Cut 158 F3->F4: the camera pulls back, the officer shrinks in the middle and a big new person fills the left.
        frames = [[{'box': [430, 150, 600, 650], 'name': 'Police Officer'}],
                  [{'box': [0, 120, 420, 700], 'name': None}, {'box': [480, 280, 560, 520], 'name': None}]]
        pose.track(frames, 1024, 576, [{'move': 'zoom_out', 'scale': 0.5, 'dx': 0.0, 'dy': 0.0}])
        self.assertEqual([p['name'] for p in frames[1]], ['unknown person 1', 'Police Officer'])

    def test_unnamed_people_are_numbered(self):
        frames = [[{'box': [0, 0, 100, 300], 'name': None}, {'box': [600, 0, 700, 300], 'name': None}]]
        pose.track(frames, 1000)
        self.assertEqual(sorted(p['name'] for p in frames[0]), ['unknown person 1', 'unknown person 2'])

    def test_swap_detection(self):
        a = person()
        b = copy.deepcopy(a)
        b['points'][9], b['points'][10] = b['points'][10], b['points'][9]
        self.assertTrue(pose.suspected_swap(a['points'], b['points'], a['box'], b['box']))
        self.assertFalse(pose.suspected_swap(a['points'], a['points'], a['box'], a['box']))
        moved = [v + 300 for v in a['box']]
        self.assertFalse(pose.suspected_swap(a['points'], b['points'], a['box'], moved))

    def test_stats(self):
        frames = [{'persons': [{**person(), 'source': 'detector', 'confidence': 0.9, 'close_up': False}]},
                  {'persons': [{**person(moves={i: (None, None) for i in range(5, 17)}), 'source': 'detector', 'confidence': 0.2, 'close_up': True}]}]
        for frame in frames:
            for p in frame['persons']:
                p['usable'] = pose.body_usable(p['points'])
        [stats] = pose.track_stats(frames)
        self.assertEqual((stats['frames'], stats['usable'], stats['close_ups'], stats['close_up_usable']), (2, 1, 1, 0))
        self.assertEqual(stats['usable_share'], 0.5)


def pose_sample(people_per_frame):
    return {'frames': [{'label': f'F{k}', 'size': list(SIZE), 'persons': people} for k, people in enumerate(people_per_frame, 1)]}


class Answers(unittest.TestCase):
    def setUp(self):
        self.sample = sample(3)
        self.window = self.sample['windows'][0]
        self.states = rules.frame_states(pose_sample([[person()], [person(moves={7: (170, 70), 9: (210, 70)})], [person(), person('unknown person 1', dx=900)]]))

    def test_window_answer_has_the_shared_shape(self):
        answer = rules.answer_for_window(self.states, self.window, self.sample['pairs'])
        self.assertEqual([s['from'] for s in answer['segments']], ['F1', 'F2'])
        tomoe = answer['segments'][0]['characters'][0]
        self.assertEqual((tomoe['speaking'], tomoe['blinking']), ('unclear', 'unclear'))
        self.assertEqual(answer['segments'][0]['camera']['move'], 'static')  # the pixel estimate
        enters = answer['segments'][1]['characters'][1]
        self.assertEqual(enters['changes'][0]['part'], 'position')
        self.assertTrue(enters['changes'][0]['description'].startswith('enters the frame'))
        claims = schema.answer_claims(answer, self.window, rules.CONFIG_C, self.sample['id'])
        self.assertTrue(any(c['category'] == 'arms_hands' for c in claims))
        self.assertFalse(any(c['category'] in ('speaking', 'blink', 'expression') for c in claims))

    def test_merge_takes_body_from_dwpose_and_face_from_qwen(self):
        c_answer = rules.answer_for_window(self.states, self.window, self.sample['pairs'])
        a_answer = {'frames': [{'frame': 'F1', 't': 0, 'characters': [{'name': 'girl in green', 'state': 'standing on the left of the frame'}]}],
                    'segments': [{'from': 'F1', 'to': 'F2', 'start': 0, 'end': 0.4, 'camera': {'move': 'zoom_in', 'detail': ''},
                                  'characters': [{'name': 'girl in green', 'walking': 'yes', 'speaking': 'yes', 'blinking': 'no',
                                                  'changes': [{'part': 'arms', 'description': 'lowers her arm'}, {'part': 'mouth', 'description': 'opens her mouth'},
                                                              {'part': 'head', 'description': 'tilts her head'}]}], 'other_changes': []},
                                 {'from': 'F2', 'to': 'F3', 'start': 0.4, 'end': 0.8, 'camera': {'move': 'static', 'detail': ''}, 'characters': [], 'other_changes': []}],
                    'summary': 's'}
        merged = rules.merge(a_answer, c_answer, self.states, self.sample['pairs'])
        first = merged['segments'][0]
        self.assertEqual(first['camera']['move'], 'static')  # pixel estimate, not the model's zoom
        [tomoe] = first['characters']
        self.assertEqual(tomoe['name'], 'Tomoe Koga')
        self.assertEqual(tomoe['walking'], 'no')  # DWPose sees the legs
        self.assertEqual(tomoe['speaking'], 'yes')  # Qwen's
        parts = [c['part'] for c in tomoe['changes']]
        self.assertIn('mouth', parts)
        self.assertIn('head', parts)
        self.assertNotIn('lowers her arm', [c['description'] for c in tomoe['changes']])
        self.assertTrue(any(c['part'] == 'arms' and 'raised to shoulder height' in c['description'] for c in tomoe['changes']))
        # F2->F3: Qwen saw nobody, DWPose saw someone enter: kept as a DWPose-only character.
        self.assertTrue(any(p['name'] == 'unknown person 1' for p in merged['segments'][1]['characters']))
        self.assertGreater(merged['merge']['from_dwpose'], 0)

    def test_merge_keeps_qwen_where_dwpose_cannot_see(self):
        blind = rules.frame_states(pose_sample([[person(moves={i: (None, None) for i in range(5, 17)})], [person(moves={i: (None, None) for i in range(5, 17)})]]))
        window = sample(2)['windows'][0]
        c_answer = rules.answer_for_window(blind, window, None)
        a_answer = {'frames': [], 'summary': '', 'segments': [{'from': 'F1', 'to': 'F2', 'start': 0, 'end': 0.4, 'camera': {'move': 'static', 'detail': ''},
                                                               'characters': [{'name': 'Tomoe', 'walking': 'yes', 'speaking': 'no', 'blinking': 'no',
                                                                               'changes': [{'part': 'arms', 'description': 'raises her arm'}]}], 'other_changes': []}]}
        [tomoe] = rules.merge(a_answer, c_answer, blind, None)['segments'][0]['characters']
        self.assertEqual(tomoe['walking'], 'yes')
        self.assertEqual(tomoe['changes'], [{'part': 'arms', 'description': 'raises her arm'}])

    def test_a_step_next_to_an_alternation_makes_the_whole_stride_walking(self):
        # F1 -> F2 the feet cross (alternation, travel); F2 -> F3 the feet open further with more travel (a half step, no flip of its own).
        sample3 = sample(3)
        states = rules.frame_states(pose_sample([[person()], [person(dx=70, moves={15: (95, 255), 16: (105, 270)})],
                                                 [person(dx=140, moves={15: (170 - 140 + 100, 270), 16: (60, 270)})]]))
        answer = rules.answer_for_window(states, sample3['windows'][0], sample3['pairs'])
        self.assertEqual([seg['characters'][0]['walking'] for seg in answer['segments']], ['yes', 'yes'])
        # the same second pair without the neighbouring alternation is undecided
        alone = rules.frame_states(pose_sample([[person(dx=70)], [person(dx=70)], [person(dx=140, moves={15: (170 - 140 + 100, 270), 16: (60, 270)})]]))
        answer = rules.answer_for_window(alone, sample3['windows'][0], sample3['pairs'])
        self.assertEqual(answer['segments'][1]['characters'][0]['walking'], 'unclear')

    def test_neighbouring_steps_must_be_close_in_time(self):
        times = {'F1': 0.0, 'F2': 0.4, 'F3': 5.0, 'F4': 5.4}
        labels = ['F1', 'F2', 'F3', 'F4']
        self.assertFalse(rules._near_flip({0: True}, 2, times, labels))   # 4.6 s between the pairs
        self.assertTrue(rules._near_flip({0: True}, 1, times, labels))
        self.assertTrue(rules._near_flip({2: True}, 1, times, labels))   # the next pair starts where this one ends

    def test_merge_says_no_walking_for_a_rider_or_a_seated_character(self):
        states = rules.frame_states(pose_sample([[person()], [person(dx=70, moves={15: (95, 255), 16: (105, 270)})]]))
        window = sample(2)['windows'][0]
        pairs = [{'from': 'F1', 'to': 'F2', 'camera': STATIC}]
        c_answer = rules.answer_for_window(states, window, pairs)
        self.assertEqual(c_answer['segments'][0]['characters'][0]['walking'], 'yes')  # the keypoints alone cannot tell a bicycle from a walk

        def merged(state_text):
            a_answer = {'frames': [{'frame': 'F1', 't': 0, 'characters': [{'name': 'Tomoe Koga', 'state': state_text}]},
                                   {'frame': 'F2', 't': 0.4, 'characters': [{'name': 'Tomoe Koga', 'state': state_text}]}],
                        'segments': [{'from': 'F1', 'to': 'F2', 'start': 0, 'end': 0.4, 'camera': {'move': 'static', 'detail': ''},
                                      'characters': [{'name': 'Tomoe Koga', 'walking': 'no', 'speaking': 'no', 'blinking': 'no', 'changes': []}], 'other_changes': []}],
                        'summary': ''}
            return rules.merge(a_answer, c_answer, states, pairs)['segments'][0]['characters'][0]['walking']
        self.assertEqual(merged('Riding a bicycle from right to left. Legs in pedaling position.'), 'no')
        self.assertEqual(merged('Sitting on a bench, legs crossed.'), 'no')
        self.assertEqual(merged('Standing on the left of the frame, arms at her sides.'), 'yes')
        self.assertTrue(rules.seated_by_text('She kneels to look at the child') and not rules.seated_by_text('Walks past the bike rack'))

    def test_hybrid_key_follows_the_merged_config(self):
        self.assertEqual(rules.hybrid_config('qwen35|inspector|roster'), 'hybrid|inspector|roster')

    def test_side_words_match_unnamed_characters(self):
        states = {'girl': {'cx': 0.2, 'height': 0.8}, 'officer': {'cx': 0.8, 'height': 0.3}}
        pairs = rules.match_characters(['unknown person 1', 'unknown person 2'], ['girl', 'officer'],
                                       {'unknown person 1': 'stands on the right, small in the background', 'unknown person 2': 'on the left side, raises her left hand'},
                                       states)
        self.assertEqual(pairs, {'unknown person 1': 'officer', 'unknown person 2': 'girl'})


class Scoring(unittest.TestCase):
    def test_ground_truth_and_body_subset(self):
        counts = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in schema.REFERENCE_FLAGS}
        counts['arms_hands'] = {'tp': 2, 'fp': 1, 'fn': 1}
        counts['speaking'] = {'tp': 0, 'fp': 0, 'fn': 2}
        gt = scoring.ground_truth(counts)
        self.assertAlmostEqual(gt['precision'], 2 / 3)
        self.assertAlmostEqual(gt['recall'], 2 / 5)
        self.assertAlmostEqual(gt['body']['recall'], 2 / 3)
        self.assertIsNone(gt['categories']['legs']['precision'])

    def test_rows_with_labels_judge_agreement_and_speaking(self):
        s = sample(3)
        s['transcript'] = [{'start': 0.1, 'end': 0.3, 'text': 'hi', 'speaker': 'Tomoe Koga', 'character': 'Tomoe Koga'},
                           {'start': 0.5, 'end': 0.7, 'text': 'yo', 'speaker': 'SPEAKER_06', 'character': None}]
        window = s['windows'][0]
        answer = {'frames': [], 'summary': '', 'segments': [
            {'from': 'F1', 'to': 'F2', 'start': 0, 'end': 0.4, 'camera': {'move': 'static', 'detail': ''}, 'other_changes': [],
             'characters': [{'name': 'Tomoe Koga', 'walking': 'no', 'speaking': 'yes', 'blinking': 'no', 'changes': [{'part': 'arms', 'description': 'raises her arm'}]}]},
            {'from': 'F2', 'to': 'F3', 'start': 0.4, 'end': 0.8, 'camera': {'move': 'static', 'detail': ''}, 'other_changes': [],
             'characters': [{'name': 'Tomoe Koga', 'walking': 'no', 'speaking': 'yes', 'blinking': 'no', 'changes': [{'part': 'legs', 'description': 'bends her knee'}]}]}]}
        records = {}
        for config in ('qwen35|observer|frames', rules.CONFIG_C):
            records[config] = {'key': config, 'config': config, 'sample': s['id'], 'window': 0, 'ok': True, 'first_try_ok': True, 'attempts': [], 'seconds': 10.0,
                               'answer': answer}
        claims = schema.answer_claims(answer, window, 'x', s['id'])
        from judge import claim_text_key
        verdicts = {claim_text_key(c): 'true' for c in claims if c['judge']}
        judge_records = {'v': {'kind': 'verify', 'ok': True, 'sample': s['id'], 'window': 0, 'verdicts': verdicts},
                         'r': {'kind': 'reference', 'ok': True, 'sample': s['id'], 'window': 0,
                               'pairs': [{'from': 'F1', 'to': 'F2', 'camera': 'static', 'categories': ['arms_hands', 'head']},
                                         {'from': 'F2', 'to': 'F3', 'camera': 'static', 'categories': []}]}}
        labels = {s['id']: {'pairs': [{'from': 'F1', 'to': 'F2', 'camera': 'static', 'flags': {'arms_hands': True, 'speaking': True}},
                                      {'from': 'F2', 'to': 'F3', 'camera': 'static', 'flags': {}}]}}
        rows = scoring.build_rows([s], records, judge_records, labels)
        row = rows['qwen35|observer|frames']
        gt = row['ground_truth']
        self.assertEqual((gt['tp'], gt['fp'], gt['fn']), (2, 2, 0))  # arms+speaking right; legs+speaking on F2→F3 wrong
        self.assertEqual(row['judge_vs_labels']['true_but_unlabelled'], 2)
        # Speaking: F1→F2 has Tomoe (marked) speaking -> right; F2→F3 only an unattributed speaker -> not scored.
        self.assertEqual((row['speaking']['yes_pairs'], row['speaking']['yes_with_speech'], row['speaking']['undecidable_yes']), (1, 1, 1))
        self.assertIsNone(rows[rules.CONFIG_C]['components']['speaking'])
        self.assertEqual(scoring.WEIGHTS['camera'], 0)
        self.assertNotIn('camera', row['score_basis'])
        # A labelled window without a valid answer counts as missed, not as left out.
        missing = scoring.build_rows([s], {'x': {**records['qwen35|observer|frames'], 'ok': False}}, judge_records, labels)
        self.assertEqual(missing['qwen35|observer|frames']['ground_truth']['fn'], 2)
        agreement = scoring.judge_agreement([s], judge_records, labels)
        self.assertEqual(agreement['reference_counts']['head']['fp'], 1)
        self.assertEqual(agreement['reference_counts']['speaking']['fn'], 1)
        self.assertTrue(any('hand-label F1' in line for line in scoring.findings(rows, True)))


if __name__ == '__main__':
    unittest.main()


class Labels(unittest.TestCase):
    def test_labels_only_apply_to_the_same_frames(self):
        import report
        s = sample(3)
        labels = {s['id']: {'pairs': [{'from': 'F1', 'to': 'F2', 'start': 0.0, 'end': 0.4, 'flags': {'head': True}},
                                      {'from': 'F2', 'to': 'F3', 'start': 0.5, 'end': 0.9, 'flags': {'head': True}}]}}
        kept = report.matching_labels(labels, [s])
        self.assertEqual([p['from'] for p in kept[s['id']]['pairs']], ['F1'])
