import json
import unittest

from app import pose_detail, prompt_context
from app.visual_contracts import frame_schema, resolve_frame

MAI = {'ref': 'B1', 'character_context_id': 'mai', 'name': 'Mai', 'box': [10, 100, 300, 900]}
RIO = {'ref': 'B2', 'character_context_id': 'rio', 'name': 'Rio', 'box': [600, 100, 900, 900]}
PEOPLE = [MAI, RIO]


def block(**changes):
    return {'body_pose': 'sitting', 'body_yaw': 45, 'head_yaw': 45, 'head_pitch': 0, 'eye_gaze': 'right', 'gaze_target': 'B2', 'gaze_object': '', **changes}


class PoseDetailTest(unittest.TestCase):
    def test_a_clean_block_is_kept_and_enriched_from_the_boxes(self):
        pose = pose_detail.resolve(block(), MAI, PEOPLE)
        self.assertEqual((pose['body_pose'], pose['body_yaw'], pose['head_yaw'], pose['eye_gaze']), ('sitting', 45, 45, 'right'))
        self.assertEqual((pose['gaze_target'], pose['gaze_target_id']), ('Rio', 'rio'))  # the handle becomes the character
        self.assertEqual((pose['body_view'], pose['head_view']), ('three_quarter_right', 'three_quarter_right'))
        self.assertEqual(pose['frame_fill'], .8)
        self.assertNotIn('flags', pose)
        self.assertNotIn('coerced', pose)

    def test_vocabulary_drift_is_coerced_never_rejected(self):
        pose = pose_detail.resolve(block(body_pose='Bent down', eye_gaze='upper left', body_yaw='47', head_yaw=-100.4, head_pitch=-20), MAI, PEOPLE)
        self.assertEqual((pose['body_pose'], pose['eye_gaze']), ('crouching', 'up_left'))
        self.assertEqual((pose['body_yaw'], pose['head_yaw'], pose['head_pitch']), (45, -105, -15))  # snapped to 15 degrees
        self.assertEqual(sorted(pose['coerced']), ['body_pose', 'body_yaw', 'eye_gaze'])
        for word, expected in (('seated', 'sitting'), ('Stands', 'standing'), ('walks', 'walking'), ('turning', 'other'), ('', 'unclear'), (None, 'unclear'), (7, 'unclear')):
            self.assertEqual(pose_detail.body_pose(word)[0], expected, word)
        for word, expected in (('none', 'not_visible'), ('eyes closed', 'closed'), ('top-right', 'up_right'), ('straight', 'center'), ('weird', 'unclear')):
            self.assertEqual(pose_detail.eye_gaze(word)[0], expected, word)

    def test_angles_are_snapped_clamped_and_may_be_null(self):
        self.assertEqual(pose_detail.snap(44, 180), 45)
        self.assertEqual(pose_detail.snap(-181, 180), 180)  # -180 and 180 are the same direction
        self.assertEqual(pose_detail.snap('about 90 degrees', 180), 90)
        self.assertEqual(pose_detail.snap(200, 90), 90)
        for value in (None, True, 'no', float('nan'), [], {}):
            self.assertIsNone(pose_detail.snap(value, 180))
        pose = pose_detail.resolve(block(body_yaw=None, head_yaw='n/a', head_pitch=None), MAI, PEOPLE)
        self.assertEqual((pose['body_yaw'], pose['head_yaw'], pose['body_view']), (None, None, None))

    def test_gaze_targets(self):
        cases = {'camera': ('camera', None, ''), 'The Camera': ('camera', None, ''), 'off screen': ('off_screen', None, ''), 'nothing': ('none', None, ''), 'b1': ('none', None, ''),
                 'Rio': ('Rio', 'rio', ''), 'object: red book': ('object', None, 'red book'), 'the window': ('object', None, 'the window'), '': ('unclear', None, '')}
        for value, expected in cases.items():
            self.assertEqual(pose_detail.gaze_target(value, '', 'B1', PEOPLE)[:3], expected, value)
        self.assertEqual(pose_detail.gaze_target('object', 'umbrella', 'B1', PEOPLE)[:3], ('object', None, 'umbrella'))
        self.assertEqual(pose_detail.gaze_target(None, None, 'B1', PEOPLE)[0], 'unclear')

    def test_geometry_disagreements_are_flagged_not_changed(self):
        pose = pose_detail.resolve(block(body_yaw=-45, head_yaw=-60, eye_gaze='left'), MAI, PEOPLE)  # Rio is on the image right
        self.assertEqual([item['code'] for item in pose['flags']], ['gaze_side'])
        self.assertEqual((pose['head_yaw'], pose['eye_gaze'], pose['gaze_target']), (-60, 'left', 'Rio'))
        self.assertIn('Rio', pose['flags'][0]['text'])
        camera = pose_detail.resolve(block(gaze_target='camera', head_yaw=90, eye_gaze='right'), MAI, PEOPLE)
        self.assertIn('camera_gaze', [item['code'] for item in camera['flags']])
        tilt = pose_detail.resolve(block(gaze_target='none', head_pitch=-45, eye_gaze='up', head_yaw=0), MAI, PEOPLE)
        self.assertIn('pitch_gaze', [item['code'] for item in tilt['flags']])
        twist = pose_detail.resolve(block(gaze_target='none', body_yaw=0, head_yaw=135, eye_gaze='center'), MAI, PEOPLE)
        self.assertIn('head_body', [item['code'] for item in twist['flags']])
        close = [{**MAI, 'box': [400, 0, 500, 900]}, {**RIO, 'box': [450, 0, 550, 900]}]  # centres too close to say a side
        self.assertEqual(pose_detail.resolve(block(body_yaw=-45, head_yaw=-60, eye_gaze='left'), close[0], close).get('flags'), None)

    def test_unusable_input_never_raises(self):
        for raw in (None, 'text', [], 5, {}, {'unrelated': 1}):
            self.assertEqual(pose_detail.resolve(raw, MAI, PEOPLE), {'status': 'missing'})
        pose = pose_detail.resolve({'body_pose': 4, 'body_yaw': {'x': 1}, 'eye_gaze': [], 'gaze_target': 3}, {'ref': 'B9', 'name': 'x', 'character_context_id': None}, [])
        self.assertEqual((pose['body_pose'], pose['body_yaw'], pose['eye_gaze']), ('unclear', None, 'unclear'))

    def test_compact_form_is_short_and_idempotent(self):
        pose = pose_detail.resolve(block(head_pitch=-30, head_yaw=-60, eye_gaze='left'), MAI, PEOPLE)
        small = pose_detail.compact(pose)
        self.assertEqual((small['state'], small['looking_at']), ('sitting', 'Rio'))
        self.assertEqual(small['torso'], 'three-quarter view toward screen-right (+45°)')  # the dictionary's own words, viewer-relative
        self.assertEqual(small['head'], 'three-quarter view toward screen-left (-60°), chin lowered')
        self.assertEqual(small['eyes'], 'toward screen-left')
        self.assertTrue(small['check'])
        self.assertEqual(pose_detail.compact(small), small)
        self.assertIsNone(pose_detail.compact({'status': 'missing'}))
        self.assertIsNone(pose_detail.compact(None))
        self.assertEqual(pose_detail.compact({'body_pose': 'unclear', 'eye_gaze': 'unclear', 'gaze_target': 'unclear', 'body_yaw': None, 'head_yaw': None, 'head_pitch': None}), None)

    def test_the_grammar_offers_only_vocabulary_and_handles(self):
        schema = pose_detail.json_schema(['B1', 'B2'])
        self.assertEqual(set(schema['required']), set(pose_detail.FIELDS))
        self.assertEqual(schema['properties']['gaze_target']['enum'][:2], ['B1', 'B2'])
        self.assertEqual(schema['properties']['head_pitch'], {'type': ['integer', 'null'], 'minimum': -90, 'maximum': 90})
        self.assertIn('crouching', schema['properties']['body_pose']['enum'])
        json.dumps(schema)

    def test_frames_carry_the_pose_block_per_character(self):
        raw = {'composition': 'MS', 'setting': 's', 'lighting': 'l', 'camera': 'c', 'background_people': 'none', 'notes': 'none',
               'visible_characters': [{'box': 'B2', 'pose_detail': block(body_pose='walking', gaze_target='camera', head_yaw=0, eye_gaze='center')},
                                      {'box': 'B1', 'pose_detail': block()}]}
        result = resolve_frame(json.dumps(raw), PEOPLE)
        self.assertEqual(result['contract'], 'frame-v2')
        mai, rio = result['visible_characters']
        self.assertEqual((mai['character'], mai['pose_detail']['gaze_target']), ('Mai', 'Rio'))
        self.assertEqual((rio['character'], rio['pose_detail']['body_pose'], rio['pose_detail']['gaze_target']), ('Rio', 'walking', 'camera'))
        self.assertEqual(pose_detail.stats(result['visible_characters']), {'entries': 2, 'missing': 0, 'coerced': 0, 'flags': 0})
        skipped = resolve_frame(json.dumps({**raw, 'visible_characters': [{'box': 'B1'}]}), PEOPLE)  # a server without the grammar may leave the block out
        self.assertEqual([item['pose_detail'] for item in skipped['visible_characters']], [{'status': 'missing'}] * 2)
        self.assertEqual(pose_detail.stats(skipped['visible_characters'])['missing'], 2)
        self.assertIn('pose_detail', frame_schema(['B1'])['properties']['visible_characters']['items']['properties'])

    def test_prompts_read_the_compact_form_only(self):
        result = resolve_frame(json.dumps({'composition': 'MS', 'setting': 's', 'lighting': 'l', 'camera': 'c', 'background_people': 'none', 'notes': 'none',
                                           'visible_characters': [{'box': 'B1', 'pose_detail': block()}, {'box': 'B2'}]}), PEOPLE)
        view = prompt_context.frame_view(result)
        self.assertNotIn('contract', view)
        mai, rio = view['visible_characters']
        self.assertEqual(set(mai['pose_detail']), {'state', 'torso', 'head', 'eyes', 'looking_at'})
        self.assertNotIn('pose_detail', rio)  # a missing block adds nothing to the prompt
        self.assertEqual(prompt_context.frame_view(view), view)  # slimming twice is safe
        self.assertEqual(prompt_context.frame_view({'composition': 'x', 'visible_characters': [{'character': 'Mai'}]})['visible_characters'], [{'character': 'Mai'}])


if __name__ == '__main__':
    unittest.main()
