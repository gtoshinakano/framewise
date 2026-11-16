import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pose_schema as ps  # noqa: E402


def full(**changes):
    entry = {'body_yaw': 0, 'body_pose': 'standing', 'head_yaw': 45, 'head_pitch': -45, 'head_roll': 'neutral', 'eye_gaze': 'down_right',
             'eye_openness': 'normal', 'gaze_target': 'B2', 'mouth_state': 'slightly_open', 'eyebrow_state': 'furrowed',
             'facial_expression': 'worried', 'camera_elevation': 'eye_level', 'shot_size': 'medium_close_up'}
    entry.update(changes)
    return entry


def with_confidence(entry, level=0.8):
    return {**entry, 'confidence': {name: level for name in ps.FIELD_ORDER if name in entry}}


def frame_entry(approach, handle='B1', **changes):
    entry = {'character_id': handle, **full(**changes)}
    return entry if approach == 'strict_detailed' else with_confidence(entry)


class ValidatorTest(unittest.TestCase):
    def test_valid_entries_for_each_approach(self):
        self.assertEqual(ps.validate_entry(frame_entry('full_confidence'), 'full_confidence', ['B1', 'B2']), [])
        self.assertEqual(ps.validate_entry(frame_entry('strict_detailed'), 'strict_detailed', ['B1', 'B2']), [])

    def test_free_angles_pass_in_a1_but_not_on_the_strict_grid(self):
        entry = frame_entry('full_confidence', head_yaw=30, head_pitch=-20)
        self.assertEqual(ps.validate_entry(entry, 'full_confidence', ['B1', 'B2']), [])
        strict = frame_entry('strict_detailed', head_yaw=30, head_pitch=-20)
        errors = ps.validate_entry(strict, 'strict_detailed', ['B1', 'B2'])
        self.assertEqual(len(errors), 2)
        self.assertIn('45-degree grid', errors[0])

    def test_angles_range_and_types(self):
        for value in (200, -181, 'left', True, float('nan')):
            self.assertTrue(ps.check_value('head_yaw', value, 'full_confidence'), value)
        self.assertIsNone(ps.check_value('head_yaw', None, 'full_confidence'))
        self.assertTrue(ps.check_value('head_pitch', 91, 'full_confidence'))

    def test_enum_values_and_abstention(self):
        self.assertIsNone(ps.check_value('eye_gaze', 'unclear', 'strict_detailed'))
        self.assertIn('not allowed', ps.check_value('eye_gaze', 'down-right', 'strict_detailed'))
        self.assertIn('not allowed', ps.check_value('body_pose', 'crouching', 'strict_detailed'))
        self.assertIsNone(ps.check_value('body_pose', 'other', 'strict_detailed'))

    def test_gaze_target_forms(self):
        ids = ['B1', 'B2']
        for good in ('B2', 'camera', 'off_screen', 'none', 'unclear', 'object:book'):
            self.assertIsNone(ps.check_value('gaze_target', good, 'full_confidence', ids), good)
        for bad in ('B3', 'object:', '', 5, 'Mai'):
            self.assertTrue(ps.check_value('gaze_target', bad, 'full_confidence', ids), bad)

    def test_missing_extra_and_unknown_character(self):
        entry = frame_entry('strict_detailed')
        del entry['eye_gaze']
        entry['mood'] = 'x'
        entry['character_id'] = 'B9'
        errors = ps.validate_entry(entry, 'strict_detailed', ['B1', 'B2'])
        self.assertTrue(any('character_id' in e for e in errors))
        self.assertTrue(any('missing fields: eye_gaze' in e for e in errors))
        self.assertTrue(any('unexpected key mood' in e for e in errors))

    def test_a3_rejects_confidence_and_a1_requires_every_field(self):
        errors = ps.validate_entry({**frame_entry('strict_detailed'), 'confidence': {}}, 'strict_detailed', ['B1', 'B2'])
        self.assertTrue(any('remove it' in e for e in errors))
        entry = frame_entry('full_confidence')
        del entry['confidence']['shot_size']
        entry['confidence']['head_yaw'] = 1.5
        errors = ps.validate_entry(entry, 'full_confidence', ['B1', 'B2'])
        self.assertTrue(any('missing shot_size' in e for e in errors))
        self.assertTrue(any('head_yaw must be a number from 0.0 to 1.0' in e for e in errors))

    def test_top5_needs_five_sorted_fields_with_matching_confidence(self):
        entry = {'character_id': 'B1', 'head_yaw': 45, 'eye_gaze': 'right', 'body_pose': 'standing', 'gaze_target': 'B2', 'facial_expression': 'happy',
                 'confidence': {'head_yaw': 0.9, 'eye_gaze': 0.8, 'body_pose': 0.8, 'gaze_target': 0.6, 'facial_expression': 0.5}}
        self.assertEqual(ps.validate_entry(entry, 'top5_confidence', ['B1', 'B2']), [])
        unsorted = {**entry, 'confidence': {**entry['confidence'], 'facial_expression': 0.95}}
        self.assertTrue(any('highest to the lowest' in e for e in ps.validate_entry(unsorted, 'top5_confidence', ['B1', 'B2'])))
        short = {k: v for k, v in entry.items() if k != 'facial_expression'}
        short['confidence'] = {k: v for k, v in short['confidence'].items() if k != 'facial_expression'}
        self.assertTrue(any('exactly 5 fields' in e for e in ps.validate_entry(short, 'top5_confidence', ['B1', 'B2'])))
        mismatch = {**entry, 'confidence': {**entry['confidence']}}
        mismatch['confidence'].pop('gaze_target')
        mismatch['confidence']['shot_size'] = 0.1
        errors = ps.validate_entry(mismatch, 'top5_confidence', ['B1', 'B2'])
        self.assertTrue(any('missing gaze_target' in e for e in errors) and any('unexpected keys shot_size' in e for e in errors))

    def test_answer_count_duplicates_and_empty_frames(self):
        entries = [frame_entry('strict_detailed', 'B1'), frame_entry('strict_detailed', 'B1')]
        errors = ps.validate_answer(entries, 'strict_detailed', ['B1', 'B2'])
        self.assertTrue(any('appears twice' in e for e in errors))
        self.assertEqual(ps.validate_answer([], 'strict_detailed', []), [])
        self.assertTrue(ps.validate_answer([], 'strict_detailed', ['B1']))
        self.assertTrue(ps.validate_answer({}, 'strict_detailed', []))

    def test_video_entries(self):
        names = ['Mai', 'Sakuta']
        snapshot = with_confidence(full(gaze_target='Sakuta'))
        entry = {'character': 'Mai', 'start': snapshot, 'end': None}
        self.assertEqual(ps.validate_video_answer([entry, {'character': 'Sakuta', 'start': None, 'end': None}], 'full_confidence', names), [])
        errors = ps.validate_video_answer([{'character': 'Nobody', 'start': snapshot}, entry], 'full_confidence', names)
        self.assertTrue(any('must be one of the names' in e for e in errors))
        self.assertTrue(any('end is missing' in e for e in errors))
        self.assertTrue(ps.validate_video_answer(None, 'full_confidence', names))

    def test_parse_json_handles_fences_and_garbage(self):
        self.assertEqual(ps.parse_json('```json\n{"a": 1}\n```'), {'a': 1})
        with self.assertRaises(ValueError):
            ps.parse_json('{"a": ')

    def test_format_errors_caps_the_list(self):
        text = ps.format_errors([f'e{i}' for i in range(20)], limit=3)
        self.assertEqual(text.count('\n'), 3)
        self.assertIn('17 more', text)


class MirrorTest(unittest.TestCase):
    def test_mirror_swaps_sides_and_keeps_the_rest(self):
        entry = frame_entry('full_confidence', 'B1', head_yaw=45, body_yaw=-90, eye_gaze='down_right', head_roll='tilt_left', gaze_target='B2')
        mirrored = ps.mirror_entry(entry, {'B1': 'B2', 'B2': 'B1'})
        self.assertEqual((mirrored['head_yaw'], mirrored['body_yaw']), (-45, 90))
        self.assertEqual((mirrored['eye_gaze'], mirrored['head_roll']), ('down_left', 'tilt_right'))
        self.assertEqual((mirrored['character_id'], mirrored['gaze_target']), ('B2', 'B1'))
        for name in ('head_pitch', 'body_pose', 'facial_expression', 'shot_size'):
            self.assertEqual(mirrored[name], entry[name])
        self.assertEqual(ps.mirror_value('head_yaw', 180), 180)
        self.assertEqual(ps.mirror_value('head_yaw', None), None)
        self.assertEqual(ps.mirror_value('eye_gaze', 'unclear'), 'unclear')
        self.assertEqual(ps.mirror_value('eye_gaze', 'center'), 'center')

    def test_circular_difference(self):
        self.assertEqual(ps.circular_difference(180, -180), 0)
        self.assertEqual(ps.circular_difference(170, -170), 20)


class PromptTest(unittest.TestCase):
    def test_blocks_differ_by_approach_and_name_every_field(self):
        blocks = {a: ps.prompt_block(a, 'frame') for a in ps.APPROACHES}
        self.assertEqual(len(set(blocks.values())), 3)
        for approach, text in blocks.items():
            for name in ps.FIELD_ORDER:
                self.assertIn(name, text, (approach, name))
            self.assertIn('pose_detail', text)
        self.assertIn('confidence', blocks['full_confidence'])
        self.assertIn('Do NOT return every field', blocks['top5_confidence'])
        self.assertIn('-135, -90, -45, 0, 45, 90, 135, 180', blocks['strict_detailed'])
        self.assertIn('up_left, up, up_right, left, center, right, down_left, down, down_right', blocks['strict_detailed'])
        self.assertIn('standing, walking, lying, sitting, running, other', blocks['strict_detailed'])
        self.assertNotIn('Do not add confidence', blocks['full_confidence'])
        self.assertIn('start and end', ps.prompt_block('strict_detailed', 'video'))

    def test_json_schema_is_serialisable_and_constrains_ids(self):
        for approach in ps.APPROACHES:
            schema = ps.json_schema(approach, ['B1', 'B2'])
            json.dumps(schema)
            self.assertEqual(schema['items']['properties']['character_id']['enum'], ['B1', 'B2'])
            self.assertEqual((schema['minItems'], schema['maxItems']), (2, 2))
        strict = ps.json_schema('strict_detailed', ['B1'])
        self.assertNotIn('confidence', strict['items']['properties'])
        self.assertEqual(strict['items']['properties']['head_yaw']['enum'][-1], None)
        self.assertEqual(ps.json_schema('full_confidence', [], 'video')['type'], 'array')


if __name__ == '__main__':
    unittest.main()
