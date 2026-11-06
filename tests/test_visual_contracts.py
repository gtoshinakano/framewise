import json
import unittest
from app.visual_contracts import validate_frame, validate_video, SUBTITLE_INSTRUCTION


class VisualContractsTest(unittest.TestCase):
    def test_frame_requires_structured_visual_fields(self):
        valid = {key: 'unknown' for key in ('composition', 'setting', 'lighting', 'camera', 'other_visual_details', 'uncertainties')}
        valid['visible_characters'] = []
        self.assertEqual(validate_frame('```json\n' + json.dumps(valid) + '\n```'), valid)
        valid['visible_characters'] = ['invented']
        with self.assertRaises(ValueError): validate_frame(json.dumps(valid))
        self.assertIn('Portuguese or English', SUBTITLE_INSTRUCTION)

    def test_video_accepts_overlapping_subject_events_without_reordering(self):
        events = [
            {'start_seconds': 0, 'end_seconds': .6, 'subject': 'Rio'},
            {'start_seconds': .6, 'end_seconds': 1.6, 'subject': 'Rio'},
            {'start_seconds': 1.6, 'end_seconds': 3.6, 'subject': 'Rio'},
            {'start_seconds': 0, 'end_seconds': 3.6, 'subject': 'Sakuta'},
        ]
        result = validate_video(json.dumps({'overview': 'scene', 'events': events}), 3.6, 10)
        self.assertEqual([event['subject'] for event in result['events']], ['Rio', 'Rio', 'Rio', 'Sakuta'])
        self.assertEqual(result['events'][-1]['episode_start_seconds'], 10)
        self.assertEqual(result['events'][-1]['episode_end_seconds'], 13.6)

    def test_video_preserves_unusual_json_without_inventing_times(self):
        payload = {'events': [{'start_seconds': 4, 'end_seconds': 5}, {'start_seconds': 'unknown'}, 'unstructured'], 'other': [1, 2]}
        self.assertEqual(validate_video(json.dumps(payload), 3, 10), payload)
        self.assertEqual(validate_video('{"events": null, "extra": true}', 3, 10), {'events': None, 'extra': True})
        for valid, expected in [('[]', []), ('null', None), ('"text"', 'text'), ('42', 42)]:
            self.assertEqual(validate_video(valid, 3, 10), expected)
        for invalid in ('not json', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_video(invalid, 3, 10)

    def test_video_preserves_original_rounded_timestamp(self):
        start, end = 161.035458, 167.792208
        event = {'start_seconds': 2.1, 'end_seconds': 6.75675}
        result = validate_video(json.dumps({'events': [event]}), end - start, start)
        self.assertEqual(result['events'][0]['end_seconds'], 6.75675)
        self.assertEqual(result['events'][0]['episode_end_seconds'], end)
