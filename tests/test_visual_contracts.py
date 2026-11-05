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

    def test_video_time_bounds_and_episode_time(self):
        event = {'start_seconds': 0.5, 'end_seconds': 1.5, 'subject': 'person', 'visual_action': 'turns', 'camera': 'fixed', 'uncertainty': 'unknown'}
        result = validate_video(json.dumps({'overview': '', 'events': [event], 'uncertainties': ''}), 2, 10)
        self.assertEqual(result['events'][0]['episode_end_seconds'], 11.5)
        event['end_seconds'] = 3
        with self.assertRaises(ValueError): validate_video(json.dumps({'overview': '', 'events': [event], 'uncertainties': ''}), 2, 10)
