import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app.frame_model import describe_frame


class FrameAdapterTest(unittest.TestCase):
    def test_validates_json_and_sends_only_subtitle_locations(self):
        result = {'composition': 'wide', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'visible_characters': [], 'other_visual_details': '', 'uncertainties': ''}
        response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}).encode()
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'frame.jpg'
            image.write_bytes(b'image')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(response)) as request:
                self.assertEqual(describe_frame('http://localhost:1', image, [[1, 2, 3, 4]], {'character_detections': {'labels_confirmed_by_queue': True}}), result)
                body = json.loads(request.call_args.args[0].data)
                self.assertIn('[[1, 2, 3, 4]]', body['messages'][0]['content'][1]['text'])
                self.assertIn('orientation relative to the camera', body['messages'][0]['content'][1]['text'])
                self.assertIn('walking is unconfirmed', body['messages'][0]['content'][1]['text'])
                self.assertIn('user-verified identity assignment', body['messages'][0]['content'][1]['text'])
                self.assertIn('labels_confirmed_by_queue', body['messages'][0]['content'][1]['text'])
                self.assertIn('data:image/jpeg;base64,', body['messages'][0]['content'][0]['image_url']['url'])

    def test_prompt_states_the_single_subtitle_language(self):
        result = {'composition': '', 'setting': '', 'lighting': '', 'camera': '', 'visible_characters': [], 'other_visual_details': '', 'uncertainties': ''}
        response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}).encode()
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'frame.jpg'
            image.write_bytes(b'image')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(response)) as request:
                describe_frame('http://localhost:1', image, [[0, 500, 1280, 700]], subtitle_language='Portuguese')
                text = json.loads(request.call_args.args[0].data)['messages'][0]['content'][1]['text']
                self.assertIn('every burned-in subtitle in this video is in Portuguese; that is certain', text)
                self.assertIn('[[0, 500, 1280, 700]]', text)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(response)) as request:
                describe_frame('http://localhost:1', image)
                self.assertNotIn('is certain', json.loads(request.call_args.args[0].data)['messages'][0]['content'][1]['text'])

    def test_reference_comparison_sends_target_and_labeled_reference_images(self):
        from app.frame_model import match_character
        result = {'character_id': 'hero', 'confidence': 'high', 'reason': 'matching hair'}
        response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}).encode()
        with patch('urllib.request.urlopen', return_value=io.BytesIO(response)) as request:
            self.assertEqual(match_character('http://localhost:1', b'target', [{'character_context_id': 'hero', 'title': 'Hero', 'image': b'reference'}]), result)
            content = json.loads(request.call_args.args[0].data)['messages'][0]['content']
            self.assertEqual(sum(item['type'] == 'image_url' for item in content), 2)
            self.assertIn('hero', json.dumps(content))

    def test_frame_analysis_sends_reference_separately(self):
        result = {'composition': '', 'setting': '', 'lighting': '', 'camera': '', 'visible_characters': [], 'other_visual_details': '', 'uncertainties': ''}
        response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}).encode()
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'frame.jpg'
            image.write_bytes(b'target')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(response)) as request:
                describe_frame('http://localhost:1', image, reference_images=[{'character_context_id': 'hero', 'title': 'Hero', 'image': b'ref'}])
                content = json.loads(request.call_args.args[0].data)['messages'][0]['content']
                self.assertEqual(sum(item['type'] == 'image_url' for item in content), 2)
                self.assertIn('first image is the target frame', json.dumps(content))
