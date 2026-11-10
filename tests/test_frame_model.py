import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from app.frame_model import describe_frame

PEOPLE = [{'ref': 'B1', 'detection_id': 'd1', 'character_context_id': 'hero', 'name': 'Hero', 'box': [10, 20, 300, 900]}]
ANSWER = {'composition': 'wide', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'visible_characters': [], 'background_people': 'none', 'notes': 'none'}


def reply(result):
    return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}], 'usage': {'completion_tokens': 5}}).encode()


class FrameAdapterTest(unittest.TestCase):
    def call(self, answer=ANSWER, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'frame.jpg'
            image.write_bytes(b'image')
            with patch('urllib.request.urlopen', return_value=io.BytesIO(reply(answer))) as request:
                result = describe_frame('http://localhost:1', image, **kwargs)
        body = json.loads(request.call_args.args[0].data)
        return result, body, body['messages'][1]['content'][1]['text']

    def test_sends_the_fixed_instruction_first_and_a_schema_for_the_listed_boxes(self):
        answer = {**ANSWER, 'visible_characters': [{'box': 'B1', 'appearance': 'red coat', 'position': 'left', 'pose': 'sitting', 'expression': 'calm', 'action': 'reading'}]}
        result, body, text = self.call(answer, subtitle_regions=[[1, 2, 3, 4]], people=PEOPLE)
        self.assertEqual(body['messages'][0]['role'], 'system')
        self.assertIn('ONE still frame', body['messages'][0]['content'])
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertEqual(body['response_format']['json_schema']['schema']['properties']['visible_characters']['items']['properties']['box']['enum'], ['B1'])
        self.assertLessEqual(body['max_tokens'], 3072)
        self.assertIn('B1: Hero, box [10, 20, 300, 900]', text)
        self.assertIn('[[1, 2, 3, 4]]', text)
        self.assertNotIn('d1', text)  # the model sees handles, never database IDs
        self.assertEqual(result['visible_characters'][0]['character_context_id'], 'hero')
        self.assertNotIn('detection_id', result['visible_characters'][0])
        self.assertEqual(result['visible_characters'][0]['character'], 'Hero')
        self.assertIn('data:image/jpeg;base64,', body['messages'][1]['content'][0]['image_url']['url'])

    def test_without_marked_characters_no_boxes_are_sent(self):
        result, body, text = self.call()
        self.assertIn('Marked characters: none', text)
        self.assertEqual(body['response_format']['json_schema']['schema']['properties']['visible_characters']['maxItems'], 0)
        self.assertEqual(result['visible_characters'], [])

    def test_series_notes_are_sent_without_detection_data(self):
        context = {'selected_context': [{'kind': 'world', 'title': 'School', 'description': 'Calm student school'}, {'kind': 'character', 'title': 'Hero', 'description': 'Red coat, long hair'}], 'additional_context': 'Episode 3',
                   'character_detections': {'detections': [{'id': 'secret-box'}]}, 'detection_note': 'note'}
        _, _, text = self.call(character_context=context)
        self.assertIn('world School: Calm student school', text)
        self.assertNotIn('Red coat', text)  # character notes carry outfits the model would copy
        self.assertIn('Episode 3', text)
        self.assertNotIn('secret-box', text)

    def test_last_frame_gets_the_changes_prompt_and_the_first_frame_notes(self):
        first = {**ANSWER, 'mode': 'full', 'frame_time': 1.0, 'visible_characters': [{'character': 'Hero', 'character_context_id': 'hero', 
                 'appearance': 'red coat', 'position': 'left', 'pose': 'sitting', 'expression': 'calm', 'action': 'reading'}]}
        answer = {**ANSWER, 'composition': 'unchanged', 'visible_characters': [{'box': 'B1', 'appearance': 'unchanged', 'position': 'unchanged', 'pose': 'standing', 'expression': 'unchanged', 'action': 'unchanged'}]}
        result, body, text = self.call(answer, people=[{**PEOPLE[0], 'new': False}], frame_label='last frame', previous=first)
        self.assertIn('CHANGES', body['messages'][0]['content'])
        self.assertIn('red coat', text)
        self.assertNotIn('frame_time', text)
        self.assertNotIn('d0', text)
        self.assertIn('[was in the first frame]', text)
        self.assertEqual(result['mode'], 'changes')
        self.assertEqual(result['visible_characters'][0]['pose'], 'standing')
        self.assertEqual(result['visible_characters'][0]['appearance'], 'unchanged')

    def test_prompt_states_the_single_subtitle_language(self):
        _, _, text = self.call(subtitle_regions=[[0, 500, 1280, 700]], subtitle_language='Portuguese')
        self.assertIn('every burned-in subtitle in this video is in Portuguese; that is certain', text)
        self.assertIn('[[0, 500, 1280, 700]]', text)
        _, _, text = self.call()
        self.assertNotIn('is certain', text)

    def test_a_server_that_rejects_the_schema_is_asked_for_plain_json(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'frame.jpg'
            image.write_bytes(b'image')
            rejected = urllib.error.HTTPError('http://x', 400, 'bad', {}, io.BytesIO(b'{}'))
            with patch('urllib.request.urlopen', side_effect=[rejected, io.BytesIO(reply(ANSWER))]) as request:
                self.assertEqual(describe_frame('http://localhost:1', image)['mode'], 'full')
            self.assertEqual(json.loads(request.call_args.args[0].data)['response_format'], {'type': 'json_object'})

    def test_truncated_answer_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'frame.jpg'
            image.write_bytes(b'image')
            cut = json.dumps({'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]}).encode()
            with patch('urllib.request.urlopen', return_value=io.BytesIO(cut)):
                with self.assertRaisesRegex(RuntimeError, 'truncated'):
                    describe_frame('http://localhost:1', image)

    def test_reference_comparison_sends_target_and_labeled_reference_images(self):
        from app.frame_model import match_character
        result = {'character_id': 'hero', 'confidence': 'high', 'reason': 'matching hair'}
        response = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}]}).encode()
        with patch('urllib.request.urlopen', return_value=io.BytesIO(response)) as request:
            self.assertEqual(match_character('http://localhost:1', b'target', [{'character_context_id': 'hero', 'title': 'Hero', 'image': b'reference'}]), result)
            content = json.loads(request.call_args.args[0].data)['messages'][0]['content']
            self.assertEqual(sum(item['type'] == 'image_url' for item in content), 2)
            self.assertIn('hero', json.dumps(content))
