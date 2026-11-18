"""The Cut summary gets the first frame as a picture, only for its Cinematic Camera section (2026-11-19)."""
import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import brief_picture, db, inference_worker, media, sequence_evidence, worker
from app.assistant import images
from app.main import app

CAMERA = ('### Cinematic Camera\n- Camera body: ARRI Alexa Mini LF\n- Lens: Cooke S4/i spherical primes\n- Focal length: 40 mm\n- Aperture: T2.8\n'
          '- Lighting: soft key from camera left\n- Color grading: muted shadows, warm highlights\n- Image texture: fine grain')
FIVE = ('### Scene\nA store.\n### Opening Frame\nA person.\n### Performance Beats\n[0.0–1.0] Camera: static.\n### Acting Notes\nThe person waits.\n'
        '### Continuity & Constraints\nStatic camera.')
SIX = FIVE + '\n' + CAMERA
JPEG = b'\xff\xd8\xff\xe0first frame'


def reply(content):
    return io.BytesIO(json.dumps({'model': 'test/actual', 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode())


class BriefPictureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','Synopsis')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,2)")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def queue(self, **body):
        response = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate', **body})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['job_id']

    def run_regen(self, replies, vision=True, picture=JPEG, slot=None):
        """Run the queued job against scripted replies; returns the requests sent and the endpoint_vision mock."""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(text) for text in replies]) as call, \
                patch.object(inference_worker, 'endpoint_vision', return_value=vision) as seeing, \
                patch.object(images, 'picture', side_effect=picture if isinstance(picture, Exception) else None, return_value=None if isinstance(picture, Exception) else picture), \
                patch.object(sequence_evidence, 'slot_tokens', return_value=slot or 60000):
            inference_worker.process_regen(*worker.claim_job(('regen',)))
        return [json.loads(item.args[0].data) for item in call.call_args_list], seeing

    def saved(self, job_id):
        row = db.one('SELECT content,prompt,settings_json FROM analysis_results WHERE job_id=?', (job_id,))
        return (row['content'], row['prompt'], json.loads(row['settings_json'])) if row else None

    def test_a_model_that_can_see_gets_the_first_frame_and_the_note(self):
        job_id = self.queue()
        sent, _ = self.run_regen([SIX])
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        system = sent[0]['messages'][0]['content']
        self.assertEqual(system, inference_worker.PRODUCTION_BRIEF_SYSTEM + brief_picture.PICTURE_NOTE)
        self.assertIn('only to decide the Cinematic Camera section', system)
        parts = sent[0]['messages'][1]['content']
        self.assertEqual([part['type'] for part in parts], ['text', 'text', 'image_url'])
        self.assertEqual(parts[1]['text'], 'Picture ' + brief_picture.CAPTION)
        self.assertEqual(parts[2]['image_url']['url'], 'data:image/jpeg;base64,' + base64.b64encode(JPEG).decode())
        self.assertEqual(json.loads(parts[0]['text'])['mode'], 'regenerate')  # the evidence is the same JSON as without a picture
        content, prompt, settings = self.saved(job_id)
        self.assertEqual(content, SIX)
        self.assertEqual(prompt, system)  # provenance keeps the instructions that were really sent
        self.assertEqual((settings['picture'], settings['usage']['pictures']), ('first frame', 1))
        self.assertNotIn('picture_skipped', settings)

    def test_a_text_only_endpoint_gets_the_same_request_as_before(self):
        job_id = self.queue()
        sent, seeing = self.run_regen([SIX], vision=False)
        self.assertEqual(sent[0]['messages'][0]['content'], inference_worker.PRODUCTION_BRIEF_SYSTEM)
        self.assertIsInstance(sent[0]['messages'][1]['content'], str)
        content, _, settings = self.saved(job_id)
        self.assertEqual(content, SIX)  # the section is still written, from the text evidence
        self.assertEqual(settings['picture_skipped'], 'the text endpoint cannot read images')
        self.assertNotIn('picture', settings)
        seeing.assert_called_once()

    def test_no_picture_when_the_request_leaves_no_room_for_it(self):
        job_id = self.queue()
        sent, _ = self.run_regen([SIX], slot=7000)
        self.assertIsInstance(sent[0]['messages'][1]['content'], str)
        self.assertEqual(self.saved(job_id)[2]['picture_skipped'], 'the request leaves no room for a picture')
        self.assertTrue(brief_picture.has_room('system', '{}', 60000))
        self.assertFalse(brief_picture.has_room('system', '{}', brief_picture.ANSWER_TOKENS + brief_picture.SPARE_TOKENS + images.TOKENS_PER_PICTURE - 1))

    def test_a_frame_that_cannot_be_cut_never_fails_the_job(self):
        job_id = self.queue()
        sent, _ = self.run_regen([SIX], picture=images.FrameError('The original video of this cut is not available'))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        self.assertIsInstance(sent[0]['messages'][1]['content'], str)
        self.assertEqual(sent[0]['messages'][0]['content'], inference_worker.PRODUCTION_BRIEF_SYSTEM)
        self.assertTrue(self.saved(job_id)[2]['picture_skipped'].startswith('the first frame could not be cut: The original video'))

    def test_an_edited_request_is_sent_exactly_with_no_picture(self):
        preview = self.client.get('/api/cuts/c1/summary-request-preview').json()
        request = {**preview['request'], 'system': 'Custom instruction', 'input': {'added': ['value']}}
        job_id = self.queue(preview_checked=True, preview_base_id=preview['base_id'], request_override=request)
        sent, seeing = self.run_regen([FIVE])
        self.assertEqual(sent[0]['messages'][0]['content'], 'Custom instruction')
        self.assertEqual(json.loads(sent[0]['messages'][1]['content']), {'added': ['value']})
        seeing.assert_not_called()
        self.assertEqual(self.saved(job_id)[0], FIVE)  # a request without the section's instructions is not forced to produce it

    def test_an_answer_without_the_camera_section_is_sent_back_naming_it(self):
        job_id = self.queue()
        sent, _ = self.run_regen([FIVE, SIX])
        self.assertEqual(len(sent), 2)
        repair = sent[1]['messages'][-1]['content']
        self.assertIn('Cinematic Camera', repair)
        self.assertEqual(sent[1]['messages'][1], sent[0]['messages'][1])  # the picture stays on the retry
        self.assertEqual(self.saved(job_id)[0], SIX)

    def test_an_answer_with_an_undecided_field_is_sent_back_naming_it(self):
        job_id = self.queue()
        sent, _ = self.run_regen([SIX.replace('Aperture: T2.8', 'Aperture: unknown'), SIX])
        self.assertIn('Aperture has no decision', sent[1]['messages'][-1]['content'])
        self.assertEqual(self.saved(job_id)[0], SIX)


if __name__ == '__main__':
    unittest.main()
