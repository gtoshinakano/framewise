import json
import unittest
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, worker


class CancelQueueTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_cancel_all_queued_across_episodes_and_preserve_running(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e2','p','E2','a.mp4','.mp4','queued','{}')")
            for job_id, episode, pipeline, status in [
                ('running', 'e', 'analyze', 'running'), ('detection', 'e', 'character_detect', 'queued'),
                ('prepare', 'e', 'prepare', 'queued'), ('voice', 'e2', 'voice_sample', 'queued'),
                ('export', 'e2', 'brief_export', 'queued'), ('finished', 'e2', 'regen', 'done')]:
                conn.execute('INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES (?,?,?,?,?)',
                             (job_id, episode, pipeline, status, json.dumps({})))
        with patch.object(worker, 'cleanup_voice_upload') as cleanup:
            response = self.client.post('/api/jobs/cancel-queued')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'cancelled': 3})
        self.assertEqual({call.args[0]['id'] for call in cleanup.call_args_list}, {'detection', 'prepare', 'voice'})
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='running'")['status'], 'running')
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='finished'")['status'], 'done')
        for job_id in ('detection', 'prepare', 'voice'):
            job = db.one('SELECT status,stage,finished_at FROM jobs WHERE id=?', (job_id,))
            self.assertEqual((job['status'], job['stage']), ('failed', 'Cancelled'))
            self.assertIsNotNone(job['finished_at'])
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")['status'], 'processing')
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e2'")['status'], 'ready')
        self.assertEqual(self.client.post('/api/jobs/cancel-queued').json(), {'cancelled': 0})
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id='export'")['status'], 'queued')
        self.assertIsNone(worker.claim_job(('character_detect', 'prepare', 'voice_sample')))
        self.assertEqual(worker.claim_job(('brief_export',))[0], 'export')

    def test_cancelling_voice_sample_removes_temporary_upload(self):
        import uuid
        from app import media
        upload_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
        folder = media.episode_dir('p', 'e') / 'voice-uploads'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / upload_id).write_bytes(b'temporary audio')
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES (?,'e','voice_sample',?)", (job_id, json.dumps({'upload_id': upload_id})))
        self.assertEqual(self.client.post('/api/jobs/cancel-queued').json()['cancelled'], 1)
        self.assertFalse((folder / upload_id).exists())
