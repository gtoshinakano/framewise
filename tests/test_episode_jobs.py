"""The episode's job list is bounded; a cut's own jobs come from /api/cuts/{id}/jobs."""
import json
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from app import db, episode_jobs, media
from app.main import app


class EpisodeJobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','Project')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}')")
            for index in range(3):
                conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES (?,?,?,?,?)", (f'c{index}', 'e', index, index, index + 1))
        self.client = TestClient(app)
        self.count = 0

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def add(self, pipeline, status='done', payload=None):
        """Jobs are created one after the other; created_at has one-second resolution, so give each its own second."""
        self.count += 1
        job_id = f'j{self.count:04d}'
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json,created_at) VALUES (?,?,?,?,?,datetime('2026-01-01 00:00:00', ?))",
                         (job_id, 'e', pipeline, status, json.dumps(payload or {}), f'+{self.count} seconds'))
        return job_id

    def listed(self):
        return [job['id'] for job in self.client.get('/api/episodes/e').json()['jobs']]

    def test_the_episode_lists_active_latest_per_pipeline_prepare_and_recent_jobs(self):
        prepare = self.add('prepare')
        old_failed_subtitles = self.add('subtitles', 'failed')
        old_detect = self.add('character_detect', payload={'cut_id': 'c1'})
        queued = self.add('analyze', 'queued', {'cuts': [{'cut_id': 'c0'}]})
        old = [self.add('regen', payload={'cut_id': 'c2'}) for _ in range(episode_jobs.RECENT + 5)]
        running = self.add('consistency_summary', 'running', {'cut_id': 'c0'})
        newest = [self.add('regen', payload={'cut_id': 'c2'}) for _ in range(episode_jobs.RECENT)]
        listed = self.listed()
        # Newest first, exactly as the full history was.
        self.assertEqual(listed[:episode_jobs.RECENT], list(reversed(newest)))
        self.assertIn(running, listed)
        self.assertIn(queued, listed)
        # The newest job of each pipeline and the newest finished Prepare stay, whatever their age.
        for job_id in (prepare, old_failed_subtitles, old_detect):
            self.assertIn(job_id, listed)
        self.assertNotIn(old[0], listed)
        self.assertEqual(listed, sorted(listed, reverse=True))
        self.assertLess(len(listed), 1 + episode_jobs.RECENT + 8)
        job = next(job for job in self.client.get('/api/episodes/e').json()['jobs'] if job['id'] == queued)
        self.assertEqual([item['id'] for item in job['cuts']], ['c0'])
        self.assertNotIn('payload_json', job)

    def test_a_finished_prepare_is_kept_behind_a_newer_unfinished_one(self):
        done = self.add('prepare', 'done')
        for _ in range(episode_jobs.RECENT + 3):
            self.add('regen', payload={'cut_id': 'c0'})
        running = self.add('prepare', 'running')
        listed = self.listed()
        self.assertIn(done, listed)
        self.assertIn(running, listed)

    def test_cut_jobs_are_the_complete_history_of_that_cut_only(self):
        mine = [self.add('regen', payload={'cut_id': 'c1'}) for _ in range(episode_jobs.RECENT + 10)]
        self.add('regen', payload={'cut_id': 'c2'})
        batch = self.add('analyze', 'queued', {'cuts': [{'cut_id': 'c0'}, {'cut_id': 'c1', 'context': {'cut': {'cut_index': 1, 'title': 'Two', 'start_time': 1, 'end_time': 2}}}]})
        self.add('sequence', payload={'step': 'bible', 'note': 'mentions c1 in text but works on no cut'})
        response = self.client.get('/api/cuts/c1/jobs')
        self.assertEqual(response.status_code, 200)
        jobs = response.json()
        self.assertEqual([job['id'] for job in jobs], [batch, *reversed(mine)])
        self.assertEqual(jobs[0]['cuts'][1], {'id': 'c1', 'cut_index': 1, 'title': 'Two', 'start_time': 1, 'end_time': 2, 'completed': False})
        # A cut that was never worked on has none; an unknown cut is a 404.
        self.assertEqual(self.client.get('/api/cuts/c0/jobs').json()[0]['id'], batch)
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('idle','e',9,9,10)")
        self.assertEqual(self.client.get('/api/cuts/idle/jobs').json(), [])
        self.assertEqual(self.client.get('/api/cuts/none/jobs').status_code, 404)

    def test_cut_jobs_have_the_shape_of_the_episode_jobs(self):
        job_id = self.add('analyze', 'queued', {'cuts': [{'cut_id': 'c0'}], 'stages': ['initial_frame'], 'workflow': 'single_cut_v2'})
        from_episode = next(job for job in self.client.get('/api/episodes/e').json()['jobs'] if job['id'] == job_id)
        self.assertEqual(self.client.get('/api/cuts/c0/jobs').json(), [from_episode])

    def test_last_job_failed_follows_the_newest_job_of_the_cut(self):
        def flag(cut_id):
            return next(cut for cut in self.client.get('/api/episodes/e').json()['cuts'] if cut['id'] == cut_id)['last_job_failed']
        failed = self.add('analyze', 'failed', {'cuts': [{'cut_id': 'c0'}]})
        self.add('regen', 'done', {'cut_id': 'c1'})
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES (?,?,'[]')", (failed, 'c0'))
        self.assertTrue(flag('c0'))
        self.assertFalse(flag('c1'))
        retry = self.add('analyze', 'done', {'cuts': [{'cut_id': 'c0'}]})
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES (?,?,'[]')", (retry, 'c0'))
        self.assertFalse(flag('c0'))


if __name__ == '__main__':
    unittest.main()
