"""Undo and redo of cut boundary edits (cut_history.py)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import db, media, worker
from app.main import app

SETTINGS = {'scene_threshold': 0.3, 'chunk_duration': 60, 'frame_interval': 1, 'proxy_width': 640}


def extract(_proxy, target, timestamp, *_args, **_kwargs):
    target.write_bytes(b'frame')
    return timestamp


class CutHistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration) VALUES ('e','p','Episode','source.mp4','.mp4','ready',?,8)", (json.dumps(SETTINGS),))
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES
                ('c1','e',0,0,2,0,1.9),('c2','e',1,2,4,2,3.9),('c3','e',2,4,8,4,7.9)""")
        (media.episode_dir('p', 'e') / 'proxy.mp4').touch()
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def settle(self):
        """Run the queued preview jobs, as the media worker would."""
        with patch.object(media, 'extract_cut_frame', side_effect=extract):
            while claimed := worker.claim_job(('cutframes',)):
                worker.process(*claimed)

    def active(self):
        return [(row['id'], row['cut_index'], row['start_time'], row['end_time'])
                for row in db.all_rows("SELECT * FROM cuts WHERE episode_id='e' AND active=1 ORDER BY cut_index")]

    def history(self):
        return self.client.get('/api/episodes/e').json()['cut_history']

    def split(self, cut_id='c2', time=3):
        response = self.client.post(f'/api/cuts/{cut_id}/split', json={'time': time})
        self.assertEqual(response.status_code, 200, response.text)
        return db.one("SELECT id FROM cuts WHERE episode_id='e' AND active=1 AND start_time=?", (time,))['id']

    def test_split_is_undone_and_redone_with_the_same_cut(self):
        new_id = self.split()
        self.assertEqual(self.history()['undo']['label'], 'Split cut 2 at 0:03.000')
        self.settle()
        response = self.client.post('/api/episodes/e/cut-edits/undo')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['selected_id'], 'c2')
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 4), ('c3', 2, 4, 8)])
        self.assertIsNone(db.one('SELECT id FROM cuts WHERE id=?', (new_id,)), 'an empty cut made by the split is removed again')
        self.assertEqual(response.json()['cut_history'], {'undo': None, 'redo': {**response.json()['cut_history']['redo'], 'label': 'Split cut 2 at 0:03.000'}})
        job = db.one("SELECT payload_json FROM jobs WHERE episode_id='e' AND status='queued' AND pipeline='cutframes'")
        self.assertEqual(json.loads(job['payload_json'])['cut_ids'], ['c2'])
        self.settle()
        response = self.client.post('/api/episodes/e/cut-edits/redo')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 3), (new_id, 2, 3, 4), ('c3', 3, 4, 8)])
        self.assertEqual(self.history()['undo']['label'], 'Split cut 2 at 0:03.000')
        self.assertIsNone(self.history()['redo'])

    def test_a_split_off_cut_with_work_is_archived_not_deleted(self):
        noted = self.split()
        self.client.put(f'/api/cuts/{noted}', json={'title': '', 'description': '', 'notes': 'Keep me', 'results': {}})
        self.settle()
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        row = db.one('SELECT active,notes,cut_index FROM cuts WHERE id=?', (noted,))
        self.assertEqual((row['active'], row['notes']), (0, 'Keep me'))
        self.assertLess(row['cut_index'], 0)
        self.settle()
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('r','c1','other','Note','x')")
        analysed = self.split('c1', 1)
        with db.connect() as conn:
            conn.execute("UPDATE analysis_results SET cut_id=? WHERE id='r'", (analysed,))
        self.settle()
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(db.one('SELECT active FROM cuts WHERE id=?', (analysed,))['active'], 0)

    def test_merge_delete_and_timing_are_reversed(self):
        self.assertEqual(self.client.post('/api/cuts/c1/merge-next').status_code, 200)
        self.assertEqual(self.history()['undo']['label'], 'Merged cut 1 with cut 2')
        self.settle()
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 4), ('c3', 2, 4, 8)])
        self.settle()
        self.assertEqual(self.client.delete('/api/cuts/c3').status_code, 200)
        self.assertEqual(self.history()['undo']['label'], 'Deleted cut 3')
        self.assertIsNone(self.history()['redo'], 'a new edit ends the redo branch')
        self.settle()
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 4), ('c3', 2, 4, 8)])
        self.settle()
        self.assertEqual(self.client.put('/api/cuts/c2/timing', json={'start_time': 2.5, 'end_time': 4}).status_code, 200)
        self.assertEqual(self.history()['undo']['label'], 'Changed the timing of cut 2')
        self.settle()
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 4), ('c3', 2, 4, 8)])
        self.assertIsNone(db.one("SELECT first_frame_time FROM cuts WHERE id='c1'")['first_frame_time'], 'moved cuts get new frames')
        self.assertIsNotNone(db.one("SELECT first_frame_time FROM cuts WHERE id='c3'")['first_frame_time'])

    def test_undo_refuses_changed_history_and_full_cut_replacement(self):
        self.split()
        self.client.put('/api/cuts/c1/analysis-enabled', json={'analysis_enabled': False})
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline) VALUES ('full','e','hardcuts')")
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 423)
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done' WHERE id='full'")
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        response = self.client.post('/api/episodes/e/cut-edits/undo')
        self.assertEqual(response.status_code, 409)
        self.assertIn('changed', response.json()['detail'])
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=2 WHERE id='c1'")
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 409, 'nothing left to undo')

    def test_split_during_preview_extraction_replaces_preview_and_keeps_original_cut_jobs(self):
        old_right = self.split()
        analysis = self.client.post('/api/cuts/c2/analysis-stages/run', json={'stage': 'initial_frame'}).json()['job_id']
        preview = worker.claim_job(('cutframes',))
        spawned = []

        def edit_during_extract(*args, **kwargs):
            if not spawned:
                spawned.append(self.split('c2', 2.5))
            return extract(*args, **kwargs)

        with patch.object(media, 'extract_cut_frame', side_effect=edit_during_extract):
            worker.process(*preview)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (preview[0],))['status'], 'cancelled')
        self.assertIsNone(db.one("SELECT first_frame_time FROM cuts WHERE id='c2'")['first_frame_time'])
        worker.release_lease(*preview)
        self.settle()
        cut = db.one("SELECT end_time,last_frame_time FROM cuts WHERE id='c2'")
        self.assertEqual(cut['end_time'], 2.5)
        self.assertLess(cut['last_frame_time'], 2.5)
        self.assertIsNotNone(db.one('SELECT first_frame_time FROM cuts WHERE id=?', (old_right,))['first_frame_time'])
        job = db.one('SELECT status,payload_json FROM jobs WHERE id=?', (analysis,))
        self.assertEqual(job['status'], 'queued')
        self.assertEqual([item['cut_id'] for item in json.loads(job['payload_json'])['cuts']], ['c2'])
        self.assertIsNone(db.one('SELECT id FROM analysis_results WHERE cut_id=?', (spawned[0],)))

    def test_merge_cancels_jobs_for_the_archived_cut_and_marks_the_survivors_results_stale(self):
        stage = '/api/cuts/c2/analysis-stages/run'
        running = self.client.post(stage, json={'stage': 'initial_frame'}).json()['job_id']
        self.assertEqual((claim := worker.claim_job(('analyze',)))[0], running)
        queued = self.client.post(stage, json={'stage': 'keyframe_analysis'}).json()['job_id']
        survivor = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'initial_frame'}).json()['job_id']
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('r','c1','cut_description','Cut summary','Before merge','model')")
        response = self.client.post('/api/cuts/c1/merge-next')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(db.one("SELECT active FROM cuts WHERE id='c2'")['active'], 0)
        for job_id in (running, queued):
            job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
            self.assertEqual(job, {'status': 'cancelled', 'error': 'Cancelled because the target cut was archived'})
        with self.assertRaises(worker.LeaseLost), db.connect() as conn:
            worker.require_lease(conn, *claim)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (survivor,))['status'], 'queued')
        self.assertEqual(db.one("SELECT stale_reason FROM analysis_results WHERE id='r'")['stale_reason'], 'Cut timing changed')

    def test_detection_inside_a_cut_and_frame_steps_are_recorded_and_full_redetection_clears(self):
        self.assertEqual(self.client.post('/api/cuts/c3/detect-hardcuts', json={'scene_threshold': 0.3}).status_code, 200)
        with patch.object(media, 'detect_cuts_in_range', return_value=[(4, 6), (6, 8)]), patch.object(media, 'extract_cut_frame', side_effect=extract):
            worker.process(*worker.claim_job(('hardcuts',)))
        self.assertEqual(self.history()['undo']['label'], 'Detected 2 cuts inside cut 3')
        found = db.one("SELECT id FROM cuts WHERE episode_id='e' AND active=1 AND start_time=6")['id']
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 4), ('c3', 2, 4, 8)])
        self.assertIsNone(db.one('SELECT id FROM cuts WHERE id=?', (found,)))
        self.settle()

        self.assertEqual(self.client.post('/api/cuts/c1/boundary-frame', json={'direction': 'later'}).status_code, 200)
        with patch.object(media, 'step_frame_boundary', return_value=2.04), patch.object(media, 'extract_cut_frame', side_effect=extract):
            worker.process(*worker.claim_job(('cutframes',)))
        self.assertEqual(self.history()['undo']['label'], 'Moved the end of cut 1 one frame later')
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.active(), [('c1', 0, 0, 2), ('c2', 1, 2, 4), ('c3', 2, 4, 8)])
        self.settle()

        self.split()
        self.settle()
        worker.enqueue('e', 'hardcuts')
        with patch.object(media, 'probe', return_value={'duration': 8, 'width': 640, 'height': 360, 'has_audio': False}), \
                patch.object(media, 'detect_cuts', return_value=[(0, 4), (4, 8)]), patch.object(media, 'extract_cut_frame', side_effect=extract):
            worker.process(*worker.claim_job(('hardcuts',)))
        self.assertEqual(self.history(), {'undo': None, 'redo': None})

    def test_detect_within_reuses_request_and_restarts_after_a_split(self):
        url = '/api/cuts/c3/detect-hardcuts'
        body = {'scene_threshold': 0.3}
        first = self.client.post(url, json=body).json()['job_id']
        self.assertEqual(self.client.post(url, json=body).json()['job_id'], first)
        claim = worker.claim_job(('hardcuts',))

        def split_during_scan(_proxy, start, end, _threshold):
            self.assertEqual((start, end), (4, 8))
            self.split('c3', 7)
            return [(4, 6), (6, 8)]

        with patch.object(media, 'detect_cuts_in_range', side_effect=split_during_scan):
            worker.process(*claim)
        worker.release_lease(*claim)
        saved = db.one('SELECT status,stage,attempt_count,not_before FROM jobs WHERE id=?', (first,))
        self.assertEqual((saved['status'], saved['stage'], saved['attempt_count']),
                         ('queued', 'Inputs changed, restarting', 0))
        self.assertIsNone(worker.claim_job(('hardcuts',)))
        with db.connect() as conn:
            conn.execute('UPDATE jobs SET not_before=0 WHERE id=?', (first,))
        with patch.object(media, 'detect_cuts_in_range', return_value=[(4, 6), (6, 7)]) as detect, \
             patch.object(media, 'extract_cut_frame', side_effect=extract):
            worker.process(*worker.claim_job(('hardcuts',)))
        self.assertEqual(detect.call_args.args[1:3], (4, 7))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (first,))['status'], 'done')
        self.assertEqual(self.active()[2], ('c3', 2, 4, 6))


if __name__ == '__main__':
    unittest.main()
