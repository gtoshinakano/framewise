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

    def test_undo_is_refused_when_the_cuts_changed_or_media_work_runs(self):
        self.split()
        # A queued preview job does not block undo; the analysis switch is not part of the timeline.
        self.client.put('/api/cuts/c1/analysis-enabled', json={'analysis_enabled': False})
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running' WHERE pipeline='cutframes'")
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 423)
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='queued' WHERE pipeline='cutframes'")
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        response = self.client.post('/api/episodes/e/cut-edits/undo')
        self.assertEqual(response.status_code, 409)
        self.assertIn('changed', response.json()['detail'])
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=2 WHERE id='c1'")
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 200)
        self.assertEqual(self.client.post('/api/episodes/e/cut-edits/undo').status_code, 409, 'nothing left to undo')

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


if __name__ == '__main__':
    unittest.main()
