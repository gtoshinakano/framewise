"""Autopilot M2: the missed-cut checker (scores, verdicts, the job, splits, questions, answers and undo)."""
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_log, cut_history, db, media, missed_cuts, worker
from app.main import app


class ScoringTest(unittest.TestCase):
    def test_parse_scores_reads_the_metadata_lines(self):
        text = ('frame:0    pts:0       pts_time:0\nlavfi.scene_score=0.000000\n'
                'frame:1    pts:1001    pts_time:0.041708\nlavfi.scene_score=0.231000\n')
        self.assertEqual(missed_cuts.parse_scores(text), ([0.0, 0.041708], [0.0, 0.231]))

    def test_candidates_are_inner_local_peaks_with_their_isolation(self):
        times = [i / 24 for i in range(96)]
        scores = [0.01] * 96
        scores[48] = 0.25          # a lone spike: a cut
        scores[70:76] = [0.12, 0.14, 0.16, 0.15, 0.13, 0.12]   # sustained: motion
        scores[2] = 0.5            # inside EDGE of the cut start: never a candidate
        found = missed_cuts.candidates(times, scores, 0.0, 4.0)
        self.assertEqual([round(item['time'] * 24) for item in found], [48, 72])
        self.assertGreater(found[0]['isolation'], 10)
        self.assertLess(found[1]['isolation'], 1.5)

    def test_verdicts(self):
        peak = {'score': 0.25, 'isolation': 8.0}
        clean = {'camera': {'move': 'unclear', 'inliers': 5, 'ratio': 0.05}, 'correlation': 0.6, 'flash': False}
        self.assertEqual(missed_cuts.verdict(peak, clean)[0], 'apply')
        same_shot = {**clean, 'camera': {'move': 'pan_left', 'inliers': 400, 'ratio': 0.8}}
        self.assertEqual(missed_cuts.verdict(peak, same_shot)[0], 'reject')
        self.assertEqual(missed_cuts.verdict(peak, {**clean, 'flash': True})[0], 'grey')
        self.assertEqual(missed_cuts.verdict({'score': 0.12, 'isolation': 8.0}, clean)[0], 'grey')
        self.assertEqual(missed_cuts.verdict({'score': 0.25, 'isolation': 1.2}, clean)[0], 'reject')
        self.assertEqual(missed_cuts.verdict(peak, {**clean, 'correlation': 0.99})[0], 'grey')

    def test_a_split_next_to_a_boundary_is_asked_and_a_misplaced_boundary_goes_to_a_person(self):
        times = [i / 24 for i in range(240)]
        scores = [0.01] * 240
        scores[24] = 0.0     # the cut starts at 1.0 s where the picture does not change (a hand-placed boundary)
        scores[32] = 0.3     # the real cut, 8 frames later
        scores[120] = 0.3    # a missed cut in the middle
        picture = {'camera': {'move': 'unclear', 'inliers': 3, 'ratio': 0.1}, 'correlation': 0.5, 'flash': False}
        with mock.patch.object(missed_cuts, 'frames_near', return_value=[]), mock.patch.object(missed_cuts, 'picture_features', return_value=picture):
            result = missed_cuts.check_cut(Path('x'), times, scores, 1.0, 9.0)
        self.assertEqual(result['split_at'], [5.0])
        self.assertEqual(result['grey'], [round(32 / 24, 6)])
        sliver = next(item for item in result['peaks'] if item['time'] == round(32 / 24, 6))
        self.assertTrue(sliver['misplaced_edge'])
        self.assertIn('misplaced', sliver['why'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is needed')
    def test_real_video_scores_frames_and_strip(self):
        with tempfile.TemporaryDirectory() as temp:
            old = db.DATA_DIR
            db.DATA_DIR = Path(temp)
            try:
                clip = Path(temp) / 'proxy.mp4'
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=24:duration=1',
                                '-f', 'lavfi', '-i', 'smptebars=size=320x180:rate=24:duration=1', '-filter_complex', '[0][1]concat=n=2:v=1[v]',
                                '-map', '[v]', '-pix_fmt', 'yuv420p', str(clip)], check=True)
                times, scores = missed_cuts.scene_scores(clip)
                self.assertEqual(len(times), len(scores))
                self.assertGreaterEqual(len(times), 47)
                cut = max(range(len(scores)), key=scores.__getitem__)
                self.assertAlmostEqual(times[cut], 1.0, delta=0.05)
                self.assertTrue(list(missed_cuts.cache_dir().glob('*.json')))  # cached for the next run
                frames = missed_cuts.frames_near(clip, times[cut])
                picture = missed_cuts.picture_features(frames, times[cut - 1], times[cut])
                self.assertLess(picture['correlation'], 0.97)
                self.assertTrue(missed_cuts.strip_jpeg(clip, times[cut]).startswith(b'\xff\xd8'))
            finally:
                db.DATA_DIR = old


def peak(moment, score=0.25, isolation=9.0, verdict='apply'):
    return {'time': moment, 'score': score, 'isolation': isolation, 'verdict': verdict, 'why': 'test', 'frame': 0,
            'picture': {'camera': {'move': 'unclear'}, 'correlation': 0.6, 'flash': False}}


class JobTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration)
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}',12)""")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,analysis_enabled) VALUES
                ('c1','e',0,0,4,1),('c2','e',1,4,8,1),('c3','e',2,8,12,0)""")
        proxy = media.episode_dir('p', 'e') / 'proxy.mp4'
        proxy.parent.mkdir(parents=True, exist_ok=True)
        proxy.write_bytes(b'not a real video')
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def active(self):
        with db.connect() as conn:
            return [(row['id'], row['cut_index'], row['start_time'], row['end_time'])
                    for row in conn.execute("SELECT * FROM cuts WHERE episode_id='e' AND active=1 ORDER BY cut_index")]

    def run_job(self, checks, body=None):
        """Queue a check and run it with ``checks`` ({cut start: check_cut result}) standing in for the pictures."""
        queued = self.client.post('/api/episodes/e/missed-cuts', json=body or {}).json()
        job_id = queued['job_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running',lease_owner='test',lease_until=? WHERE id=?", (time.time() + 60, job_id))

        def fake_check(proxy, times, scores, start, end, size=None, skip=()):
            result = checks.get(start, {'verdict': 'clean', 'split_at': [], 'grey': [], 'peaks': [], 'version': missed_cuts.VERSION})
            return {**result, 'peaks': result.get('peaks') or [peak(t) for t in [*result['split_at'], *result['grey']]]}
        with mock.patch.object(missed_cuts, 'scene_scores', return_value=([0.0], [0.0])), \
                mock.patch.object(missed_cuts, '_dimensions', return_value=(320, 180)), \
                mock.patch.object(missed_cuts, 'check_cut', side_effect=fake_check) as checked:
            missed_cuts.process(job_id, 'test')
        with db.connect() as conn:
            job = conn.execute('SELECT status,stage,error FROM jobs WHERE id=?', (job_id,)).fetchone()
        self.assertEqual(job['status'], 'done', job['error'])
        return job_id, checked

    def test_splits_the_sure_peaks_asks_about_the_rest_and_skips_cuts_with_analysis_off(self):
        _, checked = self.run_job({0: {'verdict': 'apply', 'split_at': [1.5, 2.5], 'grey': []},
                                   4: {'verdict': 'grey', 'split_at': [], 'grey': [6.0]}})
        self.assertEqual([call.args[3] for call in checked.call_args_list], [0, 4])  # c3 has analysis off
        cuts = self.active()
        self.assertEqual([(start, end) for _, _, start, end in cuts], [(0, 1.5), (1.5, 2.5), (2.5, 4), (4, 8), (8, 12)])
        self.assertEqual([index for _, index, _, _ in cuts], [0, 1, 2, 3, 4])
        with db.connect() as conn:
            decisions = conn.execute("SELECT * FROM agent_decisions WHERE kind='missed_cuts' ORDER BY rowid").fetchall()
            edits = conn.execute("SELECT action,label FROM cut_edits ORDER BY rowid").fetchall()
            previews = conn.execute("SELECT payload_json FROM jobs WHERE pipeline='cutframes' AND status='queued'").fetchall()
            checks = {row['cut_id']: row['verdict'] for row in conn.execute('SELECT * FROM missed_cut_checks')}
        self.assertEqual([row['tier'] for row in decisions], ['rules', 'rules'])
        self.assertEqual([row['action'] for row in edits], ['split', 'split'])
        self.assertTrue(previews)
        self.assertEqual(checks, {'c1': 'apply', 'c2': 'grey'})
        questions = self.client.get('/api/episodes/e/escalations', params={'reviewer': 'claude'}).json()['items']
        self.assertEqual([(item['kind'], item['evidence']['time'], item['evidence']['strip'].split('?')[0]) for item in questions],
                         [('missed_cut', 6.0, '/api/cuts/c2/missed-cut-strip')])
        report = self.client.get('/api/episodes/e/missed-cuts').json()
        self.assertEqual(len(report['splits']), 2)
        self.assertEqual(report['counts']['analysis_off'], 1)
        # A second run skips cuts it already checked with these boundaries and never asks the same peak twice.
        _, checked = self.run_job({4: {'verdict': 'grey', 'split_at': [], 'grey': [6.0]}})
        self.assertEqual(sorted(call.args[3] for call in checked.call_args_list), [0, 1.5, 2.5])  # the split pieces have new bounds
        self.assertEqual(len(self.client.get('/api/episodes/e/escalations').json()['items']), 1)

    def test_undo_merges_the_pieces_back_and_refuses_after_a_person_changed_them(self):
        self.run_job({0: {'verdict': 'apply', 'split_at': [1.5, 2.5], 'grey': []}})
        with db.connect() as conn:
            first, second = [row[0] for row in conn.execute("SELECT id FROM agent_decisions WHERE kind='missed_cuts' ORDER BY rowid")]
        self.assertEqual(self.client.post(f'/api/agent-decisions/{first}/undo').status_code, 409)  # the later split changed its right piece
        self.assertEqual(self.client.post(f'/api/agent-decisions/{second}/undo').json(), {'undone': True})
        self.assertEqual(self.client.post(f'/api/agent-decisions/{first}/undo').json(), {'undone': True})
        self.assertEqual([(start, end) for _, _, start, end in self.active()], [(0, 4), (4, 8), (8, 12)])
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM cuts WHERE episode_id='e'").fetchone()[0], 3)  # created pieces removed
            self.assertEqual(cut_history.state(conn, 'e')['undo']['action'], 'merge')
        self.run_job({0: {'verdict': 'apply', 'split_at': [2.0], 'grey': []}}, {'redo': True})
        with db.connect() as conn:
            decision = conn.execute("SELECT id FROM agent_decisions WHERE kind='missed_cuts' AND undone_at IS NULL").fetchone()[0]
        self.client.put('/api/cuts/c1/timing', json={'start_time': 0, 'end_time': 1.8})  # the person moves the boundary
        with db.connect() as conn:
            self.assertIsNotNone(conn.execute('SELECT overridden_at FROM agent_decisions WHERE id=?', (decision,)).fetchone()[0])
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').status_code, 409)

    def test_sequenced_cuts_and_ask_mode_go_to_a_person(self):
        with db.connect() as conn:
            conn.execute("""INSERT INTO sequences(id,episode_id,title,first_cut_id,last_cut_id,cuts_json,start_time,end_time)
                VALUES ('s','e','S','c2','c2','[{"id":"c2"}]',4,8)""")
        self.run_job({0: {'verdict': 'apply', 'split_at': [2.0], 'grey': []}, 4: {'verdict': 'apply', 'split_at': [6.0], 'grey': [7.0]}})
        self.assertEqual([(start, end) for _, _, start, end in self.active()][:3], [(0, 2.0), (2.0, 4), (4, 8)])
        people = self.client.get('/api/episodes/e/escalations', params={'reviewer': 'human'}).json()['items']
        self.assertEqual(sorted(item['evidence']['time'] for item in people), [6.0, 7.0])
        self.assertEqual(self.client.get('/api/episodes/e/escalations', params={'reviewer': 'claude'}).json()['items'], [])
        claude_try = self.client.post(f"/api/escalations/{people[0]['id']}/missed-cut", json={'answer': 'split', 'tier': 'claude'})
        self.assertEqual(claude_try.status_code, 409)
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'missed_cuts': 'ask'}}})
        self.run_job({2.0: {'verdict': 'apply', 'split_at': [3.0], 'grey': []}})
        self.assertNotIn(3.0, [start for _, _, start, _ in self.active()])
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'missed_cuts': 'off'}}})
        self.assertIsNone(self.client.post('/api/episodes/e/missed-cuts', json={}).json()['job_id'])

    def test_answers_split_close_or_hand_to_the_person(self):
        self.run_job({0: {'verdict': 'grey', 'split_at': [], 'grey': [1.0, 2.0, 3.0]}})
        items = {item['evidence']['time']: item['id'] for item in self.client.get('/api/episodes/e/escalations').json()['items']}
        split = self.client.post(f'/api/escalations/{items[1.0]}/missed-cut', json={'answer': 'split', 'tier': 'claude'}).json()
        self.assertEqual(split['status'], 'resolved')
        with db.connect() as conn:
            self.assertEqual(conn.execute('SELECT tier FROM agent_decisions WHERE id=?', (split['decision_id'],)).fetchone()[0], 'claude')
        self.assertIn(1.0, [start for _, _, start, _ in self.active()])
        self.assertEqual(self.client.post(f'/api/escalations/{items[2.0]}/missed-cut', json={'answer': 'no_cut', 'tier': 'claude'}).json()['status'], 'resolved')
        handed = self.client.post(f'/api/escalations/{items[3.0]}/missed-cut', json={'answer': 'unsure', 'tier': 'claude', 'note': 'a dissolve?'}).json()
        self.assertEqual(handed['status'], 'handed_to_person')
        open_items = self.client.get('/api/episodes/e/escalations').json()['items']
        self.assertEqual([(item['reviewer'], item['evidence']['claude_note']) for item in open_items], [('human', 'a dissolve?')])
        self.assertEqual(self.client.post(f'/api/escalations/{items[1.0]}/missed-cut', json={'answer': 'split'}).status_code, 409)  # closed
        self.assertEqual(self.client.post('/api/escalations/missing/missed-cut', json={'answer': 'split'}).status_code, 404)
        self.assertIs(agent_log.UNDO_KINDS['missed_cuts'], missed_cuts.undo_split)

    def test_queue_waits_for_cut_detection_and_is_registered(self):
        from app import job_kinds
        with db.connect() as conn:
            pending = worker.insert_job(conn, 'e', 'hardcuts', {'settings': {}})
            job_id = missed_cuts.queue(conn, 'e')
            payload = json.loads(conn.execute('SELECT payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()[0])
        self.assertEqual(payload['depends_on'], [pending])
        self.assertEqual(payload['policy'], {'mode': 'apply', 'claude': True})
        self.assertEqual(job_kinds.KINDS['missed_cuts'].lane, 'media')


if __name__ == '__main__':
    unittest.main()
