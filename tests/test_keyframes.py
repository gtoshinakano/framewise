import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from app import db, keyframes, media, worker
import tests.test_analysis as analysis_tests


class KeyframesTest(unittest.TestCase):
    def shot(self, frames, **settings):
        c = keyframes.config(settings)
        shot = keyframes.Shot({'id': 'c', 'start_time': 0., 'end_time': len(frames)/10}, c, {'top': .8, 'bottom': 1., 'source': 'fallback'})
        for i, frame in enumerate(frames): shot.add(i, i/10, frame)
        return shot.finish()

    def test_static_and_subtitle_only(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = a.copy(); b[85:95, 10:90] = 255
        for frames in ([a]*8, [a, b, a]):
            result = self.shot(frames)
            self.assertEqual(result['selected_frame_count'], 1)
            self.assertEqual(result['shot_change_score'], 0)

    def test_transient_and_return(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = a.copy(); b[20:40, 20:40] = 255
        result = self.shot([a, a, b, b, a, a])
        indexes = [x['frame_index'] for x in result['selected_frames']]
        self.assertIn(0, indexes); self.assertTrue(any(i in indexes for i in (2, 3)))
        self.assertIn(5, indexes)
        self.assertEqual(indexes, sorted(indexes))

    def test_config_and_cache(self):
        with self.assertRaises(ValueError): keyframes.config({'budget': 16})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'video'; path.write_bytes(b'a')
            cut = {'id': 'c', 'start_time': 0, 'end_time': 1}
            c = keyframes.config({})
            self.assertNotEqual(keyframes.cache_key(path, cut, c, {'top': .8}), keyframes.cache_key(path, cut, c, {'top': .7}))
            from unittest.mock import patch
            current = keyframes.cache_key(path, cut, c, {'top': .8})
            with patch.object(keyframes, 'VERSION', 'older-version'):
                self.assertNotEqual(current, keyframes.cache_key(path, cut, c, {'top': .8}))

    def test_duration_budgets_and_configuration(self):
        for duration, expected in [(0.1, 5), (4.999, 5), (5, 5), (5.001, 10), (10, 10), (10.001, 15), (15, 15), (40, 15)]:
            self.assertEqual(keyframes.frame_budget(duration, 15), expected)
        self.assertEqual(keyframes.frame_budget(20, 3), 3)
        self.assertEqual(keyframes.config({})['budget'], 15)
        self.assertEqual(keyframes.config({'budget': 5})['budget'], 5)
        with self.assertRaises(ValueError): keyframes.config({'state_novelty_threshold': 0})

    def test_sustained_change_preserves_distinct_stages(self):
        rng = np.random.default_rng(9)
        panorama = rng.integers(0, 256, size=(48, 150), dtype=np.uint8)
        frames = [panorama[:, i:i+48].copy() for i in range(0, 100, 2)]
        result = self.shot(frames)
        indices = [item['frame_index'] for item in result['selected_frames']]
        self.assertGreaterEqual(len(indices), 4)
        self.assertLessEqual(len(indices), 5)
        self.assertTrue(any(10 <= i <= 40 for i in indices))
        self.assertEqual(indices[0], 0)
        self.assertTrue(result['coverage_limited'])

    def test_window_and_total_limits_on_long_cut(self):
        rng = np.random.default_rng(23)
        frames = [rng.integers(0, 256, size=(32, 32), dtype=np.uint8) for _ in range(200)]
        result = self.shot(frames)
        times = [item['relative_timestamp_seconds'] for item in result['selected_frames']]
        self.assertEqual(len(times), 15)
        self.assertTrue(all(sum(int(t // 5) == window for t in times) <= 5 for window in range(4)))
        self.assertTrue(result['coverage_limited'])
        self.assertEqual(result['selected_frame_count'], len(times))
        limited = self.shot(frames, budget=3)
        self.assertLessEqual(limited['selected_frame_count'], 3)

    def test_motion_after_fifteen_seconds_and_static_long_cut(self):
        rng = np.random.default_rng(31)
        still = np.zeros((32, 32), dtype=np.uint8)
        quiet = self.shot([still] * 200)
        self.assertEqual(quiet['selected_frame_count'], 1)
        frames = [still] * 150 + [rng.integers(0, 256, size=(32, 32), dtype=np.uint8) for _ in range(50)]
        result = self.shot(frames)
        times = [frame['relative_timestamp_seconds'] for frame in result['selected_frames']]
        self.assertEqual(times[0], 0)
        self.assertTrue(any(t >= 15 for t in times))
        self.assertLessEqual(sum(t >= 15 for t in times), 5)
        self.assertTrue(result['coverage_limited'])

    def test_window_boundary_assignment(self):
        self.assertEqual(keyframes.frame_window({'relative_timestamp_seconds': 4.999}), 0)
        self.assertEqual(keyframes.frame_window({'relative_timestamp_seconds': 5.0}), 1)
        self.assertEqual(keyframes.frame_window({'relative_timestamp_seconds': 15.0}), 3)

    def test_legacy_settings_snapshot(self):
        settings = keyframes.config({})
        settings.pop('state_novelty_threshold')
        settings['budget'] = 5
        shot = keyframes.Shot({'id':'old', 'start_time':0., 'end_time':12.}, settings, {'top':.8, 'bottom':1.})
        self.assertEqual(shot.c['budget'], 5)
        self.assertGreater(shot.c['state_novelty_threshold'], 0)

    def test_variable_rate_timestamps(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'vfr.mkv'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=5:duration=1', '-vf', 'select=not(eq(n\\,2))', '-fps_mode', 'passthrough', '-c:v', 'ffv1', str(path)], check=True)
            frames = list(keyframes.decoded(path, 96, 64, 64))
            self.assertEqual([round(f[1], 2) for f in frames], [0, .2, .6, .8])

    def test_decoder(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'source.mkv'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=5:duration=1', '-c:v', 'ffv1', str(path)], check=True)
            frames = list(keyframes.decoded(path, 96, 64, 64))
            self.assertEqual(len(frames), 5)
            self.assertEqual([f[0] for f in frames], list(range(5)))
            self.assertAlmostEqual(frames[3][1], .6, places=3)


class KeyframeQueueTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_failed_selection_retry_deduplicates_and_waits_for_current_producers(self):
        initial = self.client.post('/api/cuts/c1/keyframe-selection').json()['job_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (initial,))
        hardcuts = worker.enqueue('e', 'hardcuts')
        retry = self.client.post(f'/api/jobs/{initial}/retry')
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(self.client.post(f'/api/jobs/{initial}/retry').json(), retry.json())
        job = db.one('SELECT * FROM jobs WHERE id=?', (retry.json()['job_id'],))
        self.assertEqual(json.loads(job['payload_json'])['depends_on'], [hardcuts])
        self.assertEqual(job['priority_class'], 'interactive')
        self.assertIsNone(worker.claim_job(('keyframes',)))

    def test_selection_reads_the_current_boundaries_after_queued_media(self):
        (media.episode_dir('p', 'e') / 'original.mp4').write_bytes(b'movie')
        predecessor = worker.enqueue('e', 'hardcuts')
        response = self.client.post('/api/cuts/c1/keyframe-selection')
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        self.assertEqual(self.client.post('/api/cuts/c1/keyframe-selection').json()['job_id'], job_id)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])
        self.assertEqual(payload['depends_on'], [predecessor])
        self.assertNotIn('subtitle_mask', payload)
        self.assertIsNone(worker.claim_job(('keyframes',)))
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
            conn.execute("UPDATE jobs SET status='done' WHERE id=?", (predecessor,))
        frames = [(index, index / 2, np.zeros((80, 80), dtype=np.uint8)) for index in range(5)]
        with patch.object(media, 'probe', return_value={'width': 80, 'height': 80}), \
             patch.object(keyframes, 'decoded', return_value=iter(frames)):
            worker.process(*worker.claim_job(('keyframes',)))
        result = self.client.get('/api/cuts/c1/keyframe-selection').json()
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['result']['end_timestamp_seconds'], 1.5)
        self.assertEqual(result['result']['total_frames'], 3)


class KeyframeJobTest(unittest.TestCase):
    def test_episode_job_and_stale_cache(self):
        import json
        import subprocess
        from unittest.mock import patch
        from app import db, media, worker
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with patch.object(db, 'DATA_DIR', base), patch.object(db, 'DB_PATH', base / 'db.sqlite'), patch.object(media, 'DATA_DIR', base):
                db.init_db()
                with db.connect() as conn:
                    conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
                    conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,settings_json) VALUES ('e','p','E','v.mkv','.mkv','{}')")
                    conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c','e',0,0,1)")
                source = media.episode_dir('p', 'e')/'original.mkv'
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=5:duration=1', '-c:v', 'ffv1', str(source)], check=True)
                settings = keyframes.config({})
                band = keyframes.mask_for('e', settings)
                cuts = [dict(row) for row in db.all_rows("SELECT id,start_time,end_time FROM cuts WHERE episode_id='e'")]
                job_id = worker.enqueue('e', 'keyframes', {'keyframe_settings': settings, 'subtitle_mask': band, 'cuts': cuts})
                with db.connect() as conn:
                    conn.execute("UPDATE jobs SET status='running',lease_owner='test' WHERE id=?", (job_id,))
                worker.process(job_id, 'test')
                self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
                key = keyframes.cache_key(source, cuts[0], settings, band)
                row = db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (key,))
                self.assertEqual(json.loads(row['result_json'])['total_frames'], 5)
                cuts[0]['end_time'] = .8
                self.assertIsNone(db.one('SELECT result_json FROM keyframe_results WHERE cache_key=?', (keyframes.cache_key(source, cuts[0], settings, band),)))

if __name__ == '__main__': unittest.main()
