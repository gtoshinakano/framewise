"""Real-video tests for disposable frames and analysis provenance."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from app import db, media, frame_cache
from app.main import app


class FrameRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','Project')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,duration,settings_json) VALUES ('e','p','Episode','test.mp4','.mp4','ready',3,'{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c','e',0,0,3,0,2.96)")
        self.folder = media.episode_dir('p', 'e')
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=25:d=3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(self.folder / 'original.mp4')], check=True)
        import shutil
        shutil.copyfile(self.folder / 'original.mp4', self.folder / 'proxy.mp4')
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_result_and_cut_frames_are_regenerated_without_persisting_jpegs(self):
        expected = self.folder / 'expected.jpg'
        actual = media.extract_cut_frame(self.folder / 'original.mp4', expected, 1.24, 0, 3, width=1280)
        original_bytes = expected.read_bytes()
        expected.unlink()
        snapshot = json.dumps({'cut': {'start_time': 0, 'end_time': 3}, 'sampled_frames': [{'frame_id': 'frame', 'time': actual}]})
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,context_json) VALUES ('frame','c','other',?,'view','model',?),('det','c','other',?,'{}','model',?)", (f'Frame at {actual:.6f}s', snapshot, f'Object detections at {actual:.3f}s', snapshot))
        for result in ('frame', 'det'):
            response = self.client.get(f'/api/analysis-results/{result}/frame')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, original_bytes)
        # Changes to the cut boundary must not move historical analysis frames.
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET start_time=2 WHERE id='c'")
        with frame_cache._lock:
            frame_cache._cache.clear()
        response = self.client.get('/api/analysis-results/det/frame')
        self.assertEqual(response.content, original_bytes)
        # Cut previews are generated from the current cut and need no stored JPEG.
        response = self.client.get('/api/cuts/c/frame/first')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.content.startswith(b'\xff\xd8'))
        self.assertEqual(list(self.folder.rglob('*.jpg')), [])

    def test_missing_original_uses_legacy_frame_without_deleting_it(self):
        analysis = self.folder / 'analysis'; analysis.mkdir()
        legacy = analysis / 'frame.jpg'; legacy.write_bytes(b'legacy')
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('frame','c','other','Frame at 1.000s','view','model')")
        (self.folder / 'original.mp4').unlink()
        response = self.client.get('/api/analysis-results/frame/frame')
        self.assertEqual(response.content, b'legacy')
        self.assertTrue(legacy.is_file())

    def test_cache_is_bounded_and_expires(self):
        spec = {'source': self.folder / 'original.mp4', 'time': 1.0, 'start': 0, 'end': 3, 'width': 480, 'last': False}
        a = frame_cache.request(spec).result(timeout=20)
        b = frame_cache.request(spec).result(timeout=20)
        self.assertEqual(a, b)
        with frame_cache._lock:
            frame_cache._prune(float('inf'))
            self.assertFalse(frame_cache._cache)

    def test_legacy_sampled_frame_is_recreated_then_removed(self):
        frames = self.folder / 'frames'
        media.extract_frames(self.folder / 'proxy.mp4', frames, 1.0)
        legacy = frames / 'frame_000002.jpg'
        expected = legacy.read_bytes()
        response = self.client.get('/api/episodes/e/media/frame_000002.jpg')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, expected)
        self.assertFalse(legacy.exists())
        self.assertEqual(self.client.get('/api/episodes/e/media/frame_000002.jpg').content, expected)

    def test_cleanup_verifies_linked_result_before_removing(self):
        analysis = self.folder / 'analysis'; analysis.mkdir()
        legacy = analysis / '00000000-0000-4000-8000-000000000001.jpg'
        timestamp = media.extract_cut_frame(self.folder / 'original.mp4', legacy, 1.24, 0, 3, width=1280)
        expected = legacy.read_bytes()
        snapshot = json.dumps({'cut': {'start_time': 0, 'end_time': 3}, 'sampled_frames': [{'time': timestamp}]})
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,context_json) VALUES (?,?,?,?,?,'model',?)", (legacy.stem, 'c', 'other', f'Frame at {timestamp:.6f}s', 'view', snapshot))
        item = next(item for item in self.client.get('/api/storage').json()['files'] if item['path'].endswith(legacy.name))
        response = self.client.post('/api/storage/cleanup', json={'files': [{k: item[k] for k in ('path', 'size', 'modified_ns')}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(legacy.exists())
        self.assertEqual(self.client.get(f'/api/analysis-results/{legacy.stem}/frame').content, expected)

    def test_different_legacy_image_is_preserved(self):
        analysis = self.folder / 'analysis'; analysis.mkdir()
        legacy = analysis / '00000000-0000-4000-8000-000000000002.jpg'; legacy.write_bytes(b'unique evidence')
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES (?,?,?,?,?,'model')", (legacy.stem, 'c', 'other', 'Frame at 1.000s', 'view'))
        item = next(item for item in self.client.get('/api/storage').json()['files'] if item['path'].endswith(legacy.name))
        response = self.client.post('/api/storage/cleanup', json={'files': [{k: item[k] for k in ('path', 'size', 'modified_ns')}]})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.get(f'/api/analysis-results/{legacy.stem}/frame').content, b'unique evidence')
        self.assertTrue(legacy.exists())

    def test_bulk_gc_keeps_mismatched_evidence_and_original(self):
        from app import frame_gc
        analysis = self.folder / 'analysis'; analysis.mkdir()
        good = analysis / '00000000-0000-4000-8000-000000000003.jpg'
        bad = analysis / '00000000-0000-4000-8000-000000000004.jpg'
        actual = media.extract_cut_frame(self.folder / 'original.mp4', good, 1, 0, 3, width=1280)
        good_bytes = good.read_bytes()
        bad.write_bytes(b'different legacy image')
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES (?,?,?,?,?,'model')", (good.stem, 'c', 'other', f'Frame at {actual:.6f}s', 'good'))
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES (?,?,?,?,?,'model')", (bad.stem, 'c', 'other', f'Frame at {actual:.6f}s', 'bad'))
        stats = frame_gc.collect(apply=True, progress=lambda _: None)
        self.assertEqual(stats['removed'], 1)
        self.assertEqual(stats['different'], 1)
        self.assertFalse(good.exists())
        self.assertTrue(bad.exists())
        self.assertTrue((self.folder / 'original.mp4').exists())
        self.assertEqual(self.client.get(f'/api/analysis-results/{good.stem}/frame').content, good_bytes)

    def test_missing_manifest_recreates_interval_index(self):
        # A removed manifest must not make on-demand interval frames inaccessible.
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES ('prepare','e','prepare','done',?)", (json.dumps({'settings': {'frame_interval': 1}}),))
        response = self.client.get('/api/episodes/e/frames')
        self.assertEqual(response.json()['interval'], 1)
        self.assertEqual(len(response.json()['frames']), 3)
        image = self.client.get('/api/episodes/e/media/frame_000002.jpg')
        self.assertEqual(image.status_code, 200, image.text)
        self.assertTrue(image.content.startswith(b'\xff\xd8'))
        self.assertEqual(list(self.folder.rglob('*.jpg')), [])

    def test_detection_uses_precise_time_when_rounded_labels_collide(self):
        from app import frame_source
        snapshot = json.dumps({'cut': {'start_time': 0, 'end_time': 3}, 'frame_time': 1.2404,
                               'sampled_frames': [{'time': 1.2401}, {'time': 1.2404}]})
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,context_json) VALUES ('d','c','other','Object detections at 1.240s','{}','model',?)", (snapshot,))
        self.assertEqual(frame_source.result_frame_spec('d')['time'], 1.2404)
