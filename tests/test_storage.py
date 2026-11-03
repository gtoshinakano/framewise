"""Storage cleanup must protect originals, live work, and referenced evidence."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from app import db, media
from app.main import app


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','Project')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c','e',0,0,1)")
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,error) VALUES ('j','e','analyze','failed','provider failed')")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,job_id) VALUES ('r','c','other','Frame','text','j')")
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES ('j','c',?)", (json.dumps([{'frame_file':'evidence.jpg'}]),))
        self.root = media.episode_dir('p','e')
        (self.root / 'analysis').mkdir()
        for file in ('original.mp4','analysis/r.jpg','analysis/evidence.jpg','analysis/00000000-0000-4000-8000-000000000003.jpg','proxy.partial.mp4'):
            (self.root / file).write_bytes(b'data')
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def inventory(self):
        response = self.client.get('/api/storage')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_inventory_protection_cleanup_and_stale_selection(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e2','p','Second','a.mp4','.mp4','ready','{}')")
        other = media.episode_dir('p', 'e2')
        (other / 'waveform.json').write_text('{\"peaks\": []}')
        previews = self.root / 'thumbnails'; previews.mkdir()
        (previews / 'cut_000000.jpg').write_bytes(b'data')
        inv = self.inventory()
        files = {Path(f['path']).name:f for f in inv['files']}
        self.assertFalse(files['original.mp4']['deletable'])
        self.assertFalse(files['r.jpg']['deletable'])
        self.assertTrue(files['evidence.jpg']['deletable'])
        self.assertEqual(files['evidence.jpg']['kind'], 'Recreatable run frame')
        self.assertTrue(files['00000000-0000-4000-8000-000000000003.jpg']['deletable'])
        self.assertTrue(files['proxy.partial.mp4']['deletable'])
        self.assertTrue(any(item['file'] == 'proxy.mp4' for item in inv['missing']))
        def body(item): return {'files': [{k:item[k] for k in ('path','size','modified_ns')}]}
        self.assertEqual(self.client.post('/api/storage/cleanup', json=body(files['original.mp4'])).status_code, 409)
        (self.root / 'analysis/00000000-0000-4000-8000-000000000003.jpg').write_bytes(b'changed')
        self.assertEqual(self.client.post('/api/storage/cleanup', json=body(files['00000000-0000-4000-8000-000000000003.jpg'])).status_code, 409)
        file = next(f for f in self.inventory()['files'] if f['path'].endswith('00000000-0000-4000-8000-000000000003.jpg'))
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status) VALUES ('busy','e','analyze','queued')")
        self.assertEqual(self.client.post('/api/storage/cleanup', json=body(file)).status_code, 409)
        with db.connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id='busy'")
        response = self.client.post('/api/storage/cleanup', json=body(file))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse((self.root / 'analysis/00000000-0000-4000-8000-000000000003.jpg').exists())
        self.assertTrue((self.root / 'original.mp4').exists())

    def test_run_history_and_missing_evidence(self):
        (self.root / 'analysis/evidence.jpg').unlink()
        self.assertFalse(any(item['file'] == 'analysis/evidence.jpg' for item in self.inventory()['missing']))
        (self.root / 'original.mp4').unlink()
        self.assertTrue(any(item['file'] == 'analysis/evidence.jpg' for item in self.inventory()['missing']))
        output = self.client.get('/api/jobs/j/outputs').json()
        self.assertEqual(output['results'][0]['id'], 'r')
        self.assertEqual(output['evidence'][0]['cut_id'], 'c')
        self.assertEqual(self.client.delete('/api/jobs/j').status_code, 200)
        self.assertIsNone(db.one("SELECT job_id FROM analysis_results WHERE id='r'")['job_id'])
        self.assertEqual(self.client.get('/api/jobs/j/outputs').status_code, 404)

    def test_orphan_records_and_corrupt_evidence(self):
        import sqlite3
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content) VALUES ('lost','missing','other','orphan content')")
            conn.execute("UPDATE analysis_job_cuts SET evidence_json='bad' WHERE job_id='j'")
        inv = self.inventory()
        self.assertEqual(inv['orphan_results'][0]['content'], 'orphan content')
        self.assertTrue(inv['warnings'])
        self.assertFalse(next(f for f in inv['files'] if f['path'].endswith('00000000-0000-4000-8000-000000000003.jpg'))['deletable'])
        self.assertEqual(self.client.delete('/api/storage/orphans/result/r').status_code, 409)
        self.assertEqual(self.client.delete('/api/storage/orphans/result/lost').status_code, 200)

    def test_old_voice_upload_and_unlinked_reference(self):
        uploads = self.root / 'voice-uploads'; uploads.mkdir()
        upload = uploads / '00000000-0000-4000-8000-000000000001'; upload.write_bytes(b'voice')
        os.utime(upload, (time.time()-90000, time.time()-90000))
        refdir = db.DATA_DIR / 'projects/p/voice-references'; refdir.mkdir()
        (refdir / '00000000-0000-4000-8000-000000000002.wav').write_bytes(b'voice')
        files = {f['kind']: f for f in self.inventory()['files']}
        self.assertTrue(files['Old voice upload']['deletable'])
        self.assertTrue(files['Unlinked voice reference']['deletable'])

    def test_delete_saved_evidence_preserves_completion_and_other_run_links(self):
        with db.connect() as conn:
            conn.execute("UPDATE analysis_job_cuts SET completed=1 WHERE job_id='j'")
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status) VALUES ('j2','e','analyze','failed')")
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES ('j2','c',?)", (json.dumps([{'frame_file':'evidence.jpg'}]),))
        self.assertEqual(self.client.delete('/api/jobs/j/evidence/c').status_code, 200)
        row = db.one("SELECT completed,evidence_json FROM analysis_job_cuts WHERE job_id='j'")
        self.assertEqual(row['completed'], 1)
        self.assertEqual(json.loads(row['evidence_json']), [])
        self.assertTrue(next(f for f in self.inventory()['files'] if f['path'].endswith('evidence.jpg'))['deletable'])
        self.assertEqual(self.client.delete('/api/jobs/j2/evidence/c').status_code, 200)
        # Unknown file names are not automatically eligible even after unlinking.
        self.assertEqual(next(f for f in self.inventory()['files'] if f['path'].endswith('evidence.jpg'))['kind'], 'Unknown file')

    def test_orphan_revision_chain_and_bad_metadata(self):
        import sqlite3
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content,superseded_by) VALUES ('lost1','missing','other','one','lost2')")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content,parent_id) VALUES ('lost2','missing','other','two','lost1')")
            conn.execute("UPDATE jobs SET payload_json='{' WHERE id='j'")
            conn.execute("UPDATE cuts SET results_json='{' WHERE id='c'")
        self.assertEqual(self.client.get('/api/episodes/e').status_code, 200)
        self.assertTrue(any('payload_json' in w for w in self.inventory()['warnings']))
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET payload_json=? WHERE id='j'", (json.dumps({'cuts': [{'cut_id': ['invalid']}, {'cut_id': 'c', 'context': {'cut': {'start_time': 'bad', 'title': None}}}]}),))
        job = self.client.get('/api/episodes/e').json()['jobs'][0]
        self.assertEqual(job['cuts'][0]['start_time'], None)
        self.assertEqual(job['cuts'][0]['title'], '')
        self.assertEqual(self.client.delete('/api/storage/orphans/result/lost1').status_code, 200)
        self.assertIsNone(db.one("SELECT id FROM analysis_results WHERE id='lost2'"))

    def test_prepared_media_is_explicitly_deletable_and_corruption_is_visible(self):
        (self.root / 'proxy.mp4').write_bytes(b'proxy')
        (self.root / 'waveform.json').write_text('{bad')
        inv = self.inventory()
        proxy = next(f for f in inv['files'] if f['path'].endswith('/proxy.mp4'))
        self.assertTrue(proxy['deletable'])
        self.assertFalse(proxy['cleanup_candidate'])
        self.assertTrue(any('Unreadable waveform.json' in warning for warning in inv['warnings']))
        self.assertEqual(self.client.get('/api/episodes/e/waveform').json()['peaks'], [])
        result = self.client.post('/api/storage/cleanup', json={'files': [{key: proxy[key] for key in ('path', 'size', 'modified_ns')}]})
        self.assertEqual(result.status_code, 200)
        self.assertTrue(any(item['file'] == 'proxy.mp4' for item in self.inventory()['missing']))
        self.assertTrue((self.root / 'original.mp4').exists())

    def test_orphan_evidence_and_result_frames_are_protected_until_removed(self):
        import sqlite3
        orphan_id = '00000000-0000-4000-8000-000000000004'
        evidence_name = '00000000-0000-4000-8000-000000000005.jpg'
        (self.root / 'analysis' / (orphan_id + '.jpg')).write_bytes(b'result')
        (self.root / 'analysis' / evidence_name).write_bytes(b'evidence')
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content) VALUES (?, 'missing', 'other', 'lost')", (orphan_id,))
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES ('missing-job','c',?)", (json.dumps([{'frame_file': evidence_name}]),))
        inv = self.inventory()
        self.assertFalse(next(f for f in inv['files'] if f['path'].endswith(orphan_id+'.jpg'))['deletable'])
        self.assertFalse(next(f for f in inv['files'] if f['path'].endswith(evidence_name))['deletable'])
        self.assertEqual(inv['orphan_evidence'][0]['frame_count'], 1)
        self.assertEqual(self.client.delete('/api/storage/orphan-evidence/missing-job/c').status_code, 200)
        self.assertEqual(self.client.delete('/api/storage/orphans/result/'+orphan_id).status_code, 200)
        inv = self.inventory()
        self.assertTrue(next(f for f in inv['files'] if f['path'].endswith(orphan_id+'.jpg'))['deletable'])
        self.assertTrue(next(f for f in inv['files'] if f['path'].endswith(evidence_name))['deletable'])
