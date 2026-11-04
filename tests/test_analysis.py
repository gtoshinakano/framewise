"""Contract tests for analysis snapshots, worker isolation, and revision-safe batch writes."""
import json
import os
import tempfile
import threading
import unittest
import io
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import db, media, worker, inference_worker
from app.main import app


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = Path(self.tmp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','Snapshot synopsis')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json) VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,2),('c2','e',1,2,4)")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('ctx','p','character','Hero','Green coat'),('other','p','world','World','Rain')")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_repeated_merges_refresh_only_changed_cuts(self):
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET duration=8 WHERE id='e'")
            conn.execute("UPDATE cuts SET first_frame_time=start_time,last_frame_time=end_time-0.1 WHERE episode_id='e'")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c3','e',2,4,6,4,5.9),('c4','e',3,6,8,6,7.9)")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('saved','c2','other','Note','Keep me')")
        (media.episode_dir('p', 'e') / 'proxy.mp4').touch()
        def extract(_proxy, target, timestamp, *_args, **_kwargs):
            target.write_bytes(b'frame')
            return timestamp
        for expected_archived in ('c2', 'c3'):
            response = self.client.post('/api/cuts/c1/merge-next')
            self.assertEqual(response.status_code, 200, response.text)
            job = db.one("SELECT * FROM jobs WHERE episode_id='e' ORDER BY rowid DESC LIMIT 1")
            self.assertEqual(json.loads(job['payload_json'])['cut_ids'], ['c1'])
            self.assertIsNotNone(db.one("SELECT first_frame_time FROM cuts WHERE id='c4'")['first_frame_time'])
            with patch.object(media, 'extract_cut_frame', side_effect=extract) as extract_frame:
                worker.process(*worker.claim_job(('cutframes',)))
            self.assertEqual(extract_frame.call_count, 2)
            self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job['id'],))['status'], 'done')
            self.assertEqual(db.one("SELECT active FROM cuts WHERE id=?", (expected_archived,))['active'], 0)
        self.assertEqual(db.one("SELECT content FROM analysis_results WHERE id='saved'")['content'], 'Keep me')
        cuts = db.all_rows("SELECT cut_index,start_time,end_time FROM cuts WHERE episode_id='e' AND active=1 ORDER BY cut_index")
        self.assertEqual([(c['cut_index'], c['start_time'], c['end_time']) for c in cuts], [(0, 0, 6), (1, 6, 8)])

    def test_cut_timing_refreshes_only_boundaries_that_changed(self):
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET duration=6 WHERE id='e'")
            conn.execute("UPDATE cuts SET first_frame_time=start_time,last_frame_time=end_time-0.1 WHERE episode_id='e'")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES ('c3','e',2,4,6,4,5.9)")
        response = self.client.put('/api/cuts/c2/timing', json={'start_time': 1.5, 'end_time': 4})
        self.assertEqual(response.status_code, 200, response.text)
        job = db.one("SELECT payload_json FROM jobs WHERE episode_id='e' ORDER BY rowid DESC LIMIT 1")
        self.assertEqual(json.loads(job['payload_json'])['cut_ids'], ['c1', 'c2'])
        self.assertEqual(db.one("SELECT first_frame_time FROM cuts WHERE id='c3'")['first_frame_time'], 4)

    def test_batch_snapshots_and_resume(self):
        config = {'frame_mode': 'uniform', 'frame_count': 3, 'detection_queries': ['person', 'coat'], 'context_ids': ['ctx'], 'additional_context': 'At night', 'openrouter_model': 'openrouter/free'}
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json=config).status_code, 200)
        response = self.client.post('/api/episodes/e/analyze')
        self.assertEqual(response.status_code, 200, response.text)
        job_id = response.json()['job_id']
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])
        job_summary = self.client.get('/api/episodes/e').json()['jobs'][0]
        self.assertEqual(job_summary['id'], job_id)
        self.assertNotIn('payload_json', job_summary)
        self.assertNotIn('lease_owner', job_summary)
        self.assertEqual(payload['cuts'][0]['config'], config)
        self.assertEqual([item['id'] for item in payload['cuts'][0]['context']['context_items']], ['ctx'])
        self.assertEqual(payload['cuts'][1]['config']['frame_mode'], 'uniform')
        self.assertEqual(payload['cuts'][1]['config']['frame_count'], 4)
        with db.connect() as conn:
            conn.execute("UPDATE context_items SET content='Changed' WHERE id='ctx'")
        self.assertEqual(payload['cuts'][0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertIsNone(worker.claim_job())
        claimed = worker.claim_job(('analyze',))
        self.assertEqual(claimed[0], job_id)
        calls = []
        def descriptions(job, owner, episode, cut, config):
            calls.append(cut['id'])
            return [{'time': cut['start_time'] + .05, 'description': 'Visible figure'}]
        def summary(config, context, observations, progress=None):
            return ({key: f'{key} draft' for _, _, key in inference_worker.FIELDS}, 'test/model')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=descriptions), patch.object(inference_worker, 'synthesize', side_effect=summary):
            inference_worker.process(*claimed)
        self.assertEqual(calls, ['c1', 'c2'])

        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        results = self.client.get('/api/cuts/c1/analysis-results').json()
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertEqual(results[0]['job_id'], job_id)
        compact = self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()
        self.assertEqual(len(compact), len(results))
        self.assertEqual(compact[0]['id'], results[0]['id'])
        self.assertEqual(compact[0]['context'], {})
        saved = self.client.put('/api/analysis-results/' + results[0]['id'], json={'title': 'Edited', 'content': 'Human revision'})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results?history=true').json()), 5)

    def test_default_frames_and_existing_endpoint_preset(self):
        default = self.client.get('/api/cuts/c1/analysis-config').json()
        self.assertEqual((default['frame_mode'], default['frame_count']), ('uniform', 4))
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json={**default, 'frame_mode': 'endpoints', 'frame_count': 2}).status_code, 200)
        saved = self.client.get('/api/cuts/c1/analysis-config').json()
        self.assertEqual((saved['frame_mode'], saved['frame_count']), ('endpoints', 2))

    def test_episode_cut_result_indicator_tracks_current_results(self):
        def flags():
            return {cut['id']: cut['has_analysis_results'] for cut in self.client.get('/api/episodes/e').json()['cuts']}
        self.assertEqual(flags(), {'c1': False, 'c2': False})
        created = self.client.post('/api/cuts/c1/analysis-results', json={'kind': 'other', 'content': 'Manual note'}).json()
        self.assertEqual(flags(), {'c1': True, 'c2': False})
        edited = self.client.put('/api/analysis-results/' + created['id'], json={'title': '', 'content': 'Revised'}).json()
        self.assertTrue(flags()['c1'])
        self.assertEqual(self.client.delete('/api/analysis-results/' + edited['id']).status_code, 200)
        self.assertTrue(flags()['c1'])  # Deleting a revision restores its parent.
        self.assertEqual(self.client.delete('/api/analysis-results/' + created['id']).status_code, 200)
        self.assertFalse(flags()['c1'])

    def test_job_history_identifies_batch_cuts_and_failure(self):
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET title='Opening' WHERE id='c1'")
        job_id = self.client.post('/api/episodes/e/analyze').json()['job_id']
        queued = self.client.get('/api/episodes/e').json()['jobs'][0]
        self.assertEqual([(item['id'], item['cut_index'], item['title'], item['completed']) for item in queued['cuts']],
                         [('c1', 0, 'Opening', False), ('c2', 1, '', False)])
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET title='Renamed',cut_index=9 WHERE id='c1'")
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json,completed) VALUES (?,?,?,1)", (job_id, 'c1', '[]'))
            conn.execute("UPDATE jobs SET status='running',stage='Analyzing cut 2, frame 1/2',lease_owner='owner' WHERE id=?", (job_id,))
        inference_worker.fail_job(job_id, 'owner', 'e', RuntimeError('Sample failure'))
        failed = self.client.get('/api/episodes/e').json()['jobs'][0]
        self.assertEqual(failed['stage'], 'Failed · Analyzing cut 2, frame 1/2')
        self.assertEqual(failed['error'], 'Sample failure')
        self.assertEqual([(item['title'], item['completed']) for item in failed['cuts']], [('Opening', True), ('', False)])
        self.assertNotIn('payload_json', failed)

    def test_legacy_status_migration_preserves_results_and_cleans_snapshots(self):
        snapshot = {'current_results': [{'kind': 'other', 'content': 'Previously rejected', 'status': 'rejected'}],
                    'transcript_segments': [{'text': 'Speech', 'start_time': 0, 'end_time': 1, 'status': 'approved'}]}
        with db.connect() as conn:
            conn.execute("ALTER TABLE analysis_results ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'")
            conn.execute("ALTER TABLE transcript_segments ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,content,status,context_json) VALUES ('old','c1','other','Old revision','draft',?),('keep','c1','other','Previously rejected','rejected',?),('neighbor','c2','other','Neighbor','approved',?)", (json.dumps(snapshot),)*3)
            conn.execute("UPDATE analysis_results SET superseded_by='keep' WHERE id='old'")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,status) VALUES ('speech','e',0,1,'Speech','rejected'),('tombstone','e',1,2,'Deleted speech','deleted')")
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES ('oldjob','e','analyze','failed',?)", (json.dumps({'cuts': [{'cut_id': 'c1', 'context': snapshot}]}),))
        db.init_db()
        db.init_db()  # The migration must be safe to repeat.
        self.assertNotIn('status', {r['name'] for r in db.all_rows('PRAGMA table_info(analysis_results)')})
        self.assertNotIn('status', {r['name'] for r in db.all_rows('PRAGMA table_info(transcript_segments)')})
        self.assertEqual({r['id'] for r in db.all_rows('SELECT id FROM analysis_results')}, {'old', 'keep', 'neighbor'})
        history = self.client.get('/api/cuts/c1/analysis-results?history=true').json()
        self.assertEqual(len(history), 2)
        self.assertTrue(all('status' not in row and 'status' not in row['context']['current_results'][0] for row in history))
        self.assertEqual(db.one("SELECT superseded_by FROM analysis_results WHERE id='old'")['superseded_by'], 'keep')
        transcript = self.client.get('/api/episodes/e/transcript?history=true').json()
        self.assertTrue(all('status' not in row for row in transcript))
        self.assertTrue(next(row for row in transcript if row['id'] == 'tombstone')['deleted'])
        self.assertEqual([row['id'] for row in self.client.get('/api/episodes/e/transcript').json()], ['speech'])
        context = self.client.get('/api/cuts/c1/analysis-context').json()
        self.assertEqual([r['content'] for r in context['neighbor_results']['next']], ['Neighbor'])
        self.assertEqual([r['id'] for r in context['transcript_segments']], ['speech'])
        stored = self.client.get('/api/v1/jobs/oldjob').json()['payload']['cuts'][0]['context']
        self.assertNotIn('status', stored['current_results'][0])
        self.assertNotIn('status', stored['transcript_segments'][0])
        queued = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        captured = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (queued,))['payload_json'])['cuts'][0]['context']
        self.assertEqual([r['content'] for r in captured['current_results']], ['Previously rejected'])
        self.assertNotIn('status', captured['current_results'][0])
        self.assertEqual(self.client.post('/api/cuts/c1/analysis-results', json={'kind': 'other', 'content': 'x', 'status': 'draft'}).status_code, 422)

    def test_invalid_context_and_analysis_queue(self):
        response = self.client.put('/api/cuts/c1/analysis-config', json={'context_ids': ['missing']})
        self.assertEqual(response.status_code, 400)
        queued = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(queued.status_code, 200)
        second = self.client.post('/api/cuts/c2/analyze')
        self.assertEqual(second.status_code, 200)
        self.assertEqual(self.client.post(f"/api/jobs/{queued.json()['job_id']}/cancel").status_code, 200)
        self.assertEqual(db.one("SELECT status FROM episodes WHERE id='e'")["status"], 'queued')
        self.assertEqual(worker.claim_job(('analyze',))[0], second.json()['job_id'])

    def test_upgrade_replaces_single_active_job_constraint(self):
        with db.connect() as conn:
            conn.execute("CREATE UNIQUE INDEX jobs_active_episode_idx ON jobs(episode_id) WHERE status IN ('queued','running')")
        first = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(first.status_code, 200)
        db.init_db()
        second = self.client.post('/api/cuts/c2/analyze')
        self.assertEqual(second.status_code, 200)
        db.init_db()
        self.assertEqual(len(db.all_rows("SELECT id FROM jobs WHERE episode_id='e' AND status='queued'")), 2)

    def test_queue_exclusions_editing_and_run_history(self):
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET duration=4 WHERE id='e'")
        self.assertEqual(self.client.put('/api/cuts/c2/analysis-enabled', json={'analysis_enabled': False}).status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c2/analyze').status_code, 409)
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json={'additional_context': 'First preset'}).status_code, 200)
        first = self.client.post('/api/episodes/e/analyze').json()['job_id']
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json={'additional_context': 'Second preset'}).status_code, 200)
        self.assertEqual(self.client.put('/api/cuts/c2/analysis-enabled', json={'analysis_enabled': True}).status_code, 200)
        second = self.client.post('/api/episodes/e/analyze').json()['job_id']
        snapshots = [json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job,))['payload_json'])['cuts'] for job in (first, second)]
        self.assertEqual([item['cut_id'] for item in snapshots[0]], ['c1'])
        self.assertEqual([item['cut_id'] for item in snapshots[1]], ['c1', 'c2'])
        self.assertEqual([items[0]['config']['additional_context'] for items in snapshots], ['First preset', 'Second preset'])
        db.init_db()
        claimed = worker.claim_job(('analyze',))
        self.assertEqual(claimed[0], first)
        self.assertIsNone(worker.claim_job(('analyze',)))
        self.assertEqual(self.client.put('/api/cuts/c1', json={'notes': 'Editing during analysis'}).status_code, 200)
        self.assertEqual(self.client.put('/api/cuts/c1/timing', json={'start_time': 0, 'end_time': 1.5}).status_code, 200)
        frames = worker.claim_job(('cutframes',))
        self.assertIsNotNone(frames)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            inference_worker.process(*claimed)
        history = self.client.get('/api/episodes/e').json()['jobs']
        failed = next(job for job in history if job['id'] == first)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('OPENROUTER_API_KEY', failed['error'])
        self.assertIsNotNone(failed['started_at'])
        self.assertIsNotNone(failed['finished_at'])
        self.assertEqual(snapshots[0][0]['context']['cut']['end_time'], 2)
        self.assertEqual(self.client.post(f'/api/jobs/{first}/retry').status_code, 409)
        self.assertEqual(self.client.get('/api/cuts/c2/analysis-config').status_code, 200)

    def test_queue_snapshot_excludes_concurrent_context_edits(self):
        from app import main
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json={'context_ids': ['ctx']}).status_code, 200)
        captured, release, write_finished = threading.Event(), threading.Event(), threading.Event()
        original = main.selected_analysis_context
        def hold_snapshot(cut_id, config):
            value = original(cut_id, config)
            captured.set()
            if not release.wait(5):
                raise RuntimeError('Test did not release the snapshot')
            return value
        def edit_context():
            with db.connect() as conn:
                conn.execute("UPDATE context_items SET content='Changed concurrently' WHERE id='ctx'")
            write_finished.set()
        with ThreadPoolExecutor(max_workers=2) as executor, patch.object(main, 'selected_analysis_context', side_effect=hold_snapshot):
            queued = executor.submit(self.client.post, '/api/cuts/c1/analyze')
            try:
                self.assertTrue(captured.wait(5))
                edited = executor.submit(edit_context)
                self.assertFalse(write_finished.wait(.2))
            finally:
                release.set()
            response = queued.result(timeout=5)
            edited.result(timeout=5)
        self.assertEqual(response.status_code, 200, response.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (response.json()['job_id'],))['payload_json'])
        self.assertEqual(payload['cuts'][0]['context']['context_items'][0]['content'], 'Green coat')
        self.assertEqual(db.one("SELECT content FROM context_items WHERE id='ctx'")['content'], 'Changed concurrently')

    def test_checkpoint_reuse(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        observations = [{'time': 0.05, 'description': 'A figure', 'ocr_response': '<ref>OPEN</ref><box><0><0><100><100></box>', 'ocr_prompt': 'detect words'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', return_value=observations), patch.object(inference_worker, 'synthesize', side_effect=RuntimeError('network error')):
            inference_worker.process(*claimed)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')
        self.assertEqual(json.loads(db.one('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=?', (job_id,))['evidence_json']), observations)
        retry = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(retry.status_code, 200, retry.text)
        retry_id = retry.json()['job_id']
        self.assertNotEqual(retry_id, job_id)
        self.assertEqual(db.one('SELECT retry_of FROM jobs WHERE id=?', (retry_id,))['retry_of'], job_id)
        claimed = worker.claim_job(('analyze',))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=AssertionError('should reuse checkpoint')), patch.object(inference_worker, 'synthesize', return_value=({key: 'draft' for _, _, key in inference_worker.FIELDS}, 'test/model')):
            inference_worker.process(*claimed)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (retry_id,))['status'], 'done')
        results = self.client.get('/api/cuts/c1/analysis-results').json()
        self.assertEqual(len(results), 5)
        raw = next(item for item in results if item['title'].startswith('Raw OCR'))
        self.assertIn('OPEN', raw['content'])
        self.assertEqual(raw['model_id'], inference_worker.MODEL_ID)
        summary = next(item for item in results if item['title'] == 'Cut summary')
        self.assertEqual(summary['context']['sampled_frames'][0]['ocr_response'], observations[0]['ocr_response'])
        self.assertEqual(summary['context']['assessments']['visual_evidence'], 'draft')

    def test_older_checkpoint_is_refreshed_for_ocr(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_job_cuts(job_id,cut_id,evidence_json) VALUES (?,?,?)", (job_id, 'c1', json.dumps([{'time': 0.05, 'description': 'Old frame'}])))
        fresh = [{'time': 0.05, 'description': 'New frame', 'ocr_response': 'OPEN', 'ocr_prompt': 'detect words'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', return_value=fresh) as describe, patch.object(inference_worker, 'synthesize', return_value=({key: 'draft' for _, _, key in inference_worker.FIELDS}, 'test/model')):
            inference_worker.process(*worker.claim_job(('analyze',)))
        describe.assert_called_once()
        self.assertEqual(json.loads(db.one('SELECT evidence_json FROM analysis_job_cuts WHERE job_id=?', (job_id,))['evidence_json']), fresh)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')

    def test_retry_rejects_changed_unfinished_cut(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': ''}):
            inference_worker.process(*claimed)
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        response = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(response.status_code, 409)
        self.assertIn('changed', response.json()['detail'])
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')

    def test_retry_batch_skips_completed_cuts(self):
        job_id = self.client.post('/api/episodes/e/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        described, summarized = [], []
        def describe(job, owner, episode, cut, config):
            described.append(cut['id'])
            return [{'time': cut['start_time'], 'description': 'A figure', 'ocr_response': '', 'ocr_prompt': 'detect words'}]
        def summary(config, context, observations, progress=None):
            summarized.append(context['cut']['id'])
            if context['cut']['id'] == 'c2' and summarized.count('c2') == 1:
                raise RuntimeError('Temporary provider error')
            return ({key: 'draft' for _, _, key in inference_worker.FIELDS}, 'test/model')
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=describe), patch.object(inference_worker, 'synthesize', side_effect=summary):
            inference_worker.process(*claimed)
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'failed')
            self.assertEqual(self.client.post(f'/api/jobs/{job_id}/retry').status_code, 200)
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(described, ['c1', 'c2'])
        self.assertEqual(summarized, ['c1', 'c2', 'c2'])
        self.assertEqual(len(self.client.get('/api/cuts/c1/analysis-results').json()), 5)
        self.assertEqual(len(self.client.get('/api/cuts/c2/analysis-results').json()), 5)

    def test_lost_lease_prevents_result_writes(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        claimed = worker.claim_job(('analyze',))
        def lose_lease(*args):
            with db.connect() as conn:
                conn.execute("UPDATE jobs SET lease_owner='another-worker' WHERE id=?", (job_id,))
            return [{'time': .05, 'description': 'A figure'}]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', side_effect=lose_lease):
            inference_worker.process(*claimed)
        self.assertFalse(db.all_rows('SELECT * FROM analysis_results WHERE job_id=?', (job_id,)))
        self.assertIsNone(db.one('SELECT * FROM analysis_job_cuts WHERE job_id=?', (job_id,)))
        self.assertEqual(db.one('SELECT lease_owner FROM jobs WHERE id=?', (job_id,))['lease_owner'], 'another-worker')

    def test_sampling_stays_inside_cut(self):
        self.assertEqual(inference_worker.sample_times(2, 4, {'frame_mode': 'endpoints'}), [2, 3.999999])
        times = inference_worker.sample_times(2, 4, {'frame_mode': 'uniform', 'frame_count': 5})
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 2)
        self.assertEqual(times[-1], 3.999999)

    def test_openrouter_request_and_validation(self):
        fields = {key: f'{key} draft' for _, _, key in inference_worker.FIELDS}
        sections = {key: {'content': f'{key} draft', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}
        fields['cut_summary'] = inference_worker.render_summary(sections)
        inference_worker.finalize_frame_recommendation(sections, [])
        expected_summary = inference_worker.render_summary(sections)
        response = {'model': 'provider/concrete', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({**fields, **sections})}}]}
        config = {'openrouter_model': 'openrouter/free'}
        observations = [{'time': .05, 'description': 'Visible figure'}]
        progress = []
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode())) as call:
            result, model = inference_worker.synthesize(config, {'additional_context': 'night'}, observations, progress=lambda phase, attempt, total: progress.append((phase, attempt, total)))
        self.assertEqual(result['cut_summary'], expected_summary)
        self.assertEqual(result['visual_evidence'], fields['visual_evidence'])
        self.assertEqual(result['text_evidence'], fields['text_evidence'])
        self.assertEqual(model, 'provider/concrete')
        self.assertEqual(progress, [('evidence', 1, 3), ('summary', 1, 3)])
        request = call.call_args_list[0].args[0]
        self.assertEqual(request.full_url, 'https://openrouter.ai/api/v1/chat/completions')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'openrouter/free')
        self.assertEqual(json.loads(payload['messages'][1]['content'])['sampled_frames'], observations)
        summary_payload = json.loads(call.call_args_list[1].args[0].data)
        self.assertEqual(json.loads(summary_payload['messages'][1]['content'])['assessments']['visual_evidence'], fields['visual_evidence'])
        self.assertNotIn('secret', request.data.decode())
        response['choices'][0]['finish_reason'] = 'length'
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                inference_worker.synthesize(config, {}, observations)

    def test_summary_sections_are_required_and_rendered_in_fixed_order(self):
        sections = {key: {'content': f'Body for {key}', 'confidence': 'medium'} for key, _ in reversed(inference_worker.SUMMARY_TOPICS)}
        incomplete = dict(sections)
        del incomplete['camera_movement']
        def reply(value):
            return io.BytesIO(json.dumps({'model': 'test/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(incomplete), reply(sections)]) as call:
            output, _ = inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', [key for key, _ in inference_worker.SUMMARY_TOPICS], summary_topics=True)
        rendered = inference_worker.render_summary(output)
        self.assertEqual(rendered, '\n\n'.join(f'{title} · Confidence: Medium\nBody for {key}' for key, title in inference_worker.SUMMARY_TOPICS))
        repair = json.loads(call.call_args_list[1].args[0].data)
        self.assertIn('camera_movement', repair['messages'][-1]['content'])
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(incomplete), reply(incomplete), reply(incomplete)]):
            with self.assertRaisesRegex(RuntimeError, 'camera_movement'):
                inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', [key for key, _ in inference_worker.SUMMARY_TOPICS], summary_topics=True)

    def test_summary_dialogue_prompt_uses_queued_transcript_for_analysis_and_regeneration(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin) VALUES ('t1','e',0.2,1,'Hello there.','Speaker A','ctx','manual'),('t2','e',1.8,2.2,'Wait!','',NULL,'model')")
        requests = []
        def provider(request, **kwargs):
            payload = json.loads(request.data)
            system = payload['messages'][0]['content']
            evidence = json.loads(payload['messages'][1]['content'])
            requests.append((system, evidence))
            if 'assessments' not in evidence and 'mode' not in evidence:
                value = {'visual_evidence': 'Visible figure', 'text_evidence': 'No text available'}
            else:
                value = {key: {'content': 'Not established from available evidence.', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
                value['spoken_dialogue'] = {'content': 'Speaker A (Hero), 0.2–1s: Hello there. Words: medium; speaker: medium. Unknown speaker, 1.8–2.2s: Wait! Words: low; speaker: unknown; crosses the cut boundary.', 'confidence': 'unknown'}
            return io.BytesIO(json.dumps({'model': 'test/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', return_value=[{'time': 0, 'description': 'A figure'}]), patch('urllib.request.urlopen', side_effect=provider):
            inference_worker.process(*worker.claim_job(('analyze',)))
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
            regen_id = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
            inference_worker.process_regen(*worker.claim_job(('regen',)))
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (regen_id,))['status'], 'done')
        self.assertEqual(requests[1][0], inference_worker.SUMMARY_SYSTEM)
        self.assertEqual(requests[2][0], inference_worker.SUMMARY_TEXT_SYSTEM)
        for system, evidence in (requests[1], requests[2]):
            self.assertIn('every supplied current transcript segment', system)
            self.assertIn('separate qualitative confidence for the words and the speaker attribution', system)
            self.assertIn('cannot independently hear voices', system)
            segments = evidence['context']['transcript_segments']
            self.assertEqual([item['text'] for item in segments], ['Hello there.', 'Wait!'])
            self.assertEqual(segments[0]['character_name'], 'Hero')
            self.assertEqual(segments[1]['end_time'], 2.2)
        current = next(item for item in self.client.get('/api/cuts/c1/analysis-results').json() if item['title'] == 'Cut summary')
        self.assertIn('Spoken dialogue by speaker · Confidence: Unknown', current['content'])
        self.assertIn('Hello there.', current['content'])
        self.assertIn('Wait!', current['content'])

    def test_summary_requires_spoken_dialogue_section(self):
        sections = {key: {'content': 'Evidence', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
        incomplete = {key: value for key, value in sections.items() if key != 'spoken_dialogue'}
        def reply(value):
            return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(incomplete), reply(sections)]) as call:
            output, _ = inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', [key for key, _ in inference_worker.SUMMARY_TOPICS], summary_topics=True)
        self.assertIn('spoken_dialogue', output)
        self.assertIn('spoken_dialogue', json.loads(call.call_args_list[1].args[0].data)['messages'][-1]['content'])

    def test_summary_confidence_labels_are_validated_and_repaired(self):
        sections = {key: {'content': f'Body for {key}', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
        invalid = {**sections, 'lighting': {'content': 'Bright', 'confidence': '95%'}}
        def reply(value):
            return io.BytesIO(json.dumps({'model': 'test/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        keys = [key for key, _ in inference_worker.SUMMARY_TOPICS]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(invalid), reply(sections)]) as call:
            output, _ = inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', keys, summary_topics=True)
        self.assertIn('lighting confidence must be', json.loads(call.call_args_list[1].args[0].data)['messages'][-1]['content'])
        self.assertEqual(inference_worker.render_summary(output).count('Confidence: Unknown'), len(keys))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply(invalid), reply(invalid), reply(invalid)]):
            with self.assertRaisesRegex(RuntimeError, 'lighting confidence'):
                inference_worker.chat_json('test/model', inference_worker.SUMMARY_SYSTEM, '{}', keys, summary_topics=True)

    def test_openrouter_accepts_structured_evidence(self):
        payload = {"visual_evidence": [{"observation": "coat", "confidence": "low"}], "text_evidence": [{"reading": "OPEN", "category": "scene text"}]}
        response = {'model': 'provider/x', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(payload)}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            result, _ = inference_worker.chat_json('test', 'system', 'evidence', ['visual_evidence', 'text_evidence'])
        self.assertEqual(json.loads(result['visual_evidence'])[0]['confidence'], 'low')
        self.assertEqual(json.loads(result['text_evidence'])[0]['reading'], 'OPEN')

    def test_openrouter_retries_truncation_without_echoing_partial_output(self):
        def reply(content, reason='stop'):
            return io.BytesIO(json.dumps({'model': 'provider/free', 'choices': [{'finish_reason': reason, 'message': {'content': content}}]}).encode())
        for content, reason in ((None, 'length'), (None, 'stop'), ('partial output', 'length')):
            with self.subTest(reason=reason, content=content), patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=[reply(content, reason), reply('{"visual_evidence":"visible"}')]) as call:
                progress = []
                output, model = inference_worker.chat_json('openrouter/free', 'system', 'evidence', ['visual_evidence'], progress=lambda a, n: progress.append((a, n)))
            self.assertEqual((output['visual_evidence'], model), ('visible', 'provider/free'))
            retry = json.loads(call.call_args_list[1].args[0].data)
            self.assertEqual(retry['max_tokens'], 12288)
            self.assertEqual(retry['messages'][1]['content'], 'evidence')
            self.assertNotIn('partial output', json.dumps(retry))
            self.assertEqual(progress, [(1, 3), (2, 3)])

    def test_openrouter_retries_transient_failures_but_not_authentication(self):
        good = io.BytesIO(json.dumps({'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': '{"text_evidence":"none"}'}}]}).encode())
        rate_limited = urllib.error.HTTPError('https://example.invalid', 429, 'rate limit', {}, None)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=[rate_limited, TimeoutError(), good]) as call, patch.object(inference_worker.time, 'sleep') as sleep:
            result, _ = inference_worker.chat_json('openrouter/free', 'system', 'evidence', ['text_evidence'])
        self.assertEqual(result['text_evidence'], 'none')
        self.assertEqual(call.call_count, 3)
        self.assertEqual([item.args[0] for item in sleep.call_args_list], [2, 4])
        unauthorized = urllib.error.HTTPError('https://example.invalid', 401, 'auth', {}, None)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=unauthorized) as call:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401'):
                inference_worker.chat_json('openrouter/free', 'system', 'evidence', ['text_evidence'])
        self.assertEqual(call.call_count, 1)

    def test_openrouter_repairs_missing_fields_once(self):
        fields = {key: f'{key} draft' for _, _, key in inference_worker.FIELDS}
        sections = {key: {'content': f'{key} draft', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}
        fields['cut_summary'] = inference_worker.render_summary(sections)
        incomplete = dict(fields)
        del incomplete['visual_evidence']
        first = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(incomplete)}}]}
        repaired = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({**fields, **sections})}}]}
        replies = [io.BytesIO(json.dumps(reply).encode()) for reply in (first, repaired, repaired)]
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret'}), patch('urllib.request.urlopen', side_effect=replies) as call:
            result, model = inference_worker.synthesize({'openrouter_model': 'openrouter/free'}, {}, [])
        inference_worker.finalize_frame_recommendation(sections, [])
        fields['cut_summary'] = inference_worker.render_summary(sections)
        self.assertEqual({key: result[key] for key in fields}, fields)
        self.assertEqual(model, 'provider/free')
        repair_payload = json.loads(call.call_args_list[1].args[0].data)
        self.assertIn('visual_evidence', repair_payload['messages'][-1]['content'])
        self.assertEqual(call.call_count, 3)

    def test_subtitles_default_and_prompt_rule(self):
        config = self.client.get('/api/cuts/c1/analysis-config').json()
        self.assertIn('subtitles', config['detection_queries'])
        self.assertIn('Subtitle rule', inference_worker.SYSTEM_PROMPT)
        self.assertIn('Subtitle rule', inference_worker.ANIMATION_SYSTEM)

    def test_running_summary_cancellation_unblocks_worker_and_preserves_result(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content) VALUES ('old-summary','c1','cut_description','Cut summary','Keep this')")
        job_id = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'edit', 'instruction': 'Be concise'}).json()['job_id']
        next_id = self.client.post('/api/cuts/c2/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
        claimed = worker.claim_job(('regen',))
        self.assertEqual(claimed[0], job_id)
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        sections = {key: {'content': 'New text', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS}
        def slow_provider(request, **kwargs):
            entered.set()
            try:
                if not release.wait(8):
                    raise RuntimeError('Test did not release provider')
                return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(sections)}}]}).encode())
            finally:
                returned.set()
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=slow_provider):
            thread = threading.Thread(target=inference_worker.process_regen, args=claimed)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                response = self.client.post('/api/jobs/' + job_id + '/cancel')
                self.assertEqual(response.status_code, 200, response.text)
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive(), 'Worker should stop waiting for the provider')
                job = db.one('SELECT status,stage,error,lease_owner,finished_at FROM jobs WHERE id=?', (job_id,))
                self.assertEqual((job['status'], job['stage'], job['error'], job['lease_owner']), ('failed', 'Cancelled', 'Cancelled during processing', None))
                self.assertIsNotNone(job['finished_at'])
                self.assertEqual(worker.claim_job(('regen',))[0], next_id)
            finally:
                release.set()
                thread.join(timeout=5)
                returned.wait(5)
        result = self.client.get('/api/cuts/c1/analysis-results').json()
        self.assertEqual([(item['id'], item['content']) for item in result], [('old-summary', 'Keep this')])
        self.assertEqual(self.client.post('/api/jobs/' + job_id + '/cancel').status_code, 409)

    def test_running_visual_analysis_cannot_be_cancelled(self):
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        self.assertEqual(worker.claim_job(('analyze',))[0], job_id)
        self.assertEqual(self.client.post('/api/jobs/' + job_id + '/cancel').status_code, 409)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'running')

    def test_summary_jobs_for_different_cuts_can_be_queued_without_duplicate_same_cut(self):
        first = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).status_code, 409)
        other = self.client.post('/api/cuts/c2/regenerate-summary', json={'mode': 'regenerate'})
        self.assertEqual(other.status_code, 200, other.text)
        self.assertEqual(self.client.post('/api/jobs/' + first.json()['job_id'] + '/cancel').status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).status_code, 200)

    def test_cut_summary_regeneration_creates_revision(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('anim','c1','generation_prompt','Cut summary','Old prompt','model'),('shot','c1','cut_description','Shot description','A figure','model')")
        self.assertEqual(self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'edit', 'instruction': ' '}).status_code, 400)
        queued = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'edit', 'instruction': 'Slower camera', 'current_prompt': 'Unsaved prompt'})
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(self.client.post('/api/cuts/c2/regenerate-summary', json={'mode': 'regenerate'}).status_code, 200)
        job_id = queued.json()['job_id']
        payload = json.loads(db.one("SELECT payload_json FROM jobs WHERE id=?", (job_id,))['payload_json'])
        self.assertEqual((payload['base_id'], payload['current_prompt']), ('anim', 'Unsaved prompt'))
        self.assertEqual([item['title'] for item in payload['results']], ['Shot description'])
        claimed = worker.claim_job(('regen',))
        response = {'model': 'provider/x', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({key: {'content': 'New prompt', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS})}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'k'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            inference_worker.process_regen(*claimed)
        results = self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()
        current = next(item for item in results if item['title'] == 'Cut summary')
        expected = {key: {'content': 'New prompt', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}
        inference_worker.finalize_frame_recommendation(expected, [])
        self.assertEqual((current['content'], current['parent_id'], current['origin']), (inference_worker.render_summary(expected), 'anim', 'model'))
        self.assertEqual(current['model_id'], 'provider/x')
        self.assertGreaterEqual(current['settings']['summary_generation_seconds'], 0)
        self.assertEqual(db.one("SELECT superseded_by FROM analysis_results WHERE id='anim'")['superseded_by'], current['id'])
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))['status'], 'done')

    def test_composition_recommendation_download_and_regeneration(self):
        folder = media.episode_dir('p', 'e') / 'analysis'
        folder.mkdir(exist_ok=True)
        (folder.parent / 'original.mp4').write_bytes(b'video placeholder')
        (folder / 'sample-a.jpg').write_bytes(b'first jpeg')
        (folder / 'sample-b.jpg').write_bytes(b'better jpeg')
        observations = [
            {'time': .1, 'description': 'A distant figure', 'ocr_response': '', 'ocr_prompt': 'ocr', 'frame_file': 'sample-a.jpg'},
            {'time': 1.25, 'description': 'Expressive face and clear pose', 'ocr_response': '', 'ocr_prompt': 'ocr', 'frame_file': 'sample-b.jpg'},
        ]
        requests = []
        def provider(request, **kwargs):
            payload = json.loads(request.data)
            evidence = json.loads(payload['messages'][1]['content'])
            requests.append(evidence)
            if 'assessments' not in evidence and 'mode' not in evidence:
                value = {'visual_evidence': 'Two samples', 'text_evidence': 'None'}
            else:
                value = {key: {'content': 'Evidence-based rationale', 'confidence': 'medium'} for key, _ in inference_worker.SUMMARY_TOPICS}
                value['suggested_composition_frame']['frame_id'] = evidence['sampled_frames'][-1]['frame_id']
            return io.BytesIO(json.dumps({'model': 'test/model', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}).encode())
        job_id = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(inference_worker, 'describe_frames', return_value=observations), patch('urllib.request.urlopen', side_effect=provider):
            inference_worker.process(*worker.claim_job(('analyze',)))
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
            results = self.client.get('/api/cuts/c1/analysis-results').json()
            summary = next(result for result in results if result['title'] == 'Cut summary')
            chosen = requests[-1]['sampled_frames'][-1]['frame_id']
            self.assertIn('Suggested composition frame · Confidence: Medium', summary['content'])
            self.assertEqual(summary['model_id'], 'test/model')
            self.assertGreaterEqual(summary['settings']['summary_generation_seconds'], 0)
            self.assertIn('Sample time: 1.250000 seconds', summary['content'])
            self.assertIn(f'/api/analysis-results/{chosen}/frame', summary['content'])
            with patch('app.frame_cache._generate', return_value=b'better jpeg'):
                self.assertEqual(self.client.get(f'/api/analysis-results/{chosen}/frame').content, b'better jpeg')
            regen_id = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
            inference_worker.process_regen(*worker.claim_job(('regen',)))
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (regen_id,))['status'], 'done')
            current = next(result for result in self.client.get('/api/cuts/c1/analysis-results').json() if result['title'] == 'Cut summary')
            self.assertIn(f'/api/analysis-results/{chosen}/frame', current['content'])
            self.assertEqual(self.client.delete(f'/api/analysis-results/{chosen}').status_code, 200)
            self.assertEqual(self.client.get(f'/api/analysis-results/{chosen}/frame').status_code, 404)

    def test_invalid_composition_frame_is_retried(self):
        sections = {key: {'content': 'Rationale', 'confidence': 'low'} for key, _ in inference_worker.SUMMARY_TOPICS}
        candidates = [{'frame_id': 'real-id', 'time': .5}]
        def reply(frame_id):
            sections['suggested_composition_frame']['frame_id'] = frame_id
            return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(sections)}}]}).encode())
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', side_effect=[reply('invented'), reply('real-id')]) as call:
            output, _ = inference_worker.chat_json('model', 'system', '{}', [key for key, _ in inference_worker.SUMMARY_TOPICS], summary_topics=True, frame_candidates=candidates)
        self.assertEqual(output['suggested_composition_frame']['frame_id'], 'real-id')
        self.assertIn('must match a supplied', json.loads(call.call_args_list[1].args[0].data)['messages'][-1]['content'])

    def test_summary_edit_accepts_partial_response_and_invalid_frame_without_retry(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('old','c1','cut_description','Cut summary','Original','model'),('real-frame','c1','first_frame','Frame at 0.500s','A frame','model')")
        folder = media.episode_dir('p', 'e') / 'analysis'
        folder.mkdir(exist_ok=True)
        (folder / 'real-frame.jpg').write_bytes(b'frame')
        job_id = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'edit', 'instruction': 'Expand dialogue'}).json()['job_id']
        content = json.dumps({'overview': {'content': 'Useful analysis', 'confidence': 'uncertain'}, 'suggested_composition_frame': {'content': 'A suggested view', 'frame_id': 'invented'}})
        response = {'model': 'provider/free', 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
            inference_worker.process_regen(*worker.claim_job(('regen',)))
        self.assertEqual(call.call_count, 1)
        self.assertNotIn('response_format', json.loads(call.call_args.args[0].data))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        result = next(item for item in self.client.get('/api/cuts/c1/analysis-results').json() if item['title'] == 'Cut summary')
        self.assertIn('Useful analysis', result['content'])
        self.assertIn('A suggested view', result['content'])
        self.assertNotIn('/api/analysis-results/invented/frame', result['content'])
        self.assertEqual(result['parent_id'], 'old')

    def test_summary_only_accepts_plain_text_and_marks_truncated_response(self):
        for finish, expected_suffix in [('stop', ''), ('length', "[The model's response was cut off and may be incomplete.]")]:
            with self.subTest(finish=finish):
                response = {'model': 'test/model', 'choices': [{'finish_reason': finish, 'message': {'content': 'Speaker A: Hello.'}}]}
                with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as call:
                    content, _ = inference_worker.chat_json('test/model', inference_worker.SUMMARY_TEXT_SYSTEM, '{}', [], plain_text=True)
                self.assertEqual(call.call_count, 1)
                self.assertTrue(content.startswith('Speaker A: Hello.'))
                self.assertTrue(content.endswith(expected_suffix))
        partial = '{"overview":{"content":"First section"},"spoken_dialogue":{"content":"Voice: hello"}' + "\n\n[The model's response was cut off and may be incomplete.]"
        rendered = inference_worker.render_relaxed_summary(partial, [])
        self.assertIn('First section', rendered)
        self.assertIn('Voice: hello', rendered)
        self.assertIn('response was cut off', rendered)
        self.assertNotIn('"overview"', rendered)

    def test_summary_regeneration_fails_when_changed_meanwhile(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('anim','c1','generation_prompt','Cut summary','Old','model')")
        job_id = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
        claimed = worker.claim_job(('regen',))
        self.client.put('/api/analysis-results/anim', json={'title': 'Cut summary', 'content': 'Edited by hand'})
        response = {'model': 'm', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({key: {'content': 'New', 'confidence': 'unknown'} for key, _ in inference_worker.SUMMARY_TOPICS})}}]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'k'}), patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            inference_worker.process_regen(*claimed)
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))['status'], 'failed')

    def test_result_frame_follows_revision_chain(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('det','c1','other','Object detections at 0.000s','{}','model')")
        self.assertEqual(self.client.get('/api/analysis-results/det/frame').status_code, 404)
        folder = media.episode_dir('p', 'e') / 'analysis'
        folder.mkdir()
        (folder / 'det.jpg').write_bytes(b'jpeg')
        revised = self.client.put('/api/analysis-results/det', json={'title': 'x', 'content': '{}'}).json()
        self.assertEqual(self.client.get(f"/api/analysis-results/{revised['id']}/frame").content, b'jpeg')

    def test_delete_result_reverts_revision_without_bulk_deletion(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('a','c1','other','A','one','model'),('d1','c1','other','D1','x','model'),('d2','c1','other','D2','y','model')")
        revised = self.client.put('/api/analysis-results/a', json={'title': 'A', 'content': 'two'}).json()
        self.assertEqual(self.client.delete('/api/analysis-results/a').status_code, 409)
        self.assertEqual(self.client.delete(f"/api/analysis-results/{revised['id']}").status_code, 200)
        self.assertIsNone(db.one("SELECT superseded_by FROM analysis_results WHERE id='a'")['superseded_by'])
        self.assertEqual(self.client.delete('/api/cuts/c1/analysis-results').status_code, 405)
        remaining = [item['id'] for item in self.client.get('/api/cuts/c1/analysis-results?include_context=false').json()]
        self.assertEqual(set(remaining), {'a', 'd1', 'd2'})
        self.assertEqual(self.client.delete('/api/analysis-results/missing').status_code, 404)


if __name__ == '__main__':
    unittest.main()
