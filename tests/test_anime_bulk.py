import io
import json
import unittest
from unittest.mock import patch

from PIL import Image

import tests.test_analysis as analysis_tests
from app import anime_bulk, anime_models, character_detection, db, worker


def jpeg():
    output = io.BytesIO()
    Image.new('RGB', (30, 60), 'red').save(output, format='JPEG')
    return output.getvalue()


class Detector:
    revision = 'aniref-sha'
    def detect(self, images):
        assert len(images) == 2
        return [[[5, 5, 30, 60]], [[5, 5, 30, 60]]]


class ReId:
    revision = 'reid-sha'
    def __init__(self, target_vectors):
        self.target_vectors = target_vectors
        self.calls = []
    def embed(self, images):
        self.calls.append(len(images))
        # Reference snapshots are ordered newest first: ctx2, then ctx.
        return [[0, 1], [1, 0]] if len(self.calls) == 1 else self.target_vectors


class Runtime:
    def __init__(self, reid):
        self.reid = reid
        self.reference_images = {}
        self.prepared_frames = {}
        self.same_cut_links = {}
        self.detector = Detector()
    def get_detector(self): return self.detector
    def get_reid(self): return self.reid


class BulkDetectionTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def add_references(self):
        frames = [{'position': 'first', 'time': 0, 'image_size': [100, 80]}]
        with db.connect() as conn:
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('ctx2','p','character','Other')")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('old','c1',0,2,?,'test')", (json.dumps(frames),))
            for index, character in enumerate(('ctx', 'ctx2')):
                conn.execute('INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES (?,?,?,?,?,?)',
                    (f'old{index}', 'old', 'first', '[5,5,30,60]', character, 'manual'))
                conn.execute('INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES (?,?,?)',
                    (f'ref{index}', character, f'old{index}'))

    def run_bulk(self, runtime, reviews, last_time=3.9):
        queued = self.client.post('/api/cuts/c2/character-detection')
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id, owner = worker.claim_job(('character_detect',))
        frames = [{'position': 'first', 'time': 2, 'image_size': [100, 80]},
                  {'position': 'last', 'time': last_time, 'image_size': [100, 80]}]
        images = [Image.new('RGB', (100, 80), 'red')] * 2
        with patch.object(anime_bulk, '_extract_frames', return_value=(frames, images)), \
             patch.object(character_detection, 'crop_frame', return_value=jpeg()), \
             patch.object(anime_bulk, '_review', side_effect=reviews) as review:
            anime_bulk.process(job_id, owner, runtime)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        return self.client.get('/api/cuts/c2/character-detections').json(), review

    def test_confident_embedding_bypasses_qwen_ambiguous_match_uses_qwen(self):
        self.add_references()
        runtime = Runtime(ReId([[1, 0], [1, 1]]))
        run, review = self.run_bulk(runtime, [{'character_id': 'ctx2', 'confidence': 'high', 'reason': 'distinctive features'}])
        self.assertEqual(review.call_count, 1)
        self.assertEqual(run['model_id'], anime_models.ANIREF_MODEL_ID)
        self.assertEqual([item['character_context_id'] for item in run['detections']], ['ctx', 'ctx2'])
        self.assertEqual([item['label_source'] for item in run['detections']], ['auto', 'auto'])
        self.assertEqual(run['detections'][0]['match_model_id'], anime_models.REID_MODEL_ID)
        self.assertEqual(run['detections'][1]['match_model_id'], anime_bulk.frame_model.MODEL_ID)
        self.assertEqual(len(db.all_rows('SELECT * FROM character_reference_embeddings')), 2)
        self.assertEqual(len(db.all_rows('SELECT * FROM character_detection_embeddings')), 2)
        saved = self.client.post(f"/api/character-detections/{run['detections'][0]['id']}/reference")
        self.assertEqual(saved.status_code, 200, saved.text)
        cached = db.one('SELECT * FROM character_reference_embeddings WHERE reference_id=?', (saved.json()['id'],))
        self.assertEqual(cached['model_revision'], 'reid-sha')

    def test_no_references_detects_boxes_without_loading_reid_or_qwen(self):
        runtime = Runtime(ReId([]))
        run, review = self.run_bulk(runtime, [])
        self.assertEqual(len(run['detections']), 2)
        self.assertTrue(all(item['character_context_id'] is None for item in run['detections']))
        self.assertEqual(review.call_count, 0)
        self.assertEqual(runtime.reid.calls, [])

    def test_short_unambiguous_track_reuses_reviewed_identity(self):
        self.add_references()
        runtime = Runtime(ReId([[1, 1], [1, 1]]))
        run, review = self.run_bulk(runtime, [{'character_id': 'ctx2', 'confidence': 'high', 'reason': 'reviewed'}], last_time=2.5)
        self.assertEqual(review.call_count, 1)
        self.assertEqual([item['character_context_id'] for item in run['detections']], ['ctx2', 'ctx2'])
        self.assertIn('track', run['detections'][1]['match_reason'])

    def test_deferred_qwen_review_resumes_and_still_links_the_same_cut_track(self):
        self.add_references()
        runtime = Runtime(ReId([[1, 1], [1, 1]]))
        self.assertEqual(self.client.post('/api/cuts/c2/character-detection').status_code, 200)
        job_id, owner = worker.claim_job(('character_detect',))
        frames = [{'position': 'first', 'time': 2, 'image_size': [100, 80]}, {'position': 'last', 'time': 2.5, 'image_size': [100, 80]}]
        images = [Image.new('RGB', (100, 80), 'red')] * 2
        reviews = [{'character_id': 'ctx2', 'confidence': 'high', 'reason': 'reviewed'}]
        with patch.object(anime_bulk, '_extract_frames', return_value=(frames, images)), \
             patch.object(character_detection, 'crop_frame', return_value=jpeg()), \
             patch.object(anime_bulk, '_review', side_effect=reviews) as review:
            self.assertEqual(anime_bulk.process(job_id, owner, runtime, defer_qwen=True), anime_bulk.DEFERRED)
            self.assertEqual(review.call_count, 0)
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'running')
            anime_bulk.process(job_id, owner, runtime)
        self.assertEqual(review.call_count, 1)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        run = self.client.get('/api/cuts/c2/character-detections').json()
        self.assertEqual([item['character_context_id'] for item in run['detections']], ['ctx2', 'ctx2'])
        self.assertIn('track', run['detections'][1]['match_reason'])

    def test_adjacent_cut_reuses_a_unique_appearance_without_another_qwen_call(self):
        self.add_references()
        runtime = Runtime(ReId([[1, 1], [1, 1]]))
        runtime.previous_cut = {'episode_id': 'e', 'end_time': 2, 'tracks': [{'character_id': 'ctx2', 'embedding': [1, 1]}]}
        run, review = self.run_bulk(runtime, [], last_time=2.5)
        self.assertEqual(review.call_count, 0)
        self.assertEqual([item['character_context_id'] for item in run['detections']], ['ctx2', 'ctx2'])
        self.assertIn('preceding cut', run['detections'][0]['match_reason'])

    def test_multiple_cut_jobs_share_a_single_detector_batch_and_keep_separate_results(self):
        jobs = self.client.post('/api/episodes/e/character-detection-all').json()['job_ids']
        with db.connect() as conn:
            payloads = [json.loads(conn.execute('SELECT payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()['payload_json']) for job_id in jobs]
        self.assertEqual([payload['workflow'] for payload in payloads], ['anime_bulk_v1', 'anime_bulk_v1'])
        self.assertEqual(payloads[0]['batch_id'], payloads[1]['batch_id'])
        first, owner = worker.claim_job(('character_detect',))
        runtime = Runtime(ReId([]))
        calls = []
        def detect(images):
            calls.append(len(images))
            return [[[5, 5, 30, 60]]] * len(images)
        def frames(original, cut, scratch, saved=None):
            return ([{'position': 'first', 'time': cut['start_time'], 'image_size': [100, 80]},
                     {'position': 'last', 'time': cut['end_time'] - .1, 'image_size': [100, 80]}],
                    [Image.new('RGB', (100, 80), 'red')] * 2)
        with patch.object(runtime.detector, 'detect', side_effect=detect), patch.object(anime_bulk, '_extract_frames', side_effect=frames):
            anime_bulk.process_batch(db.one('SELECT * FROM jobs WHERE id=?', (first,)), owner, runtime)
        self.assertEqual(calls, [4])
        self.assertEqual([db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'] for job_id in jobs], ['done', 'done'])
        self.assertEqual([len(self.client.get(f'/api/cuts/{cut}/character-detections').json()['detections']) for cut in ('c1', 'c2')], [2, 2])

    def test_single_cut_queue_keeps_original_workflow(self):
        job_id = self.client.post('/api/cuts/c1/character-detection').json()['job_id']
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])
        self.assertNotEqual(payload.get('workflow'), 'anime_bulk_v1')

    def test_bulk_snapshots_all_user_references_while_single_cut_keeps_its_limit(self):
        self.add_references()
        with db.connect() as conn:
            for index in range(2, 4):
                conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES (?,'old','first','[5,5,30,60]','ctx','manual')", (f'old{index}',))
                conn.execute("INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES (?,'ctx',?)", (f'ref{index}', f'old{index}'))
        single = self.client.post('/api/cuts/c2/character-detection').json()['job_id']
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (single,))['payload_json'])
        self.assertEqual(sum(ref['character_context_id'] == 'ctx' for ref in payload['references']), 2)
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done' WHERE id=?", (single,))
        bulk = self.client.post('/api/episodes/e/character-detection-all').json()['job_ids']
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (bulk[0],))['payload_json'])
        self.assertEqual(sum(ref['character_context_id'] == 'ctx' for ref in payload['references']), 3)

    def test_qwen_review_limits_reference_images_per_character(self):
        refs = [{'character_context_id': character, 'id': f'{character}{index}'}
                for character in ('a', 'b', 'c') for index in range(4)]
        runtime = Runtime(ReId([]))
        runtime.get_qwen = lambda check: 'server'
        with patch.object(anime_bulk.frame_model, 'match_character', return_value=None) as match:
            self.assertIsNone(anime_bulk._review(runtime, b'target', refs, lambda: None))
        self.assertEqual([len(call.args[2]) for call in match.call_args_list], [4, 2])


if __name__ == '__main__':
    unittest.main()
