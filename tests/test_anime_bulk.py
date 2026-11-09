import io
import json
import math
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import tests.test_analysis as analysis_tests
from app import anime_bulk, anime_gallery, anime_matching, anime_models, character_detection, db, worker

RED, BLUE, PURPLE = (220, 0, 0), (0, 0, 220), (150, 0, 150)
LEFT, RIGHT = [5, 5, 45, 75], [55, 5, 95, 75]


def jpeg(color=RED, size=(40, 70)):
    output = io.BytesIO()
    Image.new('RGB', size, color).save(output, format='JPEG')
    return output.getvalue()


def frame_image(boxes):
    image = Image.new('RGB', (100, 80), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for box, color in boxes:
        draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=color)
    return image


class Embedder:
    """Colour-based stand-in for CCIP: the vector is the crop's mean RGB."""
    model_id = 'test/ccip'
    revision = 'emb-sha'
    preprocessing = 'test-v1'
    label = 'CCIP'
    defaults = dict(anime_models.CCIP_DEFAULTS)

    def __init__(self):
        self.calls = []

    def embed(self, images):
        self.calls.append(len(images))
        vectors = []
        for image in images:
            pixels = list(image.convert('RGB').getdata())
            mean = [sum(p[i] for p in pixels) / len(pixels) + 1 for i in range(3)]
            norm = math.sqrt(sum(v * v for v in mean))
            vectors.append([v / norm for v in mean])
        return vectors


class Detector:
    revision = 'aniref-sha'

    def __init__(self, boxes):
        self.boxes = boxes  # [first-frame boxes, last-frame boxes]
        self.calls = []

    def detect(self, images):
        self.calls.append(len(images))
        return [self.boxes[i % 2] for i in range(len(images))]


class Runtime(anime_bulk.BulkRuntime):
    def __init__(self, embedder=None, boxes=None):
        super().__init__()
        self.detector = Detector(boxes or [[LEFT], [LEFT]])
        self.stub = embedder
        self.embedder_loads = 0

    def get_detector(self):
        return self.detector

    def get_embedder(self):
        self.embedder_loads += 1
        if self.stub is None:
            raise AssertionError('the embedder should not be loaded')
        return self.stub

    def get_qwen(self, lease_check):
        return 'server'


class BulkDetectionTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def add_examples(self, examples):
        """examples: (detection id, cut id, character id, colour, box). Cuts are created as needed."""
        self.colors = getattr(self, 'colors', {})
        with db.connect() as conn:
            conn.execute("INSERT OR IGNORE INTO context_items(id,project_id,kind,title) VALUES ('ctx2','p','character','Other')")
            for detection_id, cut_id, character_id, color, box in examples:
                if not conn.execute('SELECT 1 FROM cuts WHERE id=?', (cut_id,)).fetchone():
                    index = conn.execute("SELECT count(*) FROM cuts").fetchone()[0]
                    conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,analysis_enabled) VALUES (?,'e',?,?,?,0)",
                                 (cut_id, index, 10 + index * 2, 12 + index * 2))
                run_id = f'run-{cut_id}'
                if not conn.execute('SELECT 1 FROM character_detection_runs WHERE id=?', (run_id,)).fetchone():
                    frames = [{'position': 'first', 'time': 0, 'image_size': [100, 80]}, {'position': 'last', 'time': 1, 'image_size': [100, 80]}]
                    conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES (?,?,0,2,?,'test')",
                                 (run_id, cut_id, json.dumps(frames)))
                conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES (?,?,'first',?,?,'manual')",
                             (detection_id, run_id, json.dumps(box or LEFT), character_id))
                self.colors[detection_id] = color

    def standard_gallery(self):
        self.add_examples([(f'red{i}', f'g{i}', 'ctx', RED, None) for i in range(3)]
                          + [(f'blue{i}', f'h{i}', 'ctx2', BLUE, None) for i in range(3)])

    def fake_crops(self, rows, scratch, lease_check=None):
        return {row['id']: jpeg(self.colors[row['id']]) for row in rows}

    def run_episode(self, runtime, first, last, path='character-detection-all', body=None, review=None):
        response = self.client.post(f'/api/episodes/e/{path}', json=body or {})
        self.assertEqual(response.status_code, 200, response.text)
        job_id, owner = worker.claim_job(('character_detect',))

        def frames(original, cut, cache, saved=None):
            if saved:
                return saved, [frame_image(first), frame_image(last)]
            return ([{'position': 'first', 'time': cut['start_time'], 'image_size': [100, 80]},
                     {'position': 'last', 'time': cut['end_time'] - .5, 'image_size': [100, 80]}],
                    [frame_image(first), frame_image(last)])
        with patch.object(anime_bulk, '_extract_frames', side_effect=frames), \
             patch.object(anime_gallery, 'extract_crops', side_effect=self.fake_crops), \
             patch.object(anime_bulk.frame_model, 'assign_characters', side_effect=review or AssertionError('Qwen must not be called')) as qwen:
            anime_bulk.process_batch(db.one('SELECT * FROM jobs WHERE id=?', (job_id,)), owner, runtime)
        self.assertEqual({row['status'] for row in db.all_rows("SELECT status FROM jobs WHERE pipeline='character_detect'")}, {'done'})
        return {cut: self.client.get(f'/api/cuts/{cut}/character-detections').json() for cut in ('c1', 'c2')}, qwen

    def test_gallery_match_labels_boxes_and_stores_suggestions(self):
        self.standard_gallery()
        runs, qwen = self.run_episode(Runtime(Embedder(), [[LEFT, RIGHT], []]), [(LEFT, RED), (RIGHT, BLUE)], [])
        detections = runs['c2']['detections']
        self.assertEqual([d['character_context_id'] for d in detections], ['ctx', 'ctx2'])
        self.assertEqual({d['label_source'] for d in detections}, {'auto'})
        self.assertEqual(detections[0]['match_model_id'], 'test/ccip')
        self.assertIn('CCIP similarity', detections[0]['match_reason'])
        self.assertEqual(detections[0]['candidates'][0]['character_name'], 'Hero')
        self.assertEqual([c['character_id'] for c in detections[1]['candidates']], ['ctx2', 'ctx'])
        self.assertEqual(qwen.call_count, 0)
        # Gallery vectors are cached with the model, and the examples are marked as used.
        self.assertEqual(len(db.all_rows("SELECT * FROM character_detection_embeddings WHERE detection_id LIKE 'red%'")), 3)
        self.assertEqual({row['gallery_status'] for row in db.all_rows("SELECT gallery_status FROM character_detections WHERE label_source='manual'")}, {'used'})

    def test_ambiguous_box_keeps_suggestions_and_qwen_is_off_by_default(self):
        self.standard_gallery()
        runs, qwen = self.run_episode(Runtime(Embedder(), [[LEFT], []]), [(LEFT, PURPLE)], [])
        detection = runs['c2']['detections'][0]
        self.assertIsNone(detection['character_context_id'])
        self.assertEqual({c['character_id'] for c in detection['candidates']}, {'ctx', 'ctx2'})
        self.assertEqual(qwen.call_count, 0)
        self.assertEqual(db.one("SELECT phase FROM character_detection_runs WHERE cut_id='c2'")['phase'], 'complete')

    def test_optional_qwen_asks_once_per_cut_within_the_shortlist(self):
        self.standard_gallery()
        answers = {'T1': {'character_id': 'ctx', 'confidence': 'high'}, 'T2': {'character_id': None, 'confidence': 'low'}}
        runs, qwen = self.run_episode(Runtime(Embedder(), [[LEFT, RIGHT], []]), [(LEFT, PURPLE), (RIGHT, (140, 0, 160))], [],
                                      body={'qwen': True}, review=lambda server, targets, candidates, cancel_check=None: answers)
        self.assertEqual(qwen.call_count, 2)  # one request for each of the two cuts
        targets, candidates = qwen.call_args_list[0].args[1:3]
        self.assertEqual([t['key'] for t in targets], ['T1', 'T2'])
        self.assertEqual(set(candidates), {'ctx', 'ctx2'})
        detections = runs['c2']['detections']
        self.assertEqual([d['character_context_id'] for d in detections], ['ctx', None])
        self.assertEqual(detections[0]['match_model_id'], anime_bulk.frame_model.MODEL_ID)

    def test_a_character_is_assigned_once_per_frame(self):
        self.standard_gallery()
        runs, _ = self.run_episode(Runtime(Embedder(), [[LEFT, RIGHT], []]), [(LEFT, RED), (RIGHT, (200, 0, 30))], [])
        labels = [d['character_context_id'] for d in runs['c2']['detections']]
        self.assertEqual(labels.count('ctx'), 1)
        self.assertEqual(labels.count(None), 1)

    def test_manual_label_travels_along_the_same_cut_track(self):
        self.standard_gallery()
        frames = [{'position': 'first', 'time': 2, 'image_size': [100, 80]}, {'position': 'last', 'time': 2.5, 'image_size': [100, 80]}]
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES ('cur','c2',2,4,?,'test','boxes')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json) VALUES ('mine-first','cur','first',?)", (json.dumps(LEFT),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('mine-last','cur','last',?,'ctx2','manual')", (json.dumps(LEFT),))
            conn.execute("UPDATE cuts SET no_characters=1 WHERE id='c1'")
        self.colors['mine-last'] = PURPLE
        with patch.object(anime_matching, 'confident_match', return_value=None):
            runs, _ = self.run_episode(Runtime(Embedder()), [(LEFT, PURPLE)], [(LEFT, PURPLE)], path='character-detection')
        first, last = runs['c2']['detections']
        self.assertEqual((first['character_context_id'], first['label_source']), ('ctx2', 'auto'))
        self.assertIn('track', first['match_reason'])
        self.assertEqual((last['character_context_id'], last['label_source']), ('ctx2', 'manual'))

    def test_adjacent_cut_reuses_a_unique_appearance(self):
        self.standard_gallery()
        runtime = Runtime(Embedder(), [[LEFT], []])
        purple = Embedder().embed([Image.new('RGB', (4, 4), PURPLE)])[0]
        runtime.previous_cut = {'episode_id': 'e', 'end_time': 0, 'tracks': [{'character_id': 'ctx2', 'embedding': purple}]}
        with patch.object(anime_matching, 'confident_match', return_value=None):
            runs, _ = self.run_episode(runtime, [(LEFT, PURPLE)], [])
        detection = runs['c1']['detections'][0]
        self.assertEqual(detection['character_context_id'], 'ctx2')
        self.assertIn('preceding cut', detection['match_reason'])

    def test_rerun_redoes_automatic_labels_and_never_touches_manual_ones(self):
        self.standard_gallery()
        frames = [{'position': 'first', 'time': 2, 'image_size': [100, 80]}, {'position': 'last', 'time': 3.5, 'image_size': [100, 80]}]
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES ('cur','c2',2,4,?,'test','complete')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('mine','cur','first',?,'ctx','manual')", (json.dumps(LEFT),))
            # The user's label is kept even though the box looks like someone else.
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('mine-odd','cur','last',?,'ctx2','manual')", (json.dumps(LEFT),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source,match_reason) VALUES ('old-auto','cur','first',?,'ctx','auto','old Qwen')", (json.dumps(RIGHT),))
            # c1 only has manual labels: nothing to redo there.
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('done','c1',0,2,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('all-mine','done','first',?,'ctx','manual')", (json.dumps(LEFT),))
        self.colors.update({'mine': RED, 'mine-odd': RED, 'all-mine': RED})
        queued = self.client.post('/api/episodes/e/character-detection-rerun', json={'qwen': False})
        self.assertEqual(queued.json()['queued'], 1)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (queued.json()['job_ids'][0],))['payload_json'])
        self.assertEqual((payload['cut_id'], payload['phase'], payload['qwen']), ('c2', 'recognize', False))
        self.assertNotIn('references', payload)
        reset = db.one("SELECT * FROM character_detections WHERE id='old-auto'")
        self.assertEqual((reset['character_context_id'], reset['label_source'], reset['match_reason']), (None, None, None))
        self.assertEqual(db.one("SELECT phase FROM character_detection_runs WHERE id='cur'")['phase'], 'boxes')
        job_id, owner = worker.claim_job(('character_detect',))
        with patch.object(anime_bulk, '_extract_frames', return_value=(frames, [frame_image([(LEFT, RED), (RIGHT, BLUE)]), frame_image([(LEFT, RED)])])), \
             patch.object(anime_gallery, 'extract_crops', side_effect=self.fake_crops):
            anime_bulk.process_batch(db.one('SELECT * FROM jobs WHERE id=?', (job_id,)), owner, Runtime(Embedder()))
        labels = {row['id']: (row['character_context_id'], row['label_source']) for row in db.all_rows('SELECT * FROM character_detections')}
        self.assertEqual(labels['mine'], ('ctx', 'manual'))
        self.assertEqual(labels['mine-odd'], ('ctx2', 'manual'))
        self.assertEqual(labels['all-mine'], ('ctx', 'manual'))
        # It stays labelled for analysis but is not a matching example for Other.
        self.assertEqual(db.one("SELECT gallery_status FROM character_detections WHERE id='mine-odd'")['gallery_status'], 'atypical')
        self.assertEqual(labels['old-auto'], ('ctx2', 'auto'))
        self.assertIn('CCIP', db.one("SELECT match_reason FROM character_detections WHERE id='old-auto'")['match_reason'])
        self.assertEqual(self.client.post('/api/episodes/e/character-detection-rerun').json()['queued'], 1)  # the automatic label can be redone again

    def test_unidentifiable_examples_are_left_out_and_the_user_can_override(self):
        self.standard_gallery()
        self.add_examples([('tiny', 'g9', 'ctx', RED, [0, 0, 8, 8]),        # a hand-sized box
                           ('looks-blue', 'g10', 'ctx', BLUE, None),         # labelled Hero, looks like Other
                           ('forced', 'g11', 'ctx', BLUE, None),
                           ('never', 'g12', 'ctx', RED, None)])
        with db.connect() as conn:
            conn.execute("UPDATE character_detections SET recognition_use=1 WHERE id='forced'")
            conn.execute("UPDATE character_detections SET recognition_use=0 WHERE id='never'")
        runtime = Runtime(Embedder())
        with patch.object(anime_gallery, 'extract_crops', side_effect=self.fake_crops):
            gallery = runtime.get_gallery('p')
        status = {row['id']: (row['gallery_status'], row['gallery_note']) for row in db.all_rows('SELECT * FROM character_detections')}
        self.assertEqual(status['tiny'][0], 'small')
        self.assertEqual(status['looks-blue'], ('atypical', 'Looks more like Other than its own examples'))
        self.assertEqual(status['forced'][0], 'used')
        self.assertEqual(status['never'][0], 'excluded')
        self.assertEqual({item['id'] for item in gallery.items}, {'red0', 'red1', 'red2', 'blue0', 'blue1', 'blue2', 'forced'})
        response = self.client.put('/api/character-detections/looks-blue/recognition-use', json={'use': True})
        self.assertEqual(response.status_code, 200, response.text)
        row = db.one("SELECT recognition_use,gallery_status FROM character_detections WHERE id='looks-blue'")
        self.assertEqual((row['recognition_use'], row['gallery_status']), (1, None))
        self.assertEqual(self.client.put('/api/character-detections/missing/recognition-use', json={'use': None}).status_code, 404)

    def test_without_confirmed_labels_boxes_are_found_but_the_embedder_never_loads(self):
        runtime = Runtime(None, [[LEFT], [LEFT]])
        runs, qwen = self.run_episode(runtime, [(LEFT, RED)], [(LEFT, RED)], body={'qwen': True})
        self.assertEqual([len(runs[cut]['detections']) for cut in ('c1', 'c2')], [2, 2])
        self.assertTrue(all(d['character_context_id'] is None for d in runs['c2']['detections']))
        self.assertEqual(runtime.embedder_loads, 0)
        self.assertEqual(qwen.call_count, 0)

    def test_multiple_cut_jobs_share_a_single_detector_batch_and_keep_separate_results(self):
        jobs = self.client.post('/api/episodes/e/character-detection-all').json()['job_ids']
        payloads = [json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json']) for job_id in jobs]
        self.assertEqual([payload['workflow'] for payload in payloads], ['anime_bulk_v1', 'anime_bulk_v1'])
        self.assertEqual(payloads[0]['batch_id'], payloads[1]['batch_id'])
        first, owner = worker.claim_job(('character_detect',))
        runtime = Runtime(None, [[LEFT], [LEFT]])
        with patch.object(anime_bulk, '_extract_frames', return_value=([{'position': 'first', 'time': 0, 'image_size': [100, 80]},
                                                                        {'position': 'last', 'time': 1, 'image_size': [100, 80]}],
                                                                       [frame_image([(LEFT, RED)])] * 2)):
            anime_bulk.process_batch(db.one('SELECT * FROM jobs WHERE id=?', (first,)), owner, runtime)
        self.assertEqual(runtime.detector.calls, [4])
        self.assertEqual([db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'] for job_id in jobs], ['done', 'done'])

    def test_an_interrupted_batch_does_not_lock_its_episode(self):
        jobs = self.client.post('/api/episodes/e/character-detection-all').json()['job_ids']
        first, owner = worker.claim_job(('character_detect',))
        claimed = anime_bulk._claim_batch(db.one('SELECT * FROM jobs WHERE id=?', (first,)), owner)
        self.assertEqual(sorted(job_id for job_id, _ in claimed), sorted(jobs))
        with db.connect() as conn:  # the worker died: both leases run out
            conn.execute("UPDATE jobs SET lease_until=0 WHERE status='running'")
        resumed = worker.claim_job(('character_detect',))
        self.assertIn(resumed[0], jobs)
        # A job with a live lease still holds the episode.
        self.assertIsNone(worker.claim_job(('character_detect',)))

    def test_the_cut_button_uses_the_bulk_pipeline_and_keeps_manual_labels(self):
        self.standard_gallery()
        frames = [{'position': 'first', 'time': 2, 'image_size': [100, 80]}, {'position': 'last', 'time': 3.5, 'image_size': [100, 80]}]
        with db.connect() as conn:
            # The user marked the cut and switched analysis off; an explicit request still runs.
            conn.execute("UPDATE cuts SET no_characters=1,analysis_enabled=0 WHERE id='c2'")
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('cur','c2',2,4,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('mine','cur','first',?,'ctx','manual')", (json.dumps(LEFT),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('old-auto','cur','first',?,'ctx','auto')", (json.dumps(RIGHT),))
        self.colors['mine'] = RED
        response = self.client.post('/api/cuts/c2/character-detection', json={'qwen': True})
        self.assertEqual(response.status_code, 200, response.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (response.json()['job_id'],))['payload_json'])
        self.assertEqual((payload['workflow'], payload['phase'], payload['qwen'], payload['cut_id']), ('anime_bulk_v1', 'recognize', True, 'c2'))
        self.assertEqual(len(db.all_rows("SELECT id FROM jobs WHERE pipeline='character_detect'")), 1)  # only this cut
        self.assertEqual(self.client.post('/api/cuts/c2/character-detection').status_code, 409)
        job_id, owner = worker.claim_job(('character_detect',))
        with patch.object(anime_bulk, '_extract_frames', return_value=(frames, [frame_image([(LEFT, RED), (RIGHT, BLUE)]), frame_image([])])), \
             patch.object(anime_gallery, 'extract_crops', side_effect=self.fake_crops):
            anime_bulk.process_batch(db.one('SELECT * FROM jobs WHERE id=?', (job_id,)), owner, Runtime(Embedder()))
        labels = {row['id']: (row['character_context_id'], row['label_source']) for row in db.all_rows("SELECT * FROM character_detections WHERE run_id='cur'")}
        self.assertEqual(labels, {'mine': ('ctx', 'manual'), 'old-auto': ('ctx2', 'auto')})
        self.assertEqual(len(db.all_rows("SELECT id FROM character_detection_runs WHERE cut_id='c2'")), 1)  # boxes kept, no new run

    def test_the_cut_button_explains_when_there_is_nothing_to_redo(self):
        frames = [{'position': 'first', 'time': 0, 'image_size': [100, 80]}]
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('empty','c1',0,2,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('mine','c2',2,4,?,'test')", (json.dumps(frames),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d','mine','first',?,'ctx','manual')", (json.dumps(LEFT),))
        self.assertIn('No people', self.client.post('/api/cuts/c1/character-detection').json()['detail'])
        self.assertIn('already has your label', self.client.post('/api/cuts/c2/character-detection').json()['detail'])
        self.assertIsNone(db.one("SELECT id FROM jobs WHERE pipeline='character_detect'"))
        # A cut without a current run gets boxes and recognition.
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
        queued = self.client.post('/api/cuts/c1/character-detection').json()
        self.assertEqual(json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (queued['job_id'],))['payload_json'])['phase'], 'full')
        self.assertEqual(self.client.post('/api/cuts/missing/character-detection').status_code, 404)


class CcipGraphTest(unittest.TestCase):
    def test_attention_pool_conditional_becomes_its_squeeze_branch(self):
        try:
            import numpy as np
            import onnxruntime
            from onnx import TensorProto, helper
        except ImportError:
            self.skipTest('onnx is not installed')
        axes = helper.make_tensor('axes', TensorProto.INT64, [1], [0])
        then = helper.make_graph([helper.make_node('Squeeze', ['x', 'axes'], ['squeezed'])], 'then', [], [helper.make_tensor_value_info('squeezed', TensorProto.FLOAT, None)], [axes])
        other = helper.make_graph([helper.make_node('Identity', ['x'], ['same'])], 'else', [], [helper.make_tensor_value_info('same', TensorProto.FLOAT, None)])
        nodes = [helper.make_node('Shape', ['x'], ['shape']),
                 helper.make_node('Gather', ['shape', 'zero'], ['first'], axis=0),
                 helper.make_node('Equal', ['first', 'one'], ['is_one']),
                 helper.make_node('If', ['is_one'], ['output'], then_branch=then, else_branch=other)]
        graph = helper.make_graph(nodes, 'g', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 'batch', 4])],
                                  [helper.make_tensor_value_info('output', TensorProto.FLOAT, None)],
                                  [helper.make_tensor('zero', TensorProto.INT64, [], [0]), helper.make_tensor('one', TensorProto.INT64, [], [1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 14)], ir_version=8)
        patched = anime_models._squeeze_attnpool_if(model)
        self.assertNotIn('If', [node.op_type for node in patched.graph.node])
        session = onnxruntime.InferenceSession(patched.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertEqual(session.run(None, {'x': np.ones((1, 3, 4), dtype=np.float32)})[0].shape, (3, 4))

    def test_published_threshold_is_converted_to_cosine(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            variant = Path(directory) / 'v'
            variant.mkdir()
            (variant / 'model_feat.onnx').write_bytes(b'graph')
            (variant / 'metrics.json').write_text(json.dumps({'threshold': 0.2}))
            with patch.object(anime_models, '_onnx_session') as session:
                session.return_value.get_inputs.return_value = [type('Input', (), {'name': 'input'})()]
                embedder = anime_models.CcipEmbedder(directory, 'v', provider='CPUExecutionProvider')
        self.assertAlmostEqual(embedder.published_similarity, 0.6)
        self.assertEqual(embedder.model_id, 'deepghs/ccip_onnx:v')
        with self.assertRaises(RuntimeError):
            anime_models.CcipEmbedder('/nowhere', '../escape')


if __name__ == '__main__':
    unittest.main()
