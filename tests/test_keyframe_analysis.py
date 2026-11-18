"""The keyframe stage end to end with stand-ins for Qwen3.5, DWPose and frame extraction.

``seed_selection`` and ``fake_models`` are shared with the other visual-stage tests.
"""
import json
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

from app import db, inference_worker, keyframe_analysis, keyframe_schema, keyframes, media, single_cut_pipeline, worker
import tests.test_analysis as analysis_tests

FIRST_FRAME = {'mode': 'full', 'composition': 'wide', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'mood': 'calm',
               'visible_characters': [{'character': 'Hero', 'character_context_id': 'ctx', 'position': 'left', 'pose': 'standing'}],
               'background_people': [], 'props': [], 'notes': ''}


def seed_selection(cut_id='c1', count=3, step=0.5):
    """A current keyframe selection of ``count`` frames for the cut (needs ``original.mp4`` on disk, created here)."""
    cut = db.one('SELECT * FROM cuts WHERE id=?', (cut_id,))
    source = media.episode_dir('p', cut['episode_id']) / 'original.mp4'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_bytes(b'video')
    settings = keyframes.config({})
    band = keyframes.mask_for(cut['episode_id'], settings)
    frames = [{'source_frame_index': 10 + k * 12, 'timestamp_seconds': cut['start_time'] + k * step, 'relative_timestamp_seconds': k * step,
               'reason': 'initial_state' if k == 0 else 'event_peak'} for k in range(count)]
    key = keyframes.cache_key(source, cut, settings, band)
    with db.connect() as conn:
        conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES (?,?,'keyframes','done',?)",
                     (f'kf-{cut_id}-{key[:6]}', cut['episode_id'], json.dumps({'keyframe_settings': settings, 'subtitle_mask': band,
                                                                               'cuts': [{'id': cut_id, 'start_time': cut['start_time'], 'end_time': cut['end_time']}]})))
        conn.execute('INSERT OR REPLACE INTO keyframe_results(cache_key,episode_id,cut_id,result_json) VALUES (?,?,?,?)',
                     (key, cut['episode_id'], cut_id, json.dumps({'selected_frames': frames, 'subtitle_mask': band, 'shot_change_score': 10, 'change_level': 'low'})))
    return key


def answer_for(schema, name='Hero', speaking='yes'):
    """A valid observer answer for the window the grammar describes."""
    labels = schema['properties']['frames']['items']['properties']['frame']['enum']
    return {'frames': [{'frame': label, 'characters': [{'name': name, 'state': f'standing on the left at {label}'}]} for label in labels],
            'segments': [{'from': a, 'to': b, 'camera': {'move': 'zoom_in', 'detail': 'model guess'},
                          'characters': [{'name': name, 'walking': 'no', 'speaking': speaking, 'blinking': 'no',
                                          'changes': [{'part': 'head', 'description': f'turns toward screen-right between {a} and {b}'},
                                                      {'part': 'arms', 'description': 'raises the right arm'}]}],
                          'other_changes': []} for a, b in zip(labels, labels[1:])],
            'summary': 'Hero turns.'}


def frame_bytes(spec):
    image = np.full((360, 640, 3), 40, dtype=np.uint8)
    rng = np.random.default_rng(int(spec['ordinal']))
    for _ in range(80):
        x, y = int(rng.integers(0, 600)), int(rng.integers(0, 330))
        cv2.rectangle(image, (x, y), (x + 20, y + 20), tuple(int(v) for v in rng.integers(60, 255, 3)), -1)
    return cv2.imencode('.jpg', image)[1].tobytes()


@contextmanager
def fake_models(ask=None, pose=None, calls=None):
    """Patch frame extraction, the llama-server start, DWPose loading and the Qwen call for the keyframe stage."""
    calls = calls if calls is not None else []

    def chat(_url, messages, schema, cancel_check=None):
        calls.append(messages)
        answer = ask(messages, schema) if ask else answer_for(schema)
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return {'content': content, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 100, 'completion_tokens': 50}, 'timings': {}, 'grammar': True}
    with patch.object(keyframe_analysis.frame_cache, 'render', side_effect=frame_bytes), \
         patch.object(single_cut_pipeline.frame_model, 'frame_server', return_value=nullcontext('http://server')), \
         patch.object(keyframe_analysis, 'load_pose', return_value=(pose, {'model_id': 'yzd-v/DWPose', **({} if pose else {'missing': 'Not installed'})})), \
         patch.object(keyframe_analysis, 'chat', side_effect=chat):
        yield calls


class FakePose:
    """DWPose stand-in: one person whose right arm rises in the last keyframe."""
    BODY = {0: (100, 40), 1: (105, 35), 2: (95, 35), 3: (112, 38), 4: (88, 38), 5: (130, 70), 6: (70, 70), 7: (135, 110), 8: (65, 110),
            9: (137, 150), 10: (63, 150), 11: (112, 150), 12: (88, 150), 13: (113, 210), 14: (87, 210), 15: (113, 270), 16: (87, 270)}

    def __init__(self):
        self.frame = 0

    def detect(self, bgr):
        return [[60.0, 20.0, 140.0, 280.0, 0.9]]

    def keypoints(self, rgb, box):
        self.frame += 1
        moves = {7: (170, 70), 9: (210, 70)} if self.frame >= 3 else {}
        points = [[0.0, 0.0, 0.0] for _ in range(133)]
        for index, (x, y) in {**self.BODY, **moves}.items():
            points[index] = [float(x), float(y), 0.9]
        return points


@unittest.skipIf(cv2 is None, 'opencv is not installed')
class KeyframeStageTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def run_stage(self, **models):
        job_id = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'keyframe_analysis'}).json()['job_id']
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), fake_models(**models) as calls:
            inference_worker.process(*worker.claim_job(('analyze',)))
        return job_id, calls

    def current(self, kind='keyframe_analysis'):
        return next((row for row in self.client.get('/api/cuts/c1/structured-analysis').json() if row['type'] == kind), None)

    def test_qwen_only_run_saves_the_merged_answer_tables_and_queues_the_beats(self):
        key = seed_selection(count=3)
        job_id, calls = self.run_stage()
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'done', job['error'])
        self.assertEqual(len(calls), 1)  # three keyframes are one window
        result = self.current()
        content = result['content']
        self.assertEqual((content['contract'], content['method'], content['selection_key']), ('keyframes-v1', 'qwen', key))
        self.assertEqual([frame['label'] for frame in content['frames']], ['F1', 'F2', 'F3'])
        self.assertEqual([(s['from'], s['to'], s['start'], s['end']) for s in content['segments']], [('F1', 'F2', 0.0, 0.5), ('F2', 'F3', 0.5, 1.0)])
        self.assertNotEqual(content['segments'][0]['camera']['detail'], 'model guess')  # the camera is always measured
        self.assertTrue(all(change['source'] == 'qwen' for segment in content['segments'] for person in segment['characters'] for change in person['changes']))
        self.assertEqual(content['cut'], {'start_time': 0, 'end_time': 2})
        settings = json.loads(db.one('SELECT settings_json FROM analysis_results WHERE id=?', (result['id'],))['settings_json'])
        self.assertEqual((settings['method'], settings['prompt_version'], settings['dwpose']['missing']), ('qwen', 'observer-roster-v1', 'Not installed'))
        run = db.one('SELECT * FROM keyframe_analysis_runs WHERE cut_id=?', ('c1',))
        self.assertEqual((run['phase'], run['method'], run['result_id']), ('complete', 'qwen', result['id']))
        self.assertEqual(db.one('SELECT COUNT(*) AS n FROM keyframe_analysis_frames WHERE run_id=?', (run['id'],))['n'], 3)
        pairs = db.all_rows('SELECT * FROM keyframe_analysis_pairs WHERE run_id=? ORDER BY pair_index', (run['id'],))
        self.assertTrue(all(json.loads(pair['changes_json'])['characters'] for pair in pairs))
        window = db.one('SELECT * FROM keyframe_analysis_windows WHERE run_id=?', (run['id'],))
        self.assertTrue(window['ok'])
        self.assertIn('<image>', window['prompt'])
        beats = db.all_rows("SELECT payload_json FROM jobs WHERE pipeline='performance_beats' AND status='queued'")
        self.assertEqual([json.loads(row['payload_json'])['cut_id'] for row in beats], ['c1'])

    def test_the_tab_reads_the_analysis_and_knows_when_it_is_stale(self):
        self.assertEqual(self.client.get('/api/cuts/c1/keyframe-analysis').json()['status'], 'missing')
        seed_selection(count=3)
        self.run_stage()
        state = self.client.get('/api/cuts/c1/keyframe-analysis').json()
        self.assertEqual((state['status'], state['run']['phase'], state['run']['method'], state['job']['status']), ('current', 'complete', 'qwen', 'done'))
        self.assertEqual(len(state['result']['segments']), 2)
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET payload_json=json_set(payload_json,'$.keyframe_settings.budget',3) WHERE pipeline='keyframes'")
        self.assertEqual(self.client.get('/api/cuts/c1/keyframe-analysis').json()['status'], 'stale')

    def test_hybrid_takes_body_changes_from_the_keypoints(self):
        seed_selection(count=3)
        job_id, _ = self.run_stage(pose=FakePose())
        self.assertEqual(db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        content = self.current()['content']
        self.assertEqual(content['method'], 'hybrid')
        last = content['segments'][1]['characters']
        sources = {(change['part'], change['source']) for person in last for change in person['changes']}
        self.assertIn(('head', 'qwen'), sources)
        self.assertIn(('arms', 'dwpose'), sources)  # the measured arm replaced Qwen's claim about it
        self.assertTrue(any(person.get('pose') for frame in content['frames'] for person in frame['characters']))
        stored = db.one('SELECT pose_json FROM keyframe_analysis_frames ORDER BY relative_time LIMIT 1')
        self.assertEqual(len(json.loads(stored['pose_json'])[0]['points']), 133)

    def test_a_retry_resumes_from_the_saved_windows(self):
        seed_selection(count=9)  # two windows: F1-F5 and F5-F9
        def flaky(messages, schema):
            if schema['properties']['frames']['items']['properties']['frame']['enum'][0] == 'F5':
                return 'not json'  # every attempt for the second window
            return answer_for(schema)
        with patch.object(keyframe_analysis, 'PARALLEL', 1):
            job_id, _ = self.run_stage(ask=flaky)
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('no valid answer after 3 attempts', job['error'])
        self.assertEqual(db.one('SELECT COUNT(*) AS n FROM keyframe_analysis_windows WHERE ok=1')['n'], 1)
        retry = self.client.post(f'/api/jobs/{job_id}/retry').json()['job_id']
        with patch.object(single_cut_pipeline, 'ensure_models_available'), patch.object(single_cut_pipeline, 'release_gpu_models'), fake_models() as calls:
            inference_worker.process(*worker.claim_job(('analyze',)))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (retry,))['status'], 'done')
        self.assertEqual(len(calls), 1)  # only the window that failed is asked again
        self.assertEqual(db.one('SELECT COUNT(*) AS n FROM keyframe_analysis_runs')['n'], 1)
        self.assertEqual(len(self.current()['content']['segments']), 8)

    def test_target_frame_people_give_the_roster(self):
        key = seed_selection(count=3)
        with db.connect() as conn:
            frames = [{'ordinal': 10 + k * 12, 'time': k * .5, 'relative_time': k * .5, 'image_size': [1280, 720]} for k in range(3)]
            conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('t','c1',?,0,2,?)", (key, json.dumps(frames)))
            conn.execute("INSERT INTO target_slots(run_id,slot,origin,character_context_id,label_source) VALUES ('t',0,'appeared','ctx','manual')")
            conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json) VALUES ('b','t',22,0,'[640,0,1280,720]')")
        _, calls = self.run_stage()
        prompt = keyframe_analysis.keyframe_prompts.prompt_text(calls[0])
        self.assertIn('- Hero: box [500, 0, 1000, 1000] (right, first seen in F2)', prompt)
        self.assertEqual(db.one('SELECT roster_source FROM keyframe_analysis_runs')['roster_source'], 'target_slots')
        self.assertEqual(self.current()['content']['roster'][0]['character_context_id'], 'ctx')

    def test_without_a_current_selection_only_the_keyframe_stage_fails(self):
        job_id, calls = self.run_stage()
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('no current keyframe selection', job['error'])
        self.assertEqual(calls, [])

    def test_a_cut_changed_during_the_call_saves_nothing(self):
        seed_selection(count=3)

        def moving(messages, schema):
            with db.connect() as conn:
                conn.execute("UPDATE cuts SET end_time=1.5 WHERE id='c1'")
            return answer_for(schema)
        job_id, _ = self.run_stage(ask=moving)
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('cut changed', job['error'])
        self.assertIsNone(self.current())
        self.assertIsNone(db.one("SELECT id FROM jobs WHERE pipeline='performance_beats'"))


class QueueingTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_the_visual_job_queues_the_missing_selection_and_waits_for_it(self):
        source = media.episode_dir('p', 'e') / 'original.mp4'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'video')
        job_id = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'keyframe_analysis'}).json()['job_id']
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])
        [wait] = payload['wait_for_keyframes']
        selection = db.one('SELECT pipeline,payload_json FROM jobs WHERE id=?', (wait,))
        self.assertEqual(selection['pipeline'], 'keyframes')
        self.assertEqual([cut['id'] for cut in json.loads(selection['payload_json'])['cuts']], ['c1'])
        self.assertIsNone(worker.claim_job(('analyze',)))  # held while the selection is queued
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (wait,))
        self.assertEqual(worker.claim_job(('analyze',))[0], job_id)  # a failed selection fails only the stage, not the job's other work

    def test_a_pending_selection_is_reused_and_a_first_frame_only_job_waits_for_none(self):
        source = media.episode_dir('p', 'e') / 'original.mp4'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'video')
        first = self.client.post('/api/episodes/e/visual-analysis').json()['job_ids'][0]
        wait = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (first,))['payload_json'])['wait_for_keyframes']
        self.assertEqual(len(wait), 1)
        self.assertEqual(len(json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (wait[0],))['payload_json'])['cuts']), 2)
        frame_only = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'initial_frame'})
        self.assertEqual(frame_only.status_code, 409)  # the batch already covers that stage
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (first,))
        frame_only = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'initial_frame'}).json()['job_id']
        self.assertNotIn('wait_for_keyframes', json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (frame_only,))['payload_json']))

    def test_the_video_stage_is_gone(self):
        response = self.client.post('/api/cuts/c1/analysis-stages/run', json={'stage': 'video_analysis'})
        self.assertEqual(response.status_code, 410)
        self.assertIn('keyframe analysis', response.json()['detail'])


class AssembleTest(unittest.TestCase):
    def test_a_frame_shared_by_two_windows_keeps_its_first_state_and_names_link_to_the_roster(self):
        frames = [{'label': f'F{k}', 'ordinal': k, 't': k / 10, 'reason': None} for k in range(1, 6)]
        windows = keyframe_schema.make_windows(frames, 3)
        merged = [{'frames': [{'frame': 'F3', 'characters': [{'name': 'Hero', 'state': 'first'}]}], 'segments': [], 'summary': 'one'},
                  {'frames': [{'frame': 'F3', 'characters': [{'name': 'Hero', 'state': 'second'}]}], 'segments': [], 'summary': 'two'}]
        content = keyframe_analysis.assemble(frames, windows, merged, [{'name': 'Hero', 'character_context_id': 'ctx'}], {'F3': {'unknown person 1': 'standing'}}, 'hybrid', 'r', 'k')
        f3 = content['frames'][2]['characters']
        self.assertEqual(f3[0], {'name': 'Hero', 'character_context_id': 'ctx', 'state': 'first', 'pose': None})
        self.assertEqual(f3[1]['name'], 'unknown person 1')
        self.assertEqual(content['summary'], 'one two')


if __name__ == '__main__':
    unittest.main()
