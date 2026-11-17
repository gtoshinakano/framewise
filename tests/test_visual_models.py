import io
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import db, frame_model, media, single_cut_pipeline, video_model, visual_models
from app.visual_contracts import AUDIO_NOTE
import tests.test_analysis as analysis_tests

LFM = visual_models.MODELS['lfm3-omni-2b']
VIDEO_ANSWER = {'overview': 'Mia turns.', 'camera': {'movement': 'static', 'detail': 'none'},
                'characters': [{'name': 'Mia', 'activity': 'standing', 'travel': 'none', 'motion': 'Turns.', 'speaking': 'yes', 'emotion': 'calm', 'end': 'left'}],
                'events': [{'start_seconds': 0.0, 'end_seconds': 0.5, 'subject': 'Mia', 'visual_action': 'turns', 'camera': 'fixed', 'uncertainty': 'none'}],
                'uncertainties': 'none'}


def reply(content, finish='stop'):
    return io.BytesIO(json.dumps({'choices': [{'finish_reason': finish, 'message': {'content': content}}],
                                  'usage': {'prompt_tokens': 700, 'completion_tokens': 90, 'total_tokens': 790}}).encode())


def report(**available):
    with db.connect() as conn:
        conn.execute('UPDATE visual_model_state SET models_json=?,updated_at=? WHERE id=1',
                     (json.dumps({key: {'available': ok, 'detail': 'Installed' if ok else 'Not installed'} for key, ok in available.items()}), time.time()))


class RegistryTest(unittest.TestCase):
    def test_defaults_are_todays_models_and_every_role_has_one(self):
        self.assertEqual(visual_models.DEFAULTS, {'initial_frame': 'qwen35-27b', 'video_analysis': 'qwen3-vl-8b', 'character_profile': 'qwen35-27b'})
        for role, key in visual_models.DEFAULTS.items():
            self.assertTrue(visual_models.supports(key, role))
        self.assertEqual(visual_models.MODELS['qwen35-27b'].model_id, frame_model.MODEL_ID)
        self.assertEqual(visual_models.MODELS['qwen3-vl-8b'].revision, video_model.MODEL_REVISION)

    def test_lfm3_runs_every_role_and_is_pinned_in_the_manifest(self):
        self.assertEqual(set(LFM.roles), set(visual_models.ROLES))
        entry = visual_models.MANIFEST[LFM.manifest]
        self.assertEqual(LFM.revision, entry['revision'])
        self.assertEqual({item['name'] for item in entry['files']}, {LFM.path('model').name, LFM.path('projector').name})
        self.assertTrue(all(len(item['sha256']) == 64 for item in entry['files']))
        self.assertIn('--reasoning', LFM.server_args)
        self.assertTrue(LFM.hears_audio)
        for model in visual_models.MODELS.values():
            self.assertTrue(model.manifest is None or model.manifest in visual_models.MANIFEST)

    def test_video_prompt_offers_no_label_to_copy(self):
        # LFM3-Omni-2B copied the old example label "girl with red umbrella" into cuts without any umbrella.
        from app.visual_contracts import VIDEO_INSTRUCTION
        self.assertNotIn('umbrella', VIDEO_INSTRUCTION)
        self.assertIn('never invent a name', VIDEO_INSTRUCTION)

    def test_a_job_runs_its_snapshot_else_the_default(self):
        self.assertEqual(visual_models.for_job({}, 'initial_frame').key, 'qwen35-27b')
        self.assertEqual(visual_models.for_job({'models': {'video_analysis': 'lfm3-omni-2b'}}, 'video_analysis').key, 'lfm3-omni-2b')
        self.assertEqual(visual_models.for_job({'models': {'initial_frame': 'qwen3-vl-8b'}}, 'initial_frame').key, 'qwen35-27b')  # cannot run that role
        self.assertEqual(visual_models.for_job({'models': {'initial_frame': 'removed-model'}}, 'initial_frame').key, 'qwen35-27b')
        self.assertEqual(single_cut_pipeline.job_models({'models': {'initial_frame': 'lfm3-omni-2b'}})['video_analysis'].key, 'qwen3-vl-8b')

    def test_status_reports_installed_downloading_and_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            server, model, projector = folder / 'llama-server', folder / LFM.path('model').name, folder / LFM.path('projector').name
            env = {'QWEN35_SERVER_PATH': str(server), 'LFM3_OMNI_MODEL_PATH': str(model), 'LFM3_OMNI_PROJECTOR_PATH': str(projector)}
            with patch.dict('os.environ', env):
                self.assertEqual(visual_models.status('lfm3-omni-2b'), {'available': False, 'detail': 'Not installed'})
                model.with_name(model.name + '.partial').write_bytes(b'x' * 1000)
                self.assertRegex(visual_models.status('lfm3-omni-2b')['detail'], r'^Downloading \d+%$')
                model.with_name(model.name + '.partial').unlink()
                model.touch(); projector.touch()
                self.assertEqual(visual_models.status('lfm3-omni-2b')['detail'], 'llama-server runtime is missing')
                server.touch()
                self.assertEqual(visual_models.status('lfm3-omni-2b'), {'available': True, 'detail': 'Installed'})
                visual_models.require('lfm3-omni-2b')
                projector.unlink()
                with self.assertRaisesRegex(RuntimeError, 'install-model.py lfm3-omni-2b'):
                    visual_models.require('lfm3-omni-2b')

    def test_frame_server_passes_the_models_own_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            files = [Path(folder) / name for name in ('llama-server', 'model.gguf', 'mmproj.gguf')]
            for path in files:
                path.touch()
            with patch('subprocess.Popen', side_effect=RuntimeError('stop')) as popen, self.assertRaisesRegex(RuntimeError, 'stop'):
                with frame_model.frame_server(*files, extra_args=LFM.server_args):
                    pass
        args = popen.call_args.args[0]
        self.assertEqual(args[args.index('-m') + 1], str(files[1]))
        self.assertEqual(args[-2:], ['--reasoning', 'off'])


class SettingsApiTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def payload(self, job_id):
        return json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])

    def test_overview_lists_roles_models_and_the_defaults(self):
        data = self.client.get('/api/visual-models').json()
        self.assertEqual([role['role'] for role in data['roles']], list(visual_models.ROLES))
        self.assertEqual(data['selected'], visual_models.DEFAULTS)
        self.assertFalse(data['worker_reported'])
        lfm = next(model for model in data['models'] if model['key'] == 'lfm3-omni-2b')
        self.assertEqual(lfm['install'], 'python3 scripts/install-model.py lfm3-omni-2b')

    def test_choice_is_refused_for_a_wrong_role_or_a_missing_model(self):
        self.assertEqual(self.client.put('/api/visual-models', json={'role': 'initial_frame', 'model': 'qwen3-vl-8b'}).status_code, 422)
        self.assertEqual(self.client.put('/api/visual-models', json={'role': 'nope', 'model': 'qwen35-27b'}).status_code, 422)
        self.assertEqual(self.client.put('/api/visual-models', json={'role': 'initial_frame', 'model': 'lfm3-omni-2b'}).status_code, 409)  # never reported
        report(**{'lfm3-omni-2b': False})
        response = self.client.put('/api/visual-models', json={'role': 'initial_frame', 'model': 'lfm3-omni-2b'})
        self.assertEqual(response.status_code, 409)
        self.assertIn('Not installed', response.json()['detail'])
        # The default is always allowed, so the user can switch back without the worker.
        self.assertEqual(self.client.put('/api/visual-models', json={'role': 'initial_frame', 'model': 'qwen35-27b'}).status_code, 200)

    def test_queued_jobs_snapshot_the_choice_and_keep_it(self):
        report(**{'lfm3-omni-2b': True})
        for role in ('initial_frame', 'video_analysis'):
            self.assertEqual(self.client.put('/api/visual-models', json={'role': role, 'model': 'lfm3-omni-2b'}).json()['selected'][role], 'lfm3-omni-2b')
        single = self.client.post('/api/cuts/c1/analyze').json()['job_id']
        self.assertEqual(self.payload(single)['models'], {'initial_frame': 'lfm3-omni-2b', 'video_analysis': 'lfm3-omni-2b'})
        self.client.put('/api/visual-models', json={'role': 'video_analysis', 'model': 'qwen3-vl-8b'})
        batch = self.client.post('/api/episodes/e/visual-analysis').json()['job_ids'][0]
        self.assertEqual(self.payload(batch)['models'], {'initial_frame': 'lfm3-omni-2b', 'video_analysis': 'qwen3-vl-8b'})
        self.assertEqual(self.payload(single)['models']['video_analysis'], 'lfm3-omni-2b')  # queued work keeps its model


class FakeServer:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, request, timeout=None):
        self.bodies.append(json.loads(request.data))
        return self.replies.pop(0)


class LlamaVideoRuntimeTest(unittest.TestCase):
    def runtime(self):
        runtime = video_model.LlamaVideoRuntime(LFM)
        runtime.url = 'http://127.0.0.1:1'  # already started
        return runtime

    def test_sends_the_clip_as_video_and_validates_the_answer(self):
        server = FakeServer([reply(json.dumps(VIDEO_ANSWER))])
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / 'clip.mp4'
            clip.write_bytes(b'mp4 bytes')
            with patch('urllib.request.urlopen', side_effect=server):
                output, usage, attempts = self.runtime().generate(clip, 5, 'Describe the motion.', 1.0, 20.0)
        content = server.bodies[0]['messages'][0]['content']
        self.assertEqual(content[0]['type'], 'video_url')
        self.assertTrue(content[0]['video_url']['url'].startswith('data:video/mp4;base64,'))
        self.assertNotIn('fps', content[0])  # llama-server ignores it: LFM3 always reads 2 fps
        self.assertEqual(self.runtime().sample_fps, 2.0)
        self.assertEqual(content[1], {'type': 'text', 'text': 'Describe the motion.'})
        self.assertEqual((server.bodies[0]['temperature'], server.bodies[0]['response_format']), (0, {'type': 'json_object'}))
        self.assertEqual((attempts, usage['completion_tokens']), (1, 90))
        self.assertEqual(output['events'][0]['episode_start_seconds'], 20.0)

    def test_an_invalid_answer_is_kept_and_asked_again_concisely(self):
        server = FakeServer([reply('{"overview": ', 'length'), reply(json.dumps(VIDEO_ANSWER))])
        rejected = []
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / 'clip.mp4'
            clip.write_bytes(b'mp4')
            with patch('urllib.request.urlopen', side_effect=server):
                _, usage, attempts = self.runtime().generate(clip, 4, 'Prompt', 1.0, 0.0, on_invalid_response=lambda error, raw: rejected.append((error, raw)))
        self.assertEqual(attempts, 2)
        self.assertIn('generation limit', rejected[0][0])
        self.assertEqual(rejected[0][1], '{"overview": ')
        self.assertTrue(server.bodies[1]['messages'][0]['content'][1]['text'].endswith(video_model.RETRY_INSTRUCTION))
        self.assertEqual(server.bodies[1]['max_tokens'], video_model.VIDEO_TOKEN_LIMITS[1])
        self.assertEqual(usage['completion_tokens'], 180)

    def test_runtime_for_picks_the_backend_of_the_model(self):
        self.assertIsInstance(video_model.runtime_for(visual_models.MODELS['qwen3-vl-8b']), video_model.VideoRuntime)
        self.assertIsInstance(video_model.runtime_for(LFM), video_model.LlamaVideoRuntime)

    def test_a_listening_runtime_gets_the_soundtrack_and_the_audio_note(self):
        class Listening:
            model_id, revision, hears_audio, sample_fps = 'listener', 'rev-1', True, 2.0
            prompts = []
            def load(self, _path): pass
            def close(self): pass
            def generate(self, clip, fps, prompt, duration, start, reject=None, cancel=None):
                self.prompts.append(prompt)
                return {'overview': 'x', 'camera': {'movement': 'static', 'detail': 'none'}, 'characters': [], 'events': [], 'uncertainties': 'none'}, {}, 1
        calls = []
        def prepare(source, clip, start, end, width, height, fps, **kwargs):
            calls.append(kwargs)
            clip.touch()
            return {'source_start': start, 'source_end': end, 'fps': fps, 'has_audio': True}
        runtime = Listening()
        with tempfile.TemporaryDirectory() as folder, patch.object(video_model.media, 'prepare_analysis_video', side_effect=prepare):
            result = video_model.analyze_video(Path(folder) / 'proxy.mp4', {'start_time': 0, 'end_time': 2}, {'video_width': 640, 'video_height': 360, 'video_fps': 4},
                                               '/unused', folder, runtime=runtime)
        self.assertEqual(calls, [{'keep_audio': True}])
        self.assertIn(AUDIO_NOTE, runtime.prompts[0])
        self.assertEqual((result['model'], result['revision']), ('listener', 'rev-1'))
        self.assertEqual(result['windows'][0]['source']['model_sample_fps'], 2.0)


class SavedWindowTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_a_window_saved_by_another_model_is_analysed_again(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,payload_json) VALUES ('j','e','analyze','{}')")
            window = {'source': {'source_start': 0, 'source_end': 2}, 'analysis': {'events': []}}
            conn.execute("INSERT INTO analysis_video_windows(job_id,cut_id,window_index,start_time,end_time,model_revision,content_json) VALUES ('j','c1',0,0,2,?,?)",
                         (video_model.MODEL_REVISION, json.dumps(window)))

        @contextmanager
        def stage(*_args):
            yield
        cut_run = SimpleNamespace(job={'id': 'j'}, owner='o', cut_id='c1', original=Path('/missing/original.mp4'), checkpoint={}, context={}, config={}, cut={'start_time': 0, 'end_time': 2, 'cut_index': 0},
                                  strip=None, episode={'width': 1280, 'height': 720}, hints={}, stage=stage, progress=lambda _message: None, cancel_check=lambda: None,
                                  save=lambda *_args, **_kwargs: None)
        seen = {}
        def analyze(*_args, load_window=None, **_kwargs):
            seen['lfm'] = load_window(0, 0, 2)
            return {'windows': []}
        with patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=analyze), \
             patch.object(single_cut_pipeline, 'marked_people', return_value=[]):
            single_cut_pipeline._video_analysis(cut_run, '/scratch', None, LFM)
        self.assertIsNone(seen['lfm'])
        with patch.object(single_cut_pipeline.video_model, 'analyze_video', side_effect=analyze), \
             patch.object(single_cut_pipeline, 'marked_people', return_value=[]):
            single_cut_pipeline._video_analysis(cut_run, '/scratch', None, visual_models.MODELS['qwen3-vl-8b'])
        self.assertEqual(seen['lfm']['source']['source_end'], 2)


@unittest.skipUnless(shutil.which('ffmpeg'), 'needs ffmpeg')
class AudioClipTest(unittest.TestCase):
    def test_a_listening_clip_keeps_the_soundtrack_and_a_silent_one_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=25:duration=2', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)], check=True)
            def streams(path):
                info = json.loads(media.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]).stdout)
                return {stream['codec_type']: stream for stream in info['streams']}
            listening = media.prepare_analysis_video(source, Path(folder) / 'with.mp4', 0.5, 1.5, 320, 240, 4, keep_audio=True)
            silent = media.prepare_analysis_video(source, Path(folder) / 'without.mp4', 0.5, 1.5, 320, 240, 4)
            audio = streams(Path(folder) / 'with.mp4').get('audio')
            self.assertTrue(listening['has_audio'])
            self.assertEqual((audio['sample_rate'], audio['channels']), ('16000', 1))
            self.assertNotIn('audio', streams(Path(folder) / 'without.mp4'))
            self.assertNotIn('has_audio', silent)
