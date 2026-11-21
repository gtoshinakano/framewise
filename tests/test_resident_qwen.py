"""The one resident Qwen 3.6 server: pictures go to it instead of a private server, and it is not stopped for them."""
import contextlib
import fcntl
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tests.test_analysis as analysis_tests
from app import db, frame_model, gpu_device, job_kinds, local_text_runtime, visual_models

QWEN36 = visual_models.MODELS['qwen36-35b']
QWEN35 = visual_models.MODELS['qwen35-27b']
LFM = visual_models.MODELS['lfm3-omni-2b']


def runtime_serving(vision=True, url='http://127.0.0.1:9'):
    """A runtime whose server is already up (no process is started)."""
    runtime = local_text_runtime.LocalTextRuntime()
    runtime.process, runtime.url, runtime.variant, runtime.vision = Mock(), url, 'mtp', vision
    runtime.process.poll.return_value = None
    return runtime


class VisualSessionTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_a_visual_stage_uses_the_running_server_even_while_visual_work_owns_the_gpu(self):
        runtime = runtime_serving()
        runtime.begin_gpu_work()  # the visual job raised this itself, to keep new chat turns out
        with patch.object(local_text_runtime.gpu_device, 'release_held') as release, runtime.visual_session() as url:
            self.assertEqual(url, 'http://127.0.0.1:9')
            self.assertEqual(runtime.chat_users, 1)  # a switch or stop waits for the stage, like for a chat turn
            seen = []
            other = threading.Thread(target=lambda: seen.append(runtime.can_switch()))
            other.start(); other.join()
            self.assertEqual(seen, [False])  # another thread cannot switch the model under the stage
        release.assert_called_once()
        self.assertEqual(runtime.chat_users, 0)
        self.assertIsNotNone(runtime.process)  # nothing was stopped

    def test_a_server_without_vision_is_an_error_not_a_blind_stage(self):
        runtime = runtime_serving(vision=False)
        with self.assertRaisesRegex(RuntimeError, 'cannot read pictures'), runtime.visual_session():
            pass
        self.assertEqual(runtime.chat_users, 0)

    def test_the_server_is_started_when_it_is_not_running(self):
        runtime = local_text_runtime.LocalTextRuntime()

        def start(cancel_check=None, idle_load=False):
            runtime.process, runtime.variant, runtime.vision = Mock(), 'mtp', True
            return 'http://127.0.0.1:7'
        with patch.object(runtime, '_ensure', side_effect=start), runtime.visual_session() as url:
            self.assertEqual(url, 'http://127.0.0.1:7')


class ModelServerTest(unittest.TestCase):
    def test_the_resident_model_never_starts_a_private_server(self):
        runtime = Mock()
        runtime.visual_session.return_value = contextlib.nullcontext('http://resident')
        with patch.object(local_text_runtime, 'runtime', runtime), patch.object(frame_model, 'frame_server') as private, \
                frame_model.model_server(QWEN36, cancel_check=None, parallel=4, context_per_slot=8192) as url:
            self.assertEqual(url, 'http://resident')
        private.assert_not_called()

    def test_another_model_keeps_its_own_server_and_arguments(self):
        with patch.object(frame_model, 'frame_server', return_value=contextlib.nullcontext('http://private')) as private, \
                frame_model.model_server(LFM, parallel=2, context_per_slot=4096) as url:
            self.assertEqual(url, 'http://private')
        self.assertEqual(private.call_args.kwargs['extra_args'], LFM.server_args)
        self.assertEqual((private.call_args.kwargs['parallel'], private.call_args.kwargs['context_per_slot']), (2, 4096))

    def test_a_missing_worker_is_a_clear_error(self):
        with patch.object(local_text_runtime, 'runtime', None), self.assertRaisesRegex(RuntimeError, 'unavailable'), frame_model.model_server(QWEN36):
            pass

    def test_requests_in_flight_are_limited_to_the_slots_of_the_resident_server_only(self):
        with patch.dict('os.environ', {'QWEN36_SLOTS': '3'}):
            self.assertEqual(frame_model.workers(QWEN36, 4), 3)
            self.assertEqual(frame_model.workers(QWEN36, 2), 2)
            self.assertEqual(frame_model.workers(QWEN35, 4), 4)  # a private server is started with the stage's own number

    def test_thinking_is_off_for_pictures_unless_a_request_asks_for_it(self):
        self.assertEqual(frame_model.thinking_kwargs(), {'enable_thinking': False})
        self.assertEqual(frame_model.thinking_kwargs(True), {'enable_thinking': True})

        def sent(**kwargs):
            with tempfile.TemporaryDirectory() as folder:
                image = Path(folder) / 'frame.jpg'
                image.write_bytes(b'image')
                answer = {'composition': 'wide', 'setting': 'room', 'lighting': 'soft', 'camera': 'level', 'visible_characters': [], 'background_people': 'none', 'notes': 'none'}
                body = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}], 'usage': {}}).encode()
                with patch('urllib.request.urlopen', return_value=io.BytesIO(body)) as request:
                    frame_model.describe_frame('http://localhost:1', image, **kwargs)
            return json.loads(request.call_args.args[0].data)['chat_template_kwargs']
        self.assertEqual(sent(), {'enable_thinking': False})
        self.assertEqual(sent(thinking=True), {'enable_thinking': True})


class IdentityTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_results_name_the_variant_that_is_selected(self):
        mtp = visual_models.for_job({}, 'initial_frame')
        self.assertEqual((mtp.model_id, mtp.key), (local_text_runtime.VARIANTS['mtp']['id'], 'qwen36-35b'))
        self.assertIn('MTP', mtp.label)
        with db.connect() as conn:
            conn.execute("UPDATE text_generation_settings SET local_model='standard' WHERE id=1")
        standard = visual_models.for_job({}, 'keyframe_analysis')
        self.assertEqual(standard.model_id, local_text_runtime.VARIANTS['standard']['id'])
        self.assertTrue(standard.revision.startswith(local_text_runtime.VARIANTS['standard']['revision'] + '+mmproj-sha256:'))
        self.assertNotIn('MTP', standard.label)
        self.assertEqual(visual_models.general().model_id, standard.model_id)
        names = {item['key']: item for item in visual_models.overview()['models']}
        self.assertEqual(names['qwen36-35b']['label'], standard.label)  # the dialog says what the worker really runs
        self.assertTrue(QWEN36.resident and not QWEN35.resident and not LFM.resident)

    def test_which_jobs_keep_the_server(self):
        self.assertTrue(visual_models.runs_resident({}, ('initial_frame', 'keyframe_analysis')))
        self.assertFalse(visual_models.runs_resident({'models': {'keyframe_analysis': 'qwen35-27b'}}, ('initial_frame', 'keyframe_analysis')))
        self.assertTrue(visual_models.runs_resident({'models': {'initial_frame': 'lfm3-omni-2b'}}, ('keyframe_analysis',)))

    def test_only_jobs_that_run_entirely_on_the_resident_server_leave_the_device_alone(self):
        analyze, profile = job_kinds.KINDS['analyze'], job_kinds.KINDS['character_profile']
        v2 = {'workflow': 'single_cut_v2'}
        self.assertFalse(analyze.needs_device(v2))
        self.assertFalse(analyze.needs_device({**v2, 'models': {'initial_frame': 'qwen36-35b'}}))
        self.assertTrue(analyze.needs_device({**v2, 'models': {'initial_frame': 'lfm3-omni-2b'}}))  # a private server needs the memory
        self.assertTrue(analyze.needs_device({**v2, 'models': {'keyframe_analysis': 'qwen35-27b'}}))
        self.assertTrue(analyze.needs_device({'workflow': 'legacy'}))  # the older workflows load their torch models
        self.assertFalse(profile.needs_device({}))
        self.assertTrue(profile.needs_device({'models': {'character_profile': 'qwen35-27b'}}))
        # torch models (detection, the subtitle strip locator) and audio still take the whole GPU
        for name in ('character_detect', 'subtitles', 'transcribe'):
            self.assertTrue(job_kinds.KINDS[name].needs_device({}), name)
        for name, kind in job_kinds.KINDS.items():
            if 'gpu' in kind.resources and name not in ('analyze', 'character_profile'):
                self.assertTrue(kind.needs_device({}), name)


class DeviceLockTest(unittest.TestCase):
    def test_a_job_can_hand_its_device_lock_to_the_server_it_is_about_to_load(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(db, 'DATA_DIR', Path(folder)):
            def other_process_could_lock():
                handle = (Path(folder) / '.gpu.lock').open('a+b')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    return False
                finally:
                    handle.close()
            gpu_device.release_held()  # nothing held: nothing happens
            with gpu_device.hold():
                self.assertFalse(other_process_could_lock())
                gpu_device.release_held()
                self.assertTrue(other_process_could_lock())  # the server's own acquire would now succeed
            self.assertTrue(other_process_could_lock())

    def test_the_lock_belongs_to_the_thread_that_took_it(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(db, 'DATA_DIR', Path(folder)):
            seen = []
            with gpu_device.hold():
                worker = threading.Thread(target=lambda: (gpu_device.release_held(), seen.append(True)))
                worker.start(); worker.join()
                handle = (Path(folder) / '.gpu.lock').open('a+b')
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another thread's release did not free it
                handle.close()
            self.assertEqual(seen, [True])


if __name__ == '__main__':
    unittest.main()
