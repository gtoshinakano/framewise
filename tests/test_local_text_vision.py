"""The Qwen 3.6 text server starts with its vision projector (chat can read images) and falls back to text only when it cannot."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import local_text_runtime


class FakeProcess:
    """A llama-server stand-in: ``exits`` is what poll() reports from the start (None = it keeps running)."""

    def __init__(self, exits=None):
        self.exits = exits
        self.terminated = False

    def poll(self):
        return None if self.terminated else self.exits

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


class FakeResponse:
    def __init__(self, body=b'{}', status=200):
        self.body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return self.body


class VisionLaunchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name)
        self.server, self.model, self.mtp, self.projector = (folder / name for name in ('llama-server', 'standard.gguf', 'mtp.gguf', 'mmproj.gguf'))
        for path in (self.server, self.model, self.mtp, self.projector):
            path.touch()
        env = {'QWEN35_SERVER_PATH': str(self.server), 'QWEN36_MODEL_PATH': str(self.model), 'QWEN36_MTP_MODEL_PATH': str(self.mtp),
               'QWEN36_PROJECTOR_PATH': str(self.projector), 'QWEN36_MTP_ARGS': '--spec-type draft-mtp'}
        patcher = patch.dict('os.environ', env)
        patcher.start()
        self.addCleanup(patcher.stop)
        with patch.object(local_text_runtime.LocalTextRuntime, 'publish'), patch.object(local_text_runtime.LocalTextRuntime, 'publish_variants'):
            self.runtime = local_text_runtime.LocalTextRuntime()
        self.runtime.publish = lambda *args: None
        self.variant = 'standard'
        self.runtime.target_variant = lambda: self.variant
        self.starts, self.exits, self.props = [], [], {'modalities': {'vision': True}}
        popen = patch('subprocess.Popen', side_effect=self.popen)
        popen.start()
        self.addCleanup(popen.stop)
        urlopen = patch('urllib.request.urlopen', side_effect=self.urlopen)
        urlopen.start()
        self.addCleanup(urlopen.stop)

    def popen(self, args, **kwargs):
        self.starts.append(list(args))
        return FakeProcess(self.exits.pop(0) if self.exits else None)

    def urlopen(self, url, timeout=None):
        return FakeResponse(json.dumps(self.props).encode() if str(url).endswith('/props') else b'{}')

    def test_the_projector_is_loaded_and_chat_can_see(self):
        self.assertTrue(self.runtime.ensure())
        args = self.starts[0]
        self.assertEqual(args[args.index('--mmproj') + 1], str(self.projector))
        self.assertEqual(args[args.index('-m') + 1], str(self.model))
        self.assertTrue(self.runtime.vision)
        self.assertEqual(local_text_runtime.vision_status(self.runtime, 'standard'), (True, ''))

    def test_the_mtp_model_gets_the_projector_beside_its_own_arguments(self):
        self.variant = 'mtp'
        self.runtime.ensure()
        args = self.starts[0]
        self.assertEqual((args[args.index('-m') + 1], '--mmproj' in args, '--spec-type' in args), (str(self.mtp), True, True))
        self.assertTrue(self.runtime.vision)

    def test_a_server_that_says_it_has_no_vision_is_not_offered_the_tool(self):
        self.props = {'modalities': {'vision': False}}
        self.runtime.ensure()
        self.assertFalse(self.runtime.vision)
        self.assertIn('did not load', local_text_runtime.vision_status(self.runtime, 'standard')[1])

    def test_it_can_be_switched_off_or_left_out(self):
        with patch.dict('os.environ', {'QWEN36_TEXT_VISION': '0'}):
            self.runtime.ensure()
        self.assertNotIn('--mmproj', self.starts[0])
        self.assertFalse(self.runtime.vision)
        self.runtime.stop()
        self.projector.unlink()  # not installed: the text model starts exactly as before
        self.runtime.ensure()
        self.assertNotIn('--mmproj', self.starts[1])
        self.assertEqual(len(self.starts), 2)

    def test_a_start_that_fails_with_the_projector_is_retried_without_it_and_remembered(self):
        self.exits = [1]  # the first llama-server exits at once
        self.assertTrue(self.runtime.ensure())
        self.assertEqual(['--mmproj' in args for args in self.starts], [True, False])
        self.assertFalse(self.runtime.vision)
        self.assertFalse(self.runtime.failed)  # the text model works; only the images are off
        self.assertIn('standard', self.runtime.vision_failed)
        self.assertEqual(local_text_runtime.vision_status(self.runtime, 'standard')[0], False)
        self.runtime.stop()
        self.runtime.ensure()  # a later start does not try the projector again
        self.assertEqual(['--mmproj' in args for args in self.starts], [True, False, False])
        self.variant = 'mtp'  # another variant is tried on its own
        self.runtime.stop()
        self.runtime.ensure()
        self.assertTrue('--mmproj' in self.starts[-1])

    def test_a_start_that_fails_without_the_projector_too_is_an_error(self):
        self.exits = [1, 1]
        with self.assertRaises(local_text_runtime.ServerExited):
            self.runtime.ensure()
        self.assertTrue(self.runtime.failed)
        self.assertEqual(len(self.starts), 2)

    def test_stopping_the_server_forgets_that_it_could_see(self):
        self.runtime.ensure()
        self.assertTrue(self.runtime.vision)
        self.runtime.stop()
        self.assertFalse(self.runtime.vision)


if __name__ == '__main__':
    unittest.main()
