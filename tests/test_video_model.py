import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app import video_model


class Tensor:
    shape = (1, 10)
    def __getitem__(self, _key): return self
    def to(self, _device): return self


class VideoAdapterTest(unittest.TestCase):
    def test_partitions_cut_and_converts_event_timing(self):
        event = {'start_seconds': 0.0, 'end_seconds': 0.5, 'subject': 'person', 'visual_action': 'turns', 'camera': 'fixed', 'uncertainty': 'unknown'}
        response = json.dumps({'overview': 'scene', 'events': [event], 'uncertainties': 'limited'})
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                class Inputs(dict):
                    def to(self, _device): return self
                return Inputs(input_ids=Tensor())
            def batch_decode(self, *args, **kwargs): return [response]
        class Model:
            device = 'cuda'
            def to(self, _device): return self
            def eval(self): return self
            def generate(self, **kwargs): return Tensor()
        processor = Processor()
        transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor), AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None), bfloat16='bf16', inference_mode=nullcontext)
        def prepare(source, clip, start, end, width, height, fps):
            clip.touch()
            return {'source_start': start, 'source_end': end, 'width': width, 'height': height, 'fps': fps}
        class Inputs(dict):
            def to(self, _device): return self
        calls = []
        def encode(_processor, _clip, fps, prompt):
            calls.append((fps, prompt))
            return Inputs(input_ids=Tensor())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"weights.safetensors"}}')
            (path / 'weights.safetensors').touch()
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(video_model.media, 'prepare_analysis_video', side_effect=prepare), patch.object(video_model, 'video_inputs', side_effect=encode):
                result = video_model.analyze_video(path / 'original.mp4', {'start_time': 20, 'end_time': 29}, {'video_width': 640, 'video_height': 360, 'video_fps': 4}, path, path)
            self.assertEqual([(x['source']['source_start'], x['source']['source_end']) for x in result['windows']], [(20, 28), (28, 29)])
            self.assertEqual(result['windows'][1]['analysis']['events'][0]['episode_start_seconds'], 28)
            self.assertFalse((path / 'video-window.mp4').exists())
            self.assertEqual(calls[0][0], 4)

    def test_completed_windows_are_reused_without_loading_models(self):
        windows = [{'source': {'source_start': 20, 'source_end': 28}, 'analysis': {'events': []}},
                   {'source': {'source_start': 28, 'source_end': 29}, 'analysis': {'events': []}}]
        result = video_model.analyze_video('missing.mp4', {'start_time': 20, 'end_time': 29},
            {'video_width': 640, 'video_height': 360, 'video_fps': 4}, '/missing-model', '/tmp',
            load_window=lambda index, start, end: windows[index])
        self.assertEqual(result['windows'], windows)
