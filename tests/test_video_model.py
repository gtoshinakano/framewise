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
                result = video_model.analyze_video(path / 'original.mp4', {'start_time': 20, 'end_time': 29}, {'video_width': 640, 'video_height': 360, 'video_fps': 4}, path, path, context_hints={'character_detections': {'labels_confirmed_by_queue': True}})
            self.assertEqual([(x['source']['source_start'], x['source']['source_end']) for x in result['windows']], [(20, 28), (28, 29)])
            self.assertEqual(result['windows'][1]['analysis']['events'][0]['episode_start_seconds'], 28)
            self.assertFalse((path / 'video-window.mp4').exists())
            self.assertEqual(calls[0][0], 4)
            self.assertIn('position, orientation, pose, and expression', calls[0][1])
            self.assertIn('without repeating their physical appearance or clothing', calls[0][1])
            self.assertIn('walking-like pose', calls[0][1])
            self.assertIn('overlapping events for simultaneous actions', calls[0][1])
            self.assertIn('user-verified', calls[0][1])
            self.assertIn('This is an intermediate analysis window', calls[0][1])
            self.assertIn('This is the final analysis window', calls[1][1])
            self.assertIn('Exclude the next-shot frames', calls[1][1])
            self.assertIn('If the transition is unclear, keep the frames', calls[1][1])
            self.assertIn('labels_confirmed_by_queue', calls[0][1])
            self.assertIn('do not assume who is present or what they do', calls[0][1].lower())
            rejected = []
            response = 'not json'
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(video_model.media, 'prepare_analysis_video', side_effect=prepare), patch.object(video_model, 'video_inputs', side_effect=encode):
                with self.assertRaisesRegex(RuntimeError, 'Invalid Qwen3-VL video response'):
                    video_model.analyze_video(path / 'original.mp4', {'start_time': 20, 'end_time': 29}, {'video_width': 640, 'video_height': 360, 'video_fps': 4}, path, path, on_invalid_response=lambda *args: rejected.append(args))
            self.assertEqual(len(rejected), 2)
            self.assertEqual(rejected[0][:3], (0, 20.0, 28.0))
            self.assertEqual(rejected[0][4], response)

    def test_truncated_window_is_retried_with_concise_prompt_and_counted(self):
        class Tokens:
            def __init__(self, count): self.shape = (1, count)
            def __getitem__(self, _key): return self
        class Inputs(dict):
            def to(self, _device): return self
        class Processor:
            responses = iter(['{"overview":"unfinished', json.dumps({'overview': 'scene', 'events': [], 'uncertainties': ''})])
            def batch_decode(self, *_args, **_kwargs): return [next(self.responses)]
        class Model:
            device = 'cuda'
            def __init__(self): self.limits = []
            def generate(self, **kwargs):
                self.limits.append(kwargs['max_new_tokens'])
                return Tokens(10 + (2048 if len(self.limits) == 1 else 20))
        model = Model()
        prompts, rejected = [], []
        def inputs(_processor, _clip, _fps, prompt):
            prompts.append(prompt)
            return Inputs(input_ids=Tokens(10))
        with patch.object(video_model, 'video_inputs', side_effect=inputs):
            output, usage, attempts = video_model.generate_window(Processor(), model, SimpleNamespace(inference_mode=nullcontext), 'clip', 4, 'prompt', 8, 20, lambda *args: rejected.append(args))
        self.assertEqual(output['overview'], 'scene')
        self.assertEqual(model.limits, [2048, 4096])
        self.assertEqual(attempts, 2)
        self.assertIn('2048-token generation limit', rejected[0][0])
        self.assertIn('Do not repeat the same event', prompts[1])
        self.assertEqual(usage, {'prompt_tokens': 20, 'completion_tokens': 2068, 'total_tokens': 2088})

    def test_invalid_retry_remains_a_failure_and_preserves_each_response(self):
        class Inputs(dict):
            def to(self, _device): return self
        class Model:
            device = 'cuda'
            def generate(self, **_kwargs): return Tensor()
        class Processor:
            def batch_decode(self, *_args, **_kwargs): return ['{broken']
        rejected = []
        with patch.object(video_model, 'video_inputs', return_value=Inputs(input_ids=Tensor())):
            with self.assertRaisesRegex(RuntimeError, 'after 2 attempts'):
                video_model.generate_window(Processor(), Model(), SimpleNamespace(inference_mode=nullcontext), 'clip', 4, 'prompt', 8, 20, lambda *args: rejected.append(args))
        self.assertEqual([raw for _, raw in rejected], ['{broken', '{broken'])

    def test_generation_checks_cancellation_during_decoding(self):
        class Inputs(dict):
            def to(self, _device): return self
        class Cancelled(Exception): pass
        class StoppingCriteria: pass
        class StoppingCriteriaList(list): pass
        calls = []
        def check():
            calls.append(True)
            if len(calls) == 2:
                raise Cancelled()
        class Model:
            device = 'cuda'
            def generate(self, **kwargs):
                kwargs['stopping_criteria'][0](None, None)
                raise AssertionError('generation should have been interrupted')
        fake_transformers = SimpleNamespace(StoppingCriteria=StoppingCriteria, StoppingCriteriaList=StoppingCriteriaList)
        with patch.dict(sys.modules, {'transformers': fake_transformers}), patch.object(video_model, 'video_inputs', return_value=Inputs(input_ids=Tensor())):
            with self.assertRaises(Cancelled):
                video_model.generate_window(object(), Model(), SimpleNamespace(inference_mode=nullcontext), 'clip', 4, 'prompt', 8, 20, cancel_check=check)
        self.assertEqual(len(calls), 2)

    def test_completed_windows_are_reused_without_loading_models(self):
        windows = [{'source': {'source_start': 20, 'source_end': 28}, 'analysis': {'events': []}},
                   {'source': {'source_start': 28, 'source_end': 29}, 'analysis': {'events': []}}]
        result = video_model.analyze_video('missing.mp4', {'start_time': 20, 'end_time': 29},
            {'video_width': 640, 'video_height': 360, 'video_fps': 4}, '/missing-model', '/tmp',
            load_window=lambda index, start, end: windows[index])
        self.assertEqual(result['windows'], windows)
