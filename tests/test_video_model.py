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
        response = json.dumps({'overview': 'scene', 'camera': {'movement': 'static', 'detail': 'none'}, 'characters': [{'name': 'Mia', 'activity': 'standing', 'travel': 'none', 'motion': 'Turns.', 'speaking': 'yes', 'emotion': 'calm', 'end': 'left'}], 'events': [event], 'uncertainties': 'limited'})
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
                roster = [{'name': 'Mia Lee', 'character_context_id': 'mia', 'at_start': True, 'at_end': True, 'start_box': [10, 20, 300, 900], 'end_box': [600, 20, 900, 900], 'position': 'left, facing right', 'appearance': 'red coat', 'emotion': 'calm'}]
                segments = [{'start_time': 21, 'end_time': 22.5, 'text': 'secret words', 'character_name': 'Mia Lee'}]
                result = video_model.analyze_video(path / 'original.mp4', {'start_time': 20, 'end_time': 29}, {'video_width': 640, 'video_height': 360, 'video_fps': 4}, path, path,
                    context_hints={'character_detections': {'labels_confirmed_by_queue': True, 'detections': [{'id': 'secret-box'}]}, 'selected_context': [{'kind': 'world', 'title': 'School', 'description': 'A calm school'}]},
                    roster=roster, initial={'composition': 'Medium shot of Mia Lee', 'background_people': 'none'}, transcript=segments)
            self.assertEqual([(x['source']['source_start'], x['source']['source_end']) for x in result['windows']], [(20, 28), (28, 29)])
            self.assertEqual(result['windows'][1]['analysis']['events'][0]['episode_start_seconds'], 28)
            self.assertFalse((path / 'video-window.mp4').exists())
            self.assertEqual(calls[0][0], 4)
            first, second = calls[0][1], calls[1][1]
            self.assertIn('do not describe outfits', first)
            self.assertIn('Report travel only when a position actually changes', first)
            self.assertIn('overlapping events for simultaneous actions', first)
            self.assertIn('identified by the user', first)
            self.assertIn('part 1 of 2 of the cut, an intermediate part', first)
            self.assertIn('part 2 of 2 of the cut, the final part: its last frames are the end of the cut', second)
            self.assertNotIn('silent', first.lower())
            # The model is told who to follow, where they start and whether they are still there at the end.
            self.assertIn('- Mia Lee: first frame: left, facing right, box [10, 20, 300, 900]. look: red coat. emotion in the first frame: calm. last frame of the cut: in frame (right, box [600, 20, 900, 900]).', first)
            self.assertIn('Opening shot: Medium shot of Mia Lee', first)
            self.assertNotIn('Opening shot', second)
            # Speech timing without the words, relative to each window.
            self.assertIn('1.0-2.5 Mia Lee', first)
            self.assertIn('none in the transcript', second)
            self.assertNotIn('secret words', first)
            # Series notes are readable text; detection rows and ids are not sent.
            self.assertIn('world School: A calm school', first)
            self.assertNotIn('secret-box', first)
            self.assertNotIn('labels_confirmed_by_queue', first)
            # The second window continues from what the first one reported.
            self.assertIn('this clip continues the same cut', second)
            self.assertIn('- Mia Lee: left', second)
            self.assertNotIn('continues the same cut', first)
            # Answers are tied to the character record and merged into one direction for the cut.
            person = result['windows'][0]['analysis']['characters'][0]
            self.assertEqual((person['character'], person['character_context_id'], person['speaking']), ('Mia Lee', 'mia', 'yes'))
            self.assertEqual(result['contract'], video_model.VIDEO_CONTRACT)
            self.assertEqual(result['direction']['camera']['movement'], 'static')
            self.assertEqual(result['direction']['characters'][0]['character_context_id'], 'mia')
            self.assertEqual(result['direction']['end_state']['in_frame'], ['Mia Lee'])
            self.assertTrue(result['direction']['characters'][0]['in_last_frame'])
            self.assertEqual(result['windows'][0]['source']['contract'], video_model.VIDEO_CONTRACT)
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
        source = {'boundary_policy': video_model.BOUNDARY_POLICY, 'contract': video_model.VIDEO_CONTRACT, 'source_file': 'missing.mp4'}
        windows = [{'source': {'source_start': 20, 'source_end': 28, **source}, 'analysis': {'events': []}},
                   {'source': {'source_start': 28, 'source_end': 29, **source}, 'analysis': {'events': []}}]
        result = video_model.analyze_video('missing.mp4', {'start_time': 20, 'end_time': 29},
            {'video_width': 640, 'video_height': 360, 'video_fps': 4}, '/missing-model', '/tmp',
            load_window=lambda index, start, end: windows[index])
        self.assertEqual(result['windows'], windows)
        self.assertEqual(result['direction']['characters'], [])

    def test_windows_saved_under_an_older_prompt_are_analyzed_again(self):
        # A retried job from before motion directions holds windows without the contract marker.
        old = {'source': {'source_start': 20, 'source_end': 21, 'boundary_policy': video_model.BOUNDARY_POLICY, 'source_file': 'missing.mp4'}, 'analysis': {'overview': 'old', 'events': []}}
        with patch.object(video_model.VideoRuntime, 'load', side_effect=RuntimeError('model requested')):
            with self.assertRaisesRegex(RuntimeError, 'model requested'):  # it goes on to the model instead of reusing the window
                video_model.analyze_video('missing.mp4', {'start_time': 20, 'end_time': 21}, {'video_width': 640, 'video_height': 360, 'video_fps': 4}, '/missing-model', '/tmp',
                                          load_window=lambda index, start, end: old)

    def test_preview_boundaries_replace_older_unbounded_window(self):
        class Model:
            device = 'cuda'
            def to(self, _device): return self
            def eval(self): return self
        transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: object()), AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None), bfloat16='bf16')
        prepared, saved = [], []
        def prepare(source, clip, start, end, width, height, fps):
            prepared.append((source.name, start, end))
            clip.touch()
            return {'source_start': start, 'source_end': end, 'width': width, 'height': height, 'fps': fps}
        old = {'source': {'source_start': 20, 'source_end': 21}, 'analysis': {'events': []}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"weights.safetensors"}}')
            (path / 'weights.safetensors').touch()
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), \
                 patch.object(video_model.media, 'prepare_analysis_video', side_effect=prepare), \
                 patch.object(video_model, 'generate_window', return_value=({'events': []}, {}, 1)):
                result = video_model.analyze_video(path / 'proxy.mp4', {'start_time': 20, 'end_time': 21.2, 'first_frame_time': 20.04, 'last_frame_time': 20.96},
                    {'video_width': 640, 'video_height': 360, 'video_fps': 4}, path, path,
                    load_window=lambda *_: old, save_window=lambda index, window: saved.append((index, window)))
        self.assertEqual(prepared[0][0], 'proxy.mp4')
        self.assertAlmostEqual(prepared[0][1], 20.04)
        self.assertAlmostEqual(prepared[0][2], 20.960002)
        self.assertEqual(result['windows'][0]['source']['boundary_policy'], video_model.BOUNDARY_POLICY)
        self.assertEqual(result['windows'][0]['source']['source_file'], 'proxy.mp4')
        self.assertEqual(saved[0][0], 0)
