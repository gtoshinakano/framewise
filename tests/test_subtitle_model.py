import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.subtitle_model import inspect_subtitles


class Image:
    width, height = 100, 100
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def convert(self, mode): return self


class SubtitleModelTest(unittest.TestCase):
    def test_detection_is_subtitle_only_and_ocr_requires_transcript(self):
        calls = []
        class Model:
            def predict(self, image, task, **kwargs):
                calls.append((task, kwargs))
                return SimpleNamespace(response=task)
        def parse(response, width, height):
            return {'subtitles': [{'type': 'box', 'coords': [0, 80, 100, 100]}]} if response == 'detection' else {'Olá': [{'type': 'box', 'coords': [10, 85, 90, 95]}]}
        modules = {'PIL': SimpleNamespace(Image=SimpleNamespace(open=lambda path: Image())), 'wevisone_vl': SimpleNamespace(), 'wevisone_vl.visualization': SimpleNamespace(parse_predictions=parse)}
        cut = {'start_time': 0, 'end_time': 2}
        config = {'frame_mode': 'uniform', 'frame_count': 2}
        with tempfile.TemporaryDirectory() as scratch, patch.dict(sys.modules, modules), patch('app.subtitle_model.media.extract_cut_frame', side_effect=lambda source, target, sample, *a, **kw: sample):
            value = inspect_subtitles(Path('original.mp4'), cut, config, False, scratch, Model())
            self.assertEqual([kind for kind, _ in calls], ['detection', 'detection'])
            self.assertEqual(calls[0][1]['query'], ['subtitles'])
            self.assertFalse(value['ocr_performed'])
            calls.clear()
            value = inspect_subtitles(Path('original.mp4'), cut, config, True, scratch, Model())
            self.assertEqual([kind for kind, _ in calls], ['detection', 'ocr', 'detection', 'ocr'])
            self.assertEqual(value['events'][0]['text'], 'Olá')
            self.assertEqual(len(value['events'][0]['observations']), 2)
            calls.clear()
            single = inspect_subtitles(Path('original.mp4'), cut, {'frame_mode': 'uniform', 'frame_count': 1}, True, scratch, Model())
            self.assertEqual([kind for kind, _ in calls], ['detection', 'ocr', 'detection'])
            self.assertEqual([sample['selected_for_ocr'] for sample in single['samples']], [True, False])
