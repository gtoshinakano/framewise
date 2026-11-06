import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.character_detection import detect_boundary_people, select_analysis_references


class Image:
    width, height = 100, 80
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def convert(self, mode): return self


class CharacterDetectionTest(unittest.TestCase):
    def test_detects_both_boundaries_and_filters_invalid_boxes(self):
        calls, times = [], []
        class Model:
            def predict(self, image, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(response='response', prompt='prompt')
        def parse(*args):
            return {'person': [{'type': 'box', 'coords': [-5, 5, 55, 90]},
                               {'type': 'point', 'coords': [1, 2]},
                               {'type': 'box', 'coords': [20, 10, 10, 50]}]}
        def extract(source, target, sample, start, end, **kwargs):
            times.append((sample, kwargs['last']))
            return sample
        modules = {'PIL': SimpleNamespace(Image=SimpleNamespace(open=lambda _: Image())),
                   'wevisone_vl': SimpleNamespace(), 'wevisone_vl.visualization': SimpleNamespace(parse_predictions=parse)}
        with tempfile.TemporaryDirectory() as scratch, patch.dict(sys.modules, modules), patch('app.character_detection.media.extract_cut_frame', side_effect=extract):
            frames = detect_boundary_people('original.mp4', {'start_time': 2, 'end_time': 5}, scratch, Model())
        self.assertEqual([frame['position'] for frame in frames], ['first', 'last'])
        self.assertEqual([kwargs['query'] for kwargs in calls], [['person'], ['person']])
        self.assertEqual(frames[0]['boxes'], [[0, 5, 55, 80]])
        self.assertEqual(times[0][1], False)
        self.assertEqual(times[1][1], True)
        self.assertLess(times[1][0], 5)

    def test_analysis_references_cover_distinct_characters_before_extra_views(self):
        refs = [{'id': identifier, 'character_context_id': character} for identifier, character in [('a1','a'),('a2','a'),('b1','b'),('c1','c'),('d1','d')]]
        self.assertEqual([ref['id'] for ref in select_analysis_references(refs, {'a','b','c','d'})], ['a1','b1','c1','d1'])
        self.assertEqual([ref['id'] for ref in select_analysis_references(refs, {'a','b'})], ['a1','b1','a2'])
