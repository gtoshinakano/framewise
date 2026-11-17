import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app import frame_cache


class TargetPreviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source.mp4'
        self.source.write_bytes(b'video')
        self.spec = {'source': self.source, 'time': 1.0, 'start': 0.0, 'end': 2.0,
                     'width': 640, 'last': False, 'ordinal': 20}

    def test_unique_pts_uses_seek_and_is_cached(self):
        calls = [SimpleNamespace(stdout='0.0\n'), SimpleNamespace(stdout='0.9\n1.000000\n1.1\n')]
        def extract(source, output, *args, **kwargs):
            output.write_bytes(b'preview')
            return 1.0
        with frame_cache._lock:
            frame_cache._cache.clear()
        with patch.object(frame_cache.media, 'run', side_effect=calls), patch.object(frame_cache.media, 'extract_cut_frame', side_effect=extract) as seek, patch.object(frame_cache.subprocess, 'run') as fallback:
            first = frame_cache.request(self.spec).result(timeout=5)
            second = frame_cache.request(self.spec).result(timeout=5)
        self.assertEqual(first, b'preview')
        self.assertEqual(second, first)
        self.assertEqual(seek.call_count, 1)
        fallback.assert_not_called()

    def test_ambiguous_pts_falls_back_to_ordinal(self):
        calls = [SimpleNamespace(stdout='0.0\n'), SimpleNamespace(stdout='1.0\n1.0\n')]
        with patch.object(frame_cache.media, 'run', side_effect=calls), patch.object(frame_cache.media, 'extract_cut_frame') as seek, patch.object(frame_cache.subprocess, 'run', return_value=SimpleNamespace(stdout=b'exact')) as fallback:
            data = frame_cache._target_frame(self.spec, Path(self.tmp.name) / 'out.jpg')
        self.assertEqual(data, b'exact')
        seek.assert_not_called()
        self.assertIn('select=eq(n\\,20)', fallback.call_args.args[0][fallback.call_args.args[0].index('-vf') + 1])

    def test_seek_mismatch_falls_back_to_ordinal(self):
        calls = [SimpleNamespace(stdout='0.0\n'), SimpleNamespace(stdout='1.0\n')]
        with patch.object(frame_cache.media, 'run', side_effect=calls), patch.object(frame_cache.media, 'extract_cut_frame', return_value=1.04), patch.object(frame_cache.subprocess, 'run', return_value=SimpleNamespace(stdout=b'exact')) as fallback:
            data = frame_cache._target_frame(self.spec, Path(self.tmp.name) / 'out.jpg')
        self.assertEqual(data, b'exact')
        fallback.assert_called_once()

    def test_source_changed_while_extracting_is_rejected(self):
        def replace_source(spec, output):
            self.source.write_bytes(b'new-video')
            return b'jpeg'
        with patch.object(frame_cache, '_target_frame', side_effect=replace_source):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                frame_cache._generate(self.spec)

    def test_source_changes_invalidate_cache_key(self):
        first = frame_cache._key(self.spec)
        self.source.write_bytes(b'new-video')
        self.assertNotEqual(first, frame_cache._key(self.spec))

if __name__ == '__main__': unittest.main()
