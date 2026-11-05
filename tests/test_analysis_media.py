"""Preparation for the single-cut video analysis input."""
import tempfile
import json
import unittest
from pathlib import Path
from app import media


class AnalysisMediaTest(unittest.TestCase):
    def test_prepares_silent_padded_clip_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'original.mp4', Path(folder) / 'analysis.mp4'
            media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=24:duration=2', '-pix_fmt', 'yuv420p', str(source)])
            before = source.read_bytes()
            metadata = media.prepare_analysis_video(source, output, .25, 1.25)
            self.assertEqual(metadata, {'source_start': .25, 'source_end': 1.25, 'width': 640, 'height': 360, 'fps': 4, 'padded_tail': False})
            self.assertEqual((media.probe(output)['width'], media.probe(output)['height']), (640, 360))
            self.assertFalse(media.probe(output)['has_audio'])
            self.assertEqual(source.read_bytes(), before)
            metadata = media.prepare_analysis_video(source, output, 0, .1)
            self.assertTrue(metadata['padded_tail'])
            probe = media.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(output)])
            self.assertGreaterEqual(int(json.loads(probe.stdout)['streams'][0]['nb_read_frames']), 2)
            metadata = media.prepare_analysis_video(source, output, 0, .05, fps=.25)
            self.assertTrue(metadata['padded_tail'])
            self.assertIn('single_sample_time', metadata)
            self.assertEqual(media.probe(output)['width'], 640)
            with self.assertRaises(ValueError):
                media.prepare_analysis_video(source, source, 0, 1)
            with self.assertRaises(ValueError):
                media.prepare_analysis_video(source, output, 1, 0)
