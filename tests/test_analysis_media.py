"""Preparation for the single-cut video analysis input."""
import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app import media, video_model


class AnalysisMediaTest(unittest.TestCase):
    def test_frame_sampling_never_includes_the_next_cut(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'original.mp4', Path(folder) / 'analysis.mp4'
            media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                       'color=red:s=160x120:r=25:d=1', '-f', 'lavfi', '-i', 'color=blue:s=160x120:r=25:d=1',
                       '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]',
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
            for start, end, fps in ((0, 1, 4), (0, .960002, 28), (.75, 1, 4), (.96, .960002, 4)):
                with self.subTest(start=start, end=end, fps=fps):
                    media.prepare_analysis_video(source, output, start, end, width=160, height=128, fps=fps)
                    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-vf', 'scale=1:1',
                                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
                    self.assertGreaterEqual(len(decoded), 6)
                    for offset in range(0, len(decoded), 3):
                        red, _, blue = decoded[offset:offset + 3]
                        self.assertGreater(red, blue + 100, f'Next cut appeared in frame {offset // 3}')

    def test_model_receives_only_frames_through_the_saved_last_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source = path / 'proxy.mp4'
            media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                       'color=red:s=160x120:r=25:d=1', '-f', 'lavfi', '-i', 'color=blue:s=160x120:r=25:d=1',
                       '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]', '-map', '[v]',
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
            (path / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"weights.safetensors"}}')
            (path / 'weights.safetensors').touch()
            class Model:
                device = 'cuda'
                def to(self, _device): return self
                def eval(self): return self
            transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
                                           AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None), bfloat16='bf16')
            decoded = []
            def inspect(_processor, _model, _torch, clip, *_args):
                decoded.append(subprocess.run(['ffmpeg', '-v', 'error', '-i', str(clip), '-vf', 'scale=1:1',
                                               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout)
                return {'events': []}, {}, 1
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(video_model, 'generate_window', side_effect=inspect):
                result = video_model.analyze_video(source, {'start_time': 0, 'end_time': 1.4, 'first_frame_time': .04, 'last_frame_time': .96},
                    {'video_width': 160, 'video_height': 128, 'video_fps': 28, 'video_frame_budget': 32}, path, path)
            self.assertAlmostEqual(result['windows'][-1]['source']['source_end'], .960002)
            self.assertEqual(result['windows'][0]['source']['source_file'], 'proxy.mp4')
            self.assertTrue(decoded and len(decoded[0]) >= 6)
            for frames in decoded:
                for offset in range(0, len(frames), 3):
                    red, _, blue = frames[offset:offset + 3]
                    self.assertGreater(red, blue + 100)

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
