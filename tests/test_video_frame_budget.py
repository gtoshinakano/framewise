import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import media, video_model
from tests.test_video_model import Tensor


class FrameBudgetTest(unittest.TestCase):
    def test_automatic_budget_limits_encoded_frames_across_windows(self):
        response = json.dumps({'overview': 'scene', 'events': [], 'uncertainties': 'none'})
        class Processor:
            def batch_decode(self, *args, **kwargs): return [response]
        class Model:
            device = 'cuda'
            def to(self, _device): return self
            def eval(self): return self
            def generate(self, **kwargs): return Tensor()
        transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: Processor()), AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None), bfloat16='bf16', inference_mode=nullcontext)
        class Inputs(dict):
            def to(self, _device): return self
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'model.safetensors.index.json').write_text('{"weight_map":{"w":"weights.safetensors"}}')
            (path / 'weights.safetensors').touch()
            source = path / 'original.mp4'
            media.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc=size=160x120:rate=2:duration=250', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
            with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), patch.object(video_model, 'video_inputs', return_value=Inputs(input_ids=Tensor())):
                for duration in (.17, 1, 8, 9.5, 20.35, 39.16, 249):
                    with self.subTest(duration=duration):
                        config = {'video_width': 128, 'video_height': 128, 'video_fps': min(30, 56 / duration), 'video_frame_budget': 60}
                        result = video_model.analyze_video(source, {'start_time': 0, 'end_time': duration}, config, path, path)
                        counts = [w['source']['frame_count'] for w in result['windows']]
                        self.assertTrue(all(2 <= count <= 32 for count in counts), counts)
                        self.assertLessEqual(sum(counts), 60)
                        self.assertAlmostEqual(result['windows'][-1]['source']['source_end'], duration)


class FrameBudgetSettingsTest(unittest.TestCase):
    from tests.test_analysis import AnalysisTest
    setUp = AnalysisTest.setUp
    tearDown = AnalysisTest.tearDown

    def test_default_and_manual_fps_remain_configurable(self):
        from app import db
        default = self.client.get('/api/cuts/c1/analysis-config').json()
        self.assertEqual(default['video_frame_budget'], 60)
        self.assertEqual(default['video_fps'], 28)
        with db.connect() as conn:
            conn.execute("INSERT INTO cut_analysis_configs(cut_id,config_json) VALUES ('c1',?)", (json.dumps(default),))
            conn.execute("UPDATE cuts SET end_time=4 WHERE id='c1'")
        self.assertEqual(self.client.get('/api/cuts/c1/analysis-config').json()['video_fps'], 14)
        custom = {**default, 'video_frame_budget': None, 'video_fps': .1}
        self.assertEqual(self.client.put('/api/cuts/c1/analysis-config', json=custom).status_code, 200)
        self.assertEqual(self.client.get('/api/cuts/c1/analysis-config').json()['video_fps'], .1)

    def test_detected_character_context_is_included_without_selection(self):
        from app import db
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c1',0,2,'[]','test')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('box','run','first','[0,0,1,1]','ctx','manual')")
        response = self.client.post('/api/cuts/c1/analyze')
        self.assertEqual(response.status_code, 200, response.text)
        payload = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?',(response.json()['job_id'],))['payload_json'])
        self.assertEqual([item['id'] for item in payload['cuts'][0]['context']['context_items']], ['ctx'])
