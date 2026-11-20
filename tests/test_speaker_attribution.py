import json
from unittest.mock import patch

from app import db, inference_worker, speaker_attribution, worker
from tests import test_audio_worker


class AttributionTests(test_audio_worker.AudioWorkerCase):
    def test_only_open_model_lines_receive_speakers_and_they_keep_alignment(self):
        with db.connect() as conn:
            for segment_id, at, source, character, confidence, origin in [
                ('voiced', 0, 'voice', 'c', 'high', 'model'),
                ('open', 2, None, None, None, 'model'),
                ('manual', 4, 'manual', None, None, 'manual_edit'),
            ]:
                evidence = json.dumps({'evidence': [['c', .4]]})
                conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,
                    origin,character_context_id,words_json,speaker_source,confidence,evidence_json)
                    VALUES (?,'e',?,?,?,'A',?,?,?,?,?,?)""",
                    (segment_id, at, at + 1, 'Hello.', origin, character, json.dumps([[at, at + 1, 'Hello.']]),
                     source, confidence, evidence))
            job_id = speaker_attribution.insert(conn, 'e')
        self.assertEqual(worker.claim_job(('speaker_attribution',))[0], job_id)
        owner = db.one('SELECT lease_owner FROM jobs WHERE id=?', (job_id,))['lease_owner']

        def chat(_model, _system, evidence, _fields, **kwargs):
            self.assertIn('"decided": "Alice"', evidence)
            self.assertEqual(kwargs['response_format']['json_schema']['schema']['properties']['lines']['maxItems'], 1)
            answer = {'lines': [{'id': 'L2', 'speaker': 'Alice', 'confidence': 'high', 'reason': 'turn'}]}
            return kwargs['validator'](answer), 'qwen'

        with patch.object(inference_worker, 'chat_json', side_effect=chat) as model:
            speaker_attribution.process(job_id, owner)
        model.assert_called_once()
        row = db.one("SELECT * FROM transcript_segments WHERE parent_id='open'")
        self.assertEqual((row['speaker_source'], row['confidence'], row['character_context_id']), ('qwen', 'medium', 'c'))
        self.assertEqual(json.loads(row['words_json']), [[2, 3, 'Hello.']])
        self.assertEqual(json.loads(row['evidence_json'])['qwen']['reason'], 'turn')
        self.assertEqual(db.one("SELECT status FROM jobs WHERE id=?", (job_id,))['status'], 'done')
        self.assertEqual(db.one("SELECT COUNT(*) AS n FROM transcript_segments")['n'], 4)
