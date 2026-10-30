import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from app import audio_worker, db, worker


class FakeAnnotation:
    def labels(self):
        return ['SPEAKER_00']

    def __iter__(self):
        return iter([(types.SimpleNamespace(start=1., end=3.), 'SPEAKER_00')])


class Vector:
    def tolist(self):
        return [1., 0.]


class AudioWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_dir, self.old_path = db.DATA_DIR, db.DB_PATH
        db.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'db.sqlite'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Project')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,duration,has_audio,settings_json) VALUES('e','p','Episode','x.mp4','.mp4',10,1,'{}')")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES('c','p','character','Alice')")
        self.path = db.DATA_DIR / 'episode'
        self.path.mkdir()
        (self.path / 'audio.wav').touch()
        annotation = FakeAnnotation()
        self.output = types.SimpleNamespace(speaker_diarization=annotation, exclusive_speaker_diarization=annotation, speaker_embeddings=[Vector()])

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH = self.old_dir, self.old_path
        self.tmp.cleanup()

    def test_enrollment_and_transcription_store_results_under_lease(self):
        job = worker.enqueue('e', 'voice_sample', {'character_context_id': 'c', 'start_time': 1, 'end_time': 4})
        claimed = worker.claim_job(('voice_sample',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output):
            audio_worker.process(*claimed)
        sample = db.one('SELECT * FROM character_voice_samples')
        self.assertEqual(sample['character_context_id'], 'c')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')

        job = worker.enqueue('e', 'transcribe')
        claimed = worker.claim_job(('transcribe',))
        stamp = types.SimpleNamespace(start_time=1.2, end_time=1.8, text='Hello')
        fake_model = types.SimpleNamespace(transcribe=lambda **kwargs: [types.SimpleNamespace(text='Hello', language='English', time_stamps=[stamp])])
        qwen = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_model))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), float32='float32', bfloat16='bf16')
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        row = db.one('SELECT * FROM transcript_segments')
        self.assertEqual((row['text'], row['speaker_label'], row['character_context_id'], row['status']), ('Hello', 'SPEAKER_00', 'c', 'draft'))
        self.assertAlmostEqual(row['start_time'], 1.2)
        self.assertEqual(row['job_id'], job)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')

        # A rerun replaces untouched model drafts through revision history.
        old_id = row['id']
        job = worker.enqueue('e', 'transcribe')
        claimed = worker.claim_job(('transcribe',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        self.assertIsNotNone(db.one('SELECT superseded_by FROM transcript_segments WHERE id=?', (old_id,))['superseded_by'])
        self.assertEqual(len(db.all_rows("SELECT * FROM transcript_segments WHERE superseded_by IS NULL AND status='draft'")), 1)
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET origin='manual_edit',text='Reviewed',status='approved' WHERE superseded_by IS NULL AND status='draft'")
        job = worker.enqueue('e', 'transcribe')
        claimed = worker.claim_job(('transcribe',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        active = db.all_rows("SELECT * FROM transcript_segments WHERE superseded_by IS NULL AND status!='deleted'")
        self.assertEqual([row['text'] for row in active], ['Reviewed'])

    def test_ambiguous_enrollment_fails_without_creating_sample(self):
        job = worker.enqueue('e', 'voice_sample', {'character_context_id': 'c', 'start_time': 1, 'end_time': 4})
        claimed = worker.claim_job(('voice_sample',))
        self.output.speaker_diarization.labels = lambda: ['A', 'B']
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output):
            audio_worker.process(*claimed)
        self.assertIsNone(db.one('SELECT * FROM character_voice_samples'))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'failed')
