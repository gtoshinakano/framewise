import json
import tempfile
import types
import unittest
import wave
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
        self.old_dir, self.old_path, self.old_media_dir = db.DATA_DIR, db.DB_PATH, audio_worker.media.DATA_DIR
        db.DATA_DIR = audio_worker.media.DATA_DIR = Path(self.tmp.name)
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
        db.DATA_DIR, db.DB_PATH, audio_worker.media.DATA_DIR = self.old_dir, self.old_path, self.old_media_dir
        self.tmp.cleanup()

    def fake_clip(self, source, target, start, end):
        with wave.open(str(target), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            out.writeframes(b'\x01\x00' * round((end - start) * 16000))

    def test_enrollment_and_transcription_store_results_under_lease(self):
        job = worker.enqueue('e', 'voice_sample', {'character_context_id': 'c', 'start_time': 1, 'end_time': 4})
        claimed = worker.claim_job(('voice_sample',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio', side_effect=self.fake_clip), patch.object(audio_worker, 'diarize', return_value=self.output):
            audio_worker.process(*claimed)
        sample = db.one('SELECT * FROM character_voice_samples')
        self.assertEqual(sample['character_context_id'], 'c')
        self.assertTrue(audio_worker.media.voice_reference_path('p', sample['id']).is_file())
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
        self.assertEqual((row['text'], row['speaker_label'], row['character_context_id'], row['deleted']), ('Hello', 'SPEAKER_00', 'c', 0))
        self.assertAlmostEqual(row['start_time'], 1.2)
        self.assertEqual(row['job_id'], job)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')

        # A rerun replaces untouched model output and preserves human edits.
        old_id = row['id']
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin,model_id) VALUES('legacy','e',3,4,'Old text','model',?)", (audio_worker.ASR,))
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin,model_id) VALUES('rejected','e',6,7,'Human note','manual_edit',?)", (audio_worker.ASR,))
        job = worker.enqueue('e', 'transcribe')
        claimed = worker.claim_job(('transcribe',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        self.assertIsNotNone(db.one('SELECT superseded_by FROM transcript_segments WHERE id=?', (old_id,))['superseded_by'])
        self.assertIsNotNone(db.one("SELECT superseded_by FROM transcript_segments WHERE id='legacy'")['superseded_by'])
        self.assertIsNone(db.one("SELECT superseded_by FROM transcript_segments WHERE id='rejected'")['superseded_by'])
        self.assertEqual(len(db.all_rows("SELECT * FROM transcript_segments WHERE superseded_by IS NULL AND deleted=0")), 2)
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET origin='manual_edit',text='Reviewed' WHERE superseded_by IS NULL AND deleted=0 AND origin='model'")
        job = worker.enqueue('e', 'transcribe')
        claimed = worker.claim_job(('transcribe',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        active = db.all_rows("SELECT * FROM transcript_segments WHERE superseded_by IS NULL AND deleted=0")
        self.assertEqual({row['text'] for row in active}, {'Reviewed', 'Human note'})

    def test_ambiguous_enrollment_fails_without_creating_sample(self):
        job = worker.enqueue('e', 'voice_sample', {'character_context_id': 'c', 'start_time': 1, 'end_time': 4})
        claimed = worker.claim_job(('voice_sample',))
        self.output.speaker_diarization.labels = lambda: ['A', 'B']
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output):
            audio_worker.process(*claimed)
        self.assertIsNone(db.one('SELECT * FROM character_voice_samples'))
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'failed')

    def test_multiple_ranges_are_combined_and_used_for_matching(self):
        ranges = [{'start_time': 1, 'end_time': 2}, {'start_time': 5, 'end_time': 6.5}]
        job = worker.enqueue('e', 'voice_sample', {'character_context_id': 'c', 'ranges': ranges})
        claimed = worker.claim_job(('voice_sample',))
        def clip(source, target, start, end):
            with wave.open(str(target), 'wb') as out:
                out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                out.writeframes((b'\x01\x00' if start == 1 else b'\x02\x00') * round((end - start) * 16000))
        def inspect(sample):
            with wave.open(str(sample), 'rb') as wav:
                self.assertEqual(wav.getnframes(), 40000)
                self.assertEqual(wav.readframes(1), b'\x01\x00')
                wav.setpos(16000)
                self.assertEqual(wav.readframes(1), b'\x02\x00')
            return self.output
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio', side_effect=clip), patch.object(audio_worker, 'diarize', side_effect=inspect):
            audio_worker.process(*claimed)
        sample = db.one('SELECT * FROM character_voice_samples')
        self.assertEqual(json.loads(sample['ranges_json']), ranges)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')
        self.assertEqual(audio_worker.match_voices({'speaker': [1, 0]}, [{'character_id': sample['character_context_id'], 'embedding': json.loads(sample['embedding_json'])}])['speaker']['character_id'], 'c')

    def test_upload_is_processed_without_episode_audio_and_cleaned(self):
        upload_id = '5ea4cdab-c60a-418c-b45f-14583b73bcf1'
        upload = self.path / 'voice-uploads' / upload_id
        upload.parent.mkdir()
        upload.write_bytes(b'encoded audio')
        (self.path / 'audio.wav').unlink()
        job = worker.enqueue('e', 'voice_sample', {'character_context_id': 'c', 'upload_id': upload_id, 'source_filename': 'voice.mp3'})
        claimed = worker.claim_job(('voice_sample',))
        def convert(args):
            with wave.open(args[-1], 'wb') as out:
                out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                out.writeframes(b'\x01\x00' * 48000)
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker.media, 'run', side_effect=convert), patch.object(audio_worker.media, 'extract_audio') as extract, patch.object(audio_worker, 'diarize', return_value=self.output):
            audio_worker.process(*claimed)
        extract.assert_not_called()
        sample = db.one('SELECT * FROM character_voice_samples')
        self.assertEqual(sample['source_filename'], 'voice.mp3')
        self.assertTrue(audio_worker.media.voice_reference_path('p', sample['id']).is_file())
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')
        self.assertFalse(upload.exists())
