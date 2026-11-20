import json
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import audio_worker, db, worker


class FakeAnnotation:
    def labels(self):
        return ['SPEAKER_00']

    def __iter__(self):
        return iter([(types.SimpleNamespace(start=1., end=3.), 'SPEAKER_00')])


class Vector:
    def tolist(self):
        return [1., 0.]


class AudioWorkerCase(unittest.TestCase):
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
        self.fake_clip(None, self.path / 'audio.wav', 0, 10)
        self.addCleanup(patch.stopall)
        patch.object(audio_worker, 'load_diarizer', return_value=object()).start()
        patch.object(audio_worker, 'embed', return_value=np.array([1., 0.])).start()
        annotation = FakeAnnotation()
        self.output = types.SimpleNamespace(speaker_diarization=annotation, exclusive_speaker_diarization=annotation, speaker_embeddings=[Vector()])

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, audio_worker.media.DATA_DIR = self.old_dir, self.old_path, self.old_media_dir
        self.tmp.cleanup()

    def fake_clip(self, source, target, start, end):
        with wave.open(str(target), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            out.writeframes(b'\x01\x00' * round((end - start) * 16000))


class AudioWorkerTests(AudioWorkerCase):
    def test_invalid_alignment_retries_smaller_windows_without_losing_text(self):
        window = []
        def clip(_source, _target, start, end):
            window[:] = [start, end]
        def transcribe(**_kwargs):
            length = window[1] - window[0]
            stamp = types.SimpleNamespace(start_time=.2, end_time=length + 1 if length > 13 else .8, text='Hello')
            return [types.SimpleNamespace(text='Hello!', language='English', time_stamps=[stamp])]
        model = types.SimpleNamespace(transcribe=transcribe)
        with patch.object(audio_worker, 'clip_audio', side_effect=clip) as clipped:
            words = audio_worker.transcribe_window(model, self.path / 'audio.wav', self.path / 'sample.wav', 1300, 1325, [])
        self.assertEqual(clipped.call_count, 3)
        self.assertEqual([word['text'] for word in words], ['Hello!', 'Hello!'])
        self.assertAlmostEqual(words[0]['start_time'], 1300.2)
        self.assertAlmostEqual(words[1]['start_time'], 1312.7)
        self.assertTrue(all(word['speaker'] is None for word in words))

    def test_invalid_alignment_fails_after_bounded_retries(self):
        stamp = types.SimpleNamespace(start_time=float('nan'), end_time=1, text='Speech')
        model = types.SimpleNamespace(transcribe=lambda **_kwargs: [types.SimpleNamespace(text='Speech', language='English', time_stamps=[stamp])])
        with patch.object(audio_worker, 'clip_audio') as clipped:
            with self.assertRaisesRegex(RuntimeError, 'after retrying smaller excerpts'):
                audio_worker.transcribe_window(model, self.path / 'audio.wav', self.path / 'sample.wav', 0, 25, [])
        self.assertEqual(clipped.call_count, 4)

    def test_alignment_limits_small_rounding_excursions_and_zero_duration(self):
        stamps = [types.SimpleNamespace(start_time=-.01, end_time=.3, text='One'),
                  types.SimpleNamespace(start_time=10.02, end_time=10.02, text='two')]
        result = types.SimpleNamespace(text='One two.', language='English', time_stamps=stamps)
        words = audio_worker.aligned_words(result, 50, 60)
        self.assertEqual(''.join(word['text'] for word in words), 'One two.')
        self.assertEqual(words[0]['start_time'], 50)
        self.assertAlmostEqual(words[1]['start_time'], 59.99)
        self.assertEqual(words[1]['end_time'], 60)
        stamps[1].end_time = 10.5
        with self.assertRaises(audio_worker.AlignmentError):
            audio_worker.aligned_words(result, 50, 60)

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
        stamp = types.SimpleNamespace(start_time=.65, end_time=1.3, text='Hello')
        fake_model = types.SimpleNamespace(transcribe=lambda **kwargs: [types.SimpleNamespace(text='Hello', language='English', time_stamps=[stamp])])
        qwen = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_model))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), float32='float32', bfloat16='bf16')
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        row = db.one('SELECT * FROM transcript_segments')
        self.assertEqual((row['text'], row['speaker_label'], row['character_context_id'], row['deleted']), ('Hello', 'SPEAKER_00', 'c', 0))
        self.assertAlmostEqual(row['start_time'], 1.2)
        self.assertEqual(row['speaker_source'], 'voice')
        self.assertEqual(row['confidence'], 'high')
        self.assertEqual(json.loads(row['words_json'])[0][2], 'Hello')
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
        def inspect(_pipeline, sample):
            self.assertEqual(len(sample), 40000)
            self.assertEqual(sample[0], 1 / 32768)
            self.assertEqual(sample[16000], 2 / 32768)
            return self.output
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio', side_effect=clip), patch.object(audio_worker, 'diarize', side_effect=inspect):
            audio_worker.process(*claimed)
        sample = db.one('SELECT * FROM character_voice_samples')
        self.assertEqual(json.loads(sample['ranges_json']), ranges)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')
        self.assertEqual(json.loads(sample['range_embeddings_json']), [
            {'start': 1, 'end': 2, 'embedding': [1., 0.]},
            {'start': 5, 'end': 6.5, 'embedding': [1., 0.]},
        ])

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


class RangeRerunTests(AudioWorkerCase):
    """Redoing one interval changes that interval only."""

    def setUp(self):
        super().setUp()
        class Wide(FakeAnnotation):
            def __iter__(self):
                return iter([(types.SimpleNamespace(start=0., end=10.), 'SPEAKER_00')])
        wide = Wide()
        self.output = types.SimpleNamespace(speaker_diarization=wide, exclusive_speaker_diarization=wide, speaker_embeddings=[Vector()])
        model_id = audio_worker.ASR + ' + x'
        with db.connect() as conn:
            for row in (('before', 'model', 1, 2, 'Before', 'SPEAKER_04'), ('wrong', 'model', 5, 6, 'Wrnog', 'SPEAKER_09'),
                        ('hand', 'manual_edit', 6.5, 7, 'Mine', 'SPEAKER_09'), ('after', 'model', 9, 10, 'After', 'SPEAKER_04')):
                conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin,model_id) VALUES(?,'e',?,?,?,?,?,?)",
                             (row[0], *row[2:6], row[1], model_id if row[1] == 'model' else None))
        stamps = [types.SimpleNamespace(start_time=.5, end_time=1., text='Hello.'), types.SimpleNamespace(start_time=2.6, end_time=3., text='there')]
        self.model = types.SimpleNamespace(transcribe=lambda **kwargs: [types.SimpleNamespace(text='Hello. there', language='English', time_stamps=stamps)])

    def run_range(self, **extra):
        job = worker.enqueue('e', 'transcribe', {'range': {'start_time': 4, 'end_time': 8}, **extra})
        claimed = worker.claim_job(('transcribe',))
        qwen = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=lambda *a, **kw: self.model))
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), float32='float32', bfloat16='bf16')
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio') as clipped, patch.object(audio_worker, 'diarize', return_value=self.output), patch.dict('sys.modules', {'qwen_asr': qwen, 'torch': torch}):
            audio_worker.process(*claimed)
        return job, clipped

    def current(self):
        return {row['id']: row for row in db.all_rows("SELECT * FROM transcript_segments WHERE superseded_by IS NULL AND deleted=0")}

    def test_only_untouched_model_output_inside_the_interval_is_replaced(self):
        job, clipped = self.run_range()
        # Speakers are heard in the interval plus the audio around it; speech is read from the interval alone.
        clipped.assert_not_called()
        current = self.current()
        self.assertNotIn('wrong', current)
        self.assertEqual(current['hand']['text'], 'Mine')
        self.assertEqual({current[i]['text'] for i in ('before', 'after')}, {'Before', 'After'})
        new = [row for row in current.values() if row['job_id'] == job]
        # "there" lands on the hand-edited line, so it is not added over it.
        self.assertEqual([(row['text'], row['speaker_label'], row['origin']) for row in new], [('Hello.', 'SPEAKER_04', 'model')])
        self.assertAlmostEqual(new[0]['start_time'], 4.5)
        tombstone = db.one("SELECT * FROM transcript_segments WHERE parent_id='wrong'")
        self.assertEqual((tombstone['deleted'], tombstone['text'], tombstone['job_id']), (1, 'Wrnog', job))
        self.assertEqual(db.one('SELECT status,stage FROM jobs WHERE id=?', (job,))['stage'], 'Complete · 1 new, 1 replaced, 1 edited by hand kept')
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'done')

    def test_replace_manual_also_replaces_hand_edited_lines_but_keeps_history(self):
        job, _clipped = self.run_range(replace_manual=True)
        current = self.current()
        self.assertEqual(sorted(row['text'].strip() for row in current.values()), ['After', 'Before', 'Hello.', 'there'])
        self.assertIsNotNone(db.one("SELECT superseded_by FROM transcript_segments WHERE id='hand'")['superseded_by'])
        self.assertEqual(db.one('SELECT stage FROM jobs WHERE id=?', (job,))['stage'], 'Complete · 2 new, 2 replaced')

    def test_range_restarts_and_preserves_a_revision_made_while_running(self):
        job_id = worker.enqueue('e', 'transcribe', {'range': {'start_time': 4, 'end_time': 8}, 'replace_manual': True})
        initial = worker.claim_job(('transcribe',))
        words = [{'start_time': 4.5, 'end_time': 5, 'text': 'New word', 'speaker': 'SPEAKER_00'},
                 {'start_time': 6.5, 'end_time': 7, 'text': 'Overwrite', 'speaker': 'SPEAKER_00'}]

        def edit_during_generation(*_args):
            with db.connect() as conn:
                conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,origin,parent_id)
                    VALUES ('new-manual','e',6.5,7,'Keep the new edit','manual_edit','hand')""")
                conn.execute("UPDATE transcript_segments SET superseded_by='new-manual' WHERE id='hand'")
            return words

        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), \
             patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), \
             patch.object(audio_worker, 'load_asr', return_value=self.model), \
             patch.object(audio_worker, 'transcribe_window', side_effect=edit_during_generation):
            audio_worker.process(*initial)
        worker.release_lease(*initial)
        saved = db.one('SELECT status,stage,attempt_count FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(saved, {'status': 'queued', 'stage': 'Inputs changed, restarting', 'attempt_count': 0})
        self.assertIsNone(worker.claim_job(('transcribe',)))
        with db.connect() as conn:
            conn.execute('UPDATE jobs SET not_before=0 WHERE id=?', (job_id,))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), \
             patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output), \
             patch.object(audio_worker, 'load_asr', return_value=self.model), \
             patch.object(audio_worker, 'transcribe_window', return_value=words):
            audio_worker.process(*worker.claim_job(('transcribe',)))
        current = self.current()
        self.assertEqual(current['new-manual']['text'], 'Keep the new edit')
        self.assertEqual({row['text'] for row in current.values() if row['job_id'] == job_id}, {'New word'})

    def test_too_short_range_fails_without_changing_the_transcript(self):
        job = worker.enqueue('e', 'transcribe', {'range': {'start_time': 4, 'end_time': 4.2}})
        claimed = worker.claim_job(('transcribe',))
        with patch.object(audio_worker.media, 'episode_dir', return_value=self.path), patch.object(audio_worker, 'clip_audio'), patch.object(audio_worker, 'diarize', return_value=self.output):
            audio_worker.process(*claimed)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job,))['status'], 'failed')
        self.assertEqual(len(self.current()), 4)
