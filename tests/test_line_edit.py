import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import db, media, waveform_range
from app.main import app


class LineEditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'db.sqlite'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Project'),('other','Other')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,duration,has_audio,settings_json) VALUES('e','p','Episode','x.mp4','.mp4',20,1,'{}')")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES('c','p','character','Alice'),('x','other','character','Elsewhere')")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES('a','e',0,0,5),('b','e',1,5,10)")
            words = json.dumps([[4.0, 4.5, 'Hello'], [4.5, 5.2, 'there'], [5.2, 6.0, 'friend']])
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,words_json,speaker_source,confidence) VALUES('s1','e',4,6,'Hello there friend','SPEAKER_01','c','model',?, 'voice','high')", (words,))
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES('s2','e',6,7,'How are you','SPEAKER_01','model'),('s3','e',8,9,'Fine','SPEAKER_02','model')")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def current(self):
        return {row['id']: row for row in self.client.get('/api/episodes/e/transcript').json()}

    def merge(self, ids, **extra):
        body = {'segment_ids': ids, 'text': 'Hello there friend How are you', 'speaker_label': 'SPEAKER_01', 'character_context_id': None, **extra}
        return self.client.post('/api/episodes/e/transcript/merge', json=body)

    def test_merge_joins_lines_into_one_revision(self):
        response = self.merge(['s2', 's1'])
        self.assertEqual(response.status_code, 200, response.text)
        merged = response.json()
        self.assertEqual((merged['start_time'], merged['end_time']), (4, 7))
        self.assertEqual(merged['origin'], 'manual_edit')
        self.assertEqual(merged['parent_id'], 's1')
        self.assertEqual(set(self.current()), {merged['id'], 's3'})
        history = {row['id']: row for row in self.client.get('/api/episodes/e/transcript?history=true').json()}
        self.assertEqual([history['s1']['superseded_by'], history['s2']['superseded_by']], [merged['id']] * 2)
        # s2 has no words, so the merged line has none, and the speaker changed (s1 had a character).
        self.assertIsNone(merged['words_json'])
        self.assertEqual(merged['speaker_source'], 'manual')

    def test_merge_keeps_words_only_when_the_text_is_unchanged(self):
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET words_json=?, speaker_label='SPEAKER_01', character_context_id='c' WHERE id='s2'", (json.dumps([[6.0, 6.4, 'How'], [6.4, 6.6, 'are'], [6.6, 7.0, 'you']]),))
        kept = self.merge(['s1', 's2'], character_context_id='c').json()
        self.assertEqual(len(json.loads(kept['words_json'])), 6)
        self.assertEqual(kept['speaker_source'], 'voice')
        self.assertEqual(kept['confidence'], 'high')
        again = self.merge([kept['id'], 's3'], text='Hello there friend How are you, fine', character_context_id='c').json()
        self.assertIsNone(again['words_json'])

    def test_merge_refuses_stale_foreign_or_bad_requests(self):
        self.assertEqual(self.merge(['s1']).status_code, 422)
        self.assertEqual(self.merge(['s1', 'missing']).status_code, 404)
        self.assertEqual(self.merge(['s1', 's2'], character_context_id='x').status_code, 400)
        self.assertEqual(self.merge(['s1', 's2'], extra=1).status_code, 422)
        self.assertEqual(self.merge(['s1', 's2'], text='  ').status_code, 400)
        self.assertEqual(self.merge(['s1', 's2']).status_code, 200)
        self.assertEqual(self.merge(['s1', 's3']).status_code, 409)
        self.client.delete('/api/transcript-segments/s3')
        self.assertEqual(self.client.post('/api/episodes/none/transcript/merge', json={'segment_ids': ['s1', 's3'], 'text': 'x'}).status_code, 404)

    def test_split_uses_word_times_when_the_text_is_unchanged(self):
        response = self.client.post('/api/transcript-segments/s1/split', json={'at': 5.2, 'left': {'text': 'Hello there'}, 'right': {'text': 'friend'}})
        self.assertEqual(response.status_code, 200, response.text)
        left, right = response.json()
        self.assertEqual((left['start_time'], left['end_time'], right['start_time'], right['end_time']), (4, 5.2, 5.2, 6))
        self.assertEqual([word[2] for word in json.loads(left['words_json'])], ['Hello', 'there'])
        self.assertEqual([word[2] for word in json.loads(right['words_json'])], ['friend'])
        self.assertEqual((left['character_context_id'], right['character_context_id']), ('c', 'c'))
        self.assertEqual(left['parent_id'], 's1')
        self.assertEqual(right['parent_id'], 's1')
        self.assertEqual(set(self.current()), {left['id'], right['id'], 's2', 's3'})

    def test_split_with_edited_text_drops_the_alignment(self):
        left, right = self.client.post('/api/transcript-segments/s1/split', json={'at': 5, 'left': {'text': 'Hi there'}, 'right': {'text': 'friend'}}).json()
        self.assertIsNone(left['words_json'])
        self.assertIsNone(right['words_json'])

    def test_split_validation_and_stale_lines(self):
        url = '/api/transcript-segments/s2/split'
        part = {'text': 'x'}
        self.assertEqual(self.client.post(url, json={'at': 6, 'left': part, 'right': part}).status_code, 400)
        self.assertEqual(self.client.post(url, json={'at': 7, 'left': part, 'right': part}).status_code, 400)
        self.assertEqual(self.client.post(url, json={'at': 6.5, 'left': {'text': '  '}, 'right': part}).status_code, 400)
        self.assertEqual(self.client.post(url, json={'at': 6.5, 'left': {'text': ''}, 'right': part}).status_code, 422)
        self.assertEqual(self.client.post('/api/transcript-segments/none/split', json={'at': 6.5, 'left': part, 'right': part}).status_code, 404)
        self.assertEqual(self.client.post(url, json={'at': 6.5, 'left': part, 'right': part}).status_code, 200)
        self.assertEqual(self.client.post(url, json={'at': 6.5, 'left': part, 'right': part}).status_code, 409)

    def test_edits_make_the_results_of_both_neighbouring_cuts_stale(self):
        with db.connect() as conn:
            conn.executemany("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES(?,?, 'other','Cut summary','x','model')", [('ra', 'a'), ('rb', 'b')])
        self.assertEqual(self.merge(['s1', 's2']).status_code, 200)
        with db.connect() as conn:
            stale = {row['id']: row['stale_reason'] for row in conn.execute('SELECT id,stale_reason FROM analysis_results')}
        self.assertTrue(stale['ra'] and stale['rb'])


class WaveformRangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'audio.wav'
        samples = np.zeros(16000 * 4, dtype='<i2')
        samples[16000 * 2:16000 * 3] = 16384   # loud second 2..3
        with wave.open(str(self.path), 'wb') as target:
            target.setnchannels(1); target.setsampwidth(2); target.setframerate(16000)
            target.writeframes(samples.tobytes())

    def tearDown(self):
        self.tmp.cleanup()

    def test_peaks_follow_the_requested_slice(self):
        result = waveform_range.peaks(self.path, 1, 4, 6)
        self.assertEqual(len(result['peaks']), 6)
        self.assertEqual(result['peaks'][:2], [0.0, 0.0])
        self.assertEqual(result['peaks'][2:4], [0.5, 0.5])
        self.assertEqual(result['peaks'][4:], [0.0, 0.0])

    def test_range_past_the_end_is_silent_and_bad_ranges_are_refused(self):
        self.assertEqual(waveform_range.peaks(self.path, 2, 7, 5)['peaks'], [0.5, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(waveform_range.peaks(self.path.with_name('none.wav'), 0, 1, 4)['peaks'], [])
        for args in ((2, 1, 4), (-1, 1, 4), (0, 301, 4), (0, 1, 0), (0, 1, 2001)):
            with self.assertRaises(ValueError):
                waveform_range.peaks(self.path, *args)

    def test_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
            db.DATA_DIR = media.DATA_DIR = Path(tmp)
            db.DB_PATH = db.DATA_DIR / 'db.sqlite'
            try:
                db.init_db()
                with db.connect() as conn:
                    conn.execute("INSERT INTO projects(id,title) VALUES('p','P')")
                    conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,duration,has_audio,settings_json) VALUES('e','p','E','x.mp4','.mp4',4,1,'{}')")
                client = TestClient(app)
                self.assertEqual(client.get('/api/episodes/e/waveform-range?start=0&end=2&bins=10').json()['peaks'], [])
                self.assertEqual(client.get('/api/episodes/e/waveform-range?start=2&end=1').status_code, 400)
                self.assertEqual(client.get('/api/episodes/none/waveform-range?start=0&end=1').status_code, 404)
            finally:
                db.DATA_DIR, db.DB_PATH, media.DATA_DIR = old


if __name__ == '__main__':
    unittest.main()
