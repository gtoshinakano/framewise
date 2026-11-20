import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import db, media, transcript_parts
from app.main import app


class TranscriptCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'db.sqlite'
        db.init_db()
        self.words = [[2.592, 2.832, 'これ'], [2.832, 2.912, 'は'], [2.912, 3.312, '青春'], [3.312, 3.632, '気象'],
                      [3.632, 3.872, '校'], [3.872, 4.112, '軍'], [4.112, 4.352, 'ではなく']]
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Project'),('other','Other')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,duration,has_audio,settings_json) VALUES('e','p','Episode','x.mp4','.mp4',20,1,'{}'),('f','p','Two','y.mp4','.mp4',20,1,'{}')")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES('rio','p','character','Rio'),('sakuta','p','character','Sakuta'),('x','other','character','Elsewhere')")
            conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,words_json,speaker_source,confidence,evidence_json)
                            VALUES('s1','e',2.592,4.352,'これは青春気象校軍ではなく','SPEAKER_05','rio','model',?,'voice','high','{"speaker": "rio"}')""",
                         (json.dumps(self.words, ensure_ascii=False),))
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin,speaker_source,confidence) VALUES('s2','e',5,6,'料理番組か','SPEAKER_05','rio','model','voice','high'),('s3','f',1,2,'other episode','',NULL,'model',NULL,NULL)")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def post(self, corrections, episode='e'):
        return self.client.post(f'/api/episodes/{episode}/transcript/corrections',
                                json={'source': {'file': 'ep02.ja.srt', 'offset': 0.4}, 'corrections': corrections})

    def history(self):
        return {row['id']: row for row in self.client.get('/api/episodes/e/transcript?history=true').json()}

    def test_text_fix_keeps_times_words_and_speaker(self):
        response = self.post([{'segment_id': 's1', 'text': 'これは思春期症候群ではなく', 'cues': [1], 'srt_text': 'これは思春期症候群ではなく', 'note': 'mishearing'}])
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['skipped'], [])
        new = self.history()[result['applied'][0]['id']]
        self.assertEqual((new['origin'], new['parent_id'], new['text']), ('srt_edit', 's1', 'これは思春期症候群ではなく'))
        self.assertEqual((new['start_time'], new['end_time']), (2.592, 4.352))
        self.assertEqual(json.loads(new['words_json']), self.words)
        self.assertEqual((new['character_context_id'], new['speaker_source'], new['confidence']), ('rio', 'voice', 'high'))
        evidence = json.loads(new['evidence_json'])
        self.assertEqual(evidence['speaker'], 'rio')
        self.assertEqual(evidence['srt'], {'file': 'ep02.ja.srt', 'offset': 0.4, 'cues': [1], 'srt_text': 'これは思春期症候群ではなく', 'note': 'mishearing'})
        self.assertEqual(self.history()['s1']['superseded_by'], new['id'])
        # The respelled name lands on the time slots of the words it replaced.
        parts = transcript_parts.parts(new)
        self.assertEqual(''.join(part['text'] for part in parts), 'これは思春期症候群ではなく')
        self.assertEqual(parts[0]['start_time'], 2.592)
        self.assertEqual(parts[-1]['end_time'], 4.352)

    def test_speaker_change_is_manual(self):
        result = self.post([{'segment_id': 's2', 'character_context_id': 'sakuta', 'note': '(咲太) tag'}]).json()
        new = self.history()[result['applied'][0]['id']]
        self.assertEqual((new['text'], new['character_context_id'], new['speaker_source'], new['confidence']), ('料理番組か', 'sakuta', 'manual', None))

    def test_skips_stale_foreign_and_unchanged_lines_and_applies_the_rest(self):
        self.client.put('/api/transcript-segments/s2', json={'start_time': 5, 'end_time': 6, 'text': '料理番組か。', 'speaker_label': 'SPEAKER_05', 'character_context_id': 'rio'})
        result = self.post([{'segment_id': 's2', 'text': '料理番組か！'}, {'segment_id': 's3', 'text': 'x'},
                            {'segment_id': 's1', 'text': 'これは青春気象校軍ではなく'}, {'segment_id': 's1', 'character_context_id': 'x'},
                            {'segment_id': 'nope', 'text': 'x'}]).json()
        self.assertEqual(result['applied'], [])
        reasons = [item['reason'] for item in result['skipped']]
        self.assertEqual(reasons, ['the line was edited or deleted since the plan was made', 'not a line of this episode',
                                   'nothing to change', 'listed twice', 'not a line of this episode'])
        result = self.post([{'segment_id': 's1', 'character_context_id': 'x'}]).json()
        self.assertEqual(result['skipped'][0]['reason'], 'not a character of this project')

    def test_body_is_validated(self):
        self.assertEqual(self.post([]).status_code, 422)
        self.assertEqual(self.post([{'segment_id': 's1', 'text': 'a', 'extra': 1}]).status_code, 422)
        self.assertEqual(self.post([{'segment_id': 's1', 'text': 'a'}], episode='missing').status_code, 404)


if __name__ == '__main__':
    unittest.main()
