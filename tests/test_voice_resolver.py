"""Autopilot M5: unknown voices given to characters when two independent signals agree (voice_resolver.py)."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_log, db, media, speaker_attribution, voice_resolver as vr, worker
from app.main import app


def line(line_id, start, end, text='あ', label='SPEAKER_04', character=None, evidence=None):
    return {'id': line_id, 'start_time': start, 'end_time': end, 'text': text, 'speaker_label': label, 'character_context_id': character,
            'origin': 'model', 'speaker_source': None, 'evidence_json': json.dumps(evidence) if evidence else None, 'words_json': None}


class PureTest(unittest.TestCase):
    def test_phrases_join_close_open_lines_of_one_label_only(self):
        lines = [line('a', 0.0, 0.3, 'ま'), line('b', 0.4, 0.8, 'い'), line('c', 1.6, 2.0, 'さ'),       # gap 0.8 s: a new phrase
                 line('d', 2.1, 2.4, 'x', character='mai'), line('e', 2.5, 2.9, 'y'),                  # an assigned line between
                 line('f', 3.0, 3.4, 'z', label='SPEAKER_06')]                                         # another label
        found = vr.phrases(lines)
        self.assertEqual([item['line_ids'] for item in found], [['a', 'b'], ['c'], ['e'], ['f']])
        self.assertEqual(found[0]['text'], 'まい')
        self.assertEqual(found[1]['pad'], [0.8, 2.1])  # grown into the silence, never into the neighbouring lines
        self.assertEqual(vr.clip_span(found[0]), [0.0, 0.8])
        self.assertTrue(vr.phrases(lines, [(1.5, 2.2)])[1]['in_sequence'])

    def test_two_families_must_agree_and_nobody_may_disagree(self):
        self.assertEqual(vr.agree({'voice': 'mai', 'picture': 'mai'})['action'], 'apply')
        one = vr.agree({'voice': 'mai', 'label': 'mai'})  # both are sound: diarization clusters the same embeddings
        self.assertEqual((one['action'], one['why']), ('ask', 'only one kind of signal (label, voice)'))
        split = vr.agree({'voice': 'mai', 'picture': 'sakuta', 'name_tag': 'mai'})
        self.assertEqual((split['action'], split['candidates']), ('ask', ['mai', 'sakuta']))
        self.assertEqual(vr.agree({'voice': 'mai', 'picture': 'mai'}, {'mai'})['action'], 'ask')  # the line calls Mai by name
        self.assertEqual(vr.agree({'voice': None, 'picture': None})['why'], 'no signal names anyone')

    def test_who_the_picture_shows_speaking(self):
        keyframes = {'segments': [{'start': 0, 'end': 0.5, 'characters': [{'character_context_id': 'mai', 'speaking': 'yes'}]},
                                  {'start': 0.5, 'end': 1.0, 'characters': [{'character_context_id': 'sakuta', 'speaking': 'yes'},
                                                                           {'character_context_id': 'mai', 'speaking': 'no'}]}]}
        seen = vr.speaking('Keyframe analysis', json.dumps(keyframes))
        cuts = [{'id': 'c', 'start_time': 10.0, 'end_time': 11.0, 'no_characters': 0}]
        phrase = {'start': 10.1, 'end': 10.4}
        self.assertEqual(vr.picture_pick(phrase, cuts, {'c': seen}), 'mai')
        self.assertIsNone(vr.picture_pick({'start': 10.3, 'end': 10.8}, cuts, {'c': seen}))  # two speakers over the phrase
        direction = {'direction': {'characters': [{'character_context_id': 'sakuta', 'speaking': 'yes'}, {'character_context_id': 'mai', 'speaking': 'no'}]}}
        self.assertEqual(vr.picture_pick(phrase, cuts, {'c': vr.speaking('Video analysis', direction)}), 'sakuta')
        self.assertIsNone(vr.picture_pick(phrase, [{**cuts[0], 'no_characters': 1}], {'c': seen}))

    def test_addressed_names_and_tags(self):
        names = vr.name_index([{'spelling': '麻衣', 'character_context_id': 'mai', 'part': 'given'},
                               {'spelling': '梓川', 'character_context_id': 'sakuta', 'part': 'family'}], [{'id': 'mai', 'title': 'Mai Sakurajima'}])
        self.assertEqual(vr.addressed('麻衣さん、待って', names), {'mai'})
        self.assertEqual(vr.addressed('梓川くん', names), set())  # a family name is shared by relatives
        phrase = {'start': 1.0, 'end': 2.0}
        self.assertEqual(vr.tag_pick(phrase, [{'start_time': 1.0, 'end_time': 2.0, 'text': '(麻衣)そうね'}], names), 'mai')

    def test_agent_speakers_are_never_re_attributed(self):
        self.assertFalse(speaker_attribution.model_line({'origin': 'model', 'speaker_source': 'agent', 'words_json': '[]x'}))
        self.assertIn('agent', speaker_attribution.DECIDED)


class FakeScorer:
    """Voice scores by label, as if the label's audio were embedded."""
    usable = True
    by_label = {}

    def __init__(self, conn, data):
        pass

    def phrases(self, items, check=None):
        return {item['key']: self.by_label.get(item['label'], {}) for item in items}

    def labels(self, items):
        return {item['label']: self.by_label[item['label']] for item in items if item['label'] in self.by_label}


class JobTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio)
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}',40,1)""")
            for character, title in (('mai', 'Mai Sakurajima'), ('sakuta', 'Sakuta Azusagawa'), ('tomoe', 'Tomoe Koga')):
                conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES (?,?,?,?,'')", (character, 'p', 'character', title))
                conn.execute("""INSERT INTO character_voice_samples(id,character_context_id,episode_id,start_time,end_time,embedding_json,model_id)
                    VALUES (?,?,'e',0,1,'[1]','pyannote/speaker-diarization-community-1')""", (f'v-{character}', character))
            cuts = [('c1', 0, 0, 10), ('c2', 1, 10, 20), ('c3', 2, 20, 30), ('c4', 3, 30, 40)]
            for cut_id, index, start, end in cuts:
                conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES (?,?,?,?,?)', (cut_id, 'e', index, start, end))
            mai = json.dumps({'direction': {'characters': [{'character_context_id': 'mai', 'speaking': 'yes'}]}})
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('va1','c1','video','Video analysis',?,'model')", (mai,))
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('va2','c2','video','Video analysis',?,'model')", (mai,))
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('va3','c3','video','Video analysis',?,'model')", (mai,))
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('cs2','c2','text','Cut summary','x','model')")
            rows = [('a1', 1.0, 1.5, 'SPEAKER_04'), ('a2', 12.0, 12.6, 'SPEAKER_04'),   # Mai's voice, Mai speaks on screen: applied
                    ('b1', 22.0, 23.0, 'SPEAKER_06'), ('b2', 25.0, 26.0, 'SPEAKER_06'),  # Sakuta's voice, but the picture says Mai: Claude
                    ('n1', 32.0, 33.0, 'SPEAKER_07'),                                    # nobody's voice, no picture: maybe new (person)
                    ('k1', 34.0, 35.0, 'SPEAKER_01')]
            for line_id, start, end, label in rows:
                conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin)
                    VALUES (?,'e',?,?,'ね',?,'model')""", (line_id, start, end, label))
            for index in range(6):  # SPEAKER_01 is already Tomoe on six lines
                conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin)
                    VALUES (?,'e',?,?,'うん','SPEAKER_01','tomoe','manual_edit')""", (f't{index}', 36 + index * 0.5, 36.3 + index * 0.5))
        FakeScorer.by_label = {'SPEAKER_04': {'mai': 0.62, 'sakuta': 0.31, 'tomoe': 0.2}, 'SPEAKER_06': {'sakuta': 0.58, 'mai': 0.3},
                               'SPEAKER_07': {'mai': 0.12, 'sakuta': 0.1}, 'SPEAKER_01': {'tomoe': 0.55, 'mai': 0.2}}
        self.client = TestClient(app)
        self.patch = mock.patch.object(vr, 'AudioScorer', FakeScorer)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def speakers(self):
        with db.connect() as conn:
            return {row['text_id']: (row['character_context_id'], row['speaker_source'], row['origin']) for row in conn.execute(
                """SELECT coalesce((WITH RECURSIVE up(id,parent_id) AS (SELECT id,parent_id FROM transcript_segments t2 WHERE t2.id=t.id
                   UNION ALL SELECT s.id,s.parent_id FROM transcript_segments s JOIN up ON s.id=up.parent_id) SELECT id FROM up WHERE parent_id IS NULL),t.id) AS text_id,
                   t.character_context_id,t.speaker_source,t.origin FROM transcript_segments t WHERE t.episode_id='e' AND t.superseded_by IS NULL AND t.deleted=0""")}

    def run_job(self, body=None):
        job_id = self.client.post('/api/episodes/e/voice-decisions', json=body or {'attribution': False}).json()['job_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running',lease_owner='test',lease_until=? WHERE id=?", (time.time() + 60, job_id))
        vr.process(job_id, 'test')
        with db.connect() as conn:
            job = conn.execute('SELECT status,error,payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()
        self.assertEqual(job['status'], 'done', job['error'])
        return json.loads(job['payload_json'])['result']

    def test_apply_where_two_signals_agree_and_ask_the_rest(self):
        result = self.run_job()
        self.assertEqual(result['voice_scores'], 'computed')
        speakers = self.speakers()
        self.assertEqual(speakers['a1'], ('mai', 'agent', 'model'))  # the source's origin stays: the words are still the model's
        self.assertEqual(speakers['a2'][0], 'mai')
        self.assertIsNone(speakers['b1'][0])
        self.assertIsNone(speakers['n1'][0])
        self.assertEqual(speakers['k1'][0], None)  # voice + link are both sound: one family is not enough
        questions = {item['group_key']: item for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions']}
        self.assertEqual(questions['voice:SPEAKER_06']['reviewer'], 'claude')
        self.assertIn('disagree', questions['voice:SPEAKER_06']['question'])
        self.assertEqual(questions['voice:SPEAKER_07']['reviewer'], 'human')
        self.assertTrue(questions['voice:SPEAKER_07']['evidence']['new_voice'])
        with db.connect() as conn:
            stale = conn.execute("SELECT stale_reason FROM analysis_results WHERE id='cs2'").fetchone()[0]
            decision = conn.execute("SELECT * FROM agent_decisions WHERE kind='voices'").fetchone()
        self.assertEqual(stale, 'Speaker changed')
        self.assertEqual((decision['tier'], decision['cut_id']), ('local', 'c1'))
        self.assertIn('picture + voice', decision['summary'])

        # Claude settles SPEAKER_06 after looking at the frames; the person's question about SPEAKER_07 is refused to Claude.
        answered = self.client.post(f"/api/escalations/{questions['voice:SPEAKER_06']['id']}/voice",
                                    json={'choice': 'character', 'character_context_id': 'sakuta', 'tier': 'claude', 'note': 'his mouth moves'}).json()
        self.assertEqual((answered['status'], answered['lines']), ('resolved', 2))
        self.assertEqual(self.speakers()['b2'], ('sakuta', 'agent', 'model'))
        refused = self.client.post(f"/api/escalations/{questions['voice:SPEAKER_07']['id']}/voice", json={'choice': 'unsure', 'tier': 'claude'})
        self.assertEqual(refused.status_code, 409)

        # The person settles SPEAKER_01 in the app itself: its question closes on its own.
        self.client.post('/api/episodes/e/transcript/assign-voice', json={'speaker_label': 'SPEAKER_01', 'character_context_id': 'tomoe'})
        open_keys = {item['group_key'] for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions']}
        self.assertNotIn('voice:SPEAKER_01', open_keys)
        self.assertEqual(self.client.post(f"/api/escalations/{questions['voice:SPEAKER_07']['id']}/voice", json={'choice': 'leave'}).json()['status'], 'left_unknown')
        self.run_job()  # a voice the person left unknown is never asked again
        open_keys = {item['group_key'] for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions']}
        self.assertNotIn('voice:SPEAKER_07', open_keys)

    def test_undo_and_a_person_override(self):
        self.run_job()
        with db.connect() as conn:
            decision = conn.execute("SELECT id FROM agent_decisions WHERE kind='voices'").fetchone()[0]
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').json(), {'undone': True})
        self.assertEqual(self.speakers()['a1'][:2], (None, None))
        self.run_job()
        with db.connect() as conn:
            decision = conn.execute("SELECT id FROM agent_decisions WHERE kind='voices' AND undone_at IS NULL").fetchone()[0]
            current = conn.execute("SELECT * FROM transcript_segments WHERE superseded_by IS NULL AND start_time=1.0").fetchone()
        response = self.client.put(f"/api/transcript-segments/{current['id']}", json={'start_time': 1.0, 'end_time': 1.5, 'text': 'ね',
                                                                                     'speaker_label': 'SPEAKER_04', 'character_context_id': 'sakuta'})
        self.assertEqual(response.status_code, 200, response.text)
        with db.connect() as conn:
            self.assertIsNotNone(conn.execute('SELECT overridden_at FROM agent_decisions WHERE id=?', (decision,)).fetchone()[0])
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').status_code, 409)

    def test_ask_mode_off_sequences_and_queue_order(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO sequences(id,episode_id,title,first_cut_id,last_cut_id,cuts_json,start_time,end_time) VALUES ('s','e','S','c1','c1','[]',0,10)")
        result = self.run_job()
        self.assertEqual(result['left'], {'in a sequence': 1})
        self.assertIsNone(self.speakers()['a1'][0])  # inside the sequence: the user's dialogue
        self.assertEqual(self.speakers()['a2'][0], 'mai')  # the label's other line: voice + picture
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'voices': 'ask'}}})
        with db.connect() as conn:
            conn.execute('DELETE FROM sequences')
        result = self.run_job()
        self.assertEqual(result.get('lines_applied', 0), 0)
        self.assertGreaterEqual(result['asked_person'], 1)
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'voices': 'off'}}})
        self.assertIsNone(self.client.post('/api/episodes/e/voice-decisions', json={}).json()['job_id'])
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'voices': 'apply'}}})
        with db.connect() as conn:
            pending = worker.insert_job(conn, 'e', 'transcribe', {'language': 'Japanese'})
            job_id = vr.queue(conn, 'e')
            payload = json.loads(conn.execute('SELECT payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()[0])
        self.assertIn(pending, payload['depends_on'])
        self.assertIs(agent_log.UNDO_KINDS['voices'], vr.undo)

    def test_without_samples_the_saved_line_scores_are_used(self):
        with db.connect() as conn:
            conn.execute('DELETE FROM character_voice_samples')
            conn.execute("UPDATE transcript_segments SET evidence_json=? WHERE id='a1'", (json.dumps({'evidence': [['mai', 0.5], ['sakuta', 0.3]]}),))
        result = self.run_job()
        self.assertEqual(result['voice_scores'], 'none')
        self.assertEqual(self.speakers()['a1'][0], 'mai')   # stored voice + picture
        self.assertIsNone(self.speakers()['a2'][0])          # no score saved on it: one signal only


if __name__ == '__main__':
    unittest.main()
