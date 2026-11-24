"""Autopilot M3: transcript lines across a hard cut, split at a pause or trimmed (line_splits.py)."""
import array
import json
import math
import tempfile
import time
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app import agent_log, db, line_splits, media, worker
from app.main import app

WINDOW = line_splits.WINDOW


def envelope(spans, until, start=0.0, loud=0.3):
    """A level per 50 ms window: ``loud`` inside the spans (voice), 0.001 elsewhere."""
    levels = []
    for index in range(int((until - start) / WINDOW)):
        moment = start + (index + 0.5) * WINDOW
        levels.append(loud if any(a <= moment < b for a, b in spans) else 0.001)
    return levels


def row(words, text=None, origin='model'):
    return {'id': 'l1', 'episode_id': 'e', 'start_time': words[0][0], 'end_time': words[-1][1], 'text': text or ''.join(word[2] for word in words),
            'words_json': json.dumps(words, ensure_ascii=False), 'origin': origin}


class PlanTest(unittest.TestCase):
    words = [[1.0, 1.6, 'あの人、'], [2.1, 2.5, 'どういう'], [2.5, 2.7, '人'], [2.7, 3.2, 'なんだろう。']]
    voice = [(1.0, 1.6), (2.1, 3.2)]

    def test_split_on_the_cut_when_the_cut_is_inside_the_pause(self):
        plan = line_splits.plan_line(row(self.words), 1.9, envelope(self.voice, 5), 0.0)
        self.assertEqual((plan['action'], plan['at'], plan['left'], plan['right']), ('split', 1.9, 'あの人、', 'どういう人なんだろう。'))

    def test_split_at_the_pause_edge_nearest_the_cut_and_never_beyond_reach(self):
        plan = line_splits.plan_line(row(self.words), 2.3, envelope(self.voice, 5), 0.0)
        self.assertEqual(plan['action'], 'split')
        self.assertAlmostEqual(plan['at'], 2.04, places=2)  # the pause ends at 2.1; the split sits PAD into the silence
        far = [[0.2, 0.6, 'ええ、'], [0.6, 3.0, 'ずっとそうだったんだけど'], [3.0, 3.4, '。']]
        plan = line_splits.plan_line(row(far), 2.9, envelope([(0.0, 0.1), (0.2, 3.4)], 5), 0.0)
        self.assertEqual(plan['action'], 'leave')

    def test_never_inside_a_word_or_before_an_auxiliary(self):
        words = [[1.0, 1.5, 'わかっ'], [2.0, 2.2, 'た'], [2.2, 2.6, 'のなら'], [2.6, 3.0, 'はいと言いなさい。']]
        plan = line_splits.plan_line(row(words), 1.8, envelope([(1.0, 1.5), (2.0, 3.0)], 5), 0.0)
        self.assertEqual(plan['action'], 'leave')
        self.assertIn('particle', plan['why'])

    def test_trim_a_silent_tail_or_head_by_the_audio(self):
        words = [[1.0, 1.4, 'おい、'], [1.4, 2.6, 'おい。']]   # the aligner stretched the last word over the silence
        plan = line_splits.plan_line(row(words), 2.0, envelope([(1.0, 1.8)], 5), 0.0)
        self.assertEqual(plan['action'], 'trim')
        self.assertLessEqual(plan['end'], 2.0)
        self.assertGreaterEqual(plan['end'], 1.8)
        words = [[1.0, 2.4, 'えっと'], [2.4, 3.0, 'ね。']]
        plan = line_splits.plan_line(row(words), 1.8, envelope([(2.1, 3.0)], 5), 0.0)
        self.assertEqual(plan['action'], 'trim')
        self.assertGreaterEqual(plan['start'], 1.8)

    def test_leave_without_word_timings_or_when_the_text_does_not_follow_them(self):
        bare = {**row(self.words), 'words_json': None}
        self.assertEqual(line_splits.plan_line(bare, 1.9, envelope(self.voice, 5), 0.0)['action'], 'leave')
        edited = row(self.words, text='まったく違う文章です')
        self.assertEqual(line_splits.plan_line(edited, 1.9, envelope(self.voice, 5), 0.0)['action'], 'leave')


def write_wav(path, spans, seconds, rate=16000):
    samples = array.array('h')
    for index in range(int(seconds * rate)):
        moment = index / rate
        samples.append(int(8000 * math.sin(2 * math.pi * 220 * moment)) if any(a <= moment < b for a, b in spans) else 0)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())


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
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}',8,1)""")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c1','e',0,0,1.9),('c2','e',1,1.9,4.2),('c3','e',2,4.2,8)")
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('r2','c2','text','Cut summary','x','model')")
            lines = [('a', 1.0, 3.2, [[1.0, 1.6, 'あの人、'], [2.1, 2.5, 'どういう'], [2.5, 2.7, '人'], [2.7, 3.2, 'なんだろう。']], 'model'),
                     ('b', 3.5, 4.8, [[3.5, 3.9, 'うん、'], [3.9, 4.8, 'そう。']], 'srt_edit'),
                     ('c', 5.0, 5.8, [[5.0, 5.8, 'ねえ']], 'model')]
            for line_id, start, end, words, origin in lines:
                conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin,words_json,speaker_source)
                    VALUES (?,?,?,?,?,'SPEAKER_00',?,?,'voice')""", (line_id, 'e', start, end, ''.join(word[2] for word in words), origin, json.dumps(words, ensure_ascii=False)))
        folder = media.episode_dir('p', 'e')
        folder.mkdir(parents=True, exist_ok=True)
        write_wav(folder / 'audio.wav', [(1.0, 1.6), (2.1, 3.2), (3.5, 4.0), (5.0, 5.8)], 8)
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def current(self):
        with db.connect() as conn:
            return [(row['start_time'], row['end_time'], row['text'], row['origin']) for row in conn.execute(
                "SELECT * FROM transcript_segments WHERE episode_id='e' AND superseded_by IS NULL AND deleted=0 ORDER BY start_time")]

    def run_job(self, body=None):
        queued = self.client.post('/api/episodes/e/line-splits', json=body or {}).json()
        job_id = queued['job_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running',lease_owner='test',lease_until=? WHERE id=?", (time.time() + 60, job_id))
        line_splits.process(job_id, 'test')
        with db.connect() as conn:
            job = conn.execute('SELECT status,error,payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()
        self.assertEqual(job['status'], 'done', job['error'])
        return json.loads(job['payload_json'])['result']

    def test_preview_then_split_and_trim_keeping_origins_and_undo(self):
        preview = self.client.get('/api/episodes/e/line-splits').json()
        self.assertEqual(preview['counts'], {'split': 1, 'trim': 1})
        result = self.run_job()
        self.assertEqual((result['split'], result['trim']), (1, 1))
        lines = self.current()
        self.assertEqual([(text, origin) for _, _, text, origin in lines],
                         [('あの人、', 'model'), ('どういう人なんだろう。', 'model'), ('うん、そう。', 'srt_edit'), ('ねえ', 'model')])
        self.assertEqual(lines[1][0], 1.9)
        self.assertLessEqual(lines[2][1], 4.2)  # the srt line's silent tail no longer crosses the cut, and keeps every character
        with db.connect() as conn:
            decisions = conn.execute("SELECT id,tier FROM agent_decisions WHERE kind='line_splits' ORDER BY rowid").fetchall()
            stale = conn.execute("SELECT stale_reason FROM analysis_results WHERE id='r2'").fetchone()[0]
            trimmed = conn.execute("SELECT words_json FROM transcript_segments WHERE text='うん、そう。' AND superseded_by IS NULL").fetchone()[0]
        self.assertEqual([row['tier'] for row in decisions], ['rules', 'rules'])
        self.assertIsNone(stale)  # automatic audio work marks nothing stale (HARNESS.md D37)
        self.assertIsNone(trimmed)  # the stretched word no longer fits inside the new edges, so its times are dropped
        self.assertEqual(self.client.get('/api/episodes/e/line-splits').json()['counts'], {})
        self.assertEqual(self.client.get('/api/episodes/e/escalations').json()['items'], [])
        undone = self.client.post(f"/api/agent-decisions/{decisions[0]['id']}/undo").json()
        self.assertEqual(undone, {'undone': True})
        self.assertEqual(self.current()[0][:3], (1.0, 3.2, 'あの人、どういう人なんだろう。'))
        with db.connect() as conn:  # the person's undo does
            self.assertTrue(conn.execute("SELECT stale_reason FROM analysis_results WHERE id='r2'").fetchone()[0])

    def test_a_person_editing_a_piece_counts_as_an_override_and_blocks_undo(self):
        self.run_job()
        with db.connect() as conn:
            piece = conn.execute("SELECT * FROM transcript_segments WHERE text='どういう人なんだろう。' AND superseded_by IS NULL").fetchone()
            decision = conn.execute("SELECT id FROM agent_decisions WHERE kind='line_splits' AND summary LIKE 'Split%'").fetchone()[0]
        response = self.client.put(f"/api/transcript-segments/{piece['id']}", json={'start_time': 2.0, 'end_time': 3.2, 'text': piece['text'],
                                                                                   'speaker_label': 'SPEAKER_00', 'character_context_id': None})
        self.assertEqual(response.status_code, 200, response.text)
        with db.connect() as conn:
            self.assertIsNotNone(conn.execute('SELECT overridden_at FROM agent_decisions WHERE id=?', (decision,)).fetchone()[0])
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').status_code, 409)

    def test_the_persons_last_word_is_never_split_again(self):
        self.run_job()
        with db.connect() as conn:
            decision = conn.execute("SELECT id FROM agent_decisions WHERE kind='line_splits' AND summary LIKE 'Split%'").fetchone()[0]
            conn.execute("UPDATE transcript_segments SET origin='manual_edit' WHERE id='c'")
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').json(), {'undone': True})
        self.assertEqual(self.client.get('/api/episodes/e/line-splits').json()['counts'], {'by_person': 1})  # the undone split stays undone
        result = self.run_job()
        self.assertEqual((result['split'], result['by_person']), (0, 1))
        self.assertEqual(self.current()[0][2], 'あの人、どういう人なんだろう。')

    def test_a_line_asked_about_before_d37_stays_as_the_person_left_it(self):
        with db.connect() as conn:
            conn.execute("""INSERT INTO agent_escalations(id,episode_id,kind,group_key,question,status)
                VALUES ('old','e','line_split','line:a','?','resolved')""")
        result = self.run_job()
        self.assertEqual((result['split'], result['by_person']), (0, 1))

    def test_sequences_and_off_and_never_asks(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO sequences(id,episode_id,title,first_cut_id,last_cut_id,cuts_json,start_time,end_time) VALUES ('s','e','S','c1','c2','[]',0,4.2)")
        result = self.run_job()
        self.assertEqual((result['split'], result['trim'], result['in_sequence']), (0, 0, 2))  # both crossing lines overlap the sequence
        saved = self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'line_splits': 'ask'}}}).json()
        self.assertEqual(saved['profile']['modes']['line_splits'], 'apply')  # the splitter never asks
        with db.connect() as conn:
            conn.execute("DELETE FROM sequences")
        result = self.run_job()
        self.assertEqual((result['split'], result['trim']), (1, 1))
        self.assertNotIn('asked', result)
        self.assertEqual(self.client.get('/api/episodes/e/escalations').json()['items'], [])
        self.assertEqual(self.client.post('/api/escalations/x/line-split', json={'answer': 'leave'}).status_code, 404)  # no answer route
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'line_splits': 'off'}}})
        self.assertIsNone(self.client.post('/api/episodes/e/line-splits', json={}).json()['job_id'])

    def test_queue_waits_for_transcription_and_missed_cuts(self):
        with db.connect() as conn:
            pending = worker.insert_job(conn, 'e', 'transcribe', {'language': 'Japanese'})
            job_id = line_splits.queue(conn, 'e')
            payload = json.loads(conn.execute('SELECT payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()[0])
        self.assertIn(pending, payload['depends_on'])
        self.assertIs(agent_log.UNDO_KINDS['line_splits'], line_splits.undo)


if __name__ == '__main__':
    unittest.main()
