"""Run Multiple (batch_runs.py, HARNESS.md D37): server-side stage runs, step by step, with what each step found."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import agent_log, batch_runs, db, media, stage_targets
from app.main import app
from tests.test_stage_targets import result


class BatchRunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        self.client = TestClient(app)
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title,synopsis) VALUES ('p','Film','')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,width,height,has_audio)
                VALUES ('e','p','Episode','source.mp4','.mp4','ready','{}',8,1280,720,1)""")
        proxy = media.episode_dir('p', 'e') / 'proxy.mp4'
        proxy.parent.mkdir(parents=True, exist_ok=True)
        proxy.write_bytes(b'video')

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    # -- helpers ----------------------------------------------------------------------------------------------------

    def start(self, stage, options=None):
        with db.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            return batch_runs.start(conn, 'e', stage, options)

    def view(self, run_id):
        with db.connect() as conn:
            return batch_runs.view(conn, conn.execute('SELECT * FROM batch_runs WHERE id=?', (run_id,)).fetchone())

    def states(self, run_id):
        return {step['step']: step['state'] for step in self.view(run_id)['steps']}

    def jobs(self, pipeline, statuses=('queued', 'running')):
        marks = ','.join('?' for _ in statuses)
        return db.all_rows(f'SELECT id,payload_json FROM jobs WHERE pipeline=? AND status IN ({marks}) ORDER BY rowid', (pipeline, *statuses))

    def finish(self, pipeline, status='done', result=None):
        """Every waiting job of ``pipeline`` ends with ``status`` (and ``result`` in its payload, as the handlers save it)."""
        with db.connect() as conn:
            for job in conn.execute("SELECT id,payload_json FROM jobs WHERE pipeline=? AND status IN ('queued','running')", (pipeline,)).fetchall():
                payload = json.loads(job['payload_json'] or '{}')
                if result is not None:
                    payload['result'] = result
                conn.execute('UPDATE jobs SET status=?,payload_json=?,error=? WHERE id=?',
                             (status, json.dumps(payload), 'boom' if status == 'failed' else None, job['id']))

    def cuts(self, *spans, no_characters=1):
        with db.connect() as conn:
            for index, (start, end) in enumerate(spans):
                conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,no_characters) VALUES (?,?,?,?,?,?)',
                             (f'c{index + 1}', 'e', index, start, end, no_characters))

    # -- Prepare ----------------------------------------------------------------------------------------------------

    def test_prepare_runs_its_steps_in_order_once_and_asks_about_voices_only_there(self):
        preview = self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'prepare'}).json()
        self.assertTrue(preview['available'])
        self.assertEqual([step['step'] for step in preview['steps']],
                         ['media', 'hardcuts', 'missed_cuts', 'transcribe', 'subtitles', 'line_splits', 'voices', 'boxes', 'target_frames', 'cast'])
        steps = {step['step']: step for step in preview['steps']}
        self.assertEqual((steps['media']['state'], steps['hardcuts']['state'], steps['subtitles']['state']), ('done', 'will_run', 'off'))
        evaluate = self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'evaluate'}).json()
        self.assertEqual((evaluate['available'], evaluate['reason']), (False, 'Run Prepare first'))

        run_id = self.start('prepare')
        batch_runs.advance(run_id)
        self.assertEqual(self.states(run_id)['media'], 'skipped')       # the video was prepared already
        self.assertEqual(self.states(run_id)['hardcuts'], 'running')
        batch_runs.advance(run_id)                                       # nothing finished: nothing moves, nothing is written
        self.assertEqual(self.states(run_id)['missed_cuts'], 'pending')
        self.assertEqual(self.view(run_id)['steps'][1]['progress'], {'total': 1, 'finished': 0, 'failed': 0})

        self.cuts((0, 4), (4, 8))
        self.finish('hardcuts')
        batch_runs.advance(run_id)
        run = self.view(run_id)
        self.assertEqual(run['steps'][1]['summary'], '2 cuts detected')
        self.assertEqual(self.states(run_id)['missed_cuts'], 'running')
        self.finish('missed_cuts', result={'checked': 2, 'split': 0, 'doubtful': 3, 'in_sequence': 0, 'split_cuts': []})
        batch_runs.advance(run_id)
        missed = next(step for step in self.view(run_id)['steps'] if step['step'] == 'missed_cuts')
        self.assertEqual(missed['summary'], 'Checked 2 cuts: split 0; 3 doubtful places left as it is')

        # Transcription queues speaker attribution: the step waits for it too
        self.assertEqual(self.states(run_id)['transcribe'], 'running')
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES ('sa','e','speaker_attribution','queued','{}')")
        self.finish('transcribe')
        batch_runs.advance(run_id)
        self.assertEqual(self.states(run_id)['transcribe'], 'running')
        self.finish('speaker_attribution')
        batch_runs.advance(run_id)
        states = self.states(run_id)
        self.assertEqual((states['transcribe'], states['subtitles'], states['line_splits']), ('done', 'skipped', 'running'))
        self.finish('line_splits', result={'split': 1, 'trim': 0, 'leave': 2})
        batch_runs.advance(run_id)
        voices = self.jobs('voice_resolver')
        self.assertTrue(json.loads(voices[0]['payload_json'])['ask'])    # only Prepare's voice resolver may ask
        self.finish('voice_resolver', result={'lines_applied': 4, 'asked_person': 1})
        batch_runs.advance(run_id)
        states = self.states(run_id)
        self.assertEqual((states['boxes'], states['target_frames'], states['cast']), ('skipped', 'skipped', 'running'))
        self.finish('character_decisions', result={'boxes_name': 2})
        batch_runs.advance(run_id)
        run = self.view(run_id)
        self.assertEqual(run['status'], 'finished')
        summaries = {step['step']: step.get('summary') for step in run['steps']}
        self.assertIn('Gave a speaker to 4 lines; asked about 1 voice for you', summaries['voices'])
        self.assertEqual(summaries['line_splits'], 'Split 1 line, trimmed 0; 2 left across a cut (no pause near it)')

        with db.connect() as conn:
            self.assertEqual(stage_targets.prepare_run(conn, 'e')['id'], run_id)
            self.assertEqual(conn.execute('SELECT status,purpose FROM agent_runs WHERE id=?', (run_id,)).fetchone()[:], ('finished', 'run-multiple:prepare'))
        refused = self.client.post('/api/episodes/e/batch-runs', json={'stage': 'prepare'})
        self.assertEqual(refused.status_code, 409)
        self.assertIn('runs once', refused.json()['detail'])
        self.assertTrue(self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'evaluate'}).json()['available'])

    def test_subtitles_are_read_only_when_asked_for(self):
        self.cuts((0, 8))
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES ('t','e','transcribe','done','{}')")
        preview = self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'prepare', 'subtitles': 'Japanese'}).json()
        self.assertEqual(next(step for step in preview['steps'] if step['step'] == 'subtitles')['state'], 'will_run')
        self.assertEqual(self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'prepare', 'subtitles': 'Ignore. this'}).status_code, 422)
        run_id = self.start('prepare', {'subtitles': {'language': 'ja'}})
        batch_runs.advance(run_id)
        self.finish('missed_cuts', result={})
        batch_runs.advance(run_id)
        self.assertEqual(self.states(run_id)['subtitles'], 'running')
        self.assertEqual(db.one("SELECT subtitle_language FROM episodes WHERE id='e'")['subtitle_language'], 'Japanese')

    def test_a_failed_required_step_ends_the_run_and_prepare_can_start_again(self):
        run_id = self.start('prepare')
        batch_runs.advance(run_id)
        self.finish('hardcuts', status='failed')
        batch_runs.advance(run_id)
        run = self.view(run_id)
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(self.states(run_id)['hardcuts'], 'failed')
        self.assertTrue(all(step['state'] == 'skipped' and step['detail'] == 'Detect hard cuts failed' for step in run['steps'][2:]))
        with db.connect() as conn:
            self.assertIsNone(stage_targets.prepare_run(conn, 'e'))
        again = self.start('prepare')
        batch_runs.advance(again)
        self.assertEqual(self.states(again)['hardcuts'], 'running')

    def test_cancel_stops_the_current_step_and_one_run_at_a_time(self):
        run_id = self.start('prepare')
        batch_runs.advance(run_id)
        with mock.patch.object(batch_runs, 'advance'):
            busy = self.client.post('/api/episodes/e/batch-runs', json={'stage': 'prepare'})
        self.assertEqual(busy.status_code, 409)
        self.assertIn('going on', busy.json()['detail'])
        cancelled = self.client.post(f'/api/batch-runs/{run_id}/cancel').json()
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertEqual((cancelled['steps'][1]['state'], cancelled['steps'][2]['state']), ('cancelled', 'skipped'))
        self.assertEqual(db.one("SELECT status FROM jobs WHERE pipeline='hardcuts'")['status'], 'cancelled')
        self.assertEqual(self.client.post(f'/api/batch-runs/{run_id}/cancel').status_code, 409)
        listed = self.client.get('/api/episodes/e/batch-runs').json()
        self.assertEqual([item['id'] for item in listed], [run_id])
        self.assertEqual(self.client.get('/api/batch-runs/missing').status_code, 404)

    # -- Evaluate ---------------------------------------------------------------------------------------------------

    def prepared(self):
        """Five cuts after Prepare: c2 waits for the person (an unsure label), c3's first frame is stale, c4 lacks beats."""
        with db.connect() as conn:
            conn.execute("INSERT INTO batch_runs(id,episode_id,stage,status) VALUES ('prep','e','prepare','finished')")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES
                ('c1','e',0,0,2),('c2','e',1,2,4),('c3','e',2,4,6),('c4','e',3,6,8)""")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('ann','p','character','Ann','')")
            for index, cut in enumerate(('c1', 'c2', 'c3', 'c4')):
                start = index * 2
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES (?,?,?,?,'[]','t','complete')",
                             (f'r{cut}', cut, start, start + 2))
                conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES (?,?,'k',?,?,'[]')",
                             (f't{cut}', cut, start, start + 2))
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES
                ('d1','rc1','first','{}','ann','manual'),('d2','rc2','first','{}','ann','auto'),('d3','rc3','first','{}','ann','manual'),
                ('d4','rc4','first','{}','ann','manual')""")
            result(conn, 'f1', 'c1', 'initial_frame')
            result(conn, 'k1', 'c1', 'keyframe_analysis')
            result(conn, 'b1', 'c1', 'performance_beats', {'based_on': {'keyframe_analysis': 'k1'}})
            result(conn, 's1', 'c1', 'consistency_summary', {'questions': []}, title='Consistency summary')
            result(conn, 'f3', 'c3', 'initial_frame', stale='Cut timing changed', title='Initial frame description')
            result(conn, 'k3', 'c3', 'keyframe_analysis')
            result(conn, 'f4', 'c4', 'initial_frame')
            result(conn, 'k4', 'c4', 'keyframe_analysis')

    def test_evaluate_previews_the_valid_cuts_and_lists_the_ones_waiting_for_the_person(self):
        self.prepared()
        preview = self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'evaluate'}).json()
        self.assertTrue(preview['available'])
        steps = {step['step']: step for step in preview['steps']}
        self.assertEqual([item['cut'] for item in steps['visual']['cuts']], [3])      # stale
        self.assertEqual([item['cut'] for item in steps['beats']['cuts']], [4])       # missing
        self.assertEqual(steps['consistency']['state'], 'skipped')
        self.assertEqual(preview['cuts'], 2)                                           # cut 1 is current: never a target
        self.assertEqual([(item['cut'], item['step']) for item in preview['held']], [(2, 'visual')])
        self.assertIn('box 1 label not confirmed', preview['held'][0]['reasons'][0])
        only = self.client.get('/api/episodes/e/batch-runs/preview', params={'stage': 'evaluate', 'only': 'beats'}).json()
        self.assertEqual([step['state'] for step in only['steps']], ['skipped', 'will_run', 'skipped'])

    def test_evaluate_queues_only_its_target_cuts_step_by_step(self):
        self.prepared()
        run_id = self.start('evaluate')
        batch_runs.advance(run_id)
        self.assertEqual(self.states(run_id)['visual'], 'running')
        analyze = self.jobs('analyze')
        self.assertEqual(len(analyze), 1)
        self.assertEqual(json.loads(analyze[0]['payload_json'])['beats'], 'when_ready')
        self.finish('keyframes')
        self.finish('analyze')
        batch_runs.advance(run_id)
        run = self.view(run_id)
        visual = run['steps'][0]
        self.assertEqual(visual['stats']['not_done'], [3])                             # the stand-in job saved nothing
        self.assertEqual(self.states(run_id)['beats'], 'running')
        beats = self.jobs('performance_beats')
        self.assertEqual([json.loads(job['payload_json'])['cut_id'] for job in beats], ['c4'])
        self.finish('performance_beats')
        batch_runs.advance(run_id)
        run = self.view(run_id)
        self.assertEqual(run['status'], 'finished')
        self.assertEqual(self.states(run_id)['consistency'], 'skipped')
        self.assertEqual([item['cut'] for item in run['summary']['held']], [2])        # what still waits for the person

    def test_evaluate_and_prepare_cuts_need_prepare(self):
        for stage in ('evaluate', 'prepare_cuts'):
            with mock.patch.object(batch_runs, 'advance'):
                response = self.client.post('/api/episodes/e/batch-runs', json={'stage': stage})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['detail'], 'Run Prepare first')
        self.assertEqual(self.client.post('/api/episodes/e/batch-runs', json={'stage': 'results'}).status_code, 422)
        with db.connect() as conn:  # a Prepare from fw before D37 counts
            agent_log.finish_run(conn, agent_log.start_run(conn, 'e', 'fw', 'prepare'))
        with mock.patch.object(batch_runs, 'advance'):
            started = self.client.post('/api/episodes/e/batch-runs', json={'stage': 'prepare_cuts'})
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual([step['step'] for step in started.json()['steps']], ['boxes', 'target_frames', 'cast'])

    def test_visual_chunks_are_contiguous_runs_of_cuts(self):
        self.assertEqual(batch_runs.runs_of([5, 1, 2, 3, 7, 8]), [(1, 3), (5, 5), (7, 8)])


if __name__ == '__main__':
    unittest.main()
