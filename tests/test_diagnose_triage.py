"""Autopilot M1: cut health, root-cause chains, failure classification and triage with undo (docs/autopilot/HARNESS.md)."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import agent_log, db, diagnose, media, triage, worker
from app.main import app


def result(conn, result_id, cut_id, kind, content=None, stale=None, created='2026-11-20 10:00:00', title=None):
    conn.execute("""INSERT INTO analysis_results(id,cut_id,kind,title,content,origin,settings_json,stale_reason,created_at)
        VALUES (?,?,?,?,?,'model',?,?,?)""", (result_id, cut_id, 'structured', title or kind, json.dumps(content or {}),
                                              json.dumps({'structured_type': kind} if kind != 'summary' else {}), stale, created))


def failed(conn, pipeline, payload, error, finished='2026-11-21 10:00:00'):
    job_id = worker.insert_job(conn, 'e', pipeline, payload)
    conn.execute("UPDATE jobs SET status='failed',stage='Failed',error=? WHERE id=?", (error, job_id))
    conn.execute('UPDATE jobs SET finished_at=?,updated_at=? WHERE id=?', (finished, finished, job_id))  # after the status trigger
    return job_id


class ClassifyTest(unittest.TestCase):
    """The real error messages of the last weeks (HARNESS.md section 3) land on the right cause."""

    def test_real_errors(self):
        cases = {
            ('performance_beats', 'This cut has no keyframe analysis; run Keyframes first'): 'prerequisite_missing',
            ('consistency_summary', 'The performance beats of this cut are missing or older than its keyframe analysis; write them first'): 'prerequisite_stale',
            ('transcribe', 'CUDA out of memory. Tried to allocate 298.00 MiB.'): 'gpu_memory',
            ('consistency_summary', "Local model failed to start: …common_fit_params: failed to fit params to free device memory"): 'gpu_memory',
            ('voice_sample', 'miopenStatusUnknownError'): 'gpu_memory',
            ('analyze', 'OpenRouter returned an invalid analysis after 3 attempts: response was truncated'): 'provider',
            ('regen', 'Local model returned an invalid analysis after 3 attempts: animation timeline contains an invalid cut-relative interval'): 'validator_loop',
            ('regen', 'Local model request failed (HTTP 400): request (33654 tokens) exceeds the available context size (32768 tokens)'): 'context_overflow',
            ('analyze', '1 of 1 cuts failed, the others finished and were kept. cut 13: Cut preview frame timestamps are invalid; refresh the cut frames before analysis'): 'stale_previews',
            ('character_detect', 'Character comparison failed: Invalid character match response'): 'legacy_pipeline',
            ('consistency_summary', "No module named 'fastapi'"): 'code_error',
            ('analyze', '1 of 1 cuts failed. cut 326: [vost#0:0/libx264 @ 0x57] Error submitting a packet to the muxer'): 'media_error',
            ('sequence', 'Local model returned an invalid analysis after 3 attempts: beat 3 ends at 3 s'): 'out_of_scope',
            ('subtitles', 'something nobody has seen'): 'unknown',
            ('subtitles', '2 of 441 subtitle images could not be read (Subtitle reading failed: The subtitle answer did not follow the required shape). The others were saved'): 'subtitle_partial',
        }
        for (pipeline, error), cause in cases.items():
            self.assertEqual(diagnose.classify(pipeline, error)[0], cause, error)

    def test_sentence_takes_the_cut_part_and_the_validator_message(self):
        error = '2 of 9 cuts failed, the others finished and were kept. cut 70: Frame description failed: Response was truncated; cut 247: Local model returned an invalid analysis after 3 attempts: beat 2 ends late'
        self.assertEqual(diagnose.sentence(error, 247), 'beat 2 ends late')
        self.assertEqual(diagnose.sentence(error, 70), 'Frame description failed: Response was truncated')
        self.assertEqual(diagnose.sentence('OpenRouter returned an invalid analysis after one repair attempt: response was empty'), 'response was empty')


class DiagnoseTriageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,has_audio)
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}',1)""")
            conn.execute("""INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,no_characters) VALUES
                ('c1','e',0,0,4,1),('c2','e',1,4,8,1),('c3','e',2,8,12,1),('c4','e',3,12,16,1)""")
            # c1: fully analysed, but its beats were built from an older keyframe analysis
            result(conn, 'f1', 'c1', 'initial_frame')
            result(conn, 'k1', 'c1', 'keyframe_analysis', created='2026-11-20 12:00:00')
            result(conn, 'b1', 'c1', 'performance_beats', {'based_on': {'keyframe_analysis': 'k0'}})
            # c2: first frame and keyframes, no beats; c3: nothing; c4: complete with a stale summary
            result(conn, 'f2', 'c2', 'initial_frame')
            result(conn, 'k2', 'c2', 'keyframe_analysis')
            result(conn, 'f4', 'c4', 'initial_frame')
            result(conn, 'k4', 'c4', 'keyframe_analysis')
            result(conn, 'b4', 'c4', 'performance_beats', {'based_on': {'keyframe_analysis': 'k4'}})
            result(conn, 's4', 'c4', 'consistency_summary', {'questions': []})
            result(conn, 'cs4', 'c4', 'summary', stale='Transcript changed', title='Cut summary')
            # a line that runs across the c2|c3 cut
            conn.execute("""INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,character_context_id)
                VALUES ('l1','e',7.0,9.0,'across the cut',NULL)""")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def test_why_follows_the_chain_to_the_first_broken_link(self):
        with db.connect() as conn:
            chain = diagnose.why(conn, 'c1', 'consistency')
        self.assertEqual([(link['step'], link['state']) for link in chain['chain']], [('consistency', 'missing'), ('beats', 'stale')])
        self.assertIn('keyframes redone 2026-11-20', chain['root']['text'])
        self.assertEqual(chain['fix']['command'], 'fw beats EP 1')
        self.assertEqual(chain['place'], 'analysis')
        response = self.client.get('/api/cuts/c3/why')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.json()['step'], response.json()['root']['step']), ('first_frame', 'first_frame'))
        self.assertEqual(self.client.get('/api/cuts/c3/why?step=nope').status_code, 422)
        self.assertEqual(self.client.get('/api/cuts/missing/why').status_code, 404)

    def test_health_ranks_failures_first_and_never_flags_lines_across_cuts(self):
        with db.connect() as conn:
            failed(conn, 'performance_beats', {'cut_id': 'c2'}, 'This cut has no keyframe analysis; run Keyframes first')
        data = self.client.get('/api/episodes/e/health').json()
        self.assertEqual(data['cuts'][0]['cut'], 2)
        self.assertEqual(data['cuts'][0]['problems'][0]['kind'], 'failed')
        self.assertEqual(data['cuts'][0]['problems'][0]['cause'], 'prerequisite_missing')
        kinds = {cut['cut']: [problem['kind'] for problem in cut['problems']] for cut in data['cuts']}
        # A line across a hard cut is not a problem: the line splitter handled what it could in Prepare (D37)
        self.assertFalse(any('line_crossing' in found for found in kinds.values()))
        self.assertIn('stale', kinds[1])
        self.assertIn('stale', kinds[4])
        self.assertIn('missing', kinds[3])
        ranged = self.client.get('/api/episodes/e/health?from_cut=3&to_cut=4').json()
        self.assertEqual({cut['cut'] for cut in ranged['cuts']}, {3, 4})

    def test_triage_queues_the_missing_step_then_the_step_and_undo_takes_it_back(self):
        with db.connect() as conn:
            job = failed(conn, 'performance_beats', {'cut_id': 'c3'}, 'This cut has no keyframe analysis; run Keyframes first')
        plan = self.client.get('/api/episodes/e/triage').json()
        self.assertEqual([(item['job_id'], item['cause'], item['action']) for item in plan['items']], [(job, 'prerequisite_missing', 'prerequisite')])
        run = self.client.post('/api/episodes/e/agent-runs', json={'agent': 'test'}).json()
        outcome = self.client.post('/api/episodes/e/triage', json={'run_id': run['id']}).json()
        self.assertEqual(len(outcome['applied']), 1, outcome)
        with db.connect() as conn:
            queued = conn.execute("SELECT id,pipeline,retry_of,payload_json FROM jobs WHERE status='queued' ORDER BY rowid").fetchall()
            decision = conn.execute('SELECT * FROM agent_decisions').fetchone()
        pipelines = [row['pipeline'] for row in queued]
        self.assertIn('analyze', pipelines)
        beats = next(row for row in queued if row['pipeline'] == 'performance_beats')
        self.assertEqual(beats['retry_of'], job)
        visual = next(row for row in queued if row['pipeline'] == 'analyze')
        self.assertIn(visual['id'], json.loads(beats['payload_json'])['depends_on'])
        self.assertEqual(json.loads(visual['payload_json'])['stages'], ['keyframe_analysis'])
        self.assertEqual((decision['kind'], decision['tier'], decision['target_id'], decision['run_id']), ('failed_runs', 'rules', job, run['id']))
        self.assertEqual(self.client.get('/api/episodes/e/triage').json()['items'], [])  # the new run deals with the failure
        undone = self.client.post(f"/api/agent-runs/{run['id']}/undo").json()
        self.assertEqual((len(undone['undone']), undone['conflicts']), (1, []))
        with db.connect() as conn:
            self.assertFalse(conn.execute("SELECT 1 FROM jobs WHERE status='queued'").fetchone())
        self.assertEqual(len(self.client.get('/api/episodes/e/triage').json()['items']), 1)  # it fails again in the list

    def test_a_used_up_rule_escalates_once_per_cause_and_closes_when_settled(self):
        with db.connect() as conn:
            first = failed(conn, 'consistency_summary', {'cut_id': 'c1', 'config': {}}, 'Local model returned an invalid analysis after 3 attempts: questions must be a list')
        outcome = self.client.post('/api/episodes/e/triage', json={}).json()
        self.assertEqual([item['action'] for item in outcome['applied']], ['requeue'])
        with db.connect() as conn:
            requeued = conn.execute("SELECT id FROM jobs WHERE pipeline='consistency_summary' AND retry_of=?", (first,)).fetchone()['id']
            beats = conn.execute("SELECT id FROM jobs WHERE pipeline='performance_beats' AND status='queued'").fetchone()
            self.assertIsNotNone(beats)  # the stale beats are written again first
            self.assertIn(beats['id'], json.loads(conn.execute('SELECT payload_json FROM jobs WHERE id=?', (requeued,)).fetchone()[0])['depends_on'])
            conn.execute("UPDATE jobs SET status='failed',error='Local model returned an invalid analysis after 3 attempts: questions must be a list',finished_at='2026-11-22 10:00:00' WHERE id=?", (requeued,))
        plan = self.client.get('/api/episodes/e/triage').json()
        self.assertEqual([(item['action'], item['attempts']) for item in plan['items']], [('escalate', 1)])
        outcome = self.client.post('/api/episodes/e/triage', json={}).json()
        self.assertEqual(len(outcome['escalated']), 1)
        self.client.post('/api/episodes/e/triage', json={})  # asking again does not open a second question
        items = self.client.get('/api/episodes/e/escalations').json()['items']
        self.assertEqual([(item['kind'], item['group_key']) for item in items], [('failed_run', 'validator_loop:consistency_summary')])
        self.assertIn('questions must be a list', items[0]['question'])
        self.client.post(f'/api/jobs/{requeued}/dismiss')  # the person deals with it: the question closes with the dismiss
        self.assertEqual(self.client.get('/api/episodes/e/escalations').json()['items'], [])
        self.assertEqual(self.client.post('/api/episodes/e/triage', json={}).json()['closed_escalations'], 0)

    def test_obsolete_failures_are_dismissed_and_sequences_left_alone(self):
        with db.connect() as conn:
            obsolete = failed(conn, 'analyze', {'workflow': 'legacy', 'cuts': [{'cut_id': 'c4'}]},
                              'OpenRouter returned an invalid analysis after 3 attempts: response was truncated', finished='2026-11-19 10:00:00')
            conn.execute("INSERT INTO sequences(id,episode_id,title,first_cut_id,last_cut_id,cuts_json,start_time,end_time) VALUES ('s','e','S','c1','c2','[]',0,8)")
            sequence = failed(conn, 'sequence', {'sequence_id': 's', 'step': 'plan'}, 'Local model returned an invalid analysis after 3 attempts: beat 1')
        outcome = self.client.post('/api/episodes/e/triage', json={}).json()
        self.assertEqual([(item['job_id'], item['action']) for item in outcome['applied']], [(obsolete, 'dismiss')])
        self.assertEqual([item['job_id'] for item in outcome['left']], [sequence])
        with db.connect() as conn:
            self.assertIsNotNone(conn.execute('SELECT dismissed_at FROM jobs WHERE id=?', (obsolete,)).fetchone()[0])
            decision = conn.execute('SELECT id FROM agent_decisions').fetchone()[0]
            conn.execute('BEGIN IMMEDIATE')
            agent_log.undo_decision(conn, decision)
            self.assertIsNone(conn.execute('SELECT dismissed_at FROM jobs WHERE id=?', (obsolete,)).fetchone()[0])

    def test_profile_modes_and_gpu_wait(self):
        with db.connect() as conn:
            failed(conn, 'performance_beats', {'cut_id': 'c2'}, 'CUDA out of memory. Tried to allocate 2 GiB')
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'failed_runs': 'ask'}}})
        self.assertEqual([item['action'] for item in self.client.get('/api/episodes/e/triage').json()['items']], ['escalate'])
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'failed_runs': 'off'}}})
        self.assertEqual([item['action'] for item in self.client.get('/api/episodes/e/triage').json()['items']], ['leave'])
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'failed_runs': 'apply'}}})
        with db.connect() as conn:
            busy = worker.insert_job(conn, 'e', 'consistency_summary', {'cut_id': 'c4', 'config': {}})
            conn.execute("UPDATE jobs SET status='running' WHERE id=?", (busy,))
        self.assertEqual([item['action'] for item in self.client.get('/api/episodes/e/triage').json()['items']], ['wait'])
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='done' WHERE id=?", (busy,))
        outcome = self.client.post('/api/episodes/e/triage', json={}).json()
        self.assertEqual([item['action'] for item in outcome['applied']], ['rerun'])

    def test_undo_refuses_once_the_fix_started(self):
        with db.connect() as conn:
            failed(conn, 'performance_beats', {'cut_id': 'c2'}, 'Local model returned an invalid analysis after 3 attempts: beat edges')
        outcome = self.client.post('/api/episodes/e/triage', json={}).json()
        decision = outcome['applied'][0]['decision_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running' WHERE id=?", (outcome['applied'][0]['job_ids'][0],))
        self.assertEqual(self.client.post(f'/api/agent-decisions/{decision}/undo').status_code, 409)
        self.assertIs(agent_log.UNDO['jobs'], triage.undo_fix)


if __name__ == '__main__':
    unittest.main()
