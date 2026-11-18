"""Performance beats: inputs without words, the grammar, the validator, queueing, the job, and the chain behind it."""
import json
import os
import unittest
from unittest.mock import patch

from app import db, inference_worker, performance_beats as beats, prompt_context, worker
import tests.test_analysis as analysis_tests

KEYFRAMES = {'contract': 'keyframes-v1', 'method': 'hybrid', 'run_id': 'r', 'selection_key': 'k', 'cut': {'start_time': 0, 'end_time': 2},
             'roster': [{'name': 'Hero', 'character_context_id': 'ctx', 'source': 'first_frame'}],
             'frames': [{'label': 'F1', 'ordinal': 1, 't': 0.0, 'characters': [{'name': 'Hero', 'character_context_id': 'ctx', 'state': 'standing, mouth closed', 'pose': 'standing upright'}]},
                        {'label': 'F2', 'ordinal': 2, 't': 0.8, 'characters': [{'name': 'Hero', 'character_context_id': 'ctx', 'state': 'standing, mouth open', 'pose': None}]},
                        {'label': 'F3', 'ordinal': 3, 't': 1.6, 'characters': [{'name': 'Hero', 'character_context_id': 'ctx', 'state': 'standing, mouth open', 'pose': None}]}],
             'segments': [{'from': 'F1', 'to': 'F2', 'start': 0.0, 'end': 0.8, 'camera': {'move': 'static', 'detail': 'pixel estimate'},
                           'characters': [{'name': 'Hero', 'character_context_id': 'ctx', 'walking': 'no', 'speaking': 'yes', 'blinking': 'no',
                                           'changes': [{'part': 'head', 'description': 'turns toward screen-right', 'source': 'qwen'},
                                                       {'part': 'arms', 'description': 'right arm raised to shoulder height', 'source': 'dwpose'}]}], 'other_changes': []},
                          {'from': 'F2', 'to': 'F3', 'start': 0.8, 'end': 1.6, 'camera': {'move': 'pan_left', 'detail': 'pixel estimate'},
                           'characters': [{'name': 'Hero', 'character_context_id': 'ctx', 'walking': 'no', 'speaking': 'yes', 'blinking': 'yes', 'changes': []}], 'other_changes': []}],
             'windows': [{'index': 0, 'labels': ['F1', 'F2', 'F3'], 'summary': 'Hero turns and talks.'}], 'summary': 'Hero turns and talks.', 'merge': {}}
SECRET = 'PINEAPPLE'


def answer(lines=('S1',), first_end=0.8):
    return {'summary': 'Hero turns and greets someone.',
            'beats': [{'start': 0, 'end': first_end, 'camera': {'move': 'static', 'detail': ''}, 'intent': 'notices',
                       'characters': [{'name': 'Hero', 'body': 'stands, raises the right arm', 'face': 'surprised', 'gaze': 'toward screen-right', 'mouth': 'speaking'}]},
                      {'start': first_end, 'end': 2.0, 'camera': {'move': 'pan left', 'detail': ''}, 'intent': 'greets',
                       'characters': [{'name': 'Hero', 'body': 'stands', 'face': 'smiles', 'gaze': 'toward screen-right', 'mouth': 'speaking'}]}],
            'speech': [{'line': line, 'delivery': 'on_screen'} for line in lines]}


class BeatsTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def seed(self, keyframes=KEYFRAMES, transcript=True):
        with db.connect() as conn:
            for result_id, kind, content in (('first', 'initial_frame', {'mode': 'full', 'composition': 'wide', 'visible_characters': [{'character': 'Hero', 'character_context_id': 'ctx', 'pose': 'standing'}]}),
                                             ('kf', 'keyframe_analysis', keyframes)):
                conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,context_json) VALUES (?,?,?,?,?,?,?)',
                             (result_id, 'c1', 'cut_description', kind, json.dumps(content), json.dumps({'structured_type': kind}), json.dumps({'cut': {'start_time': 0, 'end_time': 2}})))
            if transcript:
                conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id,origin) VALUES ('t1','e',0.2,0.9,?,'S0','ctx','model'),('t2','e',1.0,1.5,'Hello again','S1',NULL,'model')", (SECRET + ' there',))

    def gather(self):
        with db.connect() as conn:
            return beats.gather(conn, 'c1')

    def test_the_input_has_timing_speakers_and_evidence_but_never_the_words(self):
        self.seed()
        data, ctx, based_on = self.gather()
        text = json.dumps(data)
        self.assertNotIn(SECRET, text)
        self.assertNotIn('Hello again', text)
        self.assertEqual(based_on, {'keyframe_analysis': 'kf', 'initial_frame': 'first'})
        hero, unknown = data['speech']
        self.assertEqual((hero['line'], hero['speaker'], hero['start'], hero['end'], hero['on_screen'], hero['mouth'], hero['suggested']),
                         ('S1', 'Hero', 0.2, 0.9, True, 'yes', 'on_screen'))
        self.assertEqual((unknown['speaker'], unknown['on_screen'], unknown['suggested']), ('S1', False, None))  # a voice label is never guessed
        self.assertEqual(data['anchors'], [0.0, 0.2, 0.8, 0.9, 1.0, 1.5, 1.6, 2.0])
        self.assertEqual(data['characters'], ['Hero'])
        self.assertIn('(keypoints)', text)  # measured body parts are marked
        self.assertNotIn('character_context_id', json.dumps(data['opening_frame']))
        self.assertEqual(ctx['max_beats'], 7)

    def test_suggested_delivery(self):
        self.assertEqual(beats.suggested_delivery(False, 'unclear', True), 'off_screen')
        self.assertEqual(beats.suggested_delivery(True, 'yes', True), 'on_screen')
        self.assertEqual(beats.suggested_delivery(True, 'no', True), 'inner_voice')
        self.assertIsNone(beats.suggested_delivery(True, 'unclear', True))
        self.assertIsNone(beats.suggested_delivery(False, 'no', False))

    def test_grammar_never_has_an_empty_enum(self):
        schema = beats.answer_format([], [], 1)['json_schema']['schema']
        person = schema['properties']['beats']['items']['properties']['characters']['items']['properties']
        self.assertEqual(person['name'], {'type': 'string', 'maxLength': 80})
        self.assertEqual(schema['properties']['speech']['items']['properties']['line'], {'type': 'string'})
        self.assertEqual(schema['properties']['speech']['maxItems'], 0)
        named = beats.answer_format(['Hero'], ['S1'], 3)['json_schema']['schema']
        self.assertEqual(named['properties']['speech']['minItems'], 1)
        self.assertEqual(named['properties']['beats']['maxItems'], 3)

    def test_validator_snaps_edges_links_names_and_attaches_lines(self):
        self.seed()
        _, ctx, _ = self.gather()
        out = beats.validate(answer(('S1', 'S2'), first_end=0.85), ctx)
        self.assertEqual([(beat['start'], beat['end']) for beat in out['beats']], [(0.0, 0.8), (0.8, 2.0)])  # 0.85 snapped onto the 0.8 keyframe
        self.assertEqual(out['beats'][0]['characters'][0]['character_context_id'], 'ctx')
        self.assertEqual([item['line'] for item in out['beats'][0]['speech']], ['S1'])
        self.assertEqual([item['line'] for item in out['beats'][1]['speech']], ['S2'])
        self.assertIn({'kind': 'on_screen_speaker_not_seen', 'line': 'S2', 'speaker': 'S1'}, out['checks'])
        self.assertEqual(out['contract'], 'beats-v1')

    def test_validator_sentences_go_back_to_the_model(self):
        self.seed()
        _, ctx, _ = self.gather()
        cases = [(lambda a: a['beats'][0].update(start=0.5), 'must start at 0'),
                 (lambda a: a['beats'][1].update(end=1.0), 'must end at the cut'),
                 (lambda a: a['beats'][1].update(start=1.2), 'without gaps'),
                 (lambda a: a.update(speech=[]), 'missing: S1, S2'),
                 (lambda a: a['speech'].append({'line': 'S9', 'delivery': 'on_screen'}), 'unknown line'),
                 (lambda a: (a['beats'][0].update(end=0.5), a['beats'][1].update(start=0.5)), 'not an anchor')]
        for change, sentence in cases:
            bad = answer(('S1', 'S2'))
            change(bad)
            with self.assertRaisesRegex(ValueError, sentence):
                beats.validate(bad, ctx)

    def test_queueing_reuses_a_waiting_job_and_adds_one_beside_a_running_one(self):
        with db.connect() as conn:
            first = beats.ensure_queued(conn, 'e', 'c1')
            self.assertEqual(beats.ensure_queued(conn, 'e', 'c1'), first)
            conn.execute("UPDATE jobs SET status='running' WHERE id=?", (first,))
            second = beats.ensure_queued(conn, 'e', 'c1')
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (second,))['payload_json'])['config']['text_provider'], 'openrouter')

    def run_job(self, reply):
        with db.connect() as conn:
            job_id = beats.insert(conn, 'e', 'c1')
        claimed = worker.claim_job(('performance_beats',))
        self.assertEqual(claimed[0], job_id)
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(beats, 'generate', side_effect=reply):
            beats.process(*claimed)
        return job_id

    def test_the_job_saves_beats_built_from_the_current_keyframes(self):
        self.seed()
        sent = {}

        def reply(config, data, ctx, **_):
            sent['data'] = data
            return beats.validate(answer(('S1', 'S2')), ctx), 'test/model'
        job_id = self.run_job(reply)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
        row = db.one("SELECT * FROM analysis_results WHERE job_id=?", (job_id,))
        content = json.loads(row['content'])
        self.assertEqual(content['based_on']['keyframe_analysis'], 'kf')
        self.assertEqual(json.loads(row['settings_json'])['structured_type'], 'performance_beats')
        self.assertNotIn(SECRET, row['content'] + row['context_json'])
        by_type = {'keyframe_analysis': {'id': 'kf'}, 'performance_beats': {'content': content}}
        self.assertTrue(prompt_context.beats_current(by_type))
        self.assertEqual(db.one("SELECT status FROM analysis_job_stages WHERE job_id=?", (job_id,))['status'], 'done')

    def test_a_newer_keyframe_analysis_supersedes_the_job(self):
        self.seed()

        def reply(config, data, ctx, **_):
            with db.connect() as conn:  # the keyframes were analysed again meanwhile
                conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json,parent_id) VALUES ('kf2','c1','cut_description','Keyframe analysis','{}',?, 'kf')",
                             (json.dumps({'structured_type': 'keyframe_analysis'}),))
                conn.execute("UPDATE analysis_results SET superseded_by='kf2' WHERE id='kf'")
            return beats.validate(answer(('S1', 'S2')), ctx), 'test/model'
        job_id = self.run_job(reply)
        job = db.one('SELECT status,stage FROM jobs WHERE id=?', (job_id,))
        self.assertEqual((job['status'], job['stage']), ('done', 'Superseded by a newer keyframe analysis'))
        self.assertIsNone(db.one("SELECT id FROM analysis_results WHERE job_id=?", (job_id,)))

    def test_without_keyframes_the_job_fails_with_the_reason(self):
        job_id = self.run_job(lambda *a, **k: self.fail('no model call without keyframes'))
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('run Keyframes first', job['error'])

    def test_a_manual_edit_of_the_keyframes_queues_new_beats(self):
        self.seed()
        response = self.client.put('/api/analysis-results/kf', json={'title': 'Keyframe analysis', 'content': json.dumps(KEYFRAMES)})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(db.all_rows("SELECT id FROM jobs WHERE pipeline='performance_beats' AND status='queued'")), 1)

    def test_endpoint_and_retry(self):
        self.assertEqual(self.client.post('/api/cuts/c1/performance-beats').status_code, 409)  # no keyframes yet
        self.seed()
        job_id = self.client.post('/api/cuts/c1/performance-beats').json()['job_id']
        self.assertEqual(self.client.post('/api/cuts/c1/performance-beats').status_code, 409)
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job_id,))
        retry = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (retry.json()['job_id'],))['payload_json'])['cut_id'], 'c1')


class ChainTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown
    seed = BeatsTest.seed

    def payload(self, job_id):
        return json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (job_id,))['payload_json'])

    def test_the_text_chain_writes_the_beats_first_and_the_review_reads_them(self):
        self.seed()
        body = self.client.post('/api/episodes/e/text-chain?from_cut=1').json()
        self.assertEqual(body['blocked']['missing_results'], [2])  # c2 has no analysis
        beats_job = db.one("SELECT id FROM jobs WHERE pipeline='performance_beats'")['id']
        consistency = db.one("SELECT id FROM jobs WHERE pipeline='consistency_summary'")['id']
        summary = db.one("SELECT id FROM jobs WHERE pipeline='regen'")['id']
        payload = self.payload(consistency)
        self.assertEqual((payload['wait_for'], payload['refresh_evidence']), (beats_job, True))
        self.assertIn('Performance beats:\nNot written yet.', payload['evidence']['visual_evidence_text'])
        self.assertTrue(self.payload(summary)['refresh_results'])
        self.assertIsNone(worker.claim_job(('consistency_summary',)))  # held behind the beats
        # The beats job runs; the consistency job then builds its evidence from them.
        claimed = worker.claim_job(('performance_beats',))
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), \
             patch.object(beats, 'generate', side_effect=lambda config, data, ctx, **_: (beats.validate(answer(('S1', 'S2')), ctx), 'test/model')):
            beats.process(*claimed)
        claimed = worker.claim_job(('consistency_summary',))
        self.assertEqual(claimed[0], consistency)
        seen = {}

        def generate(config, evidence, **_):
            seen['evidence'] = evidence
            return {'summary': 's', 'clear': 'c', 'confusing': 'None identified.', 'questions': []}, 'test/model'
        from app import consistency_summary
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), patch.object(consistency_summary, 'generate', side_effect=generate):
            inference_worker.process_consistency_summary(*claimed)
        self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (consistency,))['status'], 'done')
        self.assertIn('Performance beats:\nkind: performance beats', seen['evidence']['visual_evidence_text'])
        self.assertIn('intent: greets', seen['evidence']['visual_evidence_text'])
        self.assertNotIn(SECRET, seen['evidence']['visual_evidence_text'])

    def test_a_failed_beats_job_cancels_the_review_behind_it(self):
        self.seed()
        self.client.post('/api/episodes/e/text-chain?from_cut=1')
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='failed' WHERE pipeline='performance_beats'")
        self.assertIsNone(worker.claim_job(('consistency_summary', 'review_answers', 'regen')))
        self.assertEqual({row['status'] for row in db.all_rows("SELECT status FROM jobs WHERE pipeline IN ('consistency_summary','review_answers','regen')")}, {'cancelled'})

    def test_current_beats_are_not_written_again(self):
        self.seed()
        with db.connect() as conn:
            conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES (?,?,?,?,?,?)',
                         ('b', 'c1', 'cut_description', 'Performance beats', json.dumps({'beats': [], 'speech': [], 'based_on': {'keyframe_analysis': 'kf'}}),
                          json.dumps({'structured_type': 'performance_beats'})))
        self.client.post('/api/episodes/e/text-chain?from_cut=1')
        self.assertIsNone(db.one("SELECT id FROM jobs WHERE pipeline='performance_beats'"))
        consistency = db.one("SELECT payload_json FROM jobs WHERE pipeline='consistency_summary'")
        self.assertNotIn('wait_for', json.loads(consistency['payload_json']))


if __name__ == '__main__':
    unittest.main()
