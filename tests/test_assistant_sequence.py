"""Cut Assistance in a sequence chat: scoped tools, proposals that are rendered and checked, and applying them."""
import json
import time

import tests.test_sequence_pipeline as pipeline_tests
from app import db
from app.assistant import runner, sequence_tools, skills, tools


class SequenceAssistantTest(pipeline_tests.SequenceBase):
    def setUp(self):
        super().setUp()
        self.sequence_id = self.written()
        with db.connect() as conn:
            conn.execute('UPDATE assistant_worker_state SET providers_json=?,heartbeat=? WHERE id=1',
                         (json.dumps([{'id': 'fake', 'label': 'Fake', 'available': True, 'models': [{'id': 'fake-model'}], 'free_model': False}]), time.time()))
        response = self.client.post(f'/api/sequences/{self.sequence_id}/assistant/sessions', json={'provider': 'fake', 'model': 'fake-model'})
        self.assertEqual(response.status_code, 200, response.text)
        self.session = response.json()['session']
        self.ctx = tools.ToolContext(cut_id=self.session['cut_id'], session_id=self.session['id'], provider='fake', model='fake-model', sequence_id=self.sequence_id)

    def run_tool(self, name, args, ctx=None):
        content, label, ok = tools.execute(ctx or self.ctx, name, json.dumps(args))
        return json.loads(content), label, ok

    def proposals(self):
        return self.client.get(f"/api/assistant/sessions/{self.session['id']}").json()['proposals']

    def test_sessions_and_tools_are_scoped_to_the_sequence(self):
        self.assertEqual((self.session['sequence_id'], self.session['cut_id']), (self.sequence_id, 'c1'))
        self.assertEqual(len(self.client.get(f'/api/sequences/{self.sequence_id}/assistant/sessions').json()), 1)
        self.assertEqual(self.client.get('/api/cuts/c1/assistant/sessions').json(), [])  # not listed in the cut's own chat
        self.assertEqual(self.client.post('/api/sequences/missing/assistant/sessions', json={}).status_code, 404)
        names = {item['function']['name'] for item in tools.schemas('sequence')}
        self.assertEqual(names, {'get_sequence', 'get_shot', 'get_cut_evidence', 'get_sequence_transcript', 'propose_shot_edit', 'propose_plan_edit', 'propose_bible_edit',
                                 'get_characters', 'get_context_notes', 'load_skill'})
        self.assertNotIn('get_sequence', {item['function']['name'] for item in tools.schemas()})
        self.assertIn('check_h3_prompt', {item['function']['name'] for item in tools.schemas('cut')})
        _, label, ok = self.run_tool('get_result', {'type': 'video_analysis'})
        self.assertEqual((ok, label), (False, 'Unknown tool get_result'))
        _, _, ok = self.run_tool('get_sequence', {}, tools.ToolContext(cut_id='c1'))
        self.assertFalse(ok)
        self.assertEqual(self.run_tool('get_characters', {})[0], {'characters': [{'name': 'Hero', 'notes': 'Green coat'}]})
        self.assertIn('sequence_direction', skills.index_text('sequence'))
        self.assertNotIn('sequence_direction', skills.index_text())
        self.assertNotIn('h3_animation_prompt', skills.index_text('sequence'))
        prompt = runner.system_prompt(self.session)
        self.assertIn('working on ONE sequence', prompt)
        self.assertIn('cuts 1–3, 10.0 s', prompt)
        self.assertIn('Shots written: 3/3; stale: none; failing the format check: none.', prompt)
        self.assertIn('Review: 1 issue(s) in shots 1.', prompt)
        self.assertIn('ONE cut', runner.system_prompt({'cut_id': 'c1', 'sequence_id': None}))
        sequence_tools.sequence_card('missing')

    def test_reading_tools_return_compact_views(self):
        overview, label, ok = self.run_tool('get_sequence', {})
        self.assertTrue(ok, overview)
        self.assertEqual(label, 'Read sequence overview')
        self.assertEqual([(row['shot'], row['cut'], row['prompt'], row.get('lines')) for row in overview['shots']], [(1, 1, 'ok', ['L1']), (2, 2, 'ok', ['L2']), (3, 3, 'ok', ['L3'])])
        self.assertEqual((overview['shots'][0].get('opens'), overview['shots'][0]['closes']), (None, 'keep_cut'))
        self.assertEqual(overview['story']['spoken_language'], 'Japanese')
        self.assertEqual(overview['review']['issues'][0]['shot'], 1)
        self.assertEqual(self.run_tool('get_sequence', {'part': 'bible'})[0]['bible']['characters'][0]['name'], 'Hero')
        plan = self.run_tool('get_sequence', {'part': 'plan'})[0]
        self.assertEqual((plan['shots'][2]['t'], plan['shots'][2]['cut'], plan['shots'][2]['frame'], plan.get('boundaries_in_speech')), ('4.00–10.00', 3, 'cut_first_frame', None))
        shot, _, _ = self.run_tool('get_shot', {'shot': 2})
        self.assertEqual((shot['written'], shot['plan']['cut'], shot['direction']['camera']['move'], shot['check']['ok']), (True, 2, 'static', True))
        self.assertEqual(shot['plan']['opens'], {'at': 0.0, 'how': 'opens exactly on the hard cut'})
        self.assertEqual(shot['dialogue'], [{'line': 'L2', 't': '0.50–1.50', 'speaker': 'Hero', 'known_from': 'director', 'delivery': 'on_screen', 'text': 'はい'}])
        self.assertEqual(overview['shots'][1]['voices'], 'Hero on_screen')
        self.assertNotIn('prompt', shot)
        self.assertTrue(self.run_tool('get_shot', {'shot': 2, 'fields': ['prompt']})[0]['prompt'].startswith('subject_definitions:'))
        self.assertIn('the plan has 3 shot(s)', self.run_tool('get_shot', {'shot': 9})[0]['error'])
        evidence, label, _ = self.run_tool('get_cut_evidence', {'cut': 3, 'type': 'video_analysis', 'fields': ['summary', 'camera']})
        self.assertEqual((label, evidence['plays_at_in_sequence'], evidence['content']['summary']), ('Read video analysis of cut 3', '4.00–10.00', 'Hero raises a hand and speaks.'))
        self.assertTrue(self.run_tool('get_cut_evidence', {'cut': 1, 'type': 'production_brief'})[0]['missing'])
        self.assertIn('not part of this sequence', self.run_tool('get_cut_evidence', {'cut': 8, 'type': 'initial_frame'})[0]['error'])
        lines = self.run_tool('get_sequence_transcript', {})[0]['lines']
        self.assertEqual([(line['line'], line['speaker'], line['known_from']) for line in lines], [('L1', 'Hero', 'transcript'), ('L2', 'Hero', 'director'), ('L3', 'Hero', 'director')])
        self.assertEqual(self.run_tool('get_sequence_transcript', {'shot': 3})[0]['lines'][0]['t'], '1.00–2.00')

    def test_a_shot_edit_is_rendered_checked_and_applied_as_a_revision(self):
        invalid, label, ok = self.run_tool('propose_shot_edit', {'shot': 1, 'set': [{'path': 'summary', 'value': 'Hero waves before the cut.'}], 'summary': 'x'})
        self.assertEqual((ok, invalid['ok'], label), (True, False, 'Shot 1: direction has errors'))
        self.assertIn('summary contains the word "cut"', invalid['errors'])
        self.assertTrue(tools.REGISTRY['propose_shot_edit'].must_pass)
        self.assertEqual(self.proposals(), [])
        self.assertIn('does not change the shot', self.run_tool('propose_shot_edit', {'shot': 1, 'replacements': [{'find': 'Hero', 'replace': 'Hero'}], 'summary': 'x'})[0]['error'])
        self.assertIn('Text not found', self.run_tool('propose_shot_edit', {'shot': 1, 'replacements': [{'find': 'purple', 'replace': 'red'}], 'summary': 'x'})[0]['error'])
        made, label, ok = self.run_tool('propose_shot_edit', {'shot': 1, 'set': [{'path': 'voices.0.delivery', 'value': 'off_screen'}, {'path': 'voices.0.subject', 'value': None}],
                                                             'replacements': [{'find': 'green wool coat', 'replace': 'dark green wool coat'}], 'summary': 'The greeting comes from off screen'})
        self.assertEqual((ok, made['ok'], made['status'], label), (True, True, 'pending_user_approval', 'Proposed a change to shot 1'))
        self.assertIn('+[an off-screen voice is heard', made['diff'])
        proposal = self.proposals()[0]
        self.assertEqual((proposal['target'], proposal['label'], proposal['title'], proposal['status'], proposal['stale']), ('sequence_shot', 'shot prompt', 'Shot 1 prompt', 'pending', False))
        self.assertTrue(proposal['content'].startswith('subject_definitions:'))  # the card shows the rendered prompt
        self.assertTrue(any(segment['op'] == 'insert' and 'off-screen' in segment['text'] for segment in proposal['diff']))
        before = self.detail(self.sequence_id)['shots'][0]['result']
        applied = self.client.post(f"/api/assistant/proposals/{proposal['id']}/apply")
        self.assertEqual(applied.status_code, 200, applied.text)
        after = self.detail(self.sequence_id)['shots'][0]['result']
        self.assertEqual((after['origin'], after['parent_id'], applied.json()['applied_result_id']), ('assistant_edit', before['id'], after['id']))
        self.assertEqual((after['content']['voices'][0]['delivery'], after['check']['ok']), ('off_screen', True))
        self.assertIn('dark green wool coat', after['prompt'])
        settings = json.loads(db.one('SELECT settings_json FROM sequence_results WHERE id=?', (after['id'],))['settings_json'])
        self.assertEqual(settings['assistant_edit']['summary'], 'The greeting comes from off screen')
        self.assertEqual(self.client.post(f"/api/assistant/proposals/{proposal['id']}/apply").status_code, 409)

    def test_a_proposal_goes_stale_when_the_shot_changes_first(self):
        made, _, _ = self.run_tool('propose_shot_edit', {'shot': 2, 'set': [{'path': 'light', 'value': 'Cool evening light from the right.'}], 'summary': 'Light'})
        shot = self.detail(self.sequence_id)['shots'][1]['result']
        self.assertEqual(self.client.put(f"/api/sequence-results/{shot['id']}", json={'content': {**shot['content'], 'framing_lock': 'Head and shoulders only.'}}).status_code, 200)
        self.assertTrue(self.proposals()[0]['stale'])
        self.assertEqual(self.client.post(f"/api/assistant/proposals/{made['proposal_id']}/apply").status_code, 409)

    def test_plan_and_bible_edits(self):
        updated, label, ok = self.run_tool('propose_plan_edit', {'action': 'update', 'cut': 2, 'intent': 'Hero answers the friend.', 'summary': 'Sharper intent'})
        self.assertEqual((ok, label, updated['shots_after'], updated.get('shots_to_write_again'), updated.get('written_shots_that_disappear')), (True, 'Proposed a plan change (update at cut 2)', 3, None, None))
        self.assertIn('no speech crosses the hard cut between cut 1 and cut 2', self.run_tool('propose_plan_edit', {'action': 'boundary', 'cut': 1, 'choice': 'after_line', 'summary': 'x'})[0]['error'])
        self.assertIn('There is no hard cut after that cut', self.run_tool('propose_plan_edit', {'action': 'boundary', 'cut': 3, 'choice': 'keep_cut', 'summary': 'x'})[0]['error'])
        self.assertIn('Cut 9 is not part of this sequence', self.run_tool('propose_plan_edit', {'action': 'update', 'cut': 9, 'intent': 'x', 'summary': 'x'})[0]['error'])
        proposal = next(item for item in self.proposals() if item['target'] == 'sequence_plan')
        self.assertIn('2. 2.00–4.00 s (2.00 s) · cut 2 · opens keep_cut · closes keep_cut — Hero answers the friend.', proposal['content'])
        self.assertEqual(self.client.post(f"/api/assistant/proposals/{proposal['id']}/apply").status_code, 200)
        detail = self.detail(self.sequence_id)
        self.assertEqual(([shot['intent'] for shot in detail['shots']], [shot['stale'] for shot in detail['shots']], detail['plan']['origin']),
                         (['Follow the source.', 'Hero answers the friend.', 'Follow the source.'], [[], ['plan'], []], 'assistant_edit'))
        wrong, _, _ = self.run_tool('propose_bible_edit', {'set': [{'path': 'spoken_language', 'value': '日本語'}], 'summary': 'x'})
        self.assertEqual(wrong['ok'], False)
        self.assertIn('spoken_language', wrong['errors'])
        made, _, _ = self.run_tool('propose_bible_edit', {'set': [{'path': 'characters.0.outfit', 'value': 'dark green coat over a white shirt'}], 'summary': 'Coat colour'})
        self.assertTrue(made['ok'], made)
        self.assertEqual(self.client.post(f"/api/assistant/proposals/{made['proposal_id']}/apply").status_code, 200)
        detail = self.detail(self.sequence_id)
        self.assertEqual((detail['bible']['origin'], detail['bible']['content']['characters'][0]['outfit']), ('assistant_edit', 'dark green coat over a white shirt'))
        self.assertEqual(detail['shots'][0]['stale'], ['bible'])  # written from the old outfit

    def test_a_boundary_inside_speech_can_be_proposed(self):
        self.interrupt()
        sequence_id = self.create()
        self.run_step(sequence_id, 'direct')
        self.work()
        self.run_step(sequence_id, 'shots')
        self.work()
        session = self.client.post(f'/api/sequences/{sequence_id}/assistant/sessions', json={'provider': 'fake', 'model': 'fake-model'}).json()['session']
        ctx = tools.ToolContext(cut_id=session['cut_id'], session_id=session['id'], provider='fake', model='fake-model', sequence_id=sequence_id)
        plan = self.run_tool('get_sequence', {'part': 'plan'}, ctx)[0]
        boundary = plan['boundaries_in_speech'][0]
        self.assertEqual((boundary['after_cut'], boundary['lines'], boundary['choice'], boundary['decided_by'], sorted(boundary['options'])), (2, ['L3'], 'keep_cut', 'director', ['after_line', 'before_line', 'keep_cut']))
        made, label, ok = self.run_tool('propose_plan_edit', {'action': 'boundary', 'cut': 2, 'choice': 'before_line', 'reason': 'Hero says it to her face.', 'summary': 'Keep L3 whole'}, ctx)
        self.assertEqual((ok, label, made['shots_to_write_again']), (True, 'Proposed a plan change (boundary at cut 2)', [2, 3]))
        self.assertEqual(self.client.post(f"/api/assistant/proposals/{made['proposal_id']}/apply").status_code, 200)
        detail = self.detail(sequence_id)
        self.assertEqual(([shot['end_time'] for shot in detail['shots']], detail['plan']['content']['boundaries']),
                         ([2.0, 3.55, 10.0], [{'after_cut_id': 'c2', 'choice': 'before_line', 'reason': 'Hero says it to her face.', 'by': 'user'}]))
        self.assertEqual(detail['shots'][2]['opens']['reason'], 'Hero says it to her face.')

    def test_edits_that_cannot_be_made_do_not_ask_the_model_to_retry(self):
        with db.connect() as conn:
            conn.execute("DELETE FROM sequence_results WHERE type='shot'")
        result, label, ok = self.run_tool('propose_shot_edit', {'shot': 1, 'set': [{'path': 'light', 'value': 'x'}], 'summary': 'x'})
        self.assertEqual((ok, result['ok'], result.get('proposal_id'), label), (True, True, None, 'Shot 1: no change possible'))
        self.assertIn('Write shot prompts', result['not_possible'])

    def test_deleting_the_sequence_removes_its_chats(self):
        self.assertEqual(self.client.delete(f'/api/sequences/{self.sequence_id}').status_code, 200)
        self.assertEqual(self.client.get(f"/api/assistant/sessions/{self.session['id']}").status_code, 404)
