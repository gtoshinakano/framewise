"""Cut Assistance transcript corrections: planning, proposal cards, apply, stale checks and the chat flow."""
import json
import unittest
from unittest.mock import patch

import tests.test_assistant as assistant_tests
from app import db
from app.assistant import providers, runner, skills, tools, transcript_edit


class TranscriptBase(assistant_tests.AssistantBase):
    """Cut c1 spans 0.5-2.0s. Line 1 is Hero's 'こんにちは' (0.4-1.2); line 2 is the unlinked SPEAKER_02 saying 'Hello there.' in two word fragments."""

    def setUp(self):
        super().setUp()
        with db.connect() as conn:
            conn.execute("UPDATE episodes SET duration=8 WHERE id='e'")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('john','p','character','John','Tall'),('mary','p','character','Mary','Short')")
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES "
                         "('w1','e',1.3,1.5,'Hello','SPEAKER_02','model'),('w2','e',1.5,1.8,'there.','SPEAKER_02','model')")

    def propose(self, edits, summary='Fix transcript'):
        value, _, ok = self.run_tool('propose_transcript_edit', {'edits': edits, 'summary': summary})
        return value, ok

    def current(self):
        return self.client.get('/api/episodes/e/transcript').json()

    def apply(self, proposal_id):
        return self.client.post(f'/api/assistant/proposals/{proposal_id}/apply')


class PlanTest(TranscriptBase):
    def test_get_transcript_numbers_the_merged_lines(self):
        value, _, ok = self.run_tool('get_transcript', {})
        self.assertTrue(ok)
        self.assertEqual([(item['line'], item['text'], item.get('character')) for item in value['segments']], [(1, 'こんにちは', 'Hero'), (2, 'Hello there.', None)])

    def test_speaker_only_update_keeps_fragments_and_timings(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'update', 'line': 2, 'character': 'mary'}], 'Speaker is Mary')
        self.assertTrue(ok, value)
        self.assertEqual(value['status'], 'pending_user_approval')
        self.assertEqual(value['changes'], ['line 2: speaker changed → Mary'])
        self.assertNotIn('unlinked_lines_with_same_voice_label', value)
        proposal = db.one('SELECT * FROM assistant_proposals WHERE id=?', (value['proposal_id'],))
        ops = json.loads(proposal['payload_json'])['ops']
        self.assertEqual([(op['op'], op['id'], op['start_time'], op['end_time'], op['text'], op['speaker_label'], op['character_context_id']) for op in ops],
                         [('revise', 'w1', 1.3, 1.5, 'Hello', 'SPEAKER_02', 'mary'), ('revise', 'w2', 1.5, 1.8, 'there.', 'SPEAKER_02', 'mary')])
        self.assertIn('Mary (SPEAKER_02): Hello there.', proposal['content'])
        self.assertIn('SPEAKER_02: Hello there.', proposal['base_content'])
        self.assertNotIn('Mary', proposal['base_content'])
        # Nothing is saved until the user applies it.
        self.assertIsNone(db.one("SELECT character_context_id FROM transcript_segments WHERE id='w1'")['character_context_id'])

    def test_speaker_change_can_cover_a_range_of_lines(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES ('w3','e',1.85,1.95,'Right.','SPEAKER_02','manual')")
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'update', 'line': 1, 'to_line': 3, 'character': 'Mary'}])
        self.assertTrue(ok, value)
        self.assertEqual(value['changes'], ['lines 1–3: 3 changed (speaker changed → Mary)'])
        ops = json.loads(db.one('SELECT payload_json FROM assistant_proposals WHERE id=?', (value['proposal_id'],))['payload_json'])['ops']
        self.assertEqual([(op['id'], op['character_context_id']) for op in ops], [('t1', 'mary'), ('w1', 'mary'), ('w2', 'mary'), ('w3', 'mary')])
        self.assertEqual(self.apply(value['proposal_id']).status_code, 200)
        # The edited fragments now share speaker, origin and pauses, so they read as one line. Lines that
        # already match are skipped; if none would change, the model is told.
        again, ok = self.propose([{'op': 'update', 'line': 1, 'to_line': 2, 'character': 'Mary'}])
        self.assertFalse(ok)
        self.assertIn('already have that text and speaker', again['error'])
        partly, ok = self.propose([{'op': 'update', 'line': 1, 'to_line': 2, 'text': 'x'}])
        self.assertFalse(ok)
        self.assertIn('use rewrite for a range', partly['error'])

    def test_text_update_collapses_word_fragments_and_replacements_fix_a_word(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'update', 'line': 2, 'text': 'Hello, Maria!', 'character': 'Mary'}])
        self.assertTrue(ok, value)
        ops = json.loads(db.one('SELECT payload_json FROM assistant_proposals WHERE id=?', (value['proposal_id'],))['payload_json'])['ops']
        self.assertEqual([(op['op'], op['id']) for op in ops], [('revise', 'w1'), ('delete', 'w2')])
        self.assertEqual((ops[0]['start_time'], ops[0]['end_time'], ops[0]['text']), (1.3, 1.8, 'Hello, Maria!'))
        word, ok = self.propose([{'op': 'update', 'line': 2, 'replacements': [{'find': 'there', 'replace': 'Maria'}]}])
        self.assertTrue(ok, word)
        self.assertIn('Hello Maria.', db.one('SELECT content FROM assistant_proposals WHERE id=?', (word['proposal_id'],))['content'])

    def test_rewrite_shares_the_old_time_span_and_labels_a_new_speaker(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'rewrite', 'line': 1, 'to_line': 2, 'lines': [{'character': 'John', 'text': 'aaaa'}, {'character': 'Mary', 'text': 'bbbbbb'}]}])
        self.assertTrue(ok, value)
        ops = json.loads(db.one('SELECT payload_json FROM assistant_proposals WHERE id=?', (value['proposal_id'],))['payload_json'])['ops']
        self.assertEqual([(op['op'], op['id']) for op in ops], [('revise', 't1'), ('revise', 'w1'), ('delete', 'w2')])
        first, second = ops[0], ops[1]
        self.assertEqual((first['start_time'], second['end_time']), (0.4, 1.8))
        self.assertEqual(first['end_time'], second['start_time'])
        self.assertAlmostEqual(first['end_time'], 0.4 + 1.4 * 4 / 10, places=3)
        self.assertEqual((first['character_context_id'], first['speaker_label']), ('john', 'John'))
        self.assertEqual((second['character_context_id'], second['speaker_label']), ('mary', 'Mary'))

    def test_delete_and_insert(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'delete', 'line': 1}, {'op': 'insert', 'lines': [{'character': 'John', 'text': 'Wait!', 'start': 0.1, 'end': 0.5}]}])
        self.assertTrue(ok, value)
        ops = json.loads(db.one('SELECT payload_json FROM assistant_proposals WHERE id=?', (value['proposal_id'],))['payload_json'])['ops']
        self.assertEqual([op['op'] for op in ops], ['delete', 'insert'])
        self.assertEqual((ops[1]['start_time'], ops[1]['end_time']), (0.6, 1.0))  # clip-relative to episode time
        overlap, ok = self.propose([{'op': 'insert', 'lines': [{'text': 'Hello again', 'start': 0.9, 'end': 1.2}]}])
        self.assertTrue(ok, overlap)
        self.assertIn('overlaps existing line 2', overlap['warnings'][0])

    def test_errors_are_returned_to_the_model(self):
        self.ctx.session_id = self.session()
        cases = [
            ([{'op': 'update', 'line': 2, 'character': 'Jon'}], 'Did you mean: John'),
            ([{'op': 'update', 'line': 9, 'text': 'x'}], 'no transcript line 9'),
            ([{'op': 'update', 'line': 2, 'replacements': [{'find': 'nope', 'replace': 'x'}]}], 'is not in line 2'),
            ([{'op': 'update', 'line': 2, 'text': 'Hello there.'}], 'already has that text'),
            ([{'op': 'update', 'line': 2, 'text': 'x', 'replacements': [{'find': 'a', 'replace': 'b'}]}], 'either text or replacements'),
            ([{'op': 'update', 'line': 2, 'text': '  '}], 'non-empty'),
            ([{'op': 'update', 'line': 2, 'text': 'a'}, {'op': 'delete', 'line': 2}], 'already changed by another edit'),
            ([{'op': 'rewrite', 'line': 2, 'to_line': 1, 'lines': [{'text': 'x'}]}], 'not be before'),
            ([{'op': 'rewrite', 'line': 1, 'lines': [{'text': 'x', 'start': 0.0, 'end': 0.5}, {'text': 'y'}]}], 'on every new line, or on none'),
            ([{'op': 'insert', 'lines': [{'text': 'x'}]}], 'need start and end'),
            ([{'op': 'insert', 'lines': [{'text': 'x', 'start': 1.0, 'end': 0.5}]}], 'empty or outside'),
            ([{'op': 'insert', 'lines': [{'text': 'x', 'start': 0.0, 'end': 30}]}], 'empty or outside'),
            ([{'op': 'merge'}], 'needs op'),
            ([], 'at least one edit'),
        ]
        for edits, message in cases:
            value, ok = self.propose(edits)
            self.assertFalse(ok, edits)
            self.assertIn(message, value['error'], edits)
        self.assertEqual(db.one('SELECT COUNT(*) AS n FROM assistant_proposals')['n'], 0)

    def test_same_voice_label_lines_are_reported_not_changed(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES ('w3','e',1.85,1.95,'Right.','SPEAKER_02','manual')")
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'update', 'line': 2, 'character': 'Mary'}])
        self.assertTrue(ok, value)
        self.assertEqual(value['unlinked_lines_with_same_voice_label'], [3])

    def test_a_later_proposal_in_the_same_turn_replaces_the_earlier_one(self):
        self.ctx.session_id = self.session()
        with db.connect() as conn:
            conn.execute("INSERT INTO assistant_turns(id,session_id,status,provider,model) VALUES ('turn',?,'running','fake','m')", (self.ctx.session_id,))
        self.ctx.turn_id = 'turn'
        first, _ = self.propose([{'op': 'update', 'line': 2, 'character': 'Mary'}])
        second, _ = self.propose([{'op': 'update', 'line': 2, 'character': 'Mary'}, {'op': 'update', 'line': 1, 'text': 'こんばんは'}])
        self.assertTrue(second['replaced_earlier_unsaved_proposal'])
        statuses = {row['id']: row['status'] for row in db.all_rows('SELECT id,status FROM assistant_proposals')}
        self.assertEqual(statuses, {first['proposal_id']: 'discarded', second['proposal_id']: 'pending'})

    def test_planner_handles_a_cut_without_transcript(self):
        cut = {'episode_id': 'e', 'start_time': 2.0, 'end_time': 3.0}
        with self.assertRaises(transcript_edit.PlanError) as raised:
            transcript_edit.build_plan(cut, {'duration': 8}, [], [], [{'op': 'update', 'line': 1, 'text': 'x'}])
        self.assertIn('no transcript lines', str(raised.exception))
        plan = transcript_edit.build_plan(cut, {'duration': 8}, [], [{'id': 'john', 'title': 'John'}], [{'op': 'insert', 'lines': [{'character': 'John', 'text': 'Hi', 'start': 0.2, 'end': 0.8}]}])
        self.assertEqual(plan['before'], '(no transcript)')
        self.assertEqual(plan['after'], '0.20–0.80  John: Hi')


class ApplyTest(TranscriptBase):
    def test_apply_writes_assistant_edit_revisions_and_keeps_history(self):
        session_id = self.session()
        self.ctx.session_id = session_id
        value, ok = self.propose([{'op': 'update', 'line': 2, 'text': 'Hello, Maria!', 'character': 'Mary'}, {'op': 'update', 'line': 1, 'clear_character': True}], 'Fix both')
        self.assertTrue(ok, value)
        card = self.client.get(f'/api/assistant/sessions/{session_id}').json()['proposals'][0]
        self.assertEqual((card['target'], card['label'], card['status'], card['stale']), ('transcript', 'transcript', 'pending', False))
        self.assertNotIn('payload_json', card)
        self.assertTrue(any(part['op'] == 'insert' and 'Maria' in part['text'] for part in card['diff']))
        applied = self.apply(value['proposal_id'])
        self.assertEqual(applied.status_code, 200, applied.text)
        self.assertEqual((applied.json()['status'], applied.json()['applied_result_id']), ('applied', None))
        current = {segment['id']: segment for segment in self.current()}
        self.assertEqual(len(current), 2)
        new_line = next(segment for segment in current.values() if segment['text'] == 'Hello, Maria!')
        self.assertEqual((new_line['start_time'], new_line['end_time'], new_line['character_context_id'], new_line['origin'], new_line['parent_id']), (1.3, 1.8, 'mary', 'assistant_edit', 'w1'))
        cleared = next(segment for segment in current.values() if segment['text'] == 'こんにちは')
        self.assertEqual((cleared['character_context_id'], cleared['speaker_label'], cleared['origin'], cleared['parent_id']), (None, 'SPEAKER_01', 'assistant_edit', 't1'))
        # The old fragments remain as superseded history; the extra fragment has a deletion revision.
        history = {segment['id']: segment for segment in self.client.get('/api/episodes/e/transcript?history=true').json()}
        self.assertEqual(history['w1']['superseded_by'], new_line['id'])
        tombstone = history[history['w2']['superseded_by']]
        self.assertEqual((tombstone['deleted'], tombstone['parent_id'], tombstone['origin']), (True, 'w2', 'assistant_edit'))
        # The same words now read as one linked line through the assistant tools.
        lines, _, _ = self.run_tool('get_transcript', {})
        self.assertEqual([(item['text'], item.get('character')) for item in lines['segments']], [('こんにちは', None), ('Hello, Maria!', 'Mary')])
        saved = json.loads(db.one('SELECT payload_json FROM assistant_proposals WHERE id=?', (value['proposal_id'],))['payload_json'])
        self.assertEqual(len(saved['applied_segment_ids']), 3)
        self.assertEqual(self.apply(value['proposal_id']).status_code, 409)

    def test_rewrite_and_insert_apply(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'rewrite', 'line': 1, 'to_line': 2, 'lines': [{'character': 'John', 'text': 'Where were you?'}, {'character': 'Mary', 'text': 'Here.'}]},
                                  {'op': 'insert', 'lines': [{'speaker_label': 'Narrator', 'text': 'Later.', 'start': 1.4, 'end': 1.5}]}])
        self.assertTrue(ok, value)
        self.assertEqual(self.apply(value['proposal_id']).status_code, 200)
        texts = [(segment['text'], segment['character_context_id'], segment['speaker_label'], segment['origin']) for segment in self.current()]
        self.assertEqual(texts, [('Where were you?', 'john', 'John', 'assistant_edit'), ('Here.', 'mary', 'Mary', 'assistant_edit'), ('Later.', None, 'Narrator', 'assistant_edit')])

    def test_apply_is_refused_when_the_transcript_changed_after_proposing(self):
        session_id = self.session()
        self.ctx.session_id = session_id
        value, ok = self.propose([{'op': 'update', 'line': 2, 'character': 'Mary'}])
        self.assertTrue(ok, value)
        self.client.put('/api/transcript-segments/w1', json={'start_time': 1.3, 'end_time': 1.5, 'text': 'Hullo', 'speaker_label': 'SPEAKER_02'})
        card = self.client.get(f'/api/assistant/sessions/{session_id}').json()['proposals'][0]
        self.assertTrue(card['stale'])
        refused = self.apply(value['proposal_id'])
        self.assertEqual(refused.status_code, 409)
        self.assertIn('transcript changed', refused.json()['detail'])
        self.assertEqual(db.one("SELECT COUNT(*) AS n FROM transcript_segments WHERE origin='assistant_edit'")['n'], 0)

    def test_apply_is_refused_when_a_character_was_deleted(self):
        self.ctx.session_id = self.session()
        value, ok = self.propose([{'op': 'update', 'line': 2, 'character': 'Mary'}])
        self.assertTrue(ok, value)
        with db.connect() as conn:
            conn.execute("DELETE FROM context_items WHERE id='mary'")
        refused = self.apply(value['proposal_id'])
        self.assertEqual(refused.status_code, 409)
        self.assertEqual(db.one("SELECT status FROM assistant_proposals WHERE id=?", (value['proposal_id'],))['status'], 'pending')

    def test_discarding_leaves_the_transcript_alone(self):
        self.ctx.session_id = self.session()
        value, _ = self.propose([{'op': 'delete', 'line': 1}])
        discarded = self.client.post(f"/api/assistant/proposals/{value['proposal_id']}/discard")
        self.assertEqual(discarded.json()['status'], 'discarded')
        self.assertEqual(len(self.current()), 3)


class ChatFlowTest(TranscriptBase):
    def test_skill_and_tool_are_offered(self):
        skill = skills.get('fix_transcript')
        self.assertIn('propose_transcript_edit', skill['body'])
        self.assertIn('fix_transcript', skills.index_text())
        self.assertIn('propose_transcript_edit', tools.REGISTRY)
        self.assertTrue(any(item['name'] == 'fix_transcript' for item in self.client.get('/api/assistant/config').json()['skills']))

    def test_user_dictates_the_dialogue_in_chat(self):
        session_id = self.session()
        self.worker_online()
        message = 'Fix the transcription, John says: Where were you? Maria says: Right here.'
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': message, 'provider': 'fake', 'model': 'fake-model', 'skill': 'fix_transcript'}).json()['turn']
        step, offered = [], []
        proposal = {'summary': 'John and Maria', 'edits': [{'op': 'rewrite', 'line': 1, 'to_line': 2, 'lines': [{'character': 'John', 'text': 'Where were you?'}, {'character': 'Mary', 'text': 'Right here.'}]}]}
        script = [('get_transcript', {}), ('get_characters', {}), ('propose_transcript_edit', proposal)]

        def fake_stream(endpoint, messages, tool_schemas, *, on_delta, checkpoint, **kwargs):
            step.append(messages)
            offered.append(tool_schemas)
            checkpoint()
            if len(step) <= len(script):
                name, arguments = script[len(step) - 1]
                return providers.StepResult(tool_calls=[{'id': f'c{len(step)}', 'name': name, 'arguments': json.dumps(arguments)}], finish_reason='tool_calls')
            on_delta('Proposed the new dialogue; apply it on the card.', '')
            return providers.StepResult(content='Proposed the new dialogue; apply it on the card.', finish_reason='stop')

        with patch.object(providers, 'registry', return_value={'fake': assistant_tests.FakeProvider()}), patch.object(providers, 'stream_chat', side_effect=fake_stream):
            runner.run_turn(*runner.claim_turn())
        detail = self.client.get(f'/api/assistant/sessions/{session_id}').json()
        self.assertEqual(detail['turns'][0]['status'], 'done', detail['turns'][0]['error'])
        self.assertTrue(offered and all(any(schema['function']['name'] == 'propose_transcript_edit' for schema in schemas) for schemas in offered))
        self.assertIn('propose_transcript_edit', step[0][0]['content'])
        self.assertEqual(len(detail['proposals']), 1)
        card = detail['proposals'][0]
        self.assertEqual((card['target'], card['status'], card['turn_id']), ('transcript', 'pending', turn['id']))
        self.assertIn('John: Where were you?', card['content'])
        self.assertIn('Mary: Right here.', card['content'])
        self.assertEqual(self.apply(card['id']).status_code, 200)
        self.assertEqual([(segment['text'], segment['character_context_id']) for segment in self.current()], [('Where were you?', 'john'), ('Right here.', 'mary')])


if __name__ == '__main__':
    unittest.main()
