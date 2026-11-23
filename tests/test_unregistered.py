"""Unregistered cards (unregistered.py) and one question per unknown voice (HARNESS.md D29, D30)."""
import json
import time
import unittest
from unittest import mock

from app import anime_gallery, cut_needs, cut_readiness, db, keyframe_prompts, review_report, speaker_attribution, speaker_prompt, \
    sequence_package, sequence_prompt, unregistered, visual_contracts, voice_resolver as vr
from tests import test_voice_resolver as fixture


class PureTest(unittest.TestCase):
    def test_speaker_open_is_one_rule(self):
        self.assertTrue(cut_readiness.speaker_open({'character_context_id': None, 'speaker_source': None, 'confidence': None}))
        self.assertFalse(cut_readiness.speaker_open({'character_context_id': None, 'speaker_source': 'left_unknown', 'confidence': None}))
        self.assertTrue(cut_readiness.speaker_open({'character_context_id': 'mai', 'speaker_source': 'qwen', 'confidence': 'low'}))
        self.assertFalse(cut_readiness.speaker_open({'character_context_id': 'mai', 'speaker_source': 'left_unknown', 'confidence': 'low'}))
        self.assertFalse(cut_readiness.speaker_open({'character_context_id': 'mai', 'speaker_source': 'qwen', 'confidence': 'medium'}))
        self.assertFalse(speaker_attribution.model_line({'origin': 'model', 'speaker_source': 'left_unknown', 'words_json': '[]'}))

    def test_prompts_name_the_rule_only_when_a_card_is_there(self):
        self.assertNotIn(unregistered.RULE, visual_contracts.person_context([{'ref': 'B1', 'name': 'Mai'}], 'first frame'))
        self.assertIn(unregistered.RULE, visual_contracts.person_context([{'ref': 'B1', 'name': 'Mai'}, {'ref': 'B2', 'name': 'Unregistered 1'}], 'first frame'))
        roster = {'roster': [{'name': 'Mai', 'box': [0, 0, 1, 1], 'side': 'left'}]}
        self.assertNotIn(unregistered.RULE, keyframe_prompts.roster_lines(roster))
        roster['roster'].append({'name': 'Unregistered 2', 'box': [5, 0, 9, 9], 'side': 'right'})
        self.assertIn(unregistered.RULE, keyframe_prompts.roster_lines(roster))
        self.assertEqual(speaker_prompt.system_open(False), speaker_prompt.SYSTEM_OPEN)
        self.assertIn('Unregistered N (unregistered)', speaker_prompt.system_open(True))

    def test_the_consistency_review_is_told_only_when_a_card_speaks_or_shows(self):
        from app import consistency_summary
        line = {'start_time': 0.0, 'end_time': 1.0, 'text': 'はい', 'speaker_label': 'SPEAKER_00', 'character_context_id': 'u2', 'character_name': 'Unregistered 2'}
        evidence = consistency_summary.build_input([], [line], [])
        self.assertEqual(evidence['unregistered_people']['names'], ['Unregistered 2'])
        self.assertNotIn('unregistered_people', consistency_summary.build_input([], [{**line, 'character_name': 'Mai', 'character_context_id': 'mai'}], []))

    def test_a_card_on_the_shot_roster_is_a_person_without_a_sheet(self):
        roster = [{'name': 'Mai', 'character_context_id': 'mai'}, {'name': 'Unregistered 1', 'character_context_id': 'u1', 'unregistered': True}]
        subjects, pictures = sequence_prompt.assign({'subjects': [{'name': 'Mai'}, {'name': 'Unregistered 1'}]}, roster)
        self.assertEqual([(item['kind'], item['picture'], item.get('unregistered')) for item in subjects], [('character', 2, None), ('person', None, True)])
        self.assertEqual([item['role'] for item in pictures], ['initial_frame', 'character_sheet', 'original_audio'])
        linked = sequence_package.link_subjects(subjects, roster, {})
        self.assertEqual(linked[1]['kind'], 'unregistered')
        self.assertEqual(sequence_package.speaker_kind('Unregistered 1', 'u1'), 'unregistered')


class CardTest(unittest.TestCase):
    setUp = fixture.JobTest.setUp
    tearDown = fixture.JobTest.tearDown
    speakers = fixture.JobTest.speakers
    run_job = fixture.JobTest.run_job

    def card(self):
        response = self.client.post('/api/projects/p/unregistered-characters')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_cards_are_numbered_and_fixed(self):
        first, second = self.card(), self.card()
        self.assertEqual((first['title'], second['title'], first['unregistered']), ('Unregistered 1', 'Unregistered 2', 1))
        self.assertEqual(self.client.put(f"/api/context/{first['id']}", json={'kind': 'character', 'title': 'Waiter', 'content': ''}).status_code, 400)
        self.assertEqual(self.client.post('/api/projects/p/context', json={'kind': 'character', 'title': 'Unregistered 3'}).status_code, 400)
        self.client.delete(f"/api/context/{first['id']}")
        self.assertEqual(self.card()['title'], 'Unregistered 1')  # the lowest free number

    def test_a_card_is_never_a_recognition_example_or_a_voice(self):
        card = self.card()
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES ('r1','c1',0,10,'[]','t','complete')")
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source)
                VALUES ('d1','r1','first','[0,0,10,10]',?,'manual'),('d2','r1','first','[20,0,30,10]','mai','manual')""", (card['id'],))
            examples = {row['character_context_id'] for row in anime_gallery.exemplar_rows('p', conn)}
            _, names = anime_gallery.evidence(conn, 'p')
        self.assertEqual(examples, {'mai'})
        self.assertNotIn(card['id'], names)
        self.assertEqual(self.client.post('/api/character-detections/d1/reference').status_code, 400)
        upload = self.client.post('/api/episodes/e/voice-samples', json={'character_context_id': card['id'], 'start_time': 1, 'end_time': 4})
        self.assertIn(upload.status_code, (400, 422))

    def test_leave_settles_the_voice_everywhere_and_is_asked_once(self):
        self.run_job()
        questions = {item['group_key']: item for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions']}
        answered = self.client.post(f"/api/escalations/{questions['voice:SPEAKER_07']['id']}/voice", json={'choice': 'leave'}).json()
        self.assertEqual((answered['status'], answered['lines']), ('left_unknown', 1))
        self.assertEqual(self.speakers()['n1'], (None, 'left_unknown', 'manual_edit'))
        with db.connect() as conn:
            report = review_report.report(conn, 'e')
            needs = cut_needs.needs(conn, 'e')['cuts']
            readiness = cut_readiness.cut(conn, 'c4')
        self.assertNotIn('SPEAKER_07', [voice['speaker_label'] for voice in report['gates']['voices']])
        voice_need = next((item['count'] for item in needs['c4']['people'] if item['key'] == 'voice'), 0)
        self.assertEqual(voice_need, 0)  # n1 is settled; k1 (SPEAKER_01) waits for Claude, which covers every cut its lines touch
        self.assertFalse(any('32.0' in reason for reason in readiness['held']['beats']))
        self.run_job()
        open_keys = {item['group_key'] for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions']}
        self.assertNotIn('voice:SPEAKER_07', open_keys)

    def test_an_old_dismissed_voice_is_settled_at_the_next_run(self):
        self.run_job()
        with db.connect() as conn:
            conn.execute("UPDATE agent_escalations SET status='dismissed' WHERE group_key='voice:SPEAKER_07'")
        self.run_job()
        self.assertEqual(self.speakers()['n1'][1], 'left_unknown')

    def test_a_low_confidence_guess_is_asked_once_with_the_guess_as_a_candidate(self):
        with db.connect() as conn:
            conn.execute("UPDATE transcript_segments SET character_context_id='tomoe',speaker_source='qwen',confidence='low' WHERE id='n1'")
        self.run_job()
        questions = {item['group_key']: item for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions']}
        item = questions['voice:SPEAKER_07']
        self.assertIn('tomoe', [candidate['id'] for candidate in item['evidence']['candidates']])
        self.assertTrue(item['evidence']['unsure'])
        self.assertIn('Keep as it is', item['options'])
        self.client.post(f"/api/escalations/{item['id']}/voice", json={'choice': 'leave'})
        self.assertEqual(self.speakers()['n1'], ('tomoe', 'left_unknown', 'manual_edit'))  # the guess stays, never asked again

    def test_qwen_other_goes_to_the_only_card_boxed_in_the_cut(self):
        card = self.card()
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES ('r4','c4',30,40,'[]','t','complete')")
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES ('d4','r4','first','[0,0,9,9]',?,'manual')",
                         (card['id'],))
            conn.execute("""UPDATE transcript_segments SET speaker_source='qwen',confidence='low',
                evidence_json='{"qwen": {"speaker": "other", "reason": "on_screen"}}' WHERE id='n1'""")
        self.run_job()
        self.assertEqual(self.speakers()['n1'][:2], (card['id'], 'agent'))
        with db.connect() as conn:
            decision = conn.execute("SELECT * FROM agent_decisions WHERE kind='voices' AND target_id='n1'").fetchone()
        self.assertEqual(decision['tier'], 'rules')
        self.assertEqual(self.client.post(f"/api/agent-decisions/{decision['id']}/undo").json(), {'undone': True})
        self.assertEqual(self.speakers()['n1'][0], None)

    def test_two_cards_in_the_cut_are_asked_with_the_cards_offered(self):
        first, second = self.card(), self.card()
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES ('r4','c4',30,40,'[]','t','complete')")
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source)
                VALUES ('d4','r4','first','[0,0,9,9]',?,'manual'),('d5','r4','first','[10,0,19,9]',?,'manual')""", (first['id'], second['id']))
            conn.execute("""UPDATE transcript_segments SET evidence_json='{"qwen": {"speaker": "other"}}' WHERE id='n1'""")
        self.run_job()
        self.assertIsNone(self.speakers()['n1'][0])
        question = next(item for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions'] if item['group_key'] == 'voice:SPEAKER_07')
        self.assertEqual([card['name'] for card in question['evidence']['cards']], ['Unregistered 1', 'Unregistered 2'])
        # Claude never numbers an Unregistered person: the person does.
        with db.connect() as conn:
            conn.execute("UPDATE agent_escalations SET reviewer='claude' WHERE id=?", (question['id'],))
        handed = self.client.post(f"/api/escalations/{question['id']}/voice", json={'choice': 'character', 'character_context_id': second['id'], 'tier': 'claude'})
        self.assertEqual(handed.json()['status'], 'handed_to_person')
        answered = self.client.post(f"/api/escalations/{question['id']}/voice", json={'choice': 'unregistered', 'character_context_id': second['id']}).json()
        self.assertEqual(answered['status'], 'resolved')
        self.assertEqual(self.speakers()['n1'], (second['id'], 'manual', 'manual_edit'))

    def test_unregistered_without_a_number_makes_the_next_card(self):
        self.run_job()
        question = next(item for item in self.client.get('/api/episodes/e/voice-decisions').json()['questions'] if item['group_key'] == 'voice:SPEAKER_07')
        self.assertEqual(self.client.post(f"/api/escalations/{question['id']}/voice", json={'choice': 'unregistered'}).json()['status'], 'resolved')
        with db.connect() as conn:
            title = conn.execute('SELECT c.title FROM transcript_segments t JOIN context_items c ON c.id=t.character_context_id WHERE t.id IN '
                                 "(SELECT id FROM transcript_segments WHERE superseded_by IS NULL AND start_time=32.0)").fetchone()[0]
        self.assertEqual(title, 'Unregistered 1')

    def test_the_line_editor_can_leave_a_speaker_for_good(self):
        response = self.client.put('/api/transcript-segments/n1', json={'start_time': 32.0, 'end_time': 33.0, 'text': 'ね', 'speaker_label': 'SPEAKER_07',
                                                                       'character_context_id': None, 'left_unknown': True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['speaker_source'], 'left_unknown')
        with db.connect() as conn:
            self.assertNotIn('SPEAKER_07', [voice['speaker_label'] for voice in review_report.report(conn, 'e')['gates']['voices']])

    def test_attribution_never_offers_a_card_by_name(self):
        card = self.card()
        with db.connect() as conn:
            data = speaker_attribution.gather(conn, {'episode_id': 'e'})
        names = {row['title'] for row in data['characters'] if not row.get('unregistered')}
        self.assertNotIn(card['title'], names)
        self.assertIn(card['title'], {row['title'] for row in data['characters']})


if __name__ == '__main__':
    unittest.main()
