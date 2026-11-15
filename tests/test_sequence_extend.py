"""Adding the next cut to a sequence: the range grows, only the added cut's dialogue is compiled, and the direction follows."""
import json

from app import db, sequence_direction as direction
from tests.test_sequence_pipeline import BIBLE, SequenceBase


class SequenceExtendTest(SequenceBase):
    def setUp(self):
        super().setUp()
        # The bible's scenes must cover exactly the cuts of the sequence, which grows during these tests.
        self.answers['bible'] = lambda data: {**BIBLE, 'scenes': [{**BIBLE['scenes'][0], 'to_cut': self.last_cut()}]}

    def last_cut(self):
        row = db.one("SELECT c.cut_index FROM sequences s JOIN cuts c ON c.id=s.last_cut_id ORDER BY s.rowid DESC LIMIT 1")
        return row['cut_index'] + 1

    def crossing(self):
        """A third voice says ちょっと待って across the hard cut between cut 2 and cut 3 (at 4.0 s): the old end keeps only ちょっと."""
        with db.connect() as conn:
            conn.executemany("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES (?,'e',?,?,?,'SPEAKER_03','model')",
                             [('x1', 3.6, 3.9, 'ちょっと'), ('x2', 4.0, 4.6, '待って')])

    def extend(self, sequence_id, expected=200):
        response = self.client.post(f'/api/sequences/{sequence_id}/extend')
        self.assertEqual(response.status_code, expected, response.text)
        return response.json()

    def confirm(self, sequence_id):
        result = self.detail(sequence_id)['compilation']['result']
        response = self.client.post(f'/api/sequences/{sequence_id}/compilation/confirm', json={'result_id': result['id']})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def steps(self, sequence_id):
        return [row['step'] for row in db.all_rows("SELECT json_extract(payload_json,'$.step') AS step FROM jobs WHERE pipeline='sequence' AND json_extract(payload_json,'$.sequence_id')=? ORDER BY rowid", (sequence_id,))]

    def test_adding_the_next_cut_keeps_the_edits_and_rewrites_the_whole_chain(self):
        self.crossing()
        sequence_id = self.create(1, 2, review=None)
        self.work()
        result = self.detail(sequence_id)['compilation']['result']
        self.assertEqual([row['text'] for row in result['content']['lines']], ['こんにちは', 'はい', 'ちょっと'])
        edited = {**result['content'], 'lines': [{**row, 'text': 'はいはい', 'parts': []} if row['text'] == 'はい' else row for row in result['content']['lines']]}
        self.assertEqual(self.client.put(f"/api/sequence-results/{result['id']}", json={'content': edited}).status_code, 200)
        self.confirm(sequence_id)
        self.run_step(sequence_id, 'direct')
        self.work()
        self.run_step(sequence_id, 'shots')
        self.work()
        before = self.detail(sequence_id)
        self.assertTrue(before['ready'])
        self.assertEqual((before['next_cut']['number'], before['next_cut']['fits'], before['next_cut']['ready'], before['next_cut']['speech']),
                         (3, True, True, {'text_before': 'ちょっと', 'words_after': '待って'}))

        detail = self.extend(sequence_id)
        info = detail['sequence']
        self.assertEqual((info['first_cut'], info['last_cut'], info['title'], info['cuts_changed']), (1, 3, 'Cuts 1–3', False))
        self.assertEqual(info['extension'], {'cut_ids': ['c3'], 'cuts': [3], 'from_time': 4.0, 'rewrite_shots': True})
        self.assertEqual((detail['compilation']['state'], detail['next_cut']), ('running', None))  # cut 3 is the last cut of the episode
        self.extend(sequence_id, expected=409)  # its dialogue is being compiled

        compiled = len([step for step, _ in self.calls if step == 'compile'])
        self.assertEqual(self.work(), 1)  # only the compilation of the added cut; the direction waits for the review
        call = [data for step, data in self.calls if step == 'compile'][compiled:]
        self.assertEqual(len(call), 1)
        self.assertNotIn('こんにちは', call[0]['lines'])  # only the added cut is sent
        self.assertIn('待って', call[0]['lines'])
        self.assertIn('はいはい', json.dumps(call[0]['previous'], ensure_ascii=False))  # the user's last lines are its context
        detail = self.detail(sequence_id)
        lines = detail['compilation']['result']['content']['lines']
        self.assertEqual([(row['text'], row['continues_next']) for row in lines],
                         [('こんにちは', False), ('はいはい', False), ('ちょっと', True), ('待って', False), ('じゃあね', False)])
        self.assertEqual((detail['compilation']['state'], detail['compilation']['stale']), ('review', False))
        self.run_step(sequence_id, 'direct', expected=409)  # the new lines are not reviewed yet

        confirmed = self.confirm(sequence_id)
        self.assertIsNone(confirmed['sequence']['extension'])
        self.assertEqual([job['step'] for job in confirmed['jobs'] if job['status'] == 'queued'], ['bible'])
        self.work()
        after = self.detail(sequence_id)
        self.assertEqual(self.steps(sequence_id)[-6:], ['bible', 'plan', 'shot', 'shot', 'shot', 'review'])
        self.assertEqual([shot['cut'] for shot in after['shots']], [1, 2, 3])
        self.assertTrue(after['ready'])
        self.assertTrue(all(shot['result'] and not shot['stale'] for shot in after['shots']))
        self.assertIsNotNone(after['review'])
        self.assertFalse([job for job in after['jobs'] if job['status'] in ('failed', 'cancelled')])
        boundary = after['timeline']['boundaries'][-1]
        self.assertEqual(boundary['after_cut'], 2)
        self.assertTrue(boundary['speech'])  # the director was asked about the phrase the old end cut in half

    def test_an_added_cut_without_speech_redirects_at_once(self):
        with db.connect() as conn:
            conn.execute("DELETE FROM transcript_segments WHERE id='t3'")
        sequence_id = self.create(1, 2)
        self.run_step(sequence_id, 'direct')
        self.work()
        compilation = self.detail(sequence_id)['compilation']['result']['id']
        self.extend(sequence_id)
        self.work()
        detail = self.detail(sequence_id)
        self.assertEqual((detail['compilation']['result']['id'], detail['compilation']['state'], detail['compilation']['stale']), (compilation, 'confirmed', False))
        self.assertIsNone(detail['sequence']['extension'])
        self.assertEqual(self.steps(sequence_id)[-3:], ['compile', 'bible', 'plan'])  # no shot prompts existed, so the run stops after the plan as usual
        self.assertEqual([shot['cut'] for shot in detail['shots']], [1, 2, 3])
        self.assertFalse(any(shot['result'] for shot in detail['shots']))

    def test_a_save_made_while_the_added_cut_compiles_is_kept(self):
        sequence_id = self.create(1, 2)
        self.extend(sequence_id)

        def meanwhile(data):
            result = self.detail(sequence_id)['compilation']['result']
            edited = {**result['content'], 'lines': [{**row, 'text': 'こんばんは', 'parts': []} if row['id'] == 'C1' else row for row in result['content']['lines']]}
            self.assertEqual(self.client.put(f"/api/sequence-results/{result['id']}", json={'content': edited}).status_code, 200)
            del self.answers['compile']
            return self.answer(direction.COMPILE_SYSTEM, data)
        self.answers['compile'] = meanwhile
        self.work()
        detail = self.detail(sequence_id)
        self.assertEqual([row['text'] for row in detail['compilation']['result']['content']['lines']], ['こんばんは', 'はい', 'じゃあね'])
        self.assertEqual(detail['compilation']['state'], 'review')
        self.assertTrue(detail['sequence']['extension'])

    def test_refusals(self):
        busy = self.create(1, 2, review=None)  # its compilation is still queued
        self.assertIn('jobs queued or running', self.extend(busy, expected=409)['detail'])
        whole = self.create(1, 3)
        self.assertEqual(self.extend(whole, expected=409)['detail'], 'Cut 3 is the last cut of the episode')
        short = self.create(1, 2)
        with db.connect() as conn:
            conn.execute("DELETE FROM analysis_results WHERE cut_id='c3'")
        self.assertEqual(self.extend(short, expected=409)['detail'], 'Cut 3 needs a first-frame and a video analysis first; run them from the cut, then add it')
        self.assertFalse(self.detail(short)['next_cut']['ready'])
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=200 WHERE id='c3'")
            conn.execute("UPDATE episodes SET duration=200")
        self.assertIn('would last 200.0 s', self.extend(short, expected=422)['detail'])
        self.assertFalse(self.detail(short)['next_cut']['fits'])
        self.assertEqual(self.detail(short)['sequence']['last_cut'], 2)
