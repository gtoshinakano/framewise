"""Dialogue compilation: the pure alignment arithmetic, the compile job, the review gate and what Direction takes from it."""
import json
import unittest
from unittest.mock import patch

from app import db, sequence_compile as compile_math, sequence_direction as direction, worker
from tests.test_sequence_pipeline import SequenceBase


def line(index, t0, t1, text, label='SPEAKER_01', character=None):
    return {'id': f'L{index}', 't0': t0, 't1': t1, 'text': text, 'speaker_label': label, 'character_name': character, 'character_context_id': 'ctx' if character else None,
            'parts': [{'id': f't{index}', 't0': t0, 't1': t1, 'text': text}]}


class ArithmeticTest(unittest.TestCase):
    def test_pairing_scores_overlap_and_text_similarity(self):
        lines = [line(1, 1.0, 3.0, '今日はいい天気ですね')]
        subs = [{'id': 'U1', 't0': 0.8, 't1': 3.2, 'text': '今日はいい天気ですね。'}, {'id': 'U2', 't0': 9.0, 't1': 10.0, 'text': '別の台詞'}]
        pairs = compile_math.pair_subtitles(lines, subs)
        self.assertEqual([item['subtitle'] for item in pairs['L1']], ['U1'])
        self.assertGreater(pairs['L1'][0]['similarity'], 0.9)

    def test_hints_flag_a_sentence_cut_in_half_and_ignore_other_speakers(self):
        lines = [line(1, 0.0, 1.0, '私は昨日'), line(2, 1.2, 2.5, 'に行きました。'), line(3, 2.6, 3.4, 'そうですか。', label='SPEAKER_02')]
        hints = compile_math.continuity_hints(lines, [], [])
        self.assertEqual([(hint['a'], hint['b']) for hint in hints], [('L1', 'L2')])
        self.assertTrue(any('sentence ending' in reason for reason in hints[0]['reasons']))

    def test_retime_spreads_words_over_the_line_and_rebuilds_the_text(self):
        parts = compile_math.retime_parts('おはようございます', 2.0, 5.0)
        self.assertEqual((parts[0]['t0'], parts[-1]['t1']), (2.0, 5.0))
        self.assertEqual(compile_math.parts_text(parts), 'おはようございます')
        self.assertTrue(all(a['t1'] == b['t0'] for a, b in zip(parts, parts[1:])))

    def test_snap_moves_edges_onto_real_edges_and_separates_one_voice(self):
        lines = [{'t0': 1.1, 't1': 2.6, 'speaker_label': 'A'}, {'t0': 2.4, 't1': 4.0, 'speaker_label': 'A'}]
        out = compile_math.snap(lines, [1.0, 2.5, 4.05], 10.0)
        self.assertEqual((out[0]['t0'], out[1]['t1']), (1.0, 4.05))
        self.assertLessEqual(out[0]['t1'] - out[1]['t0'], compile_math.SAME_SPEAKER_OVERLAP)

    def test_validate_checks_names_and_times_and_keeps_no_delivery(self):
        # A line of an older compilation may still carry a delivery: how a line is played is decided per shot now.
        good = {'id': 'x', 't0': 0.5, 't1': 2.0, 'text': ' こんにちは ', 'speaker_label': 'SPEAKER_01', 'character': 'hero', 'delivery': 'lip_sync', 'subject': 'Hero'}
        content = compile_math.validate({'lines': [good]}, 10.0, {'Hero': 'ctx'})
        row = content['lines'][0]
        self.assertEqual((row['id'], row['text'], row['character'], row['character_context_id']), ('C1', 'こんにちは', 'Hero', 'ctx'))
        self.assertNotIn('delivery', row)
        self.assertNotIn('subject', row)
        self.assertEqual(content['voices'][0]['speaker_label'], 'SPEAKER_01')
        self.assertEqual(compile_math.validate({'lines': [{**good, 'character': None}]}, 10.0, {})['lines'][0]['character'], None)
        self.assertFalse(compile_math.needs_review({**row, 'confidence': 'high'}))
        for bad, message in (({**good, 'character': 'Nobody'}, 'not a character'), ({**good, 't1': 11.0}, '0 <= t0 < t1'), ({**good, 'text': ''}, 'no text')):
            with self.assertRaisesRegex(ValueError, message):
                compile_math.validate({'lines': [bad]}, 10.0, {'Hero': 'ctx'})

    def test_changed_text_is_retimed_and_unchanged_text_keeps_its_words(self):
        parts = [{'t0': 1.0, 't1': 1.5, 'text': 'こん'}, {'t0': 1.5, 't1': 2.0, 'text': 'にちは'}]
        kept = compile_math.validate({'lines': [{'t0': 1.0, 't1': 2.0, 'text': 'こんにちは', 'parts': parts}]}, 5.0, {})['lines'][0]
        self.assertEqual([part['text'] for part in kept['parts']], ['こん', 'にちは'])
        moved = compile_math.validate({'lines': [{'t0': 2.0, 't1': 4.0, 'text': 'こんにちは', 'parts': parts}]}, 5.0, {})['lines'][0]
        self.assertEqual((moved['parts'][0]['t0'], moved['parts'][-1]['t1']), (2.0, 4.0))
        corrected = compile_math.validate({'lines': [{'t0': 1.0, 't1': 2.0, 'text': '今日は', 'parts': parts}]}, 5.0, {})['lines'][0]
        self.assertEqual(compile_math.parts_text(corrected['parts']), '今日は')

    def test_tail_keeps_the_words_after_the_old_end_by_their_midpoint(self):
        crossing = {**line(2, 3.6, 4.6, 'ちょっと待って', 'SPEAKER_03'), 'parts': [{'id': 'a', 't0': 3.6, 't1': 3.9, 'text': 'ちょっと'}, {'id': 'b', 't0': 3.95, 't1': 4.6, 'text': '待って'}]}
        tail, crossed = compile_math.tail_lines([line(1, 0.4, 1.2, 'こんにちは'), crossing, line(3, 5.0, 6.0, 'じゃあね')], 4.0)
        self.assertEqual([(item['id'], item['t0'], item['t1'], item['text']) for item in tail], [('L2', 3.95, 4.6, '待って'), ('L3', 5.0, 6.0, 'じゃあね')])
        self.assertEqual(crossed, {'a'})  # the word the old range kept; '待って' starts before 4.0 but its middle is after it

    def test_appending_keeps_every_existing_line_and_voice_and_marks_the_phrase_cut_in_half(self):
        existing = {'lines': [{'id': 'C1', 't0': 0.4, 't1': 1.2, 'text': '直した', 'source_ids': ['t1'], 'continues_next': False},
                              {'id': 'C2', 't0': 3.6, 't1': 3.9, 'text': 'ちょっと', 'source_ids': ['a'], 'continues_next': False}],
                    'voices': [{'speaker_label': 'SPEAKER_03', 'character': 'Hero'}], 'open_questions': ['old']}
        added = [{'id': 'C1', 't0': 3.95, 't1': 4.6, 'text': '待って', 'source_ids': ['b']}]
        voices = [{'speaker_label': 'SPEAKER_03', 'character': None}, {'speaker_label': 'SPEAKER_04', 'character': None}]
        merged = compile_math.append_lines(existing, added, voices, ['new'], {'a'})
        self.assertEqual([(item['text'], item['continues_next']) for item in merged['lines'][:2]], [('直した', False), ('ちょっと', True)])
        self.assertEqual(merged['lines'][2]['text'], '待って')
        self.assertEqual([(voice['speaker_label'], voice['character']) for voice in merged['voices']], [('SPEAKER_03', 'Hero'), ('SPEAKER_04', None)])
        self.assertEqual(merged['open_questions'], ['new', 'old'])
        self.assertFalse(existing['lines'][1]['continues_next'])  # the stored revision is not changed in place


class CompileFlowTest(SequenceBase):
    def compile_answer(self, data):
        """Joins nothing, corrects line 3 from its subtitle and decides who speaks."""
        rows = [item for item in data['lines'].splitlines()]
        out = []
        for row in rows:
            line_id = row.split()[0]
            text = row.split(': ', 1)[1]
            out.append({'ids': [line_id], 'text': '字幕の言葉' if line_id == 'L3' else text, 'speaker': 'Hero' if line_id != 'L2' else None,
                        'subtitle_ids': ['U1'] if line_id == 'L3' and 'U1' in data['subtitles'] else [],
                        'confidence': 'high', 'note': '', 'continues_next': False})
        return {'lines': out, 'voices': [{'speaker_label': 'SPEAKER_02', 'character': 'Hero', 'kind': 'character', 'confidence': 'medium', 'reason': 'visibly speaks'}], 'open_questions': []}

    def test_creating_a_sequence_compiles_the_dialogue_and_direction_waits_for_the_review(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO subtitle_runs(id,episode_id,status,language) VALUES ('r','e','complete','Japanese')")
            conn.execute("INSERT INTO subtitle_lines(id,run_id,episode_id,line_index,start_time,end_time,text,confidence) VALUES ('s1','r','e',0,5.0,6.0,'字幕の言葉','high')")
        self.answers['compile'] = self.compile_answer
        sequence_id = self.create(review=None)
        self.assertEqual(self.detail(sequence_id)['compilation']['state'], 'running')
        self.run_step(sequence_id, 'direct', expected=409)  # nothing is reviewed yet
        self.assertEqual(self.work(), 1)
        detail = self.detail(sequence_id)
        compilation = detail['compilation']
        self.assertEqual(compilation['state'], 'review')
        lines = compilation['result']['content']['lines']
        self.assertEqual([(row['text'], row['character']) for row in lines], [('こんにちは', 'Hero'), ('はい', None), ('字幕の言葉', 'Hero')])
        self.assertFalse(any('delivery' in row for row in lines))
        compile_call = next(data for step, data in self.calls if step == 'compile')
        self.assertNotIn('delivery', json.dumps(direction.compile_format(['L1'], [], ['Hero'])))
        self.assertTrue(compile_call['lines'])
        self.assertEqual((lines[2]['original_text'], lines[2]['subtitle_text'], lines[2]['subtitle_ids']), ('じゃあね', '字幕の言葉', ['s1']))
        self.assertEqual([item['text'] for item in compilation['subtitles']], ['字幕の言葉'])
        self.assertEqual(detail['timeline']['lines'][2]['text'], 'じゃあね')  # the raw transcript is still what the timeline uses
        self.run_step(sequence_id, 'direct', expected=409)
        response = self.client.post(f'/api/sequences/{sequence_id}/compilation/confirm', json={'result_id': compilation['result']['id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['compilation']['state'], 'confirmed')
        self.assertEqual(self.detail(sequence_id)['timeline']['lines'][2]['text'], '字幕の言葉')
        self.assertEqual(self.run_step(sequence_id, 'direct')['queued'], 1)
        self.work()
        bible = self.detail(sequence_id)['bible']['content']
        self.assertEqual(sorted((voice['speaker_label'], voice['character'], voice['source']) for voice in bible['voices']), [('SPEAKER_01', 'Hero', 'reviewed'), ('SPEAKER_02', 'Hero', 'reviewed')])
        bible_call = next(data for step, data in self.calls if step == 'bible')
        self.assertIn('voices_are_final', bible_call)
        self.assertTrue(bible_call['dialogue_reviewed'])
        self.assertNotRegex(bible_call['transcript'], r'\[(?:lip_sync|off_screen|inner_voice|narration)\]')

    def test_a_hand_edit_clears_the_confirmation_and_the_reviewed_words_reach_the_prompt(self):
        self.answers['compile'] = self.compile_answer
        sequence_id = self.create(review=None)
        self.work()
        result = self.detail(sequence_id)['compilation']['result']
        edited = {**result['content'], 'lines': [{**row, 'text': 'こんばんは', 'parts': []} if row['id'] == 'C1' else row for row in result['content']['lines']]}
        saved = self.client.put(f"/api/sequence-results/{result['id']}", json={'content': edited})
        self.assertEqual(saved.status_code, 200, saved.text)
        compilation = saved.json()['compilation']
        self.assertEqual((compilation['state'], compilation['result']['origin']), ('review', 'manual_edit'))
        self.assertEqual(self.client.put(f"/api/sequence-results/{result['id']}", json={'content': edited}).status_code, 409)  # superseded
        bad = self.client.put(f"/api/sequence-results/{compilation['result']['id']}", json={'content': {**edited, 'lines': [{**edited['lines'][0], 'character': 'Ghost'}]}})
        self.assertEqual(bad.status_code, 422)
        self.client.post(f'/api/sequences/{sequence_id}/compilation/confirm', json={'result_id': compilation['result']['id']})
        self.run_step(sequence_id, 'direct')
        self.work()
        self.run_step(sequence_id, 'shots')
        self.work()
        first = self.detail(sequence_id)['shots'][0]['result']
        self.assertEqual((first['derived']['dialogue'][0]['text'], first['derived']['dialogue'][0]['delivery']), ('こんばんは', 'on_screen'))  # the shot decided the delivery
        self.assertIn('<Subject 1> (S1) says: <d>[Japanese] こんばんは</d>', first['prompt'])
        shot_call = next(data for step, data in self.calls if step == 'shot')
        self.assertTrue(shot_call['dialogue_reviewed'])
        self.assertNotIn('delivery_fixed', json.dumps(shot_call['dialogue']))

    def test_use_source_confirms_the_raw_transcript_and_a_silent_range_needs_no_review(self):
        sequence_id = self.create(review=None)
        self.work()
        self.client.post(f'/api/sequences/{sequence_id}/compilation/use-source')
        detail = self.detail(sequence_id)
        self.assertEqual((detail['compilation']['state'], detail['compilation']['result']['origin']), ('confirmed', 'source'))
        self.assertEqual([row['text'] for row in detail['compilation']['result']['content']['lines']], ['こんにちは', 'はい', 'じゃあね'])
        with db.connect() as conn:
            conn.execute('DELETE FROM transcript_segments')
        silent = self.create(review=None)
        self.work()
        self.assertEqual(self.detail(silent)['compilation']['state'], 'confirmed')
        self.assertEqual(self.run_step(silent, 'direct')['queued'], 1)

    def test_a_model_that_never_answers_validly_leaves_every_line_for_the_user(self):
        self.answers['compile'] = {'lines': [{'ids': ['L9'], 'text': 'x', 'speaker': None, 'subtitle_ids': [], 'confidence': 'low', 'note': '', 'continues_next': False}], 'voices': [], 'open_questions': []}
        sequence_id = self.create(review=None)
        self.work()
        content = self.detail(sequence_id)['compilation']['result']['content']
        self.assertEqual([(row['text'], row['confidence']) for row in content['lines']], [('こんにちは', 'high'), ('はい', 'low'), ('じゃあね', 'low')])
        self.assertEqual(self.detail(sequence_id)['compilation']['result']['completed_by_app'][:20], 'OpenRouter returned ')

    def test_validate_compile_demands_every_line_once_in_order(self):
        answer = {'lines': [{'ids': ['L1', 'L2'], 'text': 'a', 'speaker': None, 'subtitle_ids': [], 'confidence': 'high', 'note': '', 'continues_next': False},
                            {'ids': ['L3'], 'text': 'b', 'speaker': 'hero', 'subtitle_ids': ['U9'], 'confidence': 'high', 'note': '', 'continues_next': False}], 'voices': [], 'open_questions': []}
        ok = direction.validate_compile(answer, ['L1', 'L2', 'L3'], {'U1'}, ['Hero'])
        self.assertEqual((ok['lines'][1]['speaker'], ok['lines'][1]['subtitle_ids']), ('Hero', []))
        with self.assertRaisesRegex(ValueError, 'missing'):
            direction.validate_compile({**answer, 'lines': answer['lines'][:1]}, ['L1', 'L2', 'L3'], set(), ['Hero'])
        with self.assertRaisesRegex(ValueError, 'next lines in order'):
            direction.validate_compile({**answer, 'lines': [answer['lines'][1], answer['lines'][0]]}, ['L1', 'L2', 'L3'], set(), ['Hero'])
        with self.assertRaisesRegex(ValueError, 'not a character'):
            direction.validate_compile({**answer, 'lines': [{**answer['lines'][0], 'speaker': 'Ghost'}, answer['lines'][1]]}, ['L1', 'L2', 'L3'], set(), ['Hero'])


if __name__ == '__main__':
    unittest.main()
