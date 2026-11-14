"""Sequence timeline arithmetic: pauses, hard-cut boundaries and their timing, one shot per cut, and what is derived per shot."""
import array
import math
import tempfile
import unittest
import wave
from pathlib import Path

from app import sequence_timeline as timeline_math

RATE = 16000


def write_wav(path, spans, seconds):
    """A 16 kHz mono WAV that is loud inside ``spans`` and nearly silent elsewhere."""
    wave_form = [math.sin(2 * math.pi * 200 * index / RATE) for index in range(RATE // 200)]  # one period of a 200 Hz tone
    samples = array.array('h')
    for second in range(int(seconds * 20)):  # 50 ms blocks
        t = second / 20
        level = 9000 if any(start <= t < end for start, end in spans) else 12
        samples.extend(int(level * wave_form[index % len(wave_form)]) for index in range(RATE // 20))
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(RATE)
        out.writeframes(samples.tobytes())


def cut(cut_id, index, start, end):
    return {'id': cut_id, 'cut_index': index, 'start_time': start, 'end_time': end, 'first_frame_time': start + 0.04}


def line(start, end, text, speaker='SPEAKER_01', character=None, parts=None):
    row = {'start_time': start, 'end_time': end, 'text': text, 'speaker_label': speaker, 'character_context_id': 'ctx' if character else None, 'character_name': character}
    row['parts'] = [{'start_time': a, 'end_time': b, 'text': words} for a, b, words in parts] if parts else [dict(row)]
    return row


# Three cuts starting 100 s into the episode: a short one, one too long for a single shot, a short one.
CUTS = [cut('c1', 4, 100.0, 104.0), cut('c2', 5, 104.0, 124.0), cut('c3', 6, 124.0, 127.0)]
SPEECH = [(100.5, 103.0), (105.0, 110.0), (112.0, 118.0), (120.0, 126.0)]
LINES = [line(100.5, 103.0, 'First line.', character='Hero'),
         line(105.0, 110.0, 'Second line.'),
         line(112.0, 118.0, 'Third line.'),
         line(120.0, 126.0, 'ends after the hard cut', parts=[(120.0, 122.0, 'ends'), (122.0, 123.6, 'after'), (124.2, 125.0, 'the hard'), (125.0, 126.0, 'cut')])]


def levels(loud, seconds, window=0.05):
    """A synthetic envelope: loud inside ``loud`` spans, near silence elsewhere."""
    return [0.2 if any(a <= index * window < b for a, b in loud) else 0.001 for index in range(round(seconds / window))]


class TimelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.audio = Path(cls.tmp.name) / 'audio.wav'
        write_wav(cls.audio, SPEECH, 130)
        cls.envelope = timeline_math.audio_envelope(cls.audio, 100.0, 127.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.timeline = timeline_math.build(CUTS, LINES, self.envelope)

    def test_pauses_come_from_the_audio_level(self):
        self.assertEqual(len(self.envelope), 540)
        pauses = timeline_math.find_pauses(self.envelope)
        self.assertEqual([(round(a), round(b)) for a, b in pauses], [(0, 0), (3, 5), (10, 12), (18, 20), (26, 27)])
        self.assertEqual(timeline_math.audio_envelope(Path(self.tmp.name) / 'missing.wav', 0, 1), [])

    def test_every_hard_cut_is_a_boundary_and_only_long_cuts_get_split_points(self):
        timeline = self.timeline
        self.assertEqual((timeline['origin'], timeline['duration']), (100.0, 27.0))
        self.assertEqual([(item['id'], item['t0'], item['t1']) for item in timeline['lines']], [('L1', 0.5, 3.0), ('L2', 5.0, 10.0), ('L3', 12.0, 18.0), ('L4', 20.0, 26.0)])
        first, second = timeline['boundaries']
        self.assertEqual((first['id'], first['after_cut'], first['before_cut'], first['t'], first['time'], first['speech'], list(first['options'])), ('K1', 5, 6, 4.0, 104.0, None, ['keep_cut']))
        # The hard cut at 24 s falls inside L4: the words on each side are known, but neither move is allowed here.
        self.assertEqual((second['speech']['lines'], second['speech']['words_before'], second['speech']['words_after']), (['L4'], 'ends after', 'the hard cut'))
        self.assertEqual(list(second['options']), ['keep_cut'])
        self.assertIn('L4 starts 4.06 s before the hard cut; a boundary moves at most 3 s', second['unavailable']['before_line'])
        self.assertIn("2.06 s of cut 7's 3.00 s picture; at most half", second['unavailable']['after_line'])
        # Only the 20-second cut is offered places to divide it: its two pauses, never inside a line.
        self.assertEqual(list(timeline['splits']), ['c2'])
        self.assertEqual([(item['kind'], round(item['t'])) for item in timeline['splits']['c2']], [('pause', 11), ('pause', 19)])

    def test_a_moved_boundary_snaps_to_where_the_voice_really_stops_or_starts(self):
        cuts = [cut('a', 0, 0.0, 6.0), cut('b', 1, 6.0, 12.0)]
        lines = [line(4.0, 6.8, 'もう一度言って', speaker='A'), line(9.0, 11.0, '嫌です', speaker='B')]
        # The transcript ends L1 at 6.8 s, but the voice is heard until 7.0 s.
        boundary = timeline_math.build(cuts, lines, levels([(4.0, 7.0), (9.0, 11.0)], 12.0))['boundaries'][0]
        options = boundary['options']
        self.assertEqual((boundary['speech']['kind'], boundary['speech']['keeps']), ('inside_line', ['L1']))
        self.assertEqual((options['keep_cut']['t'], options['before_line']['t'], options['after_line']['t']), (6.0, 3.94, 7.06))
        self.assertEqual((options['before_line']['shift'], options['after_line']['shift']), (-2.06, 1.06))
        self.assertEqual((options['before_line']['drops'], options['after_line']['drops']), ({'cut': 1, 'from': 3.94, 'to': 6.0}, {'cut': 2, 'from': 0.0, 'to': 1.06}))
        # Without audio the transcript edge is used with a short pad.
        plain = timeline_math.build(cuts, lines, [])['boundaries'][0]['options']
        self.assertEqual((plain['before_line']['t'], plain['after_line']['t']), (3.85, 6.95))
        # A voice that never fades inside the gap leaves the transcript edge as it is.
        self.assertEqual(timeline_math.speech_edge(levels([(0.0, 9.5)], 20.0), 6.8, 'after', 6.8, 9.0), 6.95)
        self.assertEqual(timeline_math.speech_edge([0.2] * 400, 6.8, 'after', 6.8, 9.0), 6.95)  # a flat level says nothing
        # A voice that sets in before the transcript says is caught too, but only within SPEECH_TAIL.
        self.assertEqual(timeline_math.speech_edge(levels([(3.7, 4.5)], 10.0), 4.0, 'before', 2.0, 4.0), 3.64)
        self.assertEqual(timeline_math.speech_edge(levels([(3.0, 4.5)], 10.0), 4.0, 'before', 2.0, 4.0), 3.85)

    def test_a_boundary_never_takes_more_than_half_of_the_next_cut(self):
        cuts = [cut('a', 0, 0.0, 6.0), cut('b', 1, 6.0, 7.5), cut('c', 2, 7.5, 12.0)]
        boundary = timeline_math.build(cuts, [line(5.0, 7.2, '待って', speaker='A')], [])['boundaries'][0]
        self.assertEqual(list(boundary['options']), ['keep_cut', 'before_line'])
        self.assertIn("1.35 s of cut 2's 1.50 s picture; at most half", boundary['unavailable']['after_line'])

    def test_sentences_are_kept_together(self):
        lines = [line(2.5, 4.0, '今日は', speaker='A'), line(4.4, 7.0, 'いい天気ですね。', speaker='A'), line(9.0, 11.0, 'そうだね。', speaker='B')]
        timeline = timeline_math.build([cut('a', 0, 0.0, 5.0), cut('b', 1, 5.0, 12.0)], lines, [])
        self.assertEqual([item['continues_next'] for item in timeline['lines']], [True, False, False])
        # The cut at 5 s falls inside L2, which ends the sentence L1 began: the phrase to keep whole is L1-L2.
        boundary = timeline['boundaries'][0]
        self.assertEqual((boundary['speech']['lines'], boundary['speech']['keeps']), (['L2'], ['L1', 'L2']))
        self.assertEqual(boundary['options']['after_line']['t'], 7.15)
        self.assertIn('at most half', boundary['unavailable']['before_line'])  # 2.65 s of a 5-second cut
        halves = timeline_math.build([cut('a', 0, 0.0, 4.2), cut('b', 1, 4.2, 12.0)], lines, [])['boundaries'][0]['speech']
        self.assertEqual((halves['kind'], halves['lines'], halves['words_before'], halves['words_after']), ('between_sentence_halves', ['L1', 'L2'], '今日は', 'いい天気ですね。'))
        replaced = timeline_math.replace_voices([{'speaker': 'A', 'from': 0.0, 'to': 12.0, 'delivery': 'off_screen'}], 'A', 5.0, 12.0, [{'from': 1.0, 'delivery': 'on_screen'}, {'from': 4.0, 'delivery': 'narrative'}])
        self.assertEqual([(item['from'], item['to'], item['delivery']) for item in replaced], [(0.0, 5.0, 'off_screen'), (5.0, 9.0, 'on_screen'), (9.0, 12.0, 'narrative')])
        self.assertEqual(timeline_math.shot_voices(replaced, 5.0, 9.0), [{'speaker': 'A', 'from': 0.0, 'to': 4.0, 'delivery': 'on_screen'}])

    def test_a_hard_cut_at_the_very_edge_of_a_line_divides_only_a_sentence(self):
        cuts = [cut('a', 0, 0.0, 6.0), cut('b', 1, 6.0, 12.0)]
        parts = [(5.9, 6.4, 'ずっと'), (6.4, 7.0, '続けたい')]
        # The voice sets in 0.1 s before the hard cut and no word lies before it: nothing is divided.
        alone = timeline_math.build(cuts, [line(1.0, 3.0, '好きだった。', speaker='A'), line(5.9, 7.0, 'ずっと続けたい', speaker='A', parts=parts)], [])
        self.assertIsNone(alone['boundaries'][0]['speech'])
        # When that line goes on with the sentence before it, the hard cut lies between the two halves of the sentence.
        clause = timeline_math.build(cuts, [line(4.0, 5.85, '好きだったし', speaker='A'), line(5.9, 7.0, 'ずっと続けたい', speaker='A', parts=parts)], [])['boundaries'][0]['speech']
        self.assertEqual((clause['kind'], clause['lines'], clause['words_before'], clause['words_after']), ('between_sentence_halves', ['L1', 'L2'], '好きだったし', 'ずっと続けたい'))
        # A long pause between two halves is a natural break, even when the dialogue says the sentence goes on.
        paused = [{**line(3.0, 5.0, 'もちろん', speaker='A'), 'compiled': True, 'continues_next': True}, {**line(7.5, 9.0, 'ここよ', speaker='A'), 'compiled': True}]
        self.assertIsNone(timeline_math.build(cuts, paused, [])['boundaries'][0]['speech'])

    def test_a_long_silent_cut_can_always_be_divided(self):
        timeline = timeline_math.build([cut('long', 0, 0.0, 50.0)], [], [])
        spans = timeline_math.decide(timeline)
        self.assertEqual(len(spans), 4)
        self.assertTrue(all(span['end_time'] - span['start_time'] <= timeline_math.MAX_SHOT_SECONDS for span in spans))
        self.assertTrue(all(item['kind'] == 'quiet' for item in timeline['splits']['long']))
        events = timeline_math.build([cut('long', 0, 0.0, 20.0)], [], [], {'long': [9.5]})
        self.assertEqual([(item['kind'], item['t']) for item in events['splits']['long']], [('event', 9.5)])

    def test_one_shot_per_cut_with_the_long_cut_in_parts(self):
        spans = timeline_math.decide(self.timeline, intents={6: 'Walk.'}, scenes={5: 'S1', 6: 'S1', 7: 'S2'})
        self.assertEqual([(span['start_time'], span['end_time'], span['frame_cut_id']) for span in spans], [(100.0, 104.0, 'c1'), (104.0, 111.0, 'c2'), (111.0, 124.0, 'c2'), (124.0, 127.0, 'c3')])
        self.assertEqual([(span['intent'], span['scene']) for span in spans], [('', 'S1'), ('Walk.', 'S1'), ('Walk.', 'S1'), ('', 'S2')])
        self.assertEqual(timeline_math.check_spans(spans, self.timeline), [])
        first, second, third, fourth = timeline_math.expand(spans, self.timeline)
        self.assertEqual((first['key'], first['cut'], first['part'], first['source_cuts'], first['merged'], first['duration']), ('100000-104000', 5, None, [5], False, 4.0))
        self.assertEqual((first['generate_seconds'], first['trim_seconds']), (5.0, 4.0))  # shorter than H3's minimum
        self.assertEqual(first['frame'], {'cut': 5, 'cut_id': 'c1', 'time': 100.04, 'strategy': 'cut_first_frame', 'continues_previous_shot': False})
        self.assertEqual((first['opens'], first['closes']), ({'how': 'start'}, {'how': 'keep_cut', 'boundary': 'K1', 'shift': 0.0, 'hard_cut': 4.0}))
        self.assertEqual((second['part'], second['closes'], third['opens']), ([1, 2], {'how': 'continuation'}, {'how': 'continuation'}))
        # The second part of the long cut continues the take: its frame is where the first part stopped.
        self.assertEqual((third['frame']['strategy'], third['frame']['continues_previous_shot'], third['frame']['time']), ('frame_at_time', True, 111.0))
        # The hard cut into cut 7 divides L4: both shots know the words on each side.
        split = {'lines': ['L4'], 'before': 'ends after', 'after': 'the hard cut'}
        self.assertEqual((third['closes']['split'], fourth['opens']['split'], fourth['closes']), (split, split, {'how': 'end'}))
        self.assertEqual((third['lines'][-1]['text'], third['lines'][-1]['crosses_end'], fourth['lines'][0]['text'], fourth['lines'][0]['t0']), ('ends after', True, 'the hard cut', 0.2))
        self.assertEqual((first['lines'][0]['t0'], first['lines'][0]['t1'], first['lines'][0]['character_name']), (0.5, 3.0, 'Hero'))

    def test_spans_are_checked_against_one_shot_per_cut(self):
        spans = timeline_math.decide(self.timeline)
        check = lambda changed: ' '.join(timeline_math.check_spans(changed, self.timeline))
        self.assertIn('merges, drops or reorders cuts', check([{'start_time': 100.0, 'end_time': 111.0}, {'start_time': 111.0, 'end_time': 127.0}]))
        self.assertIn('the shot of cut 6 lasts 20.0 s', check([spans[0], {**spans[1], 'end_time': 124.0}, spans[3]]))
        self.assertIn('cut 5 lasts 4.0 s and is one shot', check([{**spans[0], 'end_time': 102.0}, {**spans[0], 'start_time': 102.0}, *spans[1:]]))
        self.assertIn('K1 (the hard cut between cut 5 and cut 6) moved +0.50 s, but no speech crosses that cut', check([{**spans[0], 'end_time': 104.5}, {**spans[1], 'start_time': 104.5}, *spans[2:]]))
        self.assertIn('must end where', check(spans[:-1]))
        self.assertIn('gap or overlap', check([spans[0], *spans[2:]]))

    def test_choices_move_boundaries_and_the_edges_carry_the_reason(self):
        cuts = [cut('a', 0, 0.0, 6.0), cut('b', 1, 6.0, 12.0)]
        timeline = timeline_math.build(cuts, [line(4.0, 6.8, 'もう一度言って', speaker='A'), line(9.0, 11.0, '嫌です', speaker='B')], levels([(4.0, 7.0), (9.0, 11.0)], 12.0))
        spans = timeline_math.decide(timeline, {'K1': 'after_line'}, {1: 'Ask.', 2: 'Refuse.'})
        self.assertEqual([(span['start_time'], span['end_time'], span['intent']) for span in spans], [(0.0, 7.06, 'Ask.'), (7.06, 12.0, 'Refuse.')])
        decided = [{'after_cut_id': 'a', 'choice': 'after_line', 'reason': 'The plea stays with her face.', 'by': 'director'}]
        first, second = timeline_math.expand(spans, timeline, decisions=decided)
        edge = {'how': 'after_line', 'boundary': 'K1', 'shift': 1.06, 'hard_cut': 6.0, 'keeps': ['L1'], 'reason': 'The plea stays with her face.', 'by': 'director'}
        self.assertEqual((first['closes'], second['opens'], first['source_cuts'], first['ends_mid_cut']), (edge, edge, [1, 2], False))
        self.assertEqual([(line['line'], line['crosses_end']) for line in first['lines']], [('L1', False)])
        self.assertEqual((second['frame']['strategy'], second['frame']['continues_previous_shot'], second['starts_mid_cut']), ('frame_at_time', False, True))
        moved = timeline_math.move_boundary(spans, timeline, 'K1', 'before_line')
        self.assertEqual([(span['start_time'], span['end_time'], span['intent']) for span in moved], [(0.0, 3.94, 'Ask.'), (3.94, 12.0, 'Refuse.')])
        self.assertEqual(timeline_math.choices_of(moved, timeline), {'K1': 'before_line'})
        after = timeline_math.expand(moved, timeline, decisions=decided)[1]
        self.assertEqual((after['opens']['how'], after['frame']['strategy'], 'reason' in after['opens']), ('before_line', 'cut_first_frame', False))  # the director's reason was for another choice
        with self.assertRaises(timeline_math.PlanError):
            timeline_math.move_boundary(spans, self.timeline, 'K9', 'keep_cut')

    def test_two_moves_may_not_empty_the_cut_between_them(self):
        cuts = [cut('a', 0, 0.0, 5.0), cut('b', 1, 5.0, 6.6), cut('c', 2, 6.6, 12.0)]
        timeline = timeline_math.build(cuts, [line(3.6, 5.3, 'あのね', speaker='A'), line(6.2, 7.0, 'うん', speaker='B')], [])
        first, second = timeline['boundaries']
        self.assertEqual((first['options']['after_line']['t'], second['options']['before_line']['t']), (5.45, 6.05))
        problems = timeline_math.check_spans(timeline_math.decide(timeline, {'K1': 'after_line', 'K2': 'before_line'}), timeline)
        self.assertEqual(problems, ['cut 2 keeps only 0.60 s of its 1.60 s picture (K1 after_line, K2 before_line); keep one of these hard cuts'])
        self.assertEqual(timeline_math.check_spans(timeline_math.decide(timeline, {'K1': 'after_line'}), timeline), [])
        with self.assertRaises(timeline_math.PlanError) as refused:
            timeline_math.decide(self.timeline, {'K2': 'after_line'})
        self.assertIn('K2: after_line is not possible here', refused.exception.errors[0])


if __name__ == '__main__':
    unittest.main()
