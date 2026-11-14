"""Sequence timeline arithmetic: pauses, candidate boundaries, plan checks and what is derived per shot."""
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

    def ids(self, kind):
        return [item['id'] for item in self.timeline['candidates'] if item['kind'] == kind]

    def test_pauses_come_from_the_audio_level(self):
        self.assertEqual(len(self.envelope), 540)
        pauses = timeline_math.find_pauses(self.envelope)
        self.assertEqual([(round(a), round(b)) for a, b in pauses], [(0, 0), (3, 5), (10, 12), (18, 20), (26, 27)])
        self.assertEqual(timeline_math.audio_envelope(Path(self.tmp.name) / 'missing.wav', 0, 1), [])

    def test_timeline_clock_lines_and_candidates(self):
        timeline = self.timeline
        self.assertEqual((timeline['origin'], timeline['duration']), (100.0, 27.0))
        self.assertEqual([(item['id'], item['t0'], item['t1']) for item in timeline['lines']], [('L1', 0.5, 3.0), ('L2', 5.0, 10.0), ('L3', 12.0, 18.0), ('L4', 20.0, 26.0)])
        candidates = timeline['candidates']
        self.assertEqual((candidates[0]['id'], candidates[0]['kind'], candidates[-1]['kind']), ('B0', 'start', 'end'))
        self.assertEqual([item['t'] for item in candidates if item['kind'] == 'cut'], [4.0, 24.0])
        # The long cut is offered boundaries in its two pauses, never inside a spoken line.
        inside = [item for item in candidates if item['kind'] == 'pause']
        self.assertEqual([round(item['t']) for item in inside], [11, 19])
        self.assertTrue(all(item['cut'] == 6 and item['pause'] >= 1.9 for item in inside))
        self.assertEqual(next(item for item in candidates if item['t'] == 24.0)['crosses_line'], 'L4')
        times = [item['t'] for item in candidates]
        self.assertEqual(times, sorted(times))
        self.assertLessEqual(max(b - a for a, b in zip(times, times[1:])), timeline_math.MAX_SHOT_SECONDS)

    def test_no_boundary_is_offered_close_to_a_hard_cut(self):
        # A pause right after a cut starts would only produce a sliver of a shot.
        lines = [line(1.8, 5.0, 'Starts late.'), line(5.6, 9.0, 'Second.')]
        timeline = timeline_math.build([cut('only', 0, 0.0, 10.0)], lines, [])
        self.assertEqual([(item['kind'], item['t']) for item in timeline['candidates'][1:-1]], [('pause', 5.3)])
        self.assertTrue(all(timeline_math.EDGE_SECONDS <= item['t'] <= 10.0 - timeline_math.EDGE_SECONDS for item in timeline['candidates'][1:-1]))

    def test_a_long_silent_cut_can_always_be_divided(self):
        timeline = timeline_math.build([cut('long', 0, 0.0, 50.0)], [], [])
        times = [item['t'] for item in timeline['candidates']]
        self.assertLessEqual(max(b - a for a, b in zip(times, times[1:])), timeline_math.MAX_SHOT_SECONDS)
        self.assertTrue(all(item['kind'] == 'quiet' for item in timeline['candidates'][1:-1]))
        events = timeline_math.build([cut('long', 0, 0.0, 20.0)], [], [], {'long': [9.5]})
        self.assertEqual([(item['kind'], item['t']) for item in events['candidates'][1:-1]], [('event', 9.5)])

    def plan(self, *pairs, **extra):
        return [{'from': a, 'to': b, 'intent': 'Show it.', 'staging': 'One take.', **extra} for a, b in pairs]

    def test_model_plan_is_checked_with_actionable_messages(self):
        first, second = self.ids('pause')
        good = self.plan(('B0', 'B1'), ('B1', first), (first, second), (second, 'B4'), ('B4', 'B5'))
        self.assertEqual(timeline_math.check_model_plan(good, self.timeline), [])
        too_long = timeline_math.check_model_plan(self.plan(('B0', 'B1'), ('B1', 'B4'), ('B4', 'B5')), self.timeline)
        self.assertEqual(len(too_long), 1)
        self.assertIn('is 20.0 s', too_long[0])
        self.assertIn(f'{first} (pause 2 s', too_long[0])
        gap = timeline_math.check_model_plan(self.plan(('B0', 'B1'), (first, second), (second, 'B5')), self.timeline)
        self.assertTrue(any('without gaps' in error for error in gap))
        self.assertTrue(any('must end at B5' in error for error in timeline_math.check_model_plan(self.plan(('B0', 'B1')), self.timeline)))
        unknown = timeline_math.check_model_plan(self.plan(('B0', 'B99')), self.timeline)
        self.assertIn('candidate ids', unknown[0])
        merged = [{'from': 'B0', 'to': first, 'intent': 'x', 'staging': ''}, *self.plan((first, second), (second, 'B4'), ('B4', 'B5'))]
        self.assertTrue(any('staging' in error for error in timeline_math.check_model_plan(merged, self.timeline)))
        wrong_frame = self.plan(('B0', 'B1'), frame_cut=7) + self.plan(('B1', first), (first, second), (second, 'B4'), ('B4', 'B5'))
        self.assertTrue(any('frame_cut 7' in error for error in timeline_math.check_model_plan(wrong_frame, self.timeline)))

    def test_auto_fix_keeps_usable_boundaries_and_adds_missing_ones(self):
        fixed = timeline_math.auto_fix(self.plan(('B0', 'B1'), ('B1', 'B4'), ('B4', 'B5')), self.timeline)
        self.assertEqual(timeline_math.check_model_plan(fixed, self.timeline), [])
        self.assertEqual((fixed[0]['from'], fixed[0]['to'], fixed[0].get('auto_fixed')), ('B0', 'B1', None))
        self.assertTrue(any(shot.get('auto_fixed') for shot in fixed))
        # Nothing usable at all: fall back to the source cuts, split where they are too long.
        fallback = timeline_math.auto_fix('nonsense', self.timeline)
        self.assertEqual(timeline_math.check_model_plan(fallback, self.timeline, strict=False), [])
        self.assertEqual(fallback[0]['frame_cut'], 5)

    def expanded(self):
        """Cut 5 alone, the long cut 6 in two parts, and its tail merged with cut 7 (carried by cut 7's picture)."""
        first, second = self.ids('pause')
        plan = [{'from': 'B0', 'to': 'B1', 'frame_cut': 5, 'scene': 'S1', 'intent': 'Open.', 'staging': ''},
                {'from': 'B1', 'to': first, 'frame_cut': 6, 'scene': 'S1', 'intent': 'First half.', 'staging': 'Stops in the pause.'},
                {'from': first, 'to': second, 'frame_cut': 6, 'scene': 'S1', 'intent': 'Second half.', 'staging': 'Continues the take.'},
                {'from': second, 'to': 'B5', 'frame_cut': 7, 'scene': 'S1', 'intent': 'End.', 'staging': 'Carried by the last picture.'}]
        self.assertEqual(timeline_math.check_model_plan(plan, self.timeline), [])
        spans = timeline_math.to_spans(plan, self.timeline)
        return spans, timeline_math.expand(spans, self.timeline)

    def test_expand_derives_frame_lengths_and_dialogue(self):
        spans, shots = self.expanded()
        self.assertEqual(timeline_math.check_spans(spans, self.timeline), [])
        first, second, third, fourth = shots
        self.assertEqual((first['key'], first['source_cuts'], first['merged'], first['duration']), ('100000-104000', [5], False, 4.0))
        self.assertEqual((first['generate_seconds'], first['trim_seconds']), (5.0, 4.0))  # shorter than H3's minimum
        self.assertEqual(first['frame'], {'cut': 5, 'cut_id': 'c1', 'time': 100.04, 'strategy': 'cut_first_frame', 'continues_previous_shot': False})
        self.assertEqual((second['starts_mid_cut'], second['ends_mid_cut'], second['trim_seconds']), (False, True, None))
        # The second part of the long cut continues the take: its frame is where the first part stopped.
        self.assertEqual((third['frame']['strategy'], third['frame']['continues_previous_shot'], third['frame']['time']), ('frame_at_time', True, third['start_time']))
        self.assertEqual((fourth['source_cuts'], fourth['merged'], fourth['frame']['cut'], fourth['frame']['strategy']), ([6, 7], True, 7, 'cut_first_frame'))
        self.assertEqual([item['line'] for item in first['lines']], ['L1'])
        self.assertEqual((first['lines'][0]['t0'], first['lines'][0]['t1'], first['lines'][0]['character_name']), (0.5, 3.0, 'Hero'))
        self.assertEqual(fourth['lines'][0]['text'], 'ends after the hard cut')

    def test_a_line_cut_by_a_boundary_is_split_by_word(self):
        spans = [{'start_time': 100.0, 'end_time': 104.0}, {'start_time': 104.0, 'end_time': 115.0}, {'start_time': 115.0, 'end_time': 124.0}, {'start_time': 124.0, 'end_time': 127.0}]
        shots = timeline_math.expand(spans, self.timeline)
        before, after = shots[2]['lines'][-1], shots[3]['lines'][0]
        self.assertEqual((before['line'], before['text'], before['crosses_start'], before['crosses_end']), ('L4', 'ends after', False, True))
        self.assertEqual((after['line'], after['text'], after['crosses_start'], after['crosses_end'], after['t0']), ('L4', 'the hard cut', True, False, 0.2))

    def test_spans_are_checked_against_the_cuts(self):
        spans, _ = self.expanded()
        self.assertTrue(any('at most 15' in error for error in timeline_math.check_spans([{'start_time': 100.0, 'end_time': 127.0}], self.timeline)))
        self.assertTrue(any('must end where' in error for error in timeline_math.check_spans(spans[:-1], self.timeline)))
        self.assertTrue(any('gap or overlap' in error for error in timeline_math.check_spans([spans[0], *spans[2:]], self.timeline)))

    def test_merge_and_split_by_hand(self):
        spans, _ = self.expanded()
        merged = timeline_math.merge_spans(spans, 0)
        self.assertEqual((len(merged), merged[0]['start_time'], merged[0]['end_time']), (3, 100.0, spans[1]['end_time']))
        with self.assertRaises(timeline_math.PlanError):
            timeline_math.merge_spans(merged, 0)  # 4 + 7 + 8 s would exceed one shot
        with self.assertRaises(timeline_math.PlanError):
            timeline_math.merge_spans(spans, len(spans) - 1)
        split = timeline_math.split_span(spans, 1, 108.0)
        self.assertEqual((len(split), split[1]['end_time'], split[2]['start_time'], split[2]['frame_cut_id']), (5, 108.0, 108.0, None))
        self.assertEqual(timeline_math.check_spans(split, self.timeline), [])
        with self.assertRaises(timeline_math.PlanError):
            timeline_math.split_span(spans, 1, 104.2)


if __name__ == '__main__':
    unittest.main()
