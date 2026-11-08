import unittest

from app.audio_matching import assign_words, match_voices, reconcile_speakers, restore_aligned_text


class AudioMatchingTests(unittest.TestCase):
    def test_restores_punctuation_and_spacing_or_reports_mismatch(self):
        words = [{'text': 'Hello'}, {'text': 'world'}]
        result = restore_aligned_text('Hello, world!', words)
        self.assertEqual(''.join(word['text'] for word in result), 'Hello, world!')
        self.assertEqual([word['text'] for word in result], ['Hello,', ' world!'])
        self.assertIsNone(restore_aligned_text('Different', words))

    def test_punctuation_stays_with_its_word_and_text_is_preserved(self):
        words = [{'text': word, 'start_time': index, 'end_time': index + .5}
                 for index, word in enumerate(['Hello', 'world', 'Next', 'yes'])]
        text = '“Hello,” (world)! Next… ¿yes?'
        result = restore_aligned_text(text, words)
        self.assertEqual([word['text'] for word in result], ['“Hello,”', ' (world)!', ' Next…', ' ¿yes?'])
        self.assertEqual(''.join(word['text'] for word in result), text)
        self.assertEqual([(word['start_time'], word['end_time']) for word in result],
                         [(word['start_time'], word['end_time']) for word in words])

    def test_assign_words_uses_overlap_and_preserves_unattributed_speech(self):
        words = [{'start_time': 1., 'end_time': 2., 'text': 'hello'},
                 {'start_time': 3., 'end_time': 4., 'text': 'there'}]
        turns = [{'start_time': .8, 'end_time': 1.2, 'speaker': 'A'},
                 {'start_time': 1.2, 'end_time': 2.5, 'speaker': 'B'}]
        self.assertEqual([w['speaker'] for w in assign_words(words, turns)], ['B', None])
        with self.assertRaises(ValueError):
            assign_words([{'start_time': 2., 'end_time': 1., 'text': 'bad'}], turns)

    def test_voice_matching_requires_confidence_and_distinct_character(self):
        refs = [{'character_id': 'alice', 'embedding': [1., 0.]},
                {'character_id': 'alice', 'embedding': [0.99, 0.01]},
                {'character_id': 'bob', 'embedding': [0., 1.]}]
        result = match_voices({'A': [1., 0.], 'B': [1., 1.], 'C': [0., 0.]}, refs)
        self.assertEqual(result['A']['character_id'], 'alice')
        self.assertNotIn('B', result)
        self.assertNotIn('C', result)


def segment(start, end, label, character=None):
    return {'start_time': start, 'end_time': end, 'speaker_label': label, 'character_context_id': character}


def turn(start, end, speaker):
    return {'start_time': start, 'end_time': end, 'speaker': speaker}


class ReconcileSpeakersTests(unittest.TestCase):
    existing = [segment(0, 4, 'SPEAKER_03', 'bob'), segment(4, 8, 'SPEAKER_07'),
                # Inside the part being redone: these labels are the suspect ones.
                segment(10, 14, 'SPEAKER_07'), segment(20, 24, 'SPEAKER_03', 'bob')]

    def test_excerpt_speakers_take_the_labels_of_the_surrounding_speech(self):
        turns = [turn(0, 4, 'A'), turn(4, 8, 'B'), turn(10, 12, 'A'), turn(12, 14, 'B'), turn(14, 18, 'C'), turn(20, 22, 'A')]
        names = reconcile_speakers(turns, self.existing, {}, (9, 19))
        self.assertEqual(names['A'], {'label': 'SPEAKER_03', 'character_id': 'bob'})
        self.assertEqual(names['B'], {'label': 'SPEAKER_07', 'character_id': None})
        # C was not heard next to anyone, so it gets a label nothing else uses.
        self.assertEqual(names['C'], {'label': 'SPEAKER_08', 'character_id': None})

    def test_old_labels_inside_the_redone_part_are_not_evidence(self):
        names = reconcile_speakers([turn(10, 14, 'A')], self.existing, {}, (9, 19))
        self.assertEqual(names['A']['label'], 'SPEAKER_08')

    def test_each_existing_label_goes_to_one_speaker_only(self):
        turns = [turn(0, 4, 'A'), turn(4, 5, 'B'), turn(5, 6, 'C')]
        names = reconcile_speakers(turns, [segment(0, 6, 'SPEAKER_01')], {}, (8, 12))
        self.assertEqual(names['A']['label'], 'SPEAKER_01')
        self.assertEqual(sorted((names['B']['label'], names['C']['label'])), ['SPEAKER_02', 'SPEAKER_03'])

    def test_brief_overlap_is_not_enough(self):
        names = reconcile_speakers([turn(0, 1, 'A')], [segment(.7, 3, 'SPEAKER_01')], {}, (8, 12))
        self.assertEqual(names['A']['label'], 'SPEAKER_02')

    def test_voice_match_reuses_the_label_of_that_character(self):
        names = reconcile_speakers([turn(12, 16, 'D')], self.existing, {'D': {'character_id': 'bob'}}, (9, 19))
        self.assertEqual(names['D'], {'label': 'SPEAKER_03', 'character_id': 'bob'})

    def test_voice_match_wins_over_a_link_inherited_from_the_label(self):
        names = reconcile_speakers([turn(0, 4, 'A')], self.existing, {'A': {'character_id': 'alice'}}, (9, 19))
        self.assertEqual(names['A'], {'label': 'SPEAKER_03', 'character_id': 'alice'})

    def test_a_link_is_inherited_only_when_most_of_the_label_carries_it(self):
        mixed = [segment(0, 2, 'SPEAKER_01', 'bob'), segment(2, 4, 'SPEAKER_01'), segment(4, 6, 'SPEAKER_01', 'alice')]
        self.assertIsNone(reconcile_speakers([turn(0, 6, 'A')], mixed, {}, (9, 12))['A']['character_id'])

    def test_no_existing_speech_starts_at_the_first_label(self):
        names = reconcile_speakers([turn(0, 2, 'X'), turn(2, 4, 'Y')], [], {}, (0, 4))
        self.assertEqual((names['X']['label'], names['Y']['label']), ('SPEAKER_00', 'SPEAKER_01'))
