import unittest

from app.audio_matching import assign_words, match_voices, restore_aligned_text


class AudioMatchingTests(unittest.TestCase):
    def test_restores_punctuation_and_spacing_or_reports_mismatch(self):
        words = [{'text': 'Hello'}, {'text': 'world'}]
        result = restore_aligned_text('Hello, world!', words)
        self.assertEqual(''.join(word['text'] for word in result), 'Hello, world!')
        self.assertIsNone(restore_aligned_text('Different', words))

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
