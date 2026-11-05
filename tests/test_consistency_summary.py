import unittest
from app.consistency_summary import build_input


class SummaryInputTest(unittest.TestCase):
    def test_preserves_transcript_speaker_and_ocr_as_separate_evidence(self):
        results = [{'type': 'initial_frame', 'content': {'composition': 'wide'}}, {'type': 'subtitle_ocr', 'content': {'events': [{'text': 'Olá', 'language': 'pt'}]}}]
        transcript = [{'start_time': 1, 'end_time': 2, 'speaker_label': 'S1', 'character_context_id': 'ch', 'text': 'Olá', 'origin': 'model'}]
        data = build_input(results, transcript, [{'id': 'ch', 'title': 'Hero', 'kind': 'character'}])
        self.assertIn('composition: wide', data['visual_evidence_text'])
        self.assertIn('S1; linked character Hero (ch)', data['transcript_text'])
        self.assertEqual(data['subtitle_ocr']['events'][0]['text'], 'Olá')
        self.assertNotIn('Olá', data['visual_evidence_text'])
