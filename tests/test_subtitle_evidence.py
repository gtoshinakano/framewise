import unittest
from app.subtitle_evidence import consolidate, subtitle_readings


class SubtitleEvidenceTest(unittest.TestCase):
    def test_only_ocr_inside_predicted_subtitle_box(self):
        detections = {'subtitles': [{'type': 'box', 'coords': [0, 80, 100, 100]}]}
        ocr = {'Olá mundo': [{'type': 'box', 'coords': [10, 82, 90, 95]}], 'Street sign': [{'type': 'box', 'coords': [2, 4, 40, 20]}]}
        self.assertEqual(subtitle_readings(detections, ocr), [{'text': 'Olá mundo', 'box': [10, 82, 90, 95]}])
        self.assertEqual(subtitle_readings({}, ocr), [])

    def test_repeated_readings_have_observation_times_not_invented_speech_times(self):
        read = lambda text: {'text': text, 'box': [1, 2, 3, 4]}
        samples = [{'time': 1, 'readings': [read('Olá mundo')]}, {'time': 2, 'readings': [read('OLÁ  mundo')]},
                   {'time': 3, 'readings': []}, {'time': 4, 'readings': [read('Olá mundo')]}]
        events = consolidate(samples)
        self.assertEqual(len(events), 2)
        self.assertEqual((events[0]['first_observed'], events[0]['last_observed']), (1, 2))
        self.assertEqual(events[0]['language'], 'unknown')
        self.assertEqual(len(events[0]['observations']), 2)

    def test_joins_words_in_reading_order_without_dropping_repeated_words(self):
        detections = {'subtitles': [{'type': 'box', 'coords': [0, 70, 200, 120]}]}
        ocr = {'tudo': [{'type': 'box', 'coords': [60, 80, 100, 100]}],
               'Olá,': [{'type': 'box', 'coords': [10, 81, 50, 101]}],
               'bem?': [{'type': 'box', 'coords': [110, 80, 150, 100]}]}
        self.assertEqual(subtitle_readings(detections, ocr)[0]['text'], 'Olá, tudo bem?')
        repeats = {'sim': [{'type': 'box', 'coords': [10, 80, 40, 100]}, {'type': 'box', 'coords': [50, 80, 80, 100]}]}
        self.assertEqual(subtitle_readings(detections, repeats)[0]['text'], 'sim sim')
