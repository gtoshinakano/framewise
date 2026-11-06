import json
import unittest

from app import db
import tests.test_analysis as analysis_tests


class TextEngineSettingsTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown
    def test_shared_selection_applies_to_cuts_and_queued_summaries(self):
        old_preset = self.client.get('/api/cuts/c1/analysis-config').json()
        self.client.put('/api/cuts/c1/analysis-config', json={**old_preset, 'text_provider': 'openrouter', 'openrouter_model': 'old/model'})
        selected = {'text_provider': 'local', 'local_thinking': True, 'openrouter_model': 'new/model'}
        self.assertEqual(self.client.put('/api/text-generation-settings', json=selected).json(), selected)
        self.assertEqual(self.client.get('/api/text-generation-settings').json(), selected)
        for cut_id in ('c1', 'c2'):
            config = self.client.get(f'/api/cuts/{cut_id}/analysis-config').json()
            self.assertEqual({key: config[key] for key in selected}, selected)
        first = self.client.post('/api/cuts/c1/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
        self.client.put('/api/text-generation-settings', json={'text_provider': 'openrouter', 'openrouter_model': 'another/model'})
        second = self.client.post('/api/cuts/c2/regenerate-summary', json={'mode': 'regenerate'}).json()['job_id']
        original = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (first,))['payload_json'])
        following = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (second,))['payload_json'])
        self.assertEqual((original['text_provider'], original['local_thinking'], original['model']), ('local', True, 'new/model'))
        self.assertEqual((following['text_provider'], following['model']), ('openrouter', 'another/model'))

    def test_invalid_model_does_not_replace_selection(self):
        self.assertEqual(self.client.put('/api/text-generation-settings', json={'openrouter_model': 'invalid model'}).status_code, 422)
        self.assertEqual(self.client.get('/api/text-generation-settings').json()['openrouter_model'], 'openrouter/free')
