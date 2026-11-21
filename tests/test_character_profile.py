import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import character_profile, db, frame_model, inference_worker, visual_models, worker

FRAMES = [{'position': 'first', 'time': 0.0, 'image_size': [1280, 720]}]
ANSWER = {'character_sheet': {'hair': 'long black hair, straight fringe', 'eyes': 'brown', 'face': 'oval', 'skin': 'fair',
                              'build': 'slim, tall', 'age': 'late teens', 'distinguishing_features': 'none visible'},
          'outfit_sheet': {'top': 'navy blazer over a white shirt', 'bottom': 'not visible', 'footwear': 'not visible',
                           'accessories': 'red ribbon tie', 'colours': 'navy, white, red'}}


def model_reply(answer=ANSWER):
    return {'choices': [{'message': {'content': json.dumps(answer)}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 900, 'completion_tokens': 120}}


class CharacterProfileTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def add_box(self, detection_id='box', run_id='run', cut_start=0):
        with db.connect() as conn:
            if not conn.execute('SELECT 1 FROM character_detection_runs WHERE id=?', (run_id,)).fetchone():
                conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES (?,'c1',?,2,?,'test')", (run_id, cut_start, json.dumps(FRAMES)))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json) VALUES (?,?,'first',?)", (detection_id, run_id, json.dumps([128, 72, 640, 720])))

    def create(self, **body):
        return self.client.post('/api/character-detections/box/new-character', json={'title': 'Aiko', 'content': 'The class president.', **body})

    def test_creates_character_labels_box_and_queues_description(self):
        self.add_box()
        response = self.create(is_main=True, episode_only=True)
        self.assertEqual(response.status_code, 200, response.text)
        character = response.json()['character']
        self.assertEqual((character['kind'], character['title'], character['episode_id'], character['is_main'], character['content']),
                         ('character', 'Aiko', 'e', 1, 'The class president.'))
        box = db.one('SELECT character_context_id,label_source FROM character_detections WHERE id=?', ('box',))
        self.assertEqual((box['character_context_id'], box['label_source']), (character['id'], 'manual'))
        job = db.one('SELECT * FROM jobs WHERE id=?', (response.json()['job_id'],))
        self.assertEqual((job['pipeline'], job['status']), ('character_profile', 'queued'))
        self.assertEqual({key: json.loads(job['payload_json'])[key] for key in ('detection_id', 'character_context_id', 'cut_id')},
                         {'detection_id': 'box', 'character_context_id': character['id'], 'cut_id': 'c1'})
        status = self.client.get(f"/api/jobs/{job['id']}/character-profile-status").json()
        self.assertEqual((status['status'], status['character_context_id']), ('queued', character['id']))
        self.assertEqual(self.create(is_main=True, episode_only=True).json()['job_id'], job['id'])

    def test_project_wide_by_default(self):
        self.add_box()
        character = self.create().json()['character']
        self.assertEqual((character['episode_id'], character['is_main']), (None, 0))

    def test_refuses_duplicate_name_reference_and_stale_run(self):
        self.add_box()
        self.assertEqual(self.client.post('/api/character-detections/box/new-character', json={'title': ' hero '}).status_code, 409)
        with db.connect() as conn:
            conn.execute("UPDATE character_detections SET character_context_id='ctx',label_source='manual' WHERE id='box'")
            conn.execute("INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES ('ref','ctx','box')")
        self.assertEqual(self.create().status_code, 409)
        with db.connect() as conn:
            conn.execute("DELETE FROM character_image_references")
            conn.execute("UPDATE character_detection_runs SET cut_start=0.5 WHERE id='run'")
        self.assertEqual(self.create().status_code, 409)
        self.assertEqual(self.client.post('/api/character-detections/missing/new-character', json={'title': 'X'}).status_code, 404)
        self.assertEqual(db.one("SELECT COUNT(*) AS n FROM context_items WHERE title='Aiko'")['n'], 0)
        self.assertIsNone(db.one("SELECT 1 FROM jobs WHERE pipeline='character_profile'"))

    def test_main_flag_saves_through_context_api(self):
        item = self.client.post('/api/projects/p/context', json={'kind': 'character', 'title': 'Ren', 'is_main': True}).json()
        self.assertEqual(item['is_main'], 1)
        item = self.client.put(f"/api/context/{item['id']}", json={'kind': 'character', 'title': 'Ren', 'is_main': False}).json()
        self.assertEqual(item['is_main'], 0)
        # Only characters can be main characters.
        self.assertEqual(self.client.post('/api/projects/p/context', json={'kind': 'world', 'title': 'City', 'is_main': True}).json()['is_main'], 0)

    def run_job(self, reply=None):
        claimed = worker.claim_job(('character_profile',))
        self.assertIsNotNone(claimed)
        with patch.object(character_profile, 'images', return_value=(b'frame', b'crop')), \
             patch.object(frame_model, 'model_server', return_value=nullcontext('http://model')), \
             patch.object(inference_worker, 'request_json', return_value=reply or model_reply()) as request:
            character_profile.process(*claimed)
        return claimed[0], request

    def test_job_appends_sheets_to_details_kept_while_queued(self):
        self.add_box()
        created = self.create().json()
        with db.connect() as conn:
            conn.execute("UPDATE context_items SET content='Edited while queued.' WHERE id=?", (created['character']['id'],))
        job_id, request = self.run_job()
        body = json.loads(request.call_args[0][0].data)
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertIn('Character name: Aiko', body['messages'][1]['content'][2]['text'])
        self.assertIn('[100, 100, 500, 1000]', body['messages'][1]['content'][2]['text'])
        content = db.one('SELECT content FROM context_items WHERE id=?', (created['character']['id'],))['content']
        self.assertTrue(content.startswith('Edited while queued.\n\n## Character sheet\nHair: long black hair, straight fringe'), content)
        self.assertIn('## Outfit sheet (Episode, cut 1)\nTop: navy blazer over a white shirt', content)
        job = db.one('SELECT status,stage,payload_json FROM jobs WHERE id=?', (job_id,))
        self.assertEqual((job['status'], job['stage']), ('done', 'Complete · Details updated'))
        result = json.loads(job['payload_json'])['result']
        self.assertEqual((result['model_id'], result['profile'], result['usage']['prompt_tokens']), (visual_models.MODELS[visual_models.DEFAULTS['character_profile']].model_id, ANSWER, 900))
        self.assertEqual(result['prompt']['system'], character_profile.PROFILE_INSTRUCTION)

    def test_job_runs_the_profile_model_chosen_when_it_was_queued(self):
        with db.connect() as conn:  # the API refuses a model the worker has not reported; set it directly
            conn.execute("INSERT INTO visual_model_settings(role,model) VALUES ('character_profile','lfm3-omni-2b')")
        self.add_box()
        created = self.create().json()
        with db.connect() as conn:
            conn.execute("DELETE FROM visual_model_settings")  # changed after queuing: the job keeps its snapshot
        claimed = worker.claim_job(('character_profile',))
        with patch.object(character_profile, 'images', return_value=(b'frame', b'crop')), \
             patch.object(frame_model, 'model_server', return_value=nullcontext('http://model')) as server, \
             patch.object(inference_worker, 'request_json', return_value=model_reply()):
            character_profile.process(*claimed)
        model = server.call_args.args[0]  # the stage asks for the job's model; model_server starts its private server with its own arguments
        self.assertEqual(model.key, 'lfm3-omni-2b')
        self.assertTrue(str(model.path('model')).endswith('LFM3-Omni-2B-BF16.gguf'))
        self.assertEqual(model.server_args, ('--reasoning', 'off'))
        result = json.loads(db.one('SELECT payload_json FROM jobs WHERE id=?', (claimed[0],))['payload_json'])['result']
        self.assertEqual((result['model_id'], result['visual_model']), ('LiquidAI/LFM3-Omni-2B-GGUF:BF16', 'lfm3-omni-2b'))
        self.assertIn('_Described by LFM3-Omni-2B', db.one('SELECT content FROM context_items WHERE id=?', (created['character']['id'],))['content'])

    def test_edit_during_description_requeues_without_appending_old_text(self):
        self.add_box()
        created = self.create().json()
        claimed = worker.claim_job(('character_profile',))

        def describe(*args, **kwargs):
            with db.connect() as conn:
                conn.execute("UPDATE context_items SET content='Edited during generation.' WHERE id=?", (created['character']['id'],))
            return ANSWER

        with patch.object(character_profile, 'images', return_value=(b'frame', b'crop')), \
             patch.object(frame_model, 'model_server', return_value=nullcontext('http://model')), \
             patch.object(character_profile, 'describe', side_effect=describe):
            character_profile.process(*claimed)
        job = db.one('SELECT status,stage,attempt_count FROM jobs WHERE id=?', (claimed[0],))
        self.assertEqual((job['status'], job['stage'], job['attempt_count']), ('queued', 'Inputs changed, restarting', 0))
        self.assertEqual(db.one('SELECT content FROM context_items WHERE id=?', (created['character']['id'],))['content'], 'Edited during generation.')
        worker.release_lease(*claimed)
        with db.connect() as conn:
            conn.execute('UPDATE jobs SET not_before=0 WHERE id=?', (claimed[0],))
        _, request = self.run_job()
        self.assertIn('Edited during generation.', json.loads(request.call_args[0][0].data)['messages'][1]['content'][2]['text'])

    def test_removed_or_relabelled_box_is_skipped_without_the_model(self):
        self.add_box()
        created = self.create().json()
        with db.connect() as conn:
            conn.execute("UPDATE character_detections SET character_context_id='ctx' WHERE id='box'")
        job_id, request = self.run_job()
        request.assert_not_called()
        self.assertEqual(db.one('SELECT stage FROM jobs WHERE id=?', (job_id,))['stage'], 'Skipped · the box was labelled with another character')
        self.assertEqual(db.one('SELECT content FROM context_items WHERE id=?', (created['character']['id'],))['content'], 'The class president.')
        with db.connect() as conn:
            conn.execute("DELETE FROM character_detections WHERE id='box'")
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/retry").status_code, 409)  # done jobs are not retried

    def test_bad_answer_fails_and_retry_requeues(self):
        self.add_box()
        self.create()
        job_id, _ = self.run_job({'choices': [{'message': {'content': '{"character_sheet": {}}'}, 'finish_reason': 'stop'}]})
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('outfit_sheet', job['error'])
        retry = self.client.post(f'/api/jobs/{job_id}/retry')
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(self.client.post(f'/api/jobs/{job_id}/retry').json(), retry.json())

    def test_validate_fills_missing_fields_and_render_formats(self):
        profile = character_profile.validate({'character_sheet': {'hair': '  short   red '}, 'outfit_sheet': {'top': ''}})
        self.assertEqual(profile['character_sheet']['hair'], 'short red')
        self.assertEqual(profile['character_sheet']['eyes'], 'not visible')
        self.assertEqual(profile['outfit_sheet']['top'], 'not visible')
        text = character_profile.render(profile, 'Ep 1', 4, 'Qwen3.5-27B', '2026-11-17')
        self.assertEqual(text.splitlines()[0], '## Character sheet')
        self.assertIn('## Outfit sheet (Ep 1, cut 4)', text)
        self.assertTrue(text.endswith('_Described by Qwen3.5-27B from the first frame of cut 4 on 2026-11-17._'))
        with self.assertRaises(ValueError):
            character_profile.validate('[]')


if __name__ == '__main__':
    unittest.main()
