import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

from app import db, media, web_search
from app.main import app

PATH = '/api/projects/p/character-registration'


class CharacterRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        directory = Path(self.tmp.name)
        for target, key, value in [(db, 'DATA_DIR', directory), (db, 'DB_PATH', directory / 'db.sqlite'),
                                   (media, 'DATA_DIR', directory)]:
            override = patch.object(target, key, value)
            override.start()
            self.addCleanup(override.stop)
        db.init_db()
        self.episode = str(uuid4())
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','Series'),('q','Other')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,duration,has_audio,settings_json)
                         VALUES(?,'p','Episode','x.mp4','.mp4',12,1,'{}')""", (self.episode,))
        (media.episode_dir('p', self.episode) / 'audio.wav').write_bytes(b'audio')
        self.client = TestClient(app)
        self.draft = {
            'character_id': str(uuid4()), 'save_id': str(uuid4()), 'title': 'Aoki Hana', 'content': 'Short dark hair.',
            'glossary': [{'part': 'family', 'spelling': '青木', 'reading': 'あおき'}],
            'outfits': [{'name': 'School uniform', 'description': 'Navy jacket.'}],
            'voice': [{'episode_id': self.episode, 'ranges': [{'start_time': 1, 'end_time': 3}, {'start_time': 4, 'end_time': 5}]}],
        }

    def save(self, draft=None, **kwargs):
        return self.client.post(PATH, data={'draft': json.dumps(draft or self.draft)}, **kwargs)

    def test_save_edit_thumbnail_and_idempotent_voice(self):
        picture = io.BytesIO()
        Image.new('RGB', (50, 80), 'green').save(picture, format='JPEG')
        response = self.save(files={'thumbnail': ('crop.jpg', picture.getvalue(), 'image/jpeg')})
        self.assertEqual(response.status_code, 200, response.text)
        saved = response.json()
        self.assertEqual(saved['glossary'][0]['reading'], 'あおき')
        self.assertEqual(saved['outfits'][0]['name'], 'School uniform')
        self.assertEqual(len(saved['voice_job_ids']), 1)
        self.assertEqual(self.save().json()['voice_job_ids'], saved['voice_job_ids'])
        character = PATH + '/' + self.draft['character_id']
        self.assertEqual(self.client.get(character + '/image').status_code, 200)
        self.assertEqual(self.client.get(character.replace('/p/', '/q/') + '/image').status_code, 404)
        edit = {**self.draft, 'version': saved['version'], 'title': 'Hana Aoki', 'glossary': saved['glossary'],
                'outfits': saved['outfits'], 'voice': []}
        response = self.save(edit)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.get(character).json()['character']['title'], 'Hana Aoki')
        self.assertEqual(self.save(edit).status_code, 409)
        self.assertEqual(self.client.get('/api/projects/p/context').json()[0]['id'], self.draft['character_id'])

    def test_invalid_voice_rolls_back_and_detection_save_uses_original_box(self):
        cut, run, box = (str(uuid4()) for _ in range(3))
        with db.connect() as conn:
            conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES(?,?,0,0,5)', (cut, self.episode))
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES(?,?,0,5,'[]','test')", (run, cut))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json) VALUES(?,?,'first','[0,0,20,40]')", (box, run))
        invalid = {**self.draft, 'detection_id': box, 'voice': [{'episode_id': self.episode, 'ranges': [{'start_time': 11, 'end_time': 15}]}]}
        self.assertEqual(self.save(invalid).status_code, 400)
        self.assertIsNone(db.one('SELECT character_context_id FROM character_detections WHERE id=?', (box,))['character_context_id'])
        self.assertEqual(self.client.get('/api/projects/p/context').json(), [])
        response = self.save({**self.draft, 'detection_id': box, 'voice': []})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()['profile_job_id'])
        self.assertEqual(db.one('SELECT character_context_id,label_source FROM character_detections WHERE id=?', (box,)),
                         {'character_context_id': self.draft['character_id'], 'label_source': 'manual'})

    def test_shared_sources_and_grounded_suggestions(self):
        sources = '/api/web-search/sources'
        self.assertEqual(len(self.client.get(sources).json()['sources']), 2)
        approved = [{'kind': 'page', 'url': 'https://example.com/characters/hana'}]
        self.assertEqual(self.client.put(sources, json={'sources': approved}).status_code, 200)
        hit = {'id': 1, 'url': approved[0]['url'], 'title': 'Hana', 'excerpt': '青木（あおき） has dark hair.'}
        raw = {'names': [{'part': 'family', 'spelling': '青木', 'reading': 'あおき',
                          'evidence': [{'source': 1, 'quote': '青木（あおき）'}]}],
               'appearance': {'value': 'Blue eyes.', 'evidence': [{'source': 1, 'quote': 'bright blue eyes'}]}}
        with patch('app.web_search.search', return_value={'hits': [hit], 'warnings': [], 'configured': True}) as search:
            self.assertEqual(self.client.post('/api/v1/tools/web-search', json={'query': 'Hana'}).json()['hits'], [hit])
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'mock-key'}), patch('app.web_search_api.request_json') as model:
                model.return_value = {'choices': [{'message': {'content': json.dumps(raw)}}]}
                response = self.client.post('/api/projects/q/character-research', json={'name': 'Hana', 'series': 'Other'})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['suggestions']['names'][0]['reading'], 'あおき')
                self.assertIsNone(response.json()['suggestions']['appearance'])
                request = json.loads(model.call_args.args[0].data)
                self.assertEqual(request['model'], 'openrouter/free')
                self.assertEqual(request['response_format']['type'], 'json_schema')
                search.assert_called_with('Hana Other', subject='Hana', selected_url=None)
                model.side_effect = [HTTPError('https://openrouter.ai', 400, 'Unsupported schema', {}, None), model.return_value]
                retry = self.client.post('/api/projects/q/character-research', json={'name': 'Hana', 'series': 'Other'})
                self.assertEqual(retry.status_code, 200, retry.text)
                self.assertEqual(json.loads(model.call_args.args[0].data)['model'], 'google/gemma-4-31b-it:free')


    def test_research_falls_back_to_the_local_model(self):
        hit = {'id': 1, 'url': 'https://example.com/characters/hana', 'title': 'Hana', 'excerpt': 'Hana has bright blue eyes.'}
        raw = {'appearance': {'value': 'Blue eyes.', 'evidence': [{'source': 1, 'quote': 'bright blue eyes'}]}}
        row = {'status': 'done', 'content': json.dumps(raw), 'error': None, 'model': 'qwen3.6-35b-a3b'}
        found = {'hits': [hit], 'warnings': [], 'configured': True}
        with patch('app.web_search.search', return_value=found), patch('app.web_search_api.chat_gateway.worker_online', return_value=True), \
                patch('app.web_search_api.chat_gateway.queue_request', return_value='r1') as queue, \
                patch('app.web_search_api.chat_gateway.snapshot', return_value=row), patch('app.web_search_api.chat_gateway.request_cancel'):
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'mock-key'}), patch('app.web_search_api.request_json', side_effect=URLError('down')):
                response = self.client.post('/api/projects/q/character-research', json={'name': 'Hana', 'series': 'Other'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['suggestions']['appearance']['value'], 'Blue eyes.')
            self.assertTrue(response.json()['model'].startswith('local:'))
            self.assertEqual(queue.call_args.args[0]['sampling']['response_format']['type'], 'json_schema')
            with patch.dict('os.environ', {'OPENROUTER_API_KEY': ''}):
                self.assertEqual(self.client.post('/api/projects/q/character-research', json={'name': 'Hana', 'series': 'Other'}).status_code, 200)
            with patch('app.web_search_api.chat_gateway.worker_online', return_value=False), \
                    patch.dict('os.environ', {'OPENROUTER_API_KEY': 'mock-key'}), patch('app.web_search_api.request_json', side_effect=URLError('down')):
                self.assertEqual(self.client.post('/api/projects/q/character-research', json={'name': 'Hana', 'series': 'Other'}).status_code, 502)

    def test_excerpts_prioritize_japanese_names_with_furigana(self):
        html = '<p>' + ('Unrelated series information. ' * 400) + '</p>'
        html += '<p><ruby>竈門<rp>(</rp><rt>かまど</rt><rp>)</rp></ruby> <ruby>炭治郎<rt>たんじろう</rt></ruby> is the protagonist.</p>'
        excerpt = web_search.Page(html).excerpt('竈門炭治郎')
        self.assertTrue(excerpt.startswith('竈門（かまど） 炭治郎（たんじろう）'))
        self.assertIn('竈門炭治郎', web_search.name_text(excerpt.splitlines()[0]))

    def test_scope_guard_blocks_redirect_and_private_dns(self):
        source = {'kind': 'directory', 'url': 'https://example.com/characters'}
        self.assertTrue(web_search.allowed(source, 'https://example.com/characters/hana'))
        self.assertFalse(web_search.allowed(source, 'https://example.com/characters-other'))
        self.assertFalse(web_search.allowed(source, 'https://other.example.com/characters'))
        with patch('app.web_search.socket.getaddrinfo', return_value=[(None, None, None, None, ('192.168.0.1', 443))]):
            with self.assertRaises(ValueError):
                web_search.public_addresses('example.com')
        with patch('app.web_search.PublicHTTPS') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 302
            response.getheader.return_value = 'https://other.example.com/characters'
            with self.assertRaises(ValueError):
                web_search.fetch(source, source['url'], float('inf'))
            self.assertEqual(connection.call_count, 1)
