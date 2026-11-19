"""JSON responses are gzipped; media, ranges and small bodies are sent as they are."""
import unittest

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient

from app.json_gzip import JsonGzip

BIG = [{'text': f'line {n}', 'speaker_label': 'SPEAKER_00'} for n in range(500)]

app = FastAPI()
app.add_middleware(JsonGzip)


@app.get('/big')
def big():
    return BIG


@app.get('/small')
def small():
    return {'ok': True}


@app.get('/frame')
def frame():
    return Response(b'\xff' * 5000, media_type='image/jpeg')


@app.get('/range')
def video_range():
    return Response(b'{"a":1}' * 500, status_code=206, media_type='application/json')


class JsonGzipTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_large_json_is_compressed_and_decodes_to_the_same_value(self):
        response = self.client.get('/big', headers={'Accept-Encoding': 'gzip'})
        self.assertEqual(response.headers['content-encoding'], 'gzip')
        self.assertIn('Accept-Encoding', response.headers['vary'])
        self.assertLess(int(response.headers['content-length']), len(response.content) // 5)
        self.assertEqual(response.json(), BIG)

    def test_a_client_without_gzip_gets_plain_json(self):
        response = self.client.get('/big', headers={'Accept-Encoding': 'identity'})
        self.assertNotIn('content-encoding', response.headers)
        self.assertEqual(response.json(), BIG)

    def test_small_json_media_and_partial_responses_pass_through(self):
        for path in ('/small', '/frame', '/range'):
            response = self.client.get(path, headers={'Accept-Encoding': 'gzip'})
            self.assertNotIn('content-encoding', response.headers, path)
            self.assertEqual(int(response.headers['content-length']), len(response.content), path)


if __name__ == '__main__':
    unittest.main()
