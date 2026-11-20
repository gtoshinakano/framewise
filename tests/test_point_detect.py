"""Double-click detection: choosing the box under a point, the worker thread and POST /api/point-detections."""
import json
import threading
import time
import unittest
from unittest.mock import patch

import tests.test_analysis as analysis_tests
from app import db, media, point_detect


class FakeImage:
    def __init__(self, size, origin=(0, 0)):
        self.size, self.origin = tuple(size), origin

    def crop(self, box):
        return FakeImage((box[2] - box[0], box[3] - box[1]), (box[0], box[1]))


class FakeDetector:
    """Answers per pass: ``full`` on the whole frame, ``crop`` (in crop coordinates) on a crop."""

    def __init__(self, full=(), crop=()):
        self.full, self.crop, self.calls = list(full), list(crop), []

    def detect(self, images, confidence=0.25, iou=0.7, scores=False):
        image = images[0]
        self.calls.append((image.size, confidence, scores))
        return [self.full if image.origin == (0, 0) and len(self.calls) == 1 else self.crop]


class PickTest(unittest.TestCase):
    def test_the_box_under_the_point_wins(self):
        found = [([0, 0, 50, 50], .9), ([100, 100, 200, 300], .4)]
        self.assertEqual(point_detect.pick(found, [150, 200], []), {'box': [100, 100, 200, 300], 'score': .4, 'pass': 'full'})
        self.assertIsNone(point_detect.pick(found, [400, 400], []))

    def test_a_person_already_boxed_is_skipped(self):
        found = [([100, 100, 200, 300], .9), ([90, 80, 260, 320], .5)]
        chosen = point_detect.pick(found, [150, 200], [[102, 98, 201, 302]])
        self.assertEqual(chosen['box'], [90, 80, 260, 320])
        self.assertIsNone(point_detect.pick(found[:1], [150, 200], [[102, 98, 201, 302]]))

    def test_close_scores_prefer_the_smaller_box(self):
        group, person = ([0, 0, 400, 400], .62), ([120, 100, 200, 300], .55)
        self.assertEqual(point_detect.pick([group, person], [150, 200], [])['box'], person[0])
        self.assertEqual(point_detect.pick([([0, 0, 400, 400], .9), person], [150, 200], [])['box'], [0, 0, 400, 400])

    def test_crop_stays_inside_the_frame(self):
        self.assertEqual(point_detect.crop_around([10, 10], [1280, 720]), [0, 0, 324, 324])
        self.assertEqual(point_detect.crop_around([1275, 715], [1280, 720]), [956, 396, 1280, 720])
        self.assertEqual(point_detect.crop_around([100, 100], [200, 150]), [25, 0, 175, 150])
        self.assertEqual(point_detect.uncrop([1, 2, 3, 4], [10, 20, 30, 40]), [11, 22, 13, 24])

    def test_detect_tries_the_crop_only_when_the_frame_misses(self):
        detector = FakeDetector(full=[([10, 10, 300, 700], .3)], crop=[([5, 5, 60, 150], .2)])
        chosen = point_detect.detect(FakeImage([1280, 720]), [100, 100], [], detector)
        self.assertEqual(chosen, {'box': [10, 10, 300, 700], 'score': .3, 'pass': 'full'})
        self.assertEqual(detector.calls, [((1280, 720), point_detect.CONFIDENCE, True)])

        detector = FakeDetector(full=[], crop=[([5, 5, 60, 150], .6), ([100, 40, 220, 330], .2)])
        chosen = point_detect.detect(FakeImage([1280, 720]), [1000, 500], [], detector)
        crop = point_detect.crop_around([1000, 500], [1280, 720])
        self.assertEqual(chosen['box'], [crop[0] + 100, crop[1] + 40, crop[0] + 220, min(720, crop[1] + 330)])
        self.assertEqual(chosen['pass'], 'crop')
        self.assertIsNone(point_detect.detect(FakeImage([1280, 720]), [10, 10], [], FakeDetector()))


class Base(unittest.TestCase):
    setUp_base = analysis_tests.AnalysisTest.setUp
    tearDown_base = analysis_tests.AnalysisTest.tearDown

    def setUp(self):
        self.setUp_base()
        folder = media.episode_dir('p', 'e')
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'original.mp4').write_bytes(b'video')
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('fr','c1',0,2,?,'test')",
                         (json.dumps([{'position': 'first', 'time': .04, 'image_size': [1280, 720]}]),))
            conn.execute("INSERT INTO character_detections(id,run_id,position,box_json) VALUES ('d1','fr','first','[0,0,100,200]')")
            conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('tr','c1','k',0,2,?)",
                         (json.dumps([{'ordinal': 0, 'time': 0, 'image_size': [640, 360], 'mirror': True},
                                      {'ordinal': 30, 'time': 1.25, 'image_size': [640, 360]}]),))
            conn.execute("INSERT INTO target_slots(run_id,slot,origin) VALUES ('tr',0,'appeared'),('tr',1,'appeared')")
            conn.execute("UPDATE target_slots SET ignored=1 WHERE run_id='tr' AND slot=1")
            conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json) VALUES ('t0','tr',30,0,'[1,1,50,90]'),('t1','tr',30,1,'[60,1,90,90]')")

    def tearDown(self):
        self.tearDown_base()

    def beat(self, when=None):
        with db.connect() as conn:
            conn.execute('UPDATE point_detection_state SET heartbeat=? WHERE id=1', (time.time() if when is None else when,))


class WorkerTest(Base):
    def insert(self, request_id='r', status='queued', lease_until=None):
        with db.connect() as conn:
            conn.execute('''INSERT INTO point_detection_requests(id,surface,run_id,frame_key,point_json,spec_json,image_size_json,existing_json,status,lease_owner,lease_until,created_at)
                            VALUES (?,'first','fr','first','[150,200]','{}','[1280,720]','[[0,0,100,200]]',?,?,?,?)''',
                         (request_id, status, 'old' if status == 'running' else None, lease_until, time.time()))

    def test_claim_and_answer(self):
        self.insert()
        row, owner = point_detect.claim()
        self.assertEqual(db.one("SELECT status FROM point_detection_requests WHERE id='r'")['status'], 'running')
        detector = FakeDetector(full=[([0, 0, 100, 200], .9), ([120, 100, 220, 400], .3)])
        point_detect.process(row, owner, lambda: detector, render=lambda _row: FakeImage([1280, 720]))
        done = db.one("SELECT * FROM point_detection_requests WHERE id='r'")
        self.assertEqual((done['status'], json.loads(done['box_json']), done['score'], done['pass']), ('done', [120, 100, 220, 400], .3, 'full'))
        self.assertIsNone(done['lease_owner'])

    def test_nothing_found_and_failure(self):
        self.insert('a'); self.insert('b')
        point_detect.process(*point_detect.claim(), lambda: FakeDetector(), render=lambda _row: FakeImage([1280, 720]))
        row = db.one("SELECT * FROM point_detection_requests WHERE id='a'")
        self.assertEqual((row['status'], row['box_json']), ('done', None))
        def broken(_row): raise RuntimeError('Could not extract frame')
        point_detect.process(*point_detect.claim(), lambda: FakeDetector(), render=broken)
        row = db.one("SELECT * FROM point_detection_requests WHERE id='b'")
        self.assertEqual(row['status'], 'failed')
        self.assertIn('Could not extract frame', row['error'])

    def test_an_expired_running_request_fails_on_claim(self):
        self.insert('stuck', status='running', lease_until=time.time() - 1)
        self.assertIsNone(point_detect.claim())
        self.assertEqual(db.one("SELECT status FROM point_detection_requests WHERE id='stuck'")['status'], 'failed')


class ApiTest(Base):
    def answer_with(self, detector):
        """Play the inference worker until the request is answered."""
        stop = threading.Event()

        def run():
            while not stop.is_set():
                claimed = point_detect.claim()
                if claimed:
                    point_detect.process(*claimed, lambda: detector, render=lambda _row: FakeImage(json.loads(_row['image_size_json'])))
                stop.wait(.02)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join(2)))

    def test_without_the_worker(self):
        response = self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'fr', 'position': 'first', 'point': [10, 10]})
        self.assertEqual(response.status_code, 503)
        self.beat(time.time() - 60)
        self.assertEqual(self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'fr', 'point': [10, 10]}).status_code, 503)

    def test_bad_requests(self):
        self.beat()
        self.assertEqual(self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'fr', 'point': [2000, 10]}).status_code, 400)
        self.assertEqual(self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'nope', 'point': [1, 1]}).status_code, 404)
        self.assertEqual(self.client.post('/api/point-detections', json={'surface': 'target', 'run_id': 'tr', 'frame_ordinal': 0, 'point': [1, 1]}).status_code, 409)
        self.assertEqual(self.client.post('/api/point-detections', json={'surface': 'target', 'run_id': 'tr', 'frame_ordinal': 7, 'point': [1, 1]}).status_code, 400)
        self.assertEqual(db.one('SELECT count(*) AS n FROM point_detection_requests')['n'], 0)

    def test_first_frame(self):
        self.beat()
        self.answer_with(FakeDetector(full=[([0, 0, 100, 200], .9), ([300, 100, 420, 600], .2)]))
        response = self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'fr', 'position': 'first', 'point': [350, 300]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'box': [300, 100, 420, 600], 'score': .2, 'pass': 'full'})
        row = db.one('SELECT * FROM point_detection_requests')
        spec = json.loads(row['spec_json'])
        self.assertEqual((spec['width'], spec['time'], spec['start'], spec['end']), (1280, .04, 0, 2))
        self.assertTrue(spec['source'].endswith('original.mp4'))
        self.assertEqual(json.loads(row['existing_json']), [[0, 0, 100, 200]])
        # Nothing was saved: the frontend adds the box through the run's add-box endpoint.
        self.assertEqual(db.one("SELECT count(*) AS n FROM character_detections WHERE run_id='fr'")['n'], 1)

    def test_target_frame_and_nobody_there(self):
        self.beat()
        self.answer_with(FakeDetector())
        response = self.client.post('/api/point-detections', json={'surface': 'target', 'run_id': 'tr', 'frame_ordinal': 30, 'point': [320, 180]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()['box'])
        row = db.one('SELECT * FROM point_detection_requests')
        spec = json.loads(row['spec_json'])
        self.assertEqual((spec['width'], spec['ordinal'], spec['time']), (640, 30, 1.25))
        # An ignored colour's box does not hide a person from the detector.
        self.assertEqual(json.loads(row['existing_json']), [[1, 1, 50, 90]])

    def test_failure_is_reported(self):
        self.beat()
        stop = threading.Event()

        def run():
            while not stop.is_set():
                claimed = point_detect.claim()
                if claimed:
                    def broken(_row): raise RuntimeError('AniRef weights are missing')
                    point_detect.process(*claimed, FakeDetector, render=broken)
                stop.wait(.02)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            response = self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'fr', 'point': [5, 5]})
        finally:
            stop.set(); thread.join(2)
        self.assertEqual(response.status_code, 502)
        self.assertIn('AniRef weights are missing', response.json()['detail'])

    def test_timeout(self):
        self.beat()
        with patch('app.point_detect_api.WAIT_SECONDS', .3):
            response = self.client.post('/api/point-detections', json={'surface': 'first', 'run_id': 'fr', 'point': [5, 5]})
        self.assertEqual(response.status_code, 504)
        self.assertEqual(db.one('SELECT status FROM point_detection_requests')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
