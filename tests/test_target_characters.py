import json
import subprocess
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import tests.test_analysis as analysis_tests
from app import anime_bulk, anime_gallery, anime_models, db, job_kinds, keyframes, media, target_characters, worker

RED, BLUE, GREEN, GOLD = (220, 0, 0), (0, 0, 220), (0, 200, 0), (230, 200, 0)
PALETTE = [RED, BLUE, GREEN, GOLD]
SIZE = (240, 80)
A, B, C, D = [5, 5, 55, 75], [65, 5, 115, 75], [125, 5, 175, 75], [185, 5, 235, 75]
ORDINALS = (10, 15, 20)


def scene(people):
    image = Image.new('RGB', SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for box, color in people:
        draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=color)
    return image


class Embedder:
    """The vector says which palette colour a crop is, so different colours are exactly unrelated."""
    model_id = 'test/ccip'
    revision = 'emb-sha'
    preprocessing = 'test-v1'
    label = 'CCIP'
    defaults = dict(anime_models.CCIP_DEFAULTS)

    def embed(self, images):
        vectors = []
        for image in images:
            pixels = list(image.convert('RGB').getdata())
            mean = [sum(p[i] for p in pixels) / len(pixels) for i in range(3)]
            nearest = min(range(len(PALETTE)), key=lambda n: sum((mean[i] - PALETTE[n][i]) ** 2 for i in range(3)))
            vectors.append([1. if n == nearest else 0. for n in range(len(PALETTE))])
        return vectors


class Detector:
    revision = 'aniref-sha'

    def __init__(self, answers):
        self.answers = answers  # id(image) -> the boxes AniRef finds in that frame, best first
        self.seen = 0

    def detect(self, images):
        self.seen += len(images)
        return [list(self.answers[id(image)]) for image in images]


class Runtime(anime_bulk.BulkRuntime):
    def __init__(self, answers):
        super().__init__()
        self.detector = Detector(answers)
        self.stub = Embedder()

    def get_detector(self):
        return self.detector

    def get_embedder(self):
        return self.stub


class TargetCharactersTest(unittest.TestCase):
    tearDown = analysis_tests.AnalysisTest.tearDown

    def setUp(self):
        analysis_tests.AnalysisTest.setUp(self)
        with db.connect() as conn:
            conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('ctx2','p','character','Other')")

    # --- fixtures -------------------------------------------------------------------------------------------
    def evaluate(self, cut_id='c1', ordinals=ORDINALS, first_reason='initial_state'):
        """A current evaluation of the cut, without a video decode."""
        source = media.episode_dir('p', 'e') / 'original.mp4'
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():  # the cache key includes the file's mtime
            source.write_bytes(b'video')
        settings = keyframes.config({})
        band = keyframes.mask_for('e', settings)
        cut = db.one('SELECT id,start_time,end_time FROM cuts WHERE id=?', (cut_id,))
        key = keyframes.cache_key(source, cut, settings, band)
        frames = [{'source_frame_index': ordinal, 'frame_index': number, 'timestamp_seconds': cut['start_time'] + .1 * number,
                   'relative_timestamp_seconds': .1 * number, 'reason': first_reason if number == 0 else 'visual_state'} for number, ordinal in enumerate(ordinals)]
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status,payload_json) VALUES (?,'e','keyframes','done',?)",
                         (f'kf-{cut_id}', json.dumps({'keyframe_settings': settings, 'subtitle_mask': band, 'cuts': [cut]})))
            conn.execute('INSERT INTO keyframe_results(cache_key,episode_id,cut_id,result_json) VALUES (?,?,?,?)',
                         (key, 'e', cut_id, json.dumps({'selected_frames': frames})))
        return key

    def first_frame(self, cut_id, people):
        """The reviewed first-frame run: people = [(box, character id or None, label source)] in editor order."""
        cut = db.one('SELECT start_time,end_time FROM cuts WHERE id=?', (cut_id,))
        with db.connect() as conn:
            conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES (?,?,?,?,?,'test')",
                         (f'fr-{cut_id}', cut_id, cut['start_time'], cut['end_time'], json.dumps([{'position': 'first', 'time': cut['start_time'], 'image_size': list(SIZE)}])))
            for number, (box, character_id, source) in enumerate(people):
                conn.execute("INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES (?,?,'first',?,?,?)",
                             (f'fd-{cut_id}-{number}', f'fr-{cut_id}', json.dumps(box), character_id, source))

    def detect(self, frames, first=None, cut_id='c1', gallery_colors=None):
        """Queue detection for one cut and run the job with a stub detector. ``frames``: [[(box, colour)]] per evaluated frame."""
        response = self.client.post(f'/api/cuts/{cut_id}/target-detection')
        self.assertEqual(response.status_code, 200, response.text)
        job_id, owner = worker.claim_job(('character_detect',))
        images = [scene(people) for people in frames]
        runtime = Runtime({id(image): [box for box, _ in people] for image, people in zip(images, frames)})
        fake_crops = lambda rows, scratch, lease_check=None: {row['id']: self.crop(gallery_colors[row['id']]) for row in rows}
        with patch.object(target_characters, '_extract_frames', return_value=images), \
             patch.object(target_characters, '_first_frame_image', return_value=scene(first or [])), \
             patch.object(anime_gallery, 'extract_crops', side_effect=fake_crops):
            target_characters.process(job_id, owner, runtime)
        return db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))

    def crop(self, color):
        import io
        output = io.BytesIO()
        Image.new('RGB', (40, 70), color).save(output, format='JPEG')
        return output.getvalue()

    def state(self, cut_id='c1'):
        response = self.client.get(f'/api/cuts/{cut_id}/target-detections')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def layout(self, state):
        """{frame ordinal: {slot: box}}"""
        result = {}
        for item in state['run']['detections']:
            result.setdefault(item['frame_ordinal'], {})[item['slot']] = item['box']
        return result

    # --- detection ------------------------------------------------------------------------------------------
    def test_slots_follow_the_first_frame_order_and_inherit_its_labels(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual'), (B, None, None)])
        job = self.detect([[(B, BLUE), (A, RED)], [(A, RED), (B, BLUE)], [(B, BLUE)]], first=[(A, RED), (B, BLUE)])
        self.assertEqual(job['status'], 'done', job)
        state = self.state()
        self.assertEqual([(s['slot'], s['character_context_id'], s['label_source'], s['origin']) for s in state['run']['slots']],
                         [(0, 'ctx', 'inherited', 'first_frame'), (1, None, None, 'first_frame')])
        layout = self.layout(state)
        # The detector listed blue first in frame 10, yet red is slot 0 everywhere; red has left in frame 20.
        self.assertEqual({slot: box for slot, box in layout[10].items()}, {0: A, 1: B})
        self.assertEqual(layout[15], {0: A, 1: B})
        self.assertEqual(layout[20], {1: B})
        self.assertFalse(state['run']['stale'])
        self.assertEqual([frame['ordinal'] for frame in state['run']['frames']], list(ORDINALS))

    def test_without_first_frame_people_the_first_evaluated_frame_sets_the_order(self):
        self.evaluate()
        job = self.detect([[(A, RED), (B, BLUE)], [(B, BLUE), (A, RED)], [(B, BLUE), (A, RED)]])
        self.assertEqual(job['status'], 'done', job)
        state = self.state()
        self.assertEqual([(s['slot'], s['origin'], s['character_context_id']) for s in state['run']['slots']], [(0, 'appeared', None), (1, 'appeared', None)])
        self.assertEqual(self.layout(state)[15], {0: A, 1: B})

    def test_a_frame_never_stores_more_than_three_boxes_and_known_people_come_first(self):
        self.evaluate()
        self.first_frame('c1', [(D, 'ctx', 'manual')])
        people = [(A, RED), (B, BLUE), (C, GREEN), (D, GOLD)]
        job = self.detect([people, people, people], first=[(D, GOLD)])
        self.assertEqual(job['status'], 'done', job)
        layout = self.layout(self.state())
        self.assertEqual(layout[10], {0: D})  # the first frame is shown as the user left it, whatever the detector would find
        for ordinal in ORDINALS[1:]:
            self.assertEqual(len(layout[ordinal]), 3)
            self.assertIn(0, layout[ordinal])  # the first-frame person, who is known, is never the one left out

    def test_people_who_appear_later_get_a_slot_and_a_name_from_the_gallery_without_joining_it(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        # The user's own label of "Other" in another cut: a green person.
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,analysis_enabled) VALUES ('g1','e',2,10,12,0)")
        self.first_frame('g1', [(A, 'ctx2', 'manual')])
        job = self.detect([[(A, RED)], [(A, RED), (C, GREEN)], [(C, GREEN)]], first=[(A, RED)], gallery_colors={'fd-g1-0': GREEN})
        self.assertEqual(job['status'], 'done', job)
        slots = {s['slot']: s for s in self.state()['run']['slots']}
        self.assertEqual((slots[1]['origin'], slots[1]['character_context_id'], slots[1]['label_source']), ('appeared', 'ctx2', 'auto'))
        # Reading the gallery added nothing to it: target boxes are not examples.
        self.assertEqual(sorted(row['id'] for row in anime_gallery.exemplar_rows('p')), ['fd-c1-0', 'fd-g1-0'])
        self.assertEqual(db.one('SELECT count(*) AS n FROM character_detections')['n'], 2)

    def test_detecting_again_keeps_what_the_user_did(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual'), (B, None, None)])
        frames = [[(A, RED), (B, BLUE)]] * 3
        self.assertEqual(self.detect(frames, first=[(A, RED), (B, BLUE)])['status'], 'done')
        run_id = self.state()['run']['id']
        self.assertEqual(self.client.put(f'/api/target-slots/{run_id}/1', json={'character_context_id': 'ctx2'}).status_code, 200)
        # A box the detector missed, drawn on the last frame for the colour that is free there.
        self.client.delete(f"/api/target-detections/{next(d['id'] for d in self.state()['run']['detections'] if d['frame_ordinal'] == 20 and d['slot'] == 1)}")
        drawn = self.client.post(f'/api/target-runs/{run_id}/boxes', json={'frame_ordinal': 20, 'box': [70, 10, 110, 70]})
        self.assertEqual(drawn.json()['slot'], 1)
        self.assertEqual(self.detect(frames, first=[(A, RED), (B, BLUE)])['status'], 'done')
        state = self.state()
        self.assertEqual(state['run']['id'], run_id)
        slots = {s['slot']: s for s in state['run']['slots']}
        self.assertEqual((slots[1]['character_context_id'], slots[1]['label_source']), ('ctx2', 'manual'))
        self.assertEqual((slots[0]['character_context_id'], slots[0]['label_source']), ('ctx', 'inherited'))
        manual = [d for d in state['run']['detections'] if d['source'] == 'manual']
        self.assertEqual([(d['frame_ordinal'], d['slot'], d['box']) for d in manual], [(20, 1, [70, 10, 110, 70])])
        # The detector's own box for that person in frame 20 was not added on top of the user's.
        self.assertEqual(len([d for d in state['run']['detections'] if d['frame_ordinal'] == 20 and d['slot'] == 1]), 1)
        self.assertEqual(db.one('SELECT count(*) AS n FROM target_detection_runs')['n'], 1)

    def test_a_slot_the_user_edited_moves_aside_when_the_first_frame_later_claims_its_number(self):
        self.evaluate()
        frames = [[(A, RED)]] * 3
        self.assertEqual(self.detect(frames)['status'], 'done')  # no first-frame people yet: the red person is slot 0
        run = self.state()['run']
        self.client.put(f"/api/target-slots/{run['id']}/0", json={'character_context_id': 'ctx2'})
        self.client.delete(f"/api/target-detections/{next(d['id'] for d in run['detections'] if d['frame_ordinal'] == 20)}")
        drawn = self.client.post(f"/api/target-runs/{run['id']}/boxes", json={'frame_ordinal': 20, 'box': [70, 10, 110, 70]})
        self.assertEqual(drawn.json()['slot'], 0)
        self.first_frame('c1', [(A, 'ctx', 'manual')])  # the user reviewed the first frame afterwards
        self.assertEqual(self.detect(frames, first=[(A, RED)])['status'], 'done')
        state = self.state()
        slots = {s['slot']: s for s in state['run']['slots']}
        self.assertEqual((slots[0]['origin'], slots[0]['character_context_id'], slots[0]['label_source']), ('first_frame', 'ctx', 'inherited'))
        self.assertEqual((slots[1]['origin'], slots[1]['character_context_id'], slots[1]['label_source']), ('appeared', 'ctx2', 'manual'))
        self.assertEqual(self.layout(state)[10], {0: A})
        self.assertEqual(self.layout(state)[20], {1: [70, 10, 110, 70]})  # the user removed A from this frame
        self.assertEqual([(d['frame_ordinal'], d['slot']) for d in state['run']['detections'] if d['source'] == 'manual'], [(20, 1)])

    # --- the first frame, exactly as the user left it ---------------------------------------------------------
    def test_the_first_evaluated_frame_is_the_first_frame_exactly_as_the_user_left_it(self):
        self.evaluate()
        odd = [5.25, 5.5, 54.75, 74.125]
        self.first_frame('c1', [(odd, 'ctx', 'manual'), (B, None, None)])
        # What the detector would say about that frame (a stranger, no one else) is not used.
        job = self.detect([[(C, GREEN)], [(A, RED), (B, BLUE)], [(A, RED), (B, BLUE)]], first=[(A, RED), (B, BLUE)])
        self.assertEqual(job['status'], 'done', job)
        state = self.state()
        self.assertEqual(self.layout(state)[10], {0: odd, 1: B})
        self.assertEqual([frame.get('mirror', False) for frame in state['run']['frames']], [True, False, False])
        self.assertEqual([(s['slot'], s['character_context_id']) for s in state['run']['slots']], [(0, 'ctx'), (1, None)])  # the stranger is no slot
        self.assertEqual(self.layout(state)[15], {0: A, 1: B})  # the frames after it follow the same slots
        self.assertEqual({d['source'] for d in state['run']['detections']}, {'auto'})

    def test_a_first_frame_left_without_people_stays_that_way(self):
        self.evaluate()
        self.first_frame('c1', [])  # reviewed: nobody is in it
        job = self.detect([[(A, RED)], [(A, RED)], [(A, RED), (B, BLUE)]], first=[])
        self.assertEqual(job['status'], 'done', job)
        state = self.state()
        self.assertNotIn(10, self.layout(state))
        self.assertTrue(state['run']['frames'][0]['mirror'])
        # Whoever shows up later still gets colours, in the order they are found.
        self.assertEqual(self.layout(state)[20], {0: A, 1: B})
        self.assertEqual({s['origin'] for s in state['run']['slots']}, {'appeared'})

    def test_a_first_frame_that_was_never_detected_is_detected_like_any_other(self):
        self.evaluate()
        self.assertEqual(self.detect([[(A, RED)]] * 3)['status'], 'done')
        state = self.state()
        self.assertEqual(self.layout(state)[10], {0: A})
        self.assertFalse(any(frame.get('mirror') for frame in state['run']['frames']))

    def test_the_selector_only_vouches_for_the_first_frame_when_it_says_so(self):
        self.evaluate(first_reason='visual_state')
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        self.assertEqual(self.detect([[(A, RED), (B, BLUE)]] * 3, first=[(A, RED)])['status'], 'done')
        state = self.state()
        self.assertEqual(self.layout(state)[10], {0: A, 1: B})
        self.assertFalse(any(frame.get('mirror') for frame in state['run']['frames']))

    def test_that_frame_cannot_be_edited_here_but_its_people_can_be_named(self):
        run = self.detected()
        shown = [d for d in run['detections'] if d['frame_ordinal'] == 10]
        self.assertEqual(len(shown), 2)
        self.assertEqual(self.client.put(f"/api/target-detections/{shown[0]['id']}/box", json={'box': [6, 6, 56, 76]}).status_code, 409)
        self.assertEqual(self.client.put(f"/api/target-detections/{shown[0]['id']}/slot", json={'slot': shown[1]['slot']}).status_code, 409)
        self.assertEqual(self.client.delete(f"/api/target-detections/{shown[0]['id']}").status_code, 409)
        self.assertEqual(self.client.post(f"/api/target-runs/{run['id']}/boxes", json={'frame_ordinal': 10, 'box': [130, 5, 170, 70]}).status_code, 409)
        self.assertEqual(self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': 'ctx2'}).status_code, 200)
        self.assertEqual(len([d for d in self.state()['run']['detections'] if d['frame_ordinal'] == 10]), 2)

    def test_detecting_again_copies_the_first_frame_as_it_is_now(self):
        run = self.detected()
        with db.connect() as conn:  # a box from before this frame was read only
            conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json,source) VALUES ('old',?,10,1,'[1,1,9,9]','manual')", (run['id'],))
            conn.execute("DELETE FROM character_detections WHERE id='fd-c1-1'")  # the user removed the second person from the first frame
            conn.execute("UPDATE character_detections SET box_json='[10, 10, 60, 70]' WHERE id='fd-c1-0'")
        self.assertEqual(self.detect([[(A, RED), (B, BLUE)]] * 3, first=[([10, 10, 60, 70], RED)])['status'], 'done')
        layout = self.layout(self.state())
        self.assertEqual(layout[10], {0: [10, 10, 60, 70]})
        self.assertNotIn('old', {d['id'] for d in self.state()['run']['detections']})

    def test_a_changed_evaluation_fails_the_job_and_changes_nothing(self):
        key = self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        self.assertEqual(self.detect([[(A, RED)]] * 3, first=[(A, RED)])['status'], 'done')
        before = self.state()['run']
        self.client.post('/api/cuts/c1/target-detection')
        with db.connect() as conn:
            conn.execute('DELETE FROM keyframe_results WHERE cache_key=?', (key,))
        job_id, owner = worker.claim_job(('character_detect',))
        target_characters.process(job_id, owner, Runtime({}))
        job = db.one('SELECT status,error FROM jobs WHERE id=?', (job_id,))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('evaluation', job['error'])
        self.assertEqual(self.state()['run']['detections'], before['detections'])
        self.assertTrue(self.state()['run']['stale'])

    # --- queueing -------------------------------------------------------------------------------------------
    def test_one_cut_needs_an_evaluation_and_cannot_be_queued_twice(self):
        self.assertEqual(self.client.post('/api/cuts/c1/target-detection').status_code, 409)
        self.evaluate()
        first = self.client.post('/api/cuts/c1/target-detection')
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(self.client.post('/api/cuts/c1/target-detection').status_code, 409)
        job = db.one('SELECT * FROM jobs WHERE id=?', (first.json()['job_id'],))
        payload = json.loads(job['payload_json'])
        self.assertEqual((job['pipeline'], payload['workflow'], payload['cut_id']), ('character_detect', 'target_frames_v1', 'c1'))
        self.assertEqual(self.client.post('/api/cuts/missing/target-detection').status_code, 404)

    def test_many_cuts_queue_the_evaluated_ones_and_report_the_rest(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,no_characters) VALUES ('c3','e',2,4,6,1)")
            conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,analysis_enabled) VALUES ('c4','e',3,6,8,0)")
        self.evaluate('c1')
        self.evaluate('c3')
        self.evaluate('c4')
        result = self.client.post('/api/episodes/e/target-detection').json()
        self.assertEqual(result['queued'], 1, result)
        self.assertEqual(result['skipped'], {'not_evaluated': [2], 'analysis_disabled': [4], 'no_characters': [3], 'up_to_date': [], 'already_queued': []})
        self.assertEqual(self.client.post('/api/episodes/e/target-detection').json()['skipped']['already_queued'], [1])
        self.assertEqual(self.client.post('/api/episodes/e/target-detection?from_cut=2').json()['queued'], 0)
        self.assertEqual(self.client.post('/api/episodes/missing/target-detection').status_code, 404)
        # Only an explicit request for the cut ignores the "no characters" mark.
        self.assertEqual(self.client.post('/api/cuts/c3/target-detection').status_code, 200)

    def test_finished_cuts_are_skipped_unless_everything_is_asked_for(self):
        self.evaluate()
        self.assertEqual(self.detect([[(A, RED)]] * 3)['status'], 'done')
        result = self.client.post('/api/episodes/e/target-detection').json()
        self.assertEqual((result['queued'], result['skipped']['up_to_date']), (0, [1]))
        self.assertEqual(self.client.post('/api/episodes/e/target-detection?only_missing=false').json()['queued'], 1)

    def test_other_episode_work_holds_the_queue(self):
        self.evaluate()
        with db.connect() as conn:
            conn.execute("INSERT INTO jobs(id,episode_id,pipeline,status) VALUES ('busy','e','analyze','running')")
        self.assertEqual(self.client.post('/api/cuts/c1/target-detection').status_code, 409)

    # --- editing --------------------------------------------------------------------------------------------
    def detected(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual'), (B, None, None)])
        self.assertEqual(self.detect([[(A, RED), (B, BLUE)]] * 3, first=[(A, RED), (B, BLUE)])['status'], 'done')
        return self.state()['run']

    def test_labelling_a_slot_labels_every_box_of_that_colour(self):
        run = self.detected()
        self.assertEqual(self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': 'ctx2'}).status_code, 200)
        slots = {s['slot']: s for s in self.state()['run']['slots']}
        self.assertEqual((slots[1]['character_name'], slots[1]['label_source']), ('Other', 'manual'))
        self.assertEqual(self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': None}).status_code, 200)
        self.assertEqual({s['slot']: s['character_context_id'] for s in self.state()['run']['slots']}, {0: 'ctx', 1: None})
        self.assertEqual(self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': 'other'}).status_code, 400)  # not a character
        self.assertEqual(self.client.put(f"/api/target-slots/{run['id']}/7", json={'character_context_id': 'ctx'}).status_code, 404)
        # The first frame's own label did not move.
        self.assertEqual(db.one("SELECT character_context_id FROM character_detections WHERE id='fd-c1-0'")['character_context_id'], 'ctx')

    def split(self):
        """A person reappears across the frame and looks different enough to get a second slot."""
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        self.assertEqual(self.detect([[(A, RED)], [(A, RED)], [(B, BLUE)]], first=[(A, RED)])['status'], 'done')
        state = self.state()
        self.assertEqual([(s['slot'], s['origin']) for s in state['run']['slots']], [(0, 'first_frame'), (1, 'appeared')])
        return state['run']

    def test_saying_a_colour_is_another_person_merges_all_its_boxes_into_one_colour(self):
        run = self.split()
        merged = self.client.post(f"/api/target-slots/{run['id']}/1/merge", json={'into': 0})
        self.assertEqual((merged.status_code, merged.json()), (200, {'moved': 1, 'dropped': 0}))
        state = self.state()
        self.assertEqual([s['slot'] for s in state['run']['slots']], [0])
        self.assertEqual(self.layout(state)[20], {0: B})
        self.assertEqual({d['slot'] for d in state['run']['detections']}, {0})
        self.assertEqual([(d['frame_ordinal'], d['source']) for d in state['run']['detections'] if d['source'] == 'manual'], [(20, 'manual')])
        # Nobody is left without a name on the timeline.
        self.assertFalse(next(c for c in self.client.get('/api/episodes/e').json()['cuts'] if c['id'] == 'c1')['has_unlabeled_target_boxes'])

    def test_a_merge_survives_detecting_again(self):
        run = self.split()
        self.client.post(f"/api/target-slots/{run['id']}/1/merge", json={'into': 0})
        self.assertEqual(self.detect([[(A, RED)], [(A, RED)], [(B, BLUE)]], first=[(A, RED)])['status'], 'done')
        state = self.state()
        self.assertEqual([s['slot'] for s in state['run']['slots']], [0])
        self.assertEqual(self.layout(state)[20], {0: B})

    def test_where_both_colours_have_a_box_in_a_frame_one_stays(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        job = self.detect([[(A, RED)], [(A, RED), (B, BLUE)], [(A, RED), (B, BLUE)]], first=[(A, RED)])
        self.assertEqual(job['status'], 'done', job)
        run = self.state()['run']
        self.assertEqual([s['slot'] for s in run['slots']], [0, 1])
        merged = self.client.post(f"/api/target-slots/{run['id']}/1/merge", json={'into': 0})
        self.assertEqual(merged.json(), {'moved': 0, 'dropped': 2})  # the target's own boxes stay
        self.assertEqual(self.layout(self.state())[15], {0: A})

    def test_the_box_the_user_placed_wins_a_clash(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        self.assertEqual(self.detect([[(A, RED)], [(A, RED), (B, BLUE)], [(A, RED), (B, BLUE)]], first=[(A, RED)])['status'], 'done')
        run = self.state()['run']
        mine = next(d for d in run['detections'] if d['frame_ordinal'] == 15 and d['slot'] == 1)
        self.client.put(f"/api/target-detections/{mine['id']}/box", json={'box': [66, 6, 116, 76]})
        merged = self.client.post(f"/api/target-slots/{run['id']}/1/merge", json={'into': 0})
        self.assertEqual(merged.json(), {'moved': 1, 'dropped': 2})
        layout = self.layout(self.state())
        self.assertEqual(layout[15], {0: [66, 6, 116, 76]})  # the box the user moved took the colour; the detector's was dropped
        self.assertEqual(layout[20], {0: A})

    def test_the_merged_colour_keeps_its_name_or_takes_the_other_ones(self):
        self.evaluate()
        self.assertEqual(self.detect([[(A, RED), (B, BLUE)], [(A, RED)], [(B, BLUE)]])['status'], 'done')
        run = self.state()['run']
        self.assertEqual([s['slot'] for s in run['slots']], [0, 1])
        self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': 'ctx2'})
        self.assertEqual(self.client.post(f"/api/target-slots/{run['id']}/1/merge", json={'into': 0}).status_code, 200)
        slot = self.state()['run']['slots'][0]
        self.assertEqual((slot['slot'], slot['character_context_id'], slot['label_source']), (0, 'ctx2', 'manual'))

    def test_a_first_frame_person_and_bad_requests_cannot_be_merged(self):
        run = self.split()
        merge = lambda slot, body: self.client.post(f"/api/target-slots/{run['id']}/{slot}/merge", json=body).status_code
        self.assertEqual(merge(0, {'into': 1}), 409)   # in the first frame as a person of their own
        self.assertEqual(merge(1, {'into': 1}), 400)
        self.assertEqual(merge(1, {'into': 6}), 404)
        self.assertEqual(merge(5, {'into': 0}), 404)
        self.client.post('/api/cuts/c1/target-detection')
        self.assertEqual(merge(1, {'into': 0}), 409)   # a detection job of the cut is pending
        self.assertEqual([s['slot'] for s in self.state()['run']['slots']], [0, 1])

    def test_locked_cast_keeps_an_extra_unmatched_and_promotion_is_manual_on_rerun(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        self.assertEqual(self.client.put('/api/cuts/c1/target-cast', json={'locked': True}).status_code, 200)
        frames = [[(A, RED)], [(A, RED), (B, BLUE)], [(A, RED), (B, BLUE)]]
        self.assertEqual(self.detect(frames, first=[(A, RED)])['status'], 'done')
        run = self.state()['run']
        self.assertTrue(run['cast_locked'])
        self.assertEqual(run['unmatched_count'], 2)
        self.assertEqual(self.layout(self.state())[15], {0: A})
        review = run['frames'][1]['review']
        self.assertIn('unmatched', review['flags'])
        candidate = next(item for item in run['candidates'] if item['frame_ordinal'] == 15)
        promoted = self.client.post(f"/api/target-runs/{run['id']}/candidates/{candidate['id']}", json={'action': 'promote'})
        self.assertEqual(promoted.status_code, 200, promoted.text)
        manual_id, = promoted.json()['boxes']
        self.assertEqual(self.detect(frames, first=[(A, RED)])['status'], 'done')
        rows = [row for row in self.state()['run']['detections'] if row['id'] == manual_id]
        self.assertEqual([(row['slot'], row['box'], row['source']) for row in rows], [(1, B, 'manual')])

    def test_ignore_and_swap_survive_detection_and_keep_the_opening_mirror(self):
        run = self.detected()
        swap = self.client.post(f"/api/target-slots/{run['id']}/0/swap-from", json={'other': 1, 'frame_ordinal': 15})
        self.assertEqual((swap.status_code, swap.json()), (200, {'swapped': 4}))
        ignore = f"/api/target-slots/{run['id']}/1/ignored"
        self.assertEqual(self.client.put(ignore, json={'ignored': True}).status_code, 200)
        frames = [[(A, RED), (B, BLUE)]] * 3
        self.assertEqual(self.detect(frames, first=frames[0])['status'], 'done')
        self.assertEqual(self.layout(self.state()), {10: {0: A}, 15: {0: B}, 20: {0: B}})
        self.assertEqual(self.client.put(ignore, json={'ignored': False}).status_code, 200)
        self.assertEqual(self.layout(self.state()), {10: {0: A, 1: B}, 15: {0: B, 1: A}, 20: {0: B, 1: A}})
        history = db.all_rows('SELECT action,before_json,after_json FROM target_corrections ORDER BY id')
        self.assertEqual([item['action'] for item in history], ['swap', 'ignore', 'restore'])
        before = json.loads(history[0]['before_json'])
        after = json.loads(history[0]['after_json'])
        self.assertEqual({(a['slot'], b['slot']) for a, b in zip(before, after)}, {(0, 1), (1, 0)})
        self.assertTrue(all(item['scores'] is not None for item in before))

    def test_rejected_candidate_and_review_marker_are_backed_by_the_run(self):
        self.evaluate()
        self.first_frame('c1', [(A, 'ctx', 'manual')])
        self.client.put('/api/cuts/c1/target-cast', json={'locked': True})
        frames = [[(A, RED)], [(A, RED), (B, BLUE)], [(A, RED)]]
        self.assertEqual(self.detect(frames, first=frames[0])['status'], 'done')
        run = self.state()['run']
        candidate, = run['candidates']
        self.assertEqual(self.client.put(f"/api/target-runs/{run['id']}/frames/15/reviewed", json={'reviewed': True}).status_code, 200)
        self.assertTrue(self.state()['run']['frames'][1]['review']['reviewed'])
        self.assertEqual(self.client.post(f"/api/target-runs/{run['id']}/candidates/{candidate['id']}", json={'action': 'reject'}).status_code, 200)
        self.assertEqual(self.detect(frames, first=frames[0])['status'], 'done')
        self.assertEqual(self.state()['run']['candidates'], [])

    def test_accepting_a_split_cannot_overwrite_a_manual_box(self):
        run = self.detected()
        selected = next(box for box in run['detections'] if box['frame_ordinal'] == 15 and box['slot'] == 0)
        self.client.put(f"/api/target-detections/{selected['id']}/box", json={'box': A})
        candidate = {'id': 'merge', 'tag': 'merge', 'box': [5, 5, 115, 75], 'frame_ordinal': 15, 'source': 'tracker',
                     'parts': [{'slot_hint': 0, 'box': A}, {'slot_hint': 1, 'box': B}]}
        with db.connect() as conn:
            conn.execute('UPDATE target_detection_runs SET candidates_json=? WHERE id=?', (json.dumps([candidate]), run['id']))
        response = self.client.post(f"/api/target-runs/{run['id']}/candidates/merge", json={'action': 'split'})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIsNotNone(db.one("SELECT id FROM target_detections WHERE id=? AND source='manual'", (selected['id'],)))

    def test_drawing_moving_swapping_and_removing_boxes(self):
        run = self.detected()
        boxes = f"/api/target-runs/{run['id']}/boxes"
        in_frame = [d for d in run['detections'] if d['frame_ordinal'] == 15]
        self.assertEqual(self.client.post(boxes, json={'frame_ordinal': 15, 'box': [1, 1, 20, 20], 'slot': 0}).status_code, 409)  # colour in use
        third = self.client.post(boxes, json={'frame_ordinal': 15, 'box': [130, 5, 170, 70]})
        self.assertEqual((third.status_code, third.json()['slot'], third.json()['created_slot']), (200, 2, True))
        self.assertEqual(self.client.post(boxes, json={'frame_ordinal': 15, 'box': [180, 5, 230, 70]}).status_code, 409)  # a fourth box
        self.assertEqual(self.client.post(boxes, json={'frame_ordinal': 15, 'box': [0, 0, 500, 20]}).status_code, 400)    # outside the frame
        self.assertEqual(self.client.post(boxes, json={'frame_ordinal': 99, 'box': [1, 1, 20, 20]}).status_code, 400)     # not an evaluated frame
        first, second = in_frame[:2]
        self.assertEqual(self.client.put(f"/api/target-detections/{first['id']}/slot", json={'slot': second['slot']}).status_code, 200)
        moved = {d['id']: d for d in self.state()['run']['detections']}
        self.assertEqual((moved[first['id']]['slot'], moved[second['id']]['slot']), (second['slot'], first['slot']))
        self.assertEqual({moved[first['id']]['source'], moved[second['id']]['source']}, {'manual'})
        self.assertEqual(self.client.put(f"/api/target-detections/{first['id']}/box", json={'box': [6, 6, 56, 76]}).status_code, 200)
        self.assertEqual(self.client.put(f"/api/target-detections/{first['id']}/box", json={'box': [6, 6, 56, 200]}).status_code, 400)
        self.assertEqual(self.client.delete(f"/api/target-detections/{second['id']}").status_code, 200)
        self.assertNotIn(second['id'], {d['id'] for d in self.state()['run']['detections']})
        self.assertEqual(self.client.delete(f"/api/target-detections/{second['id']}").status_code, 404)

    def test_nothing_is_edited_while_detection_is_pending(self):
        run = self.detected()
        self.client.post('/api/cuts/c1/target-detection')
        self.assertEqual(self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': 'ctx2'}).status_code, 409)
        self.assertEqual(self.client.post(f"/api/target-runs/{run['id']}/boxes", json={'frame_ordinal': 10, 'box': [1, 1, 20, 20]}).status_code, 409)

    # --- what the rest of the app sees ---------------------------------------------------------------------
    def test_cuts_report_unknown_people_in_target_frames_for_the_timeline(self):
        cuts = lambda: {cut['id']: cut for cut in self.client.get('/api/episodes/e').json()['cuts']}
        self.assertEqual((cuts()['c1']['has_target_boxes'], cuts()['c1']['has_unlabeled_target_boxes']), (False, False))
        run = self.detected()
        self.assertEqual((cuts()['c1']['has_target_boxes'], cuts()['c1']['has_unlabeled_target_boxes']), (True, True))
        self.assertEqual((cuts()['c1']['has_unlabeled_boxes'], cuts()['c2']['has_target_boxes']), (True, False))
        self.client.put(f"/api/target-slots/{run['id']}/1", json={'character_context_id': 'ctx2'})
        self.assertEqual((cuts()['c1']['has_target_boxes'], cuts()['c1']['has_unlabeled_target_boxes']), (True, False))
        with db.connect() as conn:  # a moved boundary makes the run stale: it no longer counts
            conn.execute("UPDATE cuts SET end_time=2.5 WHERE id='c1'")
        self.assertEqual((cuts()['c1']['has_target_boxes'], cuts()['c1']['has_unlabeled_target_boxes']), (False, False))

    def test_labelled_slots_survive_nothing_but_their_character(self):
        run = self.detected()
        with db.connect() as conn:
            conn.execute("DELETE FROM context_items WHERE id='ctx'")
        self.assertIsNone(next(s for s in self.state()['run']['slots'] if s['slot'] == 0)['character_context_id'])
        self.assertEqual(len(self.state()['run']['detections']), 6)
        self.assertEqual(run['id'], self.state()['run']['id'])

    def test_the_queue_names_and_keeps_target_work_apart_from_first_frame_detection(self):
        target = {'workflow': 'target_frames_v1', 'cut_id': 'c1'}
        first = {'workflow': 'anime_bulk_v1', 'phase': 'full', 'cut_id': 'c1'}
        kind = job_kinds.kind_of('character_detect')
        self.assertEqual(kind.name(target), 'Target frame characters')
        self.assertNotEqual(kind.keys(target, ['c1']), kind.keys(first, ['c1']))
        self.assertTrue(kind.can_cancel('running', target))
        self.assertTrue(kind.gpu)


class SelectionChangeTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_a_different_selection_deletes_the_people_found_on_the_old_frames(self):
        source = media.episode_dir('p', 'e') / 'original.mp4'
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=5:duration=2', '-c:v', 'mpeg4', str(source)], check=True)
        settings = keyframes.config({})
        band = keyframes.mask_for('e', settings)

        def evaluate():
            cut = db.one("SELECT id,start_time,end_time FROM cuts WHERE id='c1'")
            job_id = worker.enqueue('e', 'keyframes', {'keyframe_settings': settings, 'subtitle_mask': band, 'cuts': [cut]})
            with db.connect() as conn:
                conn.execute("UPDATE jobs SET status='running',lease_owner='test',lease_until=9999999999 WHERE id=?", (job_id,))
            worker.process(job_id, 'test')
            self.assertEqual(db.one('SELECT status FROM jobs WHERE id=?', (job_id,))['status'], 'done')
            return keyframes.cache_key(source, cut, settings, band)

        first = evaluate()
        with db.connect() as conn:
            conn.execute("INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('run','c1',?,0,2,'[]')", (first,))
            conn.execute("INSERT INTO target_slots(run_id,slot,origin) VALUES ('run',0,'appeared')")
            conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json) VALUES ('box','run',0,0,'[1,1,9,9]')")
        evaluate()  # nothing changed: the same selection keeps the people found on it
        self.assertEqual(db.one('SELECT count(*) AS n FROM target_detections')['n'], 1)
        with db.connect() as conn:
            conn.execute("UPDATE cuts SET end_time=1.4 WHERE id='c1'")
        self.assertNotEqual(evaluate(), first)
        self.assertEqual([db.one(f'SELECT count(*) AS n FROM {table}')['n'] for table in ('target_detection_runs', 'target_slots', 'target_detections')], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
