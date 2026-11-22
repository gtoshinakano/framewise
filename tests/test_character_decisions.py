"""Autopilot M6: first-frame names, unknown people and target-frame flags (character_decisions.py)."""
import json
import struct
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import agent_log, character_decisions as cd, db, media
from app.main import app

MAI, SAKUTA, RIO = 'ch-mai', 'ch-sakuta', 'ch-rio'


def box(box_id, cut='c1', label=None, source=None, candidates=(), rect=(100, 100, 500, 700), cut_index=0):
    return {'id': box_id, 'cut_id': cut, 'cut_index': cut_index, 'run_id': 'r', 'box': list(rect), 'image_size': [1280, 720],
            'character_context_id': label, 'label_source': source, 'candidates': list(candidates), 'small': False,
            'fill': round((rect[3] - rect[1]) / 720, 3), 'pending': False}


class DecideBoxTest(unittest.TestCase):
    def decide(self, item, others=(), near=(), relevance=None, pairs=(), vector=True):
        return cd.decide_box(item, cut_boxes=[item, *others], near=set(near), relevance=relevance or {}, look_alikes={frozenset(p) for p in pairs},
                             has_vector=vector)

    def test_names_with_a_margin_and_a_lower_bar_beside_the_same_character(self):
        self.assertEqual(self.decide(box('b', candidates=[(MAI, 0.80), (RIO, 0.60)]))['action'], 'name')
        self.assertEqual(self.decide(box('b', candidates=[(MAI, 0.64), (RIO, 0.50)]))['action'], 'claude')
        named = self.decide(box('b', candidates=[(MAI, 0.64), (RIO, 0.50)]), near=[MAI])
        self.assertEqual((named['action'], named['character'], named['rule']), ('name', MAI, 'near'))
        self.assertEqual(self.decide(box('b', candidates=[(MAI, 0.30)]))['action'], 'unknown')

    def test_already_present_duplicate_look_alikes_and_background(self):
        mai = box('m', label=MAI, source='manual')
        self.assertEqual(self.decide(box('b', candidates=[(MAI, 0.9)]), others=[mai])['action'], 'remove')  # same rectangle: a second box
        elsewhere = box('b', candidates=[(MAI, 0.9)], rect=(800, 100, 1200, 700))
        self.assertEqual(self.decide(elsewhere, others=[mai])['action'], 'claude')
        self.assertEqual(self.decide(box('b', candidates=[(MAI, 0.9), (RIO, 0.85)]), pairs=[(MAI, RIO)])['rule'], 'look_alike')
        self.assertEqual(self.decide(box('b', candidates=[(MAI, 0.9), (RIO, 0.5)]), relevance={MAI: 'background'})['action'], 'remove')

    def test_automatic_labels_are_never_confirmed_by_a_rule(self):
        decision = self.decide(box('b', label=MAI, source='auto', candidates=[(MAI, 0.97), (RIO, 0.5)]), near=[MAI])
        self.assertEqual((decision['action'], decision['rule']), ('review', 'review'))
        self.assertEqual(self.decide(box('b', label=MAI, source='auto', candidates=[(RIO, 0.9)]))['rule'], 'shortlist_disagrees')
        self.assertEqual(self.decide(box('b', label=MAI, source='manual'))['action'], 'keep')
        self.assertEqual(self.decide(box('b'), vector=False)['action'], 'skip')


class GroupTest(unittest.TestCase):
    def test_groups_by_vector_and_weighs_relevance(self):
        a, b, c = box('a', 'c1', rect=(0, 0, 100, 150)), box('b', 'c2', rect=(0, 0, 100, 140), cut_index=1), box('c', 'c3', rect=(0, 0, 10, 50), cut_index=2)
        c['small'] = True
        vectors = {'a': [1.0, 0.0], 'b': [0.99, 0.141], 'c': [0.0, 1.0]}
        groups = cd.group_unknowns([a, b, c], vectors)
        self.assertEqual(sorted(sorted(item['id'] for item in group['boxes']) for group in groups), [['a', 'b'], ['c']])
        weights = {tuple(sorted(item['id'] for item in group['boxes'])): cd.weigh(group)[0] for group in groups}
        self.assertEqual(weights, {('a', 'b'): 'borderline', ('c',): 'background'})
        self.assertEqual(cd.weigh({'boxes': [box('x', rect=(0, 0, 100, 700))]})[0], 'matters')


class DecideFrameTest(unittest.TestCase):
    slots = {0: {'character_context_id': MAI, 'ignored': 0}, 1: {'character_context_id': SAKUTA, 'ignored': 0}}

    def frame(self, flags, reviewed=False):
        return {'ordinal': 5, 'review': {'score': 0.8, 'flags': flags, 'reviewed': reviewed}}

    def test_a_drop_the_keyframe_states_explain_is_reviewed(self):
        flags = {'count_changed': {'first': 2, 'here': 1}}
        decided = cd.decide_frame(self.frame(flags), opening={0, 1}, here={0}, slots=self.slots, states={SAKUTA: 'He is no longer visible; out of frame.'})
        self.assertEqual((decided['action'], decided['rule']), ('reviewed', 'left_picture'))
        unexplained = cd.decide_frame(self.frame(flags), opening={0, 1}, here={0}, slots=self.slots, states={SAKUTA: 'Standing at the door.'})
        self.assertEqual(unexplained['action'], 'ask')
        self.assertIsNone(cd.decide_frame(self.frame(flags, reviewed=True), opening={0, 1}, here={0}, slots=self.slots, states={}))

    def test_an_extra_crop_beside_the_whole_cast_is_reviewed_unless_it_fits_a_colour(self):
        flags = {'unmatched': ['x']}
        crop = {'tag': 'unmatched', 'box': [0, 0, 300, 600], 'suggested_slot': None}
        self.assertEqual(cd.decide_frame(self.frame(flags), opening={0, 1}, here={0, 1}, slots=self.slots, states={}, crops=[crop])['rule'], 'extra_crop')
        self.assertEqual(cd.decide_frame(self.frame(flags), opening={0, 1}, here={0, 1}, slots=self.slots, states={},
                                         crops=[{**crop, 'suggested_slot': 1}])['action'], 'ask')
        self.assertEqual(cd.decide_frame(self.frame(flags), opening={0, 1}, here={0}, slots=self.slots, states={}, crops=[crop])['action'], 'ask')


def pack(vector):
    return struct.pack(f'<{len(vector)}f', *vector)


class JobTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
            conn.execute("""INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,has_audio)
                VALUES ('e','p','Episode','a.mp4','.mp4','ready','{}',20,1)""")
            for character, title in ((MAI, 'Mai'), (SAKUTA, 'Sakuta'), (RIO, 'Rio')):
                conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES (?,?,'character',?,'')", (character, 'p', title))
            for index in range(4):
                conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES (?,?,?,?,?)',
                             (f'c{index + 1}', 'e', index, index * 5.0, index * 5.0 + 5))
                conn.execute("""INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id,phase) VALUES (?,?,?,?,?,'m','complete')""",
                             (f'r{index + 1}', f'c{index + 1}', index * 5.0, index * 5.0 + 5,
                              json.dumps([{'position': 'first', 'time': index * 5.0, 'image_size': [1280, 720]}])))
            rows = [
                ('d-sak', 'r1', [100, 100, 500, 700], SAKUTA, 'manual', None),                     # cut 1: Sakuta by hand
                ('d-name', 'r2', [100, 100, 500, 700], None, None, [(SAKUTA, 0.66), (MAI, 0.5)]),  # cut 2: Sakuta beside cut 1 -> named
                ('d-dup', 'r1', [105, 100, 500, 700], None, None, [(SAKUTA, 0.9), (MAI, 0.4)]),    # cut 1: second box over Sakuta -> removed
                ('d-auto', 'r3', [600, 100, 900, 600], MAI, 'auto', [(MAI, 0.95), (RIO, 0.6)]),    # cut 3: automatic label -> Claude sheet
                ('d-unk1', 'r3', [0, 0, 300, 700], None, None, [(RIO, 0.2)]),                      # unknown, large -> matters
                ('d-unk2', 'r4', [0, 0, 300, 690], None, None, [(RIO, 0.2)]),
                ('d-bg', 'r4', [1200, 10, 1210, 40], None, None, [(RIO, 0.1)]),                     # unknown, tiny, once -> background
            ]
            for detection_id, run_id, rect, label, source, candidates in rows:
                conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source,candidates_json)
                    VALUES (?,?,'first',?,?,?,?)""", (detection_id, run_id, json.dumps(rect), label, source,
                                                       json.dumps([{'character_id': c, 'score': s} for c, s in candidates]) if candidates else None))
            for detection_id, vector in (('d-unk1', [1, 0, 0]), ('d-unk2', [0.98, 0.2, 0]), ('d-bg', [0, 0, 1])):
                conn.execute("""INSERT INTO character_detection_embeddings(detection_id,model_id,model_revision,preprocessing,dimensions,embedding)
                    VALUES (?,'ccip','1','p',3,?)""", (detection_id, pack(vector)))
            conn.execute("INSERT INTO analysis_results(id,cut_id,kind,title,content,origin) VALUES ('res2','c2','text','Cut summary','x','model')")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def run_job(self, body=None):
        queued = self.client.post('/api/episodes/e/character-decisions', json=body or {}).json()
        job_id = queued['job_id']
        with db.connect() as conn:
            conn.execute("UPDATE jobs SET status='running',lease_owner='test',lease_until=? WHERE id=?", (time.time() + 60, job_id))
        cd.process(job_id, 'test')
        with db.connect() as conn:
            job = conn.execute('SELECT status,error,payload_json FROM jobs WHERE id=?', (job_id,)).fetchone()
        self.assertEqual(job['status'], 'done', job['error'])
        return json.loads(job['payload_json'])['result']

    def label(self, detection_id):
        with db.connect() as conn:
            row = conn.execute('SELECT character_context_id,label_source FROM character_detections WHERE id=?', (detection_id,)).fetchone()
        return tuple(row) if row else None

    def test_names_removes_groups_and_asks_then_undo(self):
        result = self.run_job()
        self.assertEqual(self.label('d-name'), (SAKUTA, 'agent'))
        self.assertIsNone(self.label('d-dup'))
        self.assertIsNone(self.label('d-bg'))
        self.assertEqual(self.label('d-auto'), (MAI, 'auto'))
        self.assertEqual((result['boxes_name'], result['boxes_remove'], result['boxes_background']), (1, 1, 1))
        with db.connect() as conn:
            self.assertTrue(conn.execute("SELECT stale_reason FROM analysis_results WHERE id='res2'").fetchone()[0])
        items = {item['kind']: item for item in self.client.get('/api/episodes/e/escalations').json()['items']}
        self.assertEqual(set(items), {'box_label', 'new_person'})
        self.assertEqual(items['box_label']['reviewer'], 'claude')
        self.assertEqual(sorted(box['id'] for box in items['new_person']['evidence']['boxes']), ['d-unk1', 'd-unk2'])
        self.assertEqual(items['new_person']['reviewer'], 'human')   # two cuts with a tall figure: it matters
        # A second run asks nothing again about boxes already waiting in a question.
        again = self.run_job()
        self.assertEqual(again.get('asked_labels', 0) + again.get('asked_people', 0), 0)
        # Undo the removal of the duplicate box and the name.
        decisions = {row['target_id']: row['id'] for row in self.client.get('/api/episodes/e/agent-decisions').json()}
        self.assertEqual(self.client.post(f"/api/agent-decisions/{decisions['d-dup']}/undo").json(), {'undone': True})
        self.assertIsNotNone(self.label('d-dup'))
        self.assertEqual(self.client.post(f"/api/agent-decisions/{decisions['d-name']}/undo").json(), {'undone': True})
        self.assertEqual(self.label('d-name'), (None, None))

    def test_answers_claude_confirms_as_agent_and_the_person_names_a_new_character(self):
        self.run_job()
        items = {item['kind']: item for item in self.client.get('/api/episodes/e/escalations').json()['items']}
        labels, person = items['box_label'], items['new_person']
        refused = self.client.post(f"/api/escalations/{person['id']}/person", json={'choice': 'new', 'title': 'Kaede', 'tier': 'claude'})
        self.assertEqual(refused.status_code, 409)   # the person's question: Claude cannot answer it
        answered = self.client.post(f"/api/escalations/{labels['id']}/box-labels", json={'confirm': ['d-auto'], 'tier': 'claude'}).json()
        self.assertTrue(answered['resolved'])
        self.assertEqual(self.label('d-auto'), (MAI, 'agent'))
        created = self.client.post(f"/api/escalations/{person['id']}/person", json={'choice': 'new', 'title': 'Kaede'})
        self.assertEqual(created.status_code, 200, created.text)
        kaede = created.json()['created']['character']['id']
        self.assertEqual({self.label('d-unk1'), self.label('d-unk2')}, {(kaede, 'manual'), (kaede, 'agent')})
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM jobs WHERE pipeline='character_profile'").fetchone()[0], 1)

    def test_a_person_relabelling_an_agent_box_counts_as_an_override_and_closes_the_question(self):
        self.run_job()
        response = self.client.put('/api/character-detections/d-name', json={'character_context_id': MAI})
        self.assertEqual(response.status_code, 200, response.text)
        with db.connect() as conn:
            self.assertIsNotNone(conn.execute("SELECT overridden_at FROM agent_decisions WHERE target_id='d-name'").fetchone()[0])
        self.assertEqual(self.client.put('/api/character-detections/d-auto', json={'character_context_id': MAI}).status_code, 200)
        kinds = [item['kind'] for item in self.client.get('/api/episodes/e/escalations').json()['items']]
        self.assertNotIn('box_label', kinds)

    def test_target_frame_rule_marks_reviewed_and_asks_the_rest(self):
        with db.connect() as conn:
            frames = [{'ordinal': 1, 'time': 10.0, 'image_size': [1280, 720], 'mirror': True, 'review': {'score': 1, 'flags': {}}},
                      {'ordinal': 2, 'time': 11.0, 'image_size': [1280, 720], 'review': {'score': 0.85, 'flags': {'count_changed': {'first': 1, 'here': 0}}}},
                      {'ordinal': 3, 'time': 12.0, 'image_size': [1280, 720], 'review': {'score': 0.7, 'flags': {'suspected_merge': ['k']}}}]
            conn.execute("""INSERT INTO target_detection_runs(id,cut_id,selection_key,cut_start,cut_end,frames_json) VALUES ('t3','c3','k',10,15,?)""",
                         (json.dumps(frames),))
            conn.execute("INSERT INTO target_slots(run_id,slot,origin,character_context_id,label_source) VALUES ('t3',0,'first_frame',?, 'inherited')", (MAI,))
            conn.execute("INSERT INTO target_detections(id,run_id,frame_ordinal,slot,box_json) VALUES ('td1','t3',1,0,'[600,100,900,600]'),('td3','t3',3,0,'[600,100,900,600]')")
            conn.execute("""INSERT INTO keyframe_analysis_runs(id,cut_id,selection_key,cut_start,cut_end,input_hash,phase,method,roster_source,provenance_json)
                VALUES ('k3','c3','k',10,15,'h','complete','qwen','target','{}')""")
            conn.execute("""INSERT INTO keyframe_analysis_frames(run_id,ordinal,label,frame_time,relative_time,image_size_json,states_json)
                VALUES ('k3',2,'F2',11,1,'[1280,720]',?)""", (json.dumps({'qwen': [{'character_context_id': MAI, 'state': 'Mai has left the frame.'}]}),))
        result = self.run_job({'boxes': False})
        self.assertEqual(result['frames_reviewed'], 1)
        with db.connect() as conn:
            stored = json.loads(conn.execute("SELECT frames_json FROM target_detection_runs WHERE id='t3'").fetchone()[0])
            self.assertIsNone(conn.execute("SELECT stale_reason FROM analysis_results WHERE id='res2'").fetchone()[0])
        self.assertTrue(stored[1]['review']['reviewed'])
        self.assertFalse(stored[2]['review'].get('reviewed'))
        question = next(item for item in self.client.get('/api/episodes/e/escalations').json()['items'] if item['kind'] == 'target_frame')
        self.assertEqual([frame['ordinal'] for frame in question['evidence']['frames']], [3])
        answered = self.client.post(f"/api/escalations/{question['id']}/target-frames", json={'frames': {'3': 'person'}, 'tier': 'claude'}).json()
        self.assertTrue(answered['handed_to_person'])
        self.client.put('/api/target-runs/t3/frames/3/reviewed', json={'reviewed': True})
        self.assertNotIn('target_frame', [item['kind'] for item in self.client.get('/api/episodes/e/escalations').json()['items']])
        decision = next(item for item in self.client.get('/api/episodes/e/agent-decisions').json() if item['kind'] == 'target_frames')
        self.assertEqual(self.client.post(f"/api/agent-decisions/{decision['id']}/undo").json(), {'undone': True})

    def test_profile_off_and_registrations(self):
        self.client.put('/api/projects/p/autopilot-profile', json={'profile': {'modes': {'box_names': 'off', 'target_frames': 'off'}}})
        self.assertIsNone(self.client.post('/api/episodes/e/character-decisions', json={}).json()['job_id'])
        self.assertIs(agent_log.UNDO_KINDS['box_names'], cd.undo_boxes)
        self.assertIs(agent_log.UNDO_KINDS['target_frames'], cd.undo_target)


if __name__ == '__main__':
    unittest.main()
