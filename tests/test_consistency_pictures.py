"""The consistency reviewer looks at pictures of the cut and reads the episode around it."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import consistency_evidence, consistency_summary, db, inference_worker, media, structured_results

KEYFRAMES = {'contract': 'keyframes-v1', 'frames': [{'label': 'F1', 't': 0.0, 'characters': [{'name': 'Mai', 'state': 'standing'}]},
                                                    {'label': 'F2', 't': 1.0, 'characters': []}, {'label': 'F3', 't': 2.0, 'characters': []},
                                                    {'label': 'F4', 't': 3.5, 'characters': [{'name': 'Mai', 'state': 'sitting, eyes closed'}]}],
             'segments': [{'from': 'F1', 'to': 'F2', 'characters': [{'name': 'Mai', 'changes': [{'part': 'arms', 'description': 'raises her arm'}]}]},
                          {'from': 'F2', 'to': 'F3', 'characters': [{'name': 'Mai', 'changes': [{'part': 'expression', 'description': 'starts to smile'}]}]},
                          {'from': 'F3', 'to': 'F4', 'characters': []}]}
CUT = {'id': 'c1', 'start_time': 10.0, 'end_time': 14.0}


class PlanTest(unittest.TestCase):
    def test_pictures_follow_the_keyframe_result_and_the_spoken_lines(self):
        segments = [{'start_time': 10.5, 'end_time': 11.0, 'text': 'Hi', 'speaker_label': 'S1', 'character_context_id': 'mai', 'character_name': 'Mai'},
                    {'start_time': 11.2, 'end_time': 12.0, 'text': 'there', 'speaker_label': 'S1', 'character_context_id': 'mai', 'character_name': 'Mai'},
                    {'start_time': 12.5, 'end_time': 13.5, 'text': 'Who?', 'speaker_label': 'S2', 'character_context_id': None},
                    {'start_time': 13.9, 'end_time': 14.05, 'text': 'x', 'speaker_label': 'S3'},  # under 0.2 s inside the cut: no strip
                    {'start_time': 12.0, 'end_time': 12.4, 'text': 'gone', 'deleted': True}]
        boxes = [{'character_id': 'mai', 't': 0.0, 'box': [100, 50, 300, 650], 'size': [1280, 720]},
                 {'character_id': 'mai', 't': 2.0, 'box': [500, 60, 700, 660], 'size': [1280, 720]}]
        pictures = consistency_evidence.plan(CUT, KEYFRAMES, segments, boxes, [{'name': 'Mai', 'cut_id': 'c0', 'at': 'first', 'crop': [.1, .1, .4, .9]}], {'previous': 'c0'})
        self.assertEqual([item['label'] for item in pictures],
                         ['F1', 'F2', 'F3', 'F4', 'Line 10.50-12.00s', 'Line 12.50-13.50s', 'Reference: Mai', 'Previous cut, last frame'])
        self.assertEqual(pictures[0]['at'], ['first'])  # F1 is the first frame: no second copy of it
        self.assertEqual((pictures[0]['edge'], pictures[1]['edge'], pictures[3]['at']), (1024, 768, ['3.500s']))
        mai = pictures[4]
        self.assertEqual(mai['at'], ['0.688s', '1.062s', '1.438s', '1.812s'])  # fragments of one speaker merged, four frames across the line
        self.assertLess(mai['crop'][0], 500 / 1280)  # the box nearest the line (t=0) is used
        self.assertLess(mai['crop'][3], 0.6)  # the head, not the whole standing figure
        self.assertIsNone(pictures[5]['crop'])  # an unlinked speaker has no box: whole frames
        self.assertIn('whole frames', pictures[5]['caption'])

    def test_a_cut_without_keyframes_gets_its_first_and_last_frame(self):
        pictures = consistency_evidence.plan(CUT, None, [], [])
        self.assertEqual([(item['label'], item['at']) for item in pictures], [('First frame', ['first']), ('Last frame', ['last'])])

    def test_fit_lowers_the_detail_until_the_request_fits(self):
        pictures = consistency_evidence.plan(CUT, KEYFRAMES, [{'start_time': 10.5, 'end_time': 12, 'text': 'a', 'speaker_label': 'S1'}], [],
                                             [{'name': 'Mai', 'cut_id': 'c0', 'at': 'first'}], {'previous': 'c0', 'next': 'c2'})
        evidence = {'visual_evidence_text': 'x', 'episode': {'neighbours': {'previous': {'cut': 1, 'summary': 'long ' * 400, 'clear': 'more ' * 400}},
                                                              'dialogue_around': {'before': ['a'] * 8}}}
        fit = lambda room: consistency_evidence.fit(pictures, evidence, 'system', consistency_evidence.OUTPUT_RESERVE + room, KEYFRAMES)
        self.assertEqual(fit(100000), (pictures, evidence, 0))
        _, shrunk, level = fit(3500)  # the neighbours' long text goes first
        self.assertEqual(level, 1)
        self.assertEqual(shrunk['episode']['neighbours']['previous'], {'cut': 1, 'summary': ('long ' * 40)})
        smaller, _, level = fit(2500)
        self.assertEqual(level, 2)
        self.assertEqual([item['edge'] for item in smaller if item['kind'] == 'keyframe'], [1024, 512, 512, 512])  # the opening frame keeps its size
        fewer, _, level = fit(1900)  # no references or neighbour pictures; F2-F3 show a face change and F1, F4 are the ends, so all keyframes stay
        self.assertEqual(([item['label'] for item in fewer], level), (['F1', 'F2', 'F3', 'F4', 'Line 10.50-12.00s'], 6))
        last, _, level = fit(100)
        self.assertEqual(([item['label'] for item in last], level), (['F1'], 8))
        many = {**KEYFRAMES, 'frames': [{'label': f'F{n}', 't': n * .5} for n in range(1, 7)]}
        kept = consistency_evidence.fit(consistency_evidence.plan(CUT, many, [], []), {}, 's', consistency_evidence.OUTPUT_RESERVE + 1400, KEYFRAMES)[0]
        self.assertEqual([item['label'] for item in kept if item['kind'] == 'keyframe'], ['F1', 'F2', 'F3', 'F6'])

    def test_the_head_crop_stays_inside_the_frame(self):
        crop = consistency_evidence.head_crop([0, 0, 100, 700], [1280, 720])
        self.assertEqual(crop[:2], [0.0, 0.0])
        self.assertTrue(all(0 <= value <= 1 for value in crop))
        close = consistency_evidence.head_crop([400, 100, 900, 700], [1280, 720])  # a close-up keeps the whole box
        self.assertLessEqual(close[0], 400 / 1280)
        self.assertGreaterEqual(close[3], 0.99)


class ReviewerTest(unittest.TestCase):
    def test_the_pictures_add_their_instructions_only_when_sent(self):
        text, seen = consistency_summary.system(False), consistency_summary.system(True)
        self.assertEqual(text, consistency_summary.SYSTEM)
        self.assertIn('No pictures were supplied', text)
        for missing in ('## PICTURES', 'visual_check', 'line strip'):
            self.assertNotIn(missing, text)
        for expected in ('## PICTURES', '## LOOKING AT THE PICTURES', '**visual_check:** Written first', 'raised brows with widened eyes', 'Reference: <name>',
                         'inner_voice the speaker in the picture with a still mouth', 'A contradiction you saw in a picture about an emotion, a voice or who is in the picture is a question'):
            self.assertIn(expected, seen)
        for both in (text, seen):
            self.assertIn('episode: the rest of the episode', both)
            self.assertIn('2. Voices: who speaks each turn', both)

    def test_the_grammar_starts_with_visual_check_naming_the_offered_pictures(self):
        plain = consistency_summary.question_format()['json_schema']['schema']
        self.assertNotIn('visual_check', plain['properties'])
        schema = consistency_summary.question_format(['F1', 'Line 1.00-2.00s'])['json_schema']['schema']
        self.assertEqual(list(schema['properties'])[0], 'visual_check')
        self.assertEqual(schema['required'][0], 'visual_check')
        item = schema['properties']['visual_check']['items']['properties']
        self.assertEqual(item['picture']['enum'], ['F1', 'Line 1.00-2.00s'])
        self.assertEqual(item['verdict']['enum'], ['contradicts', 'unclear', 'agrees'])

    def test_checks_are_kept_only_for_offered_pictures_contradictions_first(self):
        answer = {'summary': 'Mai sits.', 'clear': '', 'confusing': '', 'questions': [], 'visual_check': [
            {'character': 'Mai', 'picture': 'f2', 'source': 'keyframes', 'claim': 'neutral', 'seen': 'neutral', 'verdict': 'agrees'},
            {'character': 'Mai', 'picture': 'F1', 'source': 'first_frame', 'claim': 'big smile', 'seen': 'raised brows, open mouth: surprise', 'verdict': 'contradicts'},
            {'picture': 'F9', 'claim': 'x', 'seen': 'y', 'verdict': 'agrees'}, {'picture': 'F1', 'claim': 'x', 'verdict': 'perhaps'}, 'junk']}
        stored = consistency_summary.validate(answer, labels=['F1', 'F2'])
        self.assertEqual(list(stored)[0], 'visual_check')
        self.assertEqual([(item['picture'], item['verdict']) for item in stored['visual_check']], [('F1', 'contradicts'), ('F2', 'agrees')])
        self.assertNotIn('visual_check', consistency_summary.validate(answer))  # a text-only review keeps none
        with self.assertRaisesRegex(ValueError, 'visual_check must be an array'):
            consistency_summary.validate({'summary': 'x', 'questions': []}, labels=['F1'])

    def test_answers_read_contradictions_and_neighbours_never_their_checks(self):
        from app import prompt_context, review_answers
        content = {'summary': 's', 'questions': [], 'visual_check': [{'character': 'Mai', 'picture': 'F1', 'claim': 'smile', 'seen': 'surprise', 'verdict': 'contradicts'},
                                                                      {'character': 'Mai', 'picture': 'F2', 'claim': 'calm', 'seen': 'calm', 'verdict': 'agrees'}]}
        self.assertEqual(review_answers.review_view(content)['contradicted_by_pictures'], [{'character': 'Mai', 'picture': 'F1', 'claim': 'smile', 'seen': 'surprise'}])
        self.assertNotIn('visual_check', review_answers._neighbour({'cut_index': 1}, content)['consistency_summary'])
        self.assertEqual(prompt_context.slim_neighbors({'neighbor_consistency_summaries': {'previous': content}}), {'previous': {'summary': 's'}})

    def test_generate_attaches_pictures_and_asks_for_checks(self):
        sent = {}

        def fake(model, system, evidence, fields, **kwargs):
            sent.update(system=system, images=kwargs['images'], format=kwargs['response_format'])
            return kwargs['validator']({'summary': 's', 'clear': '', 'confusing': '', 'questions': [], 'visual_check': []}), 'm'

        pictures = [({'kind': 'keyframe', 'label': 'F1', 'caption': 'keyframe F1'}, b'jpeg'), ({'kind': 'reference', 'label': 'Reference: Mai', 'caption': 'Mai'}, b'ref')]
        with patch.object(inference_worker, 'chat_json', side_effect=fake):
            output, _ = consistency_summary.generate({'openrouter_model': 'x', 'text_provider': 'local'}, {}, pictures=pictures)
            self.assertEqual(output['visual_check'], [])
            self.assertEqual(sent['images'], [('F1: keyframe F1', b'jpeg'), ('Reference: Mai: Mai', b'ref')])
            self.assertIn('## PICTURES', sent['system'])
            # Only a reference could be cut: nothing about the cut itself to check, so the review is text only.
            output, _ = consistency_summary.generate({'openrouter_model': 'x'}, {}, pictures=pictures[1:])
            self.assertNotIn('visual_check', output)
            self.assertIsNone(sent['images'])
            self.assertNotIn('visual_check', sent['format']['json_schema']['schema']['properties'])


class ChatImagesTest(unittest.TestCase):
    def test_pictures_follow_the_evidence_and_survive_a_correction(self):
        payloads, answers = [], iter(['{"summary": ""}', '{"summary": "ok", "questions": []}'])

        def request_json(request, cancel_check, timeout):
            payloads.append(json.loads(request.data))
            return {'choices': [{'message': {'content': next(answers)}, 'finish_reason': 'stop'}]}

        usage = {}
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'k'}), patch.object(inference_worker, 'request_json', side_effect=request_json), patch.object(inference_worker, 'retry_delay'):
            output, _ = inference_worker.chat_json('m', 'sys', '{"e": 1}', [], usage_out=usage, validator=consistency_summary.validate, images=[('F1: keyframe', b'\xff\xd8')])
        self.assertEqual(output['summary'], 'ok')
        self.assertEqual(usage['pictures'], 1)
        content = payloads[1]['messages'][1]['content']
        self.assertEqual([part['type'] for part in content], ['text', 'text', 'image_url'])
        self.assertEqual(content[1]['text'], 'Picture F1: keyframe')
        self.assertTrue(content[2]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(payloads[1]['messages'][2]['role'], 'assistant')  # the correction comes after the pictures
        self.assertEqual(inference_worker.user_content('{}', None), '{}')
        self.assertFalse(inference_worker.endpoint_vision('openrouter'))


class EpisodeContextTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = db.DATA_DIR, db.DB_PATH, media.DATA_DIR
        db.DATA_DIR = media.DATA_DIR = Path(self.temp.name)
        db.DB_PATH = Path(self.temp.name) / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES('p','P')")
            conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,status,settings_json,duration,width,height) VALUES('e','p','E','v.mp4','.mp4','ready','{}',100,1280,720)")
            for index, (start, end) in enumerate(((0, 10), (10, 14), (14, 20))):
                conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES(?,?,?,?,?)', (f'c{index}', 'e', index, start, end))
            for cid, name in (('mai', 'Mai'), ('rio', 'Rio')):
                conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES(?,'p','character',?)", (cid, name))
            lines = [('t1', 2, 3, 'Earlier', 'S1', 'mai'), ('t2', 8, 9, 'Before', 'S1', 'mai'), ('t3', 11, 12, 'Now', 'S1', None), ('t4', 15, 16, 'After', 'S2', 'rio'),
                     ('t5', 50, 51, 'Far', 'S1', 'mai')]
            for row in lines:
                conn.execute('INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,character_context_id) VALUES(?,?,?,?,?,?,?)', (row[0], 'e', *row[1:]))
            for cut_id, kind, content in (('c0', 'consistency_summary', {'summary': 'Mai is shocked.', 'clear': 'Still: x', 'questions': [{'q': 1}], 'visual_check': []}),
                                          ('c0', 'keyframe_analysis', {'contract': 'keyframes-v1', 'segments': [], 'frames': [{'label': 'F1', 't': 0, 'characters': [{'name': 'Mai', 'state': 'crying'}]}]}),
                                          ('c2', 'initial_frame', {'visible_characters': [{'character': 'Rio', 'expression': 'frown', 'emotion': 'annoyed'}]})):
                kind_name, title = structured_results.KINDS[kind]
                conn.execute('INSERT INTO analysis_results(id,cut_id,kind,title,content,settings_json) VALUES(?,?,?,?,?,?)',
                             (f'{cut_id}-{kind}', cut_id, kind_name, title, json.dumps(content), json.dumps({'structured_type': kind})))

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.temp.cleanup()

    def test_neighbours_dialogue_and_voice_links(self):
        with db.connect() as conn:
            context = consistency_evidence.episode_context(conn, conn.execute("SELECT * FROM cuts WHERE id='c1'").fetchone())
        self.assertEqual(context['neighbours']['previous'], {'cut': 1, 'summary': 'Mai is shocked.', 'clear': 'Still: x', 'ends_with': [{'character': 'Mai', 'state': 'crying'}]})
        self.assertEqual(context['neighbours']['next'], {'cut': 3, 'opening_faces': [{'character': 'Rio', 'expression': 'frown', 'emotion': 'annoyed'}]})
        self.assertEqual(context['dialogue_around'], {'before': ['2.000–3.000s; S1; Mai: Earlier', '8.000–9.000s; S1; Mai: Before'], 'after': ['15.000–16.000s; S2; Rio: After']})
        self.assertEqual(context['voices'], [{'speaker_label': 'S1', 'lines_linked_in_episode': {'Mai': 3}, 'unlinked_lines_in_episode': 1}])

    def test_gather_offers_the_cut_pictures_with_a_reference_of_the_speaker(self):
        with db.connect() as conn:
            conn.execute("""INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES('r','c1',10,14,?,'m')""",
                         (json.dumps([{'position': 'first', 'time': 10, 'image_size': [1280, 720]}]),))
            conn.execute("""INSERT INTO character_detections(id,run_id,position,box_json,character_context_id,label_source) VALUES('d','r','first','[100,50,300,650]','mai','manual')""")
            conn.execute("UPDATE transcript_segments SET character_context_id='mai' WHERE id='t3'")
        by_type = {item['type']: item for item in structured_results.list_results('c1')}
        pictures, keyframes, aspect = consistency_evidence.gather('c1', by_type)
        self.assertEqual(aspect, 720 / 1280)
        self.assertIsNone(keyframes)
        self.assertEqual([item['label'] for item in pictures], ['First frame', 'Last frame', 'Line 11.00-12.00s', 'Reference: Mai', 'Previous cut, last frame', 'Next cut, first frame'])
        self.assertIsNotNone(pictures[2]['crop'])
        self.assertEqual((pictures[3]['cut_id'], pictures[3]['at']), ('c1', ['first']))


class WorkerImportsTest(unittest.TestCase):
    def test_the_picture_path_never_needs_the_api_packages(self):
        # The inference worker image has no FastAPI: every module the review reaches, including lazy imports, must work without it.
        import subprocess
        import sys
        script = '''
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('fastapi', 'starlette'):
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Block())
import unittest
from tests import test_consistency_pictures as t
suite = unittest.TestSuite([t.EpisodeContextTest('test_gather_offers_the_cut_pictures_with_a_reference_of_the_speaker'),
                            t.EpisodeContextTest('test_neighbours_dialogue_and_voice_links')])
sys.exit(0 if unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful() else 1)
'''
        root = Path(__file__).resolve().parents[1]
        done = subprocess.run([sys.executable, '-c', script], cwd=root, env={**__import__('os').environ, 'PYTHONPATH': str(root / 'backend') + ':' + str(root)},
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
