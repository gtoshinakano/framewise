"""The sequence-v2 package: views, slices, identity, speaker candidates, frame honesty, issues and provenance."""
import json
import unittest

import tests.test_sequence_pipeline as pipeline_tests
from app import db, sequence_package as packages


def empty_values(value, path='$'):
    """Paths of null or empty values; the package leaves such keys out."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if item is None or item == '' or item == [] or item == {}:
                found.append(f'{path}.{key}')
            found.extend(empty_values(item, f'{path}.{key}'))
    elif isinstance(value, list):
        for number, item in enumerate(value):
            found.extend(empty_values(item, f'{path}[{number}]'))
    return found


class PackageTest(pipeline_tests.SequenceBase):
    def get(self, sequence_id, query='', status=200):
        response = self.client.get(f'/api/v1/sequences/{sequence_id}{query}')
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def test_views_state_each_fact_once_and_slice_by_shot(self):
        sequence_id = self.written()
        lean = self.get(sequence_id)
        self.assertEqual((lean['contract'], lean['view'], lean['ready'], lean['start_time'], lean['duration']), ('sequence-v2', 'lean', True, 0.0, 10.0))
        self.assertEqual(empty_values(lean), [])
        self.assertNotIn('pending', lean)  # nothing pending: left out
        self.assertEqual(lean['cuts'], [{'number': 1, 'id': 'c1', 'at': 0.0, 'duration': 2.0}, {'number': 2, 'id': 'c2', 'at': 2.0, 'duration': 2.0}, {'number': 3, 'id': 'c3', 'at': 4.0, 'duration': 6.0}])
        self.assertEqual(lean['speakers'], {'SP1': {'name': 'Hero', 'kind': 'character', 'character_context_id': 'ctx', 'speaker_labels': ['SPEAKER_01', 'SPEAKER_02']}})
        first = lean['shots'][0]
        self.assertEqual(set(first) & {'prompt', 'prompt_template', 'still_prompt', 'still', 'key', 'soundscape', 'opens', 'start_time', 't0', 'audio', 'dialogue', 'open_questions', 'result'}, set())
        self.assertEqual((first['index'], first['cut'], first['at'], first['duration'], first['generate_seconds']), (1, 1, 0.0, 2.0, 5.0))
        self.assertEqual(first['closes'], {'how': 'keep_cut', 'boundary': 'K1'})
        self.assertNotIn('closes', lean['shots'][2])  # the last shot ends the sequence
        self.assertNotIn('generate_seconds', lean['shots'][2])  # 6 s: generated as long as it is
        self.assertEqual(first['lines'], [{'id': 'L1', 'speaker': 'SP1', 't0': 0.4, 't1': 1.2, 'text': 'こんにちは', 'by': 'transcript'}])
        self.assertEqual(first['voices'], [{'speaker': 'SP1', 'from': 0.0, 'to': 2.0, 'delivery': 'on_screen', 'subject': 1}])
        self.assertEqual(first['line_map'], {'L1': '1.1'})
        self.assertEqual(first['subjects'], [{'n': 1, 'name': 'Hero', 'kind': 'character', 'character_context_id': 'ctx', 'picture': 2, 'linked_by': 'prompt'}])
        self.assertEqual(first['pictures'], [{'role': 'initial_frame'}, {'role': 'character_sheet', 'subject': 1}])
        self.assertEqual(first['initial_frame']['description_source'], 'observed')
        self.assertEqual(first['check'], {'ok': True})
        self.assertEqual(first['beats'][-1], {'start': 2.0, 'end': 5.0, 'hold': True})
        self.assertEqual(first['continuity'], {'from_previous': 'Hero stands in the same spot as before.'})
        self.assertEqual(lean['provenance']['shots']['1']['revision'], 1)
        self.assertEqual((lean['provenance']['bible']['origin'], lean['provenance']['bible']['revision']), ('model', 1))
        self.assertEqual([(issue['kind'], issue.get('shots')) for issue in lean['issues'] if issue['kind'] in ('review', 'padded_shot')], [('review', [1]), ('padded_shot', [1, 2])])
        self.assertNotIn('generation_seconds', lean['provenance']['shots']['1'])  # full view only
        self.assertEqual(lean['review'], {'verdict': 'The sequence holds together.'})
        # Full adds the prose, the stable key and the check statistics; include=prompts adds only the prompt texts.
        full = self.get(sequence_id, '?view=full')['shots'][0]
        self.assertTrue(full['prompt'].startswith('subject_definitions:'))
        self.assertIn('{start 1.1}', full['prompt_template'])
        self.assertEqual((full['key'], full['check']['ok'], full['subjects'][0]['anchors'][:10]), ('0-2000', True, 'slim build'))
        self.assertIn('stats', full['check'])
        self.assertNotIn('opens', full)
        prompts = self.get(sequence_id, '?include=prompts')['shots'][0]
        self.assertEqual(set(prompts) - set(first), {'prompt', 'prompt_template', 'still_prompt'})
        # One shot: its opening edge, its speakers and issues, no bible.
        one = self.get(sequence_id, '?shot=2')
        self.assertEqual(([shot['index'] for shot in one['shots']], one['shots'][0]['opens'], set(one['speakers'])), ([2], {'how': 'keep_cut', 'boundary': 'K1'}, {'SP1'}))
        self.assertNotIn('bible', one)
        self.assertEqual([cut['number'] for cut in one['cuts']], [2])
        self.assertTrue(all(2 in issue.get('shots', [2]) for issue in one['issues']))
        self.assertEqual(self.get(sequence_id, '?shot=1')['shots'][0]['opens'], {'how': 'start'})
        narrow = self.get(sequence_id, '?fields=lines,voices')
        self.assertEqual({key for shot in narrow['shots'] for key in shot}, {'index', 'lines', 'voices'})
        for query in ('?view=everything', '?fields=frames', '?include=clips', '?view=digest&shot=1'):
            self.get(sequence_id, query, 422)
        self.get(sequence_id, '?shot=9', 404)
        self.get('missing', '', 404)

    def test_digest_is_small_and_groups_the_same_issue(self):
        sequence_id = self.written()
        digest = self.get(sequence_id, '?view=digest')
        self.assertLess(len(json.dumps(digest, ensure_ascii=False, separators=(',', ':')).encode()), 5000)
        self.assertEqual((digest['contract'], digest['view'], digest['cuts'], digest['ready']), ('sequence-v2', 'digest', '1-3', True))
        self.assertEqual(digest['shots'][0], {'n': 1, 'cut': '1', 'at': 0.0, 'dur': 2.0, 'gen': 5.0, 'closes': 'keep_cut K1', 'frame': 'observed',
                                              'voices': ['Hero on 0-2'], 'lines': ['L1 Hero: こんにちは'], 'refs': 'P2 sheet: Hero'})
        padded = next(issue for issue in digest['issues'] if issue['kind'] == 'padded_shot')
        self.assertEqual(padded['shots'], [1, 2])
        self.assertEqual(digest['bible']['voices'][1], 'SPEAKER_02 → Hero (medium, director)')

    def test_names_link_to_records_and_unknown_lines_get_a_labelled_candidate(self):
        with db.connect() as conn:
            # Cut 2's analysis marked nobody: the prompt gets no sheet for Hero there.
            frame = {**pipeline_tests.FRAME, 'visible_characters': [], 'background_people': 'none', 'props': 'a white paper cup on the paving'}
            video = pipeline_tests.video(2, 4)
            video['direction']['characters'] = []
            conn.execute("UPDATE analysis_results SET content=? WHERE id='c2-initial_frame'", (json.dumps(frame),))
            conn.execute("UPDATE analysis_results SET content=? WHERE id='c2-video_analysis'", (json.dumps(video),))
            conn.execute("INSERT INTO transcript_segments(id,episode_id,start_time,end_time,text,speaker_label,origin) VALUES ('t9','e',7.0,7.5,'なに','','model')")

        def answer(data):
            content = pipeline_tests.shot_answer([item['speaker'] for item in data.get('speakers') or []], data['shot']['seconds'])
            if data['shot']['number'] == 2:
                content['subjects'].append({'name': 'the paper cup', 'anchors': 'white paper cup, crushed rim', 'keep': 'the cup'})
            return content
        self.answers['shot'] = answer
        sequence_id = self.written()
        lean = self.get(sequence_id)
        second, third = lean['shots'][1], lean['shots'][2]
        self.assertEqual(second['subjects'], [{'n': 1, 'name': 'Hero', 'kind': 'character', 'character_context_id': 'ctx', 'linked_by': 'name'},
                                              {'n': 2, 'name': 'the paper cup', 'kind': 'object', 'kind_guessed': True}])
        kinds = {(issue['kind'], shot) for issue in lean['issues'] for shot in issue.get('shots') or [None]}
        self.assertIn(('no_sheet', 2), kinds)
        self.assertNotIn(('unlinked_subject', 2), kinds)  # an object is not a missing person
        unknown = next(line for line in third['lines'] if line['text'] == 'なに')
        self.assertEqual(lean['speakers'][unknown['speaker']]['kind'], 'unknown')
        self.assertEqual(unknown['by'], 'unknown')
        self.assertEqual((unknown['candidate']['name'], unknown['candidate']['character_context_id'], unknown['candidate']['confidence']), ('Hero', 'ctx', 'medium'))
        self.assertIn('the only named character on screen in cut 3', unknown['candidate']['reason'])
        issue = next(issue for issue in lean['issues'] if issue['kind'] == 'unknown_speaker')
        self.assertEqual((issue['shots'], issue['line']), ([3], unknown['id']))
        self.assertIn('advice only', issue['text'])
        digest = self.get(sequence_id, '?view=digest')
        self.assertIn(f"{unknown['id']} Unknown speaker (? Hero): なに", digest['shots'][2]['lines'])

    def test_derived_frames_unwritten_shots_and_stale_templates_are_issues(self):
        self.interrupt()
        sequence_id = self.directed()
        self.assertEqual(self.client.put(f'/api/sequences/{sequence_id}/boundaries', json={'after_cut_id': 'c2', 'choice': 'after_line'}).status_code, 200)
        lean = self.get(sequence_id)
        self.assertEqual((lean['ready'], lean['pending']), (False, [1, 2, 3]))
        self.assertEqual([shot.get('written') for shot in lean['shots']], [False, False, False])
        self.assertEqual(lean['issues'][0]['severity'], 'error')
        frame = lean['shots'][2]['initial_frame']
        self.assertEqual((frame['description_source'], frame['derived_from']), ('derived', 'cut 3 first frame + its motion up to 0.8 s into the cut (shot 2 ran past the hard cut to finish a phrase)'))
        self.assertIn(('derived_frame', [3]), [(issue['kind'], issue.get('shots')) for issue in lean['issues']])
        self.assertEqual(lean['shots'][1]['closes'], {'how': 'after_line', 'boundary': 'K2', 'shift': 0.75, 'keeps': ['L3'], 'by': 'user'})
        self.assertEqual(lean['shots'][1]['source_cuts'], [2, 3])
        # Written, then the bible's spoken language changes: every shot is stale and its prompt no longer renders from its direction.
        self.run_step(sequence_id, 'shots')
        self.work()
        bible = self.detail(sequence_id)['bible']
        self.assertEqual(self.client.put(f"/api/sequence-results/{bible['id']}", json={'content': {**bible['content'], 'spoken_language': 'Korean'}}).status_code, 200)
        lean = self.get(sequence_id)
        self.assertEqual((lean['provenance']['bible']['revision'], lean['provenance']['bible']['origin']), (2, 'manual_edit'))
        self.assertEqual(lean['shots'][0]['stale'], ['bible'])
        kinds = {(issue['kind'], shot) for issue in lean['issues'] for shot in issue.get('shots') or [None]}
        self.assertTrue({('stale', 1), ('template_unavailable', 1), ('template_unavailable', 3)} <= kinds)
        self.assertEqual(len([issue for issue in lean['issues'] if issue['kind'] == 'template_unavailable']), 1)  # one text, every shot listed
        self.assertNotIn('prompt_template', self.get(sequence_id, '?view=full')['shots'][0])


class PackageHelpersTest(unittest.TestCase):
    def test_kind_guess_and_candidate_are_labelled_heuristics(self):
        frame = {'background_people': 'two students by the gate', 'props': 'a green notebook', 'setting': 'a classroom at noon'}
        self.assertEqual([packages.guess_kind(name, frame) for name in ('the notebook page', 'the dream figure', 'the classroom', 'the students', 'the light')],
                         ['object', 'person', 'place', 'person', 'unknown'])
        line = {'line': 'L7', 't0': 7.4, 't1': 8.1, 'text': 'なんだこれ', 'attribution': 'unknown'}
        before = {'line': 'L6', 't0': 4.5, 't1': 5.9, 'text': '必ずだと', 'attribution': 'reviewed', 'character_name': 'Sakuta', 'character_context_id': 'sakuta'}
        guess, why = packages.candidate(line, [before, line], {'Sakuta': 'sakuta'}, [], {'Sakuta': 'no'}, 7)
        self.assertEqual((guess['name'], guess['confidence'], why), ('Sakuta', 'medium', None))
        self.assertIn('the mouth is not seen moving in cut 7 (no)', guess['reason'])
        guess, why = packages.candidate(line, [line], {'Sakuta': 'sakuta', 'Mai': 'mai'}, ['Sakuta', 'Mai'], {}, 7)
        self.assertEqual((guess, why), (None, 'the evidence points equally to Sakuta or Mai'))
        self.assertEqual(packages.candidate(line, [line], {}, [], {}, 7)[0], None)

    def test_edges_drop_what_the_times_already_say(self):
        self.assertIsNone(packages.edge_view({'how': 'end'}))
        self.assertEqual(packages.edge_view({'how': 'keep_cut', 'boundary': 'K1', 'shift': 0.0, 'hard_cut': 2.0, 'split': {'lines': ['L2'], 'before': 'a', 'after': 'b'}}),
                         {'how': 'keep_cut', 'boundary': 'K1', 'splits': ['L2']})
        self.assertEqual(packages.edge_word({'how': 'after_line', 'boundary': 'K2', 'shift': 0.75, 'splits': ['L3']}), 'after_line K2 +0.75s splits L3')
        # A boundary the user placed by hand says so; the flag is left out everywhere else.
        placed = packages.edge_view({'how': 'before_line', 'boundary': 'K2', 'shift': -1.2, 'hard_cut': 4.0, 'placed': True, 'by': 'user', 'split': {'lines': ['L3'], 'before': 'a', 'after': 'b'}})
        self.assertEqual(placed, {'how': 'before_line', 'boundary': 'K2', 'shift': -1.2, 'splits': ['L3'], 'by': 'user', 'placed': True})
        self.assertEqual(packages.edge_word(placed), 'before_line K2 -1.2s placed splits L3')
