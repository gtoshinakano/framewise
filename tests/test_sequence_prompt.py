"""Rendering a directed shot as an H3 prompt, and the validators that guard the director's answers."""
import copy
import unittest

import tests.test_sequence_pipeline as pipeline_tests
from app import sequence_direction as direction, sequence_prompt
from app.assistant import h3_prompt

BIBLE = {'style': 'The target video is a 2D cel-shaded anime with clean ink lines.', 'spoken_language': 'Japanese'}
ROSTER = [{'name': 'Mai Sakurajima', 'character_context_id': 'mai'}, {'name': 'Sakuta Azusagawa', 'character_context_id': 'sakuta'}]
SCENARIO = {'id': 'station', 'title': 'Fujisawa Station', 'content': 'A platform.'}


def shot(duration=8.0, index=3):
    return {'index': index, 'key': 'k', 'duration': duration, 'generate_seconds': max(duration, 5.0), 'trim_seconds': duration if duration < 5 else None,
            'frame': {'cut': 1, 'cut_id': 'c1', 'time': 0.0, 'strategy': 'cut_first_frame', 'continues_previous_shot': False}}


def line(line_id, t0, t1, text, name=None):
    return {'line': line_id, 't0': t0, 't1': t1, 'text': text, 'speaker_label': 'SPEAKER_01', 'character_context_id': None, 'character_name': name,
            'attribution': 'transcript' if name else 'unknown', 'crosses_start': False, 'crosses_end': False}


DIALOGUE = [line('L4', 0.5, 2.0, 'おはよう', 'Mai Sakurajima'), line('L5', 4.5, 6.0, '遅いよ'), line('L6', 6.5, 7.5, '……', 'Sakuta Azusagawa')]


def answer():
    content = pipeline_tests.shot_answer(seconds=8.0)
    content['subjects'] = [{'name': 'mai sakurajima', 'anchors': 'long black hair, school blazer', 'keep': 'face and uniform'},
                           {'name': 'Sakuta Azusagawa', 'anchors': 'short brown hair, school blazer', 'keep': 'face and uniform'},
                           {'name': 'the station attendant', 'anchors': 'navy cap and uniform', 'keep': 'cap and uniform'}]
    content['picture_line'] = 'Mai Sakurajima stands on the platform of Fujisawa Station facing Sakuta Azusagawa.'
    content['summary'] = 'Mai greets Sakuta on the platform while the camera pushes in.'
    content['shot_line'] = 'A medium two-shot at eye level: Mai Sakurajima on the left and Sakuta Azusagawa on the right of the platform at Fujisawa station, the station attendant far behind.'
    content['beats'] = [{'end': 4.0, 'action': content['beats'][0]['action'].replace('Hero', 'Mai')}, {'end': 8.0, 'action': content['beats'][1]['action'].replace('Hero', 'Sakuta')}]
    content['camera'] = {'move': 'push_in', 'amplitude': 'small', 'speed': 'slow', 'detail': "It ends on a medium close-up of Mai's face."}
    content['speech'] = [{'line': 'L4', 'delivery': 'lip_sync', 'subject': 'Mai Sakurajima'}, {'line': 'L5', 'delivery': 'off_screen', 'subject': None},
                         {'line': 'L6', 'delivery': 'inner_voice', 'subject': 'sakuta azusagawa'}]
    content['scenario_anchors'] = 'grey platform, yellow tactile strip, overhead signs left unmarked'
    content['scenario_keep'] = 'platform layout'
    return content


class RenderTest(unittest.TestCase):
    def build(self, content=None, scenario=SCENARIO, target=None, dialogue=DIALOGUE):
        normal = direction.validate_shot(content or answer(), target or shot(), BIBLE, ROSTER, scenario, dialogue)
        return normal, *sequence_prompt.build(target or shot(), normal, BIBLE, ROSTER, scenario, dialogue)

    def test_prompt_follows_the_h3_contract_with_app_written_labels_times_and_words(self):
        normal, prompt, verdict, derived = self.build()
        self.assertTrue(verdict['ok'], verdict['errors'])
        self.assertEqual((verdict['stats']['mode'], verdict['stats']['seconds'], verdict['stats']['pictures'], verdict['stats']['audio'], verdict['stats']['subjects']), ('omni_reference', 8.0, [1, 2, 3, 4], [1], [1, 2, 3, 4]))
        self.assertEqual(h3_prompt.parse(prompt)['order'], list(h3_prompt.SECTIONS))
        self.assertEqual([subject['name'] for subject in normal['subjects']], ['Mai Sakurajima', 'Sakuta Azusagawa', 'the station attendant'])  # record names are canonical
        definitions = h3_prompt.parse(prompt)['sections']['subject_definitions'].splitlines()
        self.assertEqual(definitions[0], '<Picture 1> is the first frame of [Shot 1]: Mai Sakurajima stands on the platform of Fujisawa Station facing Sakuta Azusagawa.')
        self.assertEqual(definitions[1], '<Subject 1> is Mai Sakurajima, whose face, hair and clothing come from <Picture 2>: long black hair, school blazer.')
        self.assertEqual(definitions[3], '<Subject 3> is the station attendant, seen in <Picture 1>: navy cap and uniform.')  # nobody without a record gets a sheet
        self.assertEqual(definitions[4], '<Subject 4> is Fujisawa Station, the setting from <Picture 4>: grey platform, yellow tactile strip, overhead signs left unmarked.')
        self.assertEqual(definitions[5], '<Audio 1> is the original sound of this shot, “shot-03.wav”: the voices and ambience to keep in sync.')
        self.assertEqual(h3_prompt.subject_names(prompt), {1: 'Mai Sakurajima', 2: 'Sakuta Azusagawa', 3: 'station attendant', 4: 'Fujisawa Station'})  # the checker reads names without a leading "the"
        description = h3_prompt.parse(prompt)['sections']['detailed_description']
        self.assertTrue(description.startswith('The target video is a 2D cel-shaded anime'))
        # Names in prose become labels: full names, single name words, people without a record and the place.
        self.assertIn('[Shot 1] The shot begins from <Picture 1>: A medium two-shot at eye level: <Subject 1> on the left and <Subject 2> on the right of the platform at <Subject 4>, <Subject 3> far behind.', description)
        self.assertNotRegex(description.replace('Mai Sakurajima', ''), r'\b(?:Mai|Sakuta|Hero)\b')
        self.assertIn("The camera pushes in with small amplitude at slow speed. It ends on a medium close-up of <Subject 1>'s face.", description)
        self.assertIn('00:00.000-00:04.000:', description)
        self.assertIn('00:04.000-00:08.000:', description)
        first, second = [row for row in description.splitlines() if row.startswith('00:')]
        self.assertIn('At 00:00.500, <Subject 1> (S1) says: <d>[Japanese] おはよう</d>', first)
        self.assertIn('At 00:04.500, an off-screen voice is heard: <d>[Japanese] 遅いよ</d>', second)
        self.assertIn('At 00:06.500, the inner voice of <Subject 2> is heard while the lips stay closed: <d>[Japanese] ……</d>', second)
        self.assertEqual(h3_prompt.speaking_subjects(prompt), {1})
        self.assertIn('<Subject 2> does not speak; the lips stay closed. <Subject 3> does not speak.', description)  # the lips clause is for named characters
        self.assertNotIn('<Subject 4> does not speak', description)
        soundscape = h3_prompt.parse(prompt)['sections']['overall_soundscape']
        self.assertIn('<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:00.500.', soundscape)
        self.assertIn('An off-screen voice speaks the words of <Audio 1> from 00:04.500; no lips on screen move for them.', soundscape)
        self.assertIn('are the inner voice of <Subject 2>; the lips stay closed.', soundscape)
        self.assertNotIn('<d>', soundscape)
        self.assertEqual(h3_prompt.parse(prompt)['sections']['non_diegetic_music'], 'N/A')
        self.assertEqual([(item['label'], item['role']) for item in derived['references']],
                         [('<Picture 1>', 'initial_frame'), ('<Picture 2>', 'character_sheet'), ('<Picture 3>', 'character_sheet'), ('<Picture 4>', 'scenario'), ('<Audio 1>', 'original_audio')])
        self.assertEqual([(item['line'], item['delivery'], item['subject']) for item in derived['dialogue']], [('L4', 'lip_sync', 1), ('L5', 'off_screen', None), ('L6', 'inner_voice', 2)])
        self.assertEqual(derived['still_prompt'].splitlines()[-2:], ['Style: A 2D cel-shaded anime with clean ink lines.',
                                                                     'Constraints: one frozen moment; no text, subtitles, captions, logos or watermarks; no additional characters.'])

    def test_a_silent_shot_without_people_or_scenario(self):
        content = answer()
        content.update(subjects=[], speech=[], picture_line='An empty platform at dusk.', shot_line='A wide shot of the empty platform, tracks running to the right.')
        content['beats'] = [{'end': 8.0, 'action': ' '.join(['Wind moves a paper cup along the yellow strip and the light slowly dims on the far wall.'] * 9)}]
        content['camera'] = {'move': 'static', 'amplitude': 'large', 'speed': 'fast', 'detail': 'ignored for a static shot'}
        normal, prompt, verdict, derived = self.build(content, scenario=None, dialogue=[])
        self.assertTrue(verdict['ok'], verdict['errors'])
        self.assertEqual((normal['camera']['detail'], normal['scenario_anchors']), ('', ''))
        self.assertIn('The camera holds a static shot.', prompt)
        self.assertIn('<Audio 1> carries no speech; match its ambience and timing.', prompt)
        self.assertIn('the ambience and timing to match', prompt)
        self.assertEqual([item['label'] for item in derived['references']], ['<Picture 1>', '<Audio 1>'])

    def test_only_four_character_sheets_are_attached(self):
        roster = [{'name': f'Person{number}', 'character_context_id': f'id{number}'} for number in range(1, 6)]
        content = {'subjects': [{'name': person['name'], 'anchors': 'a', 'keep': 'b'} for person in roster]}
        subjects, references = sequence_prompt.assign(content, roster, SCENARIO)
        self.assertEqual([subject['picture'] for subject in subjects], [2, 3, 4, 5, None, 6])
        self.assertEqual([item['role'] for item in references].count('character_sheet'), sequence_prompt.MAX_SHEETS)

    def errors(self, change, **kwargs):
        content = answer()
        change(content)
        with self.assertRaises(ValueError) as raised:
            direction.validate_shot(content, kwargs.get('target') or shot(), BIBLE, ROSTER, kwargs.get('scenario', SCENARIO), kwargs.get('dialogue', DIALOGUE))
        return str(raised.exception)

    def test_validator_messages_tell_the_director_what_to_fix(self):
        self.assertIn('beat 2 contains the word "cut"', self.errors(lambda c: c['beats'][1].update(action='A quick cut to her hands.')))
        self.assertIn('summary contains a reference label', self.errors(lambda c: c.update(summary='<Subject 1> waves.')))
        self.assertIn('shot_line quotes dialogue', self.errors(lambda c: c.update(shot_line='She says <d>hello</d>.')))
        self.assertIn('beat 1 contains a timecode', self.errors(lambda c: c['beats'][0].update(action='At 00:01.500 she turns.')))
        self.assertIn('speech needs one entry for line L5', self.errors(lambda c: c['speech'].pop(1)))
        self.assertIn('speech L4: lip_sync needs "subject" to be one of your subjects (Mai Sakurajima, Sakuta Azusagawa, the station attendant)', self.errors(lambda c: c['speech'][0].update(subject='Kaede')))
        self.assertIn('camera.move must be one of: static, push_in', self.errors(lambda c: c['camera'].update(move='zoom')))
        self.assertIn('beat 2 ends at 20 s but the shot lasts 8 s', self.errors(lambda c: c['beats'][1].update(end=20)))
        self.assertIn('beat 2 ends at 3 s, which is not after the previous beat (4 s)', self.errors(lambda c: c['beats'][1].update(end=3)))
        self.assertIn('"scenario_anchors" is empty', self.errors(lambda c: c.update(scenario_anchors='')))
        self.assertIn('"picture_line" is empty', self.errors(lambda c: c.update(picture_line=' ')))
        self.assertIn('subject "the station attendant" needs anchors', self.errors(lambda c: c['subjects'][2].update(anchors='')))
        thin = self.errors(lambda c: c.update(beats=[{'end': 8.0, 'action': 'She waves.'}], shot_line='Two people.', framing_lock='', light=''))
        self.assertIn('shot_line, framing_lock, light and beats together have', thin)
        several = self.errors(lambda c: (c.update(summary='A cut.'), c['camera'].update(move='zoom')))
        self.assertTrue(several.startswith('(1) ') and '(2) ' in several)

    def test_bible_and_review_validators(self):
        ctx = {'characters': [{'id': 'mai', 'name': 'Mai Sakurajima', 'notes': ''}], 'scenarios': [{'id': 'station', 'title': 'Fujisawa Station', 'content': ''}], 'results': {}}
        timeline = {'cuts': [{'id': 'c1', 'number': 7, 't0': 0, 't1': 4}, {'id': 'c2', 'number': 8, 't0': 4, 't1': 9}],
                    'lines': [{'id': 'L1', 't0': 1, 't1': 2, 'text': 'a', 'speaker_label': 'SPEAKER_00', 'character_name': None, 'character_context_id': None}]}
        bible = copy.deepcopy(pipeline_tests.BIBLE)
        bible['scenes'] = [{'from_cut': 7, 'to_cut': 7, 'location': 'Platform', 'scenario': 'fujisawa station'}, {'from_cut': 8, 'to_cut': 8, 'location': 'Train', 'scenario': 'null'}]
        bible['characters'] = [{'name': 'MAI SAKURAJIMA', 'anchors': 'long black hair'}]
        bible['voices'] = [{'speaker_label': 'SPEAKER_00', 'character': None, 'kind': 'narrator', 'confidence': 'high', 'reason': 'Narration.'}]
        normal = direction.validate_bible(bible, ctx, timeline)
        self.assertEqual([(scene['id'], scene['scenario'], scene['scenario_context_id']) for scene in normal['scenes']], [('S1', 'Fujisawa Station', 'station'), ('S2', None, None)])
        self.assertEqual((normal['characters'][0]['name'], normal['characters'][0]['character_context_id']), ('Mai Sakurajima', 'mai'))
        self.assertEqual((normal['voices'][0]['kind'], normal['voices'][0]['character'], normal['voices'][0]['source']), ('narrator', None, 'director'))

        def problems(change):
            wrong = copy.deepcopy(bible)
            change(wrong)
            with self.assertRaises(ValueError) as raised:
                direction.validate_bible(wrong, ctx, timeline)
            return str(raised.exception)
        self.assertIn('scenes must cover cuts 7 to 8', problems(lambda b: b['scenes'][1].update(from_cut=9, to_cut=9)))
        self.assertIn('they must cover every cut up to 8', problems(lambda b: b['scenes'].pop()))
        self.assertIn('is not a scenario note; use one of: Fujisawa Station, or null', problems(lambda b: b['scenes'][0].update(scenario='Enoshima')))
        self.assertIn('has no character record; use exact names from: Mai Sakurajima', problems(lambda b: b['characters'][0].update(name='Kaede')))
        self.assertIn('voice SPEAKER_00: "Kaede" has no character record', problems(lambda b: b['voices'][0].update(character='Kaede')))
        self.assertIn('"spoken_language" must be the language name in English letters', problems(lambda b: b.update(spoken_language='日本語')))
        review = direction.validate_review({'verdict': 'Fine.', 'issues': [{'shot': 2, 'severity': 'urgent', 'problem': 'Coat changes.'}, {'shot': 5, 'problem': 'x'}, 'junk']}, 3)
        self.assertEqual(review, {'verdict': 'Fine.', 'issues': [{'shot': 2, 'severity': 'check', 'problem': 'Coat changes.', 'suggestion': ''}]})

    def test_answer_formats_constrain_ids_to_what_the_app_offers(self):
        timeline = {'candidates': [{'id': 'B0'}, {'id': 'B1'}, {'id': 'B2'}], 'cuts': [{'number': 7}, {'number': 8}]}
        plan = direction.plan_format(timeline, {'scenes': [{'id': 'S1'}]})['json_schema']['schema']['properties']['shots']['items']['properties']
        self.assertEqual((plan['from']['enum'], plan['to']['enum'], plan['frame_cut']['enum'], plan['scene']['enum']), (['B0', 'B1'], ['B1', 'B2'], [7, 8], ['S1']))
        speech = direction.shot_format(shot(), ROSTER, DIALOGUE)['json_schema']['schema']['properties']['speech']['items']['properties']
        self.assertEqual((speech['line']['enum'], speech['delivery']['enum']), (['L4', 'L5', 'L6'], list(sequence_prompt.DELIVERIES)))
        for step in direction.STEPS.values():
            self.assertIn('Return JSON only.', step['system'])


if __name__ == '__main__':
    unittest.main()
