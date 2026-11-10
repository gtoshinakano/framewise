import json
import unittest
from app.visual_contracts import (FRAME_INSTRUCTION, LAST_FRAME_INSTRUCTION, SUBTITLE_INSTRUCTION, frame_schema, full_last_frame, person_context,
                                  resolve_frame, subtitle_note, validate_video)

PEOPLE = [{'ref': 'B1', 'detection_id': 'd-mai', 'character_context_id': 'mai', 'name': 'Mai', 'box': [10, 20, 300, 900]},
          {'ref': 'B2', 'detection_id': 'd-sakuta', 'character_context_id': 'sakuta', 'name': 'Sakuta', 'box': [500, 20, 800, 900]}]


def frame(**changes):
    return {'composition': 'MS, eye level', 'setting': 'classroom', 'lighting': 'soft daylight', 'camera': 'level', 'visible_characters': [],
            'background_people': 'none', 'notes': 'none', **changes}


def entry(box, **fields):
    return {'box': box, 'appearance': 'a', 'position': 'left', 'pose': 'sitting', 'expression': 'calm', 'action': 'reading', **fields}


class VisualContractsTest(unittest.TestCase):
    def test_frame_answers_are_resolved_to_real_records(self):
        raw = frame(visible_characters=[entry('B2', pose='standing'), entry('B1')])
        result = resolve_frame('```json\n' + json.dumps(raw) + '\n```', PEOPLE)
        self.assertEqual([item['character'] for item in result['visible_characters']], ['Mai', 'Sakuta'])  # box order, not the model's order
        mai = result['visible_characters'][0]
        self.assertEqual((mai['character'], mai['character_context_id']), ('Mai', 'mai'))
        self.assertNotIn('identity_evidence', mai)  # identity is certain, so no box ID or verification text is stored
        self.assertNotIn('detection_id', mai)
        self.assertNotIn('box', mai)
        self.assertEqual(result['mode'], 'full')

    def test_orphaned_references_are_repaired(self):
        raw = frame(visible_characters=[entry('B1'), entry('b1', pose='lying'), entry('B9'), 'invented'])
        result = resolve_frame(json.dumps(raw), PEOPLE)
        self.assertEqual([item['character'] for item in result['visible_characters']], ['Mai', 'Sakuta'])  # B9 dropped, skipped B2 added
        self.assertEqual(result['visible_characters'][0]['pose'], 'sitting')  # a repeated handle keeps the first answer
        self.assertEqual(result['visible_characters'][1]['pose'], '')
        self.assertEqual(resolve_frame(json.dumps(frame(visible_characters=[entry('B1')])), [])['visible_characters'], [])
        with self.assertRaises(ValueError): resolve_frame(json.dumps(frame(visible_characters='none')), PEOPLE)
        with self.assertRaises(ValueError): resolve_frame('[]', PEOPLE)

    def test_changes_mode_marks_unchanged_fields_and_new_people(self):
        people = [{**PEOPLE[0], 'new': False}, {**PEOPLE[1], 'new': True}]
        raw = frame(composition='unchanged', visible_characters=[entry('B1', appearance='', pose='unchanged', expression='wide eyes', action='', position='unchanged'), entry('B2', appearance='uniform')])
        result = resolve_frame(json.dumps(raw), people, changes=True)
        mai, sakuta = result['visible_characters']
        self.assertEqual((mai['appearance'], mai['action'], mai['expression'], mai['new_in_frame']), ('unchanged', 'unchanged', 'wide eyes', False))
        self.assertEqual((sakuta['appearance'], sakuta['new_in_frame']), ('uniform', True))
        self.assertEqual((result['mode'], result['composition']), ('changes', 'unchanged'))

    def test_changes_mode_reads_back_as_full_states(self):
        first = resolve_frame(json.dumps(frame(visible_characters=[entry('B1', appearance='red tie', pose='sitting'), entry('B2')])), PEOPLE)
        last = resolve_frame(json.dumps(frame(composition='unchanged', visible_characters=[entry('B1', appearance='unchanged', pose='standing', expression='unchanged', position='unchanged', action='unchanged')])),
                             [{**PEOPLE[0], 'new': False}], changes=True)
        full = full_last_frame(first, last)
        self.assertEqual(full['composition'], 'MS, eye level')
        self.assertEqual([(item['character'], item['appearance'], item['pose']) for item in full['visible_characters']], [('Mai', 'red tie', 'standing')])
        self.assertIs(full_last_frame(first, first), first)  # full results pass through

    def test_schema_offers_only_the_listed_handles(self):
        schema = frame_schema(['B1', 'B2'])
        people = schema['properties']['visible_characters']
        self.assertEqual(people['items']['properties']['box']['enum'], ['B1', 'B2'])
        self.assertEqual((people['minItems'], people['maxItems']), (2, 2))
        self.assertEqual(frame_schema([])['properties']['visible_characters']['maxItems'], 0)
        self.assertEqual(list(schema['properties'])[:4], ['composition', 'setting', 'lighting', 'camera'])

    def test_prompts_are_short_and_do_not_ask_for_boilerplate(self):
        for text in (FRAME_INSTRUCTION, LAST_FRAME_INSTRUCTION):
            self.assertNotIn('walking is unconfirmed', text.lower())
            self.assertIn('one single language', text)
            self.assertIn('certain', text)
        self.assertIn('"unchanged"', LAST_FRAME_INSTRUCTION)
        self.assertIn('never describe outfits again', LAST_FRAME_INSTRUCTION)
        self.assertIn('none', person_context([], 'first frame'))
        self.assertIn('B1: Mai', person_context(PEOPLE, 'first frame'))
        self.assertIn('[new', person_context([{**PEOPLE[0], 'new': True}], 'last frame', changes=True))
        self.assertIn('one single language', SUBTITLE_INSTRUCTION)
        self.assertNotIn('Portuguese or English', SUBTITLE_INSTRUCTION)

    def test_subtitle_note_states_the_language_as_certain_and_names_the_strip(self):
        note = subtitle_note('Japanese', [[0, 540, 1280, 700]], 'image pixels')
        self.assertIn('every burned-in subtitle in this video is in Japanese; that is certain', note)
        self.assertIn('[[0, 540, 1280, 700]] in image pixels', note)
        self.assertIn('never read, describe, translate or reproduce it', note)
        self.assertEqual(subtitle_note(None, [], 'image pixels'), '')
        self.assertNotIn('certain', subtitle_note(None, [[0, 1, 2, 3]], 'image pixels'))

    def test_video_accepts_overlapping_subject_events_without_reordering(self):
        events = [
            {'start_seconds': 0, 'end_seconds': .6, 'subject': 'Rio'},
            {'start_seconds': .6, 'end_seconds': 1.6, 'subject': 'Rio'},
            {'start_seconds': 1.6, 'end_seconds': 3.6, 'subject': 'Rio'},
            {'start_seconds': 0, 'end_seconds': 3.6, 'subject': 'Sakuta'},
        ]
        result = validate_video(json.dumps({'overview': 'scene', 'events': events}), 3.6, 10)
        self.assertEqual([event['subject'] for event in result['events']], ['Rio', 'Rio', 'Rio', 'Sakuta'])
        self.assertEqual(result['events'][-1]['episode_start_seconds'], 10)
        self.assertEqual(result['events'][-1]['episode_end_seconds'], 13.6)

    def test_video_preserves_unusual_json_without_inventing_times(self):
        payload = {'events': [{'start_seconds': 4, 'end_seconds': 5}, {'start_seconds': 'unknown'}, 'unstructured'], 'other': [1, 2]}
        self.assertEqual(validate_video(json.dumps(payload), 3, 10), payload)
        self.assertEqual(validate_video('{"events": null, "extra": true}', 3, 10), {'events': None, 'extra': True})
        for valid, expected in [('[]', []), ('null', None), ('"text"', 'text'), ('42', 42)]:
            self.assertEqual(validate_video(valid, 3, 10), expected)
        for invalid in ('not json', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_video(invalid, 3, 10)

    def test_video_preserves_original_rounded_timestamp(self):
        start, end = 161.035458, 167.792208
        event = {'start_seconds': 2.1, 'end_seconds': 6.75675}
        result = validate_video(json.dumps({'events': [event]}), end - start, start)
        self.assertEqual(result['events'][0]['end_seconds'], 6.75675)
        self.assertEqual(result['events'][0]['episode_end_seconds'], end)
