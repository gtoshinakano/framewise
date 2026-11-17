import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import fake_backend  # noqa: E402
import infer  # noqa: E402
import judge  # noqa: E402
import prompts  # noqa: E402
import report  # noqa: E402
import sampler  # noqa: E402
import schema  # noqa: E402
import scoring  # noqa: E402


def frames(n, step=0.4):
    return [{'label': f'F{k}', 'k': k, 't': round((k - 1) * step, 3), 'path': f'frames/c001/F{k}.jpg'} for k in range(1, n + 1)]


def sample(n=5, sid='c001'):
    fs = frames(n)
    return {'id': sid, 'cut_number': 1, 'duration': fs[-1]['t'] + 0.2, 'start': 10.0, 'end': 10.0 + fs[-1]['t'] + 0.2, 'frames': fs,
            'windows': sampler.make_windows(fs, 5), 'roster': [{'name': 'Tomoe Koga', 'box': [500, 0, 900, 1000], 'side': 'right'}],
            'transcript': [{'start': 0.1, 'end': 0.6, 'text': 'hi', 'speaker': 'Tomoe Koga', 'character': 'Tomoe Koga'}],
            'subtitle_band': {'top': 0.8, 'bottom': 1.0}, 'no_characters': False, 'keyframes': {'origin': 'cached'},
            'pairs': [{'from': a['label'], 'to': b['label'], 'start': a['t'], 'end': b['t'], 'camera': {'move': 'static'}, 'global': 0.1, 'regional': 0.2,
                       'still': False, 'boxes': {'Tomoe Koga': 0.3}} for a, b in zip(fs, fs[1:])],
            'baseline': {'view': {'camera': {'movement': 'static', 'detail': 'none'}, 'summary': 's',
                                  'characters': [{'character': 'Tomoe Koga', 'speaking': 'yes'}],
                                  'events': [{'seconds': '0.0-0.6', 'subject': 'Tomoe Koga', 'action': 'Raises her right hand and turns her head.'}]},
                         'seconds': 40.0, 'model': 'Qwen/Qwen3-VL-8B-Instruct'}}


def good_answer(window, name='Tomoe Koga'):
    names = schema.labels(window)
    return {'frames': [{'frame': label, 'characters': [{'name': name, 'state': 'standing'}]} for label in names],
            'segments': [{'from': a, 'to': b, 'camera': {'move': 'static', 'detail': ''},
                          'characters': [{'name': name, 'walking': 'no', 'speaking': 'yes', 'blinking': 'no',
                                          'changes': [{'part': 'head', 'description': 'turns left'}]}], 'other_changes': []}
                         for a, b in zip(names, names[1:])], 'summary': 'x'}


class Windows(unittest.TestCase):
    def test_overlap_by_one(self):
        windows = sampler.make_windows(frames(9))
        self.assertEqual([[f['label'] for f in w['frames']] for w in windows], [['F1', 'F2', 'F3', 'F4', 'F5'], ['F5', 'F6', 'F7', 'F8', 'F9']])
        windows = sampler.make_windows(frames(10))
        self.assertEqual([len(w['frames']) for w in windows], [5, 5, 2])
        self.assertEqual(len(sampler.make_windows(frames(1))), 1)
        self.assertEqual([len(w['frames']) for w in sampler.make_windows(frames(4))], [4])

    def test_parse_cuts(self):
        self.assertEqual(sampler.parse_cuts('3-5, 9,1'), [1, 3, 4, 5, 9])
        with self.assertRaises(ValueError):
            sampler.parse_cuts('a-b')


class Schema(unittest.TestCase):
    def test_valid_and_coerced(self):
        window = sample()['windows'][0]
        answer = good_answer(window)
        answer['segments'][0]['camera']['move'] = 'Pan Left'
        answer['segments'][0]['characters'][0]['walking'] = True
        clean, errors = schema.validate(answer, window)
        self.assertEqual(errors, [])
        self.assertEqual(clean['segments'][0]['camera']['move'], 'pan_left')
        self.assertEqual(clean['segments'][0]['characters'][0]['walking'], 'yes')
        self.assertEqual(clean['segments'][1]['start'], 0.4)

    def test_rejects_wrong_count_and_pairs(self):
        window = sample()['windows'][0]
        answer = good_answer(window)
        answer['segments'] = answer['segments'][:2]
        self.assertTrue(schema.validate(answer, window)[1])
        answer = good_answer(window)
        answer['segments'][1]['from'] = 'F3'
        self.assertTrue(schema.validate(answer, window)[1])

    def test_duplicate_character_is_merged(self):
        window = sample()['windows'][0]
        answer = good_answer(window)
        twin = dict(answer['segments'][0]['characters'][0], walking='yes', changes=[{'part': 'arms', 'description': 'raises the arm'}])
        answer['segments'][0]['characters'].append(twin)
        clean, errors = schema.validate(answer, window)
        self.assertEqual(errors, [])
        person = clean['segments'][0]['characters']
        self.assertEqual(len(person), 1)
        self.assertEqual(person[0]['walking'], 'yes')
        self.assertEqual(len(person[0]['changes']), 2)

    def test_frames_required(self):
        window = sample()['windows'][0]
        answer = good_answer(window)
        answer['frames'] = answer['frames'][:3]
        self.assertTrue(schema.validate(answer, window)[1])
        self.assertEqual(schema.json_schema(window)['required'][0], 'frames')

    def test_schema_fixes_count(self):
        window = sample()['windows'][0]
        grammar = schema.json_schema(window)
        self.assertEqual(grammar['properties']['segments']['minItems'], 4)

    def test_parse_json_with_prose(self):
        self.assertEqual(schema.parse_json('Sure: {"a": "}"} trailing'), {'a': '}'})

    def test_claims(self):
        window = sample()['windows'][0]
        clean, _ = schema.validate(good_answer(window), window)
        claims = schema.answer_claims(clean, window, 'm|s|c', 'c001')
        cats = [c['category'] for c in claims]
        self.assertEqual(cats.count('camera'), 4)
        self.assertEqual(cats.count('speaking'), 4)
        self.assertEqual(cats.count('head'), 4)
        self.assertTrue(all(not c['judge'] for c in claims if c['category'] == 'camera'))

    def test_no_change_entries_are_not_claims(self):
        window = sample()['windows'][0]
        answer = good_answer(window)
        answer['segments'][0]['characters'][0]['changes'].append({'part': 'legs', 'description': 'no change, legs remain stationary'})
        clean, _ = schema.validate(answer, window)
        claims = schema.answer_claims(clean, window, 'm|s|c', 'c001')
        self.assertFalse([c for c in claims if c['category'] == 'legs'])
        self.assertTrue(schema.is_no_change('arm remains extended toward the girl'))
        self.assertTrue(schema.is_no_change("Tomoe's green top and white skirt remain unchanged"))
        self.assertFalse(schema.is_no_change('raises the right arm'))
        self.assertNotIn('legs', schema.segment_categories(clean['segments'][0]))
        self.assertEqual(schema.json_schema(window)['properties']['segments']['items']['properties']['characters']['maxItems'], schema.MAX_CHARACTERS)

    def test_baseline(self):
        s = sample()
        answer = schema.baseline_answer(s['baseline']['view'], s['windows'][0])
        self.assertEqual(len(answer['segments']), 4)
        first = answer['segments'][0]['characters'][0]
        self.assertEqual(first['speaking'], 'yes')
        self.assertTrue(first['changes'])  # the event overlaps the first pair
        self.assertFalse([p for p in answer['segments'][3]['characters'] if p['changes']])  # 1.2-1.6 s has no event


class Prompts(unittest.TestCase):
    def test_contexts_build(self):
        s = sample()
        images = ['AAAA'] * 5
        for style in prompts.STYLES:
            for context in prompts.CONTEXTS:
                messages = prompts.build_messages(s, s['windows'][0], style, context, images, previous={'summary': 'p', 'segments': []})
                text = prompts.prompt_text(messages)
                self.assertIn('F1 · t=0.00 s', text)
                self.assertEqual(text.count('<image>'), 5)
                if context == 'frames':
                    self.assertNotIn('Tomoe Koga', text)
                if context == 'full':
                    self.assertIn('Transcript lines', text)


class Pipeline(unittest.TestCase):
    def test_fake_run_scores_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'r1'
            (run / 'frames' / 'c001').mkdir(parents=True)
            samples = [sample(9)]
            for f in samples[0]['frames']:
                (run / f['path']).write_bytes(b'\xff\xd8fake')
            (run / 'samples.json').write_text(json.dumps(samples))
            (run / 'config.json').write_text(json.dumps({'episode': 'E', 'cuts': '1', 'window_size': 5, 'models': ['lfm3']}))
            (run / 'progress.json').write_text(json.dumps({'status': 'done', 'stages': {}}))
            client = fake_backend.FakeClient()
            for model in ('lfm3', 'qwen35'):
                store = infer.Store(run, f'answers-{model}')
                infer.run_model(store, samples, model, [(s, c) for s in prompts.STYLES for c in prompts.CONTEXTS], client, run)
                self.assertEqual(len(store.records), 2 * 6)
            answers = report.load_answers(run)
            jstore = infer.Store(run, 'judge')
            judge.run_judge(jstore, samples, answers, client, run, parallel=2)
            self.assertTrue(any(r['kind'] == 'reference' for r in jstore.records.values()))
            # Resume: nothing new to do.
            before = len(jstore.records)
            judge.run_judge(jstore, samples, answers, client, run)
            self.assertEqual(len(jstore.records), before)
            results = report.build(run)
            self.assertIn(infer.BASELINE_KEY, results['rows'])
            row = results['rows']['lfm3|observer|roster']
            self.assertIsNotNone(row['score'])
            self.assertIsNotNone(row['components']['precision'])
            self.assertTrue((run / 'report.html').read_text().startswith('<!doctype html>'))
            self.assertIn('Leaderboard', (run / 'report.md').read_text())
            report.build_index(Path(tmp))
            self.assertTrue((Path(tmp) / 'index.html').is_file())

    def test_camera_agreement(self):
        self.assertEqual(scoring.camera_agreement('zoom_in', 'zoom_in'), 1.0)
        self.assertEqual(scoring.camera_agreement('dolly_in', 'zoom_in'), 1.0)
        self.assertEqual(scoring.camera_agreement('pan_left', 'pan_right'), 0.5)
        self.assertEqual(scoring.camera_agreement('static', 'pan_right'), 0.0)
        self.assertIsNone(scoring.camera_agreement('unclear', 'static'))

    def test_names(self):
        roster = [{'name': 'Tomoe Koga'}]
        self.assertTrue(scoring.name_ok('Tomoe', roster))
        self.assertTrue(scoring.name_ok('unknown person 1', roster))
        self.assertTrue(scoring.name_ok('girl with brown hair', roster))
        self.assertFalse(scoring.name_ok('Mai', roster))

    def test_reference_validation(self):
        window = sample()['windows'][0]
        pairs = [{'from': a, 'to': b, 'camera': 'static', 'characters': [{'name': 'x', 'head': True}]} for a, b in zip(schema.labels(window), schema.labels(window)[1:])]
        clean, errors = judge.validate_reference({'pairs': pairs}, window)
        self.assertEqual(errors, [])
        self.assertEqual(clean[0]['categories'], ['head'])


if __name__ == '__main__':
    unittest.main()
