"""scripts/fw: cut ranges, episode names, step → request mapping and the gated flow, against a fake API."""
import argparse
import contextlib
import importlib.machinery
import io
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'fw'
loader = importlib.machinery.SourceFileLoader('fw_cli', str(PATH))
spec = importlib.util.spec_from_loader('fw_cli', loader)
fw = importlib.util.module_from_spec(spec)
loader.exec_module(fw)

EPISODE = {'id': 'ep-1111-2222', 'title': 'Episode 1', 'project': 'Bunny Girl', 'project_id': 'p', 'status': 'ready', 'created_at': '2026-11-01'}
CUTS = [{'id': f'c{n}', 'cut_index': n - 1, 'start_time': n * 2.0, 'end_time': n * 2.0 + 2, 'analysis_enabled': n != 3} for n in range(1, 6)]


def report(**changes):
    base = {'episode': {'id': EPISODE['id'], 'subtitle_language': 'Japanese'},
            'setup': {'media': True, 'cuts': True, 'transcript': True, 'subtitles': True, 'boxes': True},
            'gates': {'first_frame': [], 'target_frames': [], 'voices': [], 'decisions': []},
            'checks': {'speaker_attribution_stale': None},
            'progress': {'cuts': 5, 'enabled': 4, 'boxes_missing': [], 'target_missing': [], 'visual_missing': [], 'beats_missing': [],
                         'consistency_missing': [], 'summary_missing': []},
            'failures': []}
    for key, value in changes.items():
        section, _, name = key.partition('__')
        if name:
            base[section] = {**base[section], name: value}
        else:
            base[section] = value
    return base


IDLE = {'running': [], 'waiting_total': 0}


class FakeApi:
    url = 'http://framewise'

    def __init__(self, review=None, dry_run=False):
        self.calls, self.review, self.dry_run = [], review or report(), dry_run

    def get(self, path, params=None):
        if path == '/api/projects':
            return [{'id': 'p', 'title': 'Bunny Girl', 'episodes': [{k: v for k, v in EPISODE.items() if k not in ('project', 'project_id')}]}]
        if path.endswith('/cuts'):
            return {'items': CUTS, 'next_offset': None}
        if path.endswith('/review'):
            return self.review
        if path.endswith('/queue'):
            return IDLE
        raise AssertionError(path)

    def post(self, path, params=None, body=None):
        self.calls.append((path, {k: v for k, v in (params or {}).items() if v is not None}, body))
        return {'job_ids': [f'job{len(self.calls)}']}


def flags(**values):
    return argparse.Namespace(**{'redo': False, 'save': False, 'language': None, 'replace_manual': False, 'json': False,
                                 'episode': 'Episode 1', 'latest': False, 'range': 'all', **values})


class RangeTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(fw.parse_range('100-120'), (100, 120))
        self.assertEqual(fw.parse_range('100..120'), (100, 120))
        self.assertEqual(fw.parse_range('100 to 120'), (100, 120))
        self.assertEqual(fw.parse_range('7'), (7, 7))
        self.assertEqual(fw.parse_range('5-'), (5, None))
        self.assertEqual(fw.parse_range('-10'), (None, 10))
        self.assertEqual(fw.parse_range('all'), (None, None))
        for bad in ('10-5', '0-3', 'x', '-'):
            with self.assertRaises(fw.Failure):
                fw.parse_range(bad)

    def test_numbers_text(self):
        self.assertEqual(fw.numbers_text([10, 1, 2, 3, 5, 9]), '1-3, 5, 9-10')


class EpisodeTest(unittest.TestCase):
    EPISODES = [{'id': 'aaaaaaaa-1', 'title': 'Episode 1', 'project': 'Bunny Girl'}, {'id': 'bbbbbbbb-2', 'title': 'Episode 10', 'project': 'Bunny Girl'},
                {'id': 'cccccccc-3', 'title': 'Episode 1', 'project': 'Smoke test'}]

    def test_matching(self):
        match = lambda ref: [e['id'] for e in fw.match_episodes(self.EPISODES, ref)]  # noqa: E731
        self.assertEqual(match('bbbbbbbb-2'), ['bbbbbbbb-2'])
        self.assertEqual(match('bbbbbbbb'), ['bbbbbbbb-2'])
        self.assertEqual(match('Episode 1'), ['aaaaaaaa-1', 'cccccccc-3'])  # exact title wins over "Episode 10", still ambiguous
        self.assertEqual(match('bunny / episode 1'), ['aaaaaaaa-1'])
        self.assertEqual(match('smoke'), ['cccccccc-3'])


class StepTest(unittest.TestCase):
    def step(self, name, rng='all', review=None, **values):
        api = FakeApi(review)
        args = flags(step=name, range=rng, **values)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fw.cmd_step(api, args), fw.OK)
        return api.calls

    def test_batch_steps_send_the_range(self):
        self.assertEqual(self.step('boxes', '100-120'), [('/api/episodes/ep-1111-2222/character-boxes', {'from_cut': 100, 'to_cut': 120}, None)])
        self.assertEqual(self.step('visual', '5-10'),
                         [('/api/episodes/ep-1111-2222/visual-analysis', {'from_cut': 5, 'to_cut': 10, 'only_missing': True}, None)])
        self.assertEqual(self.step('summary', redo=True), [('/api/episodes/ep-1111-2222/cut-summaries', {'only_missing': False}, None)])
        self.assertEqual(self.step('recognize', '4-', redo=True)[0][:2], ('/api/episodes/ep-1111-2222/character-detection-rerun', {'from_cut': 4}))
        self.assertEqual(self.step('answers', '3')[0][1], {'from_cut': 3, 'to_cut': 3, 'auto_save': False})
        self.assertEqual(self.step('answers', save=True)[0][1], {'auto_save': True})

    def test_per_cut_steps_skip_disabled_cuts(self):
        self.assertEqual([call[0] for call in self.step('beats', '2-4')], ['/api/cuts/c2/performance-beats', '/api/cuts/c4/performance-beats'])

    def test_transcribe_range_covers_the_cuts(self):
        self.assertEqual(self.step('transcribe', '2-3'),
                         [('/api/episodes/ep-1111-2222/transcribe-range', {}, {'start_time': 4.0, 'end_time': 8.0, 'replace_manual': False})])
        self.assertEqual(self.step('transcribe')[0][0], '/api/episodes/ep-1111-2222/transcribe')

    def test_whole_episode_steps_refuse_a_range(self):
        api = FakeApi()
        with self.assertRaises(fw.Failure):
            fw.cmd_step(api, flags(step='setup', range='1-3'))
        self.assertEqual(self.step('setup'), [('/api/episodes/ep-1111-2222/prepare', {}, {'language': 'Japanese'})])

    def test_a_server_without_ranges_is_refused(self):
        api = FakeApi()
        api.get = lambda path, params=None: (_ for _ in ()).throw(fw.Failure('the server has no /review endpoint yet')) if path.endswith('/review') \
            else FakeApi.get(api, path, params)
        with self.assertRaises(fw.Failure):
            fw.cmd_step(api, flags(step='boxes', range='1-2'))
        self.assertEqual(api.calls, [])

    def test_explicit_steps_warn_about_open_checkpoints(self):
        lines = fw.gate_lines(report(gates__first_frame=[{'cut': 4, 'cut_id': 'c4', 'unlabeled_boxes': 1}]), fw.UPSTREAM['visual'])
        self.assertEqual(lines, ['unlabeled first-frame boxes: cuts 4'])


class PlanTest(unittest.TestCase):
    def test_busy_and_failures_come_first(self):
        self.assertEqual(fw.plan(report(), {'running': [{'label': 'Visual', 'status': 'running'}], 'waiting_total': 0})['action'], 'busy')
        self.assertEqual(fw.plan(report(failures=[{'id': 'j'}]), IDLE)['gate'], 'failures')

    def test_setup_needs_a_language(self):
        missing = report(setup__subtitles=False, episode={'id': 'e', 'subtitle_language': None})
        self.assertEqual(fw.plan(missing, IDLE)['gate'], 'language')
        self.assertEqual(fw.plan(missing, IDLE, 'Japanese'), {'action': 'run', 'step': 'setup', 'language': 'Japanese', 'detail': 'missing: subtitles'})

    def test_speaker_attribution_runs_again_unless_it_left_the_reason_itself(self):
        self.assertEqual(fw.plan(report(checks__speaker_attribution_stale='Name glossary changed'), IDLE)['step'], 'speakers')
        self.assertEqual(fw.plan(report(checks__speaker_attribution_stale=fw.UNANSWERED_ATTRIBUTION), IDLE)['action'], 'done')

    def test_order_of_steps_and_checkpoints(self):
        decision = lambda **changes: fw.plan(report(**changes), IDLE)  # noqa: E731
        box = [{'cut': 2, 'cut_id': 'c2', 'unlabeled_boxes': 1}]
        self.assertEqual(decision(gates__first_frame=box, progress__target_missing=[3])['gate'], 'first_frame')
        self.assertEqual(decision(progress__target_missing=[3], gates__target_frames=[{'cut': 1}])['step'], 'target-frames')
        self.assertEqual(decision(gates__target_frames=[{'cut': 1, 'frames': [], 'unnamed_people': 1}], progress__visual_missing=[2])['gate'], 'target_frames')
        self.assertEqual(decision(progress__visual_missing=[2, 4], progress__beats_missing=[1]), {'action': 'run', 'step': 'visual', 'cuts': [2, 4], 'detail': 'cuts 2, 4'})
        self.assertEqual(decision(progress__beats_missing=[1], gates__voices=[{'speaker_label': 'S1', 'lines': 2}])['step'], 'beats')
        self.assertEqual(decision(gates__voices=[{'speaker_label': 'S1', 'lines': 2}], progress__consistency_missing=[1])['gate'], 'voices')
        self.assertEqual(decision(progress__consistency_missing=[1])['step'], 'consistency')
        undrafted = [{'cut': 1, 'questions': 2, 'draft': False}, {'cut': 2, 'questions': 1, 'draft': True}]
        self.assertEqual(decision(gates__decisions=undrafted)['cuts'], [1])
        self.assertEqual(decision(gates__decisions=undrafted[1:], progress__summary_missing=[3])['gate'], 'decisions')
        self.assertEqual(decision(progress__summary_missing=[3])['step'], 'summary')
        self.assertEqual(decision()['action'], 'done')

    def test_execute_runs_per_cut_steps_only_for_the_planned_cuts(self):
        api = FakeApi()
        result = fw.execute(api, EPISODE, {'action': 'run', 'step': 'answers', 'cuts': [2, 5]}, flags())
        self.assertEqual([call[0] for call in api.calls], ['/api/cuts/c2/review-answers', '/api/cuts/c5/review-answers'])
        self.assertEqual(result['queued'], 2)
        api = FakeApi()
        fw.execute(api, EPISODE, {'action': 'run', 'step': 'visual', 'cuts': [2, 4]}, flags())
        self.assertEqual(api.calls, [('/api/episodes/ep-1111-2222/visual-analysis', {'from_cut': 2, 'to_cut': 4, 'only_missing': True}, None)])


if __name__ == '__main__':
    unittest.main()
