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
            'checks': {},
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
    return argparse.Namespace(**{'redo': False, 'save': False, 'language': None, 'json': False,
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
                         [('/api/episodes/ep-1111-2222/transcribe-range', {}, {'start_time': 4.0, 'end_time': 8.0})])  # never replaces your lines
        self.assertEqual(self.step('transcribe')[0][0], '/api/episodes/ep-1111-2222/transcribe')

    def test_whole_episode_steps_refuse_a_range(self):
        api = FakeApi()
        with self.assertRaises(fw.Failure):
            fw.cmd_step(api, flags(step='setup', range='1-3'))
        self.assertEqual(self.step('setup', language='Japanese'), [('/api/episodes/ep-1111-2222/prepare', {}, {'language': 'Japanese'})])
        self.assertEqual(self.step('setup'), [('/api/episodes/ep-1111-2222/prepare', {}, {})])  # subtitles are optional (D37)

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

    def test_setup_never_waits_for_the_optional_subtitles(self):
        missing = report(setup__subtitles=False, episode={'id': 'e', 'subtitle_language': None})
        self.assertEqual(fw.plan(missing, IDLE)['action'], 'done')
        missing = report(setup__boxes=False, episode={'id': 'e', 'subtitle_language': None})
        self.assertEqual(fw.plan(missing, IDLE), {'action': 'run', 'step': 'setup', 'language': None, 'detail': 'missing: boxes'})

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


class LinksTest(unittest.TestCase):
    def setUp(self):
        self.api = FakeApi()
        self.episode = dict(EPISODE, project='Bunny Girl')
        self.base = 'http://framewise/#/episode/ep-1111-2222/cut/'

    def test_cut_urls_name_the_place(self):
        self.assertEqual(fw.cut_url(self.api, self.episode, 'c2'), self.base + 'c2')
        self.assertEqual(fw.cut_url(self.api, self.episode, 'c2', 'analysis'), self.base + 'c2/analysis')
        self.assertEqual(fw.cut_url(self.api, self.episode, 'c2', 'characters'), self.base + 'c2/analysis/characters')
        self.assertEqual(fw.cut_url(self.api, self.episode, 'c2', 'target-frames'), self.base + 'c2/analysis/target-frames')
        with self.assertRaises(fw.Failure):
            fw.cut_url(self.api, self.episode, 'c2', 'nowhere')

    def test_numbers_become_urls_by_the_editor_number(self):
        self.assertEqual(fw.url_for_number(self.api, self.episode, 4, 'analysis'), self.base + 'c4/analysis')
        self.assertIsNone(fw.url_for_number(self.api, self.episode, 99))
        self.assertEqual(fw.url_at_time(self.api, self.episode, 5.0), self.base + 'c2/analysis')  # c2 is 4-6 s
        self.assertIsNone(fw.url_at_time(self.api, self.episode, 500))

    def test_cuts_are_read_once(self):
        fw.url_for_number(self.api, self.episode, 1)
        fw.url_for_number(self.api, self.episode, 2)
        self.assertEqual(len(self.api.__dict__['_cut_numbers']), 1)

    def test_link_lines_stop_after_the_limit_and_say_how_to_get_the_rest(self):
        lines = fw.link_lines(self.api, self.episode, list(range(1, 6)), 'analysis')
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], '  - cut 1: ' + self.base + 'c1/analysis')
        old, fw.LINK_LIMIT = fw.LINK_LIMIT, 2
        try:
            lines = fw.link_lines(self.api, self.episode, [1, 2, 3, 4, 5], 'cut')
        finally:
            fw.LINK_LIMIT = old
        self.assertEqual(len(lines), 3)
        self.assertIn('3-5 --place cut', lines[-1])

    def test_review_lists_one_url_per_cut(self):
        gates = {'first_frame': [{'cut': 4, 'cut_id': 'c4', 'unlabeled_boxes': 2}],
                 'target_frames': [{'cut': 2, 'cut_id': 'c2', 'frames': [{'ordinal': 3, 'flags': ['low_match']}], 'unnamed_people': 1}],
                 'voices': [{'speaker_label': 'SPEAKER_01', 'lines': 3, 'first_time': 5.0}],
                 'decisions': [{'cut': 5, 'cut_id': 'c5', 'questions': 2, 'draft': True}]}
        text = fw.review_text(self.api, self.episode, report(gates=gates, failures=[
            {'id': 'j1', 'label': 'Cut summary', 'error': 'bad', 'cut_span': {'first': 1, 'last': 1}}],
            checks__stale_results=[{'cut': 3, 'cut_id': 'c3', 'title': 'Cut summary', 'reason': 'Edited'}], checks__auto_labels=[], checks__no_boxes=[],
            checks__picture_contradictions=[], checks__lines_to_check={'count': 0, 'lines': []}, checks__sequences_unconfirmed=[]))
        self.assertIn(f'- cut 4 · 2 box(es): {self.base}c4/analysis/characters', text)
        self.assertIn(f'- cut 2 · frame 3 low_match, 1 unnamed person(s): {self.base}c2/analysis/target-frames', text)
        self.assertIn(f'- cut 5 · 2 question(s) (a suggested draft is ready): {self.base}c5/analysis', text)
        self.assertIn(f'first at 5.0s: {self.base}c2/analysis', text)
        self.assertIn(f'- cut 3 · Cut summary — Edited: {self.base}c3/analysis', text)
        self.assertIn(self.base + 'c1', text)  # the failed run's first cut
        self.assertTrue(text.rstrip().endswith('episode: http://framewise/#/episode/ep-1111-2222'))

    def test_a_blocked_flow_prints_the_cuts_to_open(self):
        decision = fw.plan(report(gates__first_frame=[{'cut': 4, 'cut_id': 'c4', 'unlabeled_boxes': 1}]), IDLE)
        links = fw.blocked_links(self.api, self.episode, decision)
        self.assertEqual(links[-1], '  - cut 4: ' + self.base + 'c4/analysis/characters')

    def test_explicit_steps_print_links_with_their_warnings(self):
        api = FakeApi(report(gates__first_frame=[{'cut': 4, 'cut_id': 'c4', 'unlabeled_boxes': 1}]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fw.cmd_step(api, flags(step='visual', range='all'))
        self.assertIn('warning: open checkpoints', out.getvalue())
        self.assertIn(self.base + 'c4/analysis/characters', out.getvalue())

    def test_links_command_takes_a_list(self):
        self.assertEqual(fw.parse_cut_list('4,2,3-4'), [2, 3, 4])
        with self.assertRaises(fw.Failure):
            fw.parse_cut_list('5-')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fw.cmd_links(self.api, flags(cuts='2,9', place='target-frames'))
        self.assertIn(f'- cut 2: {self.base}c2/analysis/target-frames', out.getvalue())
        self.assertIn('no active cut: 9', out.getvalue())


PROGRESS = {'cuts': 5, 'enabled': 4, 'boxes_missing': [1, 2], 'target_missing': [3], 'visual_missing': [1, 4], 'first_frame_missing': [1],
            'keyframes_missing': [1, 4], 'beats_missing': [2], 'consistency_missing': [], 'summary_missing': [5], 'analysis_off': [3]}


class AnalysisStatusTest(unittest.TestCase):
    def test_every_step_is_listed_with_its_cuts(self):
        status = fw.analysis_status(report(progress=PROGRESS))
        steps = {item['key']: item for item in status['steps']}
        self.assertEqual(steps['boxes_missing']['text'], 'Needs first-frame character detection (boxes)')
        self.assertEqual(steps['boxes_missing']['cuts'], [1, 2])
        self.assertEqual(steps['first_frame_missing']['cuts'], [1])
        self.assertEqual(steps['keyframes_missing']['cuts'], [1, 4])
        self.assertEqual(steps['keyframes_missing']['step'], 'visual')
        self.assertEqual(steps['summary_missing']['place'], 'analysis')
        self.assertNotIn('consistency_missing', steps)
        self.assertNotIn('visual_missing', steps)  # split into its two stages

    def test_an_older_server_reports_visual_analysis_as_one_step(self):
        old = {key: value for key, value in PROGRESS.items() if key not in ('first_frame_missing', 'keyframes_missing', 'analysis_off')}
        steps = fw.analysis_status(report(progress=old))['steps']
        self.assertIn(('visual_missing', [1, 4]), [(item['key'], item['cuts']) for item in steps])
        self.assertNotIn('keyframes_missing', [item['key'] for item in steps])

    def test_people_and_checks(self):
        gates = {'first_frame': [{'cut': 4, 'cut_id': 'c4', 'unlabeled_boxes': 1}], 'target_frames': [], 'decisions': [{'cut': 5, 'draft': False}],
                 'voices': [{'speaker_label': None, 'lines': 2, 'first_time': 1.0}]}
        checks = {'stale_results': [{'cut': 2, 'title': 'x', 'reason': 'y'}, {'cut': 2, 'title': 'z', 'reason': 'y'}]}
        status = fw.analysis_status(report(progress=PROGRESS, gates=gates, checks=checks))
        self.assertEqual([(item['key'], item['cuts']) for item in status['person']],
                         [('first_frame', [4]), ('decisions', [5]), ('voices', [])])
        self.assertIn('(no label) (2 lines)', status['person'][2]['text'])
        look = {item['key']: item for item in status['look']}
        self.assertEqual(look['stale_results']['cuts'], [2])

    def test_analysis_command_prints_the_needs_and_the_links(self):
        api = FakeApi(report(progress=PROGRESS))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(fw.cmd_analysis(api, flags(links=True)), fw.OK)
        text = out.getvalue()
        self.assertIn('Needs first-frame character detection (boxes): cuts 1-2  [fw boxes]', text)
        self.assertIn('Needs keyframe analysis: cuts 1, 4  [fw visual]', text)
        self.assertIn('analysis off (left out of everything below): cuts 3', text)
        self.assertIn('- cut 1: http://framewise/#/episode/ep-1111-2222/cut/c1/analysis/characters', text)
        self.assertIn('- cut 5: http://framewise/#/episode/ep-1111-2222/cut/c5/analysis', text)
        plain = io.StringIO()
        with contextlib.redirect_stdout(plain):
            fw.cmd_analysis(api, flags(links=False))
        self.assertNotIn('/cut/', plain.getvalue())


if __name__ == '__main__':
    unittest.main()
