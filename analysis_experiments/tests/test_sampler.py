import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sampler  # noqa: E402

SCHEMA = '''
CREATE TABLE episodes (id TEXT PRIMARY KEY, project_id TEXT, title TEXT, original_ext TEXT, width INT, height INT);
CREATE TABLE cuts (id TEXT PRIMARY KEY, episode_id TEXT, cut_index INT, start_time REAL, end_time REAL, active INT DEFAULT 1, first_frame_time REAL, last_frame_time REAL,
  analysis_enabled INT DEFAULT 1, no_characters INT DEFAULT 0);
CREATE TABLE analysis_results (id TEXT PRIMARY KEY, cut_id TEXT, kind TEXT, content TEXT, settings_json TEXT, context_json TEXT DEFAULT '{}', superseded_by TEXT);
CREATE TABLE context_items (id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE character_detection_runs (id TEXT PRIMARY KEY, cut_id TEXT, cut_start REAL, cut_end REAL, frames_json TEXT);
CREATE TABLE character_detections (id TEXT PRIMARY KEY, run_id TEXT, position TEXT, box_json TEXT, character_context_id TEXT);
CREATE TABLE transcript_segments (id TEXT PRIMARY KEY, episode_id TEXT, start_time REAL, end_time REAL, text TEXT, character_context_id TEXT, superseded_by TEXT, deleted INT DEFAULT 0);
'''


def build(directory, cuts):
    """``cuts``: list of (duration, [names], background, video_names, composition)."""
    path = Path(directory) / 'app.sqlite3'
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO episodes VALUES ('e1','p1','Episode 1','.mp4',1280,720)")
    conn.execute("INSERT INTO episodes VALUES ('e2','p1','Episode 2','.mp4',1280,720)")
    for ep in ('e1', 'e2'):
        folder = Path(directory) / 'projects' / 'p1' / 'episodes' / ep
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'original.mp4').write_bytes(b'x')
    for name in ('Mai', 'Sakuta', 'Rio'):
        conn.execute('INSERT INTO context_items VALUES (?,?)', (f'c-{name}', name))
    for number, (duration, names, background, video_names, composition) in enumerate(cuts):
        cut_id = f'cut{number}'
        start = number * 20.0
        conn.execute('INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time,first_frame_time,last_frame_time) VALUES (?,?,?,?,?,?,?)',
                     (cut_id, 'e1' if number % 2 else 'e2', number, start, start + duration, start, start + duration - .04))
        first = {'mode': 'full', 'composition': composition, 'background_people': 'two students' if background else 'none',
                 'visible_characters': [{'character': n, 'position': 'left', 'action': 'standing'} for n in names]}
        video = {'contract': 'direction-v2', 'direction': {'summary': 's', 'characters': [{'character': n, 'activity': 'standing'} for n in video_names]}}
        conn.execute('INSERT INTO analysis_results VALUES (?,?,?,?,?,?,NULL)', (f'a{number}', cut_id, 'first_frame', json.dumps(first), json.dumps({'structured_type': 'initial_frame', 'generation_seconds': 20}), '{}'))
        conn.execute('INSERT INTO analysis_results VALUES (?,?,?,?,?,?,NULL)', (f'v{number}', cut_id, 'cut_description', json.dumps(video), json.dumps({'structured_type': 'video_analysis'}), '{}'))
        conn.execute('INSERT INTO character_detection_runs VALUES (?,?,?,?,?)', (f'r{number}', cut_id, start, start + duration, json.dumps([{'position': 'first', 'image_size': [1280, 720]}])))
        for slot, name in enumerate(names):
            conn.execute('INSERT INTO character_detections VALUES (?,?,?,?,?)', (f'd{number}{slot}', f'r{number}', 'first', json.dumps([600 - slot * 300, 72, 800 - slot * 300, 648]), f'c-{name}'))
    conn.commit()
    conn.close()
    return path


class SamplerTest(unittest.TestCase):
    def test_candidates_are_marked_left_to_right_on_a_1000_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build(directory, [(3.0, ['Mai', 'Sakuta'], False, ['Mai', 'Sakuta'], 'two characters')])
            conn = sampler.connect_readonly(path)
            found = sampler.load_candidates(conn, directory)
            conn.close()
        self.assertEqual(len(found), 1)
        marked = found[0]['marked']
        self.assertEqual([(p['handle'], p['name']) for p in marked], [('B1', 'Sakuta'), ('B2', 'Mai')])  # Sakuta's box is further left
        self.assertEqual(marked[0]['box'], [234, 100, 391, 900])
        self.assertEqual(marked[1]['box'], [469, 100, 625, 900])

    def test_stale_runs_and_mismatched_names_are_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build(directory, [(3.0, ['Mai'], False, ['Mai'], ''), (3.0, ['Mai'], False, ['Mai'], '')])
            conn = sqlite3.connect(path)
            conn.execute("UPDATE character_detection_runs SET cut_end=cut_end+1 WHERE cut_id='cut0'")
            conn.execute("UPDATE character_detections SET character_context_id='c-Rio' WHERE run_id='r1'")
            conn.commit()
            conn.close()
            ro = sampler.connect_readonly(path)
            found = sampler.load_candidates(ro, directory)
            ro.close()
        self.assertEqual(found, [])

    def test_missing_original_and_old_contract_are_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build(directory, [(3.0, ['Mai'], False, ['Mai'], ''), (3.0, ['Mai'], False, ['Mai'], '')])
            conn = sqlite3.connect(path)
            conn.execute("UPDATE analysis_results SET content=? WHERE id='v1'", (json.dumps({'direction': {}}),))
            conn.commit()
            conn.close()
            ro = sampler.connect_readonly(path)
            self.assertEqual([c['cut_id'] for c in sampler.load_candidates(ro, directory)], ['cut0'])
            ro.close()
            (Path(directory) / 'projects/p1/episodes/e2/original.mp4').unlink()
            ro = sampler.connect_readonly(path)
            self.assertEqual(sampler.load_candidates(ro, directory), [])
            ro.close()

    def test_picks_fill_strata_deterministically_and_spread_episodes(self):
        specs = []
        for index in range(30):
            specs.append((1.5 + (index % 3) * 0.2, ['Mai'], False, ['Mai'], 'close-up of her face' if index % 5 == 0 else 'wide'))
        specs += [(5.0, ['Mai'], False, ['Mai'], 'close-up portrait'), (5.0, ['Mai'], False, ['Mai'], 'wide room shot'), (4.0, [], False, ['Mai'], 'door'), (4.0, [], False, [], 'sky')]
        with tempfile.TemporaryDirectory() as directory:
            path = build(directory, specs)
            first, missing, total = sampler.build_samples(path, directory)
            second, _, _ = sampler.build_samples(path, directory)
        self.assertEqual([s['cut_id'] for s in first], [s['cut_id'] for s in second])
        self.assertEqual(total, len(specs))
        close = next(s for s in first if s['reason'].endswith('close-up'))
        self.assertIn('close-up', close['stored']['first_frame']['composition'])
        self.assertTrue(any(s['reason'].startswith('No people') for s in first))
        self.assertTrue(any('characters enter later (mid)' in s['reason'] for s in first))
        self.assertIn('Long (8-12 s), one character', missing)
        self.assertEqual({s['id'] for s in first}, {f's{i:02d}' for i in range(1, len(first) + 1)})
        shorts = [s for s in first if s['reason'] == 'Short (2 s or less), one character']
        self.assertEqual(len({s['episode'] for s in shorts}), 2)
        self.assertEqual(sum(s['reason'].startswith('Short') for s in first), 2)

    def test_strata_total_twenty(self):
        self.assertEqual(sum(count for _, count, _, _ in sampler.STRATA), 20)


if __name__ == '__main__':
    unittest.main()
