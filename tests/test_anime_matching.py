import math
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from app import anime_matching, db


class AnimeMatchingTest(unittest.TestCase):
    def test_ranking_groups_examples_by_character_and_rejects_close_competitors(self):
        refs = [
            {'character_context_id': 'a', 'embedding': [1, 0]},
            {'character_context_id': 'a', 'embedding': [0.9, 0.1]},
            {'character_context_id': 'b', 'embedding': [0.99, 0.1]},
        ]
        ranked = anime_matching.rank_references([1, 0], refs)
        self.assertEqual([item[0] for item in ranked], ['a', 'b'])
        self.assertIsNone(anime_matching.confident_match(ranked))
        self.assertEqual(anime_matching.confident_match(ranked[:1]), 'a')
        self.assertIsNone(anime_matching.confident_match([('a', .79)]))

    def test_confident_match_defaults_and_overrides(self):
        self.assertEqual(anime_matching.confident_match([('a', .90), ('b', .86)]), 'a')
        self.assertIsNone(anime_matching.confident_match([('a', .90), ('b', .88)]))
        self.assertIsNone(anime_matching.confident_match([('a', .84)]))
        self.assertIsNone(anime_matching.confident_match([('a', .90), ('b', .86)], margin=.08))
        with unittest.mock.patch.dict('os.environ', {'ANIME_REID_MIN_MARGIN': '0.01', 'ANIME_REID_MIN_SIMILARITY': '0.5'}):
            self.assertEqual(anime_matching.confident_match([('a', .60), ('b', .58)]), 'a')
        with unittest.mock.patch.dict('os.environ', {'ANIME_REID_MIN_MARGIN': '2'}), self.assertRaises(ValueError):
            anime_matching.confident_match([('a', .90)])

    def test_invalid_embeddings_fail_instead_of_matching(self):
        for vector in ([], [0, 0], [math.nan, 1], [math.inf, 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                anime_matching.normalize(vector)
        with self.assertRaises(ValueError):
            anime_matching.rank_references([1, 0], [{'character_context_id': 'a', 'embedding': [1]}])

    def test_tracks_require_same_cut_short_gap_and_unambiguous_geometry(self):
        def detection(cut, time, box, embedding=(1, 0)):
            return {'cut_id': cut, 'time': time, 'box': box, 'image_size': [100, 100], 'embedding': embedding}
        old = [detection('a', 0, [0, 0, 50, 50])]
        self.assertEqual(anime_matching.match_tracks(old, [detection('a', .5, [1, 1, 51, 51])]), {0: 0})
        self.assertEqual(anime_matching.match_tracks(old, [detection('b', .5, [1, 1, 51, 51])]), {})
        self.assertEqual(anime_matching.match_tracks(old, [detection('a', 2, [1, 1, 51, 51])]), {})
        self.assertEqual(anime_matching.match_tracks(old, [detection('a', .5, [1, 1, 51, 51], (0, 1))]), {})
        self.assertEqual(anime_matching.match_tracks(old, [detection('a', .5, [1, 1, 51, 51])] * 2), {})

    def test_adjacent_cut_links_require_a_unique_strong_appearance(self):
        previous = [{'embedding': [1, 0]}]
        self.assertEqual(anime_matching.match_adjacent_appearances(previous, [{'embedding': [1, 0]}]), {0: 0})
        self.assertEqual(anime_matching.match_adjacent_appearances(previous, [{'embedding': [1, .5]}]), {})
        self.assertEqual(anime_matching.match_adjacent_appearances(previous, [{'embedding': [1, 0]}, {'embedding': [1, 0]}]), {})

    def test_reference_cache_is_model_specific_and_removed_with_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            old_data, old_path = db.DATA_DIR, db.DB_PATH
            db.DATA_DIR, db.DB_PATH = Path(temp), Path(temp) / 'test.sqlite3'
            try:
                db.init_db()
                with db.connect() as conn:
                    conn.execute("INSERT INTO projects(id,title) VALUES ('p','Project')")
                    conn.execute("INSERT INTO episodes(id,project_id,title,original_filename,original_ext,settings_json) VALUES ('e','p','E','a.mp4','.mp4','{}')")
                    conn.execute("INSERT INTO cuts(id,episode_id,cut_index,start_time,end_time) VALUES ('c','e',0,0,1)")
                    conn.execute("INSERT INTO context_items(id,project_id,kind,title) VALUES ('x','p','character','A')")
                    conn.execute("INSERT INTO character_detection_runs(id,cut_id,cut_start,cut_end,frames_json,model_id) VALUES ('run','c',0,1,'[]','test')")
                    conn.execute("INSERT INTO character_detections(id,run_id,position,box_json) VALUES ('d','run','first','[0,0,1,1]')")
                    conn.execute("INSERT INTO character_image_references(id,character_context_id,detection_id) VALUES ('r','x','d')")
                    anime_matching.write_cached_reference(conn, 'r', 'dino', 'revision-a', '448', [3, 4])
                    cached = anime_matching.read_cached_reference(conn, 'r', 'dino', 'revision-a', '448')
                    self.assertAlmostEqual(cached[0], .6)
                    self.assertIsNone(anime_matching.read_cached_reference(conn, 'r', 'dino', 'revision-b', '448'))
                    conn.execute("DELETE FROM character_image_references WHERE id='r'")
                    self.assertEqual(conn.execute('SELECT count(*) FROM character_reference_embeddings').fetchone()[0], 0)
            finally:
                db.DATA_DIR, db.DB_PATH = old_data, old_path


if __name__ == '__main__':
    unittest.main()
