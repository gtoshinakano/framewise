import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db, media, outfit_library, sequence_direction as direction
from app.main import app


class OutfitLibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = (db.DATA_DIR, db.DB_PATH, media.DATA_DIR)
        db.DATA_DIR = media.DATA_DIR = Path(self.tmp.name)
        db.DB_PATH = db.DATA_DIR / 'app.sqlite3'
        db.init_db()
        with db.connect() as conn:
            conn.execute("INSERT INTO projects(id,title) VALUES ('p','P'),('other','O')")
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('mai','p','character','Mai Sakurajima',''),('x','other','character','Stranger','')")
        self.client = TestClient(app)

    def tearDown(self):
        db.DATA_DIR, db.DB_PATH, media.DATA_DIR = self.old
        self.tmp.cleanup()

    def body(self, **extra):
        return {'character_context_id': 'mai', 'name': 'Casual', 'description': 'White polo-collar blouse under a lavender cardigan. Pale blue skirt, white flats.', **extra}

    def test_the_library_is_shared_by_the_project_and_scoped_to_it(self):
        created = self.client.post('/api/projects/p/outfits', json=self.body())
        self.assertEqual(created.status_code, 200, created.text)
        row = created.json()[0]
        self.assertEqual((row['character_name'], row['origin'], row['summary']), ('Mai Sakurajima', 'user', 'White polo-collar blouse under a lavender cardigan.'))
        self.assertEqual(self.client.get('/api/projects/other/outfits').json(), [])
        self.assertEqual(self.client.delete(f"/api/projects/other/outfits/{row['id']}").status_code, 404)
        self.assertEqual(self.client.post('/api/projects/other/outfits', json=self.body()).status_code, 422)  # a character of another project
        self.assertEqual(self.client.get('/api/projects/missing/outfits').status_code, 404)

    def test_names_are_unique_per_character_and_edits_clear_the_review_flag(self):
        self.assertEqual(self.client.post('/api/projects/p/outfits', json=self.body()).status_code, 200)
        self.assertEqual(self.client.post('/api/projects/p/outfits', json=self.body(name='casual')).status_code, 409)
        with db.connect() as conn:
            conn.execute('UPDATE character_outfits SET needs_review=1')
        outfit_id = self.client.get('/api/projects/p/outfits').json()[0]['id']
        updated = self.client.put(f'/api/projects/p/outfits/{outfit_id}', json=self.body(summary='Lavender cardigan look.'))
        self.assertEqual((updated.json()[0]['summary'], updated.json()[0]['needs_review']), ('Lavender cardigan look.', 0))
        self.assertEqual(self.client.delete(f'/api/projects/p/outfits/{outfit_id}').json(), [])

    def test_summary_is_the_first_sentence_cut_at_a_word(self):
        self.assertEqual(outfit_library.summarize('Short one. Second sentence.'), 'Short one.')
        long = outfit_library.summarize('word ' * 80)
        self.assertTrue(long.endswith('…') and len(long) <= outfit_library.SUMMARY_CHARS + 1)

    def test_seed_adds_known_characters_once_and_flags_doubtful_text(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('fumi','p','character',' fumika nanjou ','')")
            conn.execute('DELETE FROM app_flags WHERE name=?', (outfit_library.SEED_FLAG,))
            outfit_library.seed(conn)
            outfit_library.seed(conn)  # the flag makes a second call do nothing
            rows = {(row['character_name'].strip().lower(), row['name']): row for row in outfit_library.rows(conn, 'p')}
        self.assertIn(('fumika nanjou', 'Pale Blue Blouse'), rows)
        self.assertEqual({name for (_, name) in rows if rows[(_, name)]['character_name'].startswith('Mai')}, {'Black Bunny Costume', 'School Uniform'})
        self.assertEqual((rows[('mai sakurajima', 'School Uniform')]['needs_review'], rows[('mai sakurajima', 'Black Bunny Costume')]['needs_review']), (1, 0))
        self.assertEqual(rows[('mai sakurajima', 'School Uniform')]['origin'], 'seed')
        self.assertEqual(len(rows), 3)  # Sakuta, Kaede and the others do not exist in this project

    def test_seed_file_is_valid(self):
        entries = json.loads(outfit_library.SEED_FILE.read_text(encoding='utf-8'))
        self.assertTrue(all(entry['character'] and entry['name'] and entry['description'] for entry in entries))
        self.assertEqual(len({(entry['character'], entry['name'].casefold()) for entry in entries}), len(entries))


CTX = {'characters': [{'id': 'mai', 'name': 'Mai', 'notes': ''}], 'scenarios': [], 'notes': [], 'results': {},
       'outfits': [{'id': 'o1', 'character_context_id': 'mai', 'name': 'School Uniform', 'summary': 'Camel blazer.', 'description': 'd'}]}
TIMELINE = {'cuts': [{'number': 1, 'id': 'c1'}], 'lines': []}
DESCRIPTION = 'Black fitted bunny costume with rabbit-ear headband, white cuffs and collar, black bow tie, sheer dark tights and black heels.'


def bible(outfits, **extra):
    return {'logline': 'l', 'synopsis': 's', 'spoken_language': 'Japanese', 'scenes': [{'from_cut': 1, 'to_cut': 1, 'location': 'Hall', 'time_of_day': 'day', 'lighting': 'soft', 'mood': 'calm', 'scenario': None}],
            'characters': [{'name': 'Mai', 'role': 'lead', 'anchors': 'long dark hair', 'outfit': 'x', 'arc': 'y'}], 'outfits': outfits, **extra}


class BibleOutfitsTest(unittest.TestCase):
    def check(self, outfits, **kwargs):
        with patch.object(direction.sequence_evidence, 'voice_evidence', return_value=[]):
            return direction.validate_bible(bible(outfits), CTX, TIMELINE, **kwargs)

    def test_an_existing_outfit_is_matched_by_name_and_takes_the_library_summary(self):
        out = self.check([{'character': 'mai', 'name': 'school uniform', 'status': 'new', 'scenes': ['s1', 'S9'], 'summary': 'invented', 'description': ''}])['outfits']
        self.assertEqual(out, [{'character': 'Mai', 'character_context_id': 'mai', 'name': 'School Uniform', 'status': 'existing', 'summary': 'Camel blazer.', 'scenes': ['S1'], 'outfit_id': 'o1'}])

    def test_a_new_outfit_needs_a_complete_description_and_a_known_character(self):
        with self.assertRaises(ValueError) as caught:
            self.check([{'character': 'Mai', 'name': 'Black Bunny', 'status': 'new', 'summary': 's', 'description': 'black bunny suit'}])
        self.assertIn('not in the library (School Uniform)', str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.check([{'character': 'Nobody', 'name': 'A', 'status': 'new', 'summary': '', 'description': DESCRIPTION}])
        self.assertIn('is not one of the bible', str(caught.exception))
        entry = self.check([{'character': 'Mai', 'name': 'Black Bunny', 'status': 'new', 'summary': '', 'description': DESCRIPTION}])['outfits'][0]
        self.assertEqual((entry['status'], entry['description'], entry['summary']), ('new', DESCRIPTION, DESCRIPTION))  # no full stop inside: the whole text is the first sentence

    def test_the_director_must_name_an_outfit_for_every_character_but_old_bibles_still_validate(self):
        with self.assertRaises(ValueError) as caught:
            self.check([], require_outfits=True)
        self.assertIn('needs at least one outfit entry', str(caught.exception))
        self.assertEqual(self.check([])['outfits'], [])

    def test_adopt_creates_new_outfits_once_and_never_overwrites_the_library(self):
        tmp = tempfile.TemporaryDirectory()
        old = (db.DATA_DIR, db.DB_PATH)
        db.DATA_DIR, db.DB_PATH = Path(tmp.name), Path(tmp.name) / 'app.sqlite3'
        try:
            db.init_db()
            with db.connect() as conn:
                conn.execute("INSERT INTO projects(id,title) VALUES ('p','P')")
                conn.execute("INSERT INTO context_items(id,project_id,kind,title,content) VALUES ('mai','p','character','Mai','')")
                entry = {'character': 'Mai', 'character_context_id': 'mai', 'name': 'Black Bunny', 'status': 'new', 'summary': 'Bunny suit.', 'description': DESCRIPTION}
                first = outfit_library.adopt(conn, 'p', 'seq1', {'outfits': [entry]})['outfits'][0]
                self.assertEqual((first['status'], 'description' in first, bool(first['outfit_id'])), ('new', False, True))
                again = outfit_library.adopt(conn, 'p', 'seq1', {'outfits': [{**entry, 'description': 'a different text that must not replace it' * 3}]})['outfits'][0]
                other = outfit_library.adopt(conn, 'p', 'seq2', {'outfits': [entry]})['outfits'][0]
                rows = outfit_library.rows(conn, 'p')
            self.assertEqual((again['status'], other['status'], len(rows), rows[0]['description'], rows[0]['origin'], rows[0]['source_sequence_id']), ('new', 'existing', 1, DESCRIPTION, 'director', 'seq1'))
        finally:
            db.DATA_DIR, db.DB_PATH = old
            tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
