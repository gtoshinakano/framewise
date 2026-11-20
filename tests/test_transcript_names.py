import json
import unittest
from pathlib import Path

from app import transcript_names as N
from app.name_glossary import SEED_NAMES, SEED_OTHER

GLOSSARY = [dict(spelling=spelling, reading=reading, character_context_id=title, character_name=title, part=part)
            for title, names in SEED_NAMES.items() for part, (spelling, reading) in zip(('family', 'given'), names)]
GLOSSARY.append(dict(spelling=SEED_OTHER[0], reading=SEED_OTHER[1], character_context_id=None))
RULES = N.alias_rules(GLOSSARY)


class Names(unittest.TestCase):
    def test_katakana_readings_become_the_verified_spelling(self):
        text, changes = N.apply_aliases('二年一組のアズサガワサクタです。サクタ、カエデは？', RULES)
        self.assertEqual(text, '二年一組の梓川咲太です。咲太、かえでは？')
        self.assertTrue(changes)

    def test_two_kana_names_only_with_an_honorific_or_the_family_name(self):
        self.assertEqual(N.apply_aliases('マイさん、桜島マイだよ。', RULES)[0], '麻衣さん、桜島麻衣だよ。')
        self.assertEqual(N.apply_aliases('マイペースだね。', RULES)[0], 'マイペースだね。')
        self.assertEqual(N.apply_aliases('リオデジャネイロ', RULES)[0], 'リオデジャネイロ')

    def test_a_name_inside_a_longer_katakana_word_stays(self):
        self.assertEqual(N.apply_aliases('サクタロウ', RULES)[0], 'サクタロウ')

    def test_a_fix_may_only_swap_names(self):
        self.assertTrue(N.name_only_change('安沢川サクタです。', '梓川サクタです。', GLOSSARY))
        self.assertTrue(N.name_only_change('牧之原証拠以外', '牧之原翔子以外', GLOSSARY))
        self.assertFalse(N.name_only_change('安沢川です。', '梓川です！', GLOSSARY))                # punctuation changed
        self.assertFalse(N.name_only_change('安沢川です。', '相坂です。', GLOSSARY))                 # not a cast spelling
        self.assertFalse(N.name_only_change('おはよう。', 'おはよう、梓川。', GLOSSARY))              # a name added
        self.assertFalse(N.name_only_change('梓川です。', '梓川です。', GLOSSARY))

    def test_validate_keeps_good_fixes_and_explains_bad_ones(self):
        window = [{'id': 'L1', 'ja': '安沢川です。'}, {'id': 'L2', 'ja': 'おはよう。'}]
        fixes, errors = N.validate({'fixes': [{'id': 'L1', 'ja': '梓川です。'}, {'id': 'L2', 'ja': 'おはよう、咲太。'}]}, window, GLOSSARY)
        self.assertEqual(fixes, {'L1': '梓川です。'})
        self.assertEqual(len(errors), 1)
        self.assertIn('L2', errors[0])
    def test_all_labelled_name_fixes(self):
        fixture = Path(__file__).parents[1] / 'audio_experiment' / 'labels' / 'name_fixes.json'
        for row in json.loads(fixture.read_text())['fixes']:
            if row['verdict'] == 'unsure':
                continue
            with self.subTest(line=row['id'], episode=row['episode'], verdict=row['verdict']):
                window = [{'id': row['id'], 'ja': row['before']}]
                answer = {'fixes': [{'id': row['id'], 'ja': row['after']}]}
                fixes, errors = N.validate(answer, window, GLOSSARY)
                if row['verdict'] == 'ok':
                    self.assertEqual(fixes, {row['id']: row['after']})
                    self.assertEqual(errors, [])
                else:
                    self.assertFalse(fixes)
                    self.assertTrue(errors)
