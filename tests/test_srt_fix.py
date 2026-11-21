import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / '.agents' / 'skills' / 'framewise-srt-fix' / 'srt_fix.py'
loader = importlib.machinery.SourceFileLoader('srt_fix', str(PATH))
spec = importlib.util.spec_from_loader('srt_fix', loader)
srt = importlib.util.module_from_spec(spec)
loader.exec_module(srt)

GLOSSARY = [{'spelling': '咲太', 'character_context_id': 'sakuta'}, {'spelling': '理央', 'character_context_id': 'rio'},
            {'spelling': '梓川', 'character_context_id': 'sakuta'}, {'spelling': '梓川', 'character_context_id': 'kaede'},
            {'spelling': '麻衣', 'character_context_id': 'mai'}]
CHARACTERS = [{'id': 'sakuta', 'title': 'Sakuta Azusagawa'}, {'id': 'rio', 'title': 'Rio Futaba'},
              {'id': 'kaede', 'title': 'Kaede Azusagawa'}, {'id': 'mai', 'title': 'Mai Sakurajima'}]


def line(key, start, end, text, character='rio', source='voice'):
    return {'id': key, 'start_time': start, 'end_time': end, 'text': text, 'speaker_label': 'SPEAKER_05',
            'character_context_id': character, 'speaker_source': source, 'confidence': 'high', 'words_json': None}


def cue(number, start, end, *rows):
    return f'{number}\n00:00:{start:06.3f} --> 00:00:{end:06.3f}\n'.replace('.', ',') + '\n'.join(rows) + '\n'


def plan(lines, *cues):
    parsed = srt.parse_srt('\n'.join(cues))
    return srt.build_plan(lines, parsed, [0.0] * len(parsed), GLOSSARY, CHARACTERS)


def text_items(items):
    return [item for item in items if item['kind'] == 'text']


class ParseTests(unittest.TestCase):
    def test_tags_markers_and_lyrics(self):
        raw = '﻿' + '\r\n'.join(['1', '00:00:01,970 --> 00:00:04,700', '(理央)これは', '思春期症候群ではなく', '',
                                      '2', '00:00:05,000 --> 00:00:06,000', '[℡](南条)もしもし？', '',
                                      '3', '00:00:06,000 --> 00:00:07,000', '♪～ 歌詞', '',
                                      '4', '00:00:07,000 --> 00:00:08,000', '《心の声》(ため息)ﾌﾌ… ｴﾛいこと'])
        cues = srt.parse_srt(raw)
        self.assertEqual([item['n'] for item in cues], [1, 2, 4])
        self.assertEqual(cues[0]['lines'], [{'text': 'これは', 'tag': '理央'}, {'text': '思春期症候群ではなく', 'tag': None}])
        self.assertEqual(cues[1]['lines'], [{'text': 'もしもし？', 'tag': '南条'}])
        self.assertEqual(cues[2]['lines'][0]['text'], '心の声ﾌﾌ… ｴﾛいこと')
        self.assertEqual((cues[0]['start'], cues[0]['end']), (1.97, 4.7))
        self.assertEqual(''.join(srt.caption_content(cues[2]['lines'][0]['text'])), '心の声フフエロいこと')
        self.assertEqual(''.join(srt.caption_content('１時間 毒ｶﾞｽ｡')), '1時間毒ガス')


class OffsetTests(unittest.TestCase):
    def test_two_blocks_get_their_own_offsets(self):
        texts = ['これは思春期症候群ではない', 'あくまで量子力学の解釈だよ', 'この中に猫を入れておく', 'さてここに箱を用意した',
                 '料理番組かと思った', '何か言ったのかしら', '大丈夫よどうせ見えない', 'じゃあ僕が触れたら', 'それは私と手をつなぎたい']
        lines, cues = [], []
        for index, text in enumerate(texts):
            start = 2.0 + 4 * index + (90.0 if index >= 4 else 0.0)
            lines.append(line(f's{index}', start, start + 3, text))
            cues.append(cue(index + 1, 2.0 + 4 * index - 0.3, 2.0 + 4 * index + 2.5, text))
        runs, keyed, matched = srt.estimate_offsets(lines, srt.parse_srt('\n'.join(cues)))
        self.assertEqual((keyed, matched), (9, 9))
        self.assertEqual([(run['first'], run['last'], run['offset']) for run in runs], [(0, 3, 0.3), (4, 8, 90.3)])
        offsets = srt.cue_offsets(srt.parse_srt('\n'.join(cues)), runs)
        self.assertEqual(offsets[3:5], [0.3, 90.3])


class PlanTests(unittest.TestCase):
    def test_mishearing_is_safe_and_keeps_the_lines_punctuation(self):
        items = plan([line('s1', 1.0, 7.0, 'これは青春気象校軍ではなく、あくまで量子力学の世界における解釈だよ。')],
                     cue(1, 1.0, 4.0, '(理央)これは', '思春期症候群ではなく'), cue(2, 4.0, 7.0, 'あくまで量子力学の世界における', '解釈だよ｡'))
        [item] = text_items(items)
        self.assertEqual((item['class'], item['accepted']), ('safe', True))
        self.assertEqual(item['after'], 'これは思春期症候群ではなく、あくまで量子力学の世界における解釈だよ。')
        self.assertEqual(item['cues'], [1, 2])

    def test_spelling_variant_and_its_safe_part(self):
        items = plan([line('s1', 1.0, 4.0, 'これで蓋をして3 0分待つ。'), line('s2', 6.0, 9.0, 'ほらね、多分私が触れたものは見えなくなるのよ。')],
                     cue(1, 1.0, 4.0, 'これでﾌﾀをして 30分待つ｡'), cue(2, 6.0, 9.0, 'ほらね たぶん私が触れたものは', '見えなくなるのよ｡'))
        first, second = text_items(items)
        self.assertEqual((first['class'], first['after'], first['after_safe']), ('variant', 'これでフタをして30分待つ。', 'これで蓋をして30分待つ。'))
        self.assertEqual((second['class'], second['after_safe'], second['accepted']), ('variant', None, None))
        self.assertEqual(srt.text_to_send(first, 'decided'), 'これで蓋をして30分待つ。')
        self.assertIsNone(srt.text_to_send(second, 'decided'))
        second['accepted'] = True
        self.assertEqual(srt.text_to_send(second, 'decided'), 'ほらね、たぶん私が触れたものは見えなくなるのよ。')
        self.assertIsNone(srt.text_to_send(second, 'safe'))

    def test_a_spoken_particle_is_kept_and_kana_spelling_is_not_a_particle(self):
        items = plan([line('s1', 1.0, 4.0, 'あの人は私信気象航空軍に興味を持っている。'), line('s2', 6.0, 9.0, 'じゃあ2時半に江の田富士沢駅の改札前。')],
                     cue(1, 1.0, 4.0, 'あの人 思春期症候群に', '興味を持っている｡'), cue(2, 6.0, 9.0, 'じゃあ 2時半に', '江ノ電藤沢駅の改札前｡'))
        self.assertEqual([item['after'] for item in text_items(items)],
                         ['あの人は思春期症候群に興味を持っている。', 'じゃあ2時半に江ノ電藤沢駅の改札前。'])

    def test_words_the_captions_drop_are_kept(self):
        items = plan([line('s1', 1.0, 4.0, 'えっと、それはね、本当だよ。')], cue(1, 1.0, 4.0, 'それはね 本当だよ'))
        self.assertEqual(text_items(items), [])
        self.assertTrue(any('えっと' in item['reason'] for item in items if item['kind'] == 'info'))

    def test_a_name_from_the_glossary_is_safe(self):
        [item] = text_items(plan([line('s1', 1.0, 4.0, 'まゆさん女王様すぎ。')], cue(1, 1.0, 4.0, '麻衣さん 女王様すぎ｡')))
        self.assertEqual((item['class'], item['after']), ('safe', '麻衣さん女王様すぎ。'))

    def test_speaker_only_where_the_tag_covers_the_line(self):
        lines = [line('s1', 30.0, 31.2, '料理番組か。'), line('s2', 31.6, 33.9, '箱の中の猫はどうなってると思う？')]
        items = plan(lines, cue(9, 29.9, 32.7, '(咲太)料理番組か｡', '箱の中の猫は'), cue(10, 32.7, 34.0, 'どうなってると思う？'))
        speakers = [item for item in items if item['kind'] == 'speaker']
        self.assertEqual([(item['segment_id'], item['character_context_id'], item['class']) for item in speakers], [('s1', 'sakuta', 'review')])

    def test_speaker_is_left_alone_when_set_by_hand_unassigned_or_ambiguous(self):
        for character, source, tag in (('rio', 'manual', '咲太'), (None, 'voice', '咲太'), ('rio', 'voice', '梓川')):
            items = plan([line('s1', 30.0, 31.2, '料理番組か。', character, source)], cue(9, 29.9, 31.3, f'({tag})料理番組か｡'))
            self.assertEqual([item for item in items if item['kind'] == 'speaker'], [], (character, source, tag))
        notes = [item['reason'] for item in plan([line('s1', 30.0, 31.2, '料理番組か。', 'rio', 'manual')], cue(9, 29.9, 31.3, '(咲太)料理番組か｡'))]
        self.assertTrue(any('by hand' in note for note in notes))

    def test_captions_without_a_transcript_line_are_a_note(self):
        items = plan([line('s1', 1.0, 3.0, '今日はいい天気だ。')], cue(1, 1.0, 3.0, '今日はいい天気だ｡'), cue(2, 20.0, 22.0, 'どこにも書き起こされていない台詞'))
        self.assertEqual(text_items(items), [])
        self.assertTrue(any('no transcript line' in item['reason'] for item in items))


class ApplyTests(unittest.TestCase):
    class FakeApi:
        def __init__(self, missing=False):
            self.url, self.bodies, self.missing = 'http://server', [], missing

        def get(self, path, params=None):
            assert path.endswith('/cuts'), path
            return {'items': [{'id': 'cut-a', 'cut_index': 0, 'start_time': 0.0, 'end_time': 20.0},
                              {'id': 'cut-b', 'cut_index': 1, 'start_time': 20.0, 'end_time': 40.0}], 'next_offset': None}

        def post(self, path, params=None, body=None):
            if self.missing:
                raise srt.Failure(f'POST {path}: HTTP 404: Not Found')
            self.bodies.append(body)
            return {'applied': [{'segment_id': item['segment_id'], 'id': 'new-' + item['segment_id']} for item in body['corrections'][:1]],
                    'skipped': [{'segment_id': item['segment_id'], 'reason': 'the line was edited or deleted since the plan was made'}
                                for item in body['corrections'][1:]]}

    def write_plan(self, items):
        import json
        import tempfile
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'plan.json'
        path.write_text(json.dumps({'episode': {'id': 'e'}, 'srt': {'file': 'ep.srt'}, 'offsets': [{'offset': 0.3}], 'items': items}))
        return path

    def args(self, path, **extra):
        import argparse
        return argparse.Namespace(plan=str(path), ids=None, safe=False, skip='', dry_run=False, url=None, **extra)

    def test_sends_accepted_lines_whole_and_safe_parts_and_records_the_outcome(self):
        import json
        items = plan([line('s1', 1.0, 4.0, 'これで蓋をして3 0分待つ。'), line('s2', 30.0, 31.2, '料理番組か。')],
                     cue(1, 1.0, 4.0, 'これでﾌﾀをして 30分待つ｡'), cue(9, 29.9, 31.3, '(咲太)料理番組か｡'))
        speaker = next(item for item in items if item['kind'] == 'speaker')
        speaker['accepted'] = True
        path = self.write_plan(items)
        api = self.FakeApi()
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(srt.cmd_apply(api, self.args(path)), srt.OK)
        # the skipped speaker change at 30 s is listed with the link of the cut that holds it
        self.assertIn('http://server/#/episode/e/cut/cut-b/analysis', out.getvalue())
        [body] = api.bodies
        self.assertEqual(body['source'], {'kind': 'srt', 'file': 'ep.srt', 'offset': 0.3})
        sent = {item['segment_id']: item for item in body['corrections']}
        self.assertEqual(sent['s1']['text'], 'これで蓋をして30分待つ。')
        self.assertEqual((sent['s2']['character_context_id'], 'text' in sent['s2']), ('sakuta', False))
        saved = json.loads(path.read_text())['items']
        self.assertEqual([item.get('applied') for item in saved if item['kind'] == 'text'], ['new-s1'])
        self.assertIn('since the plan', next(item for item in saved if item['kind'] == 'speaker')['skipped'])
        api.bodies.clear()
        srt.cmd_apply(api, self.args(path))
        self.assertEqual(api.bodies, [], 'applied and skipped proposals are not sent twice')

    def test_an_old_server_is_named(self):
        path = self.write_plan(plan([line('s1', 1.0, 4.0, 'まゆさん女王様すぎ。')], cue(1, 1.0, 4.0, '麻衣さん 女王様すぎ｡')))
        with self.assertRaisesRegex(srt.Failure, 'rebuild video-analyzer'):
            srt.cmd_apply(self.FakeApi(missing=True), self.args(path))


class IdsTests(unittest.TestCase):
    def test_parse_ids(self):
        self.assertEqual(srt.parse_ids('3, 7,12-14'), {3, 7, 12, 13, 14})
        with self.assertRaises(srt.Failure):
            srt.parse_ids('a-b')


if __name__ == '__main__':
    unittest.main()
