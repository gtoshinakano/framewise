"""MiniMax H3 prompt format checks used by Cut Assistance."""
import json
import unittest

import tests.test_assistant as assistant_tests
from app import db
from app.assistant import h3_prompt, tools

EXAMPLE = """subject_definitions:
<Picture 1> is the first frame of [Shot 1]: Hermann von Salza standing behind a wooden table in the commandery hall, a map spread in front of him.
<Subject 1> is Hermann von Salza, the lean man whose face, hair and clothing come from <Picture 2>: short dark-brown beard, white wool mantle with a black cross over grey mail.
<Subject 2> is the commandery hall from <Picture 3>: whitewashed stone walls, a heavy oak table, one tall arched window.

summary:
[keyframe completion + reference generation] The shot continues from <Picture 1>: <Subject 1> traces a river on the map in <Subject 2>, presses his hand flat on the parchment and lifts his eyes toward the window, while the camera pushes in slowly from a medium shot to a medium close-up.

retention_analysis:
<Picture 1> ([Shot 1] first frame): fully_preserved - composition, palette and lighting of the opening frame.
<Subject 1> (appears in [Shot 1]): fully_preserved - face, beard, mantle and mail follow <Picture 2> exactly, with no change of clothing.
<Subject 2> (appears in [Shot 1]): fully_preserved - walls, oak table and arched window follow <Picture 3>.

detailed_description:
[Shot 1] The shot begins from <Picture 1>: a medium shot at eye level, <Subject 1> seen from the waist up behind the oak table on the left of the frame, the tall window on the right. Head to waist, never wider. Cold morning light falls through the window from the right and leaves the far wall in soft shadow; dust hangs in the beam.
00:00.000-00:02.000: <Subject 1> looks down at the map and slowly traces a river line from top to bottom with his right index finger. He does not speak; his lips stay closed. The camera pushes in with small amplitude at slow speed.
00:02.000-00:04.000: His finger stops. He presses his hand flat on the parchment, which creases under his palm, and his shoulders settle as he exhales through his nose.
00:04.000-00:06.000: He lifts his eyes from the map toward the window, and the light catches the side of his face. The push-in ends on a medium close-up of his face and the hand on the map; he holds still, watching, while dust keeps drifting through the beam.

overall_soundscape:
Quiet stone-hall room tone with faint wind against the window. A soft scrape of a fingertip on parchment, the dry crinkle of the map under his hand and one slow exhale.

non_diegetic_music:
N/A"""

PREAMBLE = '<Picture 1> is the first frame of this shot: continue exactly from it.\n<Picture 2> is the outfit sheet reference of Hermann von Salza (Mantle).\n<Audio 1> is the reference audio “cut-0154-original.wav”.'


def errors(prompt, original=EXAMPLE):
    return h3_prompt.check(prompt, original)['errors']


class H3FormatTest(unittest.TestCase):
    def test_director_example_passes_and_reports_stats(self):
        result = h3_prompt.check(EXAMPLE, EXAMPLE)
        self.assertTrue(result['ok'], result['errors'])
        self.assertEqual(result['stats']['mode'], 'omni_reference')
        self.assertEqual(result['stats']['seconds'], 6.0)
        self.assertEqual(result['stats']['pictures'], [1, 2, 3])
        self.assertTrue(h3_prompt.check(EXAMPLE)['ok'])
        fenced = h3_prompt.check('```text\n' + EXAMPLE + '\n```', EXAMPLE)
        self.assertTrue(fenced['ok'], fenced['errors'])

    def test_references_must_keep_their_numbers(self):
        renumbered = EXAMPLE.replace('<Picture 3>', '<Picture 4>')
        found = ' '.join(errors(renumbered))
        self.assertIn('<Picture 4> not in the original', found)
        self.assertIn('<Picture 3> from the original must still be defined', found)
        with_audio = EXAMPLE.replace('Quiet stone-hall', '<Audio 1> sets the ambience. Quiet stone-hall')
        self.assertIn('<Audio 1> not in the original', ' '.join(errors(with_audio)))
        for bad in ('Picture2', '@Image 1', '[image 1]', '<picture 2>', '<Picture2>'):
            self.assertTrue(any('Malformed' in e for e in errors(EXAMPLE.replace('<Picture 2>', bad, 1))), bad)

    def test_structure_rules(self):
        self.assertTrue(any('Missing sections' in e for e in errors(EXAMPLE.replace('overall_soundscape:', 'soundscape:'))))
        swapped = EXAMPLE.replace('summary:\n', 'TMP:\n').replace('retention_analysis:\n', 'summary:\n').replace('TMP:\n', 'retention_analysis:\n')
        self.assertTrue(any('order' in e for e in errors(swapped)))
        self.assertTrue(any('nothing before' in e for e in errors('Here is the fix:\n' + EXAMPLE, None)))
        cleaned = h3_prompt.check('Here is the fix:\n' + EXAMPLE, EXAMPLE)
        self.assertEqual((cleaned['ok'], cleaned['prompt']), (True, EXAMPLE))
        self.assertTrue(any('"cut"' in e for e in errors(EXAMPLE.replace('Head to waist', 'No cut. Head to waist'))))
        named = EXAMPLE.replace('Quiet stone-hall', 'The hall sound follows “cut-0154 room tone.wav”. Quiet stone-hall')
        self.assertFalse(any('"cut"' in e for e in errors(named, named)))
        self.assertTrue(any('---' in e for e in errors(EXAMPLE.replace('overall_soundscape:', '---\noverall_soundscape:'))))
        self.assertTrue(any('summary: must start' in e for e in errors(EXAMPLE.replace('[keyframe completion + reference generation] ', ''))))
        self.assertTrue(any('[Shot 1]' in e for e in errors(EXAMPLE.replace('[Shot 1] The shot begins', 'The shot begins'))))
        self.assertTrue(any('non_diegetic_music' in e for e in errors(EXAMPLE.replace('non_diegetic_music:\nN/A', 'non_diegetic_music:\nSoft strings.'))))

    def test_a_style_or_medium_word_is_refused_even_negated_but_names_and_acting_words_pass(self):
        for sentence in ('The target video is a 2D hand-drawn illustration with dark ink outlines.', 'Keep the anime look.', 'It is not cartoonish and not cel-shaded.',
                         'The animation holds the framing.'):
            found = ' '.join(errors(EXAMPLE.replace('[Shot 1] The shot begins', sentence + '\n[Shot 1] The shot begins')))
            self.assertIn('names a visual style or medium', found, sentence)
        self.assertIn('(2d, hand-drawn, illustration)', ' '.join(errors(EXAMPLE.replace('[Shot 1] The shot begins', 'A 2D hand-drawn illustration.\n[Shot 1] The shot begins'))))
        acting = EXAMPLE.replace('his shoulders settle', 'his brows drawn together, his shoulders settle').replace('whitewashed stone walls', 'painted stone walls')
        self.assertTrue(h3_prompt.check(acting, EXAMPLE)['ok'], h3_prompt.check(acting, EXAMPLE)['errors'])
        named = EXAMPLE.replace('Hermann von Salza', 'Anime Club President')  # a record name is not the prompt's wording
        self.assertTrue(h3_prompt.check(named)['ok'], h3_prompt.check(named)['errors'])
        quoted = EXAMPLE.replace('Quiet stone-hall', 'The hall sound follows “anime-room.wav”. Quiet stone-hall')
        self.assertFalse(any('style' in e for e in errors(quoted, quoted)))
        self.assertEqual(h3_prompt.style_words('Image texture: soft watercolour wash, fine grain'), [])
        self.assertEqual(h3_prompt.style_words('Image texture: soft watercolour wash, fine grain', camera=True), ['watercolour'])

    def test_beats_are_contiguous_and_within_fifteen_seconds(self):
        gap = EXAMPLE.replace('00:02.000-00:04.000', '00:02.500-00:04.000')
        self.assertTrue(any('contiguous' in e for e in errors(gap)))
        long = EXAMPLE.replace('00:04.000-00:06.000', '00:04.000-00:16.000')
        self.assertTrue(any('at most 15s' in e for e in errors(long)))
        late = EXAMPLE.replace('00:00.000-00:02.000', '00:00.500-00:02.000')
        self.assertTrue(any('00:00.000' in e for e in errors(late)))
        retimed = EXAMPLE.replace('00:04.000-00:06.000', '00:04.000-00:08.000')
        self.assertTrue(any('must end at 00:06.000' in e for e in errors(retimed)))
        self.assertTrue(h3_prompt.check(retimed, None)['ok'])

    def test_timed_lines_sit_inside_their_beat_and_audio_is_used(self):
        glued = EXAMPLE.replace('\n\noverall_soundscape:', '\nAt 00:00.500, <Subject 1> (S1) says: <d>[Japanese] ええ。</d>\n\noverall_soundscape:')
        result = h3_prompt.check(glued, EXAMPLE)
        self.assertTrue(result['ok'], result['errors'])
        self.assertTrue(any('consider moving it' in w for w in result['warnings']))
        # The director's own style (a timed voice-over line after the beats) is kept without nagging.
        self.assertFalse(any('consider moving it' in w for w in h3_prompt.check(glued, glued)['warnings']))
        inside = EXAMPLE.replace('He does not speak; his lips stay closed.', 'At 00:00.500, <Subject 1> (S1) says: <d>[Japanese] ええ。</d>')
        self.assertTrue(h3_prompt.check(inside, EXAMPLE)['ok'])
        original = PREAMBLE + '\n' + EXAMPLE.replace('<Subject 2> is the commandery', '<Audio 1> is the reference audio of the original shot.\n<Subject 2> is the commandery')
        unused = h3_prompt.check(original, original)
        self.assertTrue(any('<Audio 1> is defined but never used' in w for w in unused['warnings']))
        used = original.replace('Quiet stone-hall', '<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:00.500. Quiet stone-hall')
        self.assertFalse(any('never used' in w for w in h3_prompt.check(used, original)['warnings']))
        self.assertEqual(h3_prompt.speaking_subjects(used), {1})
        self.assertEqual(h3_prompt.subject_names(EXAMPLE), {1: 'Hermann von Salza', 2: 'commandery hall'})
        self.assertTrue(any('N/A' in e for e in errors(EXAMPLE.replace('non_diegetic_music:\nN/A', 'non_diegetic_music:\nSoft strings.'))))
        self.assertTrue(h3_prompt.check(EXAMPLE.replace('non_diegetic_music:\nN/A', 'non_diegetic_music:\nN/A.'), EXAMPLE)['ok'])

    def test_thin_descriptions_fail(self):
        start = EXAMPLE.index('00:00.000-00:02.000')
        thin = EXAMPLE[:start] + '00:00.000-00:06.000: He holds still. The camera holds a static shot.\n\noverall_soundscape:' + EXAMPLE.split('overall_soundscape:')[1]
        self.assertTrue(any('words' in e for e in errors(thin)))

    def test_dialogue_format(self):
        good = EXAMPLE.replace('He does not speak; his lips stay closed.', 'At 00:01.000, <Subject 1> (S1) says: <d>[Japanese] ええ。</d>')
        self.assertTrue(h3_prompt.check(good, EXAMPLE)['ok'], h3_prompt.check(good, EXAMPLE)['errors'])
        self.assertTrue(any('<d>[Language]' in e for e in errors(good.replace('<d>[Japanese] ', '<d>'))))
        self.assertTrue(any('Unbalanced' in e for e in errors(good.replace('</d>', ''))))

    def test_preamble_is_kept_verbatim_and_modes_are_respected(self):
        original = PREAMBLE + '\n' + EXAMPLE.replace('Quiet stone-hall', '<Audio 1> sets the ambience. Quiet stone-hall')
        self.assertTrue(h3_prompt.check(original, original)['ok'], h3_prompt.check(original, original)['errors'])
        # The app's lines are restored verbatim whether the model omits, retypes or alters them.
        body = original.split('\n', 3)[3]
        for draft in (body, original.replace('“cut-0154-original.wav”', '"cut-0154-original.wav"'), original.replace('(Mantle)', '(Cloak)')):
            result = h3_prompt.check(draft, original)
            self.assertTrue(result['ok'], result['errors'])
            self.assertEqual(result['prompt'], original)
        renamed = original.replace('Hermann von Salza, the lean man', 'Hermann, the lean man')
        self.assertTrue(any('keep record names exactly' in e for e in errors(renamed, original)))
        audio = EXAMPLE.replace('<Subject 2> is the commandery', '<Audio 1> is the reference audio “cut-0085-5f6a917f”, used as the voiceover.\n<Subject 2> is the commandery').replace('Quiet stone-hall', 'An off-screen voice speaks the words of <Audio 1>. Quiet stone-hall')
        self.assertTrue(h3_prompt.check(audio.replace('“cut-0085-5f6a917f”', '"cut-0085-5f6a917f"'), audio)['ok'])
        self.assertTrue(any('Quoted reference names' in e for e in errors(audio.replace('cut-0085-5f6a917f', 'cut-85'), audio)))
        named = EXAMPLE.replace('Quiet stone-hall', 'The hall tone follows “hall-tone.wav”. Quiet stone-hall')
        self.assertTrue(any('File names' in e and 'hall-tone.wav' in e for e in errors(EXAMPLE, named)))
        frames_only = EXAMPLE.replace('<Picture 1>', 'the first frame').replace('<Picture 2>', 'the sheet').replace('<Picture 3>', 'the hall')
        self.assertEqual(h3_prompt.mode(frames_only), 'first_last_frame')
        self.assertTrue(any('<Picture 1>' in e and 'not in the original' in e for e in errors(EXAMPLE, frames_only)))

    def test_extract_from_chat_message(self):
        message = 'Fix this H3 prompt based on results:\n\n```\n' + PREAMBLE + '\n' + EXAMPLE + '\n```'
        self.assertEqual(h3_prompt.extract(message), PREAMBLE + '\n' + EXAMPLE)
        self.assertIsNone(h3_prompt.extract('Fix the H3 prompt please'))


class H3ToolTest(assistant_tests.AssistantBase):
    def test_check_tool_compares_with_the_pasted_prompt_and_creates_a_prompt_card(self):
        session_id = self.session()
        self.ctx.session_id = session_id
        self.worker_online()
        turn = self.client.post(f'/api/assistant/sessions/{session_id}/messages', json={'content': 'Fix this H3 prompt based on results:\n\n' + EXAMPLE, 'provider': 'fake', 'model': 'fake-model'}).json()['turn']
        self.ctx.turn_id = turn['id']
        broken, label, ok = tools.execute(self.ctx, 'check_h3_prompt', json.dumps({'prompt': EXAMPLE.replace('<Picture 3>', '<Picture 5>')}))
        broken = json.loads(broken)
        self.assertFalse(broken['ok'])
        self.assertEqual(broken['compared_with'], 'your prompt')
        self.assertIn('error', label)
        self.assertIsNone(db.one('SELECT id FROM assistant_proposals'))
        fixed = EXAMPLE.replace('traces a river line', 'traces the river line')
        first = json.loads(tools.execute(self.ctx, 'check_h3_prompt', json.dumps({'prompt': fixed, 'summary': 'Article fix'}))[0])
        self.assertTrue(first['ok'], first)
        second = json.loads(tools.execute(self.ctx, 'check_h3_prompt', json.dumps({'prompt': fixed.replace('slowly', 'gently'), 'summary': 'Second draft'}))[0])
        detail = self.client.get(f'/api/assistant/sessions/{session_id}').json()
        cards = {proposal['id']: proposal for proposal in detail['proposals']}
        self.assertEqual(cards[first['proposal_id']]['status'], 'discarded')
        card = cards[second['proposal_id']]
        self.assertEqual((card['status'], card['target'], card['diff_label']), ('pending', 'h3_prompt', 'your prompt'))
        self.assertEqual({segment['text'] for segment in card['diff'] if segment['op'] != 'equal'}, {'a', 'the', 'slowly', 'gently'})
        speaking = fixed.replace('He does not speak; his lips stay closed.', 'At 00:01.000, <Subject 1> (S1) says: <d>[Japanese] こんにちは</d>')
        unlinked = json.loads(tools.execute(self.ctx, 'check_h3_prompt', json.dumps({'prompt': speaking}))[0])
        self.assertTrue(any('Hermann von Salza' in w and 'linked to Hero' in w for w in unlinked['warnings']), unlinked['warnings'])
        wrong_words = fixed.replace('He does not speak; his lips stay closed.', 'At 00:01.000, <Subject 1> (S1) says: <d>[Japanese] こんばんは</d>')
        rejected = json.loads(tools.execute(self.ctx, 'check_h3_prompt', json.dumps({'prompt': wrong_words}))[0])
        self.assertFalse(rejected['ok'])
        self.assertTrue(any('does not match the transcript' in e and 'こんにちは' in e for e in rejected['errors']))
        refused, _, ok = tools.execute(self.ctx, 'propose_result_edit', json.dumps({'type': 'h3_prompt', 'content': fixed, 'summary': 'again'}))
        self.assertIn('check_h3_prompt', json.loads(refused)['error'])
        saved = self.client.post(f"/api/assistant/proposals/{unlinked['proposal_id']}/apply").json()
        row = db.one('SELECT title,kind,origin FROM analysis_results WHERE id=?', (saved['applied_result_id'],))
        self.assertEqual(dict(row), {'title': 'H3 animation prompt', 'kind': 'generation_prompt', 'origin': 'assistant_edit'})


if __name__ == '__main__':
    unittest.main()


class TranscriptMergeTest(unittest.TestCase):
    def test_word_fragments_merge_into_speaker_lines(self):
        row = lambda start, end, text, speaker='S1': {'start_time': start, 'end_time': end, 'text': text, 'speaker_label': speaker, 'character_name': 'Mai', 'origin': 'model'}
        lines = tools.merged_transcript([row(0.43, 0.67, 'もう'), row(0.67, 1.07, '契約'), row(1.07, 1.15, 'し'), row(2.67, 2.91, 'やる'), row(2.99, 3.15, 'しか'),
                                         row(4.0, 4.2, 'Hello'), row(4.2, 4.4, 'there.'), row(4.5, 4.7, 'Hi', 'S2')])
        self.assertEqual([line['text'] for line in lines], ['もう契約し', 'やるしか', 'Hello there.', 'Hi'])
        self.assertEqual((lines[0]['start_time'], lines[0]['end_time']), (0.43, 1.15))
        self.assertEqual(tools.spoken_key('もう契約した。 Hello, there!'), 'もう契約したHellothere')
