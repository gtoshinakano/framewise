"""Re-listening to a cut's audio: the second-model reading and the Cut Assistance tool."""
import json
import os
import unittest
from unittest import mock

from app import db, media
from app.assistant import audio_transcription, tools
import tests.test_analysis as analysis_tests


# What the audio model returns; the handler turns these into the tool's numbered segments.
SEGMENTS = [
    {'start': 0.0, 'end': 1.0, 'speaker': 'S1', 'text': 'hello'},
    {'start': 1.0, 'end': 2.0, 'speaker': 'S2', 'text': 'world'},
]


class ModuleTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(audio_transcription.DEFAULT_AUDIO_MODEL, 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free')
        self.assertEqual(audio_transcription.MAX_SECONDS, 60.0)
        self.assertEqual(audio_transcription.MIN_SECONDS, 0.5)

    def test_answer_segments_clamps_and_keeps_order(self):
        text = '{"segments":[{"end":9.0,"start":2.1,"speaker":"S2","text":" second "},{"start":0.0,"end":1.2,"speaker":"S1","text":"first"},{"start":1.5,"text":"no end"},{"text":"  "}]}'
        segments = audio_transcription.answer_segments(text, 4.0)
        self.assertEqual([s['text'] for s in segments], ['second', 'first', 'no end'])  # blank text dropped, order kept
        self.assertEqual(segments[0]['end'], 4.0)  # clamped to the clip
        self.assertEqual(segments[2]['end'], None)  # no usable number
        self.assertEqual(segments[2]['start'], 1.5)

    def test_answer_segments_fixes_reversed_times(self):
        segments = audio_transcription.answer_segments('{"segments":[{"start":2.0,"end":1.0,"text":"x"}]}', 4.0)
        self.assertEqual(segments[0]['end'], 2.1)

    def test_answer_segments_strips_code_fence(self):
        segments = audio_transcription.answer_segments('```json\n{"segments":[{"start":0,"end":1,"speaker":"S1","text":"hi"}]}\n```', 2.0)
        self.assertEqual(segments[0]['text'], 'hi')

    def test_answer_segments_rejects_unreadable(self):
        for bad in ('no json here', '{"segments": "not a list"}'):
            with self.assertRaises(audio_transcription.TranscriptionError):
                audio_transcription.answer_segments(bad, 2.0)

    def test_time_clamps_and_rejects_bool(self):
        self.assertIsNone(audio_transcription._time(True, 5.0))
        self.assertIsNone(audio_transcription._time('x', 5.0))
        self.assertEqual(audio_transcription._time(10, 5.0), 5.0)


class SourceTest(analysis_tests.AnalysisTest):
    def test_settings_model_default_and_choice(self):
        self.assertEqual(audio_transcription.settings_model(), audio_transcription.DEFAULT_AUDIO_MODEL)
        with db.connect() as conn:
            conn.execute('UPDATE text_generation_settings SET openrouter_audio_model=? WHERE id=1', ('a/b:free',))
        self.assertEqual(audio_transcription.settings_model(), 'a/b:free')

    def test_audio_source_prefers_prepared_then_original_then_none(self):
        path = media.episode_dir('p', 'e')
        self.assertIsNone(audio_transcription.audio_source('p', 'e', '.mp4'))
        (path / 'original.mp4').write_bytes(b'v')
        self.assertEqual(audio_transcription.audio_source('p', 'e', '.mp4'), path / 'original.mp4')
        (path / 'audio.wav').write_bytes(b'a')
        self.assertEqual(audio_transcription.audio_source('p', 'e', '.mp4'), path / 'audio.wav')


class TranscribeTest(analysis_tests.AnalysisTest):
    def test_without_key_fails(self):
        env = {key: value for key, value in os.environ.items() if key != 'OPENROUTER_API_KEY'}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(audio_transcription.TranscriptionError) as caught:
                audio_transcription.transcribe('m', media.episode_dir('p', 'e') / 'clip.wav', 1.0)
        self.assertIn('OPENROUTER_API_KEY', str(caught.exception))

    def test_sends_audio_named_and_returns_segments(self):
        clip = media.episode_dir('p', 'e') / 'clip.wav'
        clip.write_bytes(b'RIFFfakeaudio')
        sent = {}

        def fake_answer(payload, headers):
            sent['payload'] = payload
            sent['headers'] = headers
            return json.dumps({'segments': SEGMENTS})

        with mock.patch.dict(os.environ, {**os.environ, 'OPENROUTER_API_KEY': 'sk-test'}):
            with mock.patch('app.assistant.audio_transcription._answer', side_effect=fake_answer):
                segments, speakers = audio_transcription.transcribe('a/b:free', clip, 2.0)

        self.assertEqual([s['text'] for s in segments], ['hello', 'world'])
        self.assertEqual(speakers, ['S1', 'S2'])
        # The audio goes as a named input_audio part right after the instruction.
        user = sent['payload']['messages'][1]['content']
        self.assertEqual(user[0]['type'], 'text')
        self.assertEqual(user[1]['type'], 'input_audio')
        self.assertEqual(user[1]['input_audio']['format'], 'wav')
        self.assertEqual(sent['headers']['Authorization'], 'Bearer sk-test')


class ToolTest(analysis_tests.AnalysisTest):
    def tool(self):
        spec = [item for item in tools.schemas('cut') if item['function']['name'] == 'transcribe_cut_audio']
        self.assertTrue(spec)
        self.assertEqual(sorted(spec[0]['function']['parameters']['properties']), ['end', 'start'])
        self.assertEqual(spec[0]['function']['parameters']['required'], [])
        # It serves a cut chat only (scope 'cut'): fix_transcript is a cut skill.
        self.assertFalse(any(item['function']['name'] == 'transcribe_cut_audio' for item in tools.schemas('sequence')))

    def test_no_audio_file(self):
        with self.assertRaises(tools.ToolError) as caught:
            tools.transcribe_cut_audio(tools.ToolContext(cut_id='c1'), {})
        self.assertIn('no audio', str(caught.exception))

    def test_window_too_short(self):
        media.episode_dir('p', 'e') / 'audio.wav'
        (media.episode_dir('p', 'e') / 'audio.wav').write_bytes(b'RIFF')
        with self.assertRaises(tools.ToolError) as caught:
            tools.transcribe_cut_audio(tools.ToolContext(cut_id='c1'), {'start': 0.0, 'end': 0.2})
        self.assertIn('too short', str(caught.exception))

    def test_non_number_start(self):
        with self.assertRaises(tools.ToolError):
            tools.transcribe_cut_audio(tools.ToolContext(cut_id='c1'), {'start': 'x'})

    def test_execute_reports_error_to_the_model(self):
        content, label, ok = tools.execute(tools.ToolContext(cut_id='c1'), 'transcribe_cut_audio', {})
        self.assertFalse(ok)
        self.assertIn('no audio', content.lower())

    def test_end_to_end(self):
        (media.episode_dir('p', 'e') / 'audio.wav').write_bytes(b'RIFF')
        calls = {}

        def fake_clip(source, target, start, end):
            calls['clip'] = (str(source), start, end)
            return target

        def fake_transcribe(model, clip_path, seconds):
            calls['transcribe'] = (model, seconds)
            return SEGMENTS, ['S1', 'S2']

        with mock.patch('app.assistant.audio_transcription.clip', side_effect=fake_clip):
            with mock.patch('app.assistant.audio_transcription.transcribe', side_effect=fake_transcribe):
                with mock.patch('app.assistant.audio_transcription.settings_model', return_value='a/b:free'):
                    out = tools.transcribe_cut_audio(tools.ToolContext(cut_id='c1'), {})
        self.assertEqual(out['note'], tools.LISTEN_NOTE)
        self.assertIn('propose_transcript_edit', out['note'])

        self.assertEqual(out['model'], 'a/b:free')
        self.assertEqual(out['seconds'], 2.0)
        self.assertEqual(out['voices'], ['S1', 'S2'])
        self.assertEqual([s['text'] for s in out['segments']], ['hello', 'world'])
        self.assertEqual([s['i'] for s in out['segments']], [1, 2])
        self.assertEqual(out['segments'][0]['t'], tools.span(0.0, 1.0))
        # The clip is requested in absolute episode seconds for the whole cut.
        self.assertEqual(calls['clip'][1], 0.0)
        self.assertEqual(calls['clip'][2], 2.0)
        self.assertEqual(calls['transcribe'][1], 2.0)

    def test_transcription_error_becomes_tool_error(self):
        (media.episode_dir('p', 'e') / 'audio.wav').write_bytes(b'RIFF')
        with mock.patch('app.assistant.audio_transcription.clip', return_value=media.episode_dir('p', 'e') / 'cut.wav'):
            with mock.patch('app.assistant.audio_transcription.transcribe', side_effect=audio_transcription.TranscriptionError('OpenRouter refused the audio (HTTP 400)')):
                with mock.patch('app.assistant.audio_transcription.settings_model', return_value='a/b:free'):
                    with self.assertRaises(tools.ToolError) as caught:
                        tools.transcribe_cut_audio(tools.ToolContext(cut_id='c1'), {})
        self.assertIn('HTTP 400', str(caught.exception))
