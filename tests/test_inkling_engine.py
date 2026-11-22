"""Inkling 2 Mini as a third local engine: its own server build, flags, reasoning switch and the resident-model rule."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tests.test_analysis as analysis_tests
from app import db, frame_model, local_text_runtime, visual_models

INKLING = visual_models.MODELS['inkling2-mini']
QWEN36 = visual_models.MODELS['qwen36-35b']


def select(variant):
    with db.connect() as conn:
        conn.execute('UPDATE text_generation_settings SET local_model=? WHERE id=1', (variant,))


class VariantRegistryTest(unittest.TestCase):
    def test_every_variant_names_its_server_model_projector_and_family(self):
        for name, info in local_text_runtime.VARIANTS.items():
            for key in ('label', 'id', 'revision', 'size', 'family', 'name', 'server', 'model', 'projector', 'install'):
                self.assertIn(key, info, f'{name} lacks {key}')
        self.assertEqual(local_text_runtime.family('inkling'), 'inkling')
        self.assertEqual(local_text_runtime.family('mtp'), 'qwen')
        self.assertEqual(local_text_runtime.family('remote:other'), 'remote')

    def test_inkling_has_its_own_llama_server_and_files(self):
        self.assertNotEqual(local_text_runtime.server_path('inkling'), local_text_runtime.server_path('mtp'))
        self.assertEqual(local_text_runtime.server_path('mtp'), local_text_runtime.server_path('standard'))
        with patch.dict('os.environ', {'INKLING_SERVER_PATH': '/x/llama-server', 'INKLING_MODEL_PATH': '/x/m.gguf', 'INKLING_PROJECTOR_PATH': '/x/p.gguf'}):
            self.assertEqual(local_text_runtime.server_path('inkling'), Path('/x/llama-server'))
            self.assertEqual(local_text_runtime.variant_path('inkling'), Path('/x/m.gguf'))
            self.assertEqual(local_text_runtime.projector_path('inkling'), Path('/x/p.gguf'))

    def test_results_record_the_inkling_model_and_revision(self):
        self.assertEqual(local_text_runtime.model_id_for('inkling'), local_text_runtime.INKLING_ID)
        self.assertTrue(local_text_runtime.revision_for('inkling').startswith('unsloth:'))
        self.assertEqual(local_text_runtime.known_variant('inkling'), 'inkling')
        self.assertEqual(local_text_runtime.known_variant('something-old'), 'standard')

    def test_flags_follow_the_engine(self):
        qwen = local_text_runtime.server_flags('mtp')
        self.assertIn('--reasoning-budget', qwen)
        self.assertIn('--spec-type', local_text_runtime.variant_args('mtp'))
        inkling = local_text_runtime.server_flags('inkling')
        self.assertIn('--reasoning-budget', inkling)  # a request that thinks has a cap: uncapped, Inkling used a whole answer budget on thinking
        args = local_text_runtime.variant_args('inkling')
        self.assertNotIn('--spec-type', args)  # no speculative drafter on the mainline-compatible build
        self.assertEqual(args[args.index('--temp') + 1], '1.0')  # the model card's sampling, no truncation
        self.assertEqual(args[args.index('--top-k') + 1], '0')
        with patch.dict('os.environ', {'INKLING_ARGS': '--flash-attn on'}):
            self.assertEqual(local_text_runtime.variant_args('inkling')[-2:], ['--flash-attn', 'on'])

    def test_both_reasoning_switches_are_always_sent(self):
        self.assertEqual(local_text_runtime.thinking_kwargs(False), {'enable_thinking': False, 'reasoning_effort': 'none'})
        self.assertEqual(local_text_runtime.thinking_kwargs(True), {'enable_thinking': True, 'reasoning_effort': 'high'})
        self.assertEqual(frame_model.thinking_kwargs(False), local_text_runtime.thinking_kwargs(False))

    def test_chat_sampling_is_the_engines_own(self):
        self.assertEqual(local_text_runtime.sampling_extra('mtp'), {'top_p': 0.95, 'top_k': 20})
        self.assertEqual(local_text_runtime.sampling_extra('inkling')['top_k'], 0)


class StatusTest(unittest.TestCase):
    def test_an_installed_model_without_its_server_is_not_available(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'm.gguf'
            model.write_bytes(b'x')
            env = {'INKLING_MODEL_PATH': str(model), 'INKLING_SERVER_PATH': str(Path(folder) / 'missing-server')}
            with patch.dict('os.environ', env):
                info = local_text_runtime.variant_status('inkling')
                self.assertFalse(info['available'])
                self.assertIn('llama-server is missing', info['detail'])
                (Path(folder) / 'missing-server').write_bytes(b'x')
                self.assertEqual(local_text_runtime.variant_status('inkling'), {'available': True, 'detail': 'Installed'})

    def test_an_uninstalled_selection_falls_back_to_an_installed_engine(self):
        with tempfile.TemporaryDirectory() as folder:
            qwen = Path(folder) / 'qwen.gguf'
            qwen.write_bytes(b'x')
            env = {'INKLING_MODEL_PATH': str(Path(folder) / 'none.gguf'), 'QWEN36_MTP_MODEL_PATH': str(qwen), 'QWEN36_MODEL_PATH': str(Path(folder) / 'none2.gguf')}
            with patch.dict('os.environ', env):
                self.assertEqual(local_text_runtime.resolve_variant('inkling'), 'mtp')
                self.assertEqual(local_text_runtime.resolve_variant('mtp'), 'mtp')


class ResidentRuleTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def files(self, folder, **present):
        env = {}
        for variant, installed in present.items():
            path = Path(folder) / f'{variant}.gguf'
            if installed:
                path.write_bytes(b'x')
            env[{'inkling': 'INKLING_MODEL_PATH', 'mtp': 'QWEN36_MTP_MODEL_PATH', 'standard': 'QWEN36_MODEL_PATH'}[variant]] = str(path)
        return env

    def test_the_model_of_the_selected_engine_is_resident_and_the_other_runs_privately(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', self.files(folder, inkling=True, mtp=True, standard=False)):
            select('mtp')
            self.assertTrue(visual_models.resolve(QWEN36).resident)
            self.assertFalse(visual_models.resolve(INKLING).resident)
            select('inkling')
            self.assertFalse(visual_models.resolve(QWEN36).resident)
            self.assertTrue(visual_models.resolve(INKLING).resident)

    def test_a_job_on_a_non_resident_model_takes_the_whole_device(self):
        payload = {'models': {'initial_frame': 'inkling2-mini', 'keyframe_analysis': 'inkling2-mini'}}
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', self.files(folder, inkling=True, mtp=True, standard=False)):
            select('mtp')
            self.assertFalse(visual_models.runs_resident(payload, ('initial_frame', 'keyframe_analysis')))
            select('inkling')
            self.assertTrue(visual_models.runs_resident(payload, ('initial_frame', 'keyframe_analysis')))

    def test_identity_and_label_follow_the_variant(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', self.files(folder, inkling=True, mtp=True, standard=False)):
            select('inkling')
            model = visual_models.resolve(INKLING)
            self.assertEqual(model.model_id, local_text_runtime.INKLING_ID)
            self.assertIn('+mmproj-sha256:', model.revision)
            self.assertIn('Inkling 2 Mini', model.label)
            # Qwen on a private server records its own build, not the resident engine's.
            self.assertEqual(visual_models.resolve(QWEN36).model_id, local_text_runtime.VARIANTS['mtp']['id'])

    def test_a_private_server_gets_the_engines_arguments_without_jinja_twice(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', self.files(folder, inkling=True, mtp=True, standard=False)):
            select('mtp')
            model = visual_models.resolve(INKLING)
            args = model.private_args()
            self.assertNotIn('--jinja', args)  # frame_server adds it
            self.assertIn('--temp', args)
            self.assertEqual(model.server_files()[0], local_text_runtime.server_path('inkling'))

    def test_the_overview_lists_inkling_with_its_install_command(self):
        listing = {item['key']: item for item in visual_models.overview()['models']}
        self.assertIn('inkling2-mini', listing)
        self.assertEqual(listing['inkling2-mini']['install'], 'python3 scripts/install-model.py inkling2-mini')
        self.assertNotIn('qwen35-27b', listing)
        self.assertEqual(visual_models.overview()['selected'], visual_models.DEFAULTS)


class SwitchedEngineTest(unittest.TestCase):
    setUp = analysis_tests.AnalysisTest.setUp
    tearDown = analysis_tests.AnalysisTest.tearDown

    def test_a_resident_stage_refuses_a_server_of_another_engine(self):
        runtime = local_text_runtime.LocalTextRuntime()
        runtime.process, runtime.url, runtime.variant, runtime.vision = Mock(), 'http://127.0.0.1:9', 'inkling', True
        runtime.process.poll.return_value = None
        with patch.object(runtime, '_ensure', return_value='http://127.0.0.1:9'):  # the server that is up, whatever the selection says
            with self.assertRaisesRegex(RuntimeError, 'switched to Inkling 2 Mini'), runtime.visual_session(None, ('mtp', 'standard')):
                pass
            self.assertEqual(runtime.chat_users, 0)
            with runtime.visual_session(None, ('inkling',)) as url:
                self.assertEqual(url, 'http://127.0.0.1:9')


if __name__ == '__main__':
    unittest.main()
