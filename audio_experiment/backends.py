"""The audio models: pyannote community-1 (diarization + embeddings) and Qwen3-ASR + ForcedAligner.

Same interface as fake_backend.FakeModels, so the whole pipeline runs without a GPU in the dry run. Offline only (HF_HUB_OFFLINE=1).
"""
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get('APP_PARENT', '/app'))
from app.audio_matching import restore_aligned_text  # noqa: E402  (read-only import of the app's own helper)

log = logging.getLogger('aex.models')
DIARIZATION = 'pyannote/speaker-diarization-community-1'
ASR = 'Qwen/Qwen3-ASR-1.7B'
ALIGNER = 'Qwen/Qwen3-ForcedAligner-0.6B'
RATE = 16000
DEFAULT_CLUSTERING = {'threshold': .6, 'Fa': .07, 'Fb': .8}


class AlignmentError(ValueError):
    pass


class RealModels:
    fake = False

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = None
        self.asr = None
        self.load_seconds = {}

    # ------------------------------------------------------------ diarization
    def load_diarization(self):
        if self.pipeline is not None:
            return
        from pyannote.audio import Pipeline
        started = time.monotonic()
        self.pipeline = Pipeline.from_pretrained(DIARIZATION, token=os.environ.get('HF_TOKEN') or None)
        if self.pipeline is None:
            raise RuntimeError('community-1 could not be loaded (offline cache missing?)')
        if self.device != 'cpu':
            self.pipeline.to(self.torch.device(self.device))
        self.load_seconds['diarization'] = time.monotonic() - started

    def diarize(self, wav, threshold=None, min_speakers=None, max_speakers=None, num_speakers=None):
        """-> {'turns': exclusive turns, 'labels', 'centroids': {label: vector}, 'seconds'}."""
        self.load_diarization()
        params = {'segmentation': {'min_duration_off': 0.}, 'clustering': dict(DEFAULT_CLUSTERING)}
        if threshold is not None:
            params['clustering']['threshold'] = threshold
        self.pipeline.instantiate(params)
        file = {'waveform': self.torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))[None], 'sample_rate': RATE, 'uri': 'episode'}
        hints = {k: v for k, v in (('min_speakers', min_speakers), ('max_speakers', max_speakers), ('num_speakers', num_speakers)) if v}
        started = time.monotonic()
        with self.torch.backends.cudnn.flags(enabled=False):
            output = self.pipeline(file, **hints)
        labels = list(output.speaker_diarization.labels())
        embeddings = output.speaker_embeddings
        centroids = {label: embeddings[i].tolist() for i, label in enumerate(labels)} if embeddings is not None else {}
        turns = [{'start_time': float(t.start), 'end_time': float(t.end), 'speaker': s} for t, s in output.exclusive_speaker_diarization]
        return {'turns': turns, 'labels': labels, 'centroids': centroids, 'seconds': time.monotonic() - started}

    def bind(self, number):
        self.number = number

    def embed_span(self, number, start, end, clip):
        return self.embed(clip)

    def embed(self, clip):
        """256-d speaker embedding of one clip with the pipeline's own embedding model (None when too short)."""
        self.load_diarization()
        if len(clip) < 0.3 * RATE:
            return None
        tensor = self.torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))[None, None, :]
        with self.torch.backends.cudnn.flags(enabled=False):
            vector = self.pipeline._embedding(tensor)
        vector = np.asarray(vector)[0]
        return vector if np.isfinite(vector).all() else None

    # ------------------------------------------------------------ ASR
    def load_asr(self):
        if self.asr is not None:
            return
        from qwen_asr import Qwen3ASRModel
        started = time.monotonic()
        dtype = self.torch.bfloat16 if self.device != 'cpu' else self.torch.float32
        device_map = 'cuda:0' if self.device != 'cpu' else 'cpu'
        self.asr = Qwen3ASRModel.from_pretrained(ASR, dtype=dtype, device_map=device_map, max_inference_batch_size=1, max_new_tokens=512,
                                                 forced_aligner=ALIGNER, forced_aligner_kwargs={'dtype': dtype, 'device_map': device_map})
        self.load_seconds['asr'] = time.monotonic() - started

    def transcribe(self, wav, start, end, context='', language=None, depth=0):
        """Transcribe wav[start:end]; returns {'text', 'language', 'tokens': [{start, end, text}]} on the episode clock.

        Like the app, an answer whose timestamps leave the window is retried on each half (up to 3 levels deep).
        """
        self.load_asr()
        a, b = max(0, int(start * RATE)), min(len(wav), int(end * RATE))
        clip = np.ascontiguousarray(wav[a:b], dtype=np.float32)
        duration = len(clip) / RATE
        result = self.asr.transcribe(audio=(clip, RATE), context=context or '', language=language, return_time_stamps=True)[0]
        try:
            words = []
            for item in result.time_stamps or []:
                if not item.text.strip():
                    continue
                s, e = float(item.start_time), float(item.end_time)
                if not (np.isfinite(s) and np.isfinite(e)) or e < s or s < -.08 or e > duration + .08:
                    raise AlignmentError('timestamps outside the window or reversed')
                words.append({'start': start + max(0., min(duration, s)), 'end': start + max(0., min(duration, e)), 'text': item.text.strip()})
            if result.text.strip() and not words:
                raise AlignmentError('no aligned timestamps')
            restored = restore_aligned_text(result.text, [{'start_time': w['start'], 'end_time': w['end'], 'text': w['text']} for w in words])
            if restored is None and words:
                restored = [{'start_time': words[0]['start'], 'end_time': words[-1]['end'], 'text': result.text}]
            tokens = [{'start': r['start_time'], 'end': r['end_time'], 'text': r['text']} for r in restored or []]
            return {'text': result.text, 'language': result.language, 'tokens': tokens}
        except AlignmentError as exc:
            if depth >= 3 or duration < 2.:
                return {'text': result.text, 'language': result.language, 'tokens': [], 'error': str(exc)}
            middle = (start + end) / 2
            left = self.transcribe(wav, start, middle, context, language, depth + 1)
            right = self.transcribe(wav, middle, end, context, language, depth + 1)
            return {'text': left['text'] + right['text'], 'language': left['language'], 'tokens': left['tokens'] + right['tokens'], 'split': True}

    def release(self):
        self.asr = None
        self.pipeline = None
        import gc
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
