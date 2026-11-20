"""Local model adapters for the bulk anime-character workflow.

AniRef uses Ultralytics on the existing ROCm PyTorch runtime. Two identity
embedders are available (``ANIME_REID_BACKEND``):

* ``ccip`` (default): deepghs CCIP, an anime "same character?" model. Its
  preprocessing follows imgutils.metrics.ccip (RGB, bilinear 384 x 384,
  CLIP mean/std) except that tall person crops are letterboxed on white
  instead of stretched, which matched the user's labels better. Its
  comparator is (1 - cosine) / 2, so plain cosine similarity of the
  L2-normalised features is equivalent.
* ``dinov3``: BeUnMerreHuman/Anime-Character-Re-Identification's model.py:
  RGB, 448px grey letterbox, ImageNet normalization, ONNX output.

Each embedder carries its own model id, preprocessing tag and matching
defaults, because similarity scales differ between models. No adapter
downloads weights or silently selects a different model.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from .anime_matching import normalize

ANIREF_MODEL_ID = 'SoulflareRC/AniRef40000-s-epoch40'
ANIREF_MODEL_SHA256 = '0b811109555bb6e13d0c39791aa23d7f48f841163656ced1d42386ed8350fd6c'
REID_MODEL_ID = 'BeUnMerreHuman/Anime-Character-Re-Identification:DINOv3'
REID_PREPROCESSING = 'rgb-pad-448-gray128-imagenet-v1'
CCIP_MODEL_PREFIX = 'deepghs/ccip_onnx:'
CCIP_DEFAULT_VARIANT = 'ccip-caformer_b36-24'
CCIP_PREPROCESSING = 'rgb-pad-white-384-clip-v1'

# Matching defaults per embedder, calibrated on labelled episodes of this
# project (see docs/model-integration.md). A box is labelled when its best
# character's top-3 score reaches similarity and leads by margin, or, as a
# second tier, reaches wide[0] and leads by wide[1]: a strong match needs a
# small lead, a weaker one a clear lead. track/adjacent link boxes of one
# character inside a cut and across a cut boundary; atypical is the gallery
# outlier floor.
DINOV3_DEFAULTS = {'similarity': 0.85, 'margin': 0.03, 'wide': None, 'track': 0.90, 'adjacent': 0.97, 'atypical': 0.70}
CCIP_DEFAULTS = {'similarity': 0.80, 'margin': 0.03, 'wide': (0.64, 0.10), 'track': 0.85, 'adjacent': 0.95, 'atypical': 0.75}


def create_embedder(backend=None):
    """Return the configured identity embedder (CCIP unless ANIME_REID_BACKEND=dinov3)."""
    backend = (backend or os.environ.get('ANIME_REID_BACKEND') or 'ccip').strip().lower()
    if backend == 'dinov3':
        return AnimeReId(os.environ.get('ANIME_REID_MODEL_PATH', '/models/anime-reid/dino_v3.onnx'))
    if backend == 'ccip':
        return CcipEmbedder(os.environ.get('CCIP_MODEL_DIR', '/models/ccip'),
                            os.environ.get('CCIP_VARIANT') or CCIP_DEFAULT_VARIANT)
    raise RuntimeError(f'Unknown ANIME_REID_BACKEND {backend!r}; use ccip or dinov3')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_revision(path):
    """Fingerprint an ONNX graph and every external tensor data file."""
    import onnx
    path = Path(path)
    graph = onnx.load(str(path), load_external_data=False)
    locations = {item.value for tensor in graph.graph.initializer for item in tensor.external_data
                 if item.key == 'location'}
    base = path.parent.resolve()
    sources = [(path.name, path)]
    for location in sorted(locations):
        source = (base / location).resolve()
        if not source.is_relative_to(base) or not source.is_file():
            raise RuntimeError(f'Anime Re-ID external tensor data is missing or invalid: {location}')
        sources.append((location, source))
    digest = hashlib.sha256()
    for name, source in sources:
        digest.update(name.encode('utf-8') + b'\0')
        with source.open('rb') as contents:
            for chunk in iter(lambda: contents.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


class AniRefDetector:
    def __init__(self, path, device='0'):
        import torch
        from ultralytics import YOLO
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f'AniRef weights are missing: {path}')
        if device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('AniRef requires the configured GPU; ROCm PyTorch cannot access it')
        self.revision = file_sha256(path)
        self.model = YOLO(str(path))
        if list(self.model.names.values()) != ['person']:
            raise RuntimeError('The configured AniRef model does not have the expected person class')
        self.device = device

    def detect(self, images, confidence=0.25, iou=0.7, scores=False):
        """Person boxes per image; with ``scores`` each entry is ``(box, confidence)``."""
        if not images:
            return []
        # Ultralytics preserves each source image's coordinate system in xyxy.
        results = self.model.predict(source=images, imgsz=640, conf=confidence, iou=iou,
                                     device=self.device, verbose=False, stream=False)
        if len(results) != len(images):
            raise RuntimeError('AniRef returned an unexpected number of frames')
        frames = []
        for image, result in zip(images, results):
            width, height = image.size
            boxes = []
            confidences = result.boxes.conf.cpu().tolist()
            for coords, score in zip(result.boxes.xyxy.cpu().tolist(), confidences):
                if len(coords) != 4 or any(not math.isfinite(value) for value in coords):
                    continue
                x1, y1, x2, y2 = coords
                box = [max(0, min(width, x1)), max(0, min(height, y1)),
                       max(0, min(width, x2)), max(0, min(height, y2))]
                if box[0] < box[2] and box[1] < box[3]:
                    boxes.append((box, float(score)) if scores else box)
            frames.append(boxes)
        return frames


class AnimeReId:
    model_id = REID_MODEL_ID
    preprocessing = REID_PREPROCESSING
    defaults = DINOV3_DEFAULTS
    label = 'DINOv3 Re-ID'

    def __init__(self, path, provider=None):
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f'DINOv3 anime Re-ID ONNX weights are missing: {path}')
        self.revision = onnx_revision(path)
        self.provider = provider or os.environ.get('ANIME_REID_PROVIDER', 'torch-rocm')
        if self.provider == 'torch-rocm':
            self.session = _TorchOnnxSession(path)
        else:
            self.session = _onnx_session(path, self.provider)
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError('Anime Re-ID model must have exactly one image input')
        self.input = inputs[0]
        if len(self.input.shape) != 4:
            raise RuntimeError('Anime Re-ID model must accept NCHW image input')

    def embed(self, images, batch_size=4):
        import numpy as np
        from PIL import ImageOps
        if not images:
            return []
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        # Some ONNX exports have a fixed batch dimension. Honor it rather than
        # assuming the project's batch example implies a dynamic model.
        fixed = self.input.shape[0]
        fixed = fixed if isinstance(fixed, int) and fixed > 0 else None
        step = fixed or batch_size
        vectors = []
        for offset in range(0, len(images), step):
            batch_images = images[offset:offset + step]
            batch = []
            for image in batch_images:
                padded = ImageOps.pad(image.convert('RGB'), (448, 448), color=(128, 128, 128))
                pixels = np.asarray(padded, dtype=np.float32) / 255.0
                pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
                batch.append(pixels.transpose(2, 0, 1))
            if fixed and len(batch) < fixed:
                batch.extend([batch[-1]] * (fixed - len(batch)))
            output = self.session.run(None, {self.input.name: np.stack(batch)})[0]
            if output.ndim != 2 or len(output) != len(batch):
                raise RuntimeError('Anime Re-ID returned an unexpected embedding shape')
            vectors.extend(normalize(row) for row in output[:len(batch_images)])
        return vectors


class CcipEmbedder:
    """deepghs CCIP feature extractor; cosine of its features is the CCIP comparator."""
    preprocessing = CCIP_PREPROCESSING
    defaults = CCIP_DEFAULTS
    label = 'CCIP'

    def __init__(self, directory, variant=CCIP_DEFAULT_VARIANT, provider=None):
        if not variant or '/' in variant or variant.startswith('.'):
            raise RuntimeError(f'Invalid CCIP variant: {variant!r}')
        path = Path(directory) / variant / 'model_feat.onnx'
        if not path.is_file():
            raise RuntimeError(f'CCIP weights are missing: {path}; run scripts/download-ccip.sh')
        self.variant = variant
        self.model_id = CCIP_MODEL_PREFIX + variant
        self.revision = file_sha256(path)
        metrics = path.parent / 'metrics.json'
        # The published "same character" threshold is a difference (1 - cos) / 2.
        self.published_similarity = 1 - 2 * float(json.loads(metrics.read_text())['threshold']) if metrics.is_file() else None
        self.provider = provider or os.environ.get('CCIP_PROVIDER') or os.environ.get('ANIME_REID_PROVIDER', 'torch-rocm')
        if self.provider == 'torch-rocm':
            self.session = _TorchOnnxSession(path, patch_graph=_squeeze_attnpool_if)
        else:
            self.session = _onnx_session(path, self.provider)
        self.input = self.session.get_inputs()[0]

    def embed(self, images, batch_size=16):
        import numpy as np
        from PIL import Image, ImageOps
        if not images:
            return []
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
        vectors = []
        for offset in range(0, len(images), batch_size):
            batch = []
            for image in images[offset:offset + batch_size]:
                resized = ImageOps.pad(image.convert('RGB'), (384, 384), color=(255, 255, 255))
                batch.append((np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0 - mean) / std)
            output = self.session.run(None, {self.input.name: np.stack(batch).astype(np.float32)})[0]
            if output.ndim != 2 or len(output) != len(batch):
                raise RuntimeError('CCIP returned an unexpected feature shape')
            vectors.extend(normalize(row) for row in output)
        return vectors


def _squeeze_attnpool_if(graph):
    """Replace the exporter's ``if shape[0] == 1: squeeze(0)`` guard with the squeeze.

    CCIP's attention pool has a single query token, so that dimension is
    always 1 and onnxruntime always takes the squeeze branch; onnx2torch
    cannot convert ``If``. Verified equal to onnxruntime within 1e-6.
    """
    import onnx
    for index, node in enumerate(list(graph.graph.node)):
        if node.op_type != 'If':
            continue
        then = next(attr.g for attr in node.attribute if attr.name == 'then_branch')
        if len(then.node) != 1 or then.node[0].op_type != 'Squeeze':
            raise RuntimeError('Unsupported conditional in CCIP graph')
        graph.graph.initializer.extend(then.initializer)
        graph.graph.node.remove(node)
        graph.graph.node.insert(index, onnx.helper.make_node('Squeeze', list(then.node[0].input), list(node.output), name=node.name + '_squeeze'))
    return graph


def _onnx_session(path, provider):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f'Anime Re-ID provider {provider} is unavailable; installed providers: {available}')
    options = ort.SessionOptions()
    if provider != 'CPUExecutionProvider':
        # An unannounced CPU fallback defeats the purpose of the fast path.
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    if provider not in session.get_providers():
        raise RuntimeError(f'Anime Re-ID failed to initialize {provider}')
    return session


class _TorchOnnxSession:
    """Execute the ONNX graph on ROCm through PyTorch when conversion supports it."""
    def __init__(self, path, patch_graph=None):
        from types import SimpleNamespace
        import onnx
        import torch
        from onnx2torch import convert
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError('Anime Re-ID requires ROCm PyTorch and an available GPU')
        graph = onnx.load(str(path))
        if patch_graph:
            graph = patch_graph(graph)
        initializers = {item.name for item in graph.graph.initializer}
        self.inputs = []
        for item in graph.graph.input:
            if item.name in initializers:
                continue
            shape = [dim.dim_value if dim.HasField('dim_value') else dim.dim_param or None
                     for dim in item.type.tensor_type.shape.dim]
            self.inputs.append(SimpleNamespace(name=item.name, shape=shape))
        try:
            # The DINOv3 export uses ONNX Reshape allowzero=1. onnx2torch
            # rejects it, although torch.reshape supports literal zero sizes.
            # Override only this converter while constructing the graph.
            from onnx2torch.node_converters import registry
            from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node
            reshape_key = registry.OperationDescription('', 'Reshape', 14)
            original = registry._CONVERTER_REGISTRY[reshape_key]

            class LiteralZeroReshape(torch.nn.Module):
                def forward(self, value, shape):
                    return torch.reshape(value, torch.Size(shape))

            def reshape_converter(node, graph):
                if node.attributes.get('allowzero', 0) == 1:
                    return OperationConverterResult(torch_module=LiteralZeroReshape(), onnx_mapping=onnx_mapping_from_node(node=node))
                return original(node, graph)

            class Reduce18(torch.nn.Module):
                def __init__(self, kind, keepdims, noop):
                    super().__init__()
                    self.kind, self.keepdims, self.noop = kind, bool(keepdims), bool(noop)

                def forward(self, value, axes=None):
                    if axes is None or axes.numel() == 0:
                        if self.noop:
                            return value
                        dims = tuple(range(value.ndim))
                    else:
                        dims = tuple(int(axis) for axis in axes.tolist())
                    if self.kind == 'ReduceMean':
                        return torch.mean(value, dim=dims, keepdim=self.keepdims)
                    return torch.linalg.vector_norm(value, ord=2, dim=dims, keepdim=self.keepdims)

            def reduce_converter(node, graph):
                return OperationConverterResult(
                    torch_module=Reduce18(node.operation_type, node.attributes.get('keepdims', 1), node.attributes.get('noop_with_empty_axes', 0)),
                    onnx_mapping=onnx_mapping_from_node(node=node))

            split_13 = registry._CONVERTER_REGISTRY[registry.OperationDescription('', 'Split', 13)]
            split_key = registry.OperationDescription('', 'Split', 18)

            def split_converter(node, graph):
                # DINOv3 provides explicit split sizes. The older converter has
                # identical semantics for that case; reject other variants.
                if len(node.input_values) != 2 or node.attributes.get('num_outputs') is not None:
                    raise RuntimeError('Unsupported ONNX Split-18 variant in Anime Re-ID graph')
                return split_13(node, graph)

            overrides = {reshape_key: reshape_converter, split_key: split_converter,
                         registry.OperationDescription('', 'ReduceL2', 18): reduce_converter,
                         registry.OperationDescription('', 'ReduceMean', 18): reduce_converter}
            previous = {key: registry._CONVERTER_REGISTRY.get(key) for key in overrides}
            registry._CONVERTER_REGISTRY.update(overrides)
            try:
                self.model = convert(graph).eval().to('cuda')
            finally:
                for key, converter in previous.items():
                    if converter is None:
                        registry._CONVERTER_REGISTRY.pop(key, None)
                    else:
                        registry._CONVERTER_REGISTRY[key] = converter
        except Exception as exc:
            raise RuntimeError(f'Could not convert Anime Re-ID ONNX graph for ROCm: {exc}') from exc

    def get_inputs(self):
        return self.inputs

    def run(self, output_names, feed):
        import torch
        with torch.inference_mode():
            result = self.model(*(torch.from_numpy(feed[item.name]).to('cuda') for item in self.inputs))
        if not isinstance(result, (list, tuple)):
            result = (result,)
        return [item.detach().cpu().numpy() for item in result]
