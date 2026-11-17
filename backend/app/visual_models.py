"""Which model runs each visual analysis role, and which models exist.

A *role* is one step that reads pictures: the first-frame description, the video motion direction and
the character profile of a new character. Each role can run on any model that lists it in ``roles``.
The user picks one per role (``visual_model_settings``); every queued job snapshots the choice in
``payload.models`` so changing it never affects queued or running work, and a job without the key
(queued before the choice existed) runs on ``DEFAULTS``.

To add a model: add its download to ``model_manifest.json`` (used by ``scripts/install-model.py``),
add a ``VisualModel`` here, mount its folder in the inference compose file, and, only for a new kind
of runtime, a video backend in ``video_model.runtime_for``. Nothing else names a model.

The API container cannot see the model folders (they are mounted into the inference worker only), so
the worker publishes ``status()`` of every model to ``visual_model_state`` and the API reads that.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import db, frame_model, video_model

ROLES = ('initial_frame', 'video_analysis', 'character_profile')
ROLE_LABELS = {'initial_frame': 'First frame', 'video_analysis': 'Video motion', 'character_profile': 'Character profile'}
MANIFEST = json.loads((Path(__file__).with_name('model_manifest.json')).read_text())
SERVER_ENV, SERVER_DEFAULT = 'QWEN35_SERVER_PATH', '/models/llamacpp/llama-server'  # the one bundled llama.cpp build serves every GGUF model
PUBLISH_SECONDS = 30
STALE_SECONDS = 180


@dataclass(frozen=True)
class VisualModel:
    key: str
    label: str
    short_name: str  # written into text the model produces, e.g. the character profile's heading
    model_id: str
    revision: str
    runtime: str  # 'llama-server' (GGUF + projector) or 'transformers' (a safetensors folder)
    roles: tuple
    files: dict  # file role -> (environment variable, default path inside the worker)
    server_args: tuple = ()
    hears_audio: bool = False  # the video role sends the clip with its soundtrack
    manifest: str | None = None  # model_manifest.json entry the installer downloads
    video_fps: float | None = None  # a llama-server video model's fixed sampling rate (the server resamples every clip to it)
    note: str = ''

    def path(self, name):
        env, default = self.files[name]
        return Path(os.environ.get(env) or default)

    def server_files(self):
        """(llama-server, model, projector) for ``frame_model.frame_server``."""
        return Path(os.environ.get(SERVER_ENV) or SERVER_DEFAULT), self.path('model'), self.path('projector')


def _lfm3():
    entry = MANIFEST['lfm3-omni-2b']
    return VisualModel(
        key='lfm3-omni-2b', label='LFM3-Omni 2B · BF16', short_name='LFM3-Omni-2B',
        model_id='LiquidAI/LFM3-Omni-2B-GGUF:BF16', revision=entry['revision'], runtime='llama-server',
        roles=ROLES,
        files={'model': ('LFM3_OMNI_MODEL_PATH', '/models/lfm3-omni-2b/LFM3-Omni-2B-BF16.gguf'),
               'projector': ('LFM3_OMNI_PROJECTOR_PATH', '/models/lfm3-omni-2b/mmproj-LFM3-Omni-2B-BF16.gguf')},
        server_args=('--reasoning', 'off'), hears_audio=True, manifest='lfm3-omni-2b', video_fps=2.0,
        note='2.2B omni model: sees the frames and hears the clip, much faster, less detailed.')


MODELS = {model.key: model for model in (
    VisualModel(key='qwen35-27b', label='Qwen3.5 27B · Q4_K_M', short_name='Qwen3.5-27B',
                model_id=frame_model.MODEL_ID, revision=frame_model.MODEL_REVISION, runtime='llama-server',
                roles=('initial_frame', 'character_profile'),
                files={'model': ('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
                       'projector': ('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf')},
                note='The default still-image model.'),
    VisualModel(key='qwen3-vl-8b', label='Qwen3-VL 8B · BF16', short_name='Qwen3-VL-8B',
                model_id=video_model.MODEL_ID, revision=video_model.MODEL_REVISION, runtime='transformers',
                roles=('video_analysis',), files={'model': ('QWEN3_VIDEO_MODEL_PATH', '/models/qwen3-vl-8b')},
                note='The default video model (frames only, no sound).'),
    _lfm3(),
)}
DEFAULTS = {'initial_frame': 'qwen35-27b', 'video_analysis': 'qwen3-vl-8b', 'character_profile': 'qwen35-27b'}


def supports(key, role):
    return key in MODELS and role in MODELS[key].roles


# -- files on disk (inference worker only) ---------------------------------------------------------

def _shards_complete(folder):
    try:
        shards = set(json.loads((folder / 'model.safetensors.index.json').read_text())['weight_map'].values())
        return bool(shards) and all(Path(name).name == name and (folder / name).is_file() for name in shards)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def status(key):
    """{'available', 'detail'} for one model as the worker sees its files."""
    model = MODELS[key]
    if model.runtime == 'transformers':
        ok = _shards_complete(model.path('model'))
        return {'available': ok, 'detail': 'Installed' if ok else 'Not installed or incomplete'}
    server, *files = model.server_files()
    if all(path.is_file() for path in files):
        return {'available': server.is_file(), 'detail': 'Installed' if server.is_file() else 'llama-server runtime is missing'}
    entry = MANIFEST.get(model.manifest or '')
    if entry:
        sizes = {item['name']: item['size'] for item in entry['files']}
        done = 0
        partial = False
        for path in files:
            if path.is_file():
                done += sizes.get(path.name, path.stat().st_size)
            elif path.with_name(path.name + '.partial').is_file():
                partial = True
                done += path.with_name(path.name + '.partial').stat().st_size
        if partial or done:
            return {'available': False, 'detail': f'Downloading {min(100.0, 100 * done / max(1, sum(sizes.values()))):.0f}%'}
    return {'available': False, 'detail': 'Not installed'}


def require(key):
    """Raise the error a job shows when its model is missing."""
    info = status(key)
    if not info['available']:
        raise RuntimeError(f"{MODELS[key].label} is not available ({info['detail']})" +
                           (f'; install it with: python3 scripts/install-model.py {MODELS[key].manifest}' if MODELS[key].manifest else ''))


_published = 0.0


def publish(force=False):
    """Write every model's status for the API; at most every PUBLISH_SECONDS unless forced."""
    global _published
    now = time.time()
    if not force and now - _published < PUBLISH_SECONDS:
        return
    _published = now
    try:
        report = {}
        for key in MODELS:
            try:
                report[key] = status(key)
            except OSError as exc:
                report[key] = {'available': False, 'detail': f'Unreadable: {exc}'}
        with db.connect() as conn:
            conn.execute('UPDATE visual_model_state SET models_json=?,updated_at=? WHERE id=1', (json.dumps(report), now))
    except Exception:
        pass  # a locked database must not stop the worker loop


# -- the user's choice and job snapshots -----------------------------------------------------------

def selected(conn=None):
    """{role: model key} the user chose, with the defaults where nothing (or something unknown) is saved."""
    rows = (conn.execute('SELECT role,model FROM visual_model_settings').fetchall() if conn is not None
            else db.all_rows('SELECT role,model FROM visual_model_settings'))
    chosen = {row['role']: row['model'] for row in rows}
    return {role: chosen[role] if supports(chosen.get(role), role) else DEFAULTS[role] for role in ROLES}


def snapshot(conn, roles):
    """The ``payload.models`` a job carries for the roles it runs."""
    choice = selected(conn)
    return {role: choice[role] for role in roles}


def for_job(payload, role):
    """The model a job runs for ``role``: its snapshot, else the default (jobs queued before the choice existed)."""
    models = payload.get('models') if isinstance(payload, dict) else None
    key = models.get(role) if isinstance(models, dict) else None
    return MODELS[key if supports(key, role) else DEFAULTS[role]]


def reported():
    """The worker's last report: ({key: status}, seconds since it was written or None)."""
    row = db.one('SELECT models_json,updated_at FROM visual_model_state WHERE id=1')
    try:
        report = json.loads(row['models_json']) if row else {}
    except (TypeError, ValueError):
        report = {}
    age = time.time() - row['updated_at'] if row and row['updated_at'] else None
    return (report if isinstance(report, dict) else {}), age


def overview():
    """What the Visual engine dialog shows."""
    report, age = reported()
    stale = age is None or age > STALE_SECONDS
    models = []
    for key, model in MODELS.items():
        info = report.get(key) or {}
        models.append({'key': key, 'label': model.label, 'model_id': model.model_id, 'roles': list(model.roles), 'note': model.note,
                       'hears_audio': model.hears_audio, 'install': f'python3 scripts/install-model.py {model.manifest}' if model.manifest else None,
                       'available': bool(info.get('available')), 'detail': info.get('detail') or 'The inference worker has not reported it'})
    return {'roles': [{'role': role, 'label': ROLE_LABELS[role], 'default': DEFAULTS[role]} for role in ROLES],
            'models': models, 'selected': selected(), 'worker_reported': not stale}
