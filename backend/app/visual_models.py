"""Which model runs each visual analysis role, and which models exist.

A *role* is one step that reads pictures: the first-frame description, the keyframe analysis (what changes
between the keyframes of a cut) and the character profile of a new character. Each role can run on any model that lists it in ``roles``.
The user picks one per role (``visual_model_settings``); every queued job snapshots the choice in
``payload.models`` so changing it never affects queued or running work, and a job without the key
(queued before the choice existed) runs on ``DEFAULTS``.

To add a model: add its download to ``model_manifest.json`` (used by ``scripts/install-model.py``),
add a ``VisualModel`` here and mount its folder in the inference compose file. Nothing else names a model.
A model without roles is a *helper* (DWPose, the CPU pose model the keyframe analysis merges in): it is never
chosen, only installed, and its status tells the keyframe stage whether it can run the hybrid.

The Qwen3-VL-8B video model and its ``video_analysis`` role were removed on 2026-11-18 (keyframe analysis replaced
them); a saved choice or a job snapshot for that role is ignored.

The API container cannot see the model folders (they are mounted into the inference worker only), so
the worker publishes ``status()`` of every model to ``visual_model_state`` and the API reads that.
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import db, frame_model

ROLES = ('initial_frame', 'keyframe_analysis', 'character_profile')
ROLE_LABELS = {'initial_frame': 'First frame', 'keyframe_analysis': 'Keyframes (motion)', 'character_profile': 'Character profile'}
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
    runtime: str  # 'llama-server' (GGUF + projector) or 'onnx' (CPU helper files)
    roles: tuple
    files: dict  # file role -> (environment variable, default path inside the worker), or (folder variable, default folder, file name)
    server_args: tuple = ()
    manifest: str | None = None  # model_manifest.json entry the installer downloads
    note: str = ''

    def path(self, name):
        env, default, *file_name = self.files[name]
        found = Path(os.environ.get(env) or default)
        return found / file_name[0] if file_name else found

    def server_files(self):
        """(llama-server, model, projector) for ``frame_model.frame_server``."""
        return Path(os.environ.get(SERVER_ENV) or SERVER_DEFAULT), self.path('model'), self.path('projector')


def _lfm3():
    entry = MANIFEST['lfm3-omni-2b']
    return VisualModel(
        key='lfm3-omni-2b', label='LFM3-Omni 2B · BF16', short_name='LFM3-Omni-2B',
        model_id='LiquidAI/LFM3-Omni-2B-GGUF:BF16', revision=entry['revision'], runtime='llama-server',
        roles=('initial_frame', 'character_profile'),  # describing changes between keyframes failed in the experiment (multiframe_experiment/FINDINGS.md)
        files={'model': ('LFM3_OMNI_MODEL_PATH', '/models/lfm3-omni-2b/LFM3-Omni-2B-BF16.gguf'),
               'projector': ('LFM3_OMNI_PROJECTOR_PATH', '/models/lfm3-omni-2b/mmproj-LFM3-Omni-2B-BF16.gguf')},
        server_args=('--reasoning', 'off'), manifest='lfm3-omni-2b',
        note='2.2B omni model: much faster, less detailed.')


MODELS = {model.key: model for model in (
    VisualModel(key='qwen35-27b', label='Qwen3.5 27B · Q4_K_M', short_name='Qwen3.5-27B',
                model_id=frame_model.MODEL_ID, revision=frame_model.MODEL_REVISION, runtime='llama-server',
                roles=('initial_frame', 'keyframe_analysis', 'character_profile'),
                files={'model': ('QWEN35_MODEL_PATH', '/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf'),
                       'projector': ('QWEN35_PROJECTOR_PATH', '/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf')},
                note='The default still-image model.'),
    _lfm3(),
    VisualModel(key='dwpose', label='DWPose · ONNX (CPU)', short_name='DWPose', model_id='yzd-v/DWPose',
                revision=MANIFEST['dwpose']['revision'], runtime='onnx', roles=(),
                files={'detector': ('DWPOSE_DIR', '/models/dwpose', 'yolox_l.onnx'), 'pose': ('DWPOSE_DIR', '/models/dwpose', 'dw-ll_ucoco_384.onnx')},
                manifest='dwpose',
                note='Body keypoints on the CPU, merged into the keyframe analysis (arms, legs, position, walking). Without it the keyframe analysis runs on Qwen3.5 alone.'),
)}
DEFAULTS = {'initial_frame': 'qwen35-27b', 'keyframe_analysis': 'qwen35-27b', 'character_profile': 'qwen35-27b'}


def supports(key, role):
    return key in MODELS and role in MODELS[key].roles


# -- files on disk (inference worker only) ---------------------------------------------------------

def status(key):
    """{'available', 'detail'} for one model as the worker sees its files."""
    model = MODELS[key]
    if model.runtime == 'onnx':
        server, files = None, [model.path(name) for name in model.files]
    else:
        server, *files = model.server_files()
    if all(path.is_file() for path in files):
        if server is None:
            return {'available': True, 'detail': 'Installed'}
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
                       'helper': not model.roles, 'install': f'python3 scripts/install-model.py {model.manifest}' if model.manifest else None,
                       'available': bool(info.get('available')), 'detail': info.get('detail') or 'The inference worker has not reported it'})
    return {'roles': [{'role': role, 'label': ROLE_LABELS[role], 'default': DEFAULTS[role]} for role in ROLES],
            'models': models, 'selected': selected(), 'worker_reported': not stale}
