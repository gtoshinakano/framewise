"""Shared plumbing: paths, the run folder, checkpoints, progress and timings. Stdlib only."""
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger('aex')

PROJECT_ID = 'd66e89d5-79ba-4c2b-9f9c-a4285ffa5615'
# number -> (episode id, label). Episodes 1-3 are the evaluated ones, 4-7 only feed the OP/ED search.
EPISODES = {
    1: '944df231-2b86-461a-a1b7-29b32fc24a76',
    2: 'e4d40c01-3a6e-493d-8e7b-c891db2659ae',
    3: '7c5ba9a5-e7ea-4893-94d4-58cfd31035bf',
    4: '1eca5799-2063-47f6-8ea3-24911262c239',
    5: 'e1d1a897-a066-4cd6-8f33-90456e53ccd6',
    6: '34224b1a-2a65-4c97-b4f3-a3b375572323',
    7: '47dae7a0-5e6c-4f28-b908-c5618b6d9cf6',
}
EVALUATED = (3, 2, 1)           # processing order: the held-out episode first
SAMPLE_RATE = 16000
DATA_DIR = Path(os.environ.get('DATA_DIR', '/data'))


def episode_audio(number, project_id=PROJECT_ID):
    return DATA_DIR / 'projects' / project_id / 'episodes' / EPISODES[number] / 'audio.wav'


def voice_reference(sample_id, project_id=PROJECT_ID):
    return DATA_DIR / 'projects' / project_id / 'voice-references' / f'{sample_id}.wav'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o777)  # the container user differs from the host user
    except OSError:
        pass
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


class Run:
    """One run folder: checkpoints per stage and episode, progress.json, timings.json, log.txt."""

    def __init__(self, root, run_id, deadline=None, dry=False):
        self.id, self.dir, self.dry = run_id, Path(root) / run_id, dry
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.dir, 0o777)
        except OSError:
            pass
        self.deadline = deadline          # epoch seconds, the end of the whole run
        self.lock = threading.Lock()
        self.started = time.time()
        self._progress = read_json(self.dir / 'progress.json', {}) or {}
        self._progress.setdefault('stages', {})
        self._progress.setdefault('started_at', time.strftime('%Y-%m-%d %H:%M:%S'))
        self._progress.setdefault('started_epoch', time.time())
        self._timings = read_json(self.dir / 'timings.json', {}) or {}
        self._notes = read_json(self.dir / 'notes.json', []) or []

    # ------------------------------------------------------------ files
    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def wpath(self, *parts):
        """Path for writing: parents are created and opened to every user (the container user differs from the host user)."""
        path = self.path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        for folder in [path.parent, *path.parent.parents]:
            if folder == self.dir.parent:
                break
            try:
                os.chmod(folder, 0o777)
            except OSError:
                pass
        return path

    def done(self, *parts):
        return self.path(*parts).is_file()

    def save(self, value, *parts):
        write_json(self.path(*parts), value)

    def load(self, *parts, default=None):
        return read_json(self.path(*parts), default)

    # ------------------------------------------------------------ progress
    def progress(self, stage, status=None, done=None, total=None, message=None):
        with self.lock:
            entry = self._progress['stages'].setdefault(stage, {'status': 'pending', 'done': 0, 'total': 0})
            for key, value in (('status', status), ('done', done), ('total', total), ('message', message)):
                if value is not None:
                    entry[key] = value
            self._progress['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self._progress['elapsed_seconds'] = round(time.time() - self._progress['started_epoch'])
            self._progress['status'] = self._progress.get('status') or 'running'
            write_json(self.dir / 'progress.json', self._progress)

    def set_status(self, status):
        with self.lock:
            self._progress['status'] = status
            write_json(self.dir / 'progress.json', self._progress)

    def note(self, text, kind='info'):
        """A line the report shows under 'skipped / changed / caveats'."""
        with self.lock:
            self._notes.append({'kind': kind, 'text': text, 'at': time.strftime('%Y-%m-%d %H:%M:%S')})
            write_json(self.dir / 'notes.json', self._notes)
        self.log(f'NOTE[{kind}] {text}')

    def log(self, text):
        line = f"{time.strftime('%H:%M:%S')} {text}"
        print(line, flush=True)
        try:
            with open(self.dir / 'log.txt', 'a') as handle:
                handle.write(line + '\n')
            os.chmod(self.dir / 'log.txt', 0o666)
        except OSError:
            pass

    # ------------------------------------------------------------ timings
    def timing(self, key, seconds, **extra):
        with self.lock:
            self._timings[key] = {'seconds': round(seconds, 2), **extra}
            write_json(self.dir / 'timings.json', self._timings)

    @contextmanager
    def timed(self, key, **extra):
        started = time.monotonic()
        try:
            yield
        finally:
            self.timing(key, time.monotonic() - started, **extra)

    def timings(self):
        return dict(self._timings)

    # ------------------------------------------------------------ budget
    def time_left(self):
        return None if self.deadline is None else self.deadline - time.time()

    def out_of_time(self, reserve=0):
        left = self.time_left()
        return left is not None and left < reserve


def stage_deadline(minutes):
    """Epoch second `minutes` from now (a per-stage cap)."""
    return time.time() + minutes * 60


def peak_vram_mib():
    """Highest VRAM use over every DRM card, in MiB (None without a readable counter)."""
    best = None
    try:
        for card in Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'):
            try:
                value = int(card.read_text()) // 1048576
            except (OSError, ValueError):
                continue
            best = value if best is None else max(best, value)
    except OSError:
        pass
    return best


class VramSampler(threading.Thread):
    """Samples the VRAM counter every few seconds; `peak` is the maximum seen while it ran."""

    def __init__(self, every=3.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.stop_event = every, None, threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            value = peak_vram_mib()
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)
            self.stop_event.wait(self.every)

    def finish(self):
        self.stop_event.set()
        value = peak_vram_mib()
        if value is not None:
            self.peak = value if self.peak is None else max(self.peak, value)
        return self.peak
