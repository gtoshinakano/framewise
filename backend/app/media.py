import array
import json
import math
import re
import subprocess
import tempfile
import wave
from collections import deque
from pathlib import Path

from .db import DATA_DIR


def episode_dir(project_id: str, episode_id: str) -> Path:
    path = DATA_DIR / "projects" / project_id / "episodes" / episode_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def voice_reference_path(project_id: str, sample_id: str) -> Path:
    return DATA_DIR / 'projects' / project_id / 'voice-references' / f'{sample_id}.wav'


def run(args: list[str], timeout: float | None = None):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Media command timed out') from exc
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1800:])
    return result


def probe(path: Path):
    result = run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    info = json.loads(result.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("The uploaded file has no video stream")
    audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Could not determine video duration")
    return {"duration": duration, "width": int(video["width"]), "height": int(video["height"]), "has_audio": audio}


# Frame rates derived from a frame budget (e.g. 4.505561673375263 fps) have no tidy timebase, and the mp4
# muxer then fails with "Not yet implemented in FFmpeg". A fixed track timescale avoids that.
MP4_TIMESCALE = ('-video_track_timescale', '90000')


def prepare_analysis_video(source: Path, output: Path, start: float, end: float, width: int = 640, height: int = 360, fps: float = 4, frame_limit: int | None = None,
                           keep_audio: bool = False):
    """Create a temporary cut clip and return its reproducibility metadata.

    The clip is silent unless ``keep_audio`` (for a video model that also hears the soundtrack): then the
    source's first audio track is kept as 16 kHz mono AAC and ``has_audio`` says whether there was one.
    """
    if source.resolve() == output.resolve():
        raise ValueError('Analysis clip cannot replace its source')
    if not all(math.isfinite(v) for v in (start, end, fps)) or start < 0 or end <= start or not 0 < fps <= 30:
        raise ValueError('Invalid analysis clip timing or frame rate')
    if width < 128 or width > 1920 or height < 128 or height > 1080 or width % 2 or height % 2:
        raise ValueError('Invalid analysis clip dimensions')
    # Keep the original aspect ratio; black padding supplies the requested canvas.
    scaling = f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    if end - start < 1 / fps:
        # A very short interval can yield no frame at a low output FPS. Extract
        # a real in-cut sample and duplicate it for video processors that need
        # two frames; report this so no motion is inferred from the padding.
        with tempfile.TemporaryDirectory(prefix='framewise-short-cut-') as scratch:
            still = Path(scratch) / 'frame.jpg'
            actual = extract_cut_frame(source, still, start, start, end, width=width)
            run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-loop', '1', '-framerate', str(fps), '-i', str(still),
                 '-frames:v', '2', '-an', '-vf', scaling, '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', *MP4_TIMESCALE, str(output)])
        return {'source_start': start, 'source_end': end, 'width': width, 'height': height, 'fps': fps, 'padded_tail': True, 'single_sample_time': actual}
    # Trim decoded source timestamps before the fps filter. An output-side -t
    # alone can let fps inspect and select a frame at or beyond the cut end.
    filters = f'trim=duration={end - start:.9f},setpts=PTS-STARTPTS,fps={fps},{scaling}'
    padded_tail = end - start < 2 / fps
    if padded_tail:
        # The Qwen video processor needs at least two frames, even for a very
        # short cut. Clone the final sampled frame without reading the next cut.
        filters += ',tpad=stop_mode=clone:stop=1'
    has_audio = keep_audio and probe(source)['has_audio']
    sound = ['-map', '0:v:0', '-map', '0:a:0', '-c:a', 'aac', '-ac', '1', '-ar', '16000', '-b:a', '64k'] if has_audio else ['-an']
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', f'{start:.9f}', '-t', f'{end - start:.9f}', '-i', str(source),
         *sound, '-vf', filters, *(['-frames:v', str(frame_limit)] if frame_limit is not None else []), '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', *MP4_TIMESCALE, str(output)])
    return {'source_start': start, 'source_end': end, 'width': width, 'height': height, 'fps': fps, 'padded_tail': padded_tail,
            **({'has_audio': has_audio} if keep_audio else {})}


def make_proxy(original: Path, output: Path, width: int, crf: int, preset: str):
    temp = output.with_suffix(".partial.mp4")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin", "-y", "-i", str(original),
         "-map", "0:v:0", "-map", "0:a:0?", "-vf", f"scale=w='trunc(min(iw,{width})/2)*2':h=-2",
         "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(temp)])
    temp.replace(output)


def extract_frames(proxy: Path, frames_dir: Path, interval: float):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.jpg"):
        old.unlink()
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin", "-y", "-i", str(proxy),
         "-vf", f"fps=1/{interval},scale=w='min(iw,480)':h=-2", "-pix_fmt", "yuvj420p", "-threads", "1", "-q:v", "4",
         str(frames_dir / "frame_%06d.jpg")])
    frames = [{"name": frame.name, "time": round(index * interval, 3)}
              for index, frame in enumerate(sorted(frames_dir.glob("frame_*.jpg")))]
    (frames_dir / "manifest.json").write_text(json.dumps({"source": "proxy", "interval": interval, "frames": frames}))



def frame_manifest(interval: float, duration: float):
    count = max(1, int(duration / interval + .5)) if duration > 0 else 0
    frames = [{'name': f'frame_{index + 1:06d}.jpg', 'time': round(index * interval, 3)} for index in range(count)]
    return {'source': 'proxy', 'interval': interval, 'frames': frames}


def write_frame_manifest(frames_dir: Path, interval: float, duration: float):
    """Record sample times without retaining a JPEG for every interval."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / 'manifest.json').write_text(json.dumps(frame_manifest(interval, duration)))

def extract_audio(original: Path, output: Path):
    temp = output.with_suffix(".partial.wav")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-nostdin", "-y", "-i", str(original),
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temp)])
    temp.replace(output)


def waveform(audio: Path, output: Path, duration: float):
    bins = max(200, min(2400, round(duration * 2)))
    samples_per_bin = max(1, round(duration * 8000 / bins))
    peaks = []
    current_peak = 0
    count = 0
    with subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(audio),
                           "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        while chunk := proc.stdout.read(32768):
            values = array.array("h")
            values.frombytes(chunk)
            for value in values:
                current_peak = max(current_peak, abs(value))
                count += 1
                if count >= samples_per_bin:
                    peaks.append(round(current_peak / 32768, 3))
                    current_peak, count = 0, 0
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.wait():
            raise RuntimeError(stderr[-1800:])
    if count:
        peaks.append(round(current_peak / 32768, 3))
    output.write_text(json.dumps({"peaks": peaks, "duration": duration}))


def showinfo_times(lines):
    """Read full-precision presentation times from showinfo's integer PTS."""
    time_base = None
    for line in lines:
        if 'Parsed_showinfo' not in line:
            continue
        config = re.search(r'config in time_base:\s*(\d+)/(\d+)', line)
        if config:
            time_base = int(config.group(1)) / int(config.group(2))
        stamp = re.search(r'\bn:\s*\d+\s+pts:\s*(-?\d+)', line)
        if stamp and time_base is not None:
            yield int(stamp.group(1)) * time_base


def detect_cuts(proxy: Path, duration: float, threshold: float):
    starts = [0.0]
    tail = deque(maxlen=20)
    with subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin", "-i", str(proxy),
                           "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as process:
        assert process.stderr is not None
        def lines():
            for line in process.stderr:
                tail.append(line)
                yield line
        for time in showinfo_times(lines()):
            if starts[-1] + 0.15 < time < duration - 0.15:
                starts.append(time)
        if process.wait():
            raise RuntimeError("".join(tail)[-1800:])
    return [(start, end) for start, end in zip(starts, starts[1:] + [duration])]


def detect_cuts_in_range(proxy: Path, start: float, end: float, threshold: float):
    """Detect boundaries in a bounded interval, returning episode-relative times."""
    starts = [start]
    tail = deque(maxlen=20)
    with subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin",
                           "-ss", str(start), "-i", str(proxy), "-t", str(end - start),
                           "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as process:
        assert process.stderr is not None
        def lines():
            for line in process.stderr:
                tail.append(line)
                yield line
        for time in showinfo_times(lines()):
            timestamp = start + time
            if starts[-1] + 0.15 < timestamp < end - 0.15:
                starts.append(timestamp)
        if process.wait():
            raise RuntimeError("".join(tail)[-1800:])
    return [(left, right) for left, right in zip(starts, starts[1:] + [end])]


def extract_cut_frame(source: Path, output: Path, sample_time: float, start: float, end: float, width: int | None = 480, last: bool = False, quality: int | None = None, seek_tolerance: float = 0, probe_padding: float = 0, timeout: float | None = None, exact_size: tuple[int, int] | None = None):
    """Extract a real frame inside the cut; width=None retains source dimensions."""
    temp = output.with_name(f"{output.stem}.partial.jpg")
    try:
        if last:
            # Find the final presentation timestamp, including the last frame at
            # EOF. The bounded probe avoids scanning a long cut in normal cases.
            start_info = run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "default=noprint_wrappers=1:nokey=1", str(source)], timeout=timeout).stdout.strip()
            try:
                origin = float(start_info)
                if not math.isfinite(origin):
                    origin = 0.0
            except ValueError:
                origin = 0.0
            for probe_start in (max(start, end - 5), start):
                result = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{probe_start + origin}%{end + origin + probe_padding}", "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(source)], timeout=timeout)
                values = []
                for line in result.stdout.splitlines():
                    try:
                        value = float(line.split(",", 1)[0]) - origin
                    except ValueError:
                        continue
                    # FFprobe prints times to six decimals. A boundary frame
                    # rounded down must not be mistaken for an in-cut frame.
                    if start - .000001 <= value < end - .000001:
                        values.append(value)
                if values:
                    sample_time = max(values)
                    # Seeking to a rounded-up FFprobe timestamp skips that frame.
                    seek_tolerance = max(seek_tolerance, .000002)
                    break
                if probe_start == start:
                    break
        for back in (0, .1, .25, .5, 1, 2, end - start):
            seek = max(start, sample_time - back - seek_tolerance)
            temp.unlink(missing_ok=True)
            scale = f"scale={exact_size[0]}:{exact_size[1]}:force_original_aspect_ratio=decrease,pad={exact_size[0]}:{exact_size[1]}:(ow-iw)/2:(oh-ih)/2" if exact_size else (f"scale=w='min(iw,{width})':h=-2" if width is not None else None)
            filters = f"showinfo,{scale}" if scale else "showinfo"
            result = run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin", "-y", "-ss", f"{seek:.9f}", "-i", str(source),
                          "-t", f"{end - seek:.9f}", "-frames:v", "1", "-vf", filters,
                          "-pix_fmt", "yuvj420p", "-threads", "1", *(["-q:v", str(quality)] if quality is not None else []), str(temp)], timeout=timeout)
            if temp.is_file() and temp.stat().st_size:
                timestamps = showinfo_times(result.stderr.splitlines())
                actual = seek + next(timestamps, 0)
                if not last and actual >= end - .000001:
                    # A seek lands on the first frame at or after the time, so a time that rounds up past the cut's last
                    # frame returns the next cut's first picture. Take the cut's last frame instead.
                    temp.unlink(missing_ok=True)
                    return extract_cut_frame(source, output, sample_time, start, end, width=width, last=True, quality=quality, seek_tolerance=seek_tolerance, probe_padding=probe_padding, timeout=timeout, exact_size=exact_size)
                temp.replace(output)
                return round(actual, 6)
            if seek == start:
                break
        raise RuntimeError(f"Could not extract a frame inside cut {start:.3f}–{end:.3f}s")
    finally:
        temp.unlink(missing_ok=True)


def cut_frame_times(start: float, end: float) -> tuple[float, float]:
    return start, max(start, end - min(0.000001, (end - start) / 4))


def frame_timestamps_near(source: Path, boundary: float) -> list[float]:
    """Return presentation timestamps in a bounded window around a boundary."""
    info = json.loads(run(['ffprobe', '-v', 'error', '-show_entries', 'format=start_time',
                           '-of', 'json', str(source)], timeout=30).stdout)
    try:
        origin = float(info.get('format', {}).get('start_time') or 0)
        if not math.isfinite(origin):
            origin = 0
    except (TypeError, ValueError):
        origin = 0
    result = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-read_intervals',
                  f'{max(0, boundary - 3) + origin:.6f}%{boundary + 3 + origin:.6f}',
                  '-show_entries', 'stream=time_base:frame=best_effort_timestamp', '-of', 'json', str(source)], timeout=60)
    data = json.loads(result.stdout)
    try:
        numerator, denominator = map(int, data['streams'][0]['time_base'].split('/'))
        time_base = numerator / denominator
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError('Could not determine the video frame time base') from exc
    stamps = []
    for frame in data.get('frames', []):
        try:
            stamp = int(frame['best_effort_timestamp']) * time_base - origin
            if math.isfinite(stamp):
                stamps.append(stamp)
        except (KeyError, TypeError, ValueError):
            continue
    stamps = sorted(set(stamps))
    return stamps


def step_frame_boundary(source: Path, boundary: float, direction: int, *, after_stepped_frame: bool = False) -> float:
    """Step from a frame; optionally return the frame after the stepped frame."""
    stamps = frame_timestamps_near(source, boundary)
    if direction not in (-1, 1) or not stamps:
        raise ValueError('Could not find video frames near this boundary')
    nearest = min(range(len(stamps)), key=lambda index: abs(stamps[index] - boundary))
    target = nearest + direction + int(after_stepped_frame)
    if not 0 <= target < len(stamps):
        raise ValueError('No further frame is available in that direction')
    return stamps[target]


# --------------------------------------------------------------- export images and excerpts (ffmpeg only: the API image has no Pillow)

LABEL_FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def _label(textfile: Path, size: int) -> str:
    """drawtext reading the label from a file, so no character of it needs escaping in the filter graph."""
    font = f"fontfile='{LABEL_FONT}':" if LABEL_FONT.is_file() else ''
    return f"drawtext={font}textfile='{textfile}':expansion=none:x=8:y=8:fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6"


def tile_images(items: list[tuple[Path | None, str]], output: Path, columns: int, size: tuple[int, int] = (384, 216), timeout: float | None = 180):
    """One JPEG grid: each image scaled into a ``size`` tile with its label, in rows of ``columns``. A missing image is a black tile labelled missing."""
    if not items:
        raise ValueError('No images to tile')
    width, height = size
    columns = max(1, min(columns, len(items)))
    with tempfile.TemporaryDirectory(prefix='framewise-tiles-') as temp:
        args, chains = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y'], []
        for number, (path, label) in enumerate(items):
            if path is not None and Path(path).is_file() and Path(path).stat().st_size:
                args += ['-i', str(path)]
            else:
                args += ['-f', 'lavfi', '-i', f'color=c=black:s={width}x{height}:d=1']
                label = f'{label} (missing)'
            textfile = Path(temp) / f'label-{number}.txt'
            textfile.write_text(label, encoding='utf-8')
            chains.append(f'[{number}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuvj420p,'
                          f'{_label(textfile, max(12, height // 12))}[t{number}]')
        if len(items) == 1:
            graph = chains[0].replace('[t0]', '[out]')
        else:
            layout = '|'.join(f'{(index % columns) * width}_{(index // columns) * height}' for index in range(len(items)))
            graph = ';'.join(chains) + ';' + ''.join(f'[t{index}]' for index in range(len(items))) + f'xstack=inputs={len(items)}:layout={layout}:fill=black[out]'
        temp_output = output.with_name(f'{output.stem}.partial.jpg')
        try:
            run([*args, '-filter_complex', graph, '-map', '[out]', '-frames:v', '1', '-q:v', '3', str(temp_output)], timeout)
            temp_output.replace(output)
        finally:
            temp_output.unlink(missing_ok=True)


def scale_image(source: Path, output: Path, width: int, timeout: float | None = 60):
    """A smaller copy of a JPEG, never wider than ``width``."""
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(source), '-vf', f"scale=w='min(iw,{width})':h=-2", '-frames:v', '1', '-q:v', '3', str(output)], timeout)


def crop_image(source: Path, output: Path, box: tuple[float, float, float, float], timeout: float | None = 60, height: int | None = None):
    """Crop a JPEG to ``box`` given as fractions (x1, y1, x2, y2) of its width and height, at its own resolution or at most ``height`` lines high."""
    x1, y1, x2, y2 = (min(1.0, max(0.0, float(value))) for value in box)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ValueError('The crop box is empty')
    crop = f"crop=w='trunc({x2 - x1:.6f}*iw)':h='trunc({y2 - y1:.6f}*ih)':x='trunc({x1:.6f}*iw)':y='trunc({y1:.6f}*ih)'"
    if height:
        crop += f",scale=w=-2:h='min(ih,{int(height)})'"
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(source), '-vf', crop, '-frames:v', '1', '-q:v', '2', str(output)], timeout)


def cut_clip(source: Path, output: Path, start: float, end: float, height: int = 360, timeout: float | None = 300):
    """A silent H.264 excerpt of ``[start, end)``, at most ``height`` lines high."""
    if end <= start:
        raise ValueError('The clip interval is empty')
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-ss', f'{start:.6f}', '-i', str(source), '-t', f'{end - start:.6f}', '-an',
         '-vf', f"scale=w=-2:h='min(ih,{height})'", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], timeout)


def write_wav_ranges(audio: Path, ranges: list[dict], target):
    """The ranges of a WAV (episode seconds) one after another, written to a path or a binary file object. Raises ValueError for a range outside the audio."""
    with wave.open(str(audio), 'rb') as source, wave.open(target if not isinstance(target, Path) else str(target), 'wb') as out:
        out.setparams(source.getparams())
        for item in ranges:
            start = round(item['start_time'] * source.getframerate())
            end = min(source.getnframes(), round(item['end_time'] * source.getframerate()))
            if start >= source.getnframes() or end <= start:
                raise ValueError('Excerpt is outside the available audio')
            source.setpos(start)
            out.writeframes(source.readframes(end - start))


def pad_wav(source: Path, target: Path, seconds: float):
    """A copy of a WAV with silence appended up to ``seconds``."""
    with wave.open(str(source), 'rb') as audio, wave.open(str(target), 'wb') as out:
        out.setparams(audio.getparams())
        frames = audio.getnframes()
        out.writeframes(audio.readframes(frames))
        missing = max(0, math.ceil(seconds * audio.getframerate()) - frames)
        out.writeframes(b'\x00' * (missing * audio.getnchannels() * audio.getsampwidth()))
