import array
import json
import math
import re
import subprocess
from collections import deque
from pathlib import Path

from .db import DATA_DIR


def episode_dir(project_id: str, episode_id: str) -> Path:
    path = DATA_DIR / "projects" / project_id / "episodes" / episode_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(args: list[str]):
    result = subprocess.run(args, capture_output=True, text=True)
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


def detect_cuts(proxy: Path, duration: float, threshold: float):
    starts = [0.0]
    tail = deque(maxlen=20)
    with subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin", "-i", str(proxy),
                           "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) as process:
        assert process.stderr is not None
        for line in process.stderr:
            tail.append(line)
            match = re.search(r"Parsed_showinfo[^\n]*pts_time:\s*([0-9.]+)", line)
            if match:
                time = float(match.group(1))
                if starts[-1] + 0.15 < time < duration - 0.15:
                    starts.append(time)
        if process.wait():
            raise RuntimeError("".join(tail)[-1800:])
    return [(start, end) for start, end in zip(starts, starts[1:] + [duration])]


def extract_cut_frame(source: Path, output: Path, sample_time: float, start: float, end: float, width: int = 480, last: bool = False):
    """Extract a real frame inside the cut, returning its presentation timestamp."""
    temp = output.with_name(f"{output.stem}.partial.jpg")
    try:
        if last:
            # Find the final presentation timestamp, including the last frame at
            # EOF. The bounded probe avoids scanning a long cut in normal cases.
            start_info = run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "default=noprint_wrappers=1:nokey=1", str(source)]).stdout.strip()
            try:
                origin = float(start_info)
                if not math.isfinite(origin):
                    origin = 0.0
            except ValueError:
                origin = 0.0
            for probe_start in (max(start, end - 5), start):
                result = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{probe_start + origin}%{end + origin}", "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(source)])
                values = []
                for line in result.stdout.splitlines():
                    try:
                        value = float(line.split(",", 1)[0]) - origin
                    except ValueError:
                        continue
                    if start <= value < end:
                        values.append(value)
                if values:
                    sample_time = max(values)
                    break
                if probe_start == start:
                    break
        for back in (0, .1, .25, .5, 1, 2, end - start):
            seek = max(start, sample_time - back)
            temp.unlink(missing_ok=True)
            result = run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin", "-y", "-ss", str(seek), "-i", str(source),
                          "-t", str(end - seek), "-frames:v", "1", "-vf", f"showinfo,scale=w='min(iw,{width})':h=-2",
                          "-pix_fmt", "yuvj420p", "-threads", "1", str(temp)])
            if temp.is_file() and temp.stat().st_size:
                match = re.search(r"Parsed_showinfo[^\n]*\bn:\s*0\s+pts:.*?pts_time:\s*([-+0-9.eE]+)", result.stderr)
                actual = seek + float(match.group(1)) if match else seek
                temp.replace(output)
                return round(actual, 6)
            if seek == start:
                break
        raise RuntimeError(f"Could not extract a frame inside cut {start:.3f}–{end:.3f}s")
    finally:
        temp.unlink(missing_ok=True)


def cut_frame_times(start: float, end: float) -> tuple[float, float]:
    return start, max(start, end - min(0.000001, (end - start) / 4))
