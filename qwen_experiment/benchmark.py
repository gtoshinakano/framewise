"""Benchmark the app's first-frame analysis on Qwen 3.6 vs the current Qwen 3.5.

Runs inside the inference-worker image (GPU, models and /data mounted, backend/app importable).

For a fixed set of Episode 2 first frames it rebuilds, from each cut's stored first-frame
provenance, the exact message the app sent to the first-frame stage (the same instruction, JSON
schema, marked-character handles, subtitle language and strip, and series notes), then asks each
model the same question and records inference time and the raw answer, so both runs are compared
on identical inputs. The stored Qwen3.5 timing for each cut is kept as a fourth column.

Both models are started one at a time (the 20-30 GB GGUFs do not share this GPU) and reused across
the frames at parallel=1, so a recorded time is one isolated inference, unlike the app's batched
FRAME_PARALLEL run.

    python3 /work/benchmark.py --episode "Episode 2" --count 30 --out /work
    python3 /work/benchmark.py --smoke          # one frame, last model only, checks the mmproj
    python3 /work/benchmark.py --dry            # rebuild every input, extract frames, no model
"""
import argparse
import base64
import json
import sqlite3
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.environ.get("APP_PATH", "/app"))
from app import frame_model  # noqa: E402
from app import media  # noqa: E402
from app import visual_contracts as vc  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "app.sqlite3"
RUNTIME = Path(os.environ.get("QWEN35_SERVER_PATH", "/models/llamacpp/llama-server"))

# The current first-frame model, exactly as the app configures it.
M35_MODEL = Path(os.environ.get("QWEN35_MODEL_PATH", "/models/qwen3.5-27b/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf"))
M35_PROJECTOR = Path(os.environ.get("QWEN35_PROJECTOR_PATH", "/models/qwen3.5-27b/mmproj/mmproj-Qwen3.5-27B-BF16.gguf"))
# Qwen 3.6 with its vision projector, downloaded next to the GGUF we already had.
M36_MODEL = Path(os.environ.get("QWEN36_FRAME_MODEL_PATH", "/models/qwen3.6-35b/Qwen3.6-35B-A3B-Q4_K_M.gguf"))
M36_PROJECTOR = Path(os.environ.get("QWEN36_FRAME_PROJECTOR_PATH", "/models/qwen3.6-35b/mmproj-Qwen3.6-35B-A3B-BF16.gguf"))

MODELS = {
    "qwen35": {"key": "qwen35", "label": "Qwen3.5-27B Q4_K_M", "note": "The current first-frame model.",
               "model_id": frame_model.MODEL_ID, "model": M35_MODEL, "projector": M35_PROJECTOR},
    "qwen36": {"key": "qwen36", "label": "Qwen3.6-35B-A3B Q4_K_M", "note": "Same first-frame task through the downloaded vision projector.",
               "model_id": "ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M", "model": M36_MODEL, "projector": M36_PROJECTOR},
}

REQUEST_TIMEOUT = 600


def log(message):
    print(message, flush=True)


# -- reconstructing the app's first-frame inputs from stored provenance ------------------------

def load_frame_runs(conn, episode_id):
    """Every cut with a current first-frame result, plus the provenance the app recorded with it."""
    rows = conn.execute("""
        select ct.id as cut_id, ct.cut_index, ct.title, ct.start_time, ct.end_time, ct.no_characters,
               ar.model_id, ar.created_at, ar.settings_json, ar.context_json
        from analysis_results ar join cuts ct on ar.cut_id = ct.id
        where ct.episode_id = ? and ct.active = 1 and ar.kind = 'first_frame' and ar.superseded_by is null
        order by ct.cut_index""", (episode_id,)).fetchall()
    runs = []
    for row in rows:
        item = dict(row)
        try:
            item["settings"] = json.loads(item["settings_json"] or "{}")
            item["provenance"] = json.loads(item["context_json"] or "{}")
        except json.JSONDecodeError:
            item["settings"] = item["provenance"] = {}
        runs.append(item)
    return runs


def select_cuts(runs, count):
    """A spread across the timeline: evenly spaced cuts that have a first-frame analysis."""
    if count >= len(runs):
        return list(runs)
    step = (len(runs) - 1) / (count - 1)
    return [runs[round(i * step)] for i in range(count)]


def frame_canvas(episode, width=1280):
    """Mirror of single_cut_pipeline.frame_canvas: the size a first frame is extracted at."""
    source_width, source_height = episode.get("width") or width, episode.get("height") or round(width * 9 / 16)
    scaled = min(source_width, width)
    return scaled, max(2, round(source_height * scaled / source_width / 2) * 2)


def strip_box(strip, width, height):
    """Mirror of single_cut_pipeline.strip_box: the subtitle strip as a pixel box, or None."""
    if not strip or strip.get("top") is None:
        return None
    return [0, round(strip["top"] * height), width, round(strip["bottom"] * height)]


def marked_people(character_detections, no_characters):
    """Mirror of single_cut_pipeline.marked_people for position 'first'.

    Returns None when there is no first-frame detection data (different from "nobody was marked"),
    so the pipeline treats the two cases the same way."""
    if no_characters:
        return []
    if not isinstance(character_detections, dict):
        return None
    frames = character_detections.get("frames") or []
    if not any(frame.get("position") == "first" for frame in frames):
        return None
    size = next((frame.get("image_size") for frame in frames if frame.get("position") == "first"), None)
    detections = [d for d in (character_detections.get("detections") or [])
                  if d.get("position") == "first" and d.get("character_context_id")]

    def left_edge(detection):
        box = detection.get("box")
        return box[0] if isinstance(box, list) and box else 0

    people = []
    for number, detection in enumerate(sorted(detections, key=left_edge), 1):
        box = detection.get("box")
        grid = None
        if isinstance(box, list) and len(box) == 4 and isinstance(size, list) and len(size) >= 2 and size[0] and size[1]:
            grid = [round(box[0] * 1000 / size[0]), round(box[1] * 1000 / size[1]),
                    round(box[2] * 1000 / size[0]), round(box[3] * 1000 / size[1])]
        people.append({"ref": f"B{number}", "character_context_id": detection["character_context_id"],
                       "name": detection.get("character_name") or "unnamed", "box": grid})
    return people


def build_sample(conn, episode, run, original):
    """Rebuild one cut's first-frame inputs exactly as single_cut_pipeline._frame_stage would."""
    provenance = run["provenance"]
    hints = provenance.get("model_context_hints") or {}
    strip = provenance.get("subtitle_strip")
    cut = {"no_characters": run["no_characters"]}
    people = marked_people(hints.get("character_detections"), cut["no_characters"]) or []
    canvas = frame_canvas(episode)
    box = strip_box(strip, *canvas)
    regions = [box] if box else []
    subtitle_language = (strip or {}).get("language")

    scratch = original.parent / "qwen_experiment_scratch"
    scratch.mkdir(exist_ok=True)
    image = scratch / f"first-{run['cut_id']}.jpg"
    first, _ = media.cut_frame_times(run["start_time"], run["end_time"])
    actual = media.extract_cut_frame(original, image, first, run["start_time"], run["end_time"], width=canvas[0])

    parts = ["This image is the first frame of the cut.", vc.person_context(people, "first frame")]
    note = vc.series_note(hints)
    if note:
        parts.append(note)
    parts.append(vc.subtitle_note(subtitle_language, regions, "image pixels").strip())
    messages = [{"role": "system", "content": vc.FRAME_INSTRUCTION},
                {"role": "user", "content": [{"type": "image_url",
                                              "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")}},
                                             {"type": "text", "text": "\n\n".join(part for part in parts if part)}]}]
    schema = vc.frame_schema([person["ref"] for person in people])
    settings = run["settings"]
    return {"cut_id": run["cut_id"], "cut_index": run["cut_index"], "title": run["title"] or "",
            "start_time": run["start_time"], "end_time": run["end_time"], "frame_time": actual,
            "image": f"frames/cut-{run['cut_index']:04d}.jpg",
            "marked_characters": len(people), "people": [{"ref": p["ref"], "name": p["name"]} for p in people],
            "subtitle_language": subtitle_language, "subtitle_regions": regions,
            "existing_model_id": run["model_id"], "existing_created_at": run["created_at"],
            "existing_seconds": settings.get("generation_seconds"), "existing_usage": settings.get("usage"),
            "messages": messages, "schema": schema, "image_bytes": image.read_bytes()}


# -- the request (mirrors frame_model.describe_frame, but keeps the raw answer) ------------------

def _chat_body(messages, schema):
    response_format = {"type": "json_schema",
                       "json_schema": {"name": "frame", "strict": True, "schema": schema}} if schema else {"type": "json_object"}
    return {"messages": messages, "temperature": 0, "max_tokens": frame_model.FRAME_MAX_TOKENS,
            "response_format": response_format, "chat_template_kwargs": {"enable_thinking": False}}


def _post(server_url, body):
    request = urllib.request.Request(server_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - started


def ask_frame(server_url, messages, schema):
    """One first-frame answer with its inference time; mirrors the app's grammar fallback."""
    try:
        payload, latency = _post(server_url, _chat_body(messages, schema))
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 422, 500):
            raise
        payload, latency = _post(server_url, _chat_body(messages, None))
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    choice = payload["choices"][0]
    return {"content": choice["message"]["content"], "usage": usage, "latency_seconds": latency,
            "finish_reason": choice.get("finish_reason"), "truncated": choice.get("finish_reason") == "length"}


# -- the run --------------------------------------------------------------------

def write_results(out, meta, frames):
    (out / "results.json").write_text(json.dumps({"meta": meta, "frames": frames}, ensure_ascii=False, indent=2))


def _meta(episode, canvas, strip, keys):
    models = {key: {name: (str(value) if isinstance(value, Path) else value)
                    for name, value in MODELS[key].items()} for key in keys}
    return {"episode": {"id": episode["id"], "title": episode["title"], "project_id": episode["project_id"]},
            "canvas": list(canvas), "subtitle": strip, "models": models,
            "runtime": str(RUNTIME), "request_timeout": REQUEST_TIMEOUT, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def _public(frame):
    keys = ("cut_id", "cut_index", "title", "start_time", "end_time", "frame_time", "image", "marked_characters",
            "people", "subtitle_language", "subtitle_regions", "existing_model_id", "existing_created_at",
            "existing_seconds", "existing_usage", "answers")
    return {key: frame[key] for key in keys if key in frame}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="Episode 2")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--out", default="/work")
    parser.add_argument("--models", default="qwen35,qwen36")
    parser.add_argument("--smoke", action="store_true", help="one frame, last model only")
    parser.add_argument("--dry", action="store_true", help="rebuild inputs and extract frames, no model")
    parser.add_argument("--context-per-slot", type=int, default=8192)
    parser.add_argument("--keep-frames", action="store_true", help="keep extracted frames in /work/frames")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    episode = conn.execute("select id, project_id, title, original_ext, width, height from episodes "
                           "where title = ? order by created_at", (args.episode,)).fetchone()
    if not episode:
        raise SystemExit(f"No episode titled {args.episode!r}")
    episode = dict(episode)
    runs = load_frame_runs(conn, episode["id"])
    selected = select_cuts(runs, args.count)
    original = media.episode_dir(episode["project_id"], episode["id"]) / f"original{episode['original_ext']}"
    if not original.is_file():
        raise SystemExit(f"Original video missing: {original}")

    keys = args.models.split(",")
    if args.smoke:
        keys = keys[-1:]
        selected = selected[:1]
    canvas = frame_canvas(episode)
    strip = selected[0]["provenance"].get("subtitle_strip") if selected else None
    log(f"episode={episode['title']} id={episode['id'][:8]} canvas={canvas} "
        f"subtitle={'yes' if strip else 'no'} eligible={len(runs)} frames={len(selected)} models={keys}")

    samples = []
    for run in selected:
        sample = build_sample(conn, episode, run, original)
        sample["image_bytes_saved"] = sample["image_bytes"]
        if args.keep_frames:
            (frames_dir / sample["image"].split("/")[-1]).write_bytes(sample["image_bytes"])
        samples.append(sample)
        log(f"  prepared cut {run['cut_index']} at {sample['frame_time']:.3f}s marked={sample['marked_characters']}")
    if args.dry:
        write_results(out, _meta(episode, canvas, strip, keys), [_public(s) for s in samples])
        log(f"dry run: prepared {len(samples)} inputs, no model called; results.json written")
        return

    for key in keys:
        spec = MODELS[key]
        for path in (RUNTIME, spec["model"], spec["projector"]):
            if not path.is_file():
                raise SystemExit(f"Missing file for {key}: {path}")

    # One server at a time: both GGUFs are too large to share the GPU.
    for key in keys:
        spec = MODELS[key]
        log(f"starting {spec['label']} server ...")
        started_at = time.monotonic()
        with frame_model.frame_server(RUNTIME, spec["model"], spec["projector"], parallel=1,
                                      context_per_slot=args.context_per_slot) as server_url:
            log(f"  server ready in {time.monotonic() - started_at:.1f}s at {server_url}")
            for sample in samples:
                entry = {"model": key}
                try:
                    entry.update(ask_frame(server_url, sample["messages"], sample["schema"]))
                except Exception as exc:  # a bad answer still belongs in the comparison
                    entry.update({"content": None, "error": f"{type(exc).__name__}: {exc}",
                                  "latency_seconds": None, "usage": {}, "finish_reason": None, "truncated": False})
                sample["answers"] = {**sample.get("answers", {}), key: entry}
                seconds = entry["latency_seconds"]
                detail = f"{seconds:.1f}s" if seconds is not None else str(entry.get("error"))
                log(f"  {key} cut {sample['cut_index']}: {detail}")
                write_results(out, _meta(episode, canvas, strip, keys), [_public(s) for s in samples])
    log("done; results.json written")


if __name__ == "__main__":
    main()
