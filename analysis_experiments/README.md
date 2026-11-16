# Pose-detail experiment

Question: can the models we already run (Qwen3.5-27B for the first frame, Qwen3-VL-8B for video) fill a structured
per-character block — body/head yaw, head pitch/roll, eye gaze, eye openness, gaze target, mouth, eyebrows, expression,
body pose, camera elevation, shot size — reliably enough to add it to the visual analysis?

Nothing in the app changes: no migration, no API, no edit under `backend/app`. The experiment reuses the app's prompts
(`visual_contracts`), the llama-server launcher (`frame_model.frame_server`), the video windowing
(`video_model.analyze_video`) and appends one JSON key, `pose_detail`, to each instruction.

## Run it

```bash
analysis_experiments/run.sh dry-run        # whole pipeline, fake models, CPU only (30 s); report at ?run=dry-run
analysis_experiments/run.sh start          # real run: needs an idle queue, pauses the inference worker (about 2.5-3 h)
analysis_experiments/run.sh status
analysis_experiments/run.sh stop           # stops the run; the worker is started again
analysis_experiments/run.sh start --run-id pose-20261116-1010    # same id again = resume (finished calls are skipped)
```

Report (same host and port as the app; use the full file name, the bare folder URL falls through to the app):
`http://192.168.0.6:8080/analysis-experiments/index.html` — add `?run=<id>` for a specific run. It refreshes every 15 s
while a run is active.

While a real run is active the **inference worker is stopped** (it keeps a Qwen model in VRAM even when idle, and a second GPU
process beside it once crashed this host's kernel). `run.sh start` waits for the VRAM to drain, arms a detached watcher
(`docker wait` → `docker start` of the worker) *before* it starts the container, so the worker comes back however the run ends.
Queued jobs simply wait. Each model stage is its own Python process, so GPU memory is fully released between models.

## Pipeline

`sample` → `frames` (3 approaches × 20 cuts × {normal, mirrored}) → `video` (3 × 20) → `score` → `conclude` (Qwen 3.6, 4 calls)

| File | Role |
|---|---|
| `pose_schema.py` | **Reusable**, stdlib only: fields, vocabularies, validators, prompt blocks, JSON schema, mirror maps |
| `sampler.py` | Read-only DB query → 20 cuts in 15 strata (short/long, 1–3+ characters, nobody marked, background people, no people) |
| `prompts.py`, `runners.py` | The three prompts, the retry loop, the frame and video stages, per-call checkpoints (`raw/*.jsonl`) |
| `scoring.py` | Pure: mirror / geometry / agreement checks, per-field stats, the score |
| `conclude.py`, `servers.py` | Digest → Qwen 3.6 → validated conclusion (retried with the validator's errors) |
| `fake_backend.py` | Canned answers (a quarter of first attempts broken on purpose) for `dry-run` and the tests |
| `experiment.py` | CLI and orchestration; writes `results.json` after every call |
| `../frontend/public/analysis-experiments/index.html` | The report (static page, no build) |

Output of a run: `frontend/public/analysis-experiments/runs/<run-id>/` (`samples.json`, `thumbs/`, `raw/`, `conclusions.json`,
`progress.json`, `results.json`, `log.txt`). It is gitignored; delete it when you are done.

## The three approaches

All append the same conventions (image left/right, yaw 0 = facing camera, positive = toward the image's right, pitch negative = looking down).
Abstaining is allowed per field (`unclear`, `null` for angles) and counted, not hidden.

1. **full_confidence** — all 13 fields + a 0.0–1.0 `confidence` per field; free integer angles.
2. **top5_confidence** — only the five fields the model is surest about per character, with confidence, best first.
3. **strict_detailed** — all fields, no confidence, explained value by value; 9 gaze sectors, angles on a 45° grid, `standing/walking/lying/sitting/running/other`.

Frames: `pose_detail` = one entry per marked character (`character_id` = box handle `B1…`). Video: one entry per character of the answer,
with `start` and `end` snapshots. Constrained decoding is **off** (so validator failures measure real instruction following);
`--grammar` turns on a JSON-schema grammar for the frame stage. A rejected answer is sent back with the validator's sentences, up to 3 attempts.

## Score (0–100)

Per stage (first frame, video), then 50/50:

* reliability 30 % — half first-attempt valid, half valid after retries
* quality 40 % — mean pass rate of **mirror** (frames only; a flipped frame must give negated yaw, swapped left/right gaze and roll, same everything else),
  **geometry** (who looks at whom vs. the boxes, camera gaze, shot size vs. box height, one elevation per frame, pitch vs. vertical gaze, head vs. body)
  and **agreement** with the stored free-text analysis of the same cut
* coverage 15 % — concrete values returned ÷ the 13 requested fields (top-5 is capped at 5/13 by design)
* efficiency 15 % — fastest approach's time ÷ this approach's time

There is no hand-labelled ground truth, so a high score means *consistent*, not proven correct; the report shows each frame next to the
outputs so a few cases can be judged by eye. Weights live in `scoring.WEIGHTS`; `experiment.py score --run-id …` recomputes `results.json` offline.

## Reuse the validators

Copy `pose_schema.py` anywhere: `validate_answer(pose_detail, approach, handles)` (frames), `validate_video_answer(...)`,
`prompt_block(approach, 'frame'|'video')`, `json_schema(approach, handles)`, `mirror_entry(...)`, and `format_errors(errors)` for the retry message.

## Tests

```bash
PYTHONPATH=backend python3 -B -m unittest discover -s analysis_experiments/tests -t .
```
