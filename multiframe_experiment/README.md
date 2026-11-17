# Multi-frame change experiment

Question: instead of sending whole clips to the video model, can the still-image models we already run
(**Qwen3.5-27B** and **LFM3-Omni-2B**) describe the motion between the keyframe selector's target frames well
enough to prompt MiniMax H3? Which prompt and which context do that best, without hallucinating, and at what cost?

Nothing in the app changes: no migration, no API, no edit under `backend/app`. The experiment reads the database
read-only and reuses `keyframes`, `frame_cache`, `frame_model.frame_server`, `visual_models` and `prompt_context`.

## Run it

```bash
multiframe_experiment/run.sh dry-run                        # whole pipeline, fake models, CPU only (~1 min)
multiframe_experiment/run.sh start                          # real run: needs an idle queue, pauses the inference worker
multiframe_experiment/run.sh start --run-id mf-x --cuts 200-210 --models lfm3 --contexts roster full
multiframe_experiment/run.sh status
multiframe_experiment/run.sh stop                           # the worker is started again
multiframe_experiment/run.sh start --run-id mf-x            # same id = resume (finished calls are skipped)
multiframe_experiment/run.sh report [run-id]                # rebuild report.html/report.md on the host (stdlib python3)
```

Experiment arguments (after `start` or `dry-run`):

| Flag | Default | Meaning |
|---|---|---|
| `--episode` | `Episode 2` | title or id |
| `--cuts` | `139-158` | UI cut numbers, `12,40-45` works |
| `--models` | `lfm3 qwen35` | keys of `infer.MODELS` |
| `--styles` | `animator observer` | keys of `prompts.STYLES` |
| `--contexts` | `frames roster full` | keys of `prompts.CONTEXTS` |
| `--window` | `5` | frames per call; the next call starts on the previous last frame |
| `--image-width` | `1024` | frame width sent to the models |
| `--kf key=value` | — | keyframe selector override (`budget`, `event_threshold`, …); recomputes the selection on the CPU instead of reading `keyframe_results` |
| `--no-judge`, `--judge-parallel N` | judge on, 2 | the Qwen3.5 judge stage |
| `--no-grammar` | grammar on | `json_object` instead of the `json_schema` grammar |

Reports: `http://192.168.0.6:8080/multiframe-experiment/runs/<id>/report.html` (all runs: `…/runs/index.html`).
`report.md` next to it is the same leaderboard and findings as text for agents. While a real run is active the
**inference worker is stopped** (same discipline as `analysis_experiments/run.sh`); queued jobs wait, and a
detached watcher starts the worker again however the run ends.

## Pipeline

`prepare` (CPU: cuts, keyframes, exact frames, roster, transcript, baseline, pixel metrics) → `infer --model lfm3` →
`infer --model qwen35` → `judge` (Qwen3.5) → report. Every GPU stage is its own process. Every call is a line in
`raw/*.jsonl`, and the report is rebuilt during the run (it refreshes itself every 30 s).

| File | Role |
|---|---|
| `sampler.py` | read-only DB: cuts, keyframes (cached or recomputed), frames, roster, first-frame text, transcript, baseline; windows |
| `measure.py` | cv2: camera estimate (ORB + RANSAC similarity transform) and change indices per pair and per character box |
| `prompts.py` | `STYLES` and `CONTEXTS` registries, message builder (each image preceded by `F<n> · t=… s`) |
| `schema.py` | shared answer schema, json_schema grammar, validator/coercion, atomic claims, baseline conversion |
| `infer.py` | llama-server client, retry loop (3 attempts with the validator's sentences), checkpoints |
| `judge.py` | blind judge: reference flag table + claim verdicts (true / false / not_visible) |
| `scoring.py`, `report.py` | pure stdlib scoring, `results.json`, `report.html`, `report.md` |
| `fake_backend.py` | canned answers for `dry-run` and the tests |

## What every config answers

One JSON per window: one segment per consecutive frame pair with `camera.move`, per character `walking`, `speaking`,
`blinking` (yes/no/unclear) and `changes` (`part` + description), plus `other_changes` (outfit, colour, lighting,
objects, background, effects). The app adds each pair's times. Styles differ only in framing: `animator` asks how to
animate from frame to frame, `observer` asks what visibly changed (and allows "no change"). Contexts add the roster
and subtitle band (`roster`), then the first-frame description, the window's transcript and the previous window's
answer (`full`).

## Score (0–100)

| Component | Weight | Source |
|---|---|---|
| precision | 35 | judge verdicts: true ÷ (true + false); outfit/colour/lighting weigh 0.3, objects/background/effects 0.5 |
| recall | 20 | judge reference flags per pair and category that the answer also reports |
| camera | 10 | `camera.move` vs the pixel estimate (1 exact, 0.5 same family, `unclear` not scored) |
| speaking | 10 | speaking=yes vs transcript speech in the pair interval (precision + on-screen recall); `full` is "told" |
| reliability | 10 | half valid on the first attempt, half valid after retries |
| efficiency | 15 | cheapest compute seconds per footage second ÷ this config's |

Missing components are left out and the rest renormalised (the baseline has no reliability). Also reported:
false-claim rate, not-visible rate, invented names, claims made on visually still pairs, tokens, decode speed,
per-category precision/recall, and a cost/quality frontier. `scoring.WEIGHTS` changes the weights;
`run.sh report <id>` recomputes offline.

**Caveats.** The judge is Qwen3.5-27B, the same model as one candidate. The camera estimate follows the dominant
transform, so a character who fills the frame can fool it. The baseline (stored Qwen3-VL-8B video analysis) is
mapped onto the pairs by its event times, and its camera is one value per cut.

## Ground truth (optional)

`python experiment.py labels-template --run-id <id>` (inside the container, or with `--root` on the host) writes
`labels/<cut>.json` with every reference flag `false` per pair. Set the flags that really change (a person or an
agent looking at `frames/<cut>/F<n>.jpg`), then `run.sh report <id>`. The leaderboard gains a true
precision/recall (`ground_truth` in `results.json`).

## Tests

```bash
python3 -B -m unittest discover -s multiframe_experiment/tests -t multiframe_experiment
```
