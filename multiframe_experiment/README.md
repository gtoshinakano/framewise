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
# the overnight pose run (2026-11-17): 4 Qwen setups, DWPose C, a hybrid D for each Qwen setup
multiframe_experiment/run.sh start --run-id mf-x --cuts 139-158 --resample --models qwen35 --styles observer inspector --contexts frames roster --pose \
  --merge-with 'qwen35|observer|frames' 'qwen35|observer|roster' 'qwen35|inspector|frames' 'qwen35|inspector|roster'
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
| `--pose` | off | add setup C `dwpose\|rules\|keypoints` and one D `hybrid\|<style>\|<context>` per `--merge-with` config (all CPU, see below) |
| `--merge-with` | `qwen35\|observer\|frames` | one or more Qwen configs D merges DWPose into |
| `--resample` | off | rebuild `samples.json` (frames already extracted are kept) |

Reports: `http://192.168.0.6:8080/multiframe-experiment/runs/<id>/report.html` (all runs: `…/runs/index.html`).
`report.md` next to it is the same leaderboard and findings as text for agents. While a real run is active the
**inference worker is stopped** (same discipline as `analysis_experiments/run.sh`); queued jobs wait, and a
detached watcher starts the worker again however the run ends.

## Pipeline

`prepare` (CPU: cuts, keyframes, exact frames, roster, boxes, transcript, baseline, pixel metrics) → [`pose`, CPU] →
`infer --model lfm3` → `infer --model qwen35` → [`merge`, CPU] → `judge` (Qwen3.5) → report. The bracketed stages run
only with `--pose`. Every GPU stage is its own process. Every call is a line in
`raw/*.jsonl`, and the report is rebuilt during the run (it refreshes itself every 30 s).

| File | Role |
|---|---|
| `sampler.py` | read-only DB: cuts, keyframes (cached or recomputed), frames, roster, first-frame text, transcript, baseline; windows |
| `measure.py` | cv2: camera estimate (ORB + RANSAC similarity transform) and change indices per pair and per character box |
| `prompts.py` | `STYLES` and `CONTEXTS` registries, message builder (each image preceded by `F<n> · t=… s`) |
| `schema.py` | shared answer schema, json_schema grammar, validator/coercion, atomic claims, baseline conversion |
| `infer.py` | llama-server client, retry loop (3 attempts with the validator's sentences), checkpoints |
| `judge.py` | blind judge: reference flag table + claim verdicts (true / false / not_visible) |
| `pose.py` | DWPose on the CPU (onnxruntime): person crops from the cut's boxes, YOLOX elsewhere, tracking, overlays, `pose.json` |
| `pose_rules.py` | stdlib: keypoints → per-frame states → the shared segment JSON (C), and the C + Qwen merge (D) |
| `scoring.py`, `report.py` | pure stdlib scoring, `results.json`, `report.html`, `report.md` |
| `fake_backend.py` | canned answers for `dry-run` and the tests |

## DWPose setups (`--pose`)

**Weights.** `.models/dwpose/` on the host, mounted read-only at `/models/dwpose`, downloaded once from the official
release https://huggingface.co/yzd-v/DWPose (model card licence: Apache-2.0):

| File | Role | Size | sha256 |
|---|---|---|---|
| `dw-ll_ucoco_384.onnx` | whole-body pose (133 COCO-WholeBody points, top-down 288×384, SimCC) | 134,399,116 | `724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843` |
| `yolox_l.onnx` | person detector (YOLOX-L, 640×640) | 216,746,733 | `7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411` |

```bash
mkdir -p .models/dwpose && cd .models/dwpose
for f in dw-ll_ucoco_384.onnx yolox_l.onnx; do curl -fL -o "$f" "https://huggingface.co/yzd-v/DWPose/resolve/main/$f"; done
sha256sum *.onnx
```

**C `dwpose|rules|keypoints`** (no language model). `pose.py` runs both ONNX models with `CPUExecutionProvider` only, so
it runs in the experiment container without touching the GPU (about 0.4 s wall per keyframe on this host). Person crops:
the cut's character boxes from the latest detection run (labelled or not) at the keyframe their frame matches (first
frame, and the last frame when the cut kept its end), YOLOX for every other keyframe and anyone the boxes miss. Tracks
follow boxes through the pixel camera transform (zoom, pan) and take a box's name. `pose_rules.py` turns the points into
per-frame states with the wording of `still_vocabulary.yaml`/`pose_detail` (arm elevation and elbow, hand region,
knee angles, sitting/crouching/lying/leaning, rough head and torso yaw, place in the frame) and compares consecutive
frames into the shared segment JSON. Limbs are named by their side of the screen. Blinking, speaking, mouth and
expression stay `unclear`; the camera is the pixel estimate. Overlays: `frames/<cut>/pose/F<n>.jpg`. The per-character
usability (share of keyframes with ≥ 6 of 17 body points over 0.3, mean confidence, suspected left/right swaps,
close-ups) is in the report.

**D `hybrid|<style>|<context>`** (no extra call; the first run named it `hybrid|dwpose+qwen|frames`). `merge` combines C with each `--merge-with` config: arms, hands, legs, body pose,
position and walking from DWPose wherever its points pass in both frames, everything else (head, eyes, mouth,
expression, speaking, blinking, other changes) from Qwen; the camera is the pixel estimate. Characters are paired by
name, else by the place words in Qwen's frame states against the DWPose box, else as the only pair left. D's cost is
A's plus C's. DWPose numbers are never put into a prompt.

## What every config answers

One JSON per window: one segment per consecutive frame pair with `camera.move`, per character `walking`, `speaking`,
`blinking` (yes/no/unclear) and `changes` (`part` + description), plus `other_changes` (outfit, colour, lighting,
objects, background, effects). The app adds each pair's times. Styles differ only in framing: `animator` asks how to
animate from frame to frame, `observer` asks what visibly changed (and allows "no change"), `inspector` (2026-11-17) is observer
with fixed state fields per frame (place; pose; head; eyes; mouth; arms; hands; legs) compared field by field, mouth changes as speaking,
whole-body moves as position/walking, and small background characters and entries/exits named explicitly. Contexts add the roster
and subtitle band (`roster`), then the first-frame description, the window's transcript and the previous window's
answer (`full`).

## Score (0–100)

| Component | Weight | Source |
|---|---|---|
| precision | 35 | judge verdicts: true ÷ (true + false); outfit/colour/lighting weigh 0.3, objects/background/effects 0.5 |
| recall | 20 | judge reference flags per pair and category that the answer also reports |
| camera | 0 | diagnostic only since the pose round: `camera.move` vs the pixel estimate, which the report shows as every setup's camera |
| speaking | 10 | speaking=yes vs transcript speech by a character marked on screen (a boxed name), precision + recall; speech by an unmarked speaker leaves that pair unscored; not scored for DWPose (no mouths); `full` is "told" |
| reliability | 10 | half valid on the first attempt, half valid after retries |
| efficiency | 15 | cheapest compute seconds per footage second ÷ this config's |

Missing components are left out and the rest renormalised (the baseline has no reliability). Also reported:
false-claim rate, not-visible rate, invented names, claims made on visually still pairs, tokens, decode speed,
per-category precision/recall, and a cost/quality frontier. `scoring.WEIGHTS` changes the weights;
`run.sh report <id>` recomputes offline.

**Caveats.** The judge is Qwen3.5-27B, the same model as one candidate. The camera estimate follows the dominant
transform, so a character who fills the frame can fool it. The baseline (stored Qwen3-VL-8B video analysis) is
mapped onto the pairs by its event times, and its camera is one value per cut.

## Ground truth

Make the labels **before** any model answer exists:

1. `python3 multiframe_experiment/experiment.py prepare --run-id <id> --cuts ...` in the container (CPU; see the
   dry-run command in `run.sh` for the mounts), or start from an earlier run's frames.
2. `python3 multiframe_experiment/experiment.py labels-template --run-id <id> --root frontend/public/multiframe-experiment/runs`
   writes `labels/<cut>.json`, every reference flag `false` per pair.
3. Look at `frames/<cut>/F<n>.jpg` (contact sheets: ffmpeg `xstack` in a container) and set, per pair, the flags that
   visibly change between the two keyframes (any character), the `camera` and a one-line `note`.
4. Run the experiment with the same `--run-id` (`--resample` keeps the extracted frames), then `run.sh report <id>`. Labels
   copied from another run apply only to pairs whose frame times match (`report.matching_labels`); a labelled window a setup
   failed to answer counts all its labelled changes as missed.

The report then ranks by the hand labels first: GT precision/recall/F1 per pair and category (`ground_truth` in
`results.json`), a body-only F1 (walking, arms & hands, legs, body pose, position), the per-category table, and how far
the judge agrees with the labels (its reference table per category, and per setup how many claim verdicts match).

## Tests

```bash
python3 -B -m unittest discover -s multiframe_experiment/tests -t multiframe_experiment
```
