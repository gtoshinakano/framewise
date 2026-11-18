# Hand-off: a motion pipeline from keyframes (what the experiments say to build next)

Written 2026-11-18 for the session that develops the next pipeline. It consolidates `FINDINGS.md` (chronological log of every
round, keep it for the evidence), `analysis_experiments/FINDINGS.md` (the earlier pose-detail experiment) and the decisions taken
along the way. Where a later round changed an earlier recommendation, this file states the later one.

Nothing from these experiments is in the app: `backend/app` is untouched by `multiframe_experiment/`. Everything below is a
finding about stand-alone code, or a proposal that is marked as one.

## 1. The question and where it stands

**Question.** The keyframe selector (`backend/app/keyframes.py`, `docs/keyframe-selection.md`) picks the frames where a cut changes.
Can the still-image models already installed describe what changes between those frames well enough to prompt MiniMax H3, instead
of running the video model (Qwen3-VL-8B) over the whole clip? Which prompt, which context, how to avoid hallucination, at what cost?

**Answer so far (hand labels, 14 cuts, 60 pairs, Episode 2).** Qwen3.5-27B on the keyframes, with DWPose (CPU) merged in for body
parts, finds more of what changes than the stored video analysis, at a higher compute cost per footage second:

| Setup | GT F1 (precision / recall) | Body-parts F1 | Compute per footage second |
|---|---|---|---|
| **Qwen3.5 observer/roster + DWPose hybrid (D)** | **61** (66% / 56%) | **64** | 21 s (DWPose adds 1.6 s per window on the CPU) |
| Qwen3.5 observer/roster alone (A) | 54 (70% / 44%) | 51 | 20 s |
| DWPose rules alone (C) | 39 (70% / 28%) | 50 | 0.7 s |
| Stored Qwen3-VL video analysis | 31 (40% / 25%) | 28 | 4.7–26 s depending on the cuts |

**Read these numbers with the caveats of section 6.** D's lead over A was built from changes chosen after reading both sets'
errors, so it is in-sample; the unchanged A column is the control. The score is agreement with hand labels of *visible change
categories*, never the quality of a generated H3 video: that link has not been tested at all.

## 2. The pipeline the evidence supports (build this, in this order)

1. **Frames.** The keyframe selector's frames, windows of 5 frames per call, consecutive windows share one frame, image width 1024.
2. **Qwen3.5-27B, `observer` prompt, `roster` context** (`prompts.py`; names and boxes of the characters marked in the first frame, subtitle
   band). One call per window with the `json_schema` grammar (`schema.py`: per-frame `states` first, then one segment per pair with
   per-character `walking`/`speaking`/`blinking` and `changes` by part, plus `other_changes`). Writing each frame's state first
   removed most copied changes. 47 of 47 calls in the last two runs were valid on the first attempt with the grammar. A call takes
   50-55 s for 5 frames, nearly all of it decoding about 1,400 tokens; shorter answers are the main cost lever.
3. **DWPose on the CPU** (`pose.py`, onnxruntime `CPUExecutionProvider`, weights in `.models/dwpose`, see `README.md`), person crops from
   the cut's detection boxes (names stay attached), YOLOX for the other keyframes and anyone the boxes miss, tracked through the
   camera transform. About 0.4 s wall per keyframe. It never touches the GPU.
4. **Rules** (`pose_rules.py`, stdlib) turn keypoints into per-frame states and per-pair changes in the same JSON (setup C), then the
   **merge** (`pose_rules.merge`, no extra model call) builds the hybrid D: DWPose's claim replaces Qwen's for a body part it measured
   as changed (arms/hands, legs, body pose, position, walking); where DWPose measured nothing Qwen's claim stays; head, eyes, mouth,
   speaking, blinking, expression and other changes are Qwen's.
5. **Camera from pixels only** (`measure.py`: ORB + RANSAC similarity transform on the picture without the subtitle band, and with every
   detected person masked out, `camera_without_people`). Never ask a model for the camera; give the measurement to whatever needs it.

Proposed, not built: store the result as an analysis result with its own contract name, model/revision, prompt and settings (house
rule: store provenance with each result), and an adapter that exposes the `direction` shape that `prompt_context`, the consistency
summary, the composition brief and the sequence evidence already read. Hook points: `single_cut_pipeline.py` (stages), `visual_models.py`
(roles), `prompt_context.py` (`video_view`), `consistency_summary.build_input`, `composition_brief`, `sequence_evidence.py`. Any change
there needs the usual restarts and doc updates listed in `AGENTS.md`.

## 3. Evidence, round by round

All run ids are archived in `archive/runs/` (section 8). "Judge" = a blind Qwen3.5-27B claim checker; "GT" = my hand labels.

| Run (cuts of Episode 2) | What was tested | Result |
|---|---|---|
| `mf-20261117-ep2` (139-148, stopped after 80 LFM3 calls) | LFM3-Omni-2B on the keyframes | copies the same change into every pair, misattributes actions, over-reports speaking (crying) |
| `mf-20261117-4cuts` (139, 145, 147, 151; 16 pairs) | 2 models x 2 styles x 3 contexts, judge | Qwen3.5 observer/frames 67.5 (88% / 53%) vs stored video analysis 37.1 (29% / 25%); LFM3 24-57 |
| `mf-20261117-pose` (139, 143, 144, 158; 16 pairs, first hand labels) | + DWPose C and hybrid D | GT F1 D 57, B 53, A 48, C 43, stored 19; the judge disagrees with the labels on DWPose claims (46-52% agreement) |
| `mf-20261117-ep2pose` (139-158; labels on 4 cuts) | + `inspector` style, hybrids of every Qwen setup, judge | D beats its Qwen setup in all four (+2 to +9 F1); `inspector` no better than `observer`; roster context = no invented names |
| `mf-20261118-c160-170` (160, 162-170; 44 pairs labelled **before** any answer) | new walking rule, held out, best setup only, no judge | A 55, D 55, C 37, stored 35; no false walks; walking recall 62% = Qwen's |
| `mf-20261118-final` (all 30 cuts; 14 labelled) | camera/size/position fixes and merge policy, CPU only | D 54 -> 61, body F1 53 -> 64, A unchanged at 54 (in-sample) |
| `mf-20261118-prompt` (the 14 labelled cuts) | `walker` prompt, `roster_cam` context | F1 -2 / +2 (Qwen), -2 / +1 (hybrid), walking found 8 -> 9 / 10 of 15: inside the noise, not adopted |

Hand-label detail on the 14 cuts (precision / recall), D against A: arms & hands 88/75 vs 100/40, legs 70/39 vs 75/17, position 77/67
vs 100/43, body pose 60/30 both, walking 80/53 both (8 of 15 pairs found, 2 false walks that are Qwen's own, 7 missed), speaking 67/100 both,
head 55/41 vs 73/38, eyes/gaze 9/25 both.

## 4. What the evidence says, ranked by how much it should change the design

1. **Qwen3.5-27B on keyframes beats the video model on the labels and on the judge** (precision 87-88% vs 29% on the judge; F1 54 vs 31 on
   the labels), because each claim is tied to a frame pair; the video analysis spreads one sentence over the clip and adds motion that
   does not happen. Cost per footage second is similar to higher (see section 1: the stored analysis' seconds vary 4.7-26 s per footage
   second with the cuts, Qwen on keyframes 18-22 s).
2. **The per-frame state block before the segments is the useful part** (accurate, reusable, stops most copying). The changes between two
   frames are better computed from states than written by the model; that idea is only half done (DWPose states are compared by rules, Qwen's are not).
3. **DWPose is cheap, precise on body parts and blind elsewhere.** 94-97% of person-frames have a usable body (at least 6 of 17 body points
   over 0.3), no left/right swaps seen, misses are detector misses (half-hidden people, upper-body crops). It reads no face, mouth or
   expression; its head yaw got no head change right (head stays with Qwen).
4. **The merge policy matters as much as the rules.** Replacing Qwen's claim wherever DWPose is "usable" deleted Qwen's correct arm, leg and
   position claims whenever DWPose measured nothing (the hybrid's F1 gain from 54 to 61 comes mostly from keeping them, plus the rule fixes).
5. **Walking is limited by what is in the picture.** The rule that works needs travel through the world (camera removed) *and* alternating
   ankles; it has no false walks in 60 pairs. It cannot see walkers whose legs are cropped (upper-body shots, a runner cut by the frame
   edge) or tiny ones (70 px): 7 of 15 walking pairs are missed by every setup. A prompt change recovered 1-2 of them and moved other claims.
6. **Camera: pixels, not models.** The model's own camera field has no signal (almost every pair is static; 86% agreement with the pixel
   estimate). The pixel estimate was dominated by moving faces in talking-heads cuts (four invented zooms/pans in one static cut); masking
   the people fixed that (wrong calls 5 -> 1 of 60 labelled pairs) at the price of more `unclear` (21 -> 26 of 148 pairs).
7. **Sitting, kicking, riding.** DWPose invents lower bodies for close-ups (hips in the bottom 8% of the picture, knees above the hips): never
   classify sitting/lying from those. A kick (foot above the hips) is not a walk. A bicycle is invisible to keypoints; Qwen's frame text
   ("riding", "sitting") is the gate in the merge. A seated character with visible legs and a camera pan that follows a walker are implemented
   and unit-tested on synthetic keypoints but **never measured on real cuts** (none in the labelled set).
8. **Speaking from the model is noisy** (recall 100%, precision 67% on the labels; crying counts as speaking). The transcript already says when
   someone speaks; the frames only need to say whose mouth moves. Not built.
9. **Cost is decode time.** A Qwen call is 50-55 s of which the image reading is a few seconds. A shorter answer format (states only, changes computed
   in code) is the main way to make the pipeline cheaper; untested.

## 5. Do not repeat these

- **LFM3-Omni-2B for change descriptions** (any prompt or context): recall 6-27%, precision at most 59%, invents people. Its per-frame
  *states* are decent on close-ups; that was never exploited.
- **`animator` style** (more recall, 78-82% precision vs 87-88% for `observer`); **`full` context** (first-frame text + transcript + previous answer: -15 points
  of precision, more tokens); **`inspector` style** (fixed state fields: no gain); **`frames`-only context** (invents names: 3 per 200 claims; `roster` 0).
- **Putting DWPose numbers into the Qwen prompt** (extra context lowered precision; the design is to merge afterwards, in code).
- **The Qwen judge as the ranking** for keypoint claims: it agrees with the hand labels on 67-84% of Qwen claims but only 46-59% of DWPose-only (C) claims, and its own
  reference table misses walking and most position changes. Use hand labels; use the judge only for Qwen-vs-Qwen comparisons.
- **Rescoring after the fact without saying so.** Several rule changes here were made after reading the errors; the held-out run
  (`mf-20261118-c160-170`) is the only clean measurement of the walking rule, and it was clean only until the later changes.
- Prompts: describe rules, never give filled example sentences (the model copies them); left/right always mean the viewer's.

## 6. How far to trust the numbers

- **Small and single-source:** 14 labelled cuts, 60 pairs, one series; categories have 4-30 labelled pairs, so one pair moves a category by 3-25 points. Differences under about 10 F1 points are not distinguishable.
- **The labels are mine**, written from the keyframes before any answer existed for runs `pose` and `c160-170` (`labels/`); strict (only change visible between the two keyframes), "walking" includes running and leaving the picture, "position" counts apparent movement but not camera-caused movement. Borderline slow drifts (cut 160: 6-9% of the width) were labelled as not counted, and several remaining false claims are exactly those. The labels were not changed to suit the rules.
- **In-sample changes.** Walking rule: tuned on cuts 139-158 and held out on 160-170. Size/position/enter-leave/arms/merge/camera masking: chosen after reading both sets' errors.
- **No noise floor.** No replicate of an identical run exists; the prompt test flips 9-14 claims per variant with no net direction, which looks like noise but was not measured against a re-run of the baseline.
- **The stored baseline** is whatever analysis was stored for those cuts (the video contract version was not distinguished) and its seconds vary by cut.
- **Qwen answers of the final run come from two earlier sessions** with the same prompt, window size, image width and settings (temperature 0); the 17 baseline prompts were verified byte for byte against the current code.
- **No H3 quality measurement.** Nothing here shows that better change lists make better generated video.

## 7. Open problems (known, not solved)

- Walking for cropped or tiny walkers (7 of 15 missed), and Qwen's false walks when a character is kicked or flung (cut 143).
- Position: slow drifts near the 6% threshold; tracker losses that become "leaves the frame" for a person at the picture edge (cuts 160, 170); a tiny figure standing up behind a gate read as "comes closer" (cut 167).
- Body pose recall 30% and legs 39% even in the hybrid; head recall 41%; eyes/gaze precision 9%.
- Camera `unclear` in 26 of 148 pairs (close-ups leave no backdrop to measure); the window boundary hides walking evidence in the neighbouring window.
- Sitting/following-pan rules are untested on real data.

## 8. Data and tools (all on disk, tracked under `multiframe_experiment/`, no images)

`frontend/public/multiframe-experiment/` is gitignored and may be cleaned; its useful, non-regenerable content is archived:

| Path | What |
|---|---|
| `archive/runs/<run id>/` | `config.json`, `samples.json`, `pose.json`, `labels/`, `raw/` (GPU answers of Qwen3.5, LFM3 and the judge), `report.md` of each run; 9 MB in all. Not archived because regenerable: results, reports as HTML, DWPose/hybrid answers (`experiment.py rules`), frames |
| hand labels | `archive/runs/mf-20261118-final/labels/<cut>.json`: 14 cuts, 60 pairs (139, 143, 144, 158, 160, 162-170); per pair the flags walking, speaking, blink, head, eyes_gaze, arms_hands, legs, body_pose, position, expression, the camera and a note |
| `archive/restore.sh <run id> [new id]` | copies an archived run back under `frontend/public/multiframe-experiment/runs/` for the CPU stages and reports |
| `tools/combine_runs.py` | merge runs (samples, keypoints, one Qwen config's answers, labels) into one run |
| `tools/error_lists.py <run dir> [--config ...] [-v] [-m]` | false claims and misses per body category against the labels |
| `experiment.py rules|camera|report|labels-template|prepare|infer|judge` | `rules` re-derives C and D from the stored keypoints (CPU), `camera` re-estimates the camera with people masked (CPU, cv2: run it in the inference image), `labels-template` writes label skeletons |
| `run.sh start|status|stop|dry-run|report` | GPU runs: stops the inference worker (queue must be idle), restarts it however the run ends; `dry-run` uses fake models on the CPU |
| `README.md` | pipeline, flags, scoring, DWPose weights (file names, sizes, sha256, source) and ground-truth protocol |

Replay without a GPU, for example:
```bash
multiframe_experiment/archive/restore.sh mf-20261118-final my-replay
python3 -B multiframe_experiment/experiment.py rules --run-id my-replay --root frontend/public/multiframe-experiment/runs --merge-with 'qwen35|observer|roster'
python3 -B multiframe_experiment/report.py frontend/public/multiframe-experiment/runs/my-replay
python3 -B multiframe_experiment/tools/error_lists.py frontend/public/multiframe-experiment/runs/my-replay -v
python3 -B -m unittest discover -s multiframe_experiment/tests -t multiframe_experiment      # 67 tests, host, stdlib
```
Re-running Qwen (GPU) needs the frames: `experiment.py prepare` rebuilds them from the database and is deterministic only while the
cached keyframe results and cuts are unchanged. **GPU rule:** never run a second GPU process beside the inference worker (a hand-run test once
crashed this host's kernel); `run.sh` handles it. Only DWPose and the camera estimate are CPU-only and safe beside the worker.

## 9. Suggested next steps (in order)

1. **A fresh labelled batch (about 15 cuts)** that contains what the labelled set lacks: walkers with visible legs, a seated character with visible legs, a camera pan that follows a
   walker, wide shots with small characters. Label first (`experiment.py labels-template`, then look at the keyframes), then run A and D only. This is the only fair test of the
   in-sample changes and of the untested rules.
2. **Replicate the baseline once** (about 15 minutes of GPU, 17 windows) to turn "inside the noise" into a number.
3. **Walking recall** needs another source of leg motion than DWPose ankles: optical flow inside the person's box between consecutive keyframes, or a zoomed person crop sent to DWPose/Qwen when
   the box is small, or Qwen's own walking call kept where DWPose says `unclear` (already the case). Try on the 7 missed pairs first.
4. **Shorter, cheaper answers**: states only from Qwen, changes computed in code (the first-round idea); measure time and F1 against the current answer.
5. **Speaking from transcript timing plus the mouth state** instead of a yes/no from the model.
6. **Link to the objective**: build H3 prompts from the new descriptions for a few cuts and compare with prompts from the stored video analysis (the sequence layer already
   renders and checks them); that is the experiment that decides whether the pipeline is worth integrating.
7. Only then the integration sketched in section 2 (contract name, provenance, adapter, `prompt_context` slimming, API/doc updates).

## 10. Related experiment

`analysis_experiments/` (pose-detail POC, 2026-11-16) produced the first-frame `pose_detail` block that is now in the app (see `AGENTS.md`) and the
rule "video snapshots later, as deltas". The keyframe experiment here is its successor for motion: its evidence that a still-image model on keyframes gives
per-pair, per-part changes supports that direction. Its caution still holds: no per-field confidence, no 45-degree grids, coercion before validation, one retry only for structural errors.
