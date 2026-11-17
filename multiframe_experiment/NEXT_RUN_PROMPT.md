Plan, build and run the next round of the multi-frame experiment in `multiframe_experiment/` (repo `/home/bytoshi/app-projects/video-analyzer`). Run it on Episode 2, cuts 139, 143, 144 and 158 (UI cut numbers; `cut_index` = number − 1), with **Qwen3.5-27B only**, plus a new DWPose body-pose hybrid. Build it, dry-run it, run it, and come back with a written verdict.

## Read first
- `multiframe_experiment/README.md` and `multiframe_experiment/FINDINGS.md`: the previous verdict.
- The earlier reports: `frontend/public/multiframe-experiment/runs/mf-20261117-4cuts/report.md` and `runs/mf-20261117-ep2/report.md`.
- `AGENTS.md`, especially the experiment bullets and the GPU hazard.

## What is already known (do not re-test)
- **Qwen3.5-27B, observer prompt, `frames` or `roster` context, is the best setup.** In run `mf-20261117-4cuts` it had about 87–88% judge precision and 50–53% recall, at about 55 s per 5-frame call. That is the same cost per footage second as the stored Qwen3-VL video analysis, which had 29% precision.
- **The per-frame `frames` states block, written before the segments, stops most copied changes. Keep it.**
- **Dropped:**
  - LFM3 entirely;
  - the `animator` style (lower precision);
  - the `full` context (lowered Qwen precision by about 15 points).
- **Camera from the models gave no signal:** almost every pair was static. Camera moves come from the pixel estimate (`measure.py`), not the model.
- **Speaking from the model is noisy.** Speech comes from the transcript; the frames only say whose mouth moves.
- **Fixes that are already in place; keep them:**
  - a duplicate character is merged instead of rejected;
  - "no change / remains / unchanged" statements are not claims (`schema.is_no_change`);
  - `maxItems`/`maxLength` in the grammar;
  - a retry resends at most 1500 characters of a cut-off answer.
- **Weak spots to attack:** recall (about 50%), body-part attribution, walking, position.

## Setups for this run
- **A `qwen35|observer|frames`:** the reference, unchanged.
- **B `qwen35|observer|roster`:** unchanged.
- **C `dwpose|rules`:** no language model.
  - Run DWPose whole-body on every keyframe of each window.
  - Turn the keypoints into per-frame states with rules: arm raised or lowered (shoulder-elbow-wrist angles), hand position relative to the head and torso, standing / sitting / crouching / lying / leaning from hip-knee-ankle, rough head turn from nose, eyes and ears, and position in frame.
  - Compare consecutive keyframes to produce the same segment JSON as `schema.py`: `walking`, plus `changes` with parts arms, hands, legs, body_pose, head, position.
  - Use the wording of `backend/app/still_vocabulary.yaml` and `pose_detail.py` phrases where they fit; keep the screen-left/screen-right convention.
  - Leave blinking, mouth, expression and speaking empty or "unclear": DWPose's face points are unreliable on anime.
- **D `hybrid|dwpose+qwen`:** merge C and A.
  - Body parts (arms, hands, legs, body_pose, position, walking) come from DWPose where its keypoints pass the confidence threshold. Otherwise use Qwen's.
  - Eyes, mouth, expression and other changes come from Qwen.
  - The camera comes from the pixel estimate.
  - No extra Qwen call: D is computed from the A and C answers.

Do not feed DWPose numbers into the Qwen prompt as context; extra context hurt precision last time. This run measures merging, not prompting.

## DWPose setup
- **Download approved:** only the DWPose ONNX weights, the whole-body pose model and its person detector, about 300 MB. Get them from the official DWPose release on Hugging Face (`yzd-v/DWPose`) into `.models/dwpose/` on the host. Record the file names, sha256 and source URL, and check the weights' license; report it.
- **Run it on the CPU** with onnxruntime, in the `framewise-inference-rocm:local` image (onnxruntime 1.23.2 is installed). It must not use the GPU, so it can run in the `prepare` stage without the worker paused. Mount the folder read-only.
- **Top-down pose:** use the cut's character boxes from the latest detection run (labelled or not) as person crops, so names stay attached.
  - Use DWPose's own detector only where there are no boxes, and for frames after the first.
  - Match people across keyframes by box overlap and the nearest centre.
  - Cut 144 has no boxes and no stored analyses (detector only, no baseline).
  - In cut 158 the Police Officer is boxed in the first frame; Tomoe Koga and Sakuta Azusagawa are boxed only in the last frame. The cut's frames may need both box sets.
- **Overlays:** draw each keyframe with its skeleton and confidences into `frames/<cut>/pose/`, and show them in the report's per-cut view.
- **Per character, report:**
  - the share of keyframes with a usable body (at least 6 body keypoints above the threshold);
  - average confidence;
  - suspected left/right swaps (a wrist jumping sides with an unchanged box).

## Ground truth, made before seeing any model output
1. Run `labels-template` for the new run.
2. Look at every keyframe of the 4 cuts yourself (open `frames/<cut>/F<n>.jpg`; make contact sheets with ffmpeg in a container) and fill `labels/<cut>.json` for every pair:
   - the reference flags: walking, speaking (mouth moving), blink, head, eyes_gaze, arms_hands, legs, body_pose, position, expression;
   - the camera;
   - a one-line note.
3. Do this **before** looking at A–D answers, and be strict: only changes visible between the two keyframes.

The report then shows true precision and recall (`ground_truth` in `results.json`) next to the judge's. Extend `scoring.py` so these accuracy numbers appear as their own leaderboard columns, per category too, and so the findings rank setups by them first.

## Other improvements for this run
- **Camera:** leave it out of the score (weight 0), keep it as a diagnostic, and make the pixel estimate the camera of every setup in the report.
- **Speaking:**
  - score `speaking=yes` against the transcript only for a character marked on screen;
  - leave it out of the score for setups that cannot see mouths (C);
  - D takes Qwen's.
- **Efficiency:** count DWPose CPU seconds per window as C's cost and add them to D's.
- **Baseline:** keep the stored Qwen3-VL video analysis row where it exists (not cut 144).
- **The judge stays Qwen3.5-27B.** Report agreement between the judge and the hand labels, so its bias is measured.

## How to run
1. **Code:** keep everything in `multiframe_experiment/`; no change under `backend/app`. Add the DWPose code as its own module, e.g. `pose.py` and `pose_rules.py`, and register C and D as configs so `--models qwen35 --styles observer --contexts frames roster` plus a pose flag selects exactly A–D.
2. **Tests:** add unit tests for the rules (synthetic keypoints → phrases and changes) and for the merge. Run:
   ```
   python3 -B -m unittest discover -s multiframe_experiment/tests -t multiframe_experiment
   ```
3. **Dry run:** `multiframe_experiment/run.sh dry-run --cuts 139,143,144,158 ...`. Open the report's HTML and check that it renders: run the page script through node in the `video-analyzer-frontend-watch-1` container.
4. **Real run:** `multiframe_experiment/run.sh start --run-id mf-<date>-pose --cuts 139,143,144,158 --resample ...`.
   - It needs an idle queue. It stops the inference worker, and a watcher starts it again.
   - Expect about 15–20 minutes: 2 Qwen setups × about 5 windows × about 55 s, plus the judge.
   - Watch the first Qwen answers and the pose overlays. If something clearly degrades the run, stop with `run.sh stop` and tell me before running again.
5. **Never** run ad-hoc GPU tests next to the worker (see the GPU hazard in AGENTS.md).

## Deliver
- Add a "Pose hybrid" section to `multiframe_experiment/FINDINGS.md`:
  - a ground-truth and judge table for A–D and the baseline;
  - per-category results for body parts;
  - DWPose usability on anime (the share of frames with usable keypoints, swaps, close-ups);
  - cost;
  - what is worth keeping and what is not.
- Update the README (new setups, `pose` flag, ground-truth workflow) and the `multiframe_experiment` bullet in AGENTS.md.
- Update the memory note `multiframe-experiment.md`.
- Give me the report URL: `http://192.168.0.6:8080/multiframe-experiment/runs/<id>/report.html`.
- Don't commit.
