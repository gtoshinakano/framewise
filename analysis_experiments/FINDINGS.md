# Pose-detail experiment: findings and hand-off

Run `pose-20261116-0940` (2026-11-16, 09:41 to 12:52 UTC, 3 h 11 min). Models: Qwen3.5-27B Q4_K_M (first frame, llama.cpp), Qwen3-VL-8B bf16 (video, transformers), Qwen3.6-35B-A3B (conclusions). 20 already-analysed cuts, 3 prompt approaches, frame and video stages, plus a flipped-frame (mirror) probe on frames.

Data (gitignored, files owned by uid 10001, delete with a throwaway container): `frontend/public/analysis-experiments/runs/pose-20261116-0940/`. **Copy `results.json` (1.6 MB), `samples.json` and `raw/` somewhere durable before cleaning up: this document's evidence lives there.** Report page: `http://192.168.0.6:8080/analysis-experiments/index.html?run=pose-20261116-0940`.

## 1. Verdict

Worth adding, in a smaller form than the schema asked for:

- **Add to the first-frame analysis only**, for marked characters: `body_pose`, `body_yaw`, `head_yaw`, `eye_gaze`, `gaze_target`, `camera_elevation`, `shot_size` (compute `shot_size` from the box height instead of asking, see 4.9).
- **Hold the video block.** Qwen3-VL-8B only reached 12/20 valid answers in the best approach, and the start/end snapshots are almost always identical (21 of 27 valid ones). If video gets anything, ask for what *changes*, not two full snapshots.
- **Hold the face fields** (`mouth_state`, `eyebrow_state`, `facial_expression`, `eye_openness`): they mostly return defaults and agree only 70-85% with the stored free text. `head_roll` is "neutral" in 25 of 26 answers, so it adds nothing yet.
- **Drop per-field confidence and the "top 5" style.** Confidence carries no signal (4.7).
- Style to carry forward: A3 (strict, all fields, no confidence, detailed value list) for its structure, but **not its 45-degree yaw grid** and with `not_visible` allowed.

## 2. Numbers (frame / video, A1 full+confidence, A2 top-5+confidence, A3 strict)

| | Final | Frame score | Video score | Frame first-try | Frame final | Video valid |
|---|---|---|---|---|---|---|
| A1 | 41.6 | 71.5 | 11.7 | 6/36 | 36/36 | 0/20 |
| A2 | 40.9 | 66.9 | 14.9 | 5/36 | 36/36 | 1/20 |
| A3 | 83.6 | 92.9 | 74.4 | 34/36 | 36/36 | 12/20 |

36 frame calls per approach = 20 cuts + 16 mirror probes (4 cuts have no marked character, so no mirror call).

Frame quality (mirror / geometry / agreement with the stored analysis): A1 0.80 / 0.97 / 0.81, A2 0.82 / 1.00 / 0.85, A3 0.81 / 0.96 / 0.79. **Perception quality is tied; the final-score gap is format compliance** (reliability, coverage, retries) plus the video stage.

Cost (per call, first attempt; experiment used 1 llama.cpp slot, so seconds are not comparable with production's `FRAME_PARALLEL=4`, tokens are):

| | Frame prompt tok | Frame completion tok | Frame s | Video completion tok | Video s (first / with retries) |
|---|---|---|---|---|---|
| stored production | n/a | 413 | 27.7 | 378 | 24.0 |
| A1 | 2752 | 837 | 31.7 | 671 | 32.9 / 117.7 |
| A2 | 2719 | 595 | 23.3 | 737 | 35.3 / 102.7 |
| A3 | 3484 | 616 | 23.8 | 688 | 34.1 / 91.4 |

So the block adds about +49% output tokens on the frame and +82% on the video. Frame time grows with the number of characters (A3 first attempt: ~8-12 s with none marked, ~20 s with 1, ~30 s with 2, ~41-44 s with 3).

## 3. What worked and what did not (evidence)

- **Frame validator failures are almost all output-shape mistakes**, not perception: A1 put `character_id` inside `confidence` (46 errors), A2 did not sort the five fields by confidence (41). Two deterministic repairs raise first-try pass from 6/36 to 32/36 (A1) and 5/36 to 30/36 (A2); A3 stays 34/36. The 2 A3 failures say `pose_detail must be an array with one entry per marked character`; not investigated.
- **Video failures are vocabulary drift on enum fields**, and the retry feedback does not fix them: in 4 of the 8 failing A3 cuts all three attempts returned the same rejected values. Rejected values seen: `facial_expression` `not readable`, `calm`, `concerned`, `mild concern`, `tired`, `serious`, `mild curiosity`, `curious`, `attentive`, `shy`, `defensive`, `mild surprise`, `focused`; `eye_gaze` `none` (34-39 per approach, eyes closed or face turned away) and once a character name; `body_pose` `crouching`, `bent down`, `bending` (frames: `kneeling`); `head_roll`/`eye_openness` `null`; A3 `head_pitch` -15 (off the grid). Replaying the stored raw answers through the validator after a small alias map (calm to neutral, concerned to worried, not readable/none to unclear, crouching/kneeling to other, etc.) and snapping angles to the grid gave **A3 video 12/20 to 20/20**. A1 and A2 video stay 0/20 and 1/20 because of the confidence shape, below. The alias map was written after seeing these 20 cuts, so the 20/20 is optimistic.
- **A1/A2 video cannot do per-field confidence:** the 8B model returns one scalar (`"confidence": 0.95`) per snapshot instead of an object per field (158 errors in A1). A2 video also broke "return exactly 5 fields" (132 errors).
- **Retries on the video model cost a lot and rescue little:** A3 video averaged 91 s per cut with retries against 34 s for the first attempt; of 16 windows that needed a retry 8 were rescued and 8 failed all three.
- **A3's 8 hard failures:** s01 (head_pitch -15, `calm`), s08 (`eye_gaze: none`), s11 (`eye_gaze: none`, expressions), s12 (`attentive`, `shy`, `defensive`), s13 (`crouching`, `concerned`), s14 (`eye_gaze: none`), s18 and s19 (background people added to the model's own `characters` list as "person in white shirt and blue pants" / "Unnamed person", with `not readable` expressions). All eight are fixable by coercion in the replay.
- **Confidence is uninformative** (4.7 below).
- **Yaw.** Distance from an exact mirror (flipped frame must give the negated yaw), head_yaw: within 25 degrees A1 18/26, A2 15/21, A3 12/26; within 45 degrees A1 22/26, A2 18/21, A3 18/26; beyond 45 degrees A1 4, A2 3, A3 8. The 45-degree grid makes any boundary case a full step. Qwen3.5 gets sign and target right on easy frames (checked by eye on s07: two characters facing each other, both looking at each other, gaze_target correct), magnitudes are noisy by one step, and a flipped frame tends to give smaller yaws.
- **A3 defaults.** A3 coverage is 0.98 because every field is mandatory; it falls back to safe values: `head_roll` neutral 25/26, `mouth_state` closed 20/26, `eyebrow_state` neutral 15/26, `facial_expression` neutral 13/26. A1 abstains (`unclear` ~30% on the four face fields) and uses more varied values (eyebrow `furrowed` 5 against 0 in A3). On the mirror test, A3's stability on face fields is partly this defaulting. On s07 the second character's face was in shadow: A1 abstained, A3 filled in neutral/closed.
- **Face fields on large characters** (A3, 20 characters whose box is 60%+ of the frame height): same value on the flipped frame for `facial_expression` 11/20, `mouth_state` 19/20 (mostly "closed"), `eyebrow_state` 17/20, `eye_openness` 16/20.
- **Geometry checks passed 96-100%** (gaze side vs box positions, camera gaze, shot size vs box height, pitch vs vertical gaze, head vs body). The three A3 failures were `gaze_side` cases such as `head_yaw -45` with `eye_gaze right` when the target stood to the right.
- A2's own picks (which five fields the model trusts most): body_yaw 25, head_yaw 23, body_pose 17, eye_gaze 15, head_pitch 14, mouth_state 13, shot_size 10, facial_expression 6, eye_openness 4, camera_elevation 3. It points at the structural fields, which matches the verdict.
- `camera_elevation` is `eye_level` in 23/26 (A3), 21/26 (A1).

## 4. Recommended changes for the implementation (ranked by evidence)

1. **First frame only first.** Store `pose_detail` in the first-frame result; no video snapshots until a re-test.
2. **Video later as deltas** (fields that changed plus a short `changed` list), not two 13-field snapshots.
3. **Coercion layer before validation** (code): alias map for the drifting words above, angle snapping, move `character_id` out of `confidence` if confidence is ever used, sort lists in code. Validate after coercing and only then retry.
4. **At most one retry**, only for structural errors (missing/extra entry, not JSON), never for vocabulary drift.
5. **`not_visible` (and keep `unclear`)** in the enums for gaze and the face fields; do not force a value when the eyes are closed or the face is turned away.
6. **Do not add the face fields yet**; revisit after a labelled test with close-ups only.
7. **No per-field confidence, no "top 5".** If a confidence is wanted, compute it from abstention, geometry agreement or repeated runs.
8. **Replace the 45-degree yaw grid** with free integers (round to 15 degrees in code) or buckets (`front`, `three_quarter`, `profile`, `back` plus a side).
9. **Use the boxes:** derive `shot_size` from the box height; check/repair `gaze_target` and the head/eye side against the left-right order of the boxes (`geometry_checks` in `scoring.py` already encodes the rules).
10. **Get ground truth** (hand-label 25-30 frames: head_yaw sign/bucket, eye_gaze, gaze_target, body_pose) before trusting any score; keep this harness as the regression run after prompt or model changes.

## 5. Inconsistencies found, and numbers NOT to rely on

- **Qwen3.6's written conclusions are wrong in places**: it calls the 0.01 confidence gap "excellent calibration" (it means confidence tells nothing); its overall verdict lists `shot_size` and `camera_elevation` under "fields to avoid" while its per-approach lists call them reliable; all three runs get `adopt_with_changes` although A1 and A2 produced almost no usable video output; it names a "frame efficiency 1.0" as a strength although that is only relative speed. Use the raw numbers, not `conclusions.json`.
- **Final score is not an accuracy.** It is 30% reliability, 40% consistency-based quality, 15% coverage, 15% relative speed, frame and video averaged 50/50. Consequences: no usable video output counts as quality 0 (A1/A2 video 11.7 / 14.9 are mostly "no data"), A2 coverage is capped at 0.38 by design, coverage rewards forced guesses (A3) over honest abstention (A1), and "efficiency" includes retry time.
- **Video "geometry 1.0 (64/64)" is trivial**: with no boxes only the box-free checks run. There is no mirror test for video at all.
- **Mirror-invariant fields** (`head_pitch`, `camera_elevation`, `shot_size`, `head_roll`) pass the mirror test when they are constant. Stable is not accurate.
- **Mirror yaw tolerance is 25 degrees**, stricter than one grid step; with a one-step tolerance A3 is 18/26 rather than 12/26.
- **Handles change when a frame is flipped** (they are sorted by left edge). Pair characters through `mirror_map` stored in the mirror call (`calls.frame.<sample>.mirror.mirror_map`). My first ad-hoc analysis paired them by index and was wrong; `scoring.py` uses the map.
- **Baseline seconds are not like-for-like** (production runs 4 frame slots at once); only tokens compare.
- **The replay numbers** (12 to 20, 6 to 32, 5 to 30) were built after reading the failures: expect a smaller gain on new cuts.
- **`dry-run`** in the run picker uses fake models; ignore it.
- The idle worker's 23.5 GB VRAM after the run is its Qwen3.6 text model, not a leak.

## 6. Limitations

- 20 cuts, 26 characters in frame answers, one series (episodes 2 and 4-7 of the same anime), one run per cell: run-to-run noise and confidence intervals are not measured.
- **No ground truth.** Quality = mirror consistency, box geometry and agreement with the earlier model's free text. A model can be consistently wrong.
- **Mostly large characters:** 23 of 26 boxes are 60%+ of the frame height. Wide shots and tiny faces are barely covered, so face-field results are the optimistic case.
- Constrained decoding was off (`--grammar` untested): production's frame stage uses a `json_schema` grammar, which would remove enum drift on frames. The video model (transformers) has no grammar. Whether a grammar changes yaw/gaze quality is unknown.
- Video: one window per cut in most cases, no mirror probe, no check that start vs end matches the real motion (the stored `motion` text was not used as a reference beyond `activity`/`speaking`).
- Frame calls ran with `frame_parallel=1`, so batching behaviour in production is untested.
- The 4 cuts with nobody marked produce no frame probe and no pose entries.

## 7. Notes for the implementing agent

- **Hook points** (see AGENTS.md): `visual_contracts.py` (`FRAME_INSTRUCTION`, `frame_schema`, `resolve_frame`, which today drops unknown keys and swaps `B1...` handles for `character` and `character_context_id`), `frame_model.describe_frame` (system message + per-frame user message, `json_schema` response format), `single_cut_pipeline.marked_people` (handles and 0-1000 boxes), `video_model.analyze_video` and `visual_contracts.link_video`. The experiment appended one key (`pose_detail`) to the production prompts without touching them; follow the same incremental shape. Results are JSON in `analysis_results`, so check whether a migration is needed before assuming one.
- Anything new that reaches text prompts goes through `prompt_context.py` (`frame_view`); keep it compact. If the field shows up in `/api/v1`, update `external_api.py`, `integration-guide.md` and `docs/agent-skill-prompt.md`. Restart the inference worker after changes.
- **Reusable code:** `analysis_experiments/pose_schema.py` is stdlib only: field list, enums, `validate_entry/answer`, `validate_video_*`, `format_errors`, `parse_json`, `mirror_entry`, `prompt_block(approach, kind)`, `json_schema(approach, ids, kind)` (grammar). Copy it, then change: yaw to free angles or buckets, add `not_visible`, add the coercion step, remove confidence and top-5 paths. `prompts.py` has the retry messages.
- **Prompts:** AGENTS.md says describe rules, never give filled example sentences (the model copies them). Conventions used: left/right = the image's left/right; yaw 0 faces the camera, positive toward the image's right, +-90 profile, 180 back; pitch negative = down; `gaze_target` = a handle, `camera`, `off_screen`, `object:<noun>` or `none`.
- **Rerun the harness** (GPU: only one process at a time; `run.sh` stops the inference worker, waits for VRAM to drain, restarts it afterwards; the queue must be idle): `analysis_experiments/run.sh start --run-id NAME [--approaches strict_detailed] [--limit N] [--resample] [--grammar] [--no-mirror]`, `run.sh status`, `run.sh stop`, `run.sh dry-run` (CPU, fake models). A stopped run resumes with the same `--run-id` (finished calls are skipped via `raw/*.jsonl`). The sampler is deterministic (`sha1(cut_id)` within 15 strata) so the same 20 cuts return while the DB is unchanged; `samples.json` in the run folder is the record. Unit tests: `PYTHONPATH=backend python3 -B -m unittest discover -s analysis_experiments/tests -t .` (45 tests, host).
- **Where to look in `results.json`:** `approaches.<name>.summary.<frame|video>` (rates, per-field table, tokens, seconds), `approaches.<name>.calls.frame.<sample>.<normal|mirror>` (`answer`, `attempts[]` with `errors`, `raw`, `mirror_map`, `checks[]`), `approaches.<name>.calls.video.<sample>.windows[].attempts[]`, `samples[].stored` (the earlier analysis and `baseline` seconds/tokens), `agreement_between` (cross-approach agreement: A1/A2 0.91, A1/A3 0.83, A2/A3 0.75).
- **Suggested next experiment** (about an hour of GPU with the worker paused): A3 with free-angle yaw, `not_visible`, coercion and one retry; frame with `--grammar`; video as deltas only; add about 10 wide-shot cuts (small characters) and the hand labels from 4.10; repeat each cell twice to see the noise.
- Nothing from this experiment is committed to the app; `backend/app` is untouched.
