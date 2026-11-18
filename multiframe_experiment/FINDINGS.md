# Findings: multi-frame change experiment (2026-11-17)

Runs:
- `mf-20261117-4cuts`: complete. Episode 2, cuts 139, 145, 147 and 151; 12 setups; 16 frame pairs.
- `mf-20261117-ep2`: stopped after 80 LFM3 calls on cuts 139–148, old answer format.

The reports are in `frontend/public/multiframe-experiment/runs/<id>/`.

**How far to trust the numbers.** The sample is small: 4 cuts, 16 pairs, 20–45 judged claims per setup. Differences under about 10 points are noise. The judge is Qwen3.5-27B, the same model as one candidate. I checked cuts 139, 141, 145, 147 and 151 against the frames by eye. Most verdicts were right, and Qwen's lead over LFM3 holds up by eye, so it is not only judge bias.

## Results (scores after the "remains/unchanged" fix)

| Setup | Score | Precision | Recall | s per 5-frame call | compute s per footage s |
|---|---|---|---|---|---|
| Qwen3.5 observer / frames | 67.5 | 88% | 53% | 54 | 22.0 |
| Qwen3.5 observer / roster | 66.6 | 87% | 50% | 56 | 22.9 |
| Qwen3.5 animator / roster | 66.1 | 78% | 59% | 59 | 23.9 |
| Qwen3.5 animator / frames | 65.8 | 82% | 66% | 59 | 24.0 |
| Qwen3.5 animator / full | 63.3 | 70% | 53% | 60 | 24.3 |
| Qwen3.5 observer / full | 63.2 | 73% | 53% | 60 | 24.3 |
| LFM3 observer / full | 57.4 | 59% | 22% | 13 | 5.5 |
| LFM3, other setups | 24–52 | 0–38% | 0–27% | 10–24 | 4–10 |
| Stored Qwen3-VL video analysis | 37.1 | 29% | 25% | — | 21.5 |

## What the runs show

1. **Qwen3.5-27B on keyframes is clearly better than LFM3 and better than the current video analysis, at the same cost.**
   - Qwen3.5 got 82–88% precision without the extra context, against 29% for the stored video analysis.
   - It costs about the same per second of footage: 22–24 s against 21.5 s.
   - Its claims are tied to specific frame pairs. Examples:
     - cut 147: "enters from the right, arm extends" at F1→F2, "starts speaking" at F4→F5;
     - cut 151: blink and mouth changes, frame by frame.
   - The video analysis spreads one sentence over the whole cut and adds things that don't happen: "pulls her arm" and "walking" in cut 147.
2. **LFM3-Omni-2B is not usable for describing changes.**
   - Recall is 6–27% and precision is at most 59%. Most calls take 10–14 s, up to 24 s for the worst setup.
   - It invents people (a fourth person in cut 145), misattributes actions (Tomoe's arm given to the lost child in cut 147), and copies changes across pairs.
   - Its per-frame **states** are often correct on close-ups (cut 151, cut 139's mouth). But it does not turn them into changes; it left the change lists empty in cut 139.
3. **Observer is more precise; animator finds more.**
   - Qwen precision is 87–88% for observer against 78–82% for animator; recall is 50–53% against 59–66%.
   - Animator writes more speculative motion. Observer suits H3 prompts better, where a wrong motion hurts more than a missed one.
4. **Extra context does not help Qwen.**
   - Frames only and frames plus character names score the same.
   - Adding the first-frame description, transcript and previous answer costs about 15 points of precision (70–73%) and more prompt tokens.
   - For LFM3, the extra context is the least bad setup, but still weak.
5. **Per-frame states work.** Asking for each frame's state first removed most of the copying, and Qwen's states are accurate and detailed. The states are also the most reusable part of the answer.
6. **Camera: no signal in this sample.**
   - Almost every pair is static, so every setup scores 67–71%.
   - The pixel estimate (ORB + RANSAC) agrees with what I see in these cuts. Camera movement can come from it, at no model cost.
7. **Speaking from the model is noisy.**
   - LFM3 calls crying "speaking".
   - Qwen is better, but it misses half the on-screen speech when not given the transcript.
   - The transcript already says when someone speaks; the frames only need to say whose mouth moves.
8. **Cost is almost all decode time.** A Qwen call writes about 1,500 tokens (states plus changes) and takes 54–60 s; reading the 5 images takes a few seconds. Shorter answers are the main way to make it cheaper.

## Worth trying next

1. **Qwen3.5 observer, frames only or with character names, on many more cuts.** It is the candidate to replace video analysis for motion. Two setups over the 31 windows of cuts 139–158 take about an hour. Add cuts with real camera moves and walking; none of the 4 cuts tested walking.
2. **Structured states with the changes worked out by the app.** The model fills fixed fields per character and frame: mouth open/closed, eyes open/half/closed, head yaw/pitch, arm/hand position, body pose, place in frame. The app then compares consecutive frames. This should:
   - cut Qwen's output (and so its time) a lot;
   - remove copied changes;
   - possibly make LFM3 useful as a cheap first pass, since its per-frame states are its strong side.
3. **Camera from the pixel estimate, not the model.** Validate it on cuts with pans and zooms.
4. **Speaking = transcript timing + the mouth state in each frame**, not a yes/no from the model.
5. **Hand labels for these 4 cuts (16 pairs).** A person or an agent can fill `labels-template` to get true accuracy and to check the judge's bias.

## Not worth continuing

- **LFM3 for change descriptions**, in any prompt or context; the animator prompt with character names scored zero.
- **The full context for Qwen:** less precise and slower.
- **The animator prompt** as the main prompt; keep it only if recall matters more than precision.
- **Asking the models for the camera move.**
- **Scoring "speaking" against the transcript** when the setup was given the transcript.
- **Name checks.** After the duplicate merge and the character-name rules, no setup invented names in this run.

# Pose hybrid (run `mf-20261117-pose`, 2026-11-17)

Episode 2, cuts 139, 143, 144 and 158: 4 windows, 16 frame pairs. Qwen3.5-27B only. Report:
`frontend/public/multiframe-experiment/runs/mf-20261117-pose/report.html`.

- **A** `qwen35|observer|frames`: the reference.
- **B** `qwen35|observer|roster`.
- **C** `dwpose|rules|keypoints`: DWPose on the CPU plus rules, no language model.
- **D** `hybrid|dwpose+qwen|frames`: C's body parts merged into A's answers, no extra call.

**Hand labels came first.** I labelled all 16 pairs from the keyframes before any A–D answer existed (`labels/<cut>.json`). The labels are per pair and per category, for any character. So a claim of the right category with a wrong detail still counts as a true positive (for example, C's "standing → sitting" for Tomoe's kick, where the body pose did change). The sample is tiny: the categories have 2–13 labelled pairs each, so one pair moves a per-category number by 8–50 points.

## Results

| Setup | GT F1 | GT precision | GT recall | GT body F1 (P / R) | Judge precision | Judge recall | Judge score | s per window | s compute per footage s |
|---|---|---|---|---|---|---|---|---|---|
| D hybrid | **57** | 70% (30 / 43) | **48%** | **59** (71% / 50%) | 49% | 47% | 48.7 | 60.1 | 24.9 |
| B Qwen roster | 53 | 78% (25 / 32) | 40% | 46 (82% / 32%) | 84% | 53% | 63.7 | 60.8 | 25.1 |
| A Qwen frames | 48 | 72% (23 / 32) | 37% | 47 (75% / 34%) | 80% | 47% | 60.7 | 58.2 | 24.1 |
| C DWPose rules | 43 | 76% (19 / 25) | 30% | 55 (76% / 43%) | 35% | 22% | 52.2 | **1.9** | **0.78** |
| Stored Qwen3-VL video analysis (3 cuts) | 19 | 46% (6 / 13) | 12% | 17 (60% / 10%) | 20% | 17% | 23.0 | — | 25.9 |

"Body" means walking, arms & hands, legs, body pose and position. The judge score uses the old components, with camera at weight 0.

**Body parts, hand labels (precision / recall; labelled pairs in brackets)**

| Setup | Walking (2) | Arms & hands (9) | Legs (11) | Body pose (9) | Position (13) | Head (7) |
|---|---|---|---|---|---|---|
| A | 0 / 0 | 100 / 33 | 100 / 27 | 83 / 56 | 100 / 31 | 0 / 0 |
| B | 0 / 0 | 100 / 56 | 100 / 27 | 75 / 33 | 100 / 23 | 50 / 14 |
| C | 50 / 50 | 75 / 67 | 60 / 27 | 67 / 22 | 100 / 54 | — / 0 |
| D | 20 / 50 | 75 / 67 | 60 / 27 | 83 / 56 | 100 / 54 | 0 / 0 |

**Speaking** (scored only against a boxed speaker): all Qwen setups and D reached 100% recall with 80% precision. The crying child's mouth counts against them, since crying is not in the transcript.

## What the run shows

1. **The merge helps: D finds more without losing precision.**
   - D has the best GT F1: 57 against 48 for A, the setup it is built from.
   - Recall goes from 37% to 48%, and body recall from 34% to 50%. That is 7 more true categories for 4 more false ones.
   - What DWPose adds:
     - position, including people entering the frame (cut 144 entry; Tomoe and Sakuta in cut 158 F3→F4);
     - arms;
     - one of the two walks: Sakuta in cut 139, which both Qwen setups missed.
   - The cost is A's plus about 2 s of CPU per window.
2. **C on its own is cheap but finds too little.**
   - On body parts C beats both Qwen setups (F1 55 against 46–47) at 1/30 of the cost: 0.78 s against 24 s per footage second.
   - Its overall recall is 30%, since it never reports head, eyes, mouth or expression.
   - The rules make real mistakes:
     - Tomoe's high kick reads as "standing → sitting";
     - Sakuta's dive in cut 144, seen from behind, reads as body turns;
     - a hand moving because the whole body travels is reported as a hand move;
     - the walk rule caught only one of the two walks.
3. **The Qwen judge is unfair to DWPose output, so it cannot rank C or D.**
   - Judge verdicts agree with the hand labels on 74–76% of Qwen claims but on only 46% of C's claims and 52% of D's.
   - C got 18 "false" verdicts in pairs where the labels show that category changing.
   - By my reading of the frames, the judge is wrong on roughly 6–8 of those 18 and right on about 5; the rest are uncertain.
     - Judge wrong: Sakuta walking in cut 139; "enters the frame" in 144; hands raised beside the head in 144; the officer's pedalling leg.
     - Judge right: the kick read as "sitting"; the turns in 144.
   - The judge score puts D last among the candidates (48.7), while the hand labels put it first. Both biases are present: the judge favours Qwen's wording, and C's details are sometimes wrong in a category that did change.
4. **The judge's own reference table misses a lot.**
   - Against the labels it reaches 76% precision and 54% recall.
   - It never marks walking (0 of 2) and finds only 31% of position changes.
   - It over-marks expression (43% precision).
   - So the earlier judge-recall numbers understate how much every setup misses.
5. **The camera should come from the pixel estimate.** On the 12 pairs both could decide, it matches the hand labels 12/12; the judge's camera matches 11/14.
6. **Qwen frames vs roster, now measured against hand labels:**
   - roster has slightly better F1 (53 vs 48), mainly expression and arms;
   - roster finds fewer body changes (32% vs 34% body recall).
   - Both are within noise, as in the earlier run.

## DWPose on anime

- **Usable bodies:** 33 of 34 person-frames have a usable body (≥ 6 of 17 body points over 0.3).
  - All 22 close-ups are usable (box at least 80% of the image height).
  - No suspected left/right swaps: limbs are compared by their side of the screen, and no wrist flip with a steady box occurred.
  - Mean body confidence is 0.36–0.87. It is lowest for Tomoe's blurred lean in cut 158 and highest for Sakuta, who is small in the background of 139.
- **Overlays:** the skeletons look plausible, even on cut 143's motion-blurred kick (`frames/<cut>/pose/`).
- **Failures are detector misses, not bad poses:**
  - the half-hidden lost child is never found in cut 143 F3–F5, nor in cut 158 F4;
  - in cut 158 F5 the child is found at 0.14 confidence (not usable).
- **Box anchoring and tracking work:**
  - The first-frame and last-frame character boxes name the tracks, so Tomoe and Sakuta in cut 158 are named from the last frame backwards.
  - One fix was needed during the build: matching after the pixel camera transform. Before it, the zoom-out in cut 158 moved the officer's name onto Tomoe.
- **Cost:** the CPU (onnxruntime) needs about 0.4 s wall (about 3 s of CPU time) per keyframe, 7.6 s wall for all 20 keyframes. The GPU is untouched.

## Keep

- **The D shape:** Qwen's observer answer for face, head and the rest, plus DWPose for position, enter/leave, arms and walking.
  - It gives the best F1 for almost no extra cost.
  - It is the natural form of the "structured states, app compares" idea from the first round.
- **DWPose itself:** CPU only, about 0.4 s per keyframe, usable bodies on 97% of anime person-frames, and the box anchoring for names.
- **The pixel camera estimate as the only camera.**
- **Hand labels as the ranking basis.** The labelling workflow takes about 15 minutes for 16 pairs.

## Don't keep (or fix first)

- **C alone as a replacement:** it has no head, face or speech, and recall is 30%.
- **The Qwen judge as the scorer of keypoint-style claims**, and the judge's reference table as a recall reference. Use hand labels; use the judge only for Qwen-vs-Qwen comparisons.
- **DWPose head yaw:** it got no head change right. Head stays with Qwen.

## Rule fixes worth doing before a bigger run

These are not tuned yet, on purpose: the labels were already made.

- Subtract the body's travel from hand moves.
- Read "lying / diving" from a back view using box shape and hip height, not torso tilt.
- Make the walk rule use alternating ankles.
- Never output "sitting" when one foot is above hip height (a kick).

## Next run

- About 15 cuts with real walking, sitting down/standing up and camera moves.
- Hand-label them first.
- Compare A against D only.

# Episode 2 pose run (run `mf-20261117-ep2pose`, 2026-11-17)

Episode 2, cuts 139–158 (20 cuts, 30 windows, about 2 h of GPU time). Qwen3.5-27B only, styles `observer` and `inspector` × contexts `frames` and `roster`, plus DWPose alone (C) and four hybrids (D, one per Qwen setup), plus the stored Qwen3-VL analysis. Report:
`frontend/public/multiframe-experiment/runs/mf-20261117-ep2pose/report.html` (`report.md` beside it; on this host `http://192.168.0.6:8080/multiframe-experiment/runs/mf-20261117-ep2pose/report.html`).

**How far to trust it.** The hand labels are still the same 16 pairs of cuts 139, 143, 144 and 158, so the GT columns (F1, precision, recall by category) are the same evidence as the previous run, now for 4 more setups. The other 16 cuts are scored only by the Qwen judge, which we already know is biased against keypoint claims. I did not re-check these 16 cuts against the frames by eye, so the judge-based statements below are numbers from the report, not verified verdicts.

## Results

| Setup | GT F1 | GT precision / recall | GT body F1 | Judge score | Judge precision / false claims | Invented names | s per footage s |
|---|---|---|---|---|---|---|---|
| D observer / roster | **55** | 75% / 43% | 49 | 56.4 | 68% / 31% | 0 | 22.5 |
| D observer / frames | 53 | 71% / 43% | **54** | 55.7 | 66% / 34% | 3 | 21.8 |
| D inspector / roster | 53 | 71% / 43% | 53 | 55.7 | 65% / 34% | 0 | 23.0 |
| Qwen observer / roster | 53 | 78% / 40% | 46 | **66.3** | 92% / **8%** | 0 | 21.9 |
| D inspector / frames | 52 | 70% / 41% | 51 | 55.5 | 65% / 35% | 3 | 21.9 |
| Qwen observer / frames | 48 | 72% / 37% | 47 | 64.7 | 88% / 11% | 3 | 21.2 |
| Qwen inspector / roster | 46 | 69% / 35% | 42 | 63.6 | 86% / 14% | 0 | 22.4 |
| Qwen inspector / frames | 43 | 67% / 32% | 37 | 63.8 | 87% / 13% | 3 | 21.3 |
| C DWPose rules | 38 | 76% / 25% | 49 | 49.8 | 33% / 67% | 0 | **0.63** |
| Stored Qwen3-VL analysis | 19 | 46% / 12% | 17 | 31.7 | 30% / 70% | 0 | 11.0 |

## What the run shows

1. **The hybrid helps every Qwen setup, so the first result was not a fluke of one setup.** GT F1 rises by 9 (inspector/frames), 7 (inspector/roster), 5 (observer/frames) and 2 (observer/roster). Precision stays at 70–75%, recall rises 3–11 points. The cost is about 1.5 s of CPU per window (21.8–23.0 against 21.2–22.4 s per footage second). With 16 labelled pairs, only the 5–9 point gains are more than one or two pairs.
2. **Inspector is not worth keeping.** Fixed per-frame fields compared by the app score below observer without the hybrid (GT F1 43–46 against 48–53; judge 63.6–63.8 against 64.7–66.3) and equal with it (52–53 against 53–55). It does not shorten the answer either (about 1,350–1,430 output tokens in all setups). Drop it as a prompt style.
3. **Roster is again slightly ahead of frames** (observer: GT F1 53 against 48, judge 66.3 against 64.7), and it is the only context with no invented names: frames-only setups invented names in 3 of about 200 claims in every style, roster setups in 0. That settles roster over frames for a real pipeline, since names matter there anyway. The F1 gap itself is still within noise.
4. **On 20 cuts the judge and the labels disagree strongly about DWPose claims, and I cannot yet say who is right.**
   - Judge precision on arms & hands is 30–45% for C and the hybrids against 89–96% for Qwen; on position in frame 31–40% against 93–100%; on legs 25–40%.
   - On the 4 labelled cuts the same categories have 100% (arms, position) and 60% (legs) precision for the hybrids.
   - The judge's verdicts agree with the labels on only 55–68% of hybrid claims against 67–84% for Qwen, and it says "false" on a labelled change 12–20 times per hybrid setup.
   - The previous run's by-eye check found the judge wrong on a good part of those. The 16 unlabelled cuts may hold real DWPose false claims (walking has 25–29% judge precision and 29–44% hand-label precision in every setup that has it), so the rules need the labelled set to grow before trusting either side.
5. **The walk rule over-fires.** Every DWPose-based setup has walking recall 50–100% and precision 17–44% (judge and labels agree on that). Qwen alone almost never says walking (GT 0 of 2 pairs found; the judge sees 100% recall only because its reference table is generous). The "alternating ankles" fix from the previous run is now the first thing to do.
6. **DWPose works on anime at scale:** 177 of 185 person-frames (96%) have a usable body, 112 of 117 close-ups, 0 suspected left/right swaps, 45.8 s of CPU for all 20 cuts. The misses are cut 156 (0 of 2, small and blurred), cut 142 (1 of 2) and cut 157 (2 of 3), with mean confidence 0.13–0.27.
7. **Speaking and camera:** unchanged. Speaking agreement is 57–61% for every setup (Qwen sees mouths; the transcript decides who speaks anyway). The pixel camera estimate agrees with the labels 12/12; the judge 11/14.
8. **The stored video analysis again comes last** (GT F1 19, judge 31.7), at half the cost per footage second (11.0 s against 21–23 s). That is the price comparison to remember: the new path costs twice as much, because it decodes about 1,400 tokens per window.

## Decision

- **Candidate to replace video analysis for motion: D observer / roster** (Qwen3.5 observer with roster names, plus DWPose merged by the app). Best GT F1 (55), no invented names, and the judge's low precision on its body claims is unconfirmed rather than proven.
- **Not before:** (a) the walk rule is fixed, (b) the labelled set covers at least the cuts with real walking, sitting down and standing up, a pan or zoom (about 15 more cuts, roughly 15 minutes per 16 pairs), and (c) A against D is re-run on those only. If D still has the best F1 there, wire it into `single_cut_pipeline` as a new `video_analysis` model kind under the registry in `visual_models.py`.
- **Dropped:** inspector style, frames-only context for production, LFM3 for any of this.
