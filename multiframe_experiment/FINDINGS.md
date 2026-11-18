> **Reading order for a new session:** start with `PIPELINE_HANDOFF.md` (consolidated decisions, evidence, open problems, next steps). This file is the
> chronological log of every round and keeps the evidence; **where a later section changes an earlier recommendation, the later one wins** (for example
> the "Worth trying next" list below was done in the pose and walking rounds). Archived data and replay tools: `archive/README.md`, `tools/`.
> The number of hand-labelled pairs is 60 (14 cuts); earlier sections quote the labelled pairs of their own run.

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

# Walking rule round (run `mf-20261118-c160-170`, 2026-11-18)

**What changed in `pose_rules.py`** (tuned on the cuts 139–158 keypoints in `mf-20261117-ep2pose`, never on 160–170):
- **Alternating ankles.** A walk needs the anatomical left and right ankles to trade the lead between the two keyframes, sideways or in height, both leads at least 0.12 torso lengths apart, with the same hip still on the same side of the picture (a turned body or a DWPose left/right swap is no alternation). A foot swinging 0.25 torso lengths counts only beside such a flip within 1 s in the same window.
- **Camera movement.** Travel is the person's own shift with the pair's camera transform removed, so a pan that follows a walker (the person stays still in the picture, the background slides) still counts as travel. With no camera estimate the shift must be 2.5 times larger, and a person who steps without travelling is `unclear` instead of `no`, because the camera may be following them (the same goes for anyone who fills the picture).
- **Sitting state.** Sitting, crouching or lying in either frame, or a foot above the hips (a kick), is `no`. A hip in the bottom 8% of the picture, or a knee or ankle outside it, was never seen (a bust shot), so it can no longer be sitting or lying. In the hybrid, Qwen's frame text saying sitting, riding, kneeling, crouching or lying (`SEATED`) forces `no` (the officer on the bicycle in cut 158 is "riding" for Qwen and "standing" for DWPose).
- The rule says `no` only for what it can see is not a walk, and `unclear` where the keypoints cannot decide (travel with similar legs, one ankle missing, a body cut at the waist). The hybrid then takes Qwen's answer.
- New stage `experiment.py rules --run-id <id> --merge-with ...` re-derives C and D from the stored `pose.json` after a rule change (CPU, no model). The report no longer crowns "Best overall" when a run has no judge.

**On the old keypoints (139–158)** the rule removes the false walks that made walking precision 17–44%: Tomoe's kick and the bicycle no longer say `yes`, Sakuta being flung in cut 143 is `unclear`, and Sakuta walking in (cut 140, close-up, 0.125 s apart) stays `yes`. The price is the far walker of cut 139 (70 px tall, legs hardly change): `unclear`, so one hand-labelled walk is lost (D F1 55 → 54). The hip guard removes five of six false `sitting`/`lying` states of close-ups (cuts 147, 149 and 153); one false `lying` (cut 149) and the true ones (the dive in cut 144, Sakuta sitting in cut 158) stay.

## Held-out run: Episode 2, cuts 160 and 162–170

10 cuts, 51 keyframes, 44 pairs, 13 windows. Only the best approach was run: A `qwen35|observer|roster`, C (DWPose rules) and D `hybrid|observer|roster` (C merged into A, no extra call), plus the stored Qwen3-VL analysis as the baseline. No judge (it cannot rank keypoint claims; the labels are the ranking). I labelled all 44 pairs from the keyframes before any answer existed (`labels/<cut>.json`): 13 walking pairs (Sakuta running through two station cuts and out of frame, Tomoe walking out), a real camera tilt (cut 169), a talking-heads cut with the pixel estimate calling zooms, a police desk with a bending officer. The only seated characters (Sakuta and Tomoe at the desk, cut 162) are cut at the waist, so DWPose has no legs for them. Report: `frontend/public/multiframe-experiment/runs/mf-20261118-c160-170/report.html` (on this host `http://192.168.0.6:8080/multiframe-experiment/runs/mf-20261118-c160-170/report.html`). The run took 16 minutes (prepare 26 s, DWPose 24 s of CPU, Qwen 13 calls at about 50 s).

| Setup | GT F1 | GT precision / recall | Body F1 (P / R) | Walking P / R | s per footage s |
|---|---|---|---|---|---|
| A Qwen observer / roster | **55** | 66% / 47% | 55 (86% / 41%) | 100% / 62% | 18.5 |
| D hybrid observer / roster | **55** | 56% / 54% | **56** (66% / 49%) | 100% / 62% | 19.2 |
| C DWPose rules | 36 | 57% / 27% | 43 (61% / 34%) | 100% / 23% | **0.69** |
| Stored Qwen3-VL analysis | 35 | 39% / 32% | 33 (56% / 24%) | 83% / 77% | 4.7 |

The first pass of this run (before the hip guard, which I added after seeing the old sitting states) gave C 37 and D 55 (57% / 53%, body 57), so the guard moved nothing that matters here.

## What the run shows

1. **The walking rule is precise and, on its own, still modest.** No walking claim of C, D or A is false in 44 pairs (on the 4 labelled old cuts D had 3–5 false walks per setup before the rule and 2–4 after, all of them Qwen's own). DWPose says `yes` three times (the runner in cut 166), `no` for the standing background people, and `unclear` elsewhere. D's walking recall is 62%, the same as Qwen's: it did not add a walk, and the `no` gate removed no true walk.
2. **Five of 13 walks are missed by every setup, all for the same reason: the legs are not in the picture.** Tomoe and Sakuta walking out of cut 164 are upper-body crops; the foreground runner of cut 165 is cut by the frame edge. With no ankles DWPose says `unclear`, and Qwen did not call these walks either. The stored video analysis finds more walks (77% recall) because it sees the whole clip, but it is wrong on most other categories.
3. **The hybrid no longer beats Qwen alone on F1 (55 against 55); it trades precision for recall.** On the old cuts D was ahead (57 against 48). Here D gains arms & hands (recall 27% → 64%) and loses precision on position (100% → 61%) and arms (100% → 58%). Treat the earlier "D is best" as plausible but not settled: it depends on the cuts.
4. **The new position false positives are a camera movement problem, in the position rule and not the walking rule.** D has 7 pairs with a false position claim. Four are cut 169, where the camera tilts up and DWPose reads the changing box height as "comes closer / moves away" for Mai and the two people behind her (the boxes are clipped by the frame edge, so their height changes with the tilt). Two are "leaves the frame" where the tracker lost a person (cut 160 Tomoe, cut 170 Mai at the edge), and one is a "comes closer" on a tiny figure in cut 167. A box that touches the frame edge should not feed the size rule, and a tilt (`dy`) should be removed from the box position the way `dx` and `scale` already are. Not done: it would be tuned on the same 44 pairs.
5. **The pixel camera estimate is right where it matters and wrong where faces move.** It matches the labels on 35 of the 40 pairs it decided, including the true tilt (5 of 5 in cut 169). The misses are four "zoom/pan" calls in the static talking-heads cut 160 and one "pan" in cut 164, where moving faces dominate the transform. None of them touched a walking call (those pairs have no travel and no ankles).
6. **The sitting state is untested on real sitting here.** The seated characters of cut 162 have no legs in the picture, so the gate never fired. On the old cuts it only removes false sitting/lying of close-ups; the bicycle case is handled by Qwen's text, which I checked on the stored answers, not on a new run.
7. **Qwen still has no invented names, and 0 of 13 calls failed** (reliability 100%).

## Next

- Fix the position rule for camera tilt and clipped boxes, then re-run `experiment.py rules` on both runs (CPU, minutes) and check position precision.
- Label and run cuts with a seated character and a camera pan that follows a walker (none in cuts 139–170 so far); those two are the cases the new rule was written for and could not be measured on.
- Decide D against A on the combined 8 labelled cuts + these 10; today it is a tie on F1 with a different mix of errors.

# Final pass over every tested cut (run `mf-20261118-final`, 2026-11-18)

All 30 cuts tested so far: Episode 2, cuts 139–158 and 160, 162–170 (43 windows). Nothing was run on the GPU: the Qwen3.5 observer/roster answers of the two earlier runs were kept as they were, and DWPose rules (C) and the hybrid (D) were derived again from the stored keypoints with the improved rules. Hand labels cover 14 cuts (60 pairs). Report: `frontend/public/multiframe-experiment/runs/mf-20261118-final/report.html` (on this host `http://192.168.0.6:8080/multiframe-experiment/runs/mf-20261118-final/report.html`). "Before" is the committed state `0f26fc3`; "after" is the working tree.

**Read it as in-sample for the rule changes.** I chose them after reading the error lists of both sets (the held-out cuts included), so the "after" numbers are no longer a clean held-out test. The Qwen column did not change, which is the control.

## What changed

- **Camera tilt and clipped boxes (the cut 169 false positives).** A size claim ("comes closer / moves away") needs the same amount of body in view in both frames (a tilt or a gate reveals or hides body points), a box not cut by the picture's top or bottom edge, and a camera that is not tilting; the torso (shoulders to hips, not cut by the top edge) is the size measure where both frames have one.
- **Position follows the torso, not the box.** The hips (else the shoulders) are the body's place, so raising an arm or revealing the legs no longer moves a person across the picture.
- **Enter / leave only through the border.** A person who shows up or vanishes mid-picture is a detector miss, not an entry.
- **Arms.** Elbows and wrists in the bottom 3% of the picture (a close-up ends above the hands, the points land on the lower edge) are not seen.
- **Merge policy.** DWPose still replaces Qwen's claim about a body part it measured as changed, but where it measured nothing (below its coarse thresholds, or a person it cannot read) Qwen's claim stays. Before, DWPose's silence deleted Qwen's correct claims (arms, legs, position).
- **A walk is also a leg change.** An alternating-ankle walk now adds one `legs` change ("the other foot takes the lead").
- **Camera estimate without people** (`measure.camera_without_people`, stage `experiment.py camera`). Every detected person is masked out before the ORB/RANSAC estimate. Against the 60 labelled pairs, wrong calls fall from 5 (four invented zooms/pans in the static talking-heads cut 160, one pan in cut 164) to 1 (a zoom-out in cut 143), the real tilt of cut 169 is still found in 5 of 5 pairs and two of the three labelled zoom-outs of cut 158 too, and `unclear` rises from 21 to 26 of 148 pairs (one of those is a zoom-out that was found before), because a close-up leaves no backdrop to measure. The old estimate stays in `pairs[].camera_all`.

## Results (hand labels, 60 pairs)

| Setup | Cuts | GT F1 (precision / recall) | Body F1 | After vs before |
|---|---|---|---|---|
| **D hybrid** | all 14 labelled | **61** (66% / 56%) | **64** | 54 → 61 (body 53 → 64) |
| A Qwen observer/roster | all 14 | 54 (70% / 44%) | 51 | unchanged |
| C DWPose rules | all 14 | 39 (70% / 28%) | 50 | 37 → 39 (body 46 → 50) |
| Stored Qwen3-VL analysis | all 14 | 31 (40% / 25%) | 28 | unchanged |
| D hybrid | 139–158 (4 labelled) | 63 (79% / 52%) | 62 | 54 → 63 |
| D hybrid | 160–170 (10 labelled) | 59 (60% / 59%) | 65 | 55 → 59 |

By category, D against A on all 14 labelled cuts (precision / recall): arms & hands 88% / 75% against 100% / 40%, legs 70% / 39% against 75% / 17%, position 77% / 67% against 100% / 43%, body pose 60% / 30% against 60% / 30%, walking 80% / 53% for both (the hybrid takes Qwen's call wherever the legs are not in view). D's speaking, blinking and head claims are Qwen's own. Cost: D 21.4 s of compute per footage second against 20.8 s for Qwen alone (DWPose adds 1.6 s per window of CPU); no judge was run.

## What the run shows

1. **The hybrid now beats plain Qwen on both subsets, and the gain is recall.** D gains 7 F1 points on the 14 labelled cuts (61 against 54) for 4 points of precision (66% against 70%): arms & hands recall 40% → 75%, legs 17% → 39%, position 43% → 67%. The earlier tie on the held-out cuts (55 against 55) was mostly the merge dropping Qwen's claims where DWPose was silent plus the tilt and clipping artefacts; both are fixed, but see the caveat above.
2. **Walking did not improve and cannot with these inputs.** 8 of 15 walking pairs are found by every setup, 2 false walks remain (Qwen's own, cut 143, Sakuta flung by the kick) and 7 are missed because the legs are not in the picture (upper-body crops of cut 164, the partly visible foreground runner of cut 165, the 70 px walker of cut 139). Better recall needs another signal, not another rule on the same keypoints.
3. **Remaining D false positives (17 of 73 body claims):** 4 are Qwen's own body-pose calls, 2 are Qwen's walking claims (cut 143), 3 are legs where my labels are strict (cut 158 officer and Sakuta, cut 169 Mai), 2 are arms (cuts 162 and 169) and 6 are position pairs: three in cut 160 (Sakuta's slow drift of 6–9% of the picture width, now measured because the camera is read as static, which I labelled as too small to count; in one of them Tomoe also "leaves the frame"), one in cut 167 (a tiny figure straightening up behind a gate), one in cut 169 (Mai "moves toward screen-left" while her arms cross) and one in cut 170 (Mai "leaves the frame" at the picture's edge). I did not change the labels to suit the rules.
4. **The camera is now right where it was wrong, at a price.** 1 wrong call in 60 labelled pairs against 5, but 5 more pairs say `unclear`, and in those the walking rule needs a larger shift and gives no size claim.
5. **Still unmeasured:** a seated character (the two seated characters of cut 162 have no legs in view) and a camera pan that follows a walker. The code for both is tested on synthetic keypoints only.

## Not done

- A fix for the 70 px far walker (cut 139) and for upper-body walkers needs a source of leg motion other than DWPose: Qwen's own `walking` reading, or optical flow of the box, neither tried.
- The judge was not run, so no judged precision exists for this pass.
- 16 of the 30 cuts have no hand labels (140–142 and 145–157 of the old set); their D claims were not scored.

# Targeted prompt test (run `mf-20261118-prompt`, 2026-11-18)

Question: can a prompt change fix the walking misses that the DWPose rules cannot reach (upper-body crops, a runner cut by the frame edge)? Two variants of the observer prompt, on the 14 hand-labelled cuts only (17 windows, 60 pairs), everything else fixed (same frames, roster context, new DWPose rules and merge):
- **`walker`** (style): observer + a definition of walking. "Walking" is yes when a character travels over the ground on foot (walking, running, stepping to another place, entering or leaving the picture, small or distant), judged from how the whole body moves, so also with the legs cropped out; no for turning, leaning, gesturing or shifting in place, being carried, pushed or falling, sitting or riding a vehicle.
- **`walker` + `roster_cam`** (context): the same, plus one line giving the camera measured from the background alone (people masked) for each pair of the window, "not measurable" where the estimate says `unclear`, and the instruction that a person who only seems to move because the camera moved has not changed place.

**Validation before the GPU run:** the 17 stored baseline prompts (observer/roster) are reproduced byte for byte by the current code, so the stored baseline answers are a valid control; the new prompts differ from the baseline only in the walking sentence and one camera line (diffed on a real window); 6 new tests (67 pass); an end-to-end dry run on the exact subset with the fake backend (34 calls, hybrids, report) passed. The real run made 34 calls, all valid on the first attempt, with no invented names.

**Agreed before the run:** adopt a variant only if F1 gains clearly more than the noise (about 10 points on this much data) or walking recall rises clearly. Neither happened.

| Setup (labelled cuts) | F1 (P / R) | Body F1 | Walking P / R | Speaking P / R | s per footage s |
|---|---|---|---|---|---|
| Qwen observer/roster (baseline) | 54 (70 / 44) | 51 | 80 / 53 (8 of 15) | 67 / 100 | 19.9 |
| Qwen walker/roster | 52 (68 / 42) | 54 | 82 / 60 (9) | 59 / 71 | 21.2 |
| Qwen walker/roster_cam | 56 (73 / 46) | 57 | 83 / 67 (10) | 57 / 86 | 20.6 |
| Hybrid observer/roster | 61 (66 / 56) | 64 | 80 / 53 | 67 / 100 | 20.6 |
| Hybrid walker/roster | 59 (65 / 55) | 66 | 82 / 60 | 59 / 71 | 21.9 |
| Hybrid walker/roster_cam | 62 (67 / 57) | 67 | 83 / 67 | 57 / 86 | 21.3 |

## What the test shows

1. **The definition does what it was written for, in a few pairs.** Walker recovers the two pairs of the runner cut by the frame edge in cut 165 (F1→F2, F2→F3). Walker + camera line also finds one more upper-body exit in cut 164 (F7→F8). Two of the three cropped exits of cut 164 and the far walker of cut 139 stay missed.
2. **It does not stop the false walks.** Cut 143 (Sakuta flung by the kick) is still called walking in both pairs by every variant, in spite of "pushed or falling" in the definition.
3. **The prompt change moves a lot of other claims, with no net direction.** Against the baseline, walker gains 9 true claims, loses 14, adds 9 false ones and drops 8; walker + camera gains 11, loses 9, adds 8 and drops 12. Speaking is the visible side effect: recall falls from 100% to 71% (walker) and 86% (walker + camera), mostly in the kick of cut 143, and expression claims vanish in walker. A rule about walking should not touch speaking, yet it does.
4. **So the F1 differences are inside the noise** (Qwen alone −2 and +2, hybrid −2 and +1). Walking recall rises by 1 and 2 pairs out of 15, which is not "clearly". The test does not support changing the production prompt; walker + camera is the only variant with a small positive sign everywhere (F1, body F1, walking, legs), and it costs 3% more compute per footage second.
5. **The camera column is circular for the camera line.** Qwen's camera agrees 100% with the pixel estimate once it is given that estimate; it was 86% before. The line also carries the one wrong measurement of the labelled set (a zoom-out in cut 143 labelled static).

## What this does not settle

- **The noise floor of an identical prompt is not measured.** The baseline answers are single runs from earlier sessions, and there is no replicate. The 9–14 claims that flip per variant say that any wording change shakes about a fifth of the claims; whether temperature 0 makes a re-run identical was not checked. A replicate of the baseline (about 15 minutes) would turn "inside the noise" into a number.
- **The labels and the prompt both come from me.** The walking definition was written after reading the misses on these same cuts, so even the small walking gain is in-sample. A batch of new, labelled cuts with walkers is the fair test; it is also the missing test for a seated character and a following pan.
- **Not tried:** an instruction on eyes/gaze precision (9%), a head/torso comparison step, few-shot images.
