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
