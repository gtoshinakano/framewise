# Target-frame evaluation, 2026-11-19

Phase A used the 13 active Episode 2 cuts with a surviving manual target box or slot label (113, 132, 133, 136, 137, 140, 145–148, 158, 165, 169). The app kept 43 manual boxes and two manual slot labels. The old schema did not keep the raw detector stream or deleted boxes. `scripts/target_eval.py capture` reproduced the raw stream from local AniRef and CCIP weights on CPU, read only eligible cached gallery references from other cuts, estimated camera motion from the captured frames, and ran DWPose on CPU. It did not write to the app database. The benchmark labels use the manually corrected boxes where they fit the visible person and an independent visual pass over the 73 frames to label the remaining people, extras, and merged detections. The 11 mirrored opening frames are excluded from the tracker score. These are in-sample measurements from the cuts used to tune the tracker.

## Tracking and corrections

The evaluator uses IoU ≥ 0.50 with the correct colour for a box match. Opening colours are fixed. New colours are mapped one-to-one across the whole cut, so choosing a different unused colour for a new person carries no penalty. The correction estimate counts one edit for a supported recolour or resize, and one per remaining addition or removal. It does not count a bulk removal as several actual clicks or assume that a model suggestion is correct. Manual recovery is a stricter count of the 43 edited boxes under their original colour numbers.

| Setup | Precision | Recall | F1 | Estimated corrections | Manual boxes reproduced |
|---|---:|---:|---:|---:|---:|
| Existing greedy tracker | 55.7% | 58.1% | 56.9% | 72 | 1 / 43 |
| Global assignment, camera and gallery | 90.9% | 76.9% | 83.3% | 28 | 18 / 43 |
| With the geometric splitter | 93.2% | 82.1% | 87.3% | 22 | 24 / 43 |
| With optical flow and DWPose recovery | 93.5% | 86.3% | 89.8% | 17 | 29 / 43 |

The last configuration reduced estimated corrections by **76.4%** (72 to 17). It flagged 41 of 62 non-opening frames for review. The geometric splitter proposed 10 source boxes, four of which were genuine merges (40% precision, 33% recall on 12 labelled merges). Three of those four had both child boxes in the right colours at the IoU threshold. The rest of the pipeline still evaluates the child boxes through identity, geometry and no-match costs.

## Qwen 3.6 judge

The installed Qwen 3.6 standard GGUF and vision projector returned valid enum answers for all 13 single-image contact sheets. The sheets used the first global-assignment and splitter replay, prior to the optical-flow calibration. They contained baseline errors for the judge to classify. Values below are true positives, false positives and false negatives. A dash means the denominator is zero.

| Answer | TP / FP / FN | Precision | Recall |
|---|---:|---:|---:|
| Cell `same` | 66 / 13 / 0 | 83.5% | 100% |
| Cell `different_person` | 0 / 0 / 0 | — | — |
| Cell `not_a_person` | 0 / 3 / 0 | 0% | — |
| Cell `merged_two_people` | 4 / 3 / 1 | 57.1% | 80% |
| Cell `unclear` | 0 / 0 / 18 | — | 0% |
| Unmatched `background_extra` | 0 / 1 / 17 | 0% | 0% |
| Unmatched `not_a_person` | 0 / 0 / 0 | — | — |
| Unmatched `new_character` | 8 / 20 / 12 | 28.6% | 40% |
| Unmatched `slot_1` | 1 / 16 / 4 | 5.9% | 20% |
| Unmatched `slot_2` | 1 / 1 / 0 | 50% | 100% |
| Unmatched `slot_3` | 0 / 0 / 5 | — | 0% |
| Cast changes `no` | 6 / 5 / 2 | 54.5% | 75% |
| Cast changes `yes` | 0 / 2 / 5 | 0% | 0% |

The 95% precision gate enabled **zero** Qwen actions. The judge missed every new entrant and provided too few correct placement suggestions to justify a live inference job. The production design therefore leaves cast locking under the user's control and omits the judged job. The contact-sheet builder and benchmark retain the evidence needed to revisit that decision.

## Keyframe analysis

Cuts 158, 165 and 169 had both archive hand labels and precisely matching frame ordinals. The paired comparison ran the same installed Qwen 3.6 and DWPose stack, production observer prompt, frame set, window sizes and scoring categories in both arms. The corrected arm used the stored target frames from the cuts with user corrections. The opening-only arm used their first-frame boxes and the normal unanchored DWPose detection after that. All 13 adjacent pairs had results in both arms.

| Anchors | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Opening frame only | 17 / 18 / 12 | 48.6% | 58.6% | 53.1% |
| Corrected target frames | 19 / 19 / 10 | 50.0% | 65.5% | 56.7% |

## Reproduction

- `scripts/target_eval.py inventory --db /data/app.sqlite3` identifies the eligible cuts. Run `capture --output /scratch/capture.jsonl --images /scratch/images` in the inference environment. The capture can resume per cut.
- `PYTHONPATH=backend python3 scripts/target_eval.py replay --capture /scratch/capture.jsonl --labels docs/target-eval/frame-labels.json --candidate-labels docs/target-eval/candidate-labels.json --images /scratch/images` compares the tracker variants and scores the splitter. The final arm needs the inference environment's OpenCV.
- `bash scripts/run-target-vision-eval.sh /scratch` runs the paired experiments with an idle inference worker and restores it afterward. `scripts/target_vision_eval.py motion-scores --scratch /scratch` uses `multiframe_experiment` hand labels. `judge-scores --scratch /scratch --labels docs/target-eval/judge-labels.json` scores the Phase A crop IDs. Both scoring commands accept `--results docs/target-eval/vision-results.jsonl` to use the archived Qwen answers. A regenerated task set after tracker changes needs fresh crop labels.
