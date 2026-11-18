# Multi-frame experiment — mf-20261117-pose

Status: **done** · updated 2026-11-18T02:26:55+00:00 · episode Episode 2 · cuts 139,143,144,158 · window 5 · image width 1024 · judge on

## Findings

- Ranked by hand-label F1 (true precision / recall): hybrid|dwpose+qwen|frames 57 (70% / 48%); qwen35|observer|roster 53 (78% / 40%); qwen35|observer|frames 48 (72% / 37%); dwpose|rules|keypoints 43 (76% / 30%); baseline|production|video 19 (46% / 12%).
- Body parts only (walking, arms & hands, legs, body pose, position), hand-label F1: hybrid|dwpose+qwen|frames 59 (71% / 50%); dwpose|rules|keypoints 55 (76% / 43%); qwen35|observer|frames 47 (75% / 34%); qwen35|observer|roster 46 (82% / 32%); baseline|production|video 17 (60% / 10%).
- Judge verdicts that agree with the hand labels: dwpose|rules|keypoints 46% of 37, hybrid|dwpose+qwen|frames 52% of 69, qwen35|observer|frames 74% of 43, qwen35|observer|roster 76% of 42, baseline|production|video 80% of 15.
- Best overall: qwen35|observer|roster with 63.7 (basis: efficiency, precision, recall, reliability, speaking).
- Best for dwpose: rules/keypoints at 52.2; 1.9 s per window.
- Best for hybrid: dwpose+qwen/frames at 48.7; 60.15 s per window.
- Best for qwen35: observer/roster at 63.7; 60.75 s per window.
- Mean score by style: observer 62.2, rules 52.2, dwpose+qwen 48.7.
- Mean score by model and context: dwpose/keypoints 52.2, hybrid/frames 48.7, qwen35/frames 60.7, qwen35/roster 63.7.
- Fewest false claims: qwen35|observer|roster (16% of judged claims false).
- Highest recall of the judge's reference changes: qwen35|observer|roster (53%).
- Best speaking agreement without being given the transcript: hybrid|dwpose+qwen|frames (70%).
- Cost/quality frontier (each beats every cheaper config): dwpose|rules|keypoints → qwen35|observer|frames → qwen35|observer|roster.
- Cheapest within 5 points of the best: qwen35|observer|frames (60.7, 24.09 s compute per footage second).
- Stored Qwen3-VL video analysis (baseline): 23.0 on efficiency, precision, recall, speaking; 25.9 s per footage second.

## Leaderboard

| Config | GT F1 | GT precision | GT recall | GT body F1 | Score | Precision | Recall | Camera (diag.) | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid|dwpose+qwen|frames | 57% | 70% | 48% | 59% | 48.7 | 49% | 47% | 100% | 70% | 100% | 51% | 0/41 | 0 in 0 | 60.1 | 24.88 | —/— |
| qwen35|observer|roster | 53% | 78% | 40% | 46% | 63.7 | 84% | 53% | 83% | 70% | 100% | 16% | 0/31 | 0 in 0 | 60.8 | 25.12 | 3835/1621 |
| qwen35|observer|frames | 48% | 72% | 37% | 47% | 60.7 | 80% | 47% | 83% | 70% | 100% | 20% | 0/33 | 0 in 0 | 58.2 | 24.09 | 3726/1546 |
| dwpose|rules|keypoints | 43% | 76% | 30% | 55% | 52.2 | 35% | 22% | 100% | — | 100% | 65% | 0/29 | 0 in 0 | 1.9 | 0.78 | —/— |
| baseline|production|video | 19% | 46% | 12% | 17% | 23.0 | 20% | 17% | 50% | 75% | — | 80% | 0/15 | 0 in 0 | — | 25.90 | —/— |

## Precision / recall by category

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | — / — | 14% / 29% | — / — | 0% / 0% | — / 0% | — / 0% | 67% / 100% | 0% / 25% | — / — | — / 0% |
| dwpose|rules|keypoints | 0% / — | — / 0% | — / — | — / 0% | — / 0% | 29% / 57% | 50% / 33% | 25% / 38% | 50% / 25% | — / 0% |
| hybrid|dwpose+qwen|frames | 17% / — | 44% / 100% | — / — | 100% / 25% | 100% / 50% | 33% / 57% | 57% / 33% | 70% / 62% | 60% / 25% | — / 0% |
| qwen35|observer|frames | 25% / — | 69% / 100% | — / — | 100% / 25% | 100% / 50% | 67% / 43% | 100% / 50% | 100% / 62% | 100% / 25% | — / 0% |
| qwen35|observer|roster | 33% / — | 71% / 100% | — / — | 100% / 50% | — / 0% | 80% / 71% | 100% / 50% | 100% / 50% | 100% / 25% | 100% / 29% |

## Hand labels by category (true precision / recall)

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | — / 0% | 50% / 38% | — / — | 0% / 0% | — / 0% | — / 0% | 100% / 38% | 0% / 0% | — / 0% | — / 0% |
| dwpose|rules|keypoints | 50% / 50% | — / 0% | — / — | — / 0% | — / 0% | 75% / 67% | 60% / 27% | 67% / 22% | 100% / 54% | — / 0% |
| hybrid|dwpose+qwen|frames | 20% / 50% | 80% / 100% | — / — | 0% / 0% | 0% / 0% | 75% / 67% | 60% / 27% | 83% / 56% | 100% / 54% | — / 0% |
| qwen35|observer|frames | 0% / 0% | 80% / 100% | — / — | 0% / 0% | 0% / 0% | 100% / 33% | 100% / 27% | 83% / 56% | 100% / 31% | — / 0% |
| qwen35|observer|roster | 0% / 0% | 80% / 100% | — / — | 50% / 14% | — / 0% | 100% / 56% | 100% / 27% | 75% / 33% | 100% / 23% | 67% / 67% |
| labelled pairs | 2 | 8 | 0 | 7 | 1 | 9 | 11 | 9 | 13 | 3 |

## Judge against the hand labels

Reference table: precision 76%, recall 54%, F1 63% (labels = truth).
Camera: judge agrees with the labels on 11/14 pairs, the pixel estimate on 12/12.

| Category | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| judge P / R | — / 0% | 100% / 88% | — / — | 75% / 43% | 0% / 0% | 100% / 78% | 83% / 45% | 62% / 56% | 100% / 31% | 43% / 100% |

| Config | Verdicts judged | Agree with labels | "true" on an unlabelled change | "false" on a labelled change |
|---|---|---|---|---|
| baseline|production|video | 15 | 80% | 0 | 3 |
| dwpose|rules|keypoints | 37 | 46% | 2 | 18 |
| hybrid|dwpose+qwen|frames | 69 | 52% | 6 | 27 |
| qwen35|observer|frames | 43 | 74% | 5 | 6 |
| qwen35|observer|roster | 42 | 76% | 6 | 4 |

## DWPose on anime

Usable bodies (>= 6 of 17 body points over 0.3): 33/34 person-frames (97%); close-ups usable 22/22; suspected left/right swaps 0; 7.6 s wall on the CPU.

| Cut | Character | Keyframes | Usable | Mean body confidence | Swaps | Close-ups (usable) | Box source |
|---|---|---|---|---|---|---|---|
| c139 | Sakuta Azusagawa | 5 | 5 (100%) | 0.87 | 0 | 0 (0) | box:first:auto, box:last:auto, detector |
| c139 | Crying Lost Child | 5 | 5 (100%) | 0.60 | 0 | 5 (5) | box:first:manual, detector |
| c143 | Tomoe Koga | 5 | 5 (100%) | 0.63 | 0 | 5 (5) | box:first:manual, detector |
| c143 | Sakuta Azusagawa | 5 | 5 (100%) | 0.66 | 0 | 4 (4) | box:first:manual, detector |
| c144 | unknown person 1 | 4 | 4 (100%) | 0.62 | 0 | 4 (4) | detector |
| c158 | Police Officer | 5 | 5 (100%) | 0.79 | 0 | 2 (2) | box:first:manual, detector |
| c158 | Sakuta Azusagawa | 2 | 2 (100%) | 0.74 | 0 | 2 (2) | box:last:manual, detector |
| c158 | Tomoe Koga | 2 | 2 (100%) | 0.36 | 0 | 0 (0) | box:last:manual, detector |
| c158 | unknown person 1 | 1 | 0 (0%) | 0.14 | 0 | 0 (0) | detector |

## How to read

Score weights: precision 35, recall 20, camera 0, speaking 10, reliability 10, efficiency 15 (missing components are left out and the rest renormalised).
GT = hand labels (`labels/<cut>.json`, made before any model answer): per pair and category, a reported category is a true positive when the label marks it. Precision and recall come from a blind Qwen3.5-27B judge (shares weights with one candidate). Camera is a diagnostic (weight 0) against an ORB/RANSAC pixel estimate; every setup's camera in the report is that estimate. Speaking is scored against transcript speech by a character marked on screen (boxed), and not at all for setups that cannot see mouths (DWPose).
