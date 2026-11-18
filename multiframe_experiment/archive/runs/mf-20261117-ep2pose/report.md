# Multi-frame experiment — mf-20261117-ep2pose

Status: **done** · updated 2026-11-18T02:26:55+00:00 · episode Episode 2 · cuts 139-158 · window 5 · image width 1024 · judge on

## Findings

- Ranked by hand-label F1 (true precision / recall): hybrid|observer|roster 55 (75% / 43%); hybrid|observer|frames 53 (71% / 43%); hybrid|inspector|roster 53 (71% / 43%); qwen35|observer|roster 53 (78% / 40%); hybrid|inspector|frames 52 (70% / 41%); qwen35|observer|frames 48 (72% / 37%); qwen35|inspector|roster 46 (69% / 35%); qwen35|inspector|frames 43 (67% / 32%); dwpose|rules|keypoints 38 (76% / 25%); baseline|production|video 19 (46% / 12%).
- Body parts only (walking, arms & hands, legs, body pose, position), hand-label F1: hybrid|observer|frames 54 (73% / 43%); hybrid|inspector|roster 53 (75% / 41%); hybrid|inspector|frames 51 (74% / 39%); dwpose|rules|keypoints 49 (76% / 36%); hybrid|observer|roster 49 (76% / 36%); qwen35|observer|frames 47 (75% / 34%); qwen35|observer|roster 46 (82% / 32%); qwen35|inspector|roster 42 (72% / 30%); qwen35|inspector|frames 37 (69% / 25%); baseline|production|video 17 (60% / 10%).
- Judge verdicts that agree with the hand labels: dwpose|rules|keypoints 59% of 27, hybrid|observer|frames 55% of 58, hybrid|observer|roster 67% of 48, hybrid|inspector|frames 68% of 56, hybrid|inspector|roster 68% of 57, qwen35|observer|frames 67% of 42, qwen35|observer|roster 83% of 41, qwen35|inspector|frames 84% of 43, qwen35|inspector|roster 82% of 49, baseline|production|video 86% of 14.
- Best overall: qwen35|observer|roster with 66.3 (basis: efficiency, precision, recall, reliability, speaking).
- Best for dwpose: rules/keypoints at 49.8; 1.52 s per window.
- Best for hybrid: observer/roster at 56.4; 54.38 s per window.
- Best for qwen35: observer/roster at 66.3; 52.86 s per window.
- Mean score by style: observer 60.8, inspector 59.6, rules 49.8.
- Mean score by model and context: dwpose/keypoints 49.8, hybrid/frames 55.6, hybrid/roster 56.0, qwen35/frames 64.2, qwen35/roster 65.0.
- Fewest false claims: qwen35|observer|roster (8% of judged claims false).
- Highest recall of the judge's reference changes: qwen35|observer|frames (56%).
- Best speaking agreement without being given the transcript: hybrid|inspector|frames (61%).
- Cost/quality frontier (each beats every cheaper config): dwpose|rules|keypoints → qwen35|observer|frames → qwen35|observer|roster.
- Cheapest within 5 points of the best: qwen35|observer|frames (64.7, 21.18 s compute per footage second).
- Stored Qwen3-VL video analysis (baseline): 31.7 on efficiency, precision, recall, speaking; 10.95 s per footage second.

## Leaderboard

| Config | GT F1 | GT precision | GT recall | GT body F1 | Score | Precision | Recall | Camera (diag.) | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid|observer|roster | 55% | 75% | 43% | 49% | 56.4 | 68% | 53% | 100% | 58% | 100% | 31% | 0/205 | 0 in 0 | 54.4 | 22.53 | —/— |
| hybrid|observer|frames | 53% | 71% | 43% | 54% | 55.7 | 66% | 55% | 100% | 57% | 100% | 34% | 3/220 | 0 in 0 | 52.6 | 21.81 | —/— |
| hybrid|inspector|roster | 53% | 71% | 43% | 53% | 55.7 | 65% | 54% | 100% | 60% | 100% | 34% | 0/201 | 0 in 0 | 55.5 | 22.99 | —/— |
| qwen35|observer|roster | 53% | 78% | 40% | 46% | 66.3 | 92% | 56% | 90% | 58% | 100% | 8% | 0/201 | 0 in 0 | 52.9 | 21.90 | 3533/1404 |
| hybrid|inspector|frames | 52% | 70% | 41% | 51% | 55.5 | 65% | 54% | 100% | 61% | 100% | 35% | 3/209 | 0 in 0 | 52.9 | 21.90 | —/— |
| qwen35|observer|frames | 48% | 72% | 37% | 47% | 64.7 | 88% | 56% | 90% | 57% | 100% | 11% | 3/205 | 0 in 0 | 51.1 | 21.18 | 3410/1347 |
| qwen35|inspector|roster | 46% | 69% | 35% | 42% | 63.6 | 86% | 53% | 90% | 60% | 100% | 14% | 0/198 | 0 in 0 | 54.0 | 22.36 | 3820/1431 |
| qwen35|inspector|frames | 43% | 67% | 32% | 37% | 63.8 | 87% | 52% | 90% | 61% | 100% | 13% | 3/195 | 0 in 0 | 51.3 | 21.27 | 3697/1350 |
| dwpose|rules|keypoints | 38% | 76% | 25% | 49% | 49.8 | 33% | 16% | 100% | — | 100% | 67% | 0/166 | 0 in 0 | 1.5 | 0.63 | —/— |
| baseline|production|video | 19% | 46% | 12% | 17% | 31.7 | 30% | 30% | 88% | 79% | — | 70% | 0/122 | 0 in 0 | — | 10.95 | —/— |

## Precision / recall by category

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | 50% / 100% | 32% / 86% | 100% / 15% | 32% / 45% | 27% / 5% | 7% / 24% | 100% / 25% | 40% / 23% | — / 0% | 0% / 2% |
| dwpose|rules|keypoints | 29% / 100% | — / 0% | — / 0% | 50% / 23% | — / 0% | 30% / 44% | 25% / 27% | 75% / 19% | 31% / 53% | — / 0% |
| hybrid|inspector|frames | 40% / 100% | 81% / 89% | 78% / 100% | 73% / 64% | 94% / 65% | 40% / 59% | 25% / 27% | 83% / 25% | 39% / 63% | 100% / 10% |
| hybrid|inspector|roster | 44% / 100% | 83% / 89% | 69% / 100% | 73% / 68% | 96% / 75% | 45% / 62% | 25% / 27% | 67% / 25% | 36% / 63% | 100% / 4% |
| hybrid|observer|frames | 50% / 100% | 82% / 89% | 65% / 100% | 70% / 64% | 100% / 50% | 43% / 59% | 33% / 27% | 92% / 38% | 40% / 53% | 92% / 21% |
| hybrid|observer|roster | 38% / 100% | 92% / 91% | 62% / 92% | 78% / 68% | 100% / 45% | 41% / 56% | 40% / 36% | 80% / 25% | 35% / 58% | 100% / 15% |
| qwen35|inspector|frames | 50% / 0% | 83% / 89% | 72% / 100% | 100% / 55% | 100% / 65% | 93% / 62% | 100% / 36% | 100% / 25% | 93% / 53% | 100% / 10% |
| qwen35|inspector|roster | 60% / 0% | 83% / 89% | 69% / 100% | 92% / 55% | 96% / 75% | 91% / 66% | 100% / 55% | 80% / 25% | 93% / 58% | 100% / 4% |
| qwen35|observer|frames | 86% / 100% | 87% / 89% | 56% / 100% | 93% / 50% | 100% / 50% | 89% / 69% | 67% / 27% | 100% / 50% | 100% / 53% | 100% / 21% |
| qwen35|observer|roster | 67% / 100% | 92% / 91% | 62% / 92% | 100% / 55% | 100% / 45% | 96% / 72% | 100% / 36% | 100% / 50% | 100% / 58% | 100% / 15% |

## Hand labels by category (true precision / recall)

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | — / 0% | 50% / 38% | — / — | 0% / 0% | — / 0% | — / 0% | 100% / 38% | 0% / 0% | — / 0% | — / 0% |
| dwpose|rules|keypoints | 25% / 50% | — / 0% | — / — | — / 0% | — / 0% | 100% / 67% | 60% / 27% | 100% / 11% | 100% / 38% | — / 0% |
| hybrid|inspector|frames | 20% / 50% | 67% / 100% | — / — | 50% / 14% | — / 0% | 100% / 67% | 60% / 27% | 100% / 22% | 100% / 38% | — / 0% |
| hybrid|inspector|roster | 20% / 50% | 80% / 100% | — / — | 50% / 14% | 0% / 0% | 100% / 78% | 60% / 27% | 100% / 22% | 100% / 38% | — / 0% |
| hybrid|observer|frames | 17% / 50% | 80% / 100% | — / — | 0% / 0% | 0% / 0% | 100% / 67% | 60% / 27% | 100% / 44% | 100% / 38% | — / 0% |
| hybrid|observer|roster | 25% / 50% | 80% / 100% | — / — | 50% / 14% | — / 0% | 100% / 67% | 60% / 27% | 100% / 11% | 100% / 38% | 67% / 67% |
| qwen35|inspector|frames | 0% / 0% | 67% / 100% | — / — | 50% / 14% | — / 0% | 100% / 33% | 100% / 27% | 50% / 11% | 100% / 31% | — / 0% |
| qwen35|inspector|roster | 0% / 0% | 80% / 100% | — / — | 50% / 14% | 0% / 0% | 100% / 33% | 100% / 45% | 67% / 22% | 100% / 23% | — / 0% |
| qwen35|observer|frames | 0% / 0% | 80% / 100% | — / — | 0% / 0% | 0% / 0% | 100% / 33% | 100% / 27% | 83% / 56% | 100% / 31% | — / 0% |
| qwen35|observer|roster | 0% / 0% | 80% / 100% | — / — | 50% / 14% | — / 0% | 100% / 56% | 100% / 27% | 75% / 33% | 100% / 23% | 67% / 67% |
| labelled pairs | 2 | 8 | 0 | 7 | 1 | 9 | 11 | 9 | 13 | 3 |

## Judge against the hand labels

Reference table: precision 76%, recall 51%, F1 61% (labels = truth).
Camera: judge agrees with the labels on 11/14 pairs, the pixel estimate on 12/12.

| Category | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| judge P / R | — / 0% | 100% / 88% | — / — | 50% / 14% | 0% / 0% | 100% / 67% | 100% / 55% | 57% / 44% | 100% / 38% | 43% / 100% |

| Config | Verdicts judged | Agree with labels | "true" on an unlabelled change | "false" on a labelled change |
|---|---|---|---|---|
| baseline|production|video | 14 | 86% | 1 | 1 |
| dwpose|rules|keypoints | 27 | 59% | 0 | 11 |
| hybrid|inspector|frames | 56 | 68% | 5 | 13 |
| hybrid|inspector|roster | 57 | 68% | 6 | 12 |
| hybrid|observer|frames | 58 | 55% | 6 | 20 |
| hybrid|observer|roster | 48 | 67% | 4 | 12 |
| qwen35|inspector|frames | 43 | 84% | 6 | 1 |
| qwen35|inspector|roster | 49 | 82% | 8 | 1 |
| qwen35|observer|frames | 42 | 67% | 7 | 7 |
| qwen35|observer|roster | 41 | 83% | 6 | 1 |

## DWPose on anime

Usable bodies (>= 6 of 17 body points over 0.3): 177/185 person-frames (96%); close-ups usable 112/117; suspected left/right swaps 0; 45.8 s wall on the CPU.

| Cut | Character | Keyframes | Usable | Mean body confidence | Swaps | Close-ups (usable) | Box source |
|---|---|---|---|---|---|---|---|
| c139 | Sakuta Azusagawa | 5 | 5 (100%) | 0.87 | 0 | 0 (0) | box:first:auto, box:last:auto, detector |
| c139 | Crying Lost Child | 5 | 5 (100%) | 0.60 | 0 | 5 (5) | box:first:manual, detector |
| c140 | Crying Lost Child | 10 | 10 (100%) | 0.65 | 0 | 0 (0) | box:first:manual, detector |
| c140 | unknown person 1 | 9 | 9 (100%) | 0.77 | 0 | 9 (9) | detector |
| c141 | Sakuta Azusagawa | 10 | 10 (100%) | 0.67 | 0 | 10 (10) | box:first:auto, detector |
| c141 | Crying Lost Child | 9 | 9 (100%) | 0.51 | 0 | 0 (0) | box:first:manual, detector |
| c142 | Tomoe Koga | 2 | 1 (50%) | 0.20 | 0 | 1 (1) | box:first:manual, detector |
| c142 | unknown person 1 | 2 | 1 (50%) | 0.40 | 0 | 2 (1) | detector |
| c143 | Tomoe Koga | 5 | 5 (100%) | 0.63 | 0 | 5 (5) | box:first:manual, detector |
| c143 | Sakuta Azusagawa | 5 | 5 (100%) | 0.66 | 0 | 4 (4) | box:first:manual, detector |
| c144 | unknown person 1 | 4 | 4 (100%) | 0.62 | 0 | 4 (4) | detector |
| c145 | Sakuta Azusagawa | 2 | 2 (100%) | 0.42 | 0 | 1 (1) | box:first:manual, box:last:auto |
| c145 | Tomoe Koga | 5 | 5 (100%) | 0.83 | 0 | 0 (0) | box:first:manual, box:last:manual, detector |
| c145 | Crying Lost Child | 1 | 1 (100%) | 0.63 | 0 | 0 (0) | box:first:manual |
| c146 | Sakuta Azusagawa | 5 | 4 (80%) | 0.37 | 0 | 3 (3) | box:first:manual, box:last:auto, detector |
| c147 | Crying Lost Child | 5 | 5 (100%) | 0.39 | 0 | 5 (5) | box:first:manual, detector |
| c147 | Tomoe Koga | 1 | 1 (100%) | 0.56 | 0 | 1 (1) | box:last:auto |
| c148 | Tomoe Koga | 6 | 6 (100%) | 0.74 | 0 | 6 (6) | box:first:manual, box:last:manual, detector |
| c148 | Crying Lost Child | 6 | 6 (100%) | 0.49 | 0 | 0 (0) | box:first:manual, detector |
| c148 | Sakuta Azusagawa | 6 | 6 (100%) | 0.71 | 0 | 3 (3) | box:first:manual, box:last:auto, detector |
| c149 | Tomoe Koga | 4 | 3 (75%) | 0.43 | 0 | 4 (3) | box:first:auto, detector |
| c149 | Crying Lost Child | 5 | 5 (100%) | 0.49 | 0 | 3 (3) | box:first:manual, detector |
| c150 | Sakuta Azusagawa | 10 | 10 (100%) | 0.69 | 0 | 10 (10) | box:first:auto, box:last:auto, detector |
| c150 | Tomoe Koga | 9 | 9 (100%) | 0.61 | 0 | 3 (3) | box:last:manual, detector |
| c151 | Tomoe Koga | 1 | 1 (100%) | 0.46 | 0 | 1 (1) | box:first:auto |
| c153 | Sakuta Azusagawa | 10 | 10 (100%) | 0.81 | 0 | 10 (10) | box:first:auto, detector |
| c153 | Tomoe Koga | 10 | 10 (100%) | 0.42 | 0 | 10 (10) | box:first:auto, detector |
| c153 | Crying Lost Child | 1 | 1 (100%) | 0.53 | 0 | 0 (0) | box:first:manual |
| c154 | Sakuta Azusagawa | 4 | 4 (100%) | 0.49 | 0 | 4 (4) | box:first:auto, detector |
| c154 | Tomoe Koga | 9 | 9 (100%) | 0.65 | 0 | 0 (0) | box:first:manual, detector |
| c155 | Sakuta Azusagawa | 4 | 4 (100%) | 0.43 | 0 | 4 (4) | box:first:auto, box:last:auto, detector |
| c156 | unknown person 1 | 2 | 0 (0%) | 0.13 | 0 | 2 (0) | box:first:?, box:last:? |
| c157 | Tomoe Koga | 3 | 2 (67%) | 0.27 | 0 | 3 (2) | box:first:manual, box:last:manual |
| c158 | Police Officer | 5 | 5 (100%) | 0.79 | 0 | 2 (2) | box:first:manual, detector |
| c158 | Sakuta Azusagawa | 2 | 2 (100%) | 0.74 | 0 | 2 (2) | box:last:manual, detector |
| c158 | Tomoe Koga | 2 | 2 (100%) | 0.36 | 0 | 0 (0) | box:last:manual, detector |
| c158 | unknown person 1 | 1 | 0 (0%) | 0.14 | 0 | 0 (0) | detector |

## How to read

Score weights: precision 35, recall 20, camera 0, speaking 10, reliability 10, efficiency 15 (missing components are left out and the rest renormalised).
GT = hand labels (`labels/<cut>.json`, made before any model answer): per pair and category, a reported category is a true positive when the label marks it. Precision and recall come from a blind Qwen3.5-27B judge (shares weights with one candidate). Camera is a diagnostic (weight 0) against an ORB/RANSAC pixel estimate; every setup's camera in the report is that estimate. Speaking is scored against transcript speech by a character marked on screen (boxed), and not at all for setups that cannot see mouths (DWPose).
