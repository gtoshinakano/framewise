# Multi-frame experiment — mf-20261118-prompt

Status: **done** · updated 2026-11-18T02:26:55+00:00 · episode Episode 2 · cuts c139, c143, c144, c158, c160, c162, c163, c164, c165, c166, c167, c168, c169, c170 · window 5 · image width 1024 · judge off

## Findings

- Ranked by hand-label F1 (true precision / recall): hybrid|walker|roster_cam 62 (67% / 57%); hybrid|observer|roster 61 (66% / 56%); hybrid|walker|roster 59 (65% / 55%); qwen35|walker|roster_cam 56 (73% / 46%); qwen35|observer|roster 54 (70% / 44%); qwen35|walker|roster 52 (68% / 42%); dwpose|rules|keypoints 39 (70% / 28%); baseline|production|video 31 (40% / 25%).
- Body parts only (walking, arms & hands, legs, body pose, position), hand-label F1: hybrid|walker|roster_cam 67 (77% / 59%); hybrid|walker|roster 66 (77% / 58%); hybrid|observer|roster 64 (77% / 54%); qwen35|walker|roster_cam 57 (86% / 43%); qwen35|walker|roster 54 (85% / 40%); qwen35|observer|roster 51 (84% / 37%); dwpose|rules|keypoints 50 (76% / 37%); baseline|production|video 28 (57% / 19%).
- Run without the judge: no score and no judged precision. Compute per footage second: dwpose|rules|keypoints 0.71 s, hybrid|observer|roster 20.64 s, hybrid|walker|roster 21.93 s, hybrid|walker|roster_cam 21.3 s, qwen35|observer|roster 19.93 s, qwen35|walker|roster 21.22 s, qwen35|walker|roster_cam 20.6 s (cheapest dwpose|rules|keypoints).

## Leaderboard

| Config | GT F1 | GT precision | GT recall | GT body F1 | Score | Precision | Recall | Camera (diag.) | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid|walker|roster_cam | 62% | 67% | 57% | 67% | 46.4 | — | — | 100% | 57% | 100% | — | 0/130 | 0 in 0 | 56.1 | 21.30 | —/— |
| hybrid|observer|roster | 61% | 66% | 56% | 64% | 46.7 | — | — | 100% | 58% | 100% | — | 0/117 | 0 in 0 | 54.4 | 20.64 | —/— |
| hybrid|walker|roster | 59% | 65% | 55% | 66% | 45.2 | — | — | 100% | 53% | 100% | — | 0/144 | 0 in 0 | 57.8 | 21.93 | —/— |
| qwen35|walker|roster_cam | 56% | 73% | 46% | 57% | 46.5 | — | — | 100% | 57% | 100% | — | 0/116 | 0 in 0 | 54.3 | 20.60 | 3749/1438 |
| qwen35|observer|roster | 54% | 70% | 44% | 51% | 46.8 | — | — | 86% | 58% | 100% | — | 0/101 | 0 in 0 | 52.5 | 19.93 | 3559/1391 |
| qwen35|walker|roster | 52% | 68% | 42% | 54% | 45.2 | — | — | 86% | 53% | 100% | — | 0/131 | 0 in 0 | 55.9 | 21.22 | 3668/1486 |
| dwpose|rules|keypoints | 39% | 70% | 28% | 50% | 100.0 | — | — | 100% | — | 100% | — | 0/147 | 0 in 0 | 1.9 | 0.71 | —/— |
| baseline|production|video | 31% | 40% | 25% | 28% | 34.8 | — | — | 87% | 75% | — | — | 0/72 | 0 in 0 | — | 8.85 | —/— |

## Precision / recall by category

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| dwpose|rules|keypoints | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| hybrid|observer|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| hybrid|walker|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| hybrid|walker|roster_cam | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| qwen35|observer|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| qwen35|walker|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| qwen35|walker|roster_cam | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |

## Hand labels by category (true precision / recall)

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | 83% / 67% | 22% / 64% | 29% / 40% | 60% / 32% | — / 0% | 33% / 19% | 100% / 20% | 17% / 6% | — / 0% | — / 0% |
| dwpose|rules|keypoints | 75% / 20% | — / 0% | — / 0% | 46% / 21% | — / 0% | 85% / 55% | 71% / 28% | 67% / 10% | 74% / 57% | — / 0% |
| hybrid|observer|roster | 80% / 53% | 67% / 100% | 100% / 100% | 55% / 41% | 9% / 25% | 88% / 75% | 70% / 39% | 60% / 30% | 77% / 67% | 40% / 40% |
| hybrid|walker|roster | 82% / 60% | 59% / 71% | 83% / 100% | 52% / 38% | 17% / 50% | 83% / 75% | 77% / 56% | 60% / 30% | 77% / 67% | 0% / 0% |
| hybrid|walker|roster_cam | 83% / 67% | 57% / 86% | 100% / 100% | 55% / 41% | 12% / 25% | 88% / 75% | 82% / 50% | 54% / 35% | 77% / 67% | 50% / 20% |
| qwen35|observer|roster | 80% / 53% | 67% / 100% | 100% / 100% | 73% / 38% | 9% / 25% | 100% / 40% | 75% / 17% | 60% / 30% | 100% / 43% | 40% / 40% |
| qwen35|walker|roster | 82% / 60% | 59% / 71% | 83% / 100% | 64% / 31% | 17% / 50% | 91% / 50% | 86% / 33% | 57% / 20% | 100% / 40% | 0% / 0% |
| qwen35|walker|roster_cam | 83% / 67% | 57% / 86% | 100% / 100% | 77% / 34% | 12% / 25% | 100% / 45% | 100% / 33% | 55% / 30% | 100% / 43% | 50% / 20% |
| labelled pairs | 15 | 14 | 5 | 29 | 4 | 20 | 18 | 20 | 30 | 5 |

## Judge against the hand labels

Reference table: precision —, recall —, F1 — (labels = truth).
Camera: judge agrees with the labels on 0/0 pairs, the pixel estimate on 48/49.

| Category | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| judge P / R | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |

| Config | Verdicts judged | Agree with labels | "true" on an unlabelled change | "false" on a labelled change |
|---|---|---|---|---|

## DWPose on anime

Usable bodies (>= 6 of 17 body points over 0.3): 162/171 person-frames (95%); close-ups usable 58/59; suspected left/right swaps 0; 31.7 s wall on the CPU.

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
| c160 | Sakuta Azusagawa | 5 | 5 (100%) | 0.52 | 0 | 5 (5) | box:first:auto, detector |
| c160 | Tomoe Koga | 4 | 4 (100%) | 0.40 | 0 | 0 (0) | box:first:manual, detector |
| c162 | Sakuta Azusagawa | 5 | 5 (100%) | 0.48 | 0 | 1 (1) | box:first:auto, box:last:auto, detector |
| c162 | Tomoe Koga | 5 | 5 (100%) | 0.52 | 0 | 0 (0) | box:first:manual, box:last:manual, detector |
| c162 | unknown person 1 | 5 | 5 (100%) | 0.72 | 0 | 2 (2) | detector |
| c163 | Sakuta Azusagawa | 5 | 5 (100%) | 0.55 | 0 | 5 (5) | box:first:auto, detector |
| c163 | Tomoe Koga | 5 | 5 (100%) | 0.52 | 0 | 0 (0) | box:first:auto, detector |
| c164 | Sakuta Azusagawa | 7 | 7 (100%) | 0.49 | 0 | 7 (7) | box:first:auto, detector |
| c164 | Tomoe Koga | 9 | 9 (100%) | 0.44 | 0 | 1 (1) | box:first:auto, detector |
| c165 | unknown person 1 | 5 | 5 (100%) | 0.66 | 0 | 0 (0) | detector |
| c165 | unknown person 2 | 4 | 4 (100%) | 0.90 | 0 | 0 (0) | detector |
| c165 | unknown person 3 | 5 | 5 (100%) | 0.65 | 0 | 0 (0) | detector |
| c165 | unknown person 4 | 5 | 5 (100%) | 0.57 | 0 | 0 (0) | detector |
| c165 | unknown person 5 | 4 | 3 (75%) | 0.64 | 0 | 1 (1) | detector |
| c165 | unknown person 6 | 5 | 5 (100%) | 0.54 | 0 | 0 (0) | detector |
| c165 | unknown person 7 | 4 | 4 (100%) | 0.66 | 0 | 2 (2) | detector |
| c166 | unknown person 1 | 5 | 5 (100%) | 0.82 | 0 | 0 (0) | detector |
| c166 | unknown person 2 | 4 | 4 (100%) | 0.79 | 0 | 0 (0) | detector |
| c166 | unknown person 3 | 5 | 5 (100%) | 0.83 | 0 | 0 (0) | detector |
| c166 | unknown person 4 | 4 | 4 (100%) | 0.75 | 0 | 0 (0) | detector |
| c166 | unknown person 5 | 5 | 1 (20%) | 0.35 | 0 | 0 (0) | detector |
| c167 | unknown person 1 | 3 | 3 (100%) | 0.58 | 0 | 0 (0) | box:first:?, detector |
| c168 | Sakuta Azusagawa | 5 | 5 (100%) | 0.49 | 0 | 5 (5) | box:first:auto, box:last:auto, detector |
| c169 | Mai Sakurajima | 6 | 5 (83%) | 0.45 | 0 | 6 (5) | box:first:manual, box:last:manual, detector |
| c169 | unknown person 1 | 6 | 5 (83%) | 0.50 | 0 | 0 (0) | detector |
| c169 | unknown person 2 | 6 | 5 (83%) | 0.65 | 0 | 1 (1) | detector |
| c170 | Sakuta Azusagawa | 5 | 5 (100%) | 0.50 | 0 | 0 (0) | box:first:auto, detector |
| c170 | Mai Sakurajima | 1 | 1 (100%) | 0.36 | 0 | 1 (1) | box:first:manual |

## How to read

Score weights: precision 35, recall 20, camera 0, speaking 10, reliability 10, efficiency 15 (missing components are left out and the rest renormalised).
GT = hand labels (`labels/<cut>.json`, made before any model answer): per pair and category, a reported category is a true positive when the label marks it. Precision and recall come from a blind Qwen3.5-27B judge (shares weights with one candidate). Camera is a diagnostic (weight 0) against an ORB/RANSAC pixel estimate; every setup's camera in the report is that estimate. Speaking is scored against transcript speech by a character marked on screen (boxed), and not at all for setups that cannot see mouths (DWPose).
