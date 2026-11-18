# Multi-frame experiment — mf-20261118-c160-170

Status: **done** · updated 2026-11-18T02:26:55+00:00 · episode Episode 2 · cuts 160,162-170 · window 5 · image width 1024 · judge off

## Findings

- Ranked by hand-label F1 (true precision / recall): qwen35|observer|roster 55 (66% / 47%); hybrid|observer|roster 55 (56% / 54%); dwpose|rules|keypoints 36 (57% / 27%); baseline|production|video 35 (39% / 32%).
- Body parts only (walking, arms & hands, legs, body pose, position), hand-label F1: hybrid|observer|roster 56 (66% / 49%); qwen35|observer|roster 55 (86% / 41%); dwpose|rules|keypoints 43 (61% / 34%); baseline|production|video 33 (56% / 24%).
- Run without the judge: no score and no judged precision. Compute per footage second: dwpose|rules|keypoints 0.69 s, hybrid|observer|roster 19.18 s, qwen35|observer|roster 18.49 s (cheapest dwpose|rules|keypoints).

## Leaderboard

| Config | GT F1 | GT precision | GT recall | GT body F1 | Score | Precision | Recall | Camera (diag.) | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen35|observer|roster | 55% | 66% | 47% | 55% | 48.6 | — | — | 82% | 64% | 100% | — | 0/70 | 0 in 0 | 50.0 | 18.49 | 3475/1321 |
| hybrid|observer|roster | 55% | 56% | 54% | 56% | 48.5 | — | — | 100% | 64% | 100% | — | 0/95 | 0 in 0 | 51.8 | 19.18 | —/— |
| dwpose|rules|keypoints | 36% | 57% | 27% | 43% | 100.0 | — | — | 100% | — | 100% | — | 0/118 | 0 in 0 | 1.9 | 0.69 | —/— |
| baseline|production|video | 35% | 39% | 32% | 33% | 38.9 | — | — | 87% | 75% | — | — | 0/57 | 0 in 0 | — | 4.66 | —/— |

## Precision / recall by category

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| dwpose|rules|keypoints | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| hybrid|observer|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| qwen35|observer|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |

## Hand labels by category (true precision / recall)

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | 83% / 77% | 18% / 100% | 29% / 40% | 69% / 41% | — / 0% | 33% / 27% | — / 0% | 25% / 9% | — / 0% | — / 0% |
| dwpose|rules|keypoints | 100% / 23% | — / 0% | — / 0% | 46% / 27% | — / 0% | 50% / 45% | — / 0% | 50% / 9% | 61% / 65% | — / 0% |
| hybrid|observer|roster | 100% / 62% | 55% / 100% | 100% / 100% | 55% / 50% | 9% / 33% | 58% / 64% | 0% / 0% | 60% / 27% | 61% / 65% | 0% / 0% |
| qwen35|observer|roster | 100% / 62% | 55% / 100% | 100% / 100% | 77% / 45% | 9% / 33% | 100% / 27% | 0% / 0% | 50% / 27% | 100% / 59% | 0% / 0% |
| labelled pairs | 13 | 6 | 5 | 22 | 3 | 11 | 7 | 11 | 17 | 2 |

## Judge against the hand labels

Reference table: precision —, recall —, F1 — (labels = truth).
Camera: judge agrees with the labels on 0/0 pairs, the pixel estimate on 35/40.

| Category | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| judge P / R | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |

| Config | Verdicts judged | Agree with labels | "true" on an unlabelled change | "false" on a labelled change |
|---|---|---|---|---|

## DWPose on anime

Usable bodies (>= 6 of 17 body points over 0.3): 129/137 person-frames (94%); close-ups usable 36/37; suspected left/right swaps 0; 24.1 s wall on the CPU.

| Cut | Character | Keyframes | Usable | Mean body confidence | Swaps | Close-ups (usable) | Box source |
|---|---|---|---|---|---|---|---|
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
