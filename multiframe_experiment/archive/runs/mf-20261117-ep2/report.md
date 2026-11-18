# Multi-frame experiment — mf-20261117-ep2

Status: **stopped** · updated 2026-11-18T02:26:55+00:00 · episode Episode 2 · cuts 139-158 · window 5 · image width 1024 · judge on

## Findings

- Run without the judge: no score and no judged precision. Compute per footage second: lfm3|animator|frames 6.3 s, lfm3|animator|full 4.99 s, lfm3|animator|roster 4.22 s, lfm3|observer|frames 4.96 s, lfm3|observer|full 3.35 s, lfm3|observer|roster 2.79 s (cheapest lfm3|observer|roster).

## Leaderboard

| Config | GT F1 | GT precision | GT recall | GT body F1 | Score | Precision | Recall | Camera (diag.) | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lfm3|observer|roster | — | — | — | — | 78.7 | — | — | 87% | 25% | 100% | — | 0/100 | 0 in 0 | 6.0 | 2.79 | 3392/635 |
| lfm3|observer|full | — | — | — | — | 76.5 | — | — | 88% | 43% (told) | 100% | — | 8/103 | 0 in 0 | 7.2 | 3.35 | 3800/773 |
| lfm3|animator|full | — | — | — | — | 62.9 | — | — | 98% | 40% (told) | 96% | — | 8/87 | 0 in 0 | 10.8 | 4.99 | 3995/1092 |
| lfm3|observer|frames | — | — | — | — | 57.2 | — | — | 87% | 23% | 92% | — | 4/84 | 0 in 0 | 10.7 | 4.96 | 3721/1010 |
| lfm3|animator|roster | — | — | — | — | 54.7 | — | — | 86% | 0% | 92% | — | 0/74 | 0 in 0 | 9.1 | 4.22 | 3719/858 |
| lfm3|animator|frames | — | — | — | — | 48.2 | — | — | 86% | 15% | 87% | — | 15/71 | 0 in 0 | 14.3 | 6.30 | 3884/1237 |
| baseline|production|video | — | — | — | — | 46.4 | — | — | 88% | 78% | — | — | 4/122 | 0 in 0 | — | 10.95 | —/— |

## Precision / recall by category

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| lfm3|animator|frames | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| lfm3|animator|full | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| lfm3|animator|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| lfm3|observer|frames | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| lfm3|observer|full | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |
| lfm3|observer|roster | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — | — / — |

## How to read

Score weights: precision 35, recall 20, camera 0, speaking 10, reliability 10, efficiency 15 (missing components are left out and the rest renormalised).
GT = hand labels (`labels/<cut>.json`, made before any model answer): per pair and category, a reported category is a true positive when the label marks it. Precision and recall come from a blind Qwen3.5-27B judge (shares weights with one candidate). Camera is a diagnostic (weight 0) against an ORB/RANSAC pixel estimate; every setup's camera in the report is that estimate. Speaking is scored against transcript speech by a character marked on screen (boxed), and not at all for setups that cannot see mouths (DWPose).
