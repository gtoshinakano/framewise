# Multi-frame experiment — mf-20261117-4cuts

Status: **done** · updated 2026-11-18T02:26:54+00:00 · episode Episode 2 · cuts 139,145,147,151 · window 5 · image width 1024 · judge on

## Findings

- Best overall: qwen35|observer|frames with 66.6 (basis: efficiency, precision, recall, reliability, speaking).
- Best for lfm3: observer/full at 54.9; 13.43 s per window.
- Best for qwen35: observer/frames at 66.6; 54.03 s per window.
- Mean score by style: observer 56.6, animator 48.8.
- Mean score by model and context: lfm3/frames 44.3, lfm3/full 50.5, lfm3/roster 28.8, qwen35/frames 65.5, qwen35/full 61.8, qwen35/roster 65.2.
- Fewest false claims: qwen35|observer|frames (12% of judged claims false).
- Highest recall of the judge's reference changes: qwen35|animator|frames (66%).
- Best speaking agreement without being given the transcript: qwen35|animator|roster (68%).
- Cost/quality frontier (each beats every cheaper config): lfm3|observer|frames → lfm3|observer|full → qwen35|observer|frames.
- Cheapest within 5 points of the best: qwen35|observer|frames (66.6, 21.95 s compute per footage second).
- Stored Qwen3-VL video analysis (baseline): 31.8 on efficiency, precision, recall, speaking; 21.48 s per footage second.

## Leaderboard

| Config | GT F1 | GT precision | GT recall | GT body F1 | Score | Precision | Recall | Camera (diag.) | Speaking | Reliability | False claims | Invented names | Claims on still pairs | s/window | s per footage s | Tokens in/out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen35|observer|frames | — | — | — | — | 66.6 | 88% | 53% | 71% | 58% | 100% | 12% | 0/32 | 0 in 0 | 54.0 | 21.95 | 3726/1413 |
| qwen35|observer|roster | — | — | — | — | 65.6 | 87% | 50% | 71% | 58% | 100% | 13% | 0/32 | 0 in 0 | 56.2 | 22.85 | 3857/1490 |
| qwen35|animator|roster | — | — | — | — | 64.9 | 78% | 59% | 71% | 68% | 100% | 22% | 0/32 | 0 in 0 | 58.8 | 23.89 | 3841/1561 |
| qwen35|animator|frames | — | — | — | — | 64.4 | 82% | 66% | 71% | 36% | 100% | 18% | 0/32 | 0 in 0 | 59.0 | 23.98 | 3710/1553 |
| qwen35|animator|full | — | — | — | — | 61.9 | 70% | 53% | 71% | 79% (told) | 100% | 30% | 0/32 | 0 in 0 | 59.7 | 24.27 | 4228/1578 |
| qwen35|observer|full | — | — | — | — | 61.7 | 73% | 53% | 71% | 68% (told) | 100% | 27% | 0/32 | 0 in 0 | 59.7 | 24.27 | 4244/1579 |
| lfm3|observer|full | — | — | — | — | 54.9 | 59% | 22% | 67% | 29% (told) | 100% | 40% | 0/40 | 0 in 0 | 13.4 | 5.46 | 4219/1490 |
| lfm3|observer|frames | — | — | — | — | 48.7 | 38% | 16% | 67% | 25% | 100% | 60% | 0/32 | 0 in 0 | 10.2 | 4.16 | 3724/1127 |
| lfm3|animator|full | — | — | — | — | 46.1 | 55% | 6% | 67% | 0% (told) | 100% | 40% | 0/48 | 0 in 0 | 14.1 | 5.73 | 4207/1570 |
| lfm3|observer|roster | — | — | — | — | 42.0 | 28% | 19% | 71% | 0% | 100% | 75% | 0/32 | 0 in 0 | 10.8 | 4.41 | 3838/1197 |
| lfm3|animator|frames | — | — | — | — | 39.9 | 31% | 27% | 50% | 42% | 75% | 72% | 0/20 | 0 in 0 | 19.1 | 7.76 | 4276/1397 |
| baseline|production|video | — | — | — | — | 31.8 | 29% | 25% | 71% | 75% | — | 71% | 0/17 | 0 in 0 | — | 21.48 | —/— |
| lfm3|animator|roster | — | — | — | — | 15.5 | 0% | 0% | 100% | 0% | 75% | 100% | 0/32 | 0 in 0 | 23.8 | 9.67 | 4631/1773 |

## Precision / recall by category

| Config | Walking | Speaking | Blinking | Head angle | Eyes / gaze | Arms & hands | Legs | Body pose | Position in frame | Expression |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline|production|video | 0% / — | 42% / 60% | — / 0% | 25% / 100% | 14% / 0% | 25% / 33% | — / — | — / 0% | — / 0% | — / 0% |
| lfm3|animator|frames | — / — | 25% / 44% | 100% / 50% | — / — | 40% / — | 25% / 100% | — / — | — / — | — / 0% | — / 0% |
| lfm3|animator|full | — / — | — / 0% | 100% / 33% | — / 0% | 33% / 0% | 33% / 33% | — / — | — / 0% | — / 0% | — / 0% |
| lfm3|animator|roster | — / — | — / 0% | — / 0% | — / 0% | — / 0% | — / 0% | — / — | — / 0% | — / — | — / 0% |
| lfm3|observer|frames | — / — | 29% / 30% | 100% / 33% | 0% / 0% | 100% / 0% | — / 0% | — / — | — / 0% | — / 0% | 100% / 9% |
| lfm3|observer|full | — / — | 57% / 50% | 100% / 33% | — / 0% | 0% / 0% | 100% / 33% | — / — | — / 0% | — / 0% | — / 0% |
| lfm3|observer|roster | — / — | 28% / 50% | — / 0% | — / 0% | — / 0% | 50% / 33% | — / — | — / 0% | — / 0% | — / 0% |
| qwen35|animator|frames | 50% / — | 85% / 100% | 50% / 100% | 100% / 100% | 100% / 50% | 100% / 100% | 100% / — | 0% / 100% | 100% / 100% | 100% / 9% |
| qwen35|animator|full | — / — | 72% / 100% | 50% / 100% | — / 0% | 100% / 50% | 50% / 67% | — / — | — / 0% | 100% / 100% | — / 0% |
| qwen35|animator|roster | — / — | 75% / 100% | 50% / 100% | — / 0% | 100% / 50% | 100% / 100% | — / — | 100% / 100% | 100% / 100% | — / 0% |
| qwen35|observer|frames | — / — | 100% / 90% | 33% / 100% | — / 0% | 100% / 50% | 100% / 33% | — / — | 0% / 100% | 100% / 100% | 100% / 9% |
| qwen35|observer|full | — / — | 75% / 90% | 33% / 100% | — / 0% | 100% / 50% | 50% / 67% | — / — | 100% / 100% | 100% / 100% | — / 0% |
| qwen35|observer|roster | — / — | 100% / 90% | 33% / 100% | — / 0% | 100% / 50% | 100% / 33% | — / — | 0% / 100% | 100% / 100% | — / 0% |

## How to read

Score weights: precision 35, recall 20, camera 0, speaking 10, reliability 10, efficiency 15 (missing components are left out and the rest renormalised).
GT = hand labels (`labels/<cut>.json`, made before any model answer): per pair and category, a reported category is a true positive when the label marks it. Precision and recall come from a blind Qwen3.5-27B judge (shares weights with one candidate). Camera is a diagnostic (weight 0) against an ORB/RANSAC pixel estimate; every setup's camera in the report is that estimate. Speaking is scored against transcript speech by a character marked on screen (boxed), and not at all for setups that cannot see mouths (DWPose).
