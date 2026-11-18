# Archived data of the multi-frame experiments

`frontend/public/multiframe-experiment/` is gitignored and can be cleaned. This folder keeps what cannot be regenerated cheaply:
the hand labels, every GPU answer (Qwen3.5, LFM3, the judge), the keypoints and the run configuration, plus each run's `report.md`.
Not kept because it is regenerable: frames (images; `experiment.py prepare` rebuilds them from the database while the cached keyframes
and cuts are unchanged), `results.json` and `report.html` (`report.py`), and the DWPose and hybrid answers (`experiment.py rules`).
About 9 MB, no images.

| Run | Cuts (Episode 2) | Contents |
|---|---|---|
| `runs/mf-20261117-ep2` | 139-148, stopped | LFM3 answers (80 calls, old answer format) |
| `runs/mf-20261117-4cuts` | 139, 145, 147, 151 | Qwen3.5 and LFM3 answers (12 setups), judge |
| `runs/mf-20261117-pose` | 139, 143, 144, 158 | Qwen answers, judge, keypoints, **first hand labels** |
| `runs/mf-20261117-ep2pose` | 139-158 | Qwen answers (observer/inspector x frames/roster), judge, keypoints, labels of 4 cuts |
| `runs/mf-20261118-c160-170` | 160, 162-170 | Qwen answers, keypoints, **labels of 10 cuts made before any answer** |
| `runs/mf-20261118-final` | all 30 cuts | observer/roster answers of the two runs above, keypoints, camera masked, **all 14 labelled cuts (60 pairs)** |
| `runs/mf-20261118-prompt` | the 14 labelled cuts | observer/roster plus the `walker` and `walker`+`roster_cam` answers |

A run folder holds `config.json`, `samples.json` (cuts, keyframes, windows, roster, transcript, stored baseline, pixel camera per pair),
`pose.json` (DWPose keypoints per keyframe and person, tracks), `labels/<cut>.json`, `raw/*.jsonl` (one JSON per model call with the answer,
the prompt, attempts and timings) and `report.md`.

Label file format (`labels/<cut>.json`): `{"sample", "cut_number", "frames": [...], "pairs": [{"from", "to", "start", "end", "camera", "flags": {walking, speaking, blink,
head, eyes_gaze, arms_hands, legs, body_pose, position, expression}, "note"}]}`. A flag is true when that category visibly changes between the two keyframes for any
character. Walking includes running and entering or leaving the picture on foot; position counts a move in the picture but not one caused by the camera.

Restore a run and replay it on the CPU: see `../PIPELINE_HANDOFF.md`, section 8 (`restore.sh`, `experiment.py rules`, `report.py`, `tools/error_lists.py`).
