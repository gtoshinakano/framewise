---
name: explain_result
title: Explain an analysis result
description: Explain in plain language what a specific result (initial frame, last frame, video analysis, OCR, consistency summary, brief) found and how reliable it is.
examples: What was analyzed from first image? | Explain the video analysis events
---
# Explaining a result

Map the user's words to a result type:
- "first image", "opening frame", "initial frame" → `initial_frame`
- "last image", "ending frame" → the end of the cut is in `video_analysis` (`end_of_cut` and each character's `end`). Last frames are no longer analyzed; `last_frame` only exists on cuts analyzed before that and is not kept up to date
- "video", "motion", "events", "what happens" → `video_analysis`
- "subtitles", "OCR", "on-screen text" → `get_subtitles()` (episode-wide burned-in subtitles, one language; older cuts may also have a `subtitle_ocr` result)
- "consistency", "conflicts", "review" → `consistency_summary`
- "brief", "cut summary", "production brief" → `production_brief`

## Steps
1. `get_result(type=...)`. For frames, request `fields` only when the user asks about one aspect (e.g. `["visible_characters"]` for "who is in the first image"). The initial frame gives each marked character's `appearance`, `position`, `pose`, `expression`, `emotion`, `gaze` and `action`, plus `mood` and `props`. The video analysis is a motion direction: `camera`, per character `activity`, `travel`, `motion`, `speaking`, `emotion` and `end`, timed `events`, and the video-reported end state. Historical results may also have `end_of_cut.in_frame` (a positive confirmation from last-frame character boxes; absence is not evidence of leaving). An older `last_frame` with `mode: "changes"` lists only what differs from the first frame: `unchanged` means the first frame still holds, `new_in_frame` marks a newcomer, `left_frame` names who is gone.
2. If the user asks whether the result is right, compare with one other result (for example the video analysis overview or the consistency summary) instead of guessing.

## Answer format
- One-sentence overview of what the model saw.
- Sections in this order, skipping empty ones: **Composition & camera**, **Characters** (appearance, position, pose, expression; identity and its evidence: "user-verified box", "clothing match", or "unidentified"), **Setting & lighting**, **Other details**.
- For video analysis: a short timeline `0.0–0.8s · subject · action · camera`.
- **Reliability**: the model that produced it, its stated uncertainties, and anything contradicted by other results.

Explain, don't dump: do not paste raw JSON. Times in the data are episode-relative unless labelled clip-relative; convert to clip-relative seconds when describing action within the cut.
