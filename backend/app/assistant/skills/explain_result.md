---
name: explain_result
title: Explain an analysis result
description: Explain in plain language what a specific result (initial frame, keyframes, performance beats, older video analysis or last frame, OCR, consistency summary, brief) found and how reliable it is.
examples: What was analyzed from first image? | Explain what changes between the keyframes | Explain the beats
---
# Explaining a result

Map the user's words to a result type:
- "first image", "opening frame", "initial frame" → `initial_frame`
- "last image", "ending frame" → the end of the cut is the last keyframe of `keyframe_analysis` (cuts analysed before 2026-11-18: `video_analysis`, its `end_of_cut` and each character's `end`). Last frames are no longer analyzed; `last_frame` only exists on older cuts and is not kept up to date
- "keyframes", "motion", "what changes", "what happens" → `keyframe_analysis` (older cuts: `video_analysis`)
- "beats", "acting", "performance", "delivery", "lip sync" → `performance_beats`
- "subtitles", "OCR", "on-screen text" → `get_subtitles()` (episode-wide burned-in subtitles, one language; older cuts may also have a `subtitle_ocr` result)
- "consistency", "conflicts", "review" → `consistency_summary`
- "brief", "cut summary", "production brief" → `production_brief`

## Steps
1. `get_result(type=...)`. For frames, request `fields` only when the user asks about one aspect (e.g. `["visible_characters"]` for "who is in the first image"). The initial frame gives each marked character's `appearance`, `position`, `pose`, `expression`, `emotion`, `gaze` and `action`, plus `mood` and `props`. The keyframe analysis lists each keyframe (`frame`, `t` seconds) with each character's state, then `changes` per pair of keyframes: the measured camera, per character what visibly changed (walking, speaking, blinking, then each part), marked `keypoints` when measured from body keypoints and `vision model` when read from the images; a change happened somewhere between its two keyframes. The performance beats are the timed acting plan built from it: per beat the time, camera, intent and each character's body, face, gaze and mouth, plus each transcript line's delivery (on_screen lip-synced, off_screen, inner_voice, narrative). Older cuts have a video analysis instead, a motion direction: `camera`, per character `activity`, `travel`, `motion`, `speaking`, `emotion` and `end`, timed `events`, and the video-reported end state. Historical results may also have `end_of_cut.in_frame` (a positive confirmation from last-frame character boxes; absence is not evidence of leaving). An older `last_frame` with `mode: "changes"` lists only what differs from the first frame: `unchanged` means the first frame still holds, `new_in_frame` marks a newcomer, `left_frame` names who is gone.
2. If the user asks whether the result is right, compare with one other result (for example the keyframe analysis against the beats, or the consistency summary) instead of guessing.
3. If `view_frames` is in your tool list and the claim is something visible (position, pose, gaze, expression, props, camera), look at the frame the result describes before you judge it: `first` for the initial frame, the keyframes (`F1`…) for what changed, the second where a beat starts for the beats. Say whether the picture agrees. Load `look_at_frames` when you need a zoom or several frames.

## Answer format
- One-sentence overview of what the model saw.
- Sections in this order, skipping empty ones: **Composition & camera**, **Characters** (appearance, position, pose, expression; identity and its evidence: "user-verified box", "clothing match", or "unidentified"), **Setting & lighting**, **Other details**.
- For video analysis: a short timeline `0.0–0.8s · subject · action · camera`.
- **Reliability**: the model that produced it, its stated uncertainties, and anything contradicted by other results.

Explain, don't dump: do not paste raw JSON. Times in the data are episode-relative unless labelled clip-relative; convert to clip-relative seconds when describing action within the cut.
