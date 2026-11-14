---
name: sequence_direction
title: Sequence direction
scope: sequence
description: Review or correct a sequence's direction: the bible, the shot plan and each shot's direction, from which the app renders the final MiniMax H3 prompts.
examples: Check shot 3 against its cuts and fix what is wrong | Which shots contradict each other? | Fix the review issues
---
# Correcting a sequence direction

A sequence is a range of cuts played as one short video on the **original audio**. It has three editable results:

- **Bible**: story, the style sentence, spoken language, scenes, characters (anchors, outfit), `voices` (who owns each voice label) and continuity rules.
- **Plan**: consecutive shots. Each covers an audio interval of at most 15 s and one or more source cuts, and takes its initial frame from one covered cut.
- **Shot direction**: one JSON object per shot. The app renders the six-section H3 prompt from it.

## What the app writes, and you never do
Section headers, `<Picture N>` / `<Subject N>` / `<Audio N>` labels, timecodes, the camera sentence, spoken words (`<d>…</d>`, copied from the transcript) and the audio lines. In a direction you therefore:
- write **names**, never labels. Use each subject's name exactly as it appears in `subjects`;
- write no timecodes inside sentences. Times are the numbers in `beats[].end`, in seconds from the start of the shot;
- never quote dialogue. A speaking beat describes the mouth and the acting ("her lips move as she speaks");
- never write the word "cut" (H3 reads it as an edit). Say "shot" or "take".

## Fields of a shot direction
- `subjects[]`: `name`, `anchors` (3 to 6 visible features), `keep`. A name with a character record gets a character sheet; any other name is a person seen only in the initial frame.
- `picture_line`, `keep_picture`: the opening still in one sentence, and what the video must preserve from it.
- `still`: `shot`, `subjects`, `layout`, `setting`, `lighting`. One frozen moment for the image generator.
- `summary`, `shot_line`, `framing_lock`, `light`.
- `camera`: `move` (static, push_in, pull_out, pan_left, pan_right, truck_left, truck_right, tilt_up, tilt_down, pedestal_up, pedestal_down, arc, tracking, shake), `amplitude`, `speed`, `detail`. One move per shot.
- `beats[]`: `end` and `action` (start state, one action, end state). The last beat ends at the shot's length. Do not mention the camera in a beat.
- `speech[]`: one entry per spoken line of the shot: `line` (L id), `delivery` (lip_sync, off_screen, inner_voice, narration) and `subject` (a subject name for lip_sync and inner_voice, otherwise null).
- `scenario_anchors`, `scenario_keep`: only when the scene has a scenario picture.
- `soundscape`, `continuity.from_previous`, `continuity.to_next`, `open_questions[]`.

## Evidence and precedence
Read only what the task needs: `get_shot(shot)` for the direction, the dialogue and the check; `get_cut_evidence(cut, type)` for what the cuts really show (`initial_frame` decides how a shot starts, `video_analysis` decides motion, camera and who visibly speaks, `review_answers` are the user's decisions and override everything, `production_brief` is the written summary); `get_sequence_transcript(shot)` for who says what; `get_sequence(part="bible")` for anchors, outfits and voices.

Precedence: review answers > production brief > video analysis > initial frame > neighbouring shots (continuity only).

Speech: `lip_sync` needs a speaker who is a subject of the shot and visibly speaks (video analysis `speaking: yes`, or a review answer). A speaker who is not in the picture, or a voice nobody confirmed, is `off_screen`. A speaker on screen with closed lips is `inner_voice`. Being the only person on screen is not proof.

## Procedure
1. Read the shot (or the part of the sequence) the user asks about. For a correction, read the evidence of the cuts it covers.
2. Change the direction with `propose_shot_edit`: `set` for one field (`beats.1.action`, `speech.0.delivery`, `camera.move`), `replacements` for a wording that occurs in several fields. Put every change for one shot into ONE call.
3. If the tool returns `ok: false`, the rendered prompt or the direction has errors: fix exactly those and call again. Aim for 180 to 380 words across `shot_line`, `framing_lock`, `light` and the beats.
4. Plan changes go through `propose_plan_edit` (merge, split at a candidate time, or update intent, staging or frame_cut). Bible changes go through `propose_bible_edit`. Both make the affected shots stale: tell the user which shots to write again. You cannot write a missing shot; the user runs "Write shot prompts".
5. Reply with **Changes** (one bullet per field, each with its evidence), **Kept** and **Open questions**. Do not repeat the prompt: the user sees it on the proposal card.

For the review's issues: read `get_sequence(part="review")`, then handle each shot with severity `fix`; leave `check` items to the user unless the evidence decides them.
