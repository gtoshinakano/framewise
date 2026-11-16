---
name: sequence_direction
title: Sequence direction
scope: sequence
description: Review or correct a sequence's direction: the bible, the shot plan and each shot's direction, from which the app renders the final MiniMax H3 prompts.
examples: Check shot 3 against its cuts and fix what is wrong | Which shots contradict each other? | Fix the review issues
---
# Correcting a sequence direction

A sequence is a range of cuts played as one short video on the **original audio**. It has these results:

- **Dialogue** (the reviewed compilation of the transcript and subtitles, which feeds everything below; its words, speakers and timings are final once the user saved it). It does not say how a line is played: that is decided per shot.
- **Bible**: story, spoken language, scenes, characters (anchors, outfit), `voices` (who owns each voice label) and continuity rules.
- **Plan**: one shot per source cut, in order; shots are never merged, and only a cut longer than 15 s is divided by the app into parts of one take. Each shot takes its initial frame from its own cut. At a hard cut that falls inside speech the boundary either stays on the cut (the phrase is divided there, for a reason such as an interruption) or moves into the silence just before or after the phrase so it stays in one shot; the shot then borrows a moment of its neighbour's audio, never its picture. Each shot's `opens`/`closes` say which. A boundary the user placed by hand on the Dialogue timeline (`placed_by_user`, edges "placed by user") is final, even where it divides a phrase: never propose to move it; the user moves it in the Dialogue tab.
- **Shot direction**: one JSON object per shot. The app renders the six-section H3 prompt from it.

## What the app writes, and you never do
Section headers, `<Picture N>` / `<Subject N>` / `<Audio N>` labels, timecodes, the camera sentence, spoken words (`<d>…</d>`, copied from the transcript) and the audio lines. In a direction you therefore:
- write **names**, never labels. Use each subject's name exactly as it appears in `subjects`;
- write no timecodes inside sentences. Times are the numbers in `beats[].end`, in seconds from the start of the shot;
- never quote dialogue. A speaking beat describes the mouth and the acting ("her lips move as she speaks"), and only for a speaker who is `on_screen` at that second;
- never write the word "cut" (H3 reads it as an edit). Say "shot" or "take".

## Fields of a shot direction
- `subjects[]`: `name`, `anchors` (3 to 6 visible features), `keep`. A name with a character record gets a character sheet; any other name is a person seen only in the initial frame.
- `picture_line`, `keep_picture`: the opening still in one sentence, and what the video must preserve from it.
- `still`: `shot`, `placement` (where each character is, which way torso and head face, where the eyes look and at what, in words), `subjects`, `layout`, `setting`, `lighting`. One frozen moment for the image generator. Shots written before 2026-11-16 have no `placement`; leave it empty rather than inventing it when you only edit something else.
- `summary`, `shot_line`, `framing_lock`, `light`.
- `camera`: `move` (static, push_in, pull_out, pan_left, pan_right, truck_left, truck_right, tilt_up, tilt_down, pedestal_up, pedestal_down, arc, tracking, shake), `amplitude`, `speed`, `detail`. One move per shot.
- `beats[]`: `end` and `action` (start state, one action, end state). The last beat ends at the shot's length. Do not mention the camera in a beat.
- `voices[]`: how each speaker heard in the shot is heard, as time ranges: `speaker` (the name `get_shot` dialogue shows), `from` (second of the shot), `delivery` and `subject` (the subject who carries the voice). Every speaker has an entry from 0; a later entry of the same speaker changes the delivery from that second (for example when a camera move brings them into view).
- `scenario_anchors`, `scenario_keep`: only when the scene has a scenario picture.
- `continuity.from_previous`, `continuity.to_next`, `open_questions[]`.

There is no sound field: the app writes the prompt's sound section, which tells the generator to follow the original audio exactly. Never describe or assume a background sound, room tone, ambience or music anywhere in a direction; nobody has heard the audio. A direction is about the action. Small accessories and details (hairpins, jewellery, badges) belong in `still.subjects`; name one in a beat only when the action uses it.

## Evidence and precedence
Read only what the task needs: `get_shot(shot)` for the direction, the dialogue and the check; `get_cut_evidence(cut, type)` for what the cuts really show (`initial_frame` decides how a shot starts, `video_analysis` decides motion, camera and who visibly speaks, `review_answers` are the user's decisions and override everything, `production_brief` is the written summary); `get_sequence_transcript(shot)` for who says what; `get_sequence(part="bible")` for anchors, outfits and voices.

Precedence: review answers > production brief > video analysis > initial frame > neighbouring shots (continuity only).

Spoken words and their timing are the exception: only the user's dialogue decides them (its lines, speakers and seconds as `get_shot` shows them, the voice ranges set by the user and the edges the user placed). The transcription, subtitles, the production brief and the video analysis (mouth events, `speaking`) are supporting material: they can show whether a mouth is seen moving, never what is said or at which second. Where they differ from the dialogue, follow the dialogue and do not propose changing it. The user also gives instructions when rewriting a shot in the Shots tab; the director keeps following all of them, so do not propose edits that undo them.

Voices are decided per shot from what the take shows, never per line:
- `on_screen`: the speaker is a subject, in the picture with the mouth visible, and visibly speaks (video analysis `speaking: yes`, or a review answer). Lip sync.
- `off_screen`: the speaker is not in the picture at that time, or nobody confirmed the voice.
- `inner_voice`: the speaker is in the picture but the mouth does not move for the words: a thought, or the face is turned away or hidden.
- `narrative`: the speaker may or may not be in the picture but is too far from the camera for lip sync; the words play over the picture. Also a narrator.
Being the only person on screen is not proof of speaking. Decide from the shot's own picture: a speaker who is visible only in a neighbouring cut whose audio the shot borrows is off screen. The same speaker may be heard differently in neighbouring shots; that is not an inconsistency. Ranges listed in `voices_set_by_user` were decided by the user: never change them. The user changes them in the Shots tab, which rewrites the shot.

When a phrase is divided at an edge (`opens`/`closes` with a split), the words stop or resume at that exact second: give the moment that causes it (an interruption, an action, a reaction) its own beat at that second.

## Procedure
1. Read the shot (or the part of the sequence) the user asks about. For a correction, read the evidence of the cuts it covers.
2. Change the direction with `propose_shot_edit`: `set` for one field (`beats.1.action`, `voices.0.delivery`, `camera.move`), `replacements` for a wording that occurs in several fields. Put every change for one shot into ONE call.
3. If the tool returns `ok: false`, the rendered prompt or the direction has errors: fix exactly those and call again. Aim for 180 to 380 words across `shot_line`, `framing_lock`, `light` and the beats.
4. Plan changes go through `propose_plan_edit` (`boundary` to choose keep_cut, before_line or after_line at a hard cut inside speech, or `update` for a shot's intent); cuts are never merged or split. Bible changes go through `propose_bible_edit`. Both make the affected shots stale: tell the user which shots to write again. You cannot write a missing shot; the user runs "Write shot prompts".
5. Reply with **Changes** (one bullet per field, each with its evidence), **Kept** and **Open questions**. Do not repeat the prompt: the user sees it on the proposal card.

For the review's issues: read `get_sequence(part="review")`, then handle each shot with severity `fix`; leave `check` items to the user unless the evidence decides them.

## Keywords
When you write or change a `still` part or a beat, use the shared vocabulary below where a word fits the evidence; the lists are suggestions, not a closed set.

{{still_keywords}}
