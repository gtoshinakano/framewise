---
name: h3_animation_prompt
title: H3 animation prompt
description: Fix (or draft) a MiniMax H3 / FastH3 OmniRef shot prompt in the director's six-section format, keeping every <Picture N>/<Audio N>/<Video N>/<Subject N> reference and name, and correcting its content from this cut's analysis.
examples: Fix this H3 prompt based on results
---
# Fixing a MiniMax H3 shot prompt

The user pastes a detailed prompt written by their AI director for another app. Its content may be wrong, but its **structure and references are right**. The app assigns the pictures in this order: the stills (`<Picture 1>` = the first frame), then the character sheets, then scenarios. Audio windows (often the cut's original audio) come as `<Audio N>`. The files already exist under those numbers and names.

## What never changes
- The six section headers, their order, and the language (English).
- Every `<Picture N>`, `<Audio N>`, `<Video N>` label: the same numbers, each with the same meaning. Never add, drop or renumber one.
- Names and file references as written: character/record names, outfit names ("Winter cloak"), sheet and location names, quoted audio names.
- Any lines before `subject_definitions:` (the app's picture/audio lines). **Do not retype them.** Start your prompt at `subject_definitions:`, and `check_h3_prompt` puts the original lines back exactly. It also restores “curly-quoted” names.
- The summary prefix (e.g. `[keyframe completion + reference generation]`), the mode, the style sentence (the project's style lock), and the shot length (the end of the last beat). Change one only when the user asks or it is impossible to keep.
- Each `<Subject N>` keeps its picture. You may correct its anchors, but not rebind it to another picture.

## What you fix from the analysis
Read only what you need: `get_result(type="production_brief")`, `get_result(type="video_analysis", fields=["overview","events"])`, `get_result(type="initial_frame", fields=["composition","camera","lighting","visible_characters","background_people"])`, `get_result(type="review_answers")`, `get_transcript()` when there is speech or an `<Audio N>`, and `get_characters()` for canonical names and notes. Precedence: review answers > brief > video analysis > frame descriptions > neighbouring cuts (continuity only).

- `<Picture 1>` line: who is where in the opening still (initial frame and the brief's Opening Frame). Describe only what is visible: if the still shows a waist-to-thigh close-up, nobody's face is in it.
- Subject anchors: 3–6 short visible features that match the character notes and the frame (build, hair, main garment and colour). Remove anchors that the evidence contradicts, and never invent new ones.
- `[Shot 1]` line: shot size, angle and who is where, a framing lock ("Waist to mid-thigh, never wider."), and one light sentence.
- Beats: the real action in order (video analysis events and the brief's timeline), each as a visible start state → one action → a visible end state. Keep the original length. Real cuts are often shorter than an H3 shot (about 5–15 s). Keep every event at its **source time from 00:00.000**, because the shot starts on `<Picture 1>` and must stay in sync with the original `<Audio N>`. Fill the remaining length **after** the action, with a hold or a slow continuation of the end pose. Never put a hold before the action, and never stretch one gesture across the whole shot. Remove any action, camera move or person that the evidence does not support.
- Timed lines (`At 00:00.070, …`): keep them where the original puts them (e.g. a voice-over note after the beats). Put a new timed line inside the beat whose range contains its time.
- Camera: one move, in H3 vocabulary as type + amplitude + speed: "The camera holds a static shot." / "The camera pushes in with small amplitude at slow speed toward her hands." Types: Push In / Pull Out, Pan Left / Right, Truck Left / Right, Tilt Up / Down, Pedestal Up / Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly. If the analysis says static, use a static shot.
- Speech and audio, from the transcript (times are clip-relative):
  - Quote spoken words **exactly** as `get_transcript` gives them, in their language. The checker rejects altered words.
  - If an `<Audio N>` is the original sound and the speaker is a visible subject with a supported link (a linked character or a review answer), write: "<Subject 1> speaks the words of <Audio 1> in its voice, lips in sync, from 00:00.070." and, in detailed_description, `At 00:00.070, <Subject 1> (S1) says: <d>[Japanese] ええ。</d>` with the exact source words and language.
  - If the speaker is unconfirmed or off-screen, write "An off-screen voice speaks the words of <Audio 1>; <Subject 1>'s lips stay closed." Being the only visible person, or a mouth hidden by the framing, is not evidence. A transcript segment with no `character` is unconfirmed: put the attribution under **Open questions**.
  - With no speech, write "<Subject 1> does not speak; her lips stay closed." Never keep "nobody speaks" while an `<Audio N>` carries a voice.
- retention_analysis: one line per picture and subject, kept consistent with your fixes.
- overall_soundscape: 1–4 sentences of ambience and action sounds plus a line saying what each `<Audio N>` drives (a voice, the ambience to match). Never empty, no `<d>` dialogue.
- non_diegetic_music: `N/A`.

## Format rules (checked by `check_h3_prompt`)
- Start at `subject_definitions:`. Never write the app's picture/audio lines yourself.
- Labels exactly `<Picture N>`, `<Subject N>`, `<Audio N>`, `<Video N>`, with angle brackets and a space. In descriptions use `<Subject N>`, not the bare name.
- detailed_description opens with the style sentence ("The target video is …"), then exactly one `[Shot 1]`, then timed beats `00:00.000-00:02.000: …`. Beats must be contiguous, start at 00:00.000, end at the shot length (at most 15 s), with about 2–3 beats per 10 s.
- Never write the word "cut" (not even "no cut"), never a `---` line, never lettering (write "blank" or "unmarked"). Subtitles and OCR text are never shown.
- Aim for 200–500 words in detailed_description, every sentence concrete. Keep the whole prompt under 7,000 characters.
- Modes: with no `<Picture N>` in the original (first/last frame mode), keep it picture-free. For a character-fix prompt (`<Video 1>`, summary `[video editing + …]`), keep that structure and only correct the subjects; no beats are needed.

## Procedure
1. Use the pasted prompt. If none was pasted, the saved one (`get_result(type="h3_prompt")`). If there is neither, draft from the format above with `<Picture 1>` = first frame and ask which other pictures and audio the shot sends.
2. Gather the evidence above (parallel tool calls are fine).
3. Write the corrected prompt from `subject_definitions:` to the end and call `check_h3_prompt(prompt=…, summary=…)`. It finds the original in the chat by itself, so do not pass `original` when the user pasted one. If it returns errors, fix them and call again. Address warnings when you can.
4. When it passes, the user sees the prompt in a card with Copy and Save to cut. **Do not repeat the prompt, do not write a code block, and do not call `propose_result_edit` for it.** Reply with:
   - **Changes**: bullets, one per section you changed, each with its evidence (e.g. "beats: video analysis shows a hand entering at 0.0–0.8 s, not walking").
   - **Kept**: one line confirming the labels, names and length you preserved.
   - **Open questions**: only decisions you could not make (e.g. who speaks), plus any check warnings you left.
5. If the check still fails after three attempts, show your best version in a ```text block and list the remaining errors.
