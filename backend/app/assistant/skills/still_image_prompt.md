---
name: still_image_prompt
title: Still image prompt
description: Fix, rewrite or draft the still-image generation prompt that recreates this cut's opening frame (later used as the first frame of the animation).
examples: Fix the still image generation prompt
---
# Opening-frame still image prompt

The still recreates the **first frame of this cut**. It becomes the first frame of the animation, so it must show exactly one moment and must not contain motion or story.

## Gather (only what you need)
1. The prompt to fix: the one the user pasted in this conversation. If none was pasted, `get_result(type="image_prompt")`; if that is missing too, draft a new prompt.
2. `get_result(type="production_brief")`: the Opening Frame section is the main source.
3. `get_result(type="initial_frame")`: composition, camera, setting, lighting, mood, props and visible_characters (character, appearance, position, pose, expression, emotion, gaze, action; marked characters only) and background_people (unmarked people, one line). Put each character's emotion and gaze into the image prompt: they decide the face. Each character may also carry `pose_detail` (state, torso and head facing as a view of the camera with the image side, eyes, looking_at, check): use it, in words and never as degrees, for placement and gaze, and let it beat the free-text position, pose and gaze when they disagree. Results from before emotion, gaze, mood and props existed lack those keys; even older ones may carry other_visual_details and uncertainties.
4. `get_result(type="review_answers")`: human corrections override model output.
5. `get_characters()` for exact character names and canonical appearance notes of the visible characters.
6. If `view_frames` is in your tool list, look at `first` (and zoom with `crop` on a face, hands or a held prop) and check the draft against it before you finish: framing, who stands where and faces which way, gaze, what is held. For anything visible the picture beats the written results; mention a disagreement to the user. Human answers still win.

Precedence: human answers > production brief Opening Frame > initial frame description > character notes > neighbouring cuts (never use them for this frame's content).

## Prompt structure (generator-agnostic)
1. **Shot**: shot size, camera angle and height, lens feel, and where the frame cuts the subject (e.g. "tight close-up from waist to mid-thigh, eye level, shallow depth of field").
2. **Placement and gaze**: for each character visible in the opening frame, where they are in the frame (left, centre or right, foreground to background, relative to the others), their body state, which way the torso faces and the head points, and where the eyes look and at whom or what.
3. **Subjects**: for each character visible in the opening frame, give a short identity tag (name only if user-verified or confirmed), visible appearance (hair, outfit), pose of the hands and expression. Describe only the parts that are visible.
4. **Objects and layout**: foreground, midground and background elements, and how they relate in space.
5. **Setting**: location type and environment details.
6. **Lighting and colour**: light source, softness, time of day, palette.
7. **Style**: do not state or enforce a style (no medium, art style or rendering technique); the person generating the image chooses it. Keep a style line only when the user's own prompt already has one.
8. **Constraints**: `no text, subtitles, captions, logos or watermarks`, `no additional characters`, and anything the brief's Continuity section requires.

If the user's generator uses character reference images (for example "Image 2 = Tomoe reference"), keep that syntax and give each reference one job.

## Keywords
The lists below are the app's shared vocabulary for describing a still. Use their words where they fit the evidence, so every prompt of the project reads the same; they are suggestions, not a closed set. Leave out any part you cannot support; never write "unknown" or "not visible".

{{still_keywords}}

## Rules
- One frozen moment. Describe poses, not actions (no "walks", "turns", "then").
- Subtitles and OCR text never go into the image.
- Do not add characters who are off-screen, only mentioned in dialogue, or only visible later in the cut.
- If identity is uncertain, describe the person visually and leave the name out.
- Dialogue does not belong in a still. Show a speaking expression only if the opening frame shows it.
- Prefer concrete nouns and positions over adjectives. About 60–180 words unless the user wants otherwise.

## Answer format
1. The full corrected prompt in one fenced code block (` ```text `).
2. **Changes**: bullets of what changed and why, citing the result that supports each change.
3. Open questions, only if they would change the image.
4. Offer to save it. When the user agrees, or asked to save, call `propose_result_edit(type="image_prompt", content=<prompt>, summary=...)`.
