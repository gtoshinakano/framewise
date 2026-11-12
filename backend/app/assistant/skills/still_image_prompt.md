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
3. `get_result(type="initial_frame")`: composition, camera, setting, lighting, mood, props and visible_characters (character, appearance, position, pose, expression, emotion, gaze, action; marked characters only) and background_people (unmarked people, one line). Put each character's emotion and gaze into the image prompt: they decide the face. Results from before emotion, gaze, mood and props existed lack those keys; even older ones may carry other_visual_details and uncertainties.
4. `get_result(type="review_answers")`: human corrections override model output.
5. `get_characters()` for exact character names and canonical appearance notes of the visible characters.

Precedence: human answers > production brief Opening Frame > initial frame description > character notes > neighbouring cuts (never use them for this frame's content).

## Prompt structure (generator-agnostic)
1. **Shot**: shot size, camera angle and height, lens feel, and where the frame cuts the subject (e.g. "tight close-up from waist to mid-thigh, eye level, shallow depth of field").
2. **Subjects**: for each character visible in the opening frame, give a short identity tag (name only if user-verified or confirmed), visible appearance (hair, outfit), position in frame, body and head orientation, pose, and expression. Describe only the parts that are visible.
3. **Objects and layout**: foreground, midground and background elements, and how they relate in space.
4. **Setting**: location type and environment details.
5. **Lighting and colour**: light source, softness, time of day, palette.
6. **Style**: keep the style the user's prompt already uses. Otherwise follow the project's context (for example anime cel shading) and do not invent a style.
7. **Constraints**: `no text, subtitles, captions, logos or watermarks`, `no additional characters`, and anything the brief's Continuity section requires.

If the user's generator uses character reference images (for example "Image 2 = Tomoe reference"), keep that syntax and give each reference one job.

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
