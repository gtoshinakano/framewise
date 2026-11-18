---
name: correct_result
title: Correct an analysis result
description: Correct a saved result (character name, identity, event, wording) following the user's instruction. Creates a proposal the user applies; never edits silently.
examples: Fix the character name in the keyframe analysis
---
# Correcting a saved result

## Steps
1. Identify the target result type from the user's words (keyframes or what changes → `keyframe_analysis`, beats, acting or delivery → `performance_beats`, first image → `initial_frame`, how the cut ends or what moves → `keyframe_analysis` (older cuts: `video_analysis`), brief → `production_brief`, consistency → `consistency_summary`, answers → `review_answers`, H3 → `h3_prompt`, image prompt → `image_prompt`).
2. For a name or identity fix, call `get_characters()` and use the exact canonical name. If the user's name matches no character, say so and ask before proceeding.
3. Find what to change:
   - `find_in_results(query="<wrong text>")` shows which results contain the text and where. Use it to find every occurrence cheaply.
   - Read the target result with `get_result` (use `fields` to limit it) when you need the exact wording.
4. Call `propose_result_edit` with the smallest edit that does the job:
   - `replacements: [{"find": "a person", "replace": "Tomoe Koga"}]`. `find` must be exact text from the result. A replacement applies to every occurrence unless you set `"count": 1`. Add surrounding words to `find` when a short word would also match places that must not change.
   - `set: [{"path": "windows.0.analysis.events.0.subject", "value": "Tomoe Koga"}]` replaces one JSON field of a structured result. Paths use dot notation with list indexes.
   - `content` replaces the whole text. Use it only for text results that need a full rewrite.
   - Always give a one-line `summary`.
5. If the tool reports an error (text not found, invalid path), fix the arguments and call it again. Do not ask the user to do it.

## After proposing
- Tell the user what will change and that it is **pending**: they apply or discard it on the proposal card. Never say the result is saved.
- If `find_in_results` showed the same error in other results, list them and offer to propose the same fix there. Fix only the result the user asked about unless they asked to fix "everywhere".
- Keep everything else unchanged: no rewording, no new claims, and the JSON structure stays as it is.
- When a correction is a human decision rather than evidence (e.g. "the hand belongs to Tomoe"), suggest also recording it in the review answers so future briefs use it.
