---
name: cut_status_summary
title: Summarize the cut analysis
description: Report which analysis stages exist, failed or are missing for this cut, then give an interpreted summary of what happens in it.
examples: Summarize the cut analysis so far
---
# Summarizing a cut's analysis

Goal: one answer that tells the user where the analysis stands and what the cut shows.

## Steps
1. Use the cut card in the system prompt for stage availability. Call `get_cut_status` only if you need job errors or revision counts that the card does not show.
2. Read the most consolidated evidence first, and only the fields you need:
   - `get_result(type="consistency_summary")` (summary, clear, confusing, human_review)
   - `get_result(type="production_brief")` if present
   - `get_result(type="video_analysis")` for timed events when no brief exists
   - `get_transcript()` only if dialogue matters and the card shows segments
3. Do not read every result. Two or three tool calls are usually enough.

## Answer format
### Status
A compact table: Stage | Status | Source. Stages: Character boxes, Initial frame, Last frame, Video analysis, Subtitle OCR, Consistency summary, Production brief, Review answers, H3 prompt, Image prompt. Status is one of ✅ done, ⏳ running/queued, ❌ failed (give the short error), — missing. Source is `model name` or `manual`/`assistant edit`.

### What happens in this cut
3–6 sentences: setting, who is visible in the opening frame (identity only when supported or user-verified), the main action beats with clip-relative times, camera behaviour, dialogue if any.

### Open questions
Bullet the unresolved issues (from the consistency summary's confusing/human_review, conflicts between results). Skip this section if none.

### Suggested next step
One or two bullets: the next missing stage to run, or a correction worth making.

Rules: say which result supports each claim when results disagree. Never present model output as verified fact. Keep it under ~250 words unless asked for more.
