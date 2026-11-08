---
name: fix_transcript
title: Fix the transcript
description: Correct this cut's audio transcript from the user's words - wrong text, missing or wrong speaker character, lines to split, merge, remove or add. Creates a proposal the user applies; never edits silently.
examples: Fix the transcription, John says: ... and Maria says: ... | Correct John's text to: ... | The speaker for this excerpt is Mary. Fix it
---
# Fixing the audio transcript

The transcript is speech-to-text output plus diarization, so the words, the speaker label and the character link can each be wrong. The user tells you what was really said and by whom. You record that as a **proposal** with `propose_transcript_edit`; they apply it on the card. Never say it is saved.

## Steps
1. `get_transcript()` first. It returns numbered `line`s with clip-relative times, the speaker label, the linked `character` (the key is absent when no character is linked: unknown or missing speaker) and the text. All edits refer to these line numbers.
2. If a character is involved, call `get_characters()` and use the **exact** project name. Map the user's spelling to the canonical name only when it is clearly the same character ("Jon" → "John"); if two characters fit or none does, ask the user. A voice that is not a character (narrator, announcer) goes in `speaker_label`, with no `character`.
3. Pick the smallest operation per line, then send **all** edits for the cut in one `propose_transcript_edit` call:
   - **Wrong speaker, text fine** ("the speaker for this excerpt is Mary") → `{"op":"update","line":2,"character":"Mary"}`. It keeps the word timings. `clear_character: true` removes a wrong link. Speech is split into short lines at every pause, so an excerpt often spans several: add `"to_line": 4` to relink lines 2-4 in one edit (lines that already match are skipped).
   - **Wrong words** ("correct John's text to: …") → `{"op":"update","line":2,"text":"<full corrected text>"}`. For one wrong word use `replacements: [{"find":"exact words","replace":"…"}]` instead, with `find` copied from `get_transcript`. `text` and `character` can be combined in one update.
   - **The user dictates the dialogue** ("John says: A, Maria says: B") → if the number of lines matches, `update` each line with its `text` and `character`. If it does not (one line holds two speakers, or the ASR split one sentence), `rewrite` the whole range: `{"op":"rewrite","line":1,"to_line":3,"lines":[{"character":"John","text":"A"},{"character":"Maria","text":"B"}]}`. New lines without times share the old time span in proportion to text length; give `start`/`end` (clip-relative seconds) on **every** line only when the user knows the timing.
   - **Bogus line** (noise, hallucinated text) → `{"op":"delete","line":4}`.
   - **Missing speech** → `{"op":"insert","lines":[{"character":"John","text":"…","start":1.2,"end":2.4}]}`; times are required.
4. Use the user's wording for the dialogue. Do not translate, normalize, or "improve" it; keep the original language and punctuation. Fix only what they asked.
5. If the tool reports an error (unknown character, text not found, bad line number), fix the arguments and call it again; do not make the user do it. If it returns warnings, read them: an overlap warning usually means you should `rewrite` that line instead of adding a new one.

## Ambiguity
- "this excerpt / this line / what he says" with several lines in the cut: ask which line (quote the candidates with their times) unless exactly one line fits.
- `unlinked_lines_with_same_voice_label` in the result lists other lines of this cut whose voice label matches a line you just linked. Mention them and offer to link them too; do not change them unprompted.

## After proposing
- Tell the user, briefly, which lines change (use the tool's `changes`) and that the proposal is **pending** on the card. Do not repeat the whole transcript.
- The transcript feeds later analysis and H3 prompts. Once the user has applied the fix, offer to regenerate whatever depended on the old text (summary, H3 prompt) only if they ask.
