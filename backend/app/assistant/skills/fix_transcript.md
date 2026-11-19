---
name: fix_transcript
title: Fix the transcript
description: Correct this cut's audio transcript by comparing it with the burned-in subtitles and a fresh listening of the audio on a second model. Wrong text, missing speech, lines to split, merge, remove or add, wrong speaker. Creates a proposal the user applies; never edits silently.
examples: Fix the transcription, John says: ... and Maria says: ... | Correct John's text to: ... | The speaker for this excerpt is Mary. Fix it | Re-listen to the audio and fix who says what
---
# Fixing the audio transcript

## Three sources
The transcript is speech-to-text output plus diarization, so the words, the speaker label and the character link can each be wrong. The subtitles are text read off the picture and are usually the most reliable wording, but they are not speech: they can be shortened, merged or one line late. A third reading is possible: `transcribe_cut_audio()` sends this cut's audio to another model, which hears nothing but that clip, and returns the words verbatim with a label for every voice it separates. You never decide silently which source is right: you record what the sources agree on, and you ask about what they do not.

## Steps
1. `get_transcript()` first. It returns numbered `line`s with clip-relative times, the speaker label, the linked `character` (the key is absent when no character is linked: unknown or missing speaker) and the text. All edits refer to these line numbers.
2. `get_subtitles()` — if it says `extracted: false`, say so once and work from the transcript and the audio.
3. `transcribe_cut_audio()` — the whole cut by default; `start`/`end` (clip-relative seconds) for the part the user is asking about when the cut is longer than 60 s. Call it when the user reports wrong words, a wrong or missing speaker, speech that is not in the transcript, or when the transcript and the subtitles disagree. Skip it when the user simply dictates the text.
4. `get_characters()` when a character is involved, and use the **exact** project name. Map the user's spelling to the canonical name only when it is clearly the same character ("Jon" → "John"); if two characters fit or none does, ask the user. A voice that is not a character (narrator, announcer) goes in `speaker_label`, with no `character`.
5. Pick the smallest operation per line, then send **all** edits for the cut in one `propose_transcript_edit` call.

## How to reconcile
- **Text.** Subtitles and the fresh listening agree → use that wording. Only the transcript differs → use the agreed wording and say what changed. The sources disagree with each other → keep the current text and ask the user, quoting all three. Never translate, normalize or "improve"; keep the language and punctuation as they are spoken.
- **Times.** Always keep the app's times; the fresh listening only estimates them. Never move or split a line to match a model timestamp unless the user says the timing is wrong.
- **Voices.** Two or more labels over one app line, or subtitles with two speakers, means the diarization merged them → `rewrite` that range into one line per voice, keeping the original span (`rewrite` shares it by text length; give `start`/`end` on every line only when the user knows the timing).
- **Identity.** `S1`/`S2` are voices, not characters. Link a character only when the user named it, or when the line's existing diarization label already points to that character in `get_transcript`. Otherwise keep or set `speaker_label` and leave the link empty, and say so.
- **Missing speech.** In the fresh listening (and preferably the subtitles) but not in the transcript → `insert`, with the subtitle's times when they cover it, else the model's estimate rounded to 0.1 s, and tell the user those seconds are an estimate.
- **Extra speech.** Only the fresh listening hears it, nothing else supports it → do not delete the app's line; mention it. The transcript line stays, a hallucination is only removed when the user or the subtitles support it.
- Do not delete a line just because the fresh listening omitted it: quiet speech and overlap are where it fails.

## Operations
- **Wrong speaker, text fine** ("the speaker for this excerpt is Mary") → `{"op":"update","line":2,"character":"Mary"}`. It keeps the word timings. `clear_character: true` removes a wrong link. Speech is split into short lines at every pause, so an excerpt often spans several: add `"to_line": 4` to relink lines 2-4 in one edit (lines that already match are skipped).
- **Wrong words** ("correct John's text to: …") → `{"op":"update","line":2,"text":"<full corrected text>"}`. For one wrong word use `replacements: [{"find":"exact words","replace":"…"}]` instead, with `find` copied from `get_transcript`. `text` and `character` can be combined in one update.
- **The user dictates the dialogue** ("John says: A, Maria says: B") → if the number of lines matches, `update` each line with its `text` and `character`. If it does not (one line holds two speakers, or the ASR split one sentence), `rewrite` the whole range: `{"op":"rewrite","line":1,"to_line":3,"lines":[{"character":"John","text":"A"},{"character":"Maria","text":"B"}]}`. New lines without times share the old time span in proportion to text length; give `start`/`end` (clip-relative seconds) on **every** line only when the user knows the timing.
- **Bogus line** (noise, hallucinated text) → `{"op":"delete","line":4}`.
- **Missing speech** → `{"op":"insert","lines":[{"character":"John","text":"…","start":1.2,"end":2.4}]}`; times are required.

## Ambiguity
- "this excerpt / this line / what he says" with several lines in the cut: ask which line (quote the candidates with their times) unless exactly one line fits.
- `unlinked_lines_with_same_voice_label` in the result lists other lines of this cut whose voice label matches a line you just linked. Mention them and offer to link them too; do not change them unprompted.
- the sources disagree on who speaks: quote each source's wording for that span and ask, do not guess from a face or an outfit.

## After proposing
- Tell the user, briefly, which lines change (use the tool's `changes`) and that the proposal is **pending** on the card. Do not repeat the whole transcript.
- The transcript feeds later analysis and H3 prompts. Once the user has applied the fix, offer to regenerate whatever depended on the old text (summary, H3 prompt) only if they ask.
- Say which source you followed and what still disagrees, in one line per line of dialogue.
- If the tool reports an error (unknown character, text not found, bad line number), fix the arguments and call it again; do not make the user do it. If it returns warnings, read them: an overlap warning usually means you should `rewrite` that line instead of adding a new one.
