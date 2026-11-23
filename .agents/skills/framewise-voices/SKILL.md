---
name: framewise-voices
description: "Give a Framewise episode's unknown voices (transcript lines without a character, diarization labels nobody linked) to characters automatically: the agent applies a speaker where two independent signals agree (voice score against the voice samples, who the picture shows speaking, subtitle name tags), undoable; Claude as the second reviewer looks at a frame sheet for the doubtful voices; the rest (possible new characters) goes to the person. Use when the user asks to assign, settle, resolve or fix unknown voices, unknown speakers, lines without a character, or who speaks a voice label in an episode."
---

# Framewise: unknown voices given to characters

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). Part of
the autopilot (`docs/autopilot/HARNESS.md` M5; rules and replay numbers in `docs/autopilot/M5.md`).

```
scripts/fw agent start EP --agent claude-code --purpose "voices"   # groups the changes and the Claude budget (prints RUN)
scripts/fw voices run EP --run RUN          # one audio-worker job: scores the voices, applies, asks; then: fw wait EP --jobs JOB
scripts/fw voices report EP                 # open voices, what it gave to whom (undo ids), the voice questions with cut links
scripts/fw voices sheet QUESTION_ID         # three frames over each sample phrase, one JPEG (voice-ID.jpg)
scripts/fw voices answer QUESTION_ID "Mai" --tier claude --run RUN --note "her mouth moves in P1-P3"
scripts/fw voices answer QUESTION_ID unsure --tier claude --note "two people face away"
scripts/fw agent undo DECISION_ID --decision   # put the lines back without a character (only while nobody edited them)
```

## What it does

Open lines are grouped into phrases (consecutive lines of one diarization label). Each phrase and each label get signals:
the **voice** score against the characters' voice samples, the **label** links (the label's other lines that already have a
character), the **picture** (the keyframe or video analysis shows exactly one character speaking) and subtitle **name tags**.
A character the line addresses by name is excluded. Voice and label are one family (sound); picture and tags are others.

- **Applied** when two families name the same character and nothing names another: first a whole label, then single
  phrases, then a label whose settled phrases all agree (consensus). Lines keep their origin and get
  `speaker_source='agent'`; speaker attribution never re-decides them. Each unit is an agent decision of kind `voices`.
- **Claude** gets a voice where the signals disagree or only one kind of signal speaks, when its cuts show people.
- **The person** gets a voice no sample resembles (it may be a new character), and what Claude is unsure of.
- Lines inside a sequence are left alone. Afterwards speaker attribution is queued again when open line records remain.

## Claude as the second reviewer

For each `[Claude]` voice in the report (`fw inbox EP --reviewer claude` lists them too):

1. Read the question: the candidates, each sample phrase's signals and voice scores, which name it addresses.
2. Fetch the sheet (`fw voices sheet ID`): one row per sample phrase P1…, three frames over its seconds. Look for whose
   mouth moves in the frames of each phrase. You cannot hear the audio: the voice score is the only sound evidence.
3. Answer with a character name only when the frames show that character speaking across the phrases and nothing
   contradicts it (`--tier claude --run RUN --note` what the frames show). Otherwise `unsure` with a note: the person
   answers it. Claude never answers `new`, `leave` or an Unregistered card (they go to the person: which number a minor
   person gets is the person's choice) and never answers a voice that waits for the person. When the frames show a minor
   person without a record speaking, answer `unsure` with a note saying so.

## The person

`fw voices answer ID "Name"` gives the voice to a character (written as a manual edit). `"Unregistered N"` gives it to that
Unregistered card (a minor person who speaks but has no record; the question lists the cards boxed in its cuts) and
`unregistered` to a new card with the next number. `new` closes it when the voice is a character that does not exist yet
(create it, add a voice sample, run again). `leave` keeps every line as it is (no speaker, or the unsure guess) and marks
it `left_unknown`: nothing flags those lines again (voices gate, cut needs, held reasons, Lines to check) and speaker
attribution never asks about them. Settling the voice in the app (Unknown voices block, a line's character select or its
"Don't ask again") closes the question by itself.

A voice is asked **once** (HARNESS.md D29): open lines are lines with no speaker and low-confidence guesses nobody
decided, both in the same question per voice. When speaker attribution said "other" for a phrase and its cut has exactly
one Unregistered card boxed, the resolver gives the phrase to that card itself (tier `rules`, undoable).

## Final report: the links to open

Say how many lines were given to whom (label, phrase, consensus), how many voices went to Claude and to the person, then
list one `- cut N: URL` line per cut of each open voice (the report prints them; `fw links EP N,M --place analysis`) so the
user can play the lines on the Results tab.
