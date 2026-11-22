---
name: framewise-cast
description: "Settle a Framewise episode's characters automatically: name first-frame boxes from the gallery shortlist (with the neighbouring cuts as context), unbox duplicates and background people, group unknown people across the episode into one card each, and settle flagged target frames whose flag a rule explains; Claude as the second reviewer checks automatic labels and doubtful names on box sheets and flagged target frames on frame sheets; the rest (new characters) goes to the person. Use when the user asks to name, box, confirm or settle characters, unnamed boxes, unconfirmed automatic labels, unknown people, new characters or flagged target frames of an episode."
---

# Framewise: characters that matter (box names, unknown people, target frames)

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). Part of
the autopilot (`docs/autopilot/HARNESS.md` M6; rules and replay numbers in `docs/autopilot/M6.md`).

```
scripts/fw agent start EP --agent claude-code --purpose "cast"   # groups the changes and the Claude budget (prints RUN)
scripts/fw cast run EP [RANGE] --run RUN      # one CPU job: names, unboxes, groups, settles frames, asks; then: fw wait EP --jobs JOB
scripts/fw cast report EP                     # what it applied (undo ids), the open questions with sheets and cut links
scripts/fw cast sheet QUESTION_ID             # box sheet (box, its crop, the user's example of each candidate) or target-frame sheet
scripts/fw cast answer ID confirm --tier claude --run RUN --note "same hair and uniform as Mai's example"
scripts/fw cast answer ID "confirm=B1,B3 reject=B2" --tier claude --run RUN
scripts/fw cast answer ID ok --tier claude    # target frames: every listed frame is right as boxed
scripts/fw cast answer ID "1234=ok,1250=person" --tier claude --note "F4: two people share one box"
scripts/fw cast answer ID unsure --tier claude --note "..."   # unknown person: hand it to the person
scripts/fw agent undo DECISION_ID --decision  # put a box, a name or a frame back (only while nobody changed it)
```

## What it does (rules, no model)

- **Unnamed first-frame box:** named from its stored shortlist when the leader clears 0.72 with a 0.10 lead, or 0.58 with a
  0.08 lead when the same character is boxed in the cut before or after (the context the user uses for fragments). A
  second box over a person already named in the cut is removed. A character the profile calls *background* is unboxed.
  Look-alike pairs (profile) and close shortlists go to Claude. Replay: 80/84 names right (95%).
- **Automatic label** (recognition's `auto`): never confirmed by a rule (the leader was the user's label only 72% of the
  time at ≥ 0.80 on boxes the user relabelled). Claude gets sheets of up to 12 boxes per character, least sure first,
  while the run's budget lasts; the rest wait for a later run. It is not a gate.
- **Unknown person** (nobody in the gallery resembles the box): grouped across the episode by the stored CCIP vectors.
  One small appearance is background (unboxed). Recurring (3+ cuts) or large people *matter*: one card for the person,
  who names them once and every box of the group follows. Anything in between goes to Claude first.
- **Target frames:** marked reviewed when the headcount dropped and the keyframe states say exactly those people left
  the picture, or when the only flag is an extra crop fitting no colour while everyone from the opening frame is still
  boxed (95/95 right on the user's corrections). Every other flagged frame of a cut is one question.

Agent labels are `label_source='agent'`: never gallery examples (only the user's own choices are) and a recognition
re-run leaves them alone. Cuts with analysis off or *No characters* are skipped. Each change is an agent decision of kind
`box_names` or `target_frames`, undoable while unchanged.

## Claude as the second reviewer

For each `[Claude]` question in `fw cast report EP` (or `fw inbox EP --reviewer claude`):

1. **box_label** (`Are these N box(es) Mai?`): fetch the sheet. Each row is B1…: the frame with the box, the crop, then
   the user's own example of each shortlisted character. Confirm the boxes whose face, hair and outfit match the
   proposed character's example; reject those that clearly are someone else; leave out (→ the person) any you cannot
   tell, e.g. a back view or a fragment. Never guess between look-alikes.
2. **new_person** (borderline unknown person): fetch the sheet. Answer `"Name"` only when the person is clearly an
   existing character; `background` when it is a passer-by who does not interact; otherwise `unsure` (the person decides,
   and only the person can say *new*).
3. **target_frame**: fetch the sheet (F1 = the opening frame, then each flagged frame, colours numbered as in the tab,
   grey = crops nobody has). `ok` a frame whose boxes are right as they are (everyone who matters boxed with the right
   colour, nobody boxed twice); `person` a frame where a box is missing, merged, swapped or misplaced. Claude never
   draws or recolours boxes: that is the person's (or the `framewise-target-frames` skill's) work.

Always pass `--tier claude --run RUN` and a short `--note` saying what you saw.

## The person

`fw cast answer ID confirm` (their confirmation is their own label: a gallery example), `"Name"`, `new --title "Name"
[--main] [--episode-only]` (creates the character from the clearest box and queues its profile description), `background`
or `leave`. Labelling, removing a box or marking a frame reviewed in the app closes the matching question by itself.

## Final report: the links to open

Say how many boxes were named, unboxed and left, how many frames were settled, and what went to Claude and to the person,
then list one `- cut N: URL` line per cut of each open question (the report prints them with `characters` or
`target-frames` places; `fw links EP N,M --place characters`).
