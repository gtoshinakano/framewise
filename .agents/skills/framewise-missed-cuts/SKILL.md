---
name: framewise-missed-cuts
description: "Find and fix hard cuts the Framewise cut detection missed: check an episode's cuts (scene scores, camera fit), split the sure ones automatically (undoable), and review the doubtful ones by looking at a frame strip as the second reviewer (Claude), handing what stays unclear to the person. Use when the user asks to check for missed cuts, wrong or merged cuts, cuts that contain two shots, to re-split cuts, or to review the missed-cut questions."
---

# Framewise: missed hard cuts

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). Part of
the autopilot (`docs/autopilot/HARNESS.md` M2, details and replay numbers in `docs/autopilot/M2.md`).

```
scripts/fw agent start EP --agent claude-code --purpose "missed cuts"   # prints RUN (optional, groups the decisions)
scripts/fw missed-cuts check EP [RANGE] [--run RUN] [--redo]   # queue the check (CPU, a few seconds per episode)
scripts/fw wait EP --jobs JOB
scripts/fw missed-cuts report EP [RANGE]   # splits applied (with decision ids) and open questions with their pictures
scripts/fw inbox EP --reviewer claude      # the questions waiting for Claude (any kind)
scripts/fw missed-cuts strip QUESTION_ID --out /tmp/q.jpg      # the picture to look at
scripts/fw missed-cuts answer QUESTION_ID split|no-cut|unsure --tier claude --note "why" [--run RUN]
scripts/fw agent undo DECISION_ID --decision   # merge one agent split back (only while nobody changed those cuts)
```

## What the check does on its own

- **Splits** where a lone scene-score spike has no camera fit across it (replay: 146 of 146 splits were real cuts). Each
  split is an editor cut edit (the editor's Undo works) and an agent decision.
- **Asks Claude** (`reviewer=claude`) about doubtful peaks: a weaker or less isolated spike, a flash, a split that would
  leave a piece under 0.5 s.
- **Asks the person** about cuts in a sequence (sequences are the user's), a boundary that looks misplaced, and anything
  when the Autopilot profile says "Ask me" or Claude is switched off.
- Skips cuts with analysis switched off (openings and endings) and peaks already asked about.

## Reviewing as Claude (tier 2)

For each question from `fw inbox EP --reviewer claude` (kind `missed_cut`):

1. Download the strip (`fw missed-cuts strip ID --out FILE`) and read it. Top row: two frames before the red bar, two after.
   Bottom row: eight frames over a second on each side, the first one after the peak marked red. Always read the bottom
   row: dark close-ups look alike at a glance.
2. Answer:
   - `split`: the frames after the bar are a different shot (another camera position, framing or subject), even if the
     place and the people are the same (shot/reverse-shot).
   - `no-cut`: the same shot continues: a camera move, a character moving, an effect, a flash, a dissolve.
   - `unsure`: you cannot tell from the picture. The question goes to the person with your `--note`.
3. Never answer a question whose reviewer is the person (the API refuses with 409); never split a cut of a sequence.

Count every picture you looked at and pass the total to `fw agent finish RUN --claude N`.

## Final report: the links to open

List what was split (cut number, time, decision id for undo), what you answered, and what waits for the person, ending
with one `- cut N: URL` line per cut that still needs them (from `fw missed-cuts report`, or
`fw links EP 12,40 --place cut`).
