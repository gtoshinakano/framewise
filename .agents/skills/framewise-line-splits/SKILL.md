---
name: framewise-line-splits
description: "Split Framewise transcript lines that run across a hard cut at the audio pause nearest the cut, or trim a line whose silent tail or head crosses the cut; preview first, apply as undoable agent decisions. Use when the user asks to split or fix lines that bleed across cuts, lines crossing hard cuts, dialogue spilling into the next cut, or to clean up transcript lines at cut boundaries."
---

# Framewise: lines across hard cuts

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). Part of
the autopilot (`docs/autopilot/HARNESS.md` M3 and D37; rules and replay numbers in `docs/autopilot/M3.md`).

```
scripts/fw line-splits EP              # preview: what it would split or trim, with the texts and cut links (read-only)
scripts/fw line-splits EP --all        # also the lines it would leave, with the reason
scripts/fw agent start EP --agent claude-code --purpose "line splits"   # optional: groups the changes (prints RUN)
scripts/fw line-splits EP run --run RUN   # queue it (CPU, seconds); then: fw wait EP --jobs JOB
scripts/fw agent undo DECISION_ID --decision   # put one line back (only while nobody edited its pieces)
scripts/fw agent undo RUN                      # put back everything the run changed
```

## What it does

- **Split** a line at a pause of at least 0.3 s between two words, the one nearest the cut within 1.5 s: on the cut when
  the cut is inside the pause. Never inside a word or before a particle; both pieces keep the line's exact text.
- **Trim** a line whose part across the cut is only silence (the aligner's padding) to where the voice stops or starts.
- **Leave** a line with no usable pause (a phrase bleeding into the next cut is acceptable; the user moves it while building
  a sequence), lines without word timings (episodes transcribed before 2026-11-20), and lines inside a sequence.
- Pieces keep the source line's origin; each change is an agent decision of kind `line_splits`. It **never asks** and
  marks nothing stale (D37); only the person's undo does. The Autopilot profile's *Split lines that run across a hard
  cut* mode is Apply or Off. A line it leaves is no flag anywhere: Prepare runs it once (`fw prepare EP`).

## Before running it

Run it after cut detection and the missed-cut check (`fw missed-cuts check EP`): it plans against the current cuts.
Preview first and read the split texts: every division should fall at a clause or sentence boundary. If one does not,
say so and do not run it blindly.

## Final report: the links to open

Say how many lines were split, trimmed and left (and why), then list one `- cut N: URL` line per cut whose lines changed
(from the preview), so the user can check the pieces on the Results tab's line editor.
