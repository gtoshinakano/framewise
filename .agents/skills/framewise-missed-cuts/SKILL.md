---
name: framewise-missed-cuts
description: "Find hard cuts the Framewise cut detection missed: check an episode's cuts (scene scores, camera fit), split the sure ones automatically (undoable), and list the doubtful ones for the person to look at in the editor; it never asks a question. Use when the user asks to check for missed cuts, wrong or merged cuts, cuts that contain two shots, or to re-split cuts."
---

# Framewise: missed hard cuts

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). Part of
the autopilot (`docs/autopilot/HARNESS.md` M2 and D37, details and replay numbers in `docs/autopilot/M2.md`). Prepare runs
the check **once** per episode (`fw prepare EP`); run it by hand only when the user asks.

```
scripts/fw missed-cuts check EP [RANGE] [--run RUN] [--redo]   # queue the check (CPU, a few seconds per episode)
scripts/fw wait EP --jobs JOB
scripts/fw missed-cuts report EP [RANGE]   # splits applied (with decision ids) and the cuts it was unsure of
scripts/fw agent undo DECISION_ID --decision   # merge one agent split back (only while nobody changed those cuts)
```

## What the check does

- **Splits** where a lone scene-score spike has no camera fit across it (replay: 146 of 146 splits were real cuts). Each
  split is an editor cut edit (the editor's Undo works) and an agent decision.
- **Never asks** (D37): doubtful peaks (a weaker or less isolated spike, a flash, a split that would leave a piece under
  0.5 s) and peaks in a cut of a sequence are only counted, and `report` lists them. No inbox item, nothing held: the
  person fixes missed cuts in the editor (Split at the playhead) when they see one.
- Skips cuts with analysis switched off (openings and endings), and peaks already split or asked about before D37.

## Final report: the links to open

List what was split (cut number, time, decision id for undo), then the cuts it was unsure of with one `- cut N: URL` line
each (from `fw missed-cuts report`, or `fw links EP 12,40 --place cut`), so the person can look at them in the editor.
