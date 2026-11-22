---
name: framewise-diagnose
description: "Find what is wrong in a Framewise episode and why, and fix failed runs that have a rule: per cut, ranked problems worst first (failed steps, stale results, pictures contradicting the analysis, transcript lines running across a hard cut, checkpoints for a person, the next missing step); the root-cause chain of one cut's step ('consistency failed ← beats older than keyframes'); and triage of failed runs (retry, queue the missing step first, wait for the GPU, dismiss what is obsolete, ask a person for the rest). Use when the user asks which cuts are wrong, why a cut or step failed or is missing, to root-cause failures, to fix or triage failed runs, or 'why does cut N have no summary'."
---

# Framewise: diagnose and triage

Work from the repository root. Everything goes through the Framewise API (`FRAMEWISE_URL`, default
`http://chopper.local:8080`) with `scripts/fw`. Part of the autopilot (`docs/autopilot/HARNESS.md`, milestone M1).

```
scripts/fw diagnose EP [RANGE]               # per cut, what is wrong and why, worst cuts first, one link per cut
scripts/fw diagnose EP --kind failed         # only cuts with that problem: failed, stale, contradiction, person,
                                             #   missing, line_crossing, auto_labels, no_boxes
scripts/fw why EP 142                        # the earliest step of cut 142 that is not done, and why
scripts/fw why EP 142 --step summary --all   # why this step, down its prerequisites; --all: every step's state
scripts/fw triage EP                         # failed runs nobody has dealt with: cause and the fix (read-only)
scripts/fw triage EP --go --run RUN_ID       # apply the fixes (each one an undoable agent decision)
```

`EP` is an id or text such as `"Seishun Buta Yarou / Episode 7"`; RANGE is the editor's cut numbers (`100-200`,
`7`, `5-`, `all`). `diagnose`, `why` and `triage` without `--go` only read.

## The steps of a cut and their states

Eight steps in the editor's order, each with what it reads: first-frame boxes → target-frame people → first-frame
analysis → keyframe analysis → performance beats → consistency review → answers → Cut summary. States: `ok`, `off`
(analysis switched off), `queued`/`running`, `failed`, `missing`, `stale`, `person` (a checkpoint a person settles).
`why` follows the first broken prerequisite until it reaches the root, which is what to act on:

```
cut 142 · Consistency review
  Consistency review [missing]: no consistency review yet
  ← Performance beats [stale]: built from an older keyframe analysis (keyframes redone 2026-11-20)
fix: run performance beats again  [fw beats 47dae7a0 142]
open: http://chopper.local:8080/#/episode/…/cut/…/analysis
```

Commands in brackets are ready to run (the episode is given by its id prefix).

## Triage rules (what `--go` does)

| Cause (from the error) | Fix | Tries before asking a person |
|---|---|---|
| obsolete: the output exists now, or its cuts are archived | dismiss the failure | 1 |
| a step it needs had not run (beats without keyframes) | queue that step, then the failed one after it | 2 |
| it read out-of-date results (beats older than keyframes) | queue the stale step again, then this one | 2 |
| GPU memory (CUDA/HIP out of memory, model failed to start) | rerun once no GPU job runs (else `wait`) | 2 |
| text provider returned nothing (OpenRouter empty/truncated) | queue again with the current text engine | 1 (2 if the engine changed) |
| validator loop (invalid answer after 3 attempts) | rerun once | 1 |
| older workflow, stale previews, program or media error | queue again with the current workflow / rerun | 1 |
| context overflow, unknown error | ask a person | 0 |
| sequence steps | leave alone (sequences are the user's) | — |

"Rerun" is the job's own retry (a visual run resumes its checkpoints); "queue again" uses the current settings and keeps
the user's choices in the request (a Cut summary instruction). A run whose tries are used up becomes **one question per
cause and pipeline** in the inbox (`fw inbox EP`), with the validator's message; it closes itself once none of its runs
fails any more. The project's Autopilot profile decides the mode: *Fix failed runs* = Apply (default), Ask me (every fix
becomes a question with the proposal) or Off.

Every applied fix is a decision of kind `failed_runs` on the run given with `--run` (start one with
`fw agent start EP --agent <you> --purpose triage`). `fw agent undo RUN_ID` cancels the jobs a fix queued while they
are still waiting and clears dismissals; a fix whose job already started is kept and reported.

## Report it

1. `diagnose`: the counts line, then the worst cuts with their link and the problems under each, in fw's wording. Say
   which problems a command fixes (the bracketed commands) and which need a person (`person`, contradictions).
2. `why`: the chain as printed, the fix, the link.
3. `triage --go`: what was applied (with the jobs queued), what was asked of a person (`fw inbox`), what waits for the
   GPU (run `fw triage EP --go` again once `fw wait EP` returns), what was left alone.

### Final report: the links to open

End with one `- cut N: URL` line per cut that needs a person (escalated failures, `person` problems, contradictions),
using the links fw printed or `fw links EP 12,14 --place analysis`. Never queue sequence steps; never re-run stale
results unless the user asks (that is the autopilot's stale sweep, M8).
