---
name: framewise-autopilot
description: "Run the Framewise autopilot on an episode in two phases: fw autopilot runs each gate's resolver (missed cuts, line splits, voices, box names and target frames) and every expensive step (visual analysis, beats, consistency review) only for cuts whose inputs are settled, holds the rest back for the person with the reason per cut, re-runs stale results and fixes failed runs; the same command run again after the person's fixes is phase 2. Review answers and Cut summaries run only when the profile turns them on. While it runs, Claude answers the second-reviewer questions on their sheets; it ends with the held cuts, a digest and the links to open. Use when the user asks to run the autopilot, automate or finish an episode unattended, process an episode end to end, run phase 2, or for the autopilot digest."
---

# Framewise: the autopilot (M8)

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`).
The plan, decisions and limits live in `docs/autopilot/HARNESS.md`; read its section 4 (tiers, hard stops) once.

## The loop

```
scripts/fw autopilot EP                       # the next move, nothing queued
scripts/fw autopilot EP --go                  # in the background: drives until only the person's work is left (exit 2) or done (0)
#   meanwhile: when it prints "for Claude: N open item(s)", answer them (below) so their cuts are not held back
scripts/fw inbox EP --reviewer claude         # the second reviewer's questions
scripts/fw autopilot EP --go                  # again once your answers are in: settled questions unblock their cuts
```

`fw autopilot --go` creates its agent run (purpose `autopilot`), waits on the queue by itself (the GPU work can take
hours) and ends with the HELD FOR YOU list and the digest. Run it in the background and react to its output; never poll
in a tight loop. While the GPU works it prints `for Claude: N open item(s)` whenever the second-reviewer count changes:
answer those right away, then start `fw autopilot EP --go` again after the current one exits. `--once` does one step and
exits (3 while the queue is busy), for a caller that schedules its own wake-ups (`/loop`).

## Two phases (D28)

- **Phase 1** (the first autopilot run of the episode): everything that is ready runs; a cut whose inputs are not
  settled is held back with its reasons (HELD FOR YOU, one link per cut, also in `fw digest`).
- **The person** fixes what is held: names boxes, reviews flagged target frames, names or confirms speakers, splits or
  merges cuts and lines. Tell them exactly that list and that phase 2 is the same command.
- **Phase 2** (`fw autopilot EP --go` again): re-runs what their fixes made stale and runs the missing steps, through
  the same gates. Repeat until nothing is held.

**Readiness gates** (`backend/app/cut_readiness.py`, D26):

| step | runs for a cut only when |
|---|---|
| target frames | its first-frame boxes are settled |
| visual analysis (first frame, keyframes) | first-frame boxes and target frames are settled |
| performance beats | also: first frame and keyframes current, speech settled |
| consistency review | also: beats built from the current keyframes |

A box is settled when the person or an agent labelled it, or when its gallery match is strong (top candidate ≥ 0.95,
0.06 ahead of the next). Speech is settled when every line overlapping the cut has a character, none is low confidence
unless the person or an agent set it, and transcript and speaker attribution are not stale. Visual jobs go in chunks of
16 cuts and queue beats for ready cuts as each chunk saves (`beats=when_ready`).

What one pass does, in this order (`autopilot_plan` in `scripts/fw`):

1. **triage** failed runs that have a fix rule (M1), again after every step;
2. setup, with the **missed-cut checker** (M2) as soon as cuts and the preview exist;
3. the **line splitter** (M3) after transcription, speaker attribution when stale, the **voice resolver** (M5);
4. the **character agent** (M6): box names, unknown people, flagged target frames (again after target frames are detected);
5. first-frame boxes for cuts without a detection of their current boundaries (the pieces of a split cut);
6. target frames, visual analysis, beats, consistency, each for the cuts the readiness gates allow, missing and stale
   results alike, each cut from its earliest stale stage (D7);
7. **only when the profile turns them on** (off by default, D27): the **review-answer agent** (M4) ("Your answers"),
   then Cut summaries for cuts whose questions are answered. Sequences are never touched; the digest names the
   sequences whose cuts changed (their shots stay stale, D9).

A step that queues nothing, or keeps the same cuts after two tries, is not asked again in that session (left for a
person). The profile (episode menu → Autopilot profile) switches each kind off or to Ask me, and sets the Claude budget.

## Claude as the second reviewer (between passes)

Answer only the items `fw inbox EP --reviewer claude` lists, with `--tier claude --run $RUN` and a `--note` saying what
you saw. Use each kind's skill: `framewise-missed-cuts` (strip), `framewise-voices` (frame sheet),
`framewise-review-answers` (picture sheet), `framewise-cast` (box sheet, target-frame sheet). When unsure, answer
`unsure` (or leave the box/frame out): the person gets it with your note. Never answer the person's items, never
create a character, never edit a sequence.

## Headless

`scripts/autopilot-headless.sh EP` runs this loop in `claude -p` with this skill (`deploy/framewise-autopilot.service`
and `.timer` are templates for a nightly run; install them only when the user asks).

## Final report: the links to open

End with the HELD FOR YOU list and the digest (`fw digest EP --run RUN`): applied counts per kind and tier, failures
left, the held cuts with their reasons, what waits for the person with one `- cut N: URL` line per cut, the inbox link
(`#/episode/<id>`, header → Inbox), and the sentence "fix these, then run phase 2: `fw autopilot EP --go`".
