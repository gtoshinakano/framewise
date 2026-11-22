---
name: framewise-autopilot
description: "Run the Framewise autopilot on an episode end to end: fw autopilot drives it to its Cut summaries, running each gate's resolver (missed cuts, line splits, voices, box names and target frames, review answers), holding back only the cuts that wait for a person, re-running stale results and fixing failed runs; between passes Claude answers the second-reviewer questions on their sheets; ends with a digest and the links to open. Use when the user asks to run the autopilot, automate or finish an episode unattended, process an episode end to end, or for the autopilot digest."
---

# Framewise: the autopilot (M8)

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`).
The plan, decisions and limits live in `docs/autopilot/HARNESS.md`; read its section 4 (tiers, hard stops) once.

## The loop

```
scripts/fw autopilot EP                       # the next move, nothing queued
RUN=$(scripts/fw agent start EP --agent claude-code --purpose autopilot)
scripts/fw autopilot EP --go --run $RUN       # drives until only questions are left (exit 2) or it is done (exit 0)
scripts/fw inbox EP --reviewer claude         # the second reviewer's questions
#   answer them with the resolver skills (below), within the profile's Claude budget
scripts/fw autopilot EP --go --run $RUN       # again: answers unblock cuts, settled questions re-open their steps
scripts/fw digest EP --run $RUN               # what was applied, what is left for the person, with links
scripts/fw agent finish $RUN
```

`fw autopilot --go` waits on the queue by itself (the GPU work can take hours). Run it in the background and wait for
it to exit; never poll in a tight loop. `--once` does one step and exits (3 while the queue is busy), for a caller that
schedules its own wake-ups (`/loop`).

What one pass does, in this order (`autopilot_plan` in `scripts/fw`):

1. **triage** failed runs that have a fix rule (M1), again after every step;
2. setup, with the **missed-cut checker** (M2) as soon as cuts and the preview exist;
3. the **line splitter** (M3) after transcription, speaker attribution when stale, the **voice resolver** (M5);
4. the **character agent** (M6): box names, unknown people, flagged target frames (again after target frames are detected);
5. target frames, visual analysis, beats, consistency, for every cut that is **not held** (a cut is held while its
   first-frame boxes or target frames wait for a person: the next step would be built on a guess);
6. the **review-answer agent** (M4), then Cut summaries for every cut whose questions are answered;
7. the **stale sweep** (D7): stale results up to the Cut summary are re-run from each cut's earliest stale stage
   (visual → beats → consistency → summary); a stale summary waits while its cut has unanswered questions. Sequences are
   never touched; the digest names the sequences whose cuts changed (their shots stay stale, D9).

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

End with the digest (`fw digest EP --run $RUN`): applied counts per kind and tier, failures left, what waits for the
person with one `- cut N: URL` line per cut, and the inbox link (`#/episode/<id>`, header → Inbox).
