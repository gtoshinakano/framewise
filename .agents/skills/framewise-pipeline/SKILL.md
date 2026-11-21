---
name: framewise-pipeline
description: "Run a Framewise episode through the analysis pipeline automatically, step by step in the right order, stopping wherever a person must decide (unnamed character boxes, flagged target frames, unknown voices, review questions), with scripts/fw next/drive. Use when the user asks to process, continue, advance, push through or automate an episode, or asks what the next step is."
---

# Framewise: the gated pipeline

Work from the repository root with `scripts/fw` (it talks to the Framewise API on chopper.local). The order is fixed
and lives in fw itself; do not invent another one:

1. setup (media, cut detection, transcription, subtitles, boxes and recognition)
2. speaker attribution if it is stale
3. **checkpoint: first-frame boxes without a character** → Characters panel
4. target frames
5. **checkpoint: flagged target frames or unnamed people** → Target frames tab
6. visual analysis, then performance beats
7. **checkpoint: unknown voices** → Audio → Unknown voices
8. consistency review, then suggested answers (drafts, never saved for the user)
9. **checkpoint: review questions without saved answers** → Results → Your answers
10. Cut summaries
11. done: hand over to the `framewise-review` skill for the final checklist

Failed runs are also a checkpoint: the flow stops until each one is retried or dismissed.

## Final report: the links to open

A cut has its own address, so finish every report with **one URL per cut that needs a person**, built by `fw` and
copied exactly (never typed by hand, never shortened):

```
These cuts require human review:
First-frame boxes without a character (Characters panel):
- cut 199: http://chopper.local:8080/#/episode/<episode id>/cut/<cut id>/analysis/characters
Review questions to answer (Results → Your answers):
- cut 204: http://chopper.local:8080/#/episode/<episode id>/cut/<cut id>/analysis
```

The address is `<server>/#/episode/<episode id>/cut/<cut id>[/analysis[/<place>]]`; cut ids never change, so the
links stay valid. `<place>` is `characters` (first-frame boxes), `target-frames`, `run`, `assistant`; no place opens
the Results tab (review, answers, Cut summary). Where the links come from:

- `scripts/fw review EP RANGE` prints every checkpoint and check with its link, under what the person does there.
- `scripts/fw links EP 199,204,210-212 --place target-frames` prints the links for any list of cuts.
- `scripts/fw analysis EP RANGE --links` adds a link under each "Needs …" line.
- `scripts/fw cut EP 199` gives one cut's links.

## Where this runs

The Framewise server is `chopper.local` (`http://chopper.local:8080`; `FRAMEWISE_URL` overrides it). The skill only
talks to it over HTTP, so it works the same from any machine on the LAN. When you are not on chopper (`hostname`),
there is no Docker, container or worker on this machine: never build, restart or `docker compose` anything. If the
server needs a rebuild, tell the user to do it on chopper.

## Run it

Preferred: `scripts/fw drive EP` (add `--language Japanese` if setup asks for the subtitle language). It queues the next
step, waits for the queue, and repeats until it reaches a checkpoint or the end. It needs no model calls while it waits,
so start it in the background and read its output when it exits. It can run for hours.

Step by step instead: `scripts/fw next EP` shows the next move, `scripts/fw next EP --go` queues it, and
`scripts/fw wait EP` waits for the queue. If `next` proposes the same step for the same cuts twice in a row, stop and
tell the user: something keeps that step from finishing.

Exit codes of `next`, `drive`, `wait` and `review`: `0` ok/done, `1` error, `2` waiting for a person, `3` busy.

## At a checkpoint (exit 2)

Run `scripts/fw review EP` and tell the user exactly what is waiting, by cut number **with its link**, and where in
the UI to settle it (fw prints each cut's URL under the heading of what the person does there, then the episode link).
Copy those lines into your answer: the user opens them one by one. Then stop. Continue only when the user says they are done; `fw next` will
then move on by itself. Never settle a checkpoint yourself, never work around it with the single-step commands, and
never pass `--save` to answers in this flow.

## Failed runs

`scripts/fw failures EP` lists them with their errors.
- Retry once (`scripts/fw retry JOB`) when the error says the job was interrupted, lost its lease, timed out, or the
  remote text server was unreachable.
- Otherwise do not retry: report the error to the user (validator rejections, missing model files, disk full, bad input).
- A job you already retried once that fails again: report it, do not retry it again.
- `scripts/fw dismiss JOB` only when the user agrees the failure can be ignored.

## Never

- Never edit labels, boxes, transcript text, answers, dialogue or sequences.
- Never restart containers or workers; if the queue seems stuck for a long time, tell the user.
- Never re-run stale results on your own; they are listed by `framewise-review` for the user to decide.
- Never drive two episodes at once unless the user asks: they share one GPU.
