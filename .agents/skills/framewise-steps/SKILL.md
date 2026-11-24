---
name: framewise-steps
description: "Queue one Framewise analysis step for a range of cuts with scripts/fw: detect character boxes, recognize characters, target frames, keyframes, visual analysis, performance beats, consistency review, review answers, Cut summaries, transcription, subtitles, speaker attribution. Use when the user asks to detect, trigger, run, redo, rewrite or queue one of these for some cuts or a whole episode (e.g. 'detect the boxes for cuts 100 to 120', 'trigger video analysis for cuts 5 to 10', 'rewrite the cut summary for the entire episode')."
---

# Framewise: one step over a cut range

Work from the repository root. Everything goes through `scripts/fw`, which talks to the Framewise API on chopper.local
(see below). Do not open the browser, click the UI or call the API with curl for these.

Drawing or naming a box on the Target frames of specific cuts is not a queue step — use the
`framewise-target-frames` skill, which clicks the app itself.

## Where this runs

The Framewise server is `chopper.local` (`http://chopper.local:8080`; `FRAMEWISE_URL` overrides it). The skill only
talks to it over HTTP, so it works the same from any machine on the LAN. When you are not on chopper (`hostname`),
there is no Docker, container or worker on this machine: never build, restart or `docker compose` anything. If the
server needs a rebuild, tell the user to do it on chopper.

## 1. Name the episode

`scripts/fw episodes` lists `id  Project / Episode`. Pass the id, or text such as `"Seishun Buta Yarou / Episode 6"`.
If fw answers that a name matches several episodes, use the id it prints. `--latest` means the newest episode.

## 2. Map the request to a command

Cut numbers are the ones the editor shows (they start at 1). RANGE: `100-120`, `7`, `5-` (to the end), `-10`, or `all` (the default).

| The user says | Command |
|---|---|
| detect the boxes for cuts 100 to 120 | `scripts/fw boxes EP 100-120` |
| recognize / name the characters in cuts 1-50 | `scripts/fw recognize EP 1-50` |
| redo the recognition (keep my labels) | `scripts/fw recognize EP RANGE --redo` |
| boxes and recognition together | `scripts/fw characters EP RANGE` |
| find the people on the target frames | `scripts/fw target-frames EP RANGE` (`--redo` detects again) |
| pick keyframes | `scripts/fw keyframes EP RANGE` |
| trigger video / visual analysis for cuts 5 to 10 | `scripts/fw visual EP 5-10` (`--redo` analyses done cuts again) |
| write the performance beats again | `scripts/fw beats EP RANGE` |
| run the consistency review | `scripts/fw consistency EP RANGE` (`--redo` for cuts that have one) |
| draft answers to the review questions | `scripts/fw answers EP RANGE` (suggestions; `--save` only if the user says to save them as answers) |
| write the Cut summaries | `scripts/fw summary EP RANGE` |
| rewrite the cut summary for the entire episode | `scripts/fw summary EP all --redo` |
| review + answers + summary unattended | `scripts/fw text-chain EP RANGE` (only when asked for exactly this) |
| transcribe cuts 10-12 again | `scripts/fw transcribe EP 10-12` (lines edited by hand always stay; nothing turns stale) |
| read the subtitles | `scripts/fw subtitles EP --language Japanese` (whole episode; optional, nothing waits for them or turns stale) |
| attribute speakers again | `scripts/fw speakers EP` (whole episode) |
| set the episode up | `scripts/fw setup EP` (whole episode; skips finished steps; `--language L` also reads the subtitles). Prefer `fw prepare EP` |

Without `--redo`, cuts that already have the result are skipped. Use `--redo` only when the user says redo, rewrite,
again, or all of them. When the request is unclear (which episode, which cuts, redo or not), run the command with
`--dry-run` first, show the user the request it would send, and ask.

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

## 3. Report and follow up

fw prints what it queued, what it skipped and why, and the job ids. Tell the user, including any skipped or "not ready" cuts.
If it prints `warning: open checkpoints`, pass that on word for word: the work was queued, but the person still has to
settle those items in the UI (unnamed boxes, flagged target frames, unknown voices, unanswered review questions).

To wait for the result: `scripts/fw wait EP --jobs ID,ID` (exit 0 done, 1 a job failed, with the error).
For progress at any time: `scripts/fw queue EP` or `scripts/fw status EP`.

**The report ends with links.** When the user asked you to run the step and wait for it, run
`scripts/fw review EP RANGE` once the jobs have finished and put its "THESE NEED A PERSON" part (with its URLs) at the
end of your answer. When you only queued the work, say so, and give the links fw printed under `warning: open
checkpoints`: those cuts still need the person while the work runs. Cuts the step skipped or "not ready" are listed by
number; add `scripts/fw links EP <numbers>` when the user will want to open them.

## Errors

- `this server does not understand cut ranges yet`: chopper runs older code. Tell the user to rebuild and restart
  `video-analyzer` on chopper; never fall back to queuing the whole episode instead.
- `HTTP 409 Cuts are still being detected`: wait (`scripts/fw wait EP`) and try again.
- `cannot reach http://chopper.local:8080`: the server is down or the name does not resolve here. Tell the user; do
  not swap in an IP address you guessed.
- Any other error: show it to the user. Do not retry in a loop.

## Never

- Never edit labels, boxes, transcript text, answers, dialogue or sequences: those belong to the person, in the UI.
- Never restart containers or workers. Never use Playwright or the browser to trigger work.
