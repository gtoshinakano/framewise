---
name: framewise-target-frames
description: "Put a character's box on the Target frames tab of a Framewise cut by driving the real web UI with Playwright: draw or carry a box for a person the opening frame shows, accept the tracker's suggested crop, name a colour. Use when the user asks to draw, add, put or fix a box on the target frames of a cut, name the person on a target frame, or says 'the target frames are missing X' / 'put X on frame N'."
---

# Framewise: boxes on the Target frames

Work from the repository root. This is the only skill that **clicks the app**: it opens the episode editor in
headless Chromium and does the work in the Target frames tab with real clicks and drags. Everything else —
`scripts/fw` — is the API. Use `scripts/fw target-frames EP RANGE` when the user only wants the detection
*job* queued; use this one when a person's box has to appear on specific frames.

```
node .agents/skills/framewise-target-frames/target-frames.mjs <inspect|draw|name> --episode <id> --cut <n|a-b> [options]
```

The app is `http://chopper.local:8080` unless `FRAMEWISE_URL` or `--url` says otherwise. Playwright is found from
`PLAYWRIGHT_MODULE`, then the `playwright` package, then the mise install; Chromium is `CHROMIUM_PATH` (default
`/usr/bin/chromium`) on the machine you run from. No auth, no cookies: the app is the LAN server.

When you are not on chopper (`hostname`), only the browser runs here: there is no Docker, container or worker on this
machine, so never build, restart or `docker compose` anything. A 503 from point detection means the inference worker
on chopper is down: tell the user, do not try to start it.

## The rule: the character is always the one on the first screen

The first screen of the run is the cut's opening frame, mirrored from the first-frame editor — the boxes the
user already confirmed there. A colour on it is a person the user has named. So the driver only ever works on
**one of those colours**: the person it draws or names is a colour that the opening frame shows, and the same
colour everywhere else in the cut. It never invents a new colour, never puts a box on a colour the opening
frame does not show, and never names a colour the opening frame does not show (`name` refuses unless
`--force`).

That is the whole point of the gate: the app's own keyframe analysis and its own frame review usually
already know whether that person is still in the picture on a later frame. Drawing a box where the app says
they are not is worse than drawing nothing.

## 1. Look first

`inspect` reads only the API and prints what the tab shows — the first screen's colours and names, and per
frame the colours present (`*` = the user drew or moved it) plus the tracker's candidates:

```
node .../target-frames.mjs inspect --episode <id> --cut 9
  first screen: colour 1 Sakuta Azusagawa, colour 2 Rio Futaba
  F1 (opening)  ord 950  colours 1 2
  F6  ord 957  colours 1
```

Then decide, always with `--dry-run` first: it prints the plan and the refusals and clicks nothing.

## 2. Draw or name

| The user says | Command |
|---|---|
| put Rio Futaba on the frames where she is missing | `... draw --episode <id> --cut 9 --character "Rio Futaba"` |
| add the character to cuts 5 to 10 | `... draw --episode <id> --cut 5-10` (only when the opening frame shows one person) |
| only frames 3 and 5-7 | `... draw --episode <id> --cut 9 --frames 3,5-7` |
| drag their last box instead of asking the detector | `... draw --episode <id> --cut 9 --mode carry` |
| also use a crop the tracker found for nobody | `... draw --episode <id> --cut 9 --allow-unmatched` |
| detect again first, then draw | `... draw --episode <id> --cut 9 --detect` |
| mark each frame you fixed reviewed | `... draw --episode <id> --cut 9 --mark-reviewed` |
| name colour 2 on cuts 9 to 12 | `... name --episode <id> --cut 9-12 --slot 1 --character "Rio Futaba"` |
| show me the window | add `--headed` (needs a display) or `--cdp http://127.0.0.1:9222` |

`draw` with one person on the opening frame needs no `--character`; with two or more it stops and asks which
one. It never touches a cut outside `--cut`, never deletes a box, and never changes a colour other than the
chosen one.

## What each frame action is

`draw` picks one of three, in this order, per frame that lacks the chosen colour:

1. **Accept the app's own suggestion** — a tracker candidate whose `suggested_slot` is the chosen colour:
   click `Accept` in that frame's *Suggested changes* list. Nothing is invented: the app already picked the
   place and the colour. With `--allow-unmatched`, a crop with no suggested colour is next: its box is drawn
   (not Promoted, because Promote opens a new colour) and moved to the chosen colour.
2. **Double-click where they were** (`--mode pick`, default) — the point is the centre of a crop the tracker
   found there, or of their box on the previous frame. The app's CPU detector answers the box and adds it
   like a drawn one. This needs the inference worker's point-detect thread: when it is not running the API
   answers 503 and the driver reports `the point-detection worker is not running, so no box was drawn`
   and changes nothing. Then use `--mode carry`, or tell the user to draw it.
3. **Carry their last box** (`--mode carry`) — drag the rectangle of their box from the previous frame. The
   box lands in that colour; the report says `(carried: check it is really them)` because it is a guess by
   position, and the driver only takes it when the app's evidence puts them in the picture (a crop there,
   or a keyframe state for them).

After a box lands the driver re-reads the API; if it arrived in another colour it moves it with the tab's own
*Person for the box* select, and reports the saved box either way.

## When it refuses (and why that is the right answer)

- **`no run yet` / `detection is running|queued` / `the run is stale`** — a cut-level block, exit 2. Wait,
  cancel, or add `--detect`.
- **The opening frame has no boxes** — nobody is on the first screen, so the rule has no character. The user
  boxes them in *Characters in this cut* first.
- **The keyframe state says they left** — e.g. `Not visible.` on that frame, or the run's own review flags
  `count_changed` with `here: 0`. It reports `skipped on F6` and does not draw: a box there would be a
  person the app already saw leaving.
- **The colour cannot join** — the frame already uses all its colours (`--frames` cannot fix that; a person
  has to merge or ignore a colour).
- **No name yet** — the first screen's person has no character: a person names them, then this runs with
  `--character`.
- **`--allow-unmatched` is not set** — a crop the tracker found for nobody is left alone; only the app's own
  colour suggestion is taken automatically.

`--detect` presses Detect characters before planning (or waits for a detection already queued) and plans from
the new run; with `--dry-run` it only says it would, and the plan shown is from the current run.

## Exit codes, like `fw`

`0` done or nothing to do · `1` an error · `2` a person is needed (a refusal, or a box the app could not find).

## What is not proven yet, so report it

The read-only paths and the whole browser navigation are proven against the live app, and so is the
double-click reaching the detector (it is the 503 that stops the box, not the driver). The **drag**, the
**Accept click** and **`--detect`** have never been executed on a real episode: their selectors come from the app's own
source and are checked against the live DOM, but a run that actually writes should be watched with
`--headed`, or a frame chosen where a wrong box is cheap to undo. Never run it on a cut the user has not
named, and never use `--detect` without asking: it replaces the run and its automatic boxes.
