---
name: framewise-prepare
description: "Run the one-time Prepare stage of a Framewise episode (hard cuts, missed-cut check, transcription and speakers, line splits, voices, first-frame boxes and recognition, target frames on every cut, the cast resolver; burned-in subtitles only when asked), then stop and list what the person must check; also prepare-cuts (boxes and target frames for cuts split, merged or re-boxed later). Use when the user asks to prepare, import, set up or start an episode, or to prepare cuts that changed. Can run as its own sub-agent."
---

# Framewise: Prepare (runs once)

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). The
stages and their rules are in `docs/autopilot/HARNESS.md` (D31, D32, D37). Prepare is a **Run Multiple** run: the server
carries the steps out one after another (`backend/app/batch_runs.py`, the editor's Run Multiple dialog shows the same run);
`fw` starts it, follows it and prints what each step did and found. It does everything that can really be automated
**once**, then stops: the person fixes what the Inbox and the held list show, and nothing in Prepare runs again.

```
scripts/fw prepare EP --dry-run                  # what it would do: each step done / will run / off, with its cuts
scripts/fw prepare EP                            # run it
scripts/fw prepare EP --subtitles Japanese       # also read the burned-in subtitles (optional; nothing waits for them)
scripts/fw prepare-cuts EP --dry-run             # later: cuts whose boundaries or first-frame boxes changed
scripts/fw prepare-cuts EP                       # boxes + recognition, target frames, then the cast resolver, for those cuts only
```

`EP` is an id or `"Project / Episode"` text (`fw episodes`). Run the long commands in the background; they follow the
run themselves. Never poll in a tight loop. Closing `fw` (or `--timeout`) never stops the run: it goes on in the server.

## Order (fixed in the server, `batch_runs.ORDER`)

media → hard cuts (once) → missed-cut checker (once; it splits only the peaks it is sure of and **never asks**: doubtful
places are counted in the step's summary) → transcription and speaker attribution (once) → subtitles (only with
`--subtitles`) → line splitter (once; never asks) → voice resolver (once; **the only step that asks the person about
audio**, one question per voice, in the Inbox) → boxes and recognition → target frames on every cut → cast resolver.
A resolver switched off in the Autopilot profile is skipped. Failed jobs with a fix rule are retried once; a failure of
media, hard cuts or transcription ends the run (`failed`), and `fw prepare EP` can start it again (it skips what is done).

**Prepare is done once its run finished.** `fw prepare EP` afterwards refuses and prints what it did. After that only
`prepare-cuts` touches Prepare's work. Automatic audio work (transcription, speaker attribution, line splits, voices)
and subtitles never mark anything stale and never overwrite a line the person edited.

## While it runs: Claude's items

When the Inbox has items for Claude (`fw inbox EP --reviewer claude`), answer them with the resolver skills
(`framewise-voices`, `framewise-cast`), with `--tier claude --note "what you saw"`. When unsure, answer `unsure`: the
person gets it. Never answer the person's items, never create or number a character.

## The person's last word

Whatever the person sets by hand (a label, a cleared label, a rejected box, "leave it", a cleared speaker, a merged-back
cut, an undone split) is final: no resolver asks or changes it again. Do not work around that.

## Final report: the links to open

End with what `fw` printed: each step's summary line, the inbox line (`inbox: voice 2 for you, …`) and the inbox link
(`#/episode/<id>`, header → Inbox). Then the next commands: the person answers the Inbox, then `fw prepare-cuts EP`
(only if they split, merged or re-boxed cuts) and `fw evaluate EP`.
