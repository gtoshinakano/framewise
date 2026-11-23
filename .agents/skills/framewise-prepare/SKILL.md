---
name: framewise-prepare
description: "Run the one-time Prepare stage of a Framewise episode (hard cuts, missed-cut check, transcription and speakers, subtitles, line splits, voices, first-frame boxes and recognition, target frames on every cut, the cast resolver), then stop and list what the person must check; also prepare-cuts (boxes and target frames for cuts split, merged or re-boxed later) and the line checker. Use when the user asks to prepare, import, set up or start an episode, to prepare cuts that changed, or to check or split lines across hard cuts. Can run as its own sub-agent."
---

# Framewise: Prepare (runs once)

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). The
stages and their rules are in `docs/autopilot/HARNESS.md` (D31, D32). Prepare does everything that can really be automated
**once**, then stops: the person fixes what the inbox and the held list show, and nothing in Prepare runs again.

```
scripts/fw prepare EP --dry-run                  # what it would do (steps done / to do)
scripts/fw prepare EP --language Japanese        # run it (the language of the burned-in subtitles; asked once)
scripts/fw prepare-cuts EP [RANGE] --dry-run     # later: cuts whose boundaries or first-frame boxes changed
scripts/fw prepare-cuts EP [RANGE]               # boxes + recognition, target frames, then the cast resolver, for those cuts only
scripts/fw lines EP --dry-run                    # lines across a hard cut the splitter would still look at
scripts/fw lines EP                              # the line checker and splitter
```

`EP` is an id or `"Project / Episode"` text (`fw episodes`). Run the long commands in the background; they wait on the
queue themselves and print a line when the number of questions for Claude changes. Never poll in a tight loop.

## Order (fixed in fw, `stage_next`)

setup (media, hard cuts, transcription, subtitles, boxes and recognition) → missed-cut checker → boxes for the pieces of
split cuts → line splitter → voice resolver → target frames on every cut → cast resolver (box names, unknown people,
flagged target frames). Each step runs at most once per command; failed runs with a fix rule are triaged between steps.
A resolver switched off in the Autopilot profile is skipped.

**Prepare is done once its agent run finished.** `fw prepare EP` afterwards refuses and prints what waits for the person.
After that only `prepare-cuts` (cuts the person split, merged or re-boxed: no detection of their boundaries, or a Target
frames run that is stale) and `lines` touch Prepare's work.

## While it runs: Claude's items

When it prints `for Claude: N open item(s)`, answer them with `fw inbox EP --reviewer claude` and the resolver skills
(`framewise-missed-cuts`, `framewise-voices`, `framewise-cast`), with `--tier claude --note "what you saw"`. When unsure,
answer `unsure`: the person gets it. Never answer the person's items, never create or number a character.

## The person's last word

Whatever the person sets by hand (a label, a cleared label, a rejected box, "leave it", a cleared speaker, a merged-back
cut, an undone split) is final: no resolver asks or changes it again. Do not work around that.

## Final report: the links to open

End with what `fw` printed: the inbox line (`inbox: voice 2 for you, …`), the HELD FOR YOU list with one
`- cut N: URL` per cut, and the inbox link (`#/episode/<id>`, header → Inbox). Then the next commands: the person fixes
those, then `fw prepare-cuts EP` (only if they split, merged or re-boxed cuts) and `fw evaluate EP`.
