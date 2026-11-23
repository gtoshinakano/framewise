---
name: framewise-evaluate
description: "Run the Evaluate stage of a Framewise episode: visual analysis (first frame, keyframes), performance beats and the consistency review, each only for its target cuts (missing or stale, inputs settled by the person, no open inbox item); dry run first, then one pass that never re-runs anything in a loop. Also summarizes what each stage would run on and what the person still has to do (fw targets). Use when the user asks to evaluate, analyse, run visual analysis, beats or consistency, re-run what changed, asks which cuts are targets, or what is pending. Can run as its own sub-agent."
---

# Framewise: Evaluate (target cuts only)

Work from the repository root with `scripts/fw`. The rules are in `docs/autopilot/HARNESS.md` (D31, D32).

```
scripts/fw targets EP [RANGE]                     # every stage's dry run: targets per step, cuts held for the person and why
scripts/fw evaluate EP [RANGE] --dry-run          # what evaluate would run now
scripts/fw evaluate EP [RANGE]                    # visual → beats → consistency, each once, on its targets
scripts/fw evaluate EP [RANGE] --only visual      # one step only (visual | beats | consistency)
```

**A target** is a cut whose result is missing or stale *and* whose inputs are settled: first-frame boxes named or
strongly matched, target frames reviewed, speech with a speaker, the earlier result current, and **no open inbox item**
on the cut (a missed cut holds everything; box, person and target-frame questions hold visual and after; line splits
and voices hold beats and after). A cut the person fixed becomes a target on its own because the fix made its results
stale. Held cuts are listed with their reasons and a link each; they are never queued.

Visual work goes in chunks of 16 cuts and queues beats for a cut as soon as its keyframes are saved and it is ready;
the beats step then covers the rest. Each step runs once per command: if a step fails, `fw failures EP` / `fw triage EP`
show why; run `fw evaluate EP` again after fixing.

## "What is pending?"

Run `fw targets EP` (read-only) and summarize, in this order: what each stage would run on now (cut numbers), the cuts
held for the person grouped by reason with one link each, the inbox counts (for the person, for Claude), failed runs.
Say which command runs each part; queue nothing unless the user asks.

## While it runs: Claude's items

Same as Prepare: answer `fw inbox EP --reviewer claude` items with the resolver skills, `--tier claude`, `unsure` when
unsure. Never answer the person's items.

## Final report: the links to open

End with `fw`'s report: targets left per step, the HELD FOR YOU list (`- cut N: URL`, copied exactly), the inbox line,
and the next command (`fw evaluate EP` again after the person's fixes, `fw results EP` once nothing is held).
