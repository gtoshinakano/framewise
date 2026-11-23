---
name: framewise-results
description: "Run the Results stage of a Framewise episode: the agent's answers to the consistency review questions ('Your answers') and the Cut summaries, only for cuts whose review is current and nothing earlier is open, and only when the Autopilot profile turns them on (both are off by default). Use when the user asks for Cut summaries, final results, answering review questions automatically, or finishing an episode. Can run as its own sub-agent."
---

# Framewise: Results (opt-in)

Work from the repository root with `scripts/fw`. The rules are in `docs/autopilot/HARNESS.md` (D27, D31).

```
scripts/fw results EP [RANGE] --dry-run     # targets: cuts with questions to answer, cuts whose Cut summary is missing or stale
scripts/fw results EP [RANGE]               # answers (if on), then Cut summaries (if on), each once
```

Both are **off by default** in the Autopilot profile (episode menu → Autopilot profile: *review answers*, *Cut
summaries*). With both off, `fw results` only says so: tell the user how to turn them on; never change the profile
yourself. They are the most important output, so run them only after the person resolved everything before them:
`fw targets EP` should show no held cut and an empty inbox for the cuts in question.

A Cut summary target has a current, non-stale consistency review whose questions are all answered, and no earlier step
missing or stale. Review questions are never inbox items: the person answers them in the cut dialog (Results → Your
answers), and a saved answer stays valid for the same question after the review is written again.

## Final report: the links to open

End with `fw`'s report: what was queued, the cuts still waiting (questions without an answer, earlier steps not
current) with one `- cut N: URL` each, copied exactly.
