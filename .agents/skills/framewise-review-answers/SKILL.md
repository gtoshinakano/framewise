---
name: framewise-review-answers
description: "Answer a Framewise episode's consistency review questions automatically: the local model decides each question, the agent applies what a standing rule or a trusted topic settles (undoable), and the rest stays open for the person in the cut dialog (never the inbox). Also measures the agent against the user's own answers (shadow run). Use when the user asks to answer, decide or settle the review questions, the consistency questions or production decisions of an episode or cut range, or asks how well the agent answers them."
---

# Framewise: answers to the consistency review questions

Work from the repository root with `scripts/fw` (HTTP only, `FRAMEWISE_URL`, default `http://chopper.local:8080`). Part of
the autopilot (`docs/autopilot/HARNESS.md` M4; rules and replay numbers in `docs/autopilot/M4.md`).

```
scripts/fw agent start EP --agent claude-code --purpose "review answers"   # groups the changes and the Claude budget (prints RUN)
scripts/fw review-decisions run EP [RANGE] --run RUN    # one local decider job per cut with open questions; then: fw wait EP --jobs ...
scripts/fw review-decisions report EP [RANGE]           # what it answered (undo ids), open questions with pictures, trust per topic
scripts/fw review-decisions sheet ESCALATION_ID         # the pictures the review looked at, as one JPEG (review-ID.jpg)
scripts/fw review-decisions answer ESCALATION_ID q1=a q2=b --tier claude --run RUN [--unsure q3] --note "why"
scripts/fw review-decisions run EP --shadow             # measure only: the user's answered cuts, decided with their answers hidden
scripts/fw agent undo DECISION_ID --decision            # put one cut's answers back (only while nobody edited them)
```

## What it does

Per question the local model returns a choice, fact or creative, a confidence and whether a **standing rule** of the
Autopilot profile decides it. The resolver then:

- **applies** it when a standing rule decides it with high confidence, or when the model agrees with the reviewer's
  recommended option with high confidence *and* the topic's measured agreement with the user is at least 90%
  (`trust` in the report: from shadow runs once a topic has 10, else how often the user took the recommended option);
- otherwise follows the topic's *when unsure* setting: take the recommended option, or leave the question open for the
  person, who answers it in the cut dialog (Results → Your answers). Review questions are never inbox items and never go
  to Claude (HARNESS.md D32: an episode has hundreds of them);
- never touches a question the user already answered, and never answers a creative question no rule covers.

Every applied answer is one line in the cut's *Your answers*, logged as an agent decision of kind `review_answers`; the
Cut summary turns stale as with any answer. A person's edit of those answers marks the decision overridden.

## Questions left open from before

Review questions escalated before 2026-11-24 were dismissed once. If `fw inbox EP` still lists a `review_question`
item, answer it with `fw review-decisions answer ID q1=a …` (the person's choice, or `--tier claude` with option ids and
a `--note` only when the user asks Claude to look at its sheet, `fw review-decisions sheet ID`).

## Before running it

The cut needs a current, not stale consistency review (`fw consistency EP RANGE` first). The questions of cuts that belong
to a sequence are answered like any other (production work, D7), but never use sequenced cuts for a test: a shadow run
changes nothing and is safe anywhere.

## Final report: the links to open

Say how many questions were answered (by rule, by trusted agreement), how many stay open for the person and the
trust per topic, then list one `- cut N: URL` line per cut that was answered or still waits (`fw links EP N,M --place
analysis`), so the user can check the answers on the Results tab.
