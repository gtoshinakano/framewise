---
name: framewise-review-answers
description: "Answer a Framewise episode's consistency review questions automatically: the local model decides each question, the agent applies what a standing rule or a trusted topic settles (undoable), Claude as the second reviewer looks at the picture sheet for the doubtful ones, and the rest goes to the person. Also measures the agent against the user's own answers (shadow run). Use when the user asks to answer, decide or settle the review questions, the consistency questions or production decisions of an episode or cut range, or asks how well the agent answers them."
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
- otherwise follows the topic's *when unsure* setting: take the recommended option, ask the person, or *follow the
  evidence*, which hands the cut to **Claude** (while the run's Claude budget lasts) and then to the person;
- never touches a question the user already answered, and never answers a creative question no rule covers.

Every applied answer is one line in the cut's *Your answers*, logged as an agent decision of kind `review_answers`; the
Cut summary turns stale as with any answer. A person's edit of those answers marks the decision overridden.

## Claude as the second reviewer

For each open question with `[Claude]` in the report (`fw inbox EP --reviewer claude` lists them too):

1. Read the questions, their options, the recommended one (`*`), the local proposal and why it was asked.
2. Fetch the sheet (`fw review-decisions sheet ID`) and look at it: it holds the pictures the reviewer cited (first frame,
   keyframes F1…, line strips of four frames per spoken line). One sheet per cut; never fetch frames one by one.
3. Answer with option ids only (`q1=a`), `--tier claude --run RUN`, and a one-line `--note` saying what the picture shows.
   Put every question the pictures cannot settle in `--unsure`: it goes to the person with your note. Claude never
   writes its own text and never answers a question that waits for the person.

Known weak evidence (from the user's own answers): a mouth that looks closed in a line strip is often still lip-synced
when the speaker is visible and faces the camera; a transcript timing far from the picture is usually a line bleeding
across a hard cut; a last keyframe showing another shot is a missed hard cut, not a change inside the cut. Prefer
`--unsure` over a guess on those.

## Before running it

The cut needs a current, not stale consistency review (`fw consistency EP RANGE` first). The questions of cuts that belong
to a sequence are answered like any other (production work, D7), but never use sequenced cuts for a test: a shadow run
changes nothing and is safe anywhere.

## Final report: the links to open

Say how many questions were answered (by rule, by trusted agreement, by Claude), how many went to the person and the
trust per topic, then list one `- cut N: URL` line per cut that was answered or still waits (`fw links EP N,M --place
analysis`), so the user can check the answers on the Results tab.
