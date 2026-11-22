# Autopilot harness: plan and progress tracker

Started 2026-11-22. This is the single source of truth for the automation work. **Any agent (Claude Code, Codex, OMP) that picks this up reads this file first, works on the first unfinished milestone, and updates the Status board, the Progress log and, when a choice was made, the Decision log before its session ends.** Keep it short enough to read in one go: move finished detail into `docs/autopilot/<milestone>.md` and leave a one-line summary here.

## 1. Goal

A nearly fully automated analysis flow for Framewise episodes:

- Agents **apply** decisions themselves. Each human step today is a tiny UI action repeated hundreds of times, so a human approval gate is the bottleneck we are removing. Nothing waits for a person unless it is escalated.
- **Local GPU models do most of the work** (Qwen 3.6 on the R9700, CPU tools, deterministic rules). **Claude is the second reviewer** only where the local models cannot decide, because Claude tokens (especially images) are scarce.
- **The human only handles what gets escalated**, in one inbox, grouped so that one decision settles many items (one face, one voice, one kind of question).
- Every agent decision is **traceable and reversible**: who decided, on what evidence, with what confidence, and a one-step undo of a whole run.

Done means: a new episode goes from import to Cut summaries with the user touching only the inbox, and the inbox for a typical 350-cut episode takes well under an hour.

**Scope ends at the Cut summary.** Sequences (the Sequence Director: dialogue compilation, bible, plan, shots, review) are out of scope: the human creates and revises them. The autopilot never queues, edits or re-runs a sequence step.

## 2. How to resume (session protocol)

1. Read this file top to bottom, then `AGENTS.md` (the code map; it is long, search it for the files a milestone names).
2. Look at the Status board (section 9). Work on the first milestone that is not `done`, unless the user says otherwise.
3. Before coding, read the milestone's "Reuse" list in the code; the pieces exist and must not be rebuilt.
4. Measure before and after: every resolver has a replay score (section 6). Record numbers in the Progress log with the date and the episode.
5. Follow AGENTS.md "Reload the service after every change" and the constraints in section 8.
6. Before the session ends (or the context runs low): tick the tasks done, add a dated Progress log entry (what changed, files, test results, what is next, anything half-finished), and record any decision in the Decision log. A new session must be able to continue from this file alone.

## 3. Where the human time goes today (baseline, 2026-11-22)

From the user, in order of cost:

1. **Boxing the characters that matter to the story** (first-frame names, target-frame review, new characters).
2. **Fixing audio lines**, above all splitting long lines that bleed across a hard cut at an audio pause; this improves the analysis and the performance beats.
3. **Answering the consistency-summary questions.**
4. **Finding which cuts are wrong and why**, and root-causing failed analysis.
5. (Early on) **Missed hard cuts**: checking each cut's first and last frame and re-detecting with a lower threshold.

Measured with `scripts/fw analysis` and read-only SQL on 2026-11-22:

| | Ep 2 | Ep 3 | Ep 6 |
|---|---|---|---|
| Cuts with review questions to answer | 130 | 0 | 0 |
| Cuts with target frames to review | 64 | 5 | 0 |
| Unknown voices (lines) | 1 | 1 | 241 (5 labels) |
| First-frame boxes without a name (cuts) | 4 | 0 | 0 |
| Cuts with unconfirmed automatic labels | 141 | 161 | 246 |
| Cuts whose pictures contradict the analysis | 167 | 0 | 0 |
| Current transcript lines crossing a hard cut (by more than 0.15 s each side) | 36 | 35 | 16 |

Failed jobs in the last 21 days, grouped (most are mechanical, see M1):

| Failure | Count | Root cause |
|---|---|---|
| `analyze`: OpenRouter returned empty or truncated | 49 | provider |
| `regen`: invalid cut-relative interval / missing time-coded lines | 30 | validator loop |
| `performance_beats`: no keyframe analysis | 21 | prerequisite missing |
| `consistency_summary`: beats missing or older than keyframes | 12 | prerequisite stale |
| `character_detect`: invalid character match response | 7 | legacy matcher |
| `analyze`: preview frame timestamps invalid | 6 | stale cut previews |
| `transcribe`: CUDA out of memory | 5 | GPU held by another process |
| `sequence`: beat timing validator | 3 | validator loop |

## 4. Architecture

### 4.1 Three tiers of decision

| Tier | Who | When | Cost |
|---|---|---|---|
| 1 | Rules, CPU tools, local Qwen | Default for every decision | GPU time only |
| 2 | Claude (second reviewer) | Tier 1 is unsure: confidence below the action's threshold, two signals disagree, or a picture contradicts the analysis | Claude tokens, capped per episode |
| 3 | The human (inbox) | Neither tier can settle it, or it is a hard stop (4.3) | The user's time |

Thresholds per action come from the replay harness (section 6): the threshold for "apply" is the lowest confidence at which tier 1 still meets the action's precision target on the user's own past decisions.

### 4.2 Apply, then flag (never a gate)

- A decision is applied immediately with provenance; downstream work continues. Escalated items do not block other cuts, and block their own cut only when the next step would be built on a guess that spreads (4.3).
- The app already marks derivatives stale when a box, line or answer changes, so a later human correction flows downstream with no new mechanism.
- Every applied decision is written with `origin`/`label_source = 'agent'` and a row in `agent_decisions` (M0). This is provenance, **not an approval status** (AGENTS.md forbids approval statuses; do not add one).

### 4.3 Hard stops (tier 3 even with high confidence)

- **A person the gallery has never seen** who matters to the story: the agent never creates a character record; it groups the person across the episode and escalates one card.
- **Writing to the recognition gallery.** The gallery uses only `character_detections.label_source = 'manual'` (`anime_gallery.py`), so `agent` labels are excluded automatically. Only names the user selects become gallery references (D10); agent labels never do, whatever tier 1 and tier 2 agree on. A wrong gallery example spreads to every later recognition.
- **Creative intent questions** that the preference profile (M4) does not cover.
- **Cuts that belong to a sequence** are *not* a hard stop for production work (D7): stale results there are re-run like anywhere else, because a stale cut is worse than a stale shot the user knows about. The re-run marks the sequence's shots stale and leaves them stale (D9: the human decides what to do with them; the autopilot never rewrites them); the run's digest names each affected sequence and shot. The user rule of 2026-11-21 still holds for **tests and validation runs**: never use sequenced cuts for those.

### 4.4 Where the code lives

- **Resolvers that run per cut or per episode are backend job kinds** (`job_kinds.KINDS`, the queue, leases, `JobInputs.publish`, stale marking). They run on the local GPU/CPU, inside the existing worker lanes, so they inherit restarts, retries and the queue drawer. Follow AGENTS.md "How to add a queueable feature".
- **The supervisor is an agent session with skills** (`.agents/skills/`, linked from `.claude/skills`): it drives the flow with `scripts/fw`, diagnoses, runs tier 2 on the contact sheets the backend prepares, and writes the digest. It mostly reads compact JSON, never raw frames.
- **Tier 2 runs in the supervisor's Claude Code session, on the user's plan** (D6; no Anthropic API key in the backend): the backend queues `needs_claude` items with a prepared contact sheet and a compact question; the supervisor answers them and writes the decision back through the API.
- **`scripts/fw`** grows write commands for agents (labels, voices, line splits, answers, cut re-detection). Today its docstring says it never writes labels, boxes, transcript text, answers, dialogue or sequences; that rule changes under an explicit flag (`--as-agent`), which tags every write `agent` (Decision log D3).

## 5. Escalation policy (per decision type)

Starting values; the replay harness tunes the numbers. "Apply" = tier 1 decides.

| Decision | Apply when | Claude when | Human when |
|---|---|---|---|
| Re-detect a missed hard cut (M2) | in-cut scene peak well above a learned floor, not explained by camera motion | peak is in the grey zone, or camera fit is poor | Claude unsure; cut belongs to a sequence |
| Split a line at a pause (M3) | a quiet word gap of at least 0.3 s lies near the hard cut | never (no picture needed) | never; no pause = leave the line (bleed is acceptable) |
| Answer a factual review question (M4) | the evidence or a picture check supports one option | a picture contradicts the stored analysis and the local answer disagrees with it | Claude unsure |
| Answer a creative question (M4) | the preference profile covers this kind of question | never | not covered by the profile |
| Assign a voice label to a character (M5) | at least two independent signals agree (name tag, on-screen mouth, voice score, address by name) | signals disagree | Claude unsure, or the voice may be a new character |
| Name a box with a known character (M6) | gallery top candidate with margin, and relevance says the person matters | close candidates | Claude unsure |
| Box a new person who matters (M6) | never | never | always: one card per person across the episode |
| Leave a person unboxed as background (M6) | no usable face, or one appearance with no interaction | relevance is borderline | never |
| Resolve a target-frame flag (M6) | the flag has a rule (e.g. "Not visible" in the keyframe state explains a lower headcount) | merge/swap/new entrant without a rule | Claude unsure |
| Fix a failed job (M1) | its root cause has a fix rule | never | no rule, or the fix failed twice |
| Re-run a stale result | always, for every step up to and including the Cut summary (D7), in pipeline order so each re-run reads fresh inputs; cuts in a sequence included (digest lists the shots that turn stale) | never | the same re-run failed twice (M1 rule); never for sequence steps (out of scope) |

**Relevance rule (user, 2026-11-22):** a character matters when they interact with the story and must appear in generations, even if rare (example: a crying child). Background people without faces, and characters who appear a few times without interacting, do not. **Importance beats identifiability:** the user also boxes an important character of whom only a small part is visible (e.g. an extreme-foreground shoulder or hair), because it is context for the analysis, even though such a box cannot identify anyone. The agent does the same: a fragment of a character who matters in the neighbouring cuts is boxed and named from that context (e.g. the same character fully visible in the cut before or after), with `agent` provenance. Signals: visible face (box size, crop quality), speaks or makes sound (crying counts), others look at or touch them (`gaze_target`, beats contacts), foreground or centre (`frame_fill`, placement), recurrence across cuts.

**Line splitting rule (user, 2026-11-22):** split a long line that crosses a hard cut at the audio pause nearest the cut; most long lines have one and it is generally the right point. When there is no good pause, leave it: a phrase bleeding into the next cut is acceptable, the user moves it while building the sequence.

**Missed cuts (user, 2026-11-22):** the global threshold misses some cuts; the user used to compare each cut's first and last frames and re-split with a lower threshold.

## 6. Replay harness (measurement before trust)

Each resolver is scored against decisions the user already made by hand, with the human answer hidden. Lives in `autopilot/replay/` (outside `backend/app`, like the experiments; reads a copy of the database, never writes the live one). One scorer per resolver, one report per run with the date, the episodes and the git commit. A resolver ships its thresholds from this, and its score is re-run after every change.

Replayable data on 2026-11-22:

| Resolver | Ground truth | Volume |
|---|---|---|
| Missed cuts (M2) | `jobs` with `pipeline='hardcuts'` and `payload.cut_id`: the cut, the threshold the user chose (0.13 to 0.19), `detected_cut_ids` | 60 targeted runs, 48 split the cut; plus 5 `cut_edits` splits (the log keeps only 50 rows per episode) |
| Line splits (M3) | `transcript_segments` with `origin='manual_edit'` whose parent was split | 133 manual revisions (96 current in Ep 2) |
| Review answers (M4) | `cut_brief_instructions.content` (lines `- Topic (Character): option text`) against the consistency result's `questions` | 989 cuts with saved answers |
| Voices (M5) | `transcript_segments.speaker_source='manual'`, `srt_edit` speakers, voice links | 9 manual + 58 srt edits (thin; use Ep 2/3 assignments too) |
| Box names (M6) | `character_detections.label_source='manual'` | 1,081 manual labels |
| Target-frame flags (M6) | `target_corrections` | 167 corrections; `scripts/target_eval.py` already replays them |

Precision targets (approved by the user, D12): 95% for anything that writes boxes, labels or cut boundaries; 90% for answers and voices; line splits scored as "same split point within 0.1 s" or "correctly left unsplit". Because the user's own splits are sometimes workarounds (D11), M3 disagreements are reviewed by hand before they count against the splitter.

**Replay caveat for box names:** the user's manual labels include boxes that cannot identify anyone (small extreme-foreground fragments, kept for context). Score M6's naming only on boxes the gallery itself can use (`gallery_status`, `recognition_use`); fragments are scored on "boxed or not", not on recognition.

## 7. Milestones, in order

Order rationale: M0 makes every later write safe and measurable. M1 is cheap (no model) and saves time immediately. M2 comes first in an episode's flow (a cut change invalidates everything after it) and has the best replay data. M3 is well defined and cheap. M4 is the biggest volume. M5 and M6 are the riskiest (gallery cascade) and need the inbox. The inbox UI (M7) starts as a text report from M0 on and becomes a screen when there are enough escalation kinds. M8 makes it run unattended.

### M0. Foundation: provenance, decision log, undo, policy, replay skeleton

**Built 2026-11-22** (the plan below stays as the reference; differences noted here):
- Tables in `db.py`: `agent_runs`, `agent_decisions` (`cut_id` is plain text, never a foreign key, so a log entry never stops `cut_history` from removing a cut), `agent_escalations` (one open row per `(episode, kind, group_key)`, unique partial index), `autopilot_profiles`.
- `backend/app/agent_log.py`: `start_run`/`finish_run`/`runs`; `record(conn, …)` to call in the transaction that applies a change; `undo_decision`/`undo_run` (newest first, a conflict is skipped and reported, the run is marked `undone` only when nothing conflicted); `note_override(conn, table, id)` for user edit paths; `escalate`/`escalations`/`close_escalation`/`counts`. Undo goes through `UNDO[target_table]`; `RESTORABLE` lists the columns the generic `restore_columns` may write (`character_detections`: label columns; `cuts`: `no_characters`, `analysis_enabled`). **Each milestone registers its own undo** for revision chains (transcript lines) and cut boundaries.
- `backend/app/autopilot_profile.py` + the **Autopilot profile dialog** (episode menu → Autopilot profile…, `AutopilotProfileDialog.tsx`, helpers `autopilotView.ts`): per decision kind Apply / Ask me / Off; Claude on/off and items per run; per review-question topic a standing rule and "when unsure" (follow the evidence / take the recommended option / ask me) with this project's counts (asked, answered, how often the user took the recommended option); per character Auto / Always matters / Background; look-alike pairs; free notes. A run snapshots the profile into `agent_runs.policy_json` when it starts.
- `backend/app/autopilot_api.py`: `POST /api/episodes/{id}/agent-runs`, `POST /api/agent-runs/{id}/finish`, `GET /api/episodes/{id}/agent-runs`, `GET /api/agent-runs/{id}`, `GET /api/episodes/{id}/agent-decisions`, `POST /api/agent-decisions/{id}/undo` (409 on conflict), `POST /api/agent-runs/{id}/undo`, `GET|POST /api/episodes/{id}/escalations`, `POST /api/escalations/{id}/close`, `GET|PUT /api/projects/{id}/autopilot-profile`. Decisions are never written over HTTP: the code path that applies a change logs it.
- `scripts/fw`: `agent start|finish|runs|undo`, `inbox` (open questions with cut links, applied counts), `escalate`, `close`.
- Tests: `tests/test_autopilot_foundation.py` (11), `tests/test_fw_autopilot.py` (4), `frontend/tests/autopilotView.test.mjs` (5).

**Moved to the milestones that need them:** the per-table `agent` provenance values and the `note_override` calls in user edit paths (each resolver adds them for the tables it writes, with its undo handler); `fw --as-agent` write commands (each resolver adds its own); the replay skeleton `autopilot/replay/` (starts with M2's scorer; add `autopilot` to `.dockerignore` then).

**Profile data finding (2026-11-22):** of 360 current review questions, 181 are about speaking (mouth moving, when a line starts), then pose 58, state 42, gaze 30, emotion 19, identity 15. They are factual, not taste; where the user answered, they took the recommended option 74 of 109 times (68%), and their free-form answer lines are cut-specific ("this is a flashback", "they walk, not stand"). So the profile holds standing rules per topic, not preferred answers.

- **Provenance:** `label_source='agent'` for boxes; `origin='agent_edit'` for transcript revisions, analysis results and answers (check each table's existing `origin` values: `manual_edit`, `assistant_edit`, `srt_edit`, `model`); cut edits get `cut_edits.action` plus an agent marker.
- **`agent_decisions` table:** `id, episode_id, cut_id, run_id, kind, tier (rules|local|claude|human), action, target (table + id), before_json, after_json, evidence_json, confidence, policy_version, created_at, undone_at, overridden_by_user_at`. Written in the same transaction as the change (like `cut_edits`).
- **Agent runs:** `agent_runs` (`id, episode_id, started_at, finished_at, tier2_tokens, summary_json`), so one run can be listed and undone.
- **Undo:** `POST /api/agent-runs/{id}/undo` and per decision; an undo restores `before_json` only while the target still equals `after_json` (same rule as `cut_history`), else it reports a conflict.
- **User override capture:** when the user edits something whose current revision is an agent decision, set `overridden_by_user_at`; this is the training signal for thresholds.
- **Policy file:** `docs/autopilot/policy.yaml` read by the supervisor and passed into job payloads (snapshotted like `payload.models`): per decision kind the mode and thresholds, per character notes (e.g. "always ask between X and Y"), tier-2 budget per episode.
- **Escalations:** `agent_escalations` (`id, episode_id, kind, group_key, cut_ids_json, question, evidence_json, proposal_json, status open|resolved|dismissed, resolution_json`); grouped by `group_key` (a face cluster, a voice label, a question kind).
- **`fw` agent mode:** `--as-agent` writes; `fw inbox EP` prints open escalations with cut links (the inbox before M7).
- **Replay skeleton:** `autopilot/replay/` with the DB-copy loader and one trivial scorer. Add `autopilot` to `.dockerignore` (no image uses it).
- Reuse: `cut_history.py` (before/after + conflict rule), `JobInputs.publish`, `mark_stale`, `review_report.py`, `scripts/fw` (`cut_url`, `PLACES`).

### M1. Diagnose and auto-triage

**Built 2026-11-22** (the plan below stays as the reference; differences noted here):
- `backend/app/diagnose.py` (read-only): a cut is eight steps (`STEPS`: boxes, target_frames, first_frame, keyframes, beats, consistency, answers, summary) with `PREREQS`, each in one state (`ok`, `off`, `queued`/`running`, `failed`, `missing`, `stale`, `person`), read once per episode by `Facts` (the review report, current results with their dates and stale reasons, active jobs, `runs.attention_rows`). `health` ranks problems per cut (`SEVERITY`: failed 100, stale 60, contradiction 50, person 40, next missing step 30 with its root, line across a cut 20, unconfirmed labels / no boxes 10) plus episode problems (setup, unknown voices, stale attribution/transcript, episode-level failures). `why` walks the first broken prerequisite down to the root and gives the fix with a ready `fw` command. `classify` maps an error to a cause (`RULES`, first match); `FIXES` gives each cause its action and how many tries before a person is asked. A failure is **obsolete** when every cut it left unfinished is archived or has the step's output made after it failed (legacy `analyze` failures have keys the current workflow never repeats, so `runs.attention` alone would keep them forever).
- `backend/app/triage.py`: `apply` settles every open failure in one write transaction, one savepoint each: `dismiss` (obsolete), `rerun` (`main.retry_job`, the retry route's body, now callable inside a transaction), `requeue` (current settings, the user's request choices kept; consistency/answers/summary queue the stale or missing upstream first: visual stages, beats from current keyframes, the review), `prerequisite` (beats without keyframes: keyframe stage, then beats), `wait` (GPU memory while a GPU job runs), `leave` (sequences), `escalate`. Each fix is `agent_decisions` kind `failed_runs`, tier `rules`, target the failed job, `evidence.key` = the failure's `Kind.keys` (counts attempts). `agent_log.UNDO['jobs'] = triage.undo_fix`: cancels the jobs the fix created (every job inserted in its savepoint) while all are still queued, else 409; clears an agent dismissal. Escalations: one open `failed_run` per `cause:pipeline`, runs and cuts accumulate in its evidence, closed automatically (`close_settled`) once none of its runs fails any more.
- Profile: new decision kind `failed_runs` (Apply / Ask me / Off) in `autopilot_profile.DECISIONS`; a run's snapshot wins over the project's profile.
- Routes (`autopilot_api.py`): `GET /api/episodes/{id}/health?from_cut&to_cut`, `GET /api/cuts/{id}/why?step=`, `GET /api/episodes/{id}/triage?run_id=` (the plan), `POST /api/episodes/{id}/triage {run_id, job_ids}` (apply).
- `scripts/fw`: `diagnose EP [RANGE] [--kind K] [--limit N]`, `why EP CUT [--step S] [--all]`, `triage EP [--go] [--run RUN]` (exit 2 when it asked a person, 3 when it waits for the GPU). Skill `.agents/skills/framewise-diagnose/`.
- Tests: `tests/test_diagnose_triage.py` (9), `tests/test_fw_autopilot.py` (+2).
- **Not in M1:** "suspected missed cut" in health (M2 adds it); `fw drive` calling triage between steps (M8); a fix rule for `transcribe`/`subtitles`/`character_detect` failures (no requeue path with a connection: they escalate); switching an OpenRouter failure to another provider by policy (requeue uses the current text engine, which is what the user selected).

- **Cut health:** `GET /api/episodes/{id}/health` (or an extension of `/review`): per cut a ranked list of problems in one sentence each, worst cuts first: failed steps, stale results, picture contradictions, lines crossing the cut, unconfirmed labels, missing steps, suspected missed cut (after M2).
- **Root-cause chain:** for a failed or missing step, walk its prerequisites (`job_kinds` dependencies, `summary_queue.missing_evidence`, keyframe selection, cut previews) and report the first broken link: "consistency failed ← beats older than keyframes ← keyframes re-run 2026-11-20".
- **Fix rules** (tier 1, applied by the supervisor through `fw`): prerequisite missing → queue it and retry; preview timestamps invalid → refresh previews and retry; CUDA OOM → retry when the GPU is idle; validator loop → one retry, then escalate with the validator's sentence; provider empty → switch provider per policy or escalate. Two failed fixes of the same cut and kind → escalate.
- Skill: `framewise-diagnose` (`fw diagnose EP [RANGE]`, `fw why EP CUT`).
- Reuse: `review_report.py` (`CUT_FLAGS`), `runs.attention`, `fw` retry/dismiss, `analysis-status` skill.

### M2. Missed hard-cut checker

**Built 2026-11-22** (details, replay and what went wrong on the way: `docs/autopilot/M2.md`; the plan below stays as the reference):
- `backend/app/missed_cuts.py`: per-frame scene scores in one ffmpeg pass (cached), lone peaks inside a cut (isolation = peak / strongest neighbour within 12 frames), ORB + RANSAC across the peak (a fit = same shot), flash check; verdict apply / review / reject. **Splits exactly at the peak frame** (a `cut_history` split + `agent_decisions` kind `missed_cuts`, tier `rules`) in the job's publish transaction, not through a lower detection threshold (D15). A split that would leave a piece under 0.5 s is asked about instead; next to a boundary with no picture change (hand-placed a few frames off) the person gets it. Cuts with analysis off and peaks already asked are skipped; cuts in a sequence are never split by the agent.
- Job kind `missed_cuts` (CPU, media lane, API process), queued with `POST /api/episodes/{id}/missed-cuts {from_cut,to_cut,redo,run_id}` (waits for pending prepare/hardcuts; the profile's mode and Claude switch are snapshotted in the payload; `off` queues nothing). `missed_cut_checks` keeps the last check per cut (redone when its bounds, the proxy or `VERSION` change).
- **Tier 2 is wired**: `agent_escalations.reviewer` (`claude` | `human`), `agent_log.hand_to_person`, `GET /api/episodes/{id}/escalations?reviewer=`, `GET /api/escalations/{id}`. Doubtful peaks open one `missed_cut` escalation each (group `peak:<time>`) for Claude, with a strip at `GET /api/cuts/{id}/missed-cut-strip?time=` (4 frames around the peak + 8 over ±1 s). `POST /api/escalations/{id}/missed-cut {answer: split|no_cut|unsure, tier, note, run_id}` applies it (Claude cannot answer the person's questions or split a sequenced cut; `unsure` hands it to the person).
- Undo: `agent_log.UNDO_KINDS` (kind-specific handlers win over the table's) has `missed_cuts → missed_cuts.undo_split`, which merges the two pieces back while both are exactly as the split left them (deletes the created piece when nothing refers to it) and records a `cut_edits` merge. User cut edits (`edit_cut_timeline`, editor undo/redo) call `note_override('cuts', id)`.
- Triage (M1) can re-queue a failed `missed_cuts` job.
- Replay skeleton: `autopilot/replay/{common.py,run.sh,missed_cuts.py}` (DB copy through SQLite backup, throwaway API-image container, media read-only, reports in `autopilot/replay/reports/`); `missed_cuts_audit.json` holds the real cuts the user's timeline lacks, checked by eye. `autopilot` is in `.dockerignore`.
- `scripts/fw missed-cuts check|report|strip|answer`, `fw inbox --reviewer claude|human`. Skill `.agents/skills/framewise-missed-cuts/`.
- Tests: `tests/test_missed_cuts.py` (11), `tests/test_fw_autopilot.py` (+1).
- **Replay:** 146/146 applied splits are real cuts (107 the user added, 39 the user never added, verified on the strips); 107/138 user-added cuts found, 12 more to review, 19 not found (14 without any scene peak). About 7 review items per episode.
- **Not in M2:** running it after `hardcuts` in `fw drive` (M8); a "move the boundary" answer; dissolves split by eye.

- **Score each cut on the CPU:** strongest in-cut scene score (ffmpeg `scene` without a threshold, as `media.detect_cuts_in_range` does with one) and its time; first vs last frame difference.
- **Rule out camera motion:** fit `keyframe_camera` (ORB + RANSAC) between the frames on each side of the peak; a good fit means a pan or zoom, not a cut. Also rule out fades and flashes (luma ramps, near-white or near-black frames).
- **Apply:** queue `POST /api/cuts/{id}/detect-hardcuts {scene_threshold}` just under the peak. It is undoable through `cut_history`.
- **Tier 2:** a two-frame strip around the peak (one small picture) when the peak is in the grey zone.
- Runs right after `hardcuts` in the episode flow, before any per-cut analysis. A split of a cut inside a sequence changes the sequence's cut range: escalate that one instead of applying (sequences are the user's).
- Replay: the 60 targeted re-detections (which cuts the user re-split, the threshold, whether it split) plus the 5 `cut_edits` splits; also measure false splits on cuts the user never touched.
- Reuse: `media.detect_cuts_in_range`, `media_jobs.hardcuts` (targeted mode), `keyframe_camera.py`, `media.extract_cut_frame`, `contact_sheet.py`.

### M3. Line splitter at pauses

- For each current line crossing a hard cut: candidate points are gaps between words (`words_json`) where `sequence_timeline.find_pauses` on `audio_envelope` finds at least `MIN_PAUSE` (0.3 s) of quiet; choose the one nearest the cut (window to be tuned by replay); split there with the existing endpoint `POST /api/transcript-segments/{id}/split {at, left: {text}, right: {text}}` (texts built from the words, so word times are kept). No pause near the cut → leave the line, record "left as bleed".
- Never splits inside a word. May split and retime any line, including the user's `manual_edit` and `srt_edit` lines (D11: the user's splits are sometimes workarounds or wrong, and agents may correct them). It changes timings and where a line is divided, never the words of an `srt_edit` line.
- A backend job kind (CPU, per episode), queued after transcription and after M2 (cut boundaries must be final).
- Replay: the 133 manual revisions (96 current in Ep 2): same split point within 0.1 s, or correctly left unsplit.
- Reuse: `sequence_timeline.audio_envelope/find_pauses/quietest/speech_edge`, `line_edit.split`, `transcript_parts.parts`, `lineEditView.splitTime` (the UI's rule).

### M4. Consistency answers

- **Preference profile:** mine the 989 saved answers: per question `topic` (and character), which option the user picks relative to `recommended`, and free-text habits; store as `docs/autopilot/preferences.yaml` (reviewable by the user) or a table.
- **Classify each question:** factual (what is visible, said or physically true; checkable against pictures/transcript) vs creative (how to play or stage it).
- **Apply:** factual → `review_answers` drafts with the evidence; picture contradictions → the option that matches the picture (already the recommended option there); creative → the profile's choice. Save with `auto_save` and `origin` agent.
- **Tier 2:** factual questions where the local answer disagrees with a picture check; Claude gets the picture references already chosen by `consistency_evidence`.
- **Human:** creative questions the profile does not cover; one card per question kind where possible.
- Replay: hide `cut_brief_instructions`, regenerate answers, score option agreement per topic.
- Reuse: `review_answers.py`, `consistency_summary.question_format`, `consistencyView.setAnswer` format, `consistency_evidence.py`, `fw answers --save`.

### M5. Voice resolver

- Per diarization label (then per line for leftovers): collect `[名前]`/`(名前)` subtitle and .srt name tags, names addressed in the lines (glossary), on-screen speakers with moving mouths at those seconds (first-frame boxes, keyframe mouth states, performance beats delivery), `line_voices` scores against character voice samples; re-listen with `transcribe_cut_audio` only when they disagree.
- **Apply** when two independent signals agree: `POST /api/episodes/{id}/transcript/assign-voice` for the label, or `PUT /api/transcript-segments/{id}` per line; then queue speaker attribution again.
- **Human:** a voice that may be a new character, or disagreement Claude cannot settle; one card per label with sample lines and play links.
- Replay: manual and srt speaker assignments; Ep 6's 241 open lines are the live test.
- Reuse: `line_voices.py`, `speaker_attribution`, `framewise-srt-fix` skill, `assistant/audio_transcription.py`, `voiceGroups.unknownVoices`.

### M6. Characters that matter: naming, grouping, relevance, target frames

- **Relevance score** per detected person from the signals in section 5; background → `background_people`, unboxed.
- **Known people:** apply the gallery's top candidate with margin (`candidates_json`), `label_source='agent'` (excluded from the gallery, 4.3).
- **Unknown people:** cluster unmatched boxes across the episode by CCIP embedding (`anime_models.create_embedder`); one escalation per cluster with a contact sheet, the cuts, and why they matter ("cries; Sakuta looks at them"). The human names them once (existing new-character flow, `NewCharacterDialog`/`POST /api/character-detections/{id}/new-character`), then the agent labels every box in the cluster.
- **Unconfirmed automatic labels:** tier 1 re-checks with margin and relevance; tier 2 on a contact sheet for the close ones; the human confirms in bulk from the inbox (that confirmation makes them `manual` and gallery examples).
- **Fragments of important characters:** boxed and named from the neighbouring cuts (section 5). They never reach the gallery (agent labels never do).
- **Out of scope for now (future item):** a bulk "exclude from character recognition" for the user's own manual boxes that cannot identify anyone (extreme-foreground fragments). Today this exists only per box (`PUT /api/character-detections/{id}/recognition-use`, plus the automatic `gallery_status` for boxes too small or closer to another character).
- **Target frames:** one rule per review flag (`frame_confidence` flags: low match, count changed, unmatched crop, suspected merge, compensated jump, duplicate name); write through the Target-frames endpoints directly (API-level `fw target …`), not Playwright.
- Replay: 1,081 manual labels; `scripts/target_eval.py` on the 167 `target_corrections`. Qwen did not meet the 95% gate as a target-frame judge (`docs/target-frames-evaluation.md`), which is why tier 2 is Claude here.
- Reuse: `anime_bulk.py`, `anime_gallery.py`, `target_characters.py`, `target_tracking`, `point_detect.py`, `framewise-target-frames` skill (its rules on colours), `contact_sheet.py`.

### M7. Inbox (human review UI)

- One page per episode (and a cross-episode count in the header): **Your decisions** (open escalations grouped by `group_key`, keyboard first: one key to accept the proposal or pick an option), **Applied by agent** (random sample per kind to spot-check, filter by kind/tier, undo a run), **Problems** (unfixable failures with root cause and a fix button), and deep links into the existing editors for anything complex.
- Every user action here resolves an escalation or overrides a decision, which feeds the thresholds (M0 override capture).
- Until M7 ships: `fw inbox` text report with cut links (M0).

### M8. Autopilot orchestrator

- Extend `fw drive` with the resolvers: instead of stopping at a gate, run the resolver for that gate, apply, escalate the rest, continue. Order of an episode: setup → hardcuts → **M2** → transcription → **M3** → **M5** → first-frame boxes → **M6** → target frames → **M6 flags** → visual → beats → consistency → **M4** → Cut summaries; **M1** triage between every step; then a **stale sweep** (D7): re-run every stale result up to the Cut summary in pipeline order (`fw` today never re-runs stale results; the autopilot does), and repeat until nothing up to the Cut summary is stale.
- Runs headless on chopper (`claude -p` with the `framewise-autopilot` skill from a systemd timer, or `/loop`, on the user's Claude plan), sleeping on `fw wait` while the GPU works; a tier-2 budget per run; never queues a sequence step.
- Ends every run with a digest: applied counts per kind, escalations with links, failures fixed and not fixed, tier-2 tokens spent (an artifact page or `fw digest`).

## 8. Constraints that bind this work (from AGENTS.md and memory)

- No approval status on results, transcript segments or subtitle lines. Provenance (`agent`) is allowed; approval is not.
- Never add automatic or agent labels to the recognition gallery.
- Never run *test or validation* jobs on cuts that belong to a sequence (it stales shots); check `sequences` first or pick other cuts. Production stale re-runs are allowed there (D7).
- Sequence steps are out of scope: never queue, edit or re-run them.
- No ad-hoc ROCm tests beside the running llama-server (kernel oops); test on CPU or through queued jobs.
- Long work goes in the queue (job kinds), never in an HTTP request.
- Restart the inference or audio worker after changing files they import; check that no job is `running` first.
- Do not download models unless a task requires them.
- Reports from skills end with one openable URL per cut (`fw cut_url`, `PLACES`).
- Test recipe and suite: see memory `backend-test-setup`; frontend tests in `frontend/tests/*.test.mjs`.

## 9. Status board

| Milestone | Status | Owner/session | Notes |
|---|---|---|---|
| M0 Foundation | done (2026-11-22) | Claude Code | per-table provenance, undo handlers and override hooks land with each resolver |
| M1 Diagnose and triage | done (2026-11-22) | Claude Code | live-checked on a copy of the database; not yet run with `--go` on the live service |
| M2 Missed-cut checker | replay (2026-11-22) | Claude Code | 146/146 replay splits real; not yet run on the live service (not restarted at the user's request) |
| M3 Line splitter | not started | | |
| M4 Consistency answers | not started | | |
| M5 Voice resolver | not started | | |
| M6 Characters that matter | not started | | |
| M7 Inbox UI | not started (text inbox in M0) | | |
| M8 Autopilot orchestrator | not started | | |

Statuses: `not started`, `design`, `in progress`, `replay` (built, being scored), `live` (applied on a real episode), `done`.

## 10. Open questions

1. ~~Tier 2 transport~~ → D6.
2. ~~Precision targets~~ → D12.
3. ~~Agent labels and the gallery~~ → D10.
4. ~~Stale re-run policy~~ → D7, D9.
5. ~~Preference profile storage~~ → D14.
6. ~~Lines edited by hand~~ → D11.

## 11. Decision log

- **D1 (2026-11-22, user):** agents apply decisions; no human approval gate. Escalations go to an inbox and are non-blocking.
- **D2 (2026-11-22, user):** local GPU first; Claude only as second reviewer where the small models cannot decide.
- **D3 (2026-11-22):** `fw` gets write commands for agents behind `--as-agent`; every agent write carries `agent` provenance.
- **D4 (2026-11-22, user):** relevance rule, line-splitting rule and missed-cut practice as written in section 5.
- **D5 (2026-11-22):** milestone order M0 → M8 as in section 7.
- **D6 (2026-11-22, user):** tier 2 (Claude) runs in a Claude Code session on the user's plan; no Anthropic API key in the backend.
- **D7 (2026-11-22, user):** stale results are re-run automatically for every step up to and including the Cut summary; a stale result should never surprise the user. Cuts in a sequence are included; the digest reports the shots that turn stale.
- **D8 (2026-11-22, user):** the Sequence Director is out of scope; the human creates and revises sequences.
- **D9 (2026-11-22, user):** shots made stale by an autopilot re-run stay stale; the human decides what to do with them.
- **D10 (2026-11-22, user):** only names the user selects become recognition-gallery references; agent labels never do.
- **D11 (2026-11-22, user):** agents may split and retime any transcript line, including lines the user split or edited; the user's edits are sometimes workarounds and agents may correct them.
- **D12 (2026-11-22, user):** precision targets approved: 95% for boxes, labels and cut boundaries; 90% for answers and voices.
- **D14 (2026-11-22, user):** the profile is a table edited in the app (Autopilot profile dialog), not a file in the repo; it replaces the planned `docs/autopilot/policy.yaml` and `preferences.yaml`.
- **D15 (2026-11-22):** a missed cut is split exactly at the peak frame (a cut_history split), not by re-detecting with a lower threshold: the threshold splits at every frame above it and blocked 49 of 87 good splits in the replay. A split that would leave a piece under 0.5 s is asked, never applied. Cuts with analysis off (openings, endings) are not checked.
- **D16 (2026-11-22):** tier 2 is an escalation with `reviewer='claude'`; Claude answers through the resolver's answer endpoint with `tier=claude`, and `unsure` hands the item to the person with Claude's note. Claude never answers an item that is the person's.
- **D13 (2026-11-22, user):** importance beats identifiability for boxing: important characters are boxed even when only a fragment is visible. A bulk "exclude from recognition" for such manual boxes is a future item, out of scope now.

## 12. Progress log

Newest first. One entry per session: date, milestone, what changed (files), tests and numbers, what is next, anything left half-done.

- **2026-11-22: M2 built and replayed** (see M2 "Built" and `docs/autopilot/M2.md`). Files: `backend/app/{missed_cuts.py,missed_cuts_api.py,job_kinds.py,agent_log.py,autopilot_api.py,db.py,main.py,triage.py,diagnose.py}`, `autopilot/replay/*`, `scripts/fw`, `.agents/skills/framewise-missed-cuts/` (+ link), `.dockerignore`, tests. Replay over Episodes 1-7: 146 splits, all real (107 user-added, 39 audited by eye); recall 107/138. Live check on a database copy of Episode 7: 4 splits (all real), 6 questions for Claude, 1 for a person. **The service was not rebuilt** (the user asked to wait until the next step): `video-analyzer` needs `docker compose build video-analyzer && docker compose up -d video-analyzer` (API process only: the media worker runs the job there; no inference or audio worker change). The schema change (`missed_cut_checks`, `agent_escalations.reviewer`) applies on that start. Next: M3 (line splitter).
- **2026-11-22: M1 built** (see M1 "Built"). Files: `backend/app/{diagnose.py,triage.py,autopilot_api.py,autopilot_profile.py,main.py}` (`retry_job` split out of the retry route), `scripts/fw`, `.agents/skills/framewise-diagnose/` (+ `.claude/skills` link), tests listed in M1. Measured on a copy of the live database: `health` takes 0.09-0.18 s per episode (Ep 2: 301 of 350 cuts with a problem: 352 picture contradictions, 97 stale, 194 person, 221 missing, 72 lines across cuts; Ep 6: 47 lines across cuts, 246 unconfirmed labels); only one failure is open on all episodes (Ep 7 cut 84, Cut summary validator loop → requeue, attempt 1 of 1); its apply → decision → `fw agent undo` round trip was run on the copy. Next: M2 (missed-cut checker) with the replay skeleton `autopilot/replay/`.
- **2026-11-22: M0 built** (see M0 "Built"). Files: `backend/app/{db.py,agent_log.py,autopilot_profile.py,autopilot_api.py,main.py}`, `scripts/fw`, `frontend/src/{AutopilotProfileDialog.tsx,autopilotView.ts,EditorHeader.tsx,App.tsx}`, tests listed in M0. Next: M1 (diagnose and triage), then M2 with the replay skeleton.
- **2026-11-22: decisions D9–D13 recorded** (sequence shots stay stale, gallery only from the user, agents may retime the user's lines, precision bars approved, fragments of important characters are boxed). Only question 5 is open, to decide in M4. Next: M0 design.
- **2026-11-22: decisions D6–D8 recorded** (tier 2 on the Claude Code plan, stale re-runs up to the Cut summary, sequences out of scope). Open: questions 2, 3, 5, 6. Next unchanged: M0 design.
- **2026-11-22: harness created.** Brainstorm with the user; baseline measured (section 3), replay data counted (section 6), plan written. No code yet. Next: M0 design (schema of `agent_decisions`, `agent_runs`, `agent_escalations`; provenance values per table; `fw --as-agent`), then implementation.
