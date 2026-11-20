# Audio experiment: findings

Status: written on 2026-11-19 **before** the first real run. The measured-today section is real (CPU work on the stored data); the verdict block is filled by
`audio_experiment/run.sh report <id>` from the finished run's `results.json`. Everything else below is design intent and the porting plan.

## Verdict (filled by the run)

<!-- verdict:start -->
_No real run has finished yet. `run.sh report <id>` replaces this block with the run's verdict, the ranking table and the time estimate._
<!-- verdict:end -->

## Round 1 scored against labels (2026-11-20)

Run `audio-20261119-2055` (116 min). The report's own anchors (voice-sample ranges) cannot judge the voice methods: each character has ONE sample whose
ranges sit seconds apart, so leaving out only the overlapping range still left the same scene in. Instead, 169 lines were labelled by listening:
`typical` (dry-run page: episode openings, consecutive lines, mapped onto this run's lines by time; the 7 lines labelled on both pages agree) and
`hard` (the report's disagreement picks). Both are in `labels/round1.json`.

| method | typical (75) | hard (91) |
|---|---|---|
| Q1 Qwen with everything | 96% | 71% |
| Q1s (shifted windows) | 97% | 77% |
| Q2 without subtitles | 96% | 57% |
| Q3 profiles / R1 episode voices | 96% | 68% |
| B0 stored transcript | 93% | 48% |
| V3 voice only | 89% (98.5% when it answers) | 43% (87% when it answers) |
| voice when it decides, else Q1 (offline) | 97% | 78% |

* Voice and Qwen agree on 76% of all 1,092 lines and are then ~98% right; they disagree on 2% (the voice is right ~70% there); on 22% the voice decides
  nothing, almost all lines under 0.8 s, and Qwen gets ~70% of those. Qwen overruled a clear voice match in 11 of its 26 hard-line errors.
* Qwen's confidence is not usable (high on 70% of its hard-line errors); it almost never says low or unknown, so R1 re-decided 5–17 lines per episode
  and the second-opinion stage read 0 lines.
* Subtitles help on hard lines (+14 points); speech profiles (Q3), R1 as built and the diarization variants did not.
* Pause-cut windows: unfinished lines 347 → 24 (Ep1), speaker switches inside a sentence 31 → 15 (Ep2). Names as ASR context barely help
  (梓川 and 咲太 never appear); Qwen's text fixes wrote wrong names (安積川, 相坂, 作矢) because its prompt had no Japanese spellings.
* The ED ranges start ~25 s late (Ep2 L435 and Ep3 L313 are lyrics kept as dialogue); LFM3 heard that stretch as singing and every sampled dialogue window as speech.
* Leaving out every reference within 60 s moves voice-only from 89% to 88% on typical lines: the leak was real but small.
* Possible label slips to re-check: Ep3 L22 「少なくとも国見は覚えていなかった。」 (labelled Yuuma, sounds like Sakuta's narration) and Ep2 L435 (ED lyrics, labelled Mai).

Round 2 (README) tests what this points to. Its thresholds were picked on these labels, so only labels from the round-2 spot check are a clean test.

## Measured before the run (CPU, stored data, 2026-11-19)

* **The stored transcript** is 3,084 / 3,160 / 2,772 word rows for Episodes 1 / 2 / 3 (median row 2 characters), 402 / 451 / 376 lines after the app's own merge. The transcribe job took 285-291 s per episode.
  Sentences (by final punctuation, speaker ignored) holding rows of **more than one speaker label: 11 / 31 / 27** of 356 / 422 / 301.
* **OP/ED search works and is consistent.** Fingerprints of all 7 episodes: the **OP is in all seven**, 88-90 s long (Ep1 0:52-2:22, Ep2 1:02-2:30, Ep3 1:10-2:38, Ep4 0:38-2:06, Ep5 0:01-1:29, Ep6 0:53-2:21, Ep7 3:02-4:32).
  Two episodes' encodes drift by about 1 s over the OP (a fixed-offset check missed them; the matcher follows the drift). The **ED is shared only by Episodes 2, 3 and 5** (about 1:00 of 1:20-1:24), so a "recurs in most episodes" rule would find the OP but not the ED.
  Episode 7 has a 3-minute cold open before its OP. Each pair takes seconds on the CPU, the whole search (7 fingerprints, 21 pairs) about 15 s.
* **Garbage in the stored transcript** (Episode 3, the first 140 s and the ED): Chinese and English lyrics assigned to a random voice. Inside the detected ranges the stored transcript has 15 / 12 / 21 lines; **Japanese OP lyrics are transcribed as Japanese**, so script
  alone cannot tell them from dialogue (14 of Episode 1's 15 look like Japanese sentences). The ranges themselves, not a text check, are what removes them. The report lists those lines so you can confirm that none is real dialogue.
* **Real models run offline on the CPU with the wrappers used here** (smoke test on a 52 s excerpt): pyannote takes an in-memory waveform, `pipeline.instantiate` changes the clustering threshold, `min/max_speakers` hints work, `pipeline._embedding` gives 256-d vectors;
  `Qwen3ASRModel.transcribe(..., context=..., language='Japanese', return_time_stamps=True)` returns tokens with the sentence punctuation already attached to the word.
* The GPU stages themselves (diarization settings, ASR, llama-server) were **not** run before launch: the dry run uses fake models, so a crash in a real GPU stage is the main risk of the run.

## What the run will tell you (read the report in this order)

1. **Verdict** and the ranking table: anchor accuracy with the references left out, contradictions, agreement, stability. Anchor n is small (a few dozen to ~150 lines), so read it as an ordering, not as final accuracy.
2. **V3 vs Q1 vs R1:** does the line's own voice beat the cluster's; does Qwen add on top of the voice; does the episode-level resolution fix the weak lines?
3. **Q1 vs Q2:** whether the Portuguese subtitles help or mislead (they were reported to mislead Qwen when it reconciled text).
4. **A1 vs A2 vs A3:** pause-cut windows vs fixed windows (lines cut at window edges, short lines, speaker switches inside a sentence), names as context, forced Japanese.
5. **OP/ED ranges** with timestamps to check in a minute, lines removed, lines possibly lost.
6. **Time table** and the per-episode estimate of the recommended pipeline against the current one.
7. **Disagreement samples with audio**, and `spotcheck.html` if you want to add real labels in 10 minutes (`run.sh report <id> labels.json` then scores them).

## Porting plan (fits AGENTS.md; to be confirmed or cut by the verdict)

1. **One episode-level dialogue result** instead of one row per token: lines with `start`, `end`, `text`, child `words[]` (token, start, end), `speaker_label` (diarization), `character_context_id` (nullable), `confidence` (high/medium/low), `reason`, `sources` (voice scores, Qwen run id, second opinion), revision links like `transcript_segments`.
   No approval status in the schema, API, prompts or UI. `transcript_segments` stays readable as history; a migration keeps current word rows as `origin='legacy'` lines or leaves them in place until the episode is re-transcribed.
2. **One reader for every consumer** (`prompt_context`, `analysis-context`, `subtitle`-style `cut_context`, the assistant's `get_transcript`, `/api/v1`, `sequence_timeline`, `sequence_compile`): they read lines through a single module instead of re-merging word rows (`merged_transcript` and `speech_lines` both become thin views).
3. **Pipeline in the queue:** register `transcribe` v2 in `job_kinds` as a chain of kinds: `audio_regions` (OP/ED fingerprint search across the project's episodes, CPU, cached per episode) → `diarize` (masked, GPU, audio worker) → `transcribe` (pause windows, audio worker) → `speakers` (text lane, `worker.TEXT_PIPELINES`, windows of 40 lines, validated, json_schema) → optional `reread` (audio worker, weak lines only).
   Each publishes with `JobInputs.publish`, records fingerprint, model ids and prompt in the result, and marks dependants stale (consistency summary, Cut summary, sequences' dialogue compilation).
4. **Characters:** voice references stay per range (store ranges' embeddings, not only the sample's), per-line embeddings are stored with the line so linking a voice to a character later re-labels without audio work; user-confirmed lines become enrolment evidence, never automatic labels (same rule as the character gallery).
5. **UI:** Audio tab shows lines (not words) with the speaker select and a confidence chip; "weak lines" filter; the existing Unknown-voices block assigns clusters. No review gate.
6. **OP/ED** ranges as an episode setting (`audio_ranges`), editable, shown on the timeline; masked for diarization and ASR, kept out of dialogue.

## Risks and caveats

* Small anchor sets (voice samples are single-speaker clips picked by the user; Episode 3 has none) and no real labels: a high score means "agrees with the evidence we have".
* Stored-transcript B0 and the app's cluster rule use the very samples the anchors come from (circular), so they are shown as such and never ranked as leave-one-out.
* Thresholds of the voice decision (line: 0.35 / margin 0.06; cluster by range: 0.45 / 0.05; app rule 0.7 / 0.08) are not tuned on held-out data. The report's `V3raw` (top-1 without a threshold) shows the ceiling of the voice evidence.
* Qwen3.6 on a line is only as good as its candidates: a character without a voice sample can be named from dialogue and context only.
* Qwen3-ASR with the names as context may echo names that were not said; A2/A3 are scored for that through the verified-name hits and the CER between readings, not against a truth.
* Anything the run skipped for time is listed in the report ("Changed, skipped and caveats").
* A better model would be needed for vocal separation (clean dialogue under music) and for a real audio-language model (Qwen3-Omni); neither is installed and nothing was downloaded.
