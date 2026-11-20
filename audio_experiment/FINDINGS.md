# Audio experiment: findings

Status (2026-11-20): **finished; the recommended flow is fixed and ready to port.** Round 1 (`audio-20261119-2055`) compared the approaches, round 2
(`audio-r2-20261120-0208`) tested the changes round 1 pointed to. Both were scored against 169 lines labelled by listening (`labels/round1.json`).
**Implementation agent: read `IMPLEMENTATION.md`, not the rest of this folder.**

## Verdict

<!-- verdict:start -->
* **Speaker:** a clear match of the line's OWN voice against per-range voice references is final (96 % right, 70 % of lines). A short line gets a second try
  from its clip grown into the surrounding silence (77 %). Qwen3.6 answers only the remaining ~25 %, reading the whole conversation with the settled speakers
  in it (82 %). Overall 88–89 % on the labels against 69 % for today's transcript (hard lines 81 % against 48 %; typical lines 97 % against 94 %).
* **Lines, not words:** pause-cut ASR windows and one speaker per sentence remove most broken sentences and speaker switches inside a sentence.
* **OP/ED:** the cross-episode audio fingerprint finds the OP and the shared ED; masking them removes the lyrics that were transcribed as dialogue.
  The LFM3 music map added only one 13 s edge and is deferred.
* **Names:** turning the kana reading of a verified name into its spelling is always right (84 changes); a Qwen name pass fixes kanji mishearings
  (48 of 62 right) but needs a stricter validator (12 wrong fixes, all of a few kinds, in `labels/name_fixes.json`).
* **Cost:** about 8.5 min per 25-minute episode (diarization 74 s, ASR 170 s, voice clips 15 s, Qwen speakers 150–245 s, names 80–145 s) against 3.2 min today.
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

## Round 2 (2026-11-20, `audio-r2-20261120-0208`, 18 min)

Built on round 1's checkpoints (same line ids). Every voice score left out the reference ranges within 60 s of the clip (an evaluation device only).

| method (labels: all 156 / hard 85 / typical 71) | all | hard | typical |
|---|---|---|---|
| **voice cascade → Q4** (round 2 flow) | **88%** | **81%** | 97% |
| voice cascade → Q1 | 88% | 80% | 97% |
| voice line clip only → Q1 | 87% | 79% | 97% |
| Q1 (round 1) | 83% | 71% | 97% |
| stored transcript (B0) | 69% | 47% | 94% |

* **Who decided and how often it was right (labelled lines):** the line's own clip 92/96 (96 %), the padded clip 10/13 (77 %), the diarization turn 6/10 (60 %),
  Qwen Q4 31/38 (82 %). Share of all 1,092 lines: own clip 70 %, padded 4.5 %, turn 2.7 %, Qwen 22 %.
* **The turn is not good enough to be final:** sending those lines to Qwen instead gives 140/157 (89 %). The flow to port uses the own clip and the padded clip only.
* **Q4 is not more accurate than Q1 on the same lines (31 vs 30 of 38) but costs less** (it asks ~23 % of the lines): it is the one to port.
* **Confidence that production can compute** (no Q1 there): high = own-clip voice 96 %; medium = padded-clip voice, or Qwen where the line's weak voice
  top-1 agrees (32/39, 82 %); low = Qwen otherwise (16/22, 73 %). The "agrees with round-1 Qwen" rule of the report is not available in the app.
* **Thresholds:** top ≥ 0.45 and gap ≥ 0.06 hold up; the sweep shows 0.50 is more precise (96 %) but no better overall. They were picked on these labels.
* **Music map (LFM3, 900 windows in 91 s):** only 1 of 677 dialogue windows was called music. It grew Ep2's ED by 13 s (dropping the lyric line L435) and an OP
  edge by 2 s (dropping Ep1 L8 「うん。」, probably real). Ep3's ED already started where the singing does; Ep1 has no ED. Not worth porting yet.
* **Names:** verified spellings went from almost none to dozens per episode (梓川 0 → 19, 咲太 0 → 40, 麻衣 0 → 73 over the three episodes).
  Aliases: 84 changes, all right. Qwen pass: 62 fixes, 48 right, 12 wrong, 2 unsure. The wrong ones: a non-name word turned into a name (テスラ, 恩借, クン,
  南作た, ゴフマ); another proper noun forced into a cast name (ミネガハラ高校 = 峰ヶ原高校 → 牧之原高校, three times); a given name expanded to the full name
  (まえさん → 桜島麻衣さん); a neighbouring character swallowed (マイサなら → 麻衣なら).
* Not checked: the round-2 spot check was not labelled, so no labels exist that the thresholds were not chosen on. `score_app.py` scores the app's
  transcript against the same labels after the port.

## Measured before round 1 (CPU, stored data, 2026-11-19)

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

## Porting

`IMPLEMENTATION.md` is the plan for the app. It replaces the porting plan written before round 1 (a separate dialogue table and a chain of five job kinds):
the measured flow fits the existing `transcript_segments` table with one row per line and two job kinds.

## Risks and caveats

* 169 labels from three episodes of one series, and the voice thresholds were chosen on them: expect a few points less on new material. Each character has one voice sample.
* Stored-transcript B0 and the app's cluster rule use the very samples the anchors come from (circular), so they are shown as such and never ranked as leave-one-out.
* Thresholds of the voice decision (line: 0.35 / margin 0.06; cluster by range: 0.45 / 0.05; app rule 0.7 / 0.08) are not tuned on held-out data. The report's `V3raw` (top-1 without a threshold) shows the ceiling of the voice evidence.
* Qwen3.6 on a line is only as good as its candidates: a character without a voice sample can be named from dialogue and context only.
* Qwen3-ASR with the names as context may echo names that were not said; A2/A3 are scored for that through the verified-name hits and the CER between readings, not against a truth.
* Anything the run skipped for time is listed in the report ("Changed, skipped and caveats").
* A better model would be needed for vocal separation (clean dialogue under music) and for a real audio-language model (Qwen3-Omni); neither is installed and nothing was downloaded.
