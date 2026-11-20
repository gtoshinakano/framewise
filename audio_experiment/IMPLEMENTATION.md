# Porting the audio experiment into the app

For the agent that implements the new transcription flow. The experiment is finished; this file states what to build, what to copy from, and how
to check it. **Do not re-run or extend the experiment, and do not read the rest of `audio_experiment/` beyond what this file names.**
Why each decision was made, with the numbers: `FINDINGS.md`, sections "Verdict", "Round 1 scored against labels" and "Round 2". AGENTS.md applies as usual
(queue discipline, `JobInputs.publish`, revisions, no approval status, rebuild and restart after the change).

## What to read

| Read | For |
|---|---|
| this file | the flow, the data, the jobs, the checks |
| `lines.py`: `merge_intervals`, `subtract_intervals`, `overlap`, `inside_any`, `quietest_cut`, `plan_windows`, `TurnIndex`, `split_sentences`, `split_by_speaker`, `build_lines`, `join_text`, `ends_sentence` | ASR windows cut at pauses, sentence lines with one speaker |
| `oped.py`: `fingerprint`, `_halves`, `match_pair` and what it calls, `consolidate`, `label_ranges` | OP/ED search across the project's episodes |
| `round2_rules.py`: `pad_span`, `cascade`, `decide`, `split_by_mask` (ignore `guarded_scores`/`REF_GUARD`, `borrow`, `extend_ranges`, `confidence`, labels helpers) | the voice decision |
| `voice_tables.py`: `scores`, `decide`, `ranked` (ignore `overlapping_ranges` and the cluster helpers) | per-character score from per-range references |
| `voices.py`: `build_references` only | references per sample range |
| `backends.py`: `RealModels.diarize`, `embed`, `transcribe` | the model calls that produced the numbers (in-memory waveform, `pipeline._embedding`, aligner checks) |
| `round2.py`: `SYSTEM_OPEN`, `open_items`, `attribute_open`, `call_names` | the Qwen speaker step and the name pass |
| `qwen.py`: `answer_schema`, `validate`, `make_windows`, `user_message`, `call_window`, `parse_json` | the speaker grammar, validator and retry |
| `names.py` | name aliases, name-pass prompt, schema and validator (the validator needs tightening, below) |
| tests: `tests/test_lines.py`, `tests/test_oped.py`, `tests/test_round2.py` (Names, Voice, Music.split_by_mask), `tests/test_qwen.py` | port the ones for code you port |
| fixtures: `labels/round1.json`, `labels/name_fixes.json`, `score_app.py` | acceptance checks |

**Ignore:** `report*.py`, `analysis.py`, `round2_analysis.py`, `scoring.py`, `glossary.py` proposals, `voices.py` enrolment and cluster resolution, `stages.py`
(apart from the call order you can read in `stage_audio`), `fake_backend.py`, `run.sh`, `experiment.py`, the variants Q1s/Q2/Q3/R1/ENROL/A2/A3/A0,
the diarization settings other than the default, the second opinion, the LFM3 music map and the `REF_GUARD` ±60 s exclusion (an evaluation device: production uses every reference).

## The flow

Per episode, in this order:

1. **OP/ED ranges (CPU).** Fingerprint every episode of the project that has `audio.wav` (cache the fingerprint per audio file identity under
   `DATA_DIR/cache/fingerprints/`, safe to delete like the frame cache), match this episode against the others, keep ranges seen in at least 2 other
   episodes (`consolidate(min_support=2)`), label them with `label_ranges`. A project with fewer than 3 episodes gets no automatic ranges. Save them as an
   episode setting (below); the user can add, move or delete ranges. Seconds per project, no GPU.
2. **Diarization on masked audio.** Load `audio.wav`, zero the samples inside the ranges, pass the waveform in memory (`backends.RealModels.diarize`; the experiment
   instantiated `{'segmentation': {'min_duration_off': 0.}, 'clustering': {'threshold': .6, 'Fa': .07, 'Fb': .8}}`, community-1's defaults). Keep the
   `cudnn.flags(enabled=False)` guard of the app.
3. **ASR in pause-cut windows.** `plan_windows(speech_regions, rms, hop, mask)` with the diarization turns dilated by 0.2 s minus the mask as regions
   (`stages.speech_regions`, `oped.rms_levels`): windows of at most 25 s cut at the quietest point, never across a range. Keep the app's alignment checks and
   the split-in-half retry (`transcribe_window`). Speech that diarization missed is not transcribed; that is what the measured numbers include.
4. **Lines.** `build_lines(tokens, TurnIndex(turns))` per window: a sentence ends at final punctuation or a 1.2 s pause; it is split only where the speaker
   changes at a pause of at least 0.3 s; the line's speaker is the majority of its words' overlap. Drop lines with at least half of their span inside a range.
5. **Voice per line** (still in the audio job, pyannote loaded):
   - references: one 256-d embedding per voice-sample **range** of at least 1 s, made with the same call as the lines (`pipeline._embedding` on the clip,
     `backends.RealModels.embed`); a sample whose ranges are all shorter keeps one from its longest range; an uploaded sample is embedded from its saved wav.
     The stored `embedding_json` (diarization centroid of the joined sample) is NOT comparable and is not used here.
   - a line's score per character = mean of its top-2 cosines over that character's references (`voice_tables.scores`, nothing excluded).
   - **own clip** (line span, at least 0.6 s): decide at top ≥ **0.45** and gap ≥ **0.06** → final, `speaker_source='voice'`, `confidence='high'`.
   - else **padded clip** (`pad_span`: grown into the silence before and after, at most 1.0 s each side, never into a neighbouring line): same rule → final,
     `speaker_source='voice_padded'`, `confidence='medium'`.
   - else keep the best top-3 scores as evidence for Qwen. **Do not port the diarization-turn source**: it was right 6 of 10 times, worse than Qwen.
6. **Name aliases** (deterministic, `names.apply_aliases` with the project's glossary instead of `glossary.PARTS`): the katakana reading of a cast name standing
   alone becomes its spelling; a two-kana name only next to its family name or before an honorific. All 84 changes in round 2 were right.
7. **Speakers by Qwen** (text lane, separate job): only the open lines are asked, the whole conversation is shown with `decided` speakers, windows of 40 lines
   overlapping by 4, an answer kept from the window where the line sits farther from an edge (`attribute_open`). Use `SYSTEM_OPEN` as written, but drop
   `text_fix`/`fix_reason` from the schema and rule 6 (names have their own pass). Inputs per line: id, times, text, diarization label, top-3 voice evidence,
   the characters labelled in the first frame of the cut it starts in, and the episode's burned-in subtitle text overlapping it (`pt`; +14 points on hard
   lines in round 1). `confidence='medium'` when the line's top-1 voice evidence names the same character, else `'low'`; `speaker_source='qwen'`.
   `unknown`/`other` leave `character_context_id` empty.
8. **Name pass by Qwen** (same text job, after step 7): every line's text with the cast spellings and readings and the project's other proper nouns, windows
   of 60 lines, `names.SYSTEM` + a list of the other names ("these are not characters; never change one into a character name"), only fixes that pass the
   tightened validator are written.

Measured on the labels (FINDINGS): own clip 96 % right on 70 % of the lines, padded clip 77 % on 4.5 %, Qwen 82 % on the rest; 88–89 % overall against 69 %
for today's transcript. Confidence buckets: high 96 %, medium 82 %, low 73 %. Time per 25-minute episode about 8.5 min against 3.2 min.

## Data

* **`transcript_segments`: one row per line for new runs.** Add nullable columns `words_json` (`[[start, end, text], …]` from the aligner),
  `speaker_source` (`voice`, `voice_padded`, `qwen`, `manual`), `confidence` (`high`, `medium`, `low`) and `evidence_json` (top-3 voice scores and which clip,
  the Qwen reason). Old word rows stay as they are until the episode is transcribed again. **Confidence is model evidence, not an approval status**: nothing
  filters, gates or asks the user to approve by it, no prompt receives it as a status; a hand edit of the speaker sets `speaker_source='manual'` and clears
  `confidence`; a hand edit of the text clears `words_json` (readers fall back to spreading the text over the span).
* **Word parts:** several readers need words inside a line (a sequence edge or a hard cut that falls inside a line). Add one helper (for example in
  `assistant/transcript_edit.py` beside `merged_transcript`) that returns a row's parts from `words_json`, else the row itself, and use it in
  `sequence_store.transcript_lines` (`parts`) and wherever fragments are split by time. `merged_transcript` keeps working (it merges consecutive lines of one
  speaker) and the assistant's transcript edits keep working at line level. Audit and run the tests of every reader:
  `sequence_store`, `sequence_timeline.speech_lines`, `sequence_compile` (fragment grouping, `source_hash`), `performance_beats.speech_lines`,
  `consistency_evidence.spoken_lines`, `subtitle_scan.choose_probe_times`, `prompt_context`, `analysis_context`, `external_api`, `assistant/transcript_edit.py`,
  `assistant/tools.py`, `main.assign_voice`, `storage.py`, the Audio tab and `CutAudioPreview`.
* **`character_voice_samples.range_embeddings_json`** (`[{start, end, embedding}]`), filled by the `voice_sample` job for new samples. Existing samples: the
  transcribe job computes missing ones in memory before scoring (40 references took 15 s) and does not write them back; a later `voice_sample` re-run stores them.
* **Episode audio ranges:** a table such as `episode_audio_ranges (id, episode_id, start_time, end_time, kind 'OP'|'ED'|'repeat'|'manual', source
  'fingerprint'|'user', matched_episodes_json)`. A setting, not a revisioned result: user rows win over fingerprint rows; a new search replaces only
  `source='fingerprint'` rows. Shown on the timeline; changing them marks the transcript stale.
* **Name glossary (per project):** rows `(spelling, reading, character_context_id nullable)`. A row with a character is a cast name part (family or given name,
  give the part: `family`/`given`); a row without one is another proper noun (峰ヶ原高校, places). Entered by the user (Context library or Characters menu);
  never proposed by a model (round 1's model proposals invented a name). **Seed this project's glossary once** (decided by the user, 2026-11-20) from
  `glossary.PARTS` (the user's verified spellings with their standard readings, matched to characters by title) plus the other name 峰ヶ原高校
  (みねがはらこうこう), with a one-off idempotent step; no other project is seeded. Characters without glossary rows are skipped by steps 6 and 8.

## Jobs

* **`transcribe` (audio lane, existing kind):** steps 1–6 for the full-episode path; publishes line rows like today (tombstones for replaced model rows,
  hand-edited rows protected) and, **in the same transaction**, queues `speaker_attribution` with `depends_on` the transcribe job when any line is open or a
  glossary exists. Record in `jobs.input_json`/the rows: fingerprint, model ids, thresholds (`0.45/0.06`, pad 1.0 s, min clip 0.6 s), the ranges used.
  `transcribe_range` uses the same steps 2–6 on its interval and queues `speaker_attribution` with `payload.range`.
* **`speaker_attribution` (new kind, text lane):** register it in `job_kinds.KINDS` (lane `text`, handler, resources `{'text'}`, scope reading
  `transcript`, `subtitles`, `detections`, `context` and writing `transcript`, keys per episode and range, priority `batch`, gather/inputs/invalidates like the
  other text kinds), which also puts it in `worker.TEXT_PIPELINES` and the remote routing. Gather at start: current lines of the episode (or range), their
  evidence, characters, first-frame roster per cut, subtitle lines, glossary. Publish with `JobInputs.publish`: write a model revision only for lines whose
  current row is the one gathered and whose `speaker_source` is not `manual`; a changed input requeues it like other jobs. Its invalidation marks the same
  dependants as a transcript change (`structured_results.invalidate_transcript`).
* **Subtitles finishing later:** a completed subtitle run marks the episode's speaker attribution stale (shown, with a button to run it again); it does not
  queue it by itself (AGENTS: edits leave derivatives stale until the user asks).

## Name-pass validator (tighten before porting)

`names.name_only_change` lets any short span become a name. `labels/name_fixes.json` holds round 2's 62 Qwen fixes with a verdict: the new validator must
accept all 48 `ok` and reject all 12 `wrong` (2 `unsure` are not tested). The wrong ones fall into four kinds; reject:
1. a replaced span written only in kana whose reading is far from the chosen name's reading (テスラ → 桜島麻衣, クン → 国見, ゴフマ → 佑真);
2. a span that is, or reads like, a glossary *other name* (ミネガハラ高校 is 峰ヶ原高校), plus the other-names list in the prompt;
3. a full name (family + given) replacing a span that holds only a given name or less (まえさん → 桜島麻衣さん; the right fix was 麻衣さん);
4. a replacement that swallows a neighbouring kana that belongs to the sentence (マイサなら → 麻衣なら; ガサクタが → 咲太が).
Kanji mishearings (安沢川 → 梓川, 桜島舞 → 桜島麻衣, 佐久田 → 咲太) are the fixes worth keeping. Readings are in the glossary; compare kana with a plain edit
distance after converting katakana to hiragana. Port the `Names` tests of `tests/test_round2.py` and add one that replays the fixture.

## UI

Audio tab, Lines view: one row per line with the speaker select (exists), a small chip on `medium` and `low` lines ("check"), a filter "Lines to check", and
the episode's OP/ED ranges as a lane on the timeline with add/edit/delete. The Unknown voices block keeps working on `speaker_label`. No review gate, no
approval state.

## Done means

1. Ported unit tests pass, plus the name fixture test.
2. Rebuild and recreate the API, the audio worker and the inference worker (AGENTS: check no job is running first), re-transcribe Episodes 1–3 of the project
   `d66e89d5-…` (the one the labels come from) after seeding its glossary, let `speaker_attribution` finish, then run
   `python3 audio_experiment/score_app.py`. Today it prints typical 94 %, hard 48 %, all 69 %. Expect about typical ≥ 95 %, hard ≥ 78 %, all ≥ 86 %
   (round 2: 97 / 81 / 88, with the turn source removed 89 % overall). It also prints the verified spellings found: 梓川, 咲太 and 麻衣 should each appear
   dozens of times.
3. No lines inside the OP ranges (Ep1 0:52–2:22, Ep2 1:02–2:30, Ep3 1:10–2:38) or the ED ranges (Ep2 from 24:08, Ep3 from 23:40).
   Known limit: Ep2's ED range from fingerprints starts at 24:08 while the singing starts at about 23:44, so the lyric line 「語れない、眠れない…」 at 23:44
   stays until the user widens that range (round 2's LFM3 map caught it; it is deferred, not needed for this check).
4. Time per episode around 8–9 min on this host, model loads included in the job log.
5. AGENTS.md code map updated (audio pipeline, new kind, new columns and tables), `integration-guide.md` and `external_api.py` models updated if `/api/v1`
   returns the new fields.
