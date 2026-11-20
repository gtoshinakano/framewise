# Audio experiment: clean dialogue lines with the right speaker

Question: what is the best way to turn an episode's audio into clean dialogue LINES with the right SPEAKER, using only models already
downloaded (Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B, pyannote community-1, Qwen3.6-35B-A3B, optionally LFM3-Omni-2B)? The user does not want to
review audio by hand, so approaches where Qwen and the local signals do the work and few lines stay uncertain are favoured.

Nothing in the app changes: no migration, no API, no edit under `backend/app`, `frontend/src`, compose files or images. The experiment reads a
**copy** of the database (sqlite backup API through the API container, source opened `mode=ro`), reads the episodes' `audio.wav` through a
read-only mount, and imports one pure helper of the app (`audio_matching.restore_aligned_text`). Nothing is downloaded (`HF_HUB_OFFLINE=1`); no network model is used.

## Run it

```bash
audio_experiment/run.sh dry-run              # whole pipeline, fake models, CPU, 3.5-minute excerpt (about 1 min); report at <base>/dry-run/report.html
audio_experiment/tests/run.sh                # unit tests (CPU, throwaway container)
audio_experiment/run.sh start                # real run: needs an idle queue; stops BOTH workers for the GPU phases; about 4-5 h, hard stop after 6.9 h
audio_experiment/run.sh status
audio_experiment/run.sh stop                 # stops the run; both workers are started again
audio_experiment/run.sh start --resume audio-20261119-2100     # same id continues: finished checkpoints are skipped
audio_experiment/run.sh report [ID]          # rebuild report.html / report.md / spotcheck.html (+ the verdict block of FINDINGS.md) on the host
audio_experiment/run.sh serve                # http.server on <host>:8095 for the reports (the app on :8080 only shows them after an image rebuild)
# after labelling some lines in spotcheck.html (Export JSON):
audio_experiment/run.sh report ID labels.json    # same as: experiment.py rescore ID --labels labels.json
```

Report: `http://192.168.0.6:8095/runs/<id>/report.html` (all runs: `.../runs/index.html`), `report.md` beside it (same text for agents),
`spotcheck.html` for real labels, `results.json` with every number, `clips/` with the audio of the sampled lines.

### What `start` does

`start` checks the queue (jobs, assistant turns, chat gateway all idle), copies the database, then launches a **detached driver** (survives your session) and a
**detached watcher** that starts both workers again however the driver ends. The driver runs the phases one after another, each in its own container
(never two on the GPU), with a cap per phase and a deadline for the whole run:

| phase | container | what | cap |
|---|---|---|---|
| `prepare` | CPU, workers still up | database copy → `prep.json`; OP/ED search across the 7 episodes → `oped.json` | 40 min |
| `glossary` | Qwen3.6 | proposed Japanese spellings of the characters that have none verified (unverified, marked) | 25 min |
| `audio` | pyannote + Qwen3-ASR | diarization settings, ASR variants, lines, per-line embeddings, the app's own pipeline re-run on Episode 3 (time baseline) | 130 min |
| `qwen` | Qwen3.6 | Q1, R1, Q2, Q3, Q1s (see below) | 240 min |
| `second` | Qwen3-ASR | local second opinion for the weak lines | 40 min |
| `lfm3` | LFM3-Omni | does the bundled llama-server accept audio with this projector? if yes: speech/singing/music per 10 s | 30 min |
| `final` | CPU, workers back up | results, report, clips | 20 min |

Stages inside a phase are ordered by value and check the deadline; whatever is skipped is listed in the report under "Changed, skipped and caveats".

## Round 2 (2026-11-20)

Round 2 builds on a finished round-1 run (its checkpoints are read, never changed) and keeps its line ids, so labels made on either run apply.
It tests the four changes the round-1 labels pointed to (see FINDINGS.md):

* **Voice first.** A line whose own voice match is clear (top ≥ 0.45, gap ≥ 0.06, every reference within 60 s of the clip left out) is final; Qwen never answers it.
* **Short lines get a usable clip.** A line the voice did not settle is embedded again from its clip grown into the silence around it (≤ 1 s each side, never into a neighbour) and from its diarization turn (≤ 3 s past the line).
* **Q4:** Qwen answers only the open lines, but reads the whole conversation with the decided speakers in it (round 1's R1 sent the weak lines alone).
* **Names:** kana readings of the verified names are spelled deterministically (`names.py`), then a Qwen pass that is given every spelling and reading may only swap a misheard name; any other change is rejected.
* **OP/ED edges:** LFM3 labels every 5 s; a fingerprint range grows over the adjacent singing (music only between singing), and ≥ 20 s of singing becomes a range.

Confidence comes from agreement (voice decided and round-1 Qwen agrees → high), never from Qwen's own field.

```bash
audio_experiment/run.sh round2-dry                       # fakes, CPU, on top of the dry run (run.sh dry-run first)
audio_experiment/run.sh round2 [--base audio-20261119-2055]   # about 40 min; phases music (LFM3), voice (pyannote), qwen2 (Qwen3.6), final2 (CPU)
audio_experiment/run.sh report2 [ID] [labels-export.json ...]  # rebuild on the host with labels/*.json plus exports of the round-2 spot check
```

Labels: `labels/round1.json` holds the 2026-11-20 spot checks as time spans (`typical`: the dry-run page, `hard`: round 1's disagreement picks). The round-2
spot check lists lines that round 2 changed or left low-confidence; its export uses the base run's line ids and is scored as set `round2`, the only labels
the thresholds were not chosen on. Files: `round2.py` (stages), `round2_rules.py` (pure rules), `round2_analysis.py` (results2.json and the report), `names.py`.

## Variants

* **B0** the stored transcript merged with the app's rule (`merged_transcript`, gap 0.6 s); its job duration from `jobs`. **A0** (Episode 3 only): the current pipeline re-run (D0 + 25 s windows) for a fair time baseline.
* **OP/ED** (CPU, first): repeated-audio search (Haitsma-Kalker-style sub-fingerprints per 0.1 s, half-word votes on offsets, verification along a drifting diagonal) across all 7 episodes;
  1.b signals inside one episode (lyric-looking ASR, loud stretches with little diarized speech); 1.c LFM3 only if the audio probe works. The ranges are masked for every later stage.
* **A1** windows cut at pauses (speech regions from the masked diarization, RMS minimum near the 25 s cap, never across a mask), speaker per sentence (majority of the word overlap; split only at sentence-final punctuation or a 0.3 s pause where the speaker changes), punctuation attached to the previous word, lines keep `words[]`.
  **A2** = A1 + character names as ASR context (verified spellings + Qwen's proposals marked sure). **A3** = A2 + language forced to Japanese.
* **D0** pyannote on the whole audio (the app's way), **D1** on the OP/ED-masked audio (base for everything), **D2** thresholds 0.45 / 0.75 / 0.9, speaker hints 3-8, and max 6 with 0.75.
  **D3** per-line embeddings with the pipeline's own embedding model against every voice-sample RANGE (leave-one-out) and against the cluster centroids.
* **Q1** Qwen3.6 per line (windows of 40 lines, 4 overlap; json_schema grammar, thinking on, validator sentences back): text, times, diarization label, top-3 voice candidates, characters in the cut's first frame, the loose Portuguese subtitle, neighbours.
  **Q2** = Q1 without the subtitles. **Q3** = Q1 + speech profiles (pronouns, endings, address terms) counted from confident lines of the OTHER episodes. **Q1s** = Q1 with windows shifted by half a window (stability).
* **R1** episode-level voice resolution: clusters linked to characters from their own lines' evidence, merged clusters reported, only the weak lines re-decided by Qwen. **ENROL** per-character centroids from the confident lines of the other episodes (Ep3 learns from 1+2).
* **Second opinion** for the weak lines: Qwen3-ASR again on that sentence's clip with the neighbours as context, CER against the first reading, the tight clip embedded and re-checked.

## Scoring without ground truth

Anchors: lines mostly inside a **voice-sample range** (36 ranges of 10 samples, Episodes 1-2) and the **30 hand/assistant-corrected rows of Episode 2**. For every line all reference
ranges overlapping it are left out (leave-one-out), so the reference audio never labels itself; characters with a single range have no reference for those lines (the report shows that cost honestly).
The stored transcript B0 and the app's cluster rule (`diar:*:app`) cannot be made leave-one-out (their links are made from the same samples) and are marked `refs left out: no`.
Also: addressed-by-name contradictions (a line saying "サクタさん" attributed to Sakuta), first-person pronoun entropy per character, agreement between independent methods, Q1 stability under shifted windows,
speaker on screen (reported, never scored as error), voices per setting, % unknown / low confidence; segmentation (edges vs the subtitle cue edges ±0.35 s, short and unfinished lines, lines cut by a window edge, speaker changes inside a sentence);
text (garbage rate, verified-name hits, CER between independent readings, every Qwen text fix listed). No weighted composite: methods are ranked by anchor accuracy, then contradictions, then agreement and stability.

**Portuguese subtitles** are a loose translation: they are only used for sentence boundaries/timing, speaker-turn hints and rough meaning, and never to change the Japanese.
**The "Minute by minute breakdown" context item of Episode 1 is not used.**

## Files

| File | Role |
|---|---|
| `run.sh` | launcher: start/status/stop/dry-run/report/serve; detached driver and watcher |
| `experiment.py` | CLI: `phase <name>`, `all --dry`, `rescore` |
| `stages.py` | the stages (prepare, glossary, audio, qwen, second, lfm3) with per-window/per-episode checkpoints |
| `backends.py`, `fake_backend.py` | real models / fakes with the same interface |
| `qwen.py` | llama-server launcher, attribution prompt, grammar, validator, window runner, glossary, speech profiles |
| `oped.py` | fingerprints, pair matching, consolidation, 1.b helpers |
| `lines.py`, `scoring.py`, `voice_tables.py`, `glossary.py` | pure helpers (stdlib): segmentation, scores, leave-one-out voice tables, names |
| `voices.py` | numpy: reference embeddings, cluster resolution, enrolment |
| `analysis.py`, `oped_report.py`, `oped_signals.py`, `report.py`, `findings_block.py` | results.json and the report (stdlib, run on the host too) |
| `tests/` | unit tests (`tests/run.sh`) |

Run folder (`frontend/public/audio-experiment/runs/<id>/`, gitignored): `prep.json`, `oped.json`, `glossary.json`, `ep{N}/{diar,asr,lines,qwen}/…`, `emb/`, `timings.json`, `notes.json`, `progress.json`, `log.txt`, `driver.log`,
`results.json`, `report.*`, `spotcheck.html`, `clips/`. The database copy lives in `audio_experiment/work/<id>/` (private, never under the served folder).

## While it runs

The inference **and** audio workers are stopped (a second GPU process beside llama-server once crashed this host's kernel); queued jobs wait. Both come back by themselves when the run ends or `stop` is used.
