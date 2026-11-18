# Qwen 3.6 vs Qwen 3.5 first-frame benchmark (throwaway)

A one-off experiment outside the app: it re-runs the app's **first-frame analysis** on real
first frames from Episode 2 with two vision models and reports the inference time and the
answer of every call side by side. It answers one question: on this machine, how does
**Qwen 3.6-35B-A3B** (text model we already have, given a vision projector) compare with the
**Qwen 3.5-27B** the app uses today for first frames?

Nothing here changes the app: no database, no API, no worker logic. The app's own first-frame
path (`backend/app/single_cut_pipeline.py` → `frame_model.py` → `visual_contracts.py`) is left
untouched; the benchmark simply reuses its prompt, grammar and frame extraction.

## How it works

`benchmark.py` runs **inside the inference container** (it needs `llama-server`, `/models` and
the app code), and builds every input the exact way the app does, then asks the model:

1. `media.extract_cut_frame` pulls the first frame of the cut at its real display time
   (`cuts.start_time`, the same value `get_frame_time` uses).
2. It reuses the cut's **stored `marked_people` boxes** (what the user drew, what the app sent)
   and the episode's subtitle strip, so the model sees the same handles and the same
   `SUBTITLE_RULE` note.
3. `visual_contracts.frame_message` builds the identical `system` + `user` messages and the same
   JSON-schema grammar.
4. It starts **one server at a time** (`parallel=1`, the app's 8192-token slot) and reuses it
   across all cuts. Both models get the same message, `temperature=0`, JSON-schema grammar with
   the app's `json_object` fallback.

A vision-capable **Qwen 3.5-27B** is the first-frame model we already have (`QWEN35_MODEL_PATH`
+ `QWEN35_PROJECTOR_PATH`). **Qwen 3.6-35B-A3B** is the app's text model; a vision projector
(`mmproj-Qwen3.6-35B-A3B-BF16.gguf`, the official multimodal projector) makes it vision-capable
in `llama-server`. **Nothing else is downloaded** and **no app file is edited** — `llama-server`
just loads the extra file.

Each call records the **full request round-trip** (encode + inference + response) in
`latency_seconds`, plus the server's prompt/completion tokens, so time and output are comparable
between models. The app batches first frames at `FRAME_PARALLEL=4`, so absolute times here differ
from a queued run; the comparison between the two models on identical inputs is what matters.

## Files

- `benchmark.py` — the experiment. Picks N cuts, builds the app's frame message, calls each model,
  writes `results.json` (and `frames/cut-XXXX.jpg` with `--keep-frames`).
- `report.py` — renders `results.json` into `report.html` (summary stats + side-by-side answers).
  Standalone, stdlib only, runs on the host.
- `run.sh` — `start` / `status` / `stop` / `report` / `dry`. Like the other experiments, `start`
  stops the inference worker, waits for VRAM to drain (one GPU process at a time), launches the
  benchmark container, and arms a watcher that starts the worker again when it ends.
- `download-qwen36-mmproj.sh` — fetches only the Qwen 3.6 vision projector into
  `.models/qwen3.6-35b/` (resumable, sha256-verified).
- `results.json`, `frames/`, `report.html` — generated (gitignored).

## Run

```bash
./qwen_experiment/download-qwen36-mmproj.sh          # one time
./qwen_experiment/run.sh dry --count 5               # rebuild inputs, no model, worker untouched
./qwen_experiment/run.sh start --count 30            # GPU run (worker stops, comes back after)
./qwen_experiment/run.sh status                      # progress
./qwen_experiment/run.sh report                      # build report.html (host, after it finishes)
./qwen_experiment/run.sh stop                        # stop early; worker comes back
```

Open `report.html` in a browser. It shows per-model median/mean/p95 inference time, tokens and
throughput, and every first frame with the two models' answers side by side (the faster call of
each pair is green).

## Prerequisite

Only Qwen 3.6 needs an extra file: `scripts/download-qwen36.sh` fetches the text-only GGUF
without a vision projector, so this experiment pulls the matching projector itself. `run.sh start`
refuses to run if either model, the projector, or `llama-server` is missing.

## Caveats

- **Qwen 3.6 here is the text model with a vision projector**, not a purpose-trained vision
  model; treat its first-frame answers as "the text model looking at a frame".
- Qwen 3.6 may answer in plain JSON when it struggles with the grammar, and its field set can
  differ from Qwen 3.5's; both are kept in the report.
- These are the models' answers, not verified ground truth: they can disagree and both can be
  wrong, especially on identities, `no_characters` cuts, tiny boxes or cut edges.
