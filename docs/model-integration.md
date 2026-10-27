# Model integration plan

The requested [WeVisOne-VL paper](https://huggingface.co/papers/2610.31329) and [official implementation](https://github.com/Tencent/WeVisOne-VL) describe image-level detection, referring localization, counting, OCR, segmentation, and matting. The official repository currently shows 2B and 4B variants and a Python 3.10 installation example. Its documented device choices include CUDA, NPU, and CPU; ROCm is not explicitly documented. No model package or weights are installed in Framewise yet.

## Intended place in the pipeline

1. Keep import, proxy creation, audio, waveform, chunks, and hard cuts in the small CPU app.
2. Hard-cut detection now saves first and last preview frames per cut and records their sample timestamps in the cut API. Use these as initial visual inputs, retain the original as the quality reference, and sample additional frames per chunk as needed.
3. Run WeVisOne-VL on selected frames for object, character candidate, OCR, mask, and other spatial evidence. Save structured boxes/masks as separate artifacts under that episode's data directory, with an index in SQLite. Use the `/models` volume for cached weights.
4. Use a separate temporal analysis method for camera movement, action, shot description, and continuity across frames. The documented WeVisOne-VL tasks alone do not establish motion or audio meaning.
5. Assemble the cut's input with `/api/cuts/{id}/analysis-context`: project synopsis, applicable character/world/scenario notes, overlapping chunk notes, approved transcript overlap, and previous/next cut context. Freeze that input into `analysis_results.context_json` for reproducible retries.
6. Append model results to `analysis_results` with `origin='model'`, `model_id`, `job_id`, prompt, settings, and context snapshot. Do not overwrite human revisions or approved results. Store longer structured predictions or binary masks in dedicated artifact tables/files when an inference adapter is implemented.

Redetection archives unmatched cuts instead of deleting them, so their manual notes and model provenance remain available. A matching boundary can reactivate the same cut ID and its results.

## Device and deployment boundary

Keep a separate inference worker image for the selected ROCm or CUDA PyTorch runtime. Mount the same `/data` and `/models` volumes and pass `/dev/kfd` and `/dev/dri` for ROCm or the NVIDIA runtime for CUDA. Validate the chosen WeVisOne-VL version with the ROCm stack before enabling model jobs; the upstream quick start documents CUDA/CPU but does not establish ROCm compatibility. Do not install a CUDA wheel in the base app container.

The media worker now claims durable SQLite jobs with a renewable lease, heartbeat, attempt count, and queued settings payload. A separate inference worker should claim only its own pipeline kinds and preserve the same lease ownership checks before writing results. Keep media preparation idempotent so a restarted worker can resume safely. A model adapter should load weights only when the chosen pipeline is run and only if they are absent from `/models`.

## Later audio and prompt work

`transcript_segments` stores timed text, speaker labels, optional links to character context records, approval state, model/job provenance, and revision links. Human edits append a new revision; deletes append a tombstone, so model output can remain inspectable under history. A transcription worker can populate it; voice clustering can add stable speaker identities after its schema is settled. The context endpoint selects current approved transcript segments overlapping each cut, so audiovisual matching can start from aligned evidence.

Prompt generation should create separate `generation_prompt` results for character sheets, worlds, objects, first/last animation frames, and audio. Preserve each prompt version, the evidence used, and user edits. Output from generation services belongs in new artifacts and should not replace reference media.
