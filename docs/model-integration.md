# Model integration plan

The requested [WeVisOne-VL paper](https://huggingface.co/papers/2610.31329) and [official implementation](https://github.com/Tencent/WeVisOne-VL) describe image-level detection, referring localization, counting, OCR, segmentation, and matting. The official repository currently shows 2B and 4B variants and a Python 3.10 installation example. Its documented device choices include CUDA, NPU, and CPU; ROCm is not explicitly documented. Framewise has a separate pinned inference worker and download scripts. A host GPU test loaded WeVisOne and confirmed that free-form requests returned bounding-box output, so Qwen3-VL-2B-Instruct supplies natural-language captions while WeVisOne supplies configurable object detections. The container deployment still requires validation.

## Intended place in the pipeline

1. Keep import, proxy creation, audio, waveform, chunks, and hard cuts in the small CPU app.
2. Hard-cut detection now saves first and last preview frames per cut and records their sample timestamps in the cut API. Use these as initial visual inputs, retain the original as the quality reference, and sample additional frames per chunk as needed.
3. Run WeVisOne-VL detection and OCR and Qwen3-VL captioning on selected original-video frames. Store descriptions and parsed detection boxes with timestamps in SQLite. Masks are not generated. Both pinned checkpoints are mounted from persistent host directories.
4. A text model assesses individual visual and OCR claims with qualitative confidence before synthesizing a cut summary; its motion, action, and continuity suggestions are estimates, not measured temporal analysis. The documented WeVisOne-VL tasks alone do not establish motion or audio meaning.
5. Assemble the cut's input with `/api/cuts/{id}/analysis-context`: project synopsis, applicable character/world/scenario notes, overlapping chunk notes, current transcript overlap, and previous/next cut context. Freeze that input into `analysis_results.context_json` for reproducible retries.
6. Append model results to `analysis_results` with `origin='model'`, `model_id`, `job_id`, prompt, settings, and context snapshot. Do not overwrite existing results or human revisions. Store longer structured predictions or binary masks in dedicated artifact tables/files when an inference adapter is implemented.

Redetection archives unmatched cuts instead of deleting them, so their manual notes and model provenance remain available. A matching boundary can reactivate the same cut ID and its results.

## Device and deployment boundary

Keep a separate inference worker image for the selected ROCm or CUDA PyTorch runtime. Mount the same `/data` and `/models` volumes and pass `/dev/kfd` and `/dev/dri` for ROCm or the NVIDIA runtime for CUDA. Validate the chosen WeVisOne-VL version with the ROCm stack before enabling model jobs; the upstream quick start documents CUDA/CPU but does not establish ROCm compatibility. Do not install a CUDA wheel in the base app container.

The media worker now claims durable SQLite jobs with a renewable lease, heartbeat, attempt count, and queued settings payload. A separate inference worker should claim only its own pipeline kinds and preserve the same lease ownership checks before writing results. Keep media preparation idempotent so a restarted worker can resume safely. The adapter loads the mounted local weights lazily on the first job. The separate download script pins and verifies the cached files; the worker does not download model weights.

## Later audio and prompt work

`transcript_segments` stores timed text, speaker labels, optional links to character context records, a deletion marker, model/job provenance, and revision links. Human edits append a new revision; deletes append a tombstone, so model output can remain inspectable under history. The audio worker populates it and can suggest character links from voice references. The context endpoint includes current transcript segments overlapping each cut, except deleted segments, so audiovisual matching can start from aligned evidence.

The cut summary combines visual evidence and context for still-image and animation planning. Confidence is qualitative rather than a calibrated probability. Preserve each prompt version, the evidence used, and user edits. Output from generation services belongs in new artifacts and should not replace reference media.
