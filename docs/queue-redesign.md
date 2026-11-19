# Queue redesign

This inventory records the behavior at `af57f81`. Each row stays open until its replacement is implemented and verified. Multiple routes are grouped when they use the same guard or protect the same operation.

**Real conflict** keeps a validity check or deduplicates an identical request. **Dependency** orders work whose inputs are being produced. **User wins** commits the edit and checks running inputs before publishing. **Scope** limits serialization to the data or resource in use.

## API refusals and implicit locks

| Location and current gate | What it protects | Classification | Replacement |
| --- | --- | --- | --- |
| `main.delete_project`, `main.delete_episode` | A queued or leased worker still needs the media and database rows | Real conflict | Keep deletion protection, including workers finishing cancellation |
| `worker.enqueue` and its callers in `main.py` (prepare, hard cuts, frame boundary move, audio and voice jobs) and `keyframes_api.py` | An episode already has work | Scope | Resolved. Job requests deduplicate and use relevant prerequisites. Uploaded voice references queue independently of the episode's media |
| `main.py` timeline timing, split, merge, delete, undo, redo and `cut_history.busy` | Full prepare or hard-cut replacement would race a cut edit | Real conflict | Resolved. Retains 423 for work replacing the cut list |
| Timeline timing and history paths when `cutframes` is queued or running | Preview images and times match the boundaries a worker read | User wins | Resolved. Edits cancel obsolete previews and enqueue replacements. A merge cancels jobs targeting the archived cut |
| Timeline timing and restore revision checks | Stale clients could overwrite a newer edit | Real conflict | Keep the existing compare-and-save checks |
| `POST /api/cuts/{id}/detect-hardcuts` and `step_cut_boundary` | Existing preview and boundary data must match the edit | Scope / real conflict | Resolved. Both read the live boundary at start and restart if it changes while working |
| `keyframes_api.enqueue_selection`, `enqueue_cut_selection` | Keyframe selection needs valid cut boundaries | Dependency | Resolved. Select current active cuts and subtitle mask after prerequisites. Save each completed selection with its fingerprint |
| `main.extract_episode_subtitles` and subtitle queue validation | Subtitle probes may need transcription. The guard also includes voice samples | Dependency | Depend on transcription when present. Schedule voice samples using resource scopes |
| `main.py` single first-frame recognition, `target_characters_api.py` single and bulk detection | A request already covers this cut | Real conflict | Return the existing job for the same selection and choices |
| `main.py` bulk character boxes/recognition and `target_characters_api.py` bulk target detection | A different job is present in the episode | Scope | Queue admission resolved. Recognition waits for its box jobs, and target detection waits for its selection. Episode box and recognition requests can choose targets after media work |
| `target_characters_api.py` detection request without current target frames | Detection needs a current evaluated selection | Dependency | Resolved. Reuses or queues selection and reads the current one at start |
| `target_characters_api.py` shared target-edit lookup, cast lock/unlock, frame/slot/label changes, redetection edits | A detector may replace the edited run or use the old anchors | User wins | Resolved. The edit commits, generated analysis becomes stale and running detection restarts from the edited anchors |
| `main.py` removal of a stale detection frame | An existing episode job might use the frame | User wins | Resolved for queue admission. Frame validity and saved references govern removal, and detection publication revalidates its inputs |
| `worker.process` cutframe boundary move checks for other episode jobs | Old assumptions about the source and boundaries might be published | Scope | Schedule shared data access and check cut inputs before saving |
| `main.py` single, batch, only-missing and retry visual analysis | Prepare/hard cuts/detection can change current cuts or character evidence | Dependency | Add the applicable prerequisites and gather cuts, boxes, results and context at start |
| `main.py` visual queue/retry overlap checks and legacy text retry | Duplicate work or a different stage on the same cut | Real conflict / scope | Deduplicate identical requests. Schedule different stage requests against their data scopes |
| `main.py` visual work while a consistency summary exists in the queue | A reader could run beside a writer | Scope | Enforce conflicts through read/write domains |
| `main.py` performance beats queue and retry | Keyframe analysis is pending or a beats request already exists | Dependency / real conflict | Wait for keyframe analysis and reuse a matching beats request |
| `main.py` one-cut consistency summary and `summary_queue` validation | Visual evidence or beats have yet to finish, or consistency is already queued | Dependency / real conflict | Wait for producers and reuse matching requests |
| `main.py` episode consistency batch, all-text, review answers, text chain and summaries-from-here | Busy cuts are rejected or skipped. Some routes also block on any prepare/hardcut/detection job | Dependency | Accept requests behind applicable producers and evaluate missing inputs at start |
| `main.py` Cut summary preview, queue and retry | A summary is pending, media work holds the episode, or the client preview uses an obsolete base | Dependency / real conflict | Reuse matching requests, depend on producers, retain explicit stale preview checks |
| `main.py` composition compaction request/retry | A compaction for the same evidence is pending, or the requested evidence is unavailable | Real conflict / dependency | Return an existing exact request and gather inputs after prerequisites |
| `main.py` character description retry | A matching profile request is already active | Real conflict | Return the existing request |
| `main.py` retry of other jobs and generic SQLite integrity-error translations | Any active job is treated as a refusal | Scope | Apply shared queue admission and preserve actual validation errors |
| `sequence_edit.extend`, `use_source_dialogue`, `set_points`, `change_voices` and sequence delete/cancel paths in `sequences.py` | An edit can alter the range, compilation, markers, or voices while work uses an older revision | User wins | Save edits, validate running inputs, and stop or supersede work for a deleted sequence |
| `sequence_edit.queue` compile/direct/bible/plan/shots/review, `sequences.py` route descriptions | An episode job, another step, or a matching shot/review is pending | Dependency / real conflict | Use step dependencies and return existing IDs for identical requests |
| `sequence_edit` manual shot instructions and `continue_extension` | Running shot work may ignore new notes or existing pending work may hold the next step | User wins | Queue explicit requests with fresh inputs. Invalidate already-written output on edits |
| `assistant/api.py` send, regenerate while running and the per-session active reply guard | A turn depends on earlier turns and regeneration replaces an identified turn | Dependency / real conflict | Queue conversation turns in order and deduplicate an identical regeneration |
| `assistant/api.py` proposal apply and dismiss, `main.py` analysis and transcript revisions, `sequence_edit` compilation/shot edit revisions | An existing revision or already-resolved proposal would be overwritten | Real conflict | Keep explicit revision comparisons for hand edits |
| `storage.py` inventory cleanup, run history and saved evidence deletion, orphan result/transcript and orphan evidence cleanup | Active work may need a specific run, checkpoint, file, or revision. Several guards check the entire episode | Scope | Protect actual consumers and restorable checkpoints, with unrelated jobs free to proceed |
| `main.remove_regenerated_legacy` | Regenerated legacy frame could replace evidence currently used by a job | Scope | Check consumers of the particular frame and keep byte equality verification |
| `main.py` cancel and `queue_api.py` dismiss | Invalid lifecycle transitions or a kind that cannot safely stop | Real conflict | Preserve action capabilities. A repeated cancel can report the current state |
| `external_api.py` export download before completion | The requested archive is unavailable | Dependency | Keep the download readiness contract. Continue accepting export requests |

## Claiming and input freshness

| Location and current gate | What it protects | Classification | Replacement |
| --- | --- | --- | --- |
| `worker.claim_job` episode hold and export exception | Broad serialization of processing | Scope | Use shared scope and exclusive resource checks |
| `worker.claim_job` remote lane owner prefix and `ALONGSIDE_REMOTE` | Remote text can coexist with local analysis in one special case | Scope | Remote text has no local GPU resource. Check data access normally |
| `worker.claim_job` text beside text | Text server accepts more than one slot | Scope | Express shared local text slots and exclusive GPU use with resource data |
| `worker.claim_job` cutframes alongside analyze/regen | CPU previews can be independent of analysis | Scope | Declare preview writes and boundary/evidence reads |
| `worker.claim_job` `WAITING_PIPELINES`, `wait_for`, failed or missing parent sweep | Only certain text jobs recognize dependencies | Dependency | Resolved. Each claimed kind waits for every job in `payload.depends_on` to finish successfully |
| `worker.claim_job` and `keyframes.ensure_selection` `wait_for_keyframes` | Visual work waits for selection even if a selection later fails | Dependency | Resolved. Visual jobs use the common list and are cancelled if selection fails or is cancelled |
| `worker.cancel_jobs` and Cut summary `depends_on_consistency` | Cancellation and evidence selection follow different parent fields | Dependency | Cascade through the same dependency list. Read current evidence at start |
| `worker.claim_job` remote provider filter and local/remote lane limits | Workers only claim jobs they can execute | Scope | Retain eligible lanes and pool capacity. Keep claim order and lease rules |
| `main.py` visual `analysis_config_for_cut`, `selected_analysis_context`, references, current results, `base_result_ids`, `CutRun.save` | Visual jobs carry frozen evidence and reject a changed result base at publication | User wins | Resolved for the current visual workflow. Queue choices and cut IDs, gather from a read transaction, and compare each cut's inputs during publication. Store fingerprints and preserve usable checkpoints |
| `summary_queue` and `consistency_summary.refresh` `refresh_evidence` and `base_id` | Rebuilds prompt evidence after beats, only for a flagged job | User wins | Always gather live evidence and guard the save |
| `review_answers.refresh` and `inference_worker` dependency check in review answers | Only chained work refreshes consistency questions at start | User wins | Gather questions and the current answers base consistently |
| `main.py` and `inference_worker.process_regen` `refresh_answers`, `refresh_results`, `base_id` and preceding consistency result lookup | A queued summary mixes frozen data and inputs from preceding jobs, and fails on a changed result base | User wins | Gather all current inputs and compare at publication. Keep the request's chosen instructions |
| `sequence_edit`, `sequence_pipeline` `base_id`, shot/compile source and voice checks | Old worker output could overwrite current sequence results or edited voices | User wins | Gather current versions and use the common input guard with sequence and shot scopes |
| `performance_beats.process` “Superseded by a newer keyframe analysis” | Prevents old beats from saving after the underlying keyframes changed | User wins | Requeue with a quiet period and record the inputs that were used |
| `main.py` legacy staged analysis retry and retry of unfinished visual cuts | Captured frame evidence must match its cut | User wins | Preserve historical provenance. Restart affected work using current inputs where applicable |
| `anime_bulk`, `target_characters`, `keyframes`, transcription, subtitle, and character-profile commits | Domain-specific protections allow some changes but may still publish obsolete inputs | User wins | Check relevant reads and revision bases inside publication transactions |
| `prompt_context.beats_current`, `sequence_view` shot stale, compilation source checks and `composition_brief` | Readers need to distinguish evidence from an obsolete upstream revision | User wins | Expose stale reasons and a user-triggered rerun. An edit does not queue its downstream results |

## Editor controls and refresh

| Location and current gate | What it protects | Classification | Replacement |
| --- | --- | --- | --- |
| `App`, `EditorHeader` media and character menus | Buttons disable while episode jobs exist | Scope | Keep valid actions available and show queue state |
| `App`, `CutInspector` and `useCutEdits` timing fields, Detect within, retry on 423 | Fights repeated preview jobs and broad media locks | User wins / real conflict | Let previews coexist with edits. Display the full-replacement conflict |
| `AnalysisDialog`, `AnalysisControls`, `runState.preflight`, `SummaryRequestEditor`, `AnimationPromptPanel`, `ConsistencySummary`, `PerformanceBeats` | Episode or cut work disables evaluation, visual stages and text actions. Batch messages reflect skipped busy cuts | Dependency / real conflict | Queue behind producers and expose duplicate IDs. Keep form validation and unsaved draft guards |
| `CharacterDetectionPanel`, `TargetFrames`, `targetView.detectReason` | Detection blocks resubmission, drawing, moving, label/cast edits and cleanup | User wins / real conflict | Keep edit controls open and send new anchors or reuse a matching queue request |
| `TranscriptEditor`, `TranscribeRangePanel`, `SubtitlePanel` | Any episode job can disable audio, voice samples or subtitles | Scope / dependency | Transcription and voice actions accept requests during other jobs. Subtitle admission is tracked above |
| `DialogueTab`, `SequenceDialog`, `sequenceView` | Active sequence jobs gate step actions, edits, extension, and shot rewrites | User wins / dependency | Keep controls open and reflect prerequisites and stale results |
| `CutAssistant` send/regenerate gates and GPU-busy send blocker | One reply owns a session while in progress | Dependency | Queue subsequent replies and preserve their order |
| `Storage` cleanup-candidate controls | Backend candidate state includes an episode-wide active-job hold | Scope | Display specific resource conflicts |
| `RunUi.ActionButton` pending and individual request guards throughout editor | Prevents duplicate mutations while an HTTP request is being sent. Some callers pass a job state as pending | Real conflict | Retain request-in-flight protection. Show existing jobs separately |
| `App` Projects and Project Detail 5 s refresh | Job badges on multi-episode screens | Scope | Resolved. Both use the shared host event feed |
| `App.Editor` 2.5 s status tick and idle full refresh | Queue changes and updated editor evidence | Scope | Resolved. A single episode connection refreshes status, data and the prepare-version waveform cache |
| `useCutStages` 4 s, `AnalysisTimeline` 5 s | Results/progress of in-flight work | Scope | Resolved. Episode events trigger reads of the affected views |
| `TargetFrames` 3 s selection, 3 s target detection, 5 s analysis | Selection and work status | Scope | Resolved. Target frames shares the episode subscription |
| `NewCharacterDialog` 3 s, `CharacterDetectionPanel` 3 s | Profile/detection job completion | Scope | Resolved. Both refresh on episode events |
| `QueueDrawer` status poll and `SequenceDialog` 2.5 s | Queue rows, sequence job progress and stale state | Scope | Resolved. Both refresh on episode events, including sequence export progress |
| `CutAssistant` 10 s configuration and 3 s session refresh | Provider availability and turn lists | Scope | Resolved. Shared events update provider and session state. The turn stream carries token deltas |
| `LocalModelControl` 2.5 s and `VisualEngineControl` 5 s | Host model startup and download status, including when no job exists | Scope | Resolved. Host notifications cover model status and expired worker heartbeats |
| `QueueDrawer` and `RunUi` 1 s ticks and `CutAssistant` 0.5 s tick | Relative and elapsed clocks | Scope | Resolved. Local clock ticks make no API requests |

## Phase 3 notes

- Retain stable cut IDs and per-cut input fingerprints. Analyze already has per-cut result checkpoints, per-stage progress, and model-major execution. Retries can preserve checkpoints knowing the matching input revisions.
- Keep batch membership, selected models, original request identity, and phase counts available when one logical request is later expressed as multiple claimable jobs.
- The dependency graph records why work waits. Distinguish that from resource contention when a future scheduler can move an interactive job ahead of ready batch work.
- Local text slots use a shared server on the GPU. Vision and audio need exclusive use. Remote lanes can be independently available while conflicts over SQLite-backed data still apply.
- Preserve the stopping lease until the worker releases its resource. A priority label by itself gives no permission to preempt a running job.
