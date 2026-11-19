# Queue redesign

This inventory records the behavior at `af57f81`. The final column records the implemented resolution. Multiple routes are grouped when they use the same guard or protect the same operation.

**Real conflict** keeps a validity check or deduplicates an identical request. **Dependency** orders work whose inputs are being produced. **User wins** commits the edit and checks running inputs before publishing. **Scope** limits serialization to the data or resource in use.

## API refusals and implicit locks

| Location and original gate | What it protects | Classification | Resolution |
| --- | --- | --- | --- |
| `main.delete_project`, `main.delete_episode` | A queued or leased worker still needs the media and database rows | Real conflict | Resolved. Keep deletion protection for the selected project or episode, including workers finishing cancellation. |
| `worker.enqueue` and its callers in `main.py` (prepare, hard cuts, frame boundary move, audio and voice jobs) and `keyframes_api.py` | An episode already has work | Scope | Resolved. Job requests deduplicate and use relevant prerequisites. Uploaded voice references queue independently of the episode's media. |
| `main.py` timeline timing, split, merge, delete, undo, redo and `cut_history.busy` | Full prepare or hard-cut replacement would race a cut edit | Real conflict | Resolved. Retains 423 for work replacing the cut list. |
| Timeline timing and history paths when `cutframes` is queued or running | Preview images and times match the boundaries a worker read | User wins | Resolved. Edits cancel obsolete previews and enqueue replacements. A merge cancels jobs targeting the archived cut. |
| Timeline timing and restore revision checks | Stale clients could overwrite a newer edit | Real conflict | Resolved. Keep compare-and-save for manual edits based on a saved revision. |
| `POST /api/cuts/{id}/detect-hardcuts` and `step_cut_boundary` | Existing preview and boundary data must match the edit | Scope / real conflict | Resolved. Both read the live boundary at start and restart if it changes while working. |
| `keyframes_api.enqueue_selection`, `enqueue_cut_selection` | Keyframe selection needs valid cut boundaries | Dependency | Resolved. Select current active cuts and subtitle mask after prerequisites. Save each completed selection with its fingerprint. |
| `main.extract_episode_subtitles` and subtitle queue validation | Subtitle probes may need transcription. The guard also includes voice samples | Dependency | Resolved. Subtitle extraction waits for pending transcription and uses the latest speech timing. Voice samples can queue independently. |
| `main.py` single first-frame recognition, `target_characters_api.py` single and bulk detection | A request already covers this cut | Real conflict | Resolved. Return the active job ID for the same choices and prerequisites. |
| `main.py` bulk character boxes/recognition and `target_characters_api.py` bulk target detection | A different job is present in the episode | Scope | Resolved. Recognition waits for its box jobs, and target detection waits for the selection and relevant opening-frame detector. Episode-wide requests select live targets after cut replacement. |
| `target_characters_api.py` detection request without current target frames | Detection needs a current evaluated selection | Dependency | Resolved. Reuses or queues selection and reads the current one at start. |
| `target_characters_api.py` shared target-edit lookup, cast lock/unlock, frame/slot/label changes, redetection edits | A detector may replace the edited run or use the old anchors | User wins | Resolved. The edit commits, generated analysis becomes stale and running detection restarts from the edited anchors. |
| `main.py` removal of a stale detection frame | An existing episode job might use the frame | User wins | Resolved. Frame validity and saved references govern removal. Running detections revalidate their inputs at publication. |
| `worker.process` cutframe boundary move checks for other episode jobs | Old assumptions about the source and boundaries might be published | Scope | Resolved. Schedule the affected boundaries and preview writes by scope. Validate the neighboring cut and boundary before saving. |
| `main.py` single, batch, only-missing and retry visual analysis | Prepare/hard cuts/detection can change current cuts or character evidence | Dependency | Resolved. Queue the relevant media, detection and keyframe prerequisites. Gather cut evidence at start. Deduplicate retries and retain valid checkpoints. |
| `main.py` visual queue/retry overlap checks and legacy text retry | Duplicate work or a different stage on the same cut | Real conflict / scope | Resolved. Return the existing ID for identical requests. Schedule different choices by their data scope. Explicit historical text retries retain the user-edited request. |
| `main.py` visual work while a consistency summary exists in the queue | A reader could run beside a writer | Scope | Resolved. Schedule by read and write domains. Consistency reviews of neighboring cuts wait when they read each other’s results. |
| `main.py` performance beats queue and retry | Keyframe analysis is pending or a beats request already exists | Dependency / real conflict | Resolved. Beats wait for the selected cut's evidence producers and reuse matching requests. |
| `main.py` one-cut consistency summary and `summary_queue` validation | Visual evidence or beats have yet to finish, or consistency is already queued | Dependency / real conflict | Resolved. Requests wait for evidence producers and deduplicate by cut and choices. |
| `main.py` episode consistency batch, all-text, review answers, text chain and summaries-from-here | Busy cuts are rejected or skipped. Some routes also block on any prepare/hardcut/detection job | Dependency | Resolved. Queue behind evidence producers and reuse matching work. An episode request behind cut replacement selects its cuts after the replacement, including an episode with no cuts yet. |
| `main.py` Cut summary preview, queue and retry | A summary is pending, media work holds the episode, or the client preview uses an obsolete base | Dependency / real conflict | Resolved. Queue behind applicable producers, reuse matching requests, and retain explicit stale preview checks. |
| `main.py` composition compaction request/retry | A compaction for the same evidence is pending, or the requested evidence is unavailable | Real conflict / dependency | Resolved. Reuse an exact active request and gather current fields at start. A changed hash stops an old cache from projecting obsolete evidence. |
| `main.py` character description retry | A matching profile request is already active | Real conflict | Resolved. Return the existing description request. Check the live box and Details before appending the profile. |
| `main.py` retry of other jobs and generic SQLite integrity-error translations | Any active job is treated as a refusal | Scope | Resolved. Apply shared deduplication and prerequisites to retries. Current revision and archived-cut checks still apply where required by the requested operation. |
| `sequence_edit.extend`, `use_source_dialogue`, `set_points`, `change_voices` and sequence delete/cancel paths in `sequences.py` | An edit can alter the range, compilation, markers, or voices while work uses an older revision | User wins | Resolved. Edits save through running work. Deletion cancels the sequence's jobs, and current writes compare their inputs and output base. |
| `sequence_edit.queue` compile/direct/bible/plan/shots/review, `sequences.py` route descriptions | An episode job, another step, or a matching shot/review is pending | Dependency / real conflict | Resolved. Queue step prerequisites and reuse matching jobs. Keep the dialogue confirmation and stale manual-revision checks. |
| `sequence_edit` manual shot instructions and `continue_extension` | Running shot work may ignore new notes or existing pending work may hold the next step | User wins | Resolved. Explicit rewrites read current inputs. Voice and timing edits mark affected shots stale. |
| `assistant/api.py` send, regenerate while running and the per-session active reply guard | A turn depends on earlier turns and regeneration replaces an identified turn | Dependency / real conflict | Resolved. Queue replies in session order and deduplicate identical requests. A retry of a current reply with different choices is queued. Keep the latest-reply check when replacing chat history. |
| `assistant/api.py` proposal apply and dismiss, `main.py` analysis and transcript revisions, `sequence_edit` compilation/shot edit revisions | An existing revision or already-resolved proposal would be overwritten | Real conflict | Resolved. Keep explicit revision comparisons for hand edits and proposal application. |
| `storage.py` inventory cleanup, run history and saved evidence deletion, orphan result/transcript and orphan evidence cleanup | Active work may need a specific run, checkpoint, file, or revision. Several guards check the entire episode | Scope | Resolved. Protect active run records, the files and checkpoints they use, and structured checkpoint integrity. Deleting successful history preserves the prerequisites it already fulfilled. |
| `main.remove_regenerated_legacy` | Regenerated legacy frame could replace evidence currently used by a job | Scope | Resolved. Check references to the particular JPEG and verify byte equality before removing it. |
| `main.py` cancel and `queue_api.py` dismiss | Invalid lifecycle transitions or a kind that cannot safely stop | Real conflict | Resolved. Check the selected run’s lifecycle and whether its kind can stop. Finished or non-cancellable runs keep their current state. |
| `external_api.py` export download before completion | The requested archive is unavailable | Dependency | Resolved. Keep download readiness. Export requests behind a pending plan or boundary producer queue immediately. Workers assemble archives from the inputs at start. |

## Claiming and input freshness

| Location and original gate | What it protects | Classification | Resolution |
| --- | --- | --- | --- |
| `worker.claim_job` episode hold and export exception | Broad serialization of processing | Scope | Resolved. Resolve read and write scopes and live resource leases under the claim transaction. Downloads serialize only against work that writes their inputs. |
| `worker.claim_job` remote lane owner prefix and `ALONGSIDE_REMOTE` | Remote text can coexist with local analysis in one special case | Scope | Resolved. Remote text uses the ordinary data scope rules without leasing the local GPU. |
| `worker.claim_job` text beside text | Text server accepts more than one slot | Scope | Resolved. Independent local text requests for one episode can share model slots. Local visual and audio work need exclusive GPU access across episodes. |
| `worker.claim_job` cutframes alongside analyze/regen | CPU previews can be independent of analysis | Scope | Resolved. Ordinary preview writes can run beside analysis. A frame-step boundary writer waits for cut readers of those boundaries. |
| `worker.claim_job` `WAITING_PIPELINES`, `wait_for`, failed or missing parent sweep | Only certain text jobs recognize dependencies | Dependency | Resolved. Each claimed kind waits for every job in `payload.depends_on` to finish successfully. |
| `worker.claim_job` and `keyframes.ensure_selection` `wait_for_keyframes` | Visual work waits for selection even if a selection later fails | Dependency | Resolved. Visual jobs use the common list and are cancelled if selection fails or is cancelled. |
| `worker.cancel_jobs` and Cut summary `depends_on_consistency` | Cancellation and evidence selection follow different parent fields | Dependency | Resolved. Cancel descendants through the shared list. Each handler reads the current evidence when it starts. |
| `worker.claim_job` remote provider filter and local/remote lane limits | Workers only claim jobs they can execute | Scope | Resolved. Keep provider routing, chat reservation and lane capacity. Preserve interrupted-first ordering and the existing cache order. Priority class is metadata. |
| `main.py` visual `analysis_config_for_cut`, `selected_analysis_context`, references, current results, `base_result_ids`, `CutRun.save` | Visual jobs carry frozen evidence and reject a changed result base at publication | User wins | Resolved. Queue choices and cut IDs, gather at start, and compare each cut’s inputs during publication. Preserve valid per-cut checkpoints and provenance, including for legacy analysis. |
| `summary_queue` and `consistency_summary.refresh` `refresh_evidence` and `base_id` | Rebuilds prompt evidence after beats, only for a flagged job | User wins | Resolved. Every summary gathers current evidence and picture inputs and checks the fingerprint before saving. |
| `review_answers.refresh` and `inference_worker` dependency check in review answers | Only chained work refreshes consistency questions at start | User wins | Resolved. Every draft reads current questions and answers. It restarts on input edits and preserves decisions saved after the request. |
| `main.py` and `inference_worker.process_regen` `refresh_answers`, `refresh_results`, `base_id` and preceding consistency result lookup | A queued summary mixes frozen data and inputs from preceding jobs, and fails on a changed result base | User wins | Resolved. Gather current answers and results, retain explicit request choices, keep hand edits, and save the actual model request for provenance. |
| `sequence_edit`, `sequence_pipeline` `base_id`, shot/compile source and voice checks | Old worker output could overwrite current sequence results or edited voices | User wins | Resolved. Gather current revisions and restart on any change. Retain manual output and append extension speech to the edited dialogue. |
| `performance_beats.process` “Superseded by a newer keyframe analysis” | Prevents old beats from saving after the underlying keyframes changed | User wins | Resolved. Requeue with a quiet period, record the revision IDs, and keep any hand edit made after execution started. |
| `main.py` legacy staged analysis retry and retry of unfinished visual cuts | Captured frame evidence must match its cut | User wins | Resolved. Legacy analysis retries read current data and re-extract frames only for changed boundaries. An explicitly edited historical frame request skips a cut whose saved frame bounds no longer match. |
| `anime_bulk`, `target_characters`, `keyframes`, transcription, subtitle, and character-profile commits | Domain-specific protections allow some changes but may still publish obsolete inputs | User wins | Resolved. All named handlers verify current inputs during publication, record what they read and restart on changes. Retain hand-applied revisions and manual detection anchors. |
| `prompt_context.beats_current`, `sequence_view` shot stale, compilation source checks and `composition_brief` | Readers need to distinguish evidence from an obsolete upstream revision | User wins | Resolved. Expose stale reasons and explicit reruns. Saved upstream revisions and metadata edits mark affected derivatives stale without enqueueing them. |

## Editor controls and refresh

| Location and original gate | What it protects | Classification | Resolution |
| --- | --- | --- | --- |
| `App`, `EditorHeader` media and character menus | Buttons disable while episode jobs exist | Scope | Resolved. Keep valid actions available and show queued dependencies and active jobs. |
| `App`, `CutInspector` and `useCutEdits` timing fields, Detect within, retry on 423 | Fights repeated preview jobs and broad media locks | User wins / real conflict | Resolved. Edits proceed during preview rebuilds and refresh obsolete previews. Show the full-media replacement conflict directly. |
| `AnalysisDialog`, `AnalysisControls`, `runState.preflight`, `SummaryRequestEditor`, `AnimationPromptPanel`, `ConsistencySummary`, `PerformanceBeats` | Episode or cut work disables evaluation, visual stages and text actions. Batch messages reflect skipped busy cuts | Dependency / real conflict | Resolved. Accept work behind relevant producers. Reuse exact active requests and keep form validation and manual revision checks. |
| `CharacterDetectionPanel`, `TargetFrames`, `targetView.detectReason` | Detection blocks resubmission, drawing, moving, label/cast edits and cleanup | User wins / real conflict | Resolved. Keep controls open during detection. Edits become anchors on restart and the same detection request reuses its job ID. |
| `TranscriptEditor`, `TranscribeRangePanel`, `SubtitlePanel` | Any episode job can disable audio, voice samples or subtitles | Scope / dependency | Resolved. These actions accept valid requests during other jobs. Extraction shows its prerequisite, and a stale track has an Extract again action. |
| `DialogueTab`, `SequenceDialog`, `sequenceView` | Active sequence jobs gate step actions, edits, extension, and shot rewrites | User wins / dependency | Resolved. Steps and edits stay available. Dialogue retains a local draft when a newer compilation arrives, and shot rewrites are explicit. |
| `CutAssistant` send/regenerate gates and GPU-busy send blocker | One reply owns a session while in progress | Dependency | Resolved. Send remains available during replies. Follow-ups show their queued status and keep conversational order. |
| `Storage` cleanup-candidate controls | Backend candidate state includes an episode-wide active-job hold | Scope | Resolved. Show file-specific use and integrity reasons in the candidate list. |
| `RunUi.ActionButton` pending and individual request guards throughout editor | Prevents duplicate mutations while an HTTP request is being sent. Some callers pass a job state as pending | Real conflict | Resolved. Disable duplicate clicks while the HTTP mutation is in flight. Show queued and running jobs separately. |
| `App` Projects and Project Detail 5 s refresh | Job badges on multi-episode screens | Scope | Resolved. Both use the shared host event feed. |
| `App.Editor` 2.5 s status tick and idle full refresh | Queue changes and updated editor evidence | Scope | Resolved. A single episode connection refreshes status, data and the prepare-version waveform cache. |
| `useCutStages` 4 s, `AnalysisTimeline` 5 s | Results/progress of in-flight work | Scope | Resolved. Episode events trigger reads of the affected views. |
| `TargetFrames` 3 s selection, 3 s target detection, 5 s analysis | Selection and work status | Scope | Resolved. Target frames shares the episode subscription. |
| `NewCharacterDialog` 3 s, `CharacterDetectionPanel` 3 s | Profile/detection job completion | Scope | Resolved. Both refresh on episode events. |
| `QueueDrawer` status poll and `SequenceDialog` 2.5 s | Queue rows, sequence job progress and stale state | Scope | Resolved. Both refresh on episode events, including sequence export progress. |
| `CutAssistant` 10 s configuration and 3 s session refresh | Provider availability and turn lists | Scope | Resolved. Shared events update provider and session state. The turn stream carries token deltas. |
| `LocalModelControl` 2.5 s and `VisualEngineControl` 5 s | Host model startup and download status, including when no job exists | Scope | Resolved. Host notifications cover model status and expired worker heartbeats. |
| `QueueDrawer` and `RunUi` 1 s ticks and `CutAssistant` 0.5 s tick | Relative and elapsed clocks | Scope | Resolved. Local clock ticks make no API requests. |

## Verification

- The full container backend suite passed all 947 tests and 55 subtests. The frontend production build and all 122 frontend tests passed. Both final suites ran with zero skips.
- `docker compose up -d` and both worker restarts completed. The rebuilt API, inference and audio images were running when `/api/health` returned `{"ok":true}`.
- Chromium queued visual analysis during opening-frame detection, split a cut while its preview lease was held, saved a moved target box during tracking, and received queue drawer updates through one episode event stream. The in-flight states used controlled leases. The media worker generated the replacement previews and the split retained the original cut ID on its first half.
- The disposable app imported and prepared clips with and without audio, detected three cuts, sought and decoded a cut in Chromium, and saved a note. A container restart preserved the note, split, silent-clip status, and original upload checksum. An invalid MP4 failed during preparation and an unsupported extension returned HTTP 400.
- Separate processes contended for the same GPU lock, and the second acquired it after the first released it. Both rebuilt worker images allocated a tensor and ran a calculation on the Radeon GPU under that lock.

### Earlier verification failures and skips

- The host baseline had 24 failures and nine skips with missing media or ONNX dependencies. Affected tests included frame sampling, evidence cleanup and recovery, keyframes, target selection, frame budgets, import, timing edits, and silent clips. The skips covered the AniRef ONNX interpreter comparison, the five subtitle video scan tests, the brief's original-resolution boundary frame export, sequence image tiling and cropping, and the audio-preserving listening clip test. Container subsets without ONNX also skipped the AniRef test. The final full container suite includes FFmpeg, FFprobe, ONNX and ONNX Runtime.
- Intermediate queue tests flagged obsolete snapshot expectations in analysis, audio, character detection, frame budgets, keyframes, and legacy retry. Character batch coverage exposed a thread-order assumption. Scope tests flagged obsolete concurrency expectations for adjacent consistency summaries and sequence shots, plus missing episode settings in a GPU-lock test fixture. Updated assertions cover live inputs and the intended scope, and the fixtures now supply their dependencies.
- An initial media workflow test assumed synchronous preview updates, and early browser test runs used an unsuitable fixture container format, assumed malformed MP4s failed synchronously, read queue dependency labels in the wrong shape, or matched duplicate UI controls. These checks were rerun after correcting the fixtures and assertions.
- The first full Compose build timed out downloading a CPU Torch wheel selected by audio dependency resolution. The audio image now pins the existing ROCm Torch and NumPy versions and its rebuilt image passed the audio runtime imports. The next full build timed out downloading the pinned inference ROCm SDK from AMD. The inference image was rebuilt with the current source and the previously installed, matching ROCm runtime. A separate backend test setup timed out on PyPI and the suite was rerun using already installed test dependencies.
- The first direct GPU smoke check reached its timeout while the configured resident text server held the lock. Both checks passed with the inference service temporarily stopped. After it started again, the API health check passed.

<details>
<summary>Backend tests that failed in earlier runs, using their names at the time</summary>

- `test_analysis.py` — `test_batch_snapshots_and_resume`
- `test_analysis_media.py` — `test_fractional_boundary_first_frame_is_not_skipped`, `test_fractional_frame_rate_boundaries_exclude_next_shot`, `test_frame_sampling_never_includes_the_next_cut`, `test_model_receives_only_frames_through_the_saved_last_preview`, `test_prepares_silent_padded_clip_without_overwriting_source`
- `test_assistant.py` — `test_chat_session_can_refuse_to_wait`, `test_visual_work_waits_for_chat_and_chat_waits_for_visual_work`
- `test_audio_api.py` — `test_many_generated_words_are_queued_as_analysis_context_without_approval`, `test_transcribe_range_validation_and_queued_payload`
- `test_batch_pipeline.py` — `test_run_batch_runs_stages_in_model_order_for_all_cuts`
- `test_batch_queueing.py` — `test_characters_must_be_settled_before_cuts_are_queued_from_a_cut`, `test_episode_visual_analysis_queues_one_job_for_every_cut`, `test_only_missing_stages_are_queued_and_finished_cuts_are_skipped`, `test_queued_cut_context_carries_the_episode_subtitles_that_overlap_it`
- `test_brief.py` — `test_export_uses_snapshotted_first_frame`
- `test_character_detection_api.py` — `test_analysis_receives_frame_labels_and_reference_images`, `test_automatic_matches_are_unverified_and_queued_labels_are_snapshotted`, `test_batch_queue_does_not_confirm_labels`, `test_bulk_detection_continues_after_failure_and_requeues_only_failed`, `test_detect_label_reference_and_snapshot`
- `test_character_profile.py` — `test_edit_during_description_requeues_without_appending_old_text`
- `test_composition_brief.py` — `test_export_can_run_alongside_analysis`
- `test_frame_recovery.py` — `test_bulk_gc_keeps_mismatched_evidence_and_original`, `test_cache_is_bounded_and_expires`, `test_cleanup_verifies_linked_result_before_removing`, `test_detection_uses_precise_time_when_rounded_labels_collide`, `test_different_legacy_image_is_preserved`, `test_legacy_sampled_frame_is_recreated_then_removed`, `test_missing_manifest_recreates_interval_index`, `test_missing_original_uses_legacy_frame_without_deleting_it`, `test_result_and_cut_frames_are_regenerated_without_persisting_jpegs`
- `test_keyframe_analysis.py` — `test_a_pending_selection_is_reused_and_a_first_frame_only_job_waits_for_none`, `test_the_visual_job_queues_the_missing_selection_and_waits_for_it`
- `test_keyframes.py` — `test_decoder`, `test_episode_job_and_stale_cache`, `test_variable_rate_timestamps`
- `test_keyframes_cut_api.py` — `test_queue_snapshots_only_current_cut`
- `test_local_text.py` — `test_failed_legacy_analysis_can_retry_as_cancellable_local_text`, `test_text_jobs_of_one_episode_can_run_together_but_other_jobs_still_wait`
- `test_remote_text.py` — `test_remote_lane_runs_text_beside_visual_analysis_without_holding_it_back`, `test_routing_splits_local_provider_jobs_between_lanes`
- `test_runs.py` — `test_gpu_is_exclusive_across_episodes_and_until_cancelled_job_releases`
- `test_sequence_pipeline.py` — `test_sequence_jobs_share_their_episode_with_text_jobs_only`
- `test_target_characters.py` — `test_a_different_selection_deletes_the_people_found_on_the_old_frames`
- `test_text_chain.py` — `test_later_jobs_wait_for_the_job_before_them`
- `test_video_frame_budget.py` — `test_automatic_budget_limits_encoded_frames_across_windows`, `test_detected_character_context_is_included_without_selection`
- `test_workflow.py` — `test_detect_inside_selected_cut_preserves_neighbors_and_archives`, `test_frame_step_moves_both_cuts_and_preserves_results`, `test_import_prepare_detect_edit_and_persist`, `test_low_frame_rate_cut_end_uses_decoded_frame`, `test_low_frame_rate_no_audio_workflow`

</details>

## Phase 3 notes

- Retain stable cut IDs and per-cut input fingerprints. Analyze already has per-cut result checkpoints, per-stage progress, and model-major execution. Retries can preserve checkpoints knowing the matching input revisions.
- Keep batch membership, selected models, original request identity, and phase counts available when one logical request is later expressed as multiple claimable jobs.
- The dependency graph records why work waits. Distinguish that from resource contention when a future scheduler can move an interactive job ahead of ready batch work.
- Local text slots use a shared server on the GPU. Vision and audio need exclusive use. Remote lanes can be independently available while conflicts over SQLite-backed data still apply.
- Preserve the stopping lease until the worker releases its resource. A priority label by itself gives no permission to preempt a running job.

The existing AniRef groups and model-major analysis batch remain the execution units. Per-cut splitting, GPU yielding and priority-based scheduling are reserved for Phase 3.
