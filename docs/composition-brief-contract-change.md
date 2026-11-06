# Composition brief contract change (in development)

The existing brief and export keep their default response format. The opt-in
`view=composition` introduces a separate cut schema and query parameters
`include`, `fields`, `only_ready`, and `updated_since`. Composition exports use
an independent leased worker and add full-size boundary frames and manifest
metadata. Queued exports are excluded from the global queued-analysis cancel
operation. Existing summary generation now retries and rejects truncated,
overlength, reasoning-leaking, or structurally invalid production briefs
instead of saving them as current revisions.

This change is not complete: the current deterministic projection can omit
visual-event details and the fixture range exceeds the 2 KB-per-cut and 50 KB
range limits. The latest fixture response is approximately 70 KB for 25 cuts;
the live measurement and cut 49 example are in
`composition-brief-size-draft.md`. The size-budget regression test
intentionally fails until both the compactness and the required evidence
coverage are satisfied. Do not treat this draft view as a final composition
contract. Complete dialogue is retained even when it alone exceeds the size
target.
