# Semantic workload qualification v3

Status: **inactive candidate pending review**

Workload qualification v3 separates answer-quality eligibility from controlled-latency
eligibility. It reuses the frozen workload-control-v2 process classifier and
`B_CONSECUTIVE_10` decision as a performance condition. It does not relabel or modify
any v2 calibration, approval, result, or invalid diagnostic.

An answer-quality result may be eligible only after all scheduled cases are captured
and evaluated exactly once, artifact and scorer finalization succeeds, model/index/
service identities are verified, AC power and Low Power Mode requirements hold, awake
protection remains active, and workload observation closes with complete evidence.
Captured analyst, model, and tool failures remain benchmark outcomes rather than
turning the entire answer-quality run into an infrastructure failure.

Controlled latency additionally requires no real Chrome or Safari process, no active
ChatGPT/Codex supervision process, no sustained external CPU trigger under
`B_CONSECUTIVE_10`, valid frozen-v2 sampling cadence, and no monitor error. Timing from
an unqualified environment is diagnostic only.

Ambient workload findings never stop the case loop. The frozen launcher continues the
in-flight and remaining cases, then the v3 finalizer records independent eligibility
objects in `workload_qualification_v3_completion.json`. The old generic
`official_baseline_eligible` field is deliberately absent from that authoritative v3
record so downstream claims must select an evidence dimension.

`scripts/diagnostics/rehearse_semantic_workload_qualification_v3.py` runs the exact
observer without semantic cases. Its default result is observational: a completed
rehearsal exits successfully even when controlled latency is false. The optional
`--require-controlled-latency` flag provides a future pre-consumption gate. Rehearsal
outputs are private local artifacts and grant no execution authority.

Before any future semantic attempt, separately investigate why a nominal 120-second
analyst timeout produced a 440-second case duration and prove an explicit end-to-end
case ceiling. Any later attempt also requires a new reviewed namespace, approval, and
explicit authorization; the consumed local-once-v1 namespace cannot be reused.
