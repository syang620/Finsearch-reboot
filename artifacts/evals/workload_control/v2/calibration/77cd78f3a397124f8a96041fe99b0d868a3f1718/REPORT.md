# Workload-control v2 corrected calibration report

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED`**

Selected policy: **`B_CONSECUTIVE_10`**

This report supersedes the comparison at `7d68a8c` after independent review found
that its constant family keys could aggregate or hand off CPU across distinct
executables, and that arbitrary commands containing `qdrant` could receive a
service exemption. The corrected evaluator follows the preregistered stable key:
category + canonical executable + nearest identified application/service ancestor.
Only exact required-service executables receive the narrow exemption.

The candidate set, parameters, selection preference, and acceptance criteria did
not change. Preserved raw observations were re-evaluated. Because the old S2
capture omitted zero-CPU supervision processes, S2 alone was recaptured with the
corrected collector before the final comparison. No semantic case, retrieval, or
model inference ran.

## Corrected results

- Supervised idle: 0 B10 CPU episodes / 900 samples; maximum group streak 9.
- Terminal-only idle: 0 episodes / 900 samples; maximum streak 1.
- Required service activity: 0 episodes / 180 samples; maximum streak 1.
- Sustained interference: 3/3 detected at 9.9996–9.9999 seconds.
- Six 1.5-second bursts: 0 episodes; maximum streak 1.
- Real Chrome: hard rule detected in the first sampled interval.
- Process evidence: 100% classified or retained as unknown.

The terminal capture retained every known supervision executable. It recorded no
ChatGPT/Codex UI, renderer, or active service, but retained four launchd-owned
`browser_crashpad_handler` processes at 0.0% CPU for all 900 samples. Terminal-only
control monitoring is therefore viable with that explicit inert-handler limitation;
semantic execution autonomy remains untested.

The offline PR32 replay is now labeled a lower bound. Its sequential detail
snapshot omitted processes hidden by the broader v1 exemption filter and omitted
sub-threshold CPU, so zero observed B10 episodes cannot support an exact zero claim.
Historical statuses remain unchanged.

## Review findings addressed

1. Distinct executables/ancestors no longer aggregate or hand off a B10 streak.
2. A Qdrant-named client remains CPU-scored; exact service executables alone are
   excluded.
3. Each comparison input records its actual capture implementation SHA.
4. Zero-CPU supervision executables are retained; S2 was recaptured and audited.
5. Historical PR32 persistence/occupancy/burden replay is described as a lower
   bound rather than exact.

## Limitations and release status

The calibration covers 33 clean minutes, not a multi-hour run. The supervised
nine-sample streak is one sample below the selected boundary. The evidence does
not establish timeout causality or semantic quality. PR31 remains frozen, draft,
release-blocked, and unmerged. This calibration does not authorize the semantic
baseline; a separately reviewed opt-in integration is still required.

## Verification

The corrected diagnostics suite passed **27/27**. The full repository suite
produced **1,205 passed**, **47 subtests passed**, **2 failed**, and **25 warnings**.
The two failures are the unchanged known planner-alias route and retrieval
no-tool-call retry-count expectations; neither path changed in this PR.
