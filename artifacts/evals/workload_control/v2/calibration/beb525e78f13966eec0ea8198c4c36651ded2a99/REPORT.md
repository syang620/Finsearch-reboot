# Workload-control v2 final workload-proved calibration report

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED`**

Selected policy: **`B_CONSECUTIVE_10`**

This correctly SHA-keyed comparison supersedes prior derived comparisons for
release use. The final analyzer fail-closes on complete scenario identity/count/
cadence, global AC/LPM/browser controls, awake liveness, terminal-only viability,
S3 frozen identity/event binding, and proof that all six registered S5 workloads
actually executed. All finite floats are canonicalized to nine decimal places
before JSON serialization. Candidates, ordering, thresholds, scenarios, and
acceptance criteria did not change. Complete `baa2dd2` raw captures contained all
required evidence and were re-evaluated without recapture. No semantic case,
retrieval, reranker, or model inference ran.

## Results

- Six captures match registered identity, preregistration hash, duration,
  one-second cadence, and exact sample count (2,370 total).
- AC/LPM hold in every sample; only the harness-owned S6 browser PID appears.
- Awake protection is active in all 2,370 samples and alive before cleanup.
- All 10 S3 frozen-provenance/event-binding checks pass.
- All six short bursts have unique matching start/exit PIDs, successful exits,
  registered timing/duration, and retained controlled CPU ≥50%; B10 produces zero
  short-burst episodes.
- Clean scenarios have zero B10 CPU/hard episodes across 1,980 samples.
- Sustained interference is detected 3/3 at 9.999918750, 9.999962458, and
  10.004894167 seconds; Chrome is detected in the first sampled interval.
- Terminal-only has zero active supervision/browser records; four inert,
  launchd-owned crash handlers remain visible at 0.0% CPU across 900/900 samples.
- Process evidence is 100% classified or retained as unknown.

The contaminated supervised batch and two invalid terminal attempts remain
preserved under the capture SHA and excluded. Historical PR32 replay remains a
lower bound; no historical status changes.

## Claim limits and verification

This supports only that B10 met the preregistered control-only calibration on this
machine/session. It does not establish multi-hour false-invalidation rates,
timeout causality, semantic quality, or benchmark validity. PR31 remains frozen,
draft, blocked, and unmerged; PR33 remains draft and must not merge automatically.

Focused diagnostics pass **42/42**. The full suite records **1,220 passed**, **47
subtests passed**, the same **2 known pre-existing failures**, and **25 warnings**.
A second documented analyzer run over identical inputs produced the exact same
comparison bytes and SHA-256.
