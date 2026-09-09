# Workload-control v2 final complete-gate calibration report

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED`**

Selected policy: **`B_CONSECUTIVE_10`**

This correctly SHA-keyed comparison supersedes prior derived comparisons for
release use. The final analyzer fail-closes on every registered control: complete
scenario identity/count/cadence, global AC and Low Power Mode, browser identity,
per-sample awake liveness, terminal-only viability, and frozen S3 model/index/
service identity bound to the successful S3 workload event. The candidates,
ordering, thresholds, scenarios, and acceptance criteria did not change. Complete
raw captures from `baa2dd2` contained all required evidence and were re-evaluated
without recapture. No semantic case, retrieval, reranker, or model inference ran.

## Results

- All six captures match registered scenario identity, preregistration hash,
  duration, one-second cadence, and exact sample count (2,370 total).
- AC power and Low Power Mode off hold in every sample; only the harness-owned S6
  browser PID appears, within its recorded start/stop interval.
- Awake protection is active in all 2,370 samples and alive before cleanup.
- All 10 S3 frozen-provenance and event-binding checks pass, including exact
  Ollama and Qdrant identities and successful child exit.
- Clean scenarios have 0 B10 CPU/hard episodes across 900 supervised-idle, 900
  terminal-only, and 180 required-service samples.
- Sustained interference is detected 3/3 at 9.9999, 10.0000, and 10.0049 seconds.
- Six short bursts produce 0 B10 episodes; real Chrome is detected in the first
  sampled interval.
- Terminal-only has zero active supervision/browser records; four inert,
  launchd-owned crash handlers remain visible at 0.0% CPU across 900/900 samples.
- Process evidence is 100% classified or retained as unknown.

The browser-contaminated supervised batch and two invalid terminal attempts remain
preserved under the capture SHA and excluded. Historical PR32 replay remains a
lower bound; no historical status changes.

## Claim limits and status

The evidence supports only that B10 met this preregistered control-only calibration
on this machine/session. It does not establish multi-hour false-invalidation
rates, timeout causality, semantic quality, or benchmark validity. PR31 remains
frozen, draft, blocked, and unmerged. PR33 remains draft and must not merge
automatically.

Focused diagnostics pass **39/39**. The full suite records **1,217 passed**, **47
subtests passed**, the same **2 known pre-existing failures**, and **25 warnings**.
