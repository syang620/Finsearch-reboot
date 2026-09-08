# Workload-control v2 calibration report

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED`**

Selected policy: **`B_CONSECUTIVE_10`**

The selected rule requires the same scored stable process group to reach at least
50% aggregate CPU for ten consecutive one-second samples. AC power, Low Power Mode
off, awake protection, actual-browser absence, and frozen service/model/index
identity remain independent hard requirements.

## Results

- Clean supervised idle: 0 B10 CPU episodes in 900 samples; maximum group streak 9.
- Clean terminal-only idle: 0 episodes in 900 samples; maximum streak 1.
- Required read-only service activity: 0 episodes in 180 samples; maximum streak 3.
- Sustained interference: 3/3 detections at 9.9996–9.9999 seconds.
- Six 1.5-second bursts: 0 episodes; maximum streak 4.
- Real Chrome workload: hard rule detected in the first sampled interval.
- Identity retention: 100% classified or retained as unknown.

The complete machine-readable matrix, candidate acceptance, latency results,
unknown-process contribution, terminal assessment, and historical replay are in
`comparison.json`. The preregistered instantaneous, persistence, occupancy, and
burden candidates are all reported, including poor results.

## Integrity and limitations

The preregistration hash is
`0c6c4ac93cee1b23e33ea2ea93ded0a9b56092cd9898045b44a0bb7595463483`.
Raw observations are deterministic-gzip files and are bound by `files_sha256.json`.
No semantic case, retrieval, or inference ran.

Publication replaces the user-home prefix and email addresses embedded in process
commands with `$USER_HOME` and `$EMAIL`. This pre-freeze privacy pass changed two
raw-file hashes but no measured or classified field used by any candidate policy.
The comparison was regenerated from the sanitized evidence, and the superseded
pre-publication comparison remains preserved as non-release diagnostic history.

An initial analyzer incorrectly credited a policy already active before deliberate
load as a zero-second detection. That output is preserved as invalid diagnostic
analysis. The frozen corrected analyzer requires an attributable false-to-true
transition. It did not change B10's selection.

The 33 clean minutes do not establish an hour-scale or multi-hour false-trigger
rate. The nine-sample supervised streak leaves a one-sample margin. Terminal-only
monitoring is viable with procedural limitations; semantic execution was not
tested. Historical statuses are unchanged and no timeout-causality claim is made.

PR31 remains frozen, draft, release-blocked, and unmerged. This report does not
authorize a semantic baseline until a separately reviewed v2 launcher integration
records the frozen contract hash.

## Verification

The focused diagnostics suite passed **22/22**. The full repository suite, using
its documented importlib collection mode, produced **1,200 passed**, **47 subtests
passed**, **2 failed**, and **25 warnings**. Both failures are the unchanged known
failures recorded before this work: `alias_recognition/alias_002` routes to KB
rather than its expected structured route, and the retrieval no-tool-call test
observes one attempt rather than two. Default pytest import mode also reproduced
the existing installed-`tests` namespace collision during collection; it was not
counted as a test result.
