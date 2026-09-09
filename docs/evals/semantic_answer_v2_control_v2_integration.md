# Semantic-v2 workload-control opt-in integration

Status: **candidate — no semantic case execution authorized**

This separately versioned adapter is the narrow bridge from the reviewed PR33
calibration to PR31. It leaves the PR31 benchmark, gold, deterministic scorer,
runtime, models, prompts, retries, 120-second timeout, case order, original v2
launcher, v2.1 launcher, and all prior diagnostics unchanged.

The new launcher requires both of the following explicit inputs:

- `--workload-control-v2 B_CONSECUTIVE_10`
- the GitHub comment ID of a clean Codex review of the exact integration head

It refuses any other policy and verifies the frozen PR33 preregistration,
contract, classifier, and reviewed ancestry by SHA-256. The original PR31 quality
approval and v2.1 launcher approval remain required independently.

After the second fixed preflight-settling boundary, the adapter samples continuously
at one-second cadence until the frozen semantic launcher has closed its normal
artifacts. Each raw sample retains power state, exact-browser evidence, awake
state, process identity/classification evidence, the legacy instantaneous signal,
and the selected B10 decision. AC loss, Low Power Mode, a real Chrome/Safari
browser, or an active ChatGPT/Codex supervision process is an immediate hard
failure. As in the reviewed terminal-only calibration, zero-CPU launchd-owned
crash handlers are retained and disclosed but do not violate terminal-only mode.
CPU invalidation requires the same scored
stable group at or above 50% for ten consecutive samples. Unknown processes stay
visible and scored. Model, service, and index identity remain the frozen launcher's
existing fail-closed checks.

The frozen launcher still executes and scores every captured case. Its append-only
`completion.json` is deliberately closed as `pending_workload_control_v2` and can
never claim eligibility by itself. After the monitor closes, the adapter writes a
separate authoritative `workload_control_v2_completion.json` and hash manifest;
it never overwrites the frozen launcher's artifacts. The adapter only replaces
the operational eligibility decision: the old instantaneous CPU
observations remain in the case records but no longer invalidate the run. A hard
failure, B10 trigger, cadence failure, monitor error, incomplete capture/scoring,
or failed model/index/cleanup verification makes the attempt `invalid_diagnostic`.
Raw control evidence and the trigger are preserved. No result is relabeled after
the fact, and this integration does not retroactively validate either prior
diagnostic.

This PR is infrastructure validation only. It must receive a clean exact-head
review before a separately explicit authorization may allocate one terminal-only
60-case attempt. That future authorization must use a new durable single-attempt
wrapper/marker; the exhausted September 8 permission is never reset or reused.
