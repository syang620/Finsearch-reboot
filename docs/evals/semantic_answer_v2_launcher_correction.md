# Semantic v2: separately versioned preflight correction

Status: launcher v2.1 has passed fresh exact-head review and its own run-contract
freeze. No semantic v2 baseline case has started yet. The user explicitly authorized
this evaluation-only correction on 2026-09-08; it is not production tuning.

## Diagnosis and preserved record

The original optimization freeze remains at implementation
`31797803dc0ef262068fa55a8a76745d15ac758f`, manifest SHA-256
`653a65e778a5633c4d1adc2d90ea7a56a578b81695e69d679f9d69915e54bec0`.
Its launcher, dataset, source/scorer hashes, judge validation, disabled-judge
decision and quality approval are unchanged.

At 12:20 UTC on 2026-09-08 the original launcher failed its first workload check
after importing retrieval libraries. A second preflight imported the libraries
before a 30-second idle period. Its 12:22:54 UTC check recorded AC power, Low
Power Mode off, zero browsers and no heavy process. At 12:23:01 UTC it failed
the second check, immediately after index verification; the failure snapshot
reported only Python at 89.4% CPU. No SHA-keyed output/cache directory or
`started.json` was created. No benchmark query or analyst call occurred. The
second preflight did perform the existing non-benchmark service-health probes.
These are tool-observed diagnostic observations, not a completed baseline or
continuous process trace. The workstation was quiet again after process exit.

The original guard observes CPU by process name, including the evaluator itself.
CPU-intensive setup can therefore trip the competing-workload check. Neither
removing Python from that check nor weakening its 50% threshold is justified.

## Minimal behavior difference

`run_semantic_baseline_v2_1.py` adds a fixed, bounded 30-second settling period
after imports/configuration and another after index verification/planner setup.
It then applies the original strict check, including Python and all external
processes. Each before/after observation and elapsed delay is recorded. A busy
post-settling check still fails. These periods occur before any benchmark case;
there are no case-level delays or retries. Awake protection includes preflight.

The seeded schedule, retrieval resources, model/provider configuration, timeout,
retry policy, analyst inputs, case timing, capture, scoring, per-case workload
checks and final validity decision are unchanged. An AST regression verifies
the original case loop and finalization, apart from moving awake cleanup around
the whole launch. Existing helpers are imported unchanged, not reimplemented.

The original launcher and frozen manifests are never edited. The new launcher
requires an additional clean exact-head review and committed, hash-bound record
at `data/evals/semantic_answer/launcher_v2_1/manifest.json`. It verifies all original
frozen files and permits only this separately versioned launcher/contract in the
evaluation source/data diff. The new contract does not unfreeze v2 labels,
membership, scoring or judge policy. The original single-pass guard remains:
any existing baseline `started.json` blocks another run, irrespective of a new
documentation SHA. This correction does not authorize selective reruns.

## Remaining

The first launcher review on `0e3f09b` found one compatibility omission: the
original optional `--env-file` argument. It is restored with the same
`load_dotenv(..., override=False)` behavior before baseline execution; inherited
credentials remain supported. Two new regressions cover both launch modes.

Previous checkpoint verification: 39 launcher/baseline-control tests passed
(20 new launcher tests). Full suite: 1,159 passed, 47 subtests passed, the same two pre-existing failures
(planner alias route and retrieval attempt count), and 25 unchanged warnings.
Original frozen file hashes, production/data diffs, privacy and whitespace checks
pass. No dependency changes.

After the credential-option fix: 41 launcher/control tests pass (22 new);
full suite 1,161 passed, 47 subtests passed, the same two failures and 25 warnings.
The read-only setup diagnostic at `0e3f09b` reproduced Python at 90.4% CPU after
index inspection, then no heavy process after the fixed 30-second wait. No
benchmark questions were executed; this is preflight evidence only.

The [clean launcher review](https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-5585393657)
inspected exact `f16f96d7b416a195b739cadc87bfe0b1c9a63b3b`. The separate launcher
contract is frozen with SHA-256
`ce1646e554cdb3f8fb583db0ea93eeb278daa13a937c641320335c33c24845ad`.
The diagnostic is preserved under
`artifacts/evals/semantic_answer/v2/preflight_diagnostics/0e3f09b09d6b95d610a96e9114f8046bb9027465/`,
with its own file hash manifest. It is not baseline performance evidence.

Remaining: one controlled unchanged-system baseline; fixed
30-case source audit; immutable reporting; fresh final review. Do not merge
PR31 automatically or tune production behavior.
