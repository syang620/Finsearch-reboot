# Semantic v2 baseline: stopped diagnostic, not a release baseline

Status: **incomplete and invalid for controlled performance claims**. The user
requested stopping and preserving the run after a workload violation. No new
attempt is authorized. PR31 remains unmerged; the full benchmark task is not
complete, and there is no current-system semantic performance baseline to cite.

## Frozen implementation and observations

The one attempt used `29bfec8a662efdd1dd78fe986eb753d655315646`, the original
frozen v2 dataset/scorer/disabled-judge policy plus the separately reviewed v2.1
preflight launcher. Both strict preflight checks passed after their fixed
30-second settling periods. No production behavior, model, timeout, retry,
retrieval setting, label, metric rule or workload threshold was changed.

The run started at 12:52:34 UTC on 2026-09-08. It captured:

| Case | Primary execution outcome | Captured wall time |
| --- | --- | --- |
| SEM2_MSFT_2024_05 | Analyst timeout; ineligible for semantic credit | 429.1 s |
| SEM2_MSFT_2025_04 | Analyst timeout; ineligible for semantic credit | 314.5 s |

At the start of the second case, the workload snapshot recorded ChatGPT at
59.1% CPU, above the frozen 50% heavy-process threshold. This makes the attempt
diagnostic-only. AC power, Low Power Mode off and zero browsers were recorded
at those case boundaries. Do not infer that the CPU burst caused either timeout;
no causal comparison was performed.

The user selected **stop and preserve now**. A graceful interrupt stopped the
third case, `SEM2_AMZN_2023_01`, during retrieval without a final output. Thus two
cases have captured outcomes, one was interrupted, and 57 were never started.
The remaining 58 cases have no captured outcome; they are not semantic errors,
timeouts, refusals or successful answers. The two captured timeouts likewise
remain execution losses, not semantically incorrect claims.

Cleanup wrote `completion.json` at 13:07:44 UTC and the process exited with code
130. It records `invalid_diagnostic`, incomplete capture/evaluation, and the
workload violation. Model identities and both index snapshots were unchanged;
there were no scoring or runtime-cleanup errors. No official deterministic
summary was produced. The fixed 30-case source audit was not completed; the
disabled secondary judge was not called on this attempt.

## Immutable evidence and checks

The five original runner files are preserved unchanged under
`artifacts/evals/semantic_answer/v2/baselines/29bfec8a662efdd1dd78fe986eb753d655315646/`.
The directory name identifies the attempted baseline workflow, not eligibility.
Its hash manifest SHA-256 is
`dea0d285930472c36f64e3f90163db6f2ec766d9e2eb50b425b581d09d42ab26`.
The separate `diagnostic_disposition/29bfec8a662efdd1dd78fe986eb753d655315646/`
record explains the user stop without modifying the original completion record
or raw artifact manifest.

All captured hashes verify. Both deterministic rows reproduce exactly from raw
outputs with the frozen scorer, and neither receives positive numeric credit.
Historical/dataset hashes and privacy checks pass. The preflight diagnostic,
judge calibration and prior historical evidence remain unchanged. The reviewed
source had 1,161 passing tests, 47 passing subtests, the same two pre-existing
failures and 25 warnings; these evidence-only additions do not change behavior.

## Claim and continuation boundary

This record supports describing the benchmark construction, strict validation,
versioning, failed-judge decision and honest diagnostic handling. It supports
**no controlled answer-quality, full-benchmark timeout, semantic correctness,
retrieval gain or latency performance claim**. Do not extrapolate two captured
cases to 60 or compare this partial run as an optimization baseline.

Do not rerun or relax the workload rule automatically. Any later attempt needs
explicit user authorization and a separately reviewed execution plan that
preserves this failed attempt and the frozen evaluation contract. The existing
single-pass launcher correctly refuses to start again while this attempt exists.
Fresh review of this diagnostic disposition is the remaining preservation step;
it would not make the incomplete benchmark baseline complete or authorize merge.
