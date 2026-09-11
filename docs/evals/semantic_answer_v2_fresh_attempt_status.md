# Fresh semantic-v2 attempt: stopped and preserved

Status: **invalid diagnostic; PR31 remains blocked**. The one newly authorized
attempt was consumed, ran once, and stopped under its preregistered operational
failure rule. It is not a completed controlled baseline and is not “baseline 1.”
No further attempt, merge, or joint optimization-readiness declaration is authorized
under the exhausted permission. The fixed 30-case source audit remains pending.

## Frozen implementation and observations

Clean evaluated implementation: `488e112a64b51fb2a5ad159194df993b2ff04f11`.
The external permission wrapper received clean review at `26993ea6f6`; its separate
approval commit changed no benchmark, scorer, launcher, runtime, model, timeout,
retry, workload threshold or judge policy. Both strict preflight checks passed.
Capture began at 14:14:08 UTC on 2026-09-08, on AC with Low Power Mode off, no
browser processes and no heavy non-model processes at those checkpoints.

| Captured case | Primary execution outcome | Wall time |
| --- | --- | --- |
| SEM2_MSFT_2024_05 | Grounding fail-closed; ineligible | 543.6 s |
| SEM2_MSFT_2025_04 | Analyst timeout; ineligible | 306.5 s |

After case 2, at 14:28:18 UTC, the frozen control snapshot recorded one browser
process and `git` at 59.8% CPU, exceeding the unchanged 50% threshold. AC and
Low Power Mode settings were still correct. This observed violation invalidates
the controlled run; it does **not** prove who launched those processes, how long
they ran, or whether they caused the timeout. The later quiet cleanup snapshot
does not erase the violation.

The frozen loop had started case 3, `SEM2_AMZN_2023_01`, before the violation was
observed through monitoring. The already approved stop rule was applied by
interrupting the verified child launcher. Two final outcomes were captured, one
case was interrupted without a final output, and 57 were unstarted. The 58
uncaptured cases are not semantic failures or inferred timeouts. Neither captured
output is eligible for semantic credit; retained rejected prose is not an answer.

Finalization began at 14:29:01 UTC (the runner assigns `finished_at` on entering
its cleanup block). Cleanup and verification subsequently succeeded with unchanged
model identities and both index snapshots, and no scoring or runtime-cleanup
errors. Their exact completion time is not separately recorded; they finished
before the wrapper's outcome at 14:29:11 UTC. Child return code `-2` records SIGINT; the outer shell
reported 254. `completion.json` explicitly records `invalid_diagnostic`, incomplete
capture/evaluation and a control violation. No official deterministic summary was
created. No full-corpus judge or source-subset assessment was run.

## Immutable publication and verification

All five runner files are copied byte-for-byte, with complete file-set equality,
to `artifacts/evals/semantic_answer/v2/controlled_baselines/488e112a64b51fb2a5ad159194df993b2ff04f11/`.
That directory name identifies the workflow, not eligibility. Manifest SHA-256:
`f5b579cf2457a5a9a574fc9a88ad1e77fec9a493f210b6df74eea638dc256fa0`.
Original staging is retained; local console logs remain private.

Consumption/outcome records and a separate disposition are preserved under
`artifacts/evals/semantic_answer/v2/attempt_authorizations/SEMANTIC-V2-FRESH-20260908/488e112a64b51fb2a5ad159194df993b2ff04f11/`.
The durable marker remains consumed and must not be deleted or reset. The earlier
`29bfec8` diagnostic and every frozen benchmark/scorer/launcher file remain unchanged.

Both captured deterministic rows reproduce exactly from the original raw outputs
with the frozen scorer. All three numeric requirements remain unassessed with no
positive credit. File-set/hash, historical immutability, privacy and diff checks
pass. Prelaunch verification on the reviewed wrapper had 1,178 passing tests,
47 passing subtests, the same two pre-existing failures and 25 warnings. These
publication/report additions do not change behavior.

## Claims and continuation

This evidence supports describing controlled-run enforcement and honest failure
preservation, not answer-quality, latency, timeout-rate or before/after gains.
Do not extrapolate either partial diagnostic to 60 cases or combine them into a
baseline. Judge calibration remains failed and full-corpus judging disabled.

A valid full 60-case capture, the predetermined 30-case source audit, denominator
checks and fresh final review are still required before PR31 can merge. Another
attempt or any investigation needs new direction; this record grants neither a
retry nor a relaxed control. Retrieval v3 and semantic v2 stay unchanged, and no
joint optimization-readiness declaration is made.
