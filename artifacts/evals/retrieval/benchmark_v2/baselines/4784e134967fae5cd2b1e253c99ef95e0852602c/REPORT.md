# PR27 review follow-up — interrupted measurement

- Evaluated SHA: `4784e134967fae5cd2b1e253c99ef95e0852602c`.
- Unchanged dataset freeze: `0cc20ce37c42d18412cfb518f8b06f775d07257b`.
- Source change: offline float comparison tolerance only, with regressions and
  documentation. No ranking, scoring-formula, retrieval or configuration change.
- Focused tests: 67 passed. Full suite: 872 passed, 47 subtests passed, two
  existing failures (planner alias_002 and retrieval no-tool-call attempt count),
  one existing table-renderer warning. pytest remains 9.0.2, environment-only.
- Existing `2d50cfe` baseline verifies with the new verifier, without edits.

## Terminal state

**Incomplete: 90/480 query-mode pairs. Do not use as a full baseline.**

The first 89 pairs completed under the power/browser controls. Pair 90,
`KBV2_AAPL_2025_03` / `hybrid_reranker`, began with zero browser processes; its
after-call sample recorded five. The runner stopped with
`Power/browser controls changed during retrieval; preserve partial run`.
The final sample recorded 15 browser processes, including busy Chrome renderers.
AC power stayed connected and Low Power Mode stayed off. There were no retrieval
errors in the completed pairs; the fatal condition was workload-control loss.
Both index fingerprints and the embedding model digest were unchanged.

Raw rows, errors, partial summary and provenance are retained. Their metric
arithmetic and hashes can be verified, but `complete` must remain false. There
is no interpolation, silent resumption, outcome-based selection, overwrite, or
claim that this attempt replaces the original complete baseline. A subsequent
attempt requires a clean new freeze/output identity and restored controls.

## Read-only verification

```text
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/verify_benchmark_v2.py --baseline artifacts/evals/retrieval/benchmark_v2/baselines/4784e134967fae5cd2b1e253c99ef95e0852602c
```

The expected verifier result is 90 pairs, `complete: false`, zero missing
labels, and verified hashes/metrics. This report does not authorize merge.
