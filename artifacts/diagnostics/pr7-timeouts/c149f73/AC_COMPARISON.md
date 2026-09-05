# PR7 / PR25: single AC-power / awake analyst comparison

**Result: 5/6 cases were timeout-free, versus 0/6 in the battery replay.**
Three produced valid grounded answers; two retained their PR24 grounding failures.
KB_002 still timed out. Power controls held, but background workload isolation was
imperfect. This is diagnostic evidence, **not a replacement gate or a release
exception**. PR25 remains frozen and blocked.

## Protocol and integrity

- One sequential batch, September 5, 2026, approximately **08:47:31–09:05:52
  America/New_York**. No repeated trials and no full 15-case gate.
- Exact runtime `c149f73d073a306cf34d938955ae6cc739191528`, exact six captured PR7
  packets, original analyst/model code and existing calculator/repair behavior.
- Same model `ollama/qwen2.5:14b-instruct`, digest
  `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`,
  same Ollama 0.33.2 service version, temperature 0, task-dependent output
  limits, five-context limit and **120 seconds per model call**.
- Python 3.11.14 / pytest 9.0.2 in the existing `finsearch-arm` environment.
  No dependencies or product files changed.
- All six packet hashes, initial request-message hashes, tool-schema hashes,
  provider prompt-token counts and per-call settings verified against the frozen
  inputs/runtime. All 18 provider requests were correlated with the observer.
- The five timeout-free cases generated exactly the same per-call token counts
  as PR24: **17/17 completed calls**. This establishes matching output lengths,
  not independently proven identical factual prose.
- The original failed gate and prior battery diagnostic files were preserved.
  Main/PR25 remains at `3179826109af6f47d2a80c36a6cc6ead7401e2c9`, open draft.

## Results

Generation time and throughput below come from completed provider responses and
exclude prompt evaluation/model loading. Throughput is total generated tokens
divided by total generation time across that case's completed calls.

| Case | Model calls | First call, s | Total generation, s | Weighted tokens/s | Final result |
| --- | ---: | ---: | ---: | ---: | --- |
| KB_002 | 1 | 120.0 | Censored | 8.63 partial sample | Timeout |
| KB_003 | 1 | 91.1 | 54.9 | 8.89 | Valid grounded answer |
| HYBRID_002 | 2 | 102.4 | 155.7 | 8.52 | Valid grounded answer |
| HYBRID_003 | 6 | 86.4 | 281.4 | 8.87 | Fail-closed grounding error |
| ANALYST_001 | 7 | 63.4 | 233.5 | 9.38 | Fail-closed grounding error |
| ANALYST_003 | 1 | 71.6 | 37.9 | 10.16 | Valid grounded answer |

KB_002's cancelled response has no completed generation-duration metadata. Its
last server sample was 734 generated tokens at 8.63 tokens/s; this is neither a
final token count nor a completed-output rate. A zero sum of completed generation
durations in the machine-readable record must **not** be interpreted as zero
generation time: that record is explicitly marked censored.

Completed calls ranged from about **8.38 to 10.16 tokens/s**, versus roughly
5–6 in the battery replay. Recovery is substantial but not uniformly 9–10.
Full case elapsed times were 121.2, 91.1, 188.2, 337.9, 290.4 and 71.6 seconds.
Multi-call case totals may exceed 120 seconds without violating the unchanged
per-call limit. The first case's wall time also includes tool startup/control
overhead; its model call itself was cancelled at 120.005 seconds.

HYBRID_003 and ANALYST_001 ended with `ANALYST_GROUNDING_INVALID`, as in PR24.
Neither is counted as a valid answer. No grounding rules were weakened or
post-hoc answer repairs introduced.

## System conditions and limitations

The machine initially had substantial background work after waking. No cases
were started until two workload samples, 20 seconds apart, showed at least 85%
aggregate CPU idle and no sampled process consuming 50% of one core. This was a
diagnostic preflight rule, not a change to product/release thresholds.

The AC profile and active Low Power Mode setting were verified. A temporary
`caffeinate -dims` guard covered the batch. All **49 power samples** (including
preflight/case-boundary/postflight samples) show AC power and Low Power Mode off.
The historical power log contains no sleep events during the run. The temporary
calculator service and keep-awake guard were released normally afterward.

However, **"no other meaningful workloads" was not fully maintained/verified**:

- The 35 during-run workload samples show 69.9–94.8% aggregate CPU idle, median
  84.5%.
- Safari reached 93.6% of one core at 08:54:54 and 87.6% at 09:04:55. A Chrome
  helper reached 53.2% at 08:58:36. These are per-core percentages, not fractions
  of the entire ten-core machine.
- The guard enforced power at case boundaries and recorded workload activity,
  but did not stop user applications or abort on browser bursts. Do not describe
  this as a fully idle/isolated experiment. These bursts are not proven to have
  caused any particular latency outcome, including the earlier KB_002 failure.
- Power/load were sampled rather than continuously profiled. Thermal/GPU-clock
  effects and unsampled activity are not isolated.

Both analyst-only batches began with no loaded model. KB_002 was the cold first
case; its chat runner startup was approximately 6.8 seconds and its cached
prompt-token count was zero. PR24's original gate had a warmer prompt cache for
that case. Additionally, KB_002 is the sole case whose captured PR7 prompt differs
from PR24, by reversing two retrieved content blocks (not by increasing input
size). The current comparison kept the captured PR7 input exact; it did not alter
ordering to improve the result. Cold/cache effects and required answer length
remain possible contributors, not diagnosed sole causes.

## Interpretation / next decision

This materially strengthens the evidence that the failed PR7 gate was affected
by host/model-service performance: five previously timed-out inputs now complete
without a timeout using unchanged code, with normal or near-normal throughput.
No deterministic resolver or analyst product bug has been demonstrated.

It is **not** the clean 6/6 recovery or fully workload-isolated comparison that
would settle attribution. KB_002 remains unresolved. Do not claim that all six
failures were conclusively explained by Low Power Mode, or that the strict gate
passed. No PR24 exception carries forward and no PR7 exception is granted here.

Recommended follow-up, **not executed**: a small, explicitly bounded KB_002-only
diagnostic with browser/background activity controlled more tightly and
cold/warm state recorded. Do not tune runtime, extend 120 seconds, or rerun the
full gate to address this residual diagnostic question. Keep PR25 blocked until
the remaining uncertainty is accepted explicitly or investigated further.

## Added files / commands / checks

Runtime/dependency diff: **empty**. Only the following new, uncommitted local
diagnostic files were added in `codex/pr7-timeout-investigation`:

- `scripts/evals/agents/replay_pr7_ac_packets.py`: power/awake/workload controls
  around the prior, unchanged diagnostic runner; checks outside analyst execution.
- `scripts/evals/agents/summarize_pr7_ac_comparison.py`: exact-input/settings
  assertions, provider timing correlations and diagnostic summary.
- `ac-control-01/{system_samples.jsonl,provenance.json,completion.json}`.
- `analyst-replay-ac-01/{provenance.json,per_case.jsonl,completion.json}`.
- `ac-comparison.json`, `ac-verification.json`, and this report.

The full per-call generation, prefill, load and request durations, statuses,
token counts and provider-log line ranges are in `ac-comparison.json`; raw model
responses are in `analyst-replay-ac-01/per_case.jsonl`. The workload caveat and
17-call token-count comparison are recorded in `ac-verification.json`.

Commands run from the separate frozen checkout, using local path variables:

```sh
PYTHONPATH=src python scripts/evals/agents/replay_pr7_ac_packets.py \
  --per-query "$PR7_EVIDENCE_ROOT/c149f73/per_query.jsonl" \
  --provider-log "$PR7_PROVIDER_LOG" \
  --output-dir artifacts/diagnostics/pr7-timeouts/c149f73/analyst-replay-ac-01 \
  --control-dir artifacts/diagnostics/pr7-timeouts/c149f73/ac-control-01
PYTHONPATH=src python scripts/evals/agents/summarize_pr7_ac_comparison.py \
  --diagnostic-root artifacts/diagnostics/pr7-timeouts/c149f73 \
  --per-query "$PR7_EVIDENCE_ROOT/c149f73/per_query.jsonl" \
  --provider-log "$PR7_PROVIDER_LOG" \
  --output artifacts/diagnostics/pr7-timeouts/c149f73/ac-comparison.json
```

Additional checks: Python syntax compilation; read-only `pmset`, `top`, provider
inventory/version, `git diff/status/rev-parse` and `gh pr view 25`; baseline-file
hash comparison with original Git objects; exact replay hashes/settings and
18 request correlations; sleep-event check; new-file whitespace/privacy checks.
No dependency installation, full regression suite, repeated trial or live gate
was run. These commands document this batch, not authorization for another run.
