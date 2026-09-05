# KB_002: one isolated AC/awake follow-up

**Succeeded: one model call, valid grounded answer, no retry, no timeout.**
The frozen captured PR7 input completed at normal throughput under the original
120-second limit. This meets the user's stated evidence criterion for considering
a narrowly scoped PR7 **timeout** exception. No exception has been granted or
published by this diagnostic run, and no merge or full gate was performed.

## Measurement

| Measurement | Result |
| --- | ---: |
| Model-call wall time | 109.629 s |
| Entire case wall time, including setup | 110.542 s |
| Provider-reported total | 109.625 s |
| Model loading | 6.865 s |
| Prompt evaluation | 23.228 s / 2,593 tokens |
| Generation | 79.524 s / 844 tokens |
| Completed generation throughput | **10.613 tokens/s** |
| Model calls / retries | **1 / 0** |
| Analyst result | **ok**, valid grounded answer |

The outcome is an actual completed response, not a projected completion from a
partial generation sample. No output/schema/grounding rewrites were introduced.

For context, the prior battery replay timed out near 6 tokens/s, and the prior
AC batch timed out with a last partial sample of 734 tokens at 8.63 tokens/s.
This complete answer is longer than the PR24 answer (844 versus 691 generated
tokens); the captured PR7 input has a different ordering of two content blocks
than PR24, as documented in the original investigation. This follow-up preserved
the PR7 input exactly rather than altering that ordering.

Illustrative timing only: 844 tokens at 8.63 tokens/s would take about 97.8 seconds
to generate; adding this run's roughly 30.1 seconds of other provider work gives
about 127.9 seconds. That explains why this case has limited latency margin at
the slower rate, without claiming identical historical overhead or a completed
historical answer that was never returned. No timeout increase is proposed.

## Controls and verification

- Runtime remained `c149f73d073a306cf34d938955ae6cc739191528`. No product,
  requirements, lockfile, model/client, provider, prompt or retry-policy edits.
- The original KB_002 JSONL row was copied verbatim into `captured_case.jsonl`.
  Packet hash, initial request messages, tool schema, 2,593 provider prompt tokens,
  temperature 0, output limit 2,048 and timeout 120 s all verified exactly.
- Same model `ollama/qwen2.5:14b-instruct`, digest
  `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`;
  same Ollama 0.33.2 and existing `finsearch-arm` environment. Python/pytest
  versions are retained in `run/provenance.json`.
- Safari and Chrome were quit normally, without force quitting or deleting their
  profiles. Browser-process absence was verified before and throughout sampling.
  No other heavy application required termination; no system services were stopped.
- Two preflight samples showed 90.95% and 94.24% aggregate CPU idle. All 13 power/
  browser samples held AC power, active Low Power Mode off and browser absence.
- Ten during-run workload samples showed 84.6–93.3% aggregate CPU idle (median
  89.73%). There were **zero sampled non-model processes at or above 50% of one
  CPU core**, unlike the previous AC batch's browser bursts. No control violation
  occurred. Sampling is not continuous profiling and does not isolate every
  possible thermal/GPU effect.
- A temporary `caffeinate -dims` guard kept the machine awake. The guard and
  temporary calculator service were released normally after the run. Browsers
  remain closed; no automatic restart of user applications was performed.
- The original observer and analyst runner were reused unchanged. Only the
  diagnostic case list was bound to KB_002 and fed the verbatim singleton row;
  the original analyst implementation executed unchanged.

## Release disposition

The sequence is now:

- Battery diagnostic: 0/6 timeout-free.
- Prior AC/awake diagnostic with imperfect workload isolation: 5/6 timeout-free.
- This separate, browser-closed KB_002 diagnostic: **1/1 timeout-free and valid**.

These are separate diagnostic experiments, **not a newly passed six-case batch
or 15-case release gate**. They provide strong evidence for environmental
contamination of the original timeout results and support a narrowly documented
PR7 timeout exception under the user's decision rule.

The original PR7 gate remains **8/15 and failed**, preserved exactly. Its
non-timeout grounding failure is not covered by this timeout investigation or an
implied exception. PR25 remains an unchanged open draft at
`3179826109af6f47d2a80c36a6cc6ead7401e2c9`, pending an explicit release decision.
No PR24 exception carries forward automatically.

## Diff, files and commands

Runtime/dependency diff: **empty**. Added one diagnostic script and nine local
evidence/report files, uncommitted on `codex/pr7-timeout-investigation`:

- `scripts/evals/agents/replay_pr7_kb002_isolated.py`.
- This directory: `captured_case.jsonl`, `system_samples.jsonl`,
  `control_provenance.json`, `control_completion.json`, `verification.json`,
  `REPORT.md`.
- `run/provenance.json`, `run/per_case.jsonl`, `run/completion.json`.

The original gate and earlier diagnostic artifacts were not edited or replaced.
Full per-call data, result and raw model response are in `run/per_case.jsonl`;
`verification.json` records timing, exact-input checks, control results and
correlated provider-log line ranges.

The one replay command, using local path variables from the separate checkout:

```sh
PYTHONPATH=src python scripts/evals/agents/replay_pr7_kb002_isolated.py \
  --per-query "$PR7_EVIDENCE_ROOT/c149f73/per_query.jsonl" \
  --provider-log "$PR7_PROVIDER_LOG" \
  --output-dir artifacts/diagnostics/pr7-timeouts/c149f73/kb002-isolated-01
```

Other commands/checks: normal AppleScript quit requests for Safari/Chrome;
read-only `pmset`, `ps`, `top` and provider inventory; source/packet/message/tool
schema hash assertions; provider request correlation; Python syntax compilation;
new-file whitespace/privacy checks; `git diff/status/rev-parse` and read-only
`gh pr view 25`. No dependencies, full regression suite, repeated trial, full live
gate, exception publication or merge was performed.
