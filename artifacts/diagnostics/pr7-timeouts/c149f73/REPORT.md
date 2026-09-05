# PR7 / PR25 timeout investigation

Status: investigation complete; **6/6 isolated replays reproduce the timeouts**.
Evidence strongly supports local inference/host conditions rather than the PR7
resolver extraction. PR25 remains frozen and blocked pending an explicit release
decision; this report is not a replacement gate and grants no exception.

## Scope and isolation

- Runtime under investigation: `c149f73d073a306cf34d938955ae6cc739191528`.
- PR24 evaluated baseline: `7062f48c9d929f2925ffa6820cfd08e18c41ecac`;
  merged PR24: `2860c430036f3fa9e9488663fed9d086a03820af`.
- Separate local branch: `codex/pr7-timeout-investigation`, based on `c149f73`.
- No runtime edits, dependencies, timeout changes, provider tuning, retrieval,
  resolver, or grounding changes. No full 15-case gate rerun.
- Original PR24 and PR25 evidence is read-only. New files live under a separate
  diagnostic directory, outside the SHA-keyed release baseline directories.

## Findings from the original runs

All six PR7 timeouts occurred on the **first analyst model invocation**. Each
trace has two initial messages, no completed tool calls, no grounding attempts,
and no retry timing. `agent_invoke_ms` is 120,011–120,030; the analyst stage is
120,034–120,104 ms. The limit is per model invocation, not the entire analyst
stage: PR24's longer multi-call stages are not timeout-policy violations.

### Inputs did not grow

| Case | Contexts | User prompt characters, both runs | Provider prompt tokens, both runs | PR24 model calls | PR7 model calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| KB_002 | 3 | 7,686 | 2,593 | 1 | 1, timed out |
| KB_003 | 3 | 18,836 | 4,304 | 1 | 1, timed out |
| HYBRID_002 | 4 | 8,606 | 3,029 | 2 | 1, timed out |
| HYBRID_003 | 4 | 11,473 | 3,880 | 6 | 1, timed out |
| ANALYST_001 | 3 | 15,111 | 4,099 | 7 | 1, timed out |
| ANALYST_003 | 3 | 17,547 | 4,111 | 1 | 1, timed out |

The system prompt is also unchanged (1,035 characters). All contexts fit within
the unchanged five-context visibility limit. Full serialized packet sizes and
total serialized context sizes are identical per case across the two runs.

Five user prompts are byte-for-byte identical. Their packets differ only in
`plan_id`, which is not in the analyst prompt. KB_002 additionally reverses the
first two retrieved content blocks (Segments / Segment Operating Performance),
without adding content. Its `ctx_1` / `ctx_2` identifiers therefore refer to
different blocks across runs; matching context-ID lists alone would not establish
evidence identity. This is a KB-only route, not a resolver decision.

SHA-256 checks establish identical analyst implementation, grounding validator,
contracts, model client and capability policy at evaluated PR24, merged PR24 and
`c149f73`. The runtime diff is confined to the resolver extraction and its
orchestrator call sites. The original 170/170 resolver parity and 7/7 captured SEC
execution/argument parity evidence is preserved, not rerun or replaced here.

### Time was spent inside the local model service

The provider log contains one HTTP 500 / `2m0s` cancellation for each failed
analyst call, with prompt processing and generation samples preceding it.
These are not silent orchestration stalls or calculator waits.

| Case | PR24 first provider call | PR24 completed decode, tokens/s | PR7 last decode sample, tokens/s | PR7 generated tokens at last sample |
| --- | ---: | ---: | ---: | ---: |
| KB_002 | 95 s | 9.05 | 5.95 before sleep; 0.12 including sleep | 415 |
| KB_003 | 103 s | 8.73 | 5.59 | 311 |
| HYBRID_002 | 91 s | 10.42 | 6.33 | 418 |
| HYBRID_003 | 79 s | 10.22 | 5.97 | 251 |
| ANALYST_001 | 69 s | 10.20 | 5.54 | 230 |
| ANALYST_003 | 70 s | 10.20 | 5.61 | 328 |

PR7 values are partial-generation samples, not completed-output throughput or
final token counts. They cannot establish what the eventual answer would have
been. Completed PR24 first calls used 691, 488, 655, 431, 304 and 385 generated
tokens respectively. PR24 first-call prompt evaluation took approximately
19, 47, 29, 38, 40 and 33 seconds. PR7 cancelled calls have partial prefill
progress records, not completed prefill-duration metadata; do not invent an
exact prefill/decoding breakdown for them.

The unchanged planner-only control cases also slowed: 23.3 → 41.5 seconds and
29.3 → 52.5 seconds. Neither invokes the resolver or analyst. This supports a
broader inference-service slowdown, not an analyst-only PR7 regression.

PR24 had calculator activity in HYBRID_003 and ANALYST_001; their recorded tool
execution totals were only 53 ms and 65 ms. PR7 never reached those calls. Raw
trace entries include both generated tool calls and tool activity; their counts
are retained in `comparison.json` and are not equated with model invocation
counts. The latter are correlated from provider requests.

### Recorded host/service conditions

- Both manifests identify `ollama/qwen2.5:14b-instruct` with digest
  `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
- The server records the same 49/49 GPU-offloaded chat-model layers and 32,768
  context configuration. Recorded free system memory at initial chat-model load
  was 8.0 GiB for PR24 and 8.8 GiB for PR7; these are snapshots, not continuous
  memory-pressure measurements.
- Before PR7 KB_002, the service evicted/reloaded embedding and chat models due
  to its available-memory prediction. That call began with only five cached
  prompt tokens, versus 810 in PR24; cache warmth was not identical.
- The macOS power log records clamshell sleep at **18:19:02 on September 4**,
  a two-second dark wake at 18:34:57, and lid wake at **19:17:44**. All are local
  America/New_York timestamps, on battery. The timeout completed at 19:17:58.
  This accounts for KB_002's roughly 58.6-minute wall-clock excess; it must not
  be interpreted as 63 minutes of continuous analyst computation.
- A subsequent historical `powerd` query found a concrete configuration change:
  **September 4, 08:36:36**, power source changed to battery and Low Power Mode
  changed from **0 to 1**. This is between PR24's September 3 evening run and
  PR25's September 4 evening run. The transition is preserved in
  `power_history.json`; the earlier current-state-only limitation in
  `support.json` is supplemented by this new historical evidence.
- The replay also ran on battery with Low Power Mode configured on. No power
  settings were changed. No per-call historical thermal, CPU/GPU frequency or
  competing-process samples were captured. Current absence of thermal warnings
  does not prove historical absence. The recorded power transition strengthens
  environmental attribution but is not a controlled experiment proving that
  Low Power Mode alone caused every timeout.

## Diagnostic replay

One sequential batch of only the six captured PR7 analyst packets is recorded in
`analyst-replay-01/`. It uses the exact frozen runtime, model digest, 120-second
limit, temperature, task-dependent output limits, context limit, calculator
runtime and existing retry policy. A pass-through observer records provider-call
timings and returned metadata without changing requests or responses.

This starts from the current machine/service state (initially no loaded models),
not a recreation of the original retrieval/cache history. There is no planner,
retrieval, SEC call, resolver or full-gate evaluation. Results are diagnostic only;
they cannot improve the recorded 8/15 release-gate result.

The batch completed successfully as a diagnostic procedure: **6/6 reproduced
`ANALYST_MODEL_TIMEOUT` on the first call**, with no retry, calculator execution,
or grounding validation. Each underlying HTTP request hit its unchanged read
timeout at 120,004–120,007 ms. The analyst's result remained fail-closed.

| Case | Replay last decode sample, tokens/s | Tokens at last sample | Result |
| --- | ---: | ---: | --- |
| KB_002 | 5.99 | 400 | First-call timeout |
| KB_003 | 5.40 | 302 | First-call timeout |
| HYBRID_002 | 5.81 | 460 | First-call timeout |
| HYBRID_003 | 5.34 | 297 | First-call timeout |
| ANALYST_001 | 5.25 | 280 | First-call timeout |
| ANALYST_003 | 5.15 | 247 | First-call timeout |

All six replay packet hashes, initial model-message hashes, tool-schema hashes,
model-input token counts, per-task output limits and 120-second timeout settings
match the captured inputs / frozen implementation. Six corresponding provider
HTTP 500 cancellations are retained. The temporary calculator service was shut
down normally afterward. `verification.json` records the checks; this does not
mean the six financial answers passed.

## Attribution and release disposition

1. **Proven:** the original six limits were consumed in initial local-model calls,
   with active prefill/decoding. There were no analyst retries or calculator calls
   to blame. Host sleep explains KB_002's additional wall-clock gap.
2. **Strongly supported:** this is an inference/host-performance issue outside
   the PR7 resolver change. Inputs and the analyst stack are unchanged (apart
   from KB_002 retrieval ordering), the model runs substantially more slowly,
   planner-only controls also slowed, and all six timeouts reproduce with no
   resolver, retrieval or SEC execution. A battery/Low Power Mode transition
   between baseline runs is now documented.
3. **Not established:** how much slowdown came from Low Power Mode versus other
   host load, cache, thermal or service effects. No controlled AC-power/awake
   contrast has been performed. No deterministic product bug was identified.
4. **Disposition:** do not change PR25 runtime, raise 120 seconds, or rerun the
   full gate. Its original 8/15 result and separate grounding failure remain.
   No PR24 exception carries forward. Any PR7 exception must be newly and
   explicitly approved; if causal certainty is insufficient, keep it blocked.

Recommended next decision: authorize a controlled **analyst-only** contrast on
AC power, awake, with power mode and service state recorded, before treating
Low Power Mode as the isolated cause. This is an optional follow-up, not executed
here. It needs no grounding/timeout changes and is not a release-gate substitute.

## Evidence and limitations

- `comparison.json`: frozen input sizes/hashes, trace counts, stage timings and
  correlated provider requests with numeric generation samples.
- `support.json`: three-version source hashes, context-source identities,
  path-redacted provider excerpts with original line numbers, and sleep events.
- `analyst-replay-01/`: independent replay provenance, per-case results and
  per-call provider metadata, correlated provider requests and completion state.
- `power_history.json`: historical host power-mode transitions.
- `verification.json`: diagnostic integrity checks, not release scoring.
- Provider calls are correlated by sequential case duration and the manifest's
  completion timestamp, with two-second tolerance; they are not traced by
  request IDs across application and provider. Token counts and durations supply
  additional cross-checks. Six distinct 120-second failures match six trace
  timeouts. Both retained SQLite stores contain only control-case root
  checkpoints, so original analyst message checkpoints are unavailable.
- The relative causal contribution of power mode, contention, thermal state,
  cache and service state is not isolated. Do not turn a recorded configuration
  difference into proof of its sole causal responsibility.

## Local change / verification summary

Runtime diff: **empty**. No existing file or baseline artifact was modified.
PR25 remains an open draft at `3179826109af6f47d2a80c36a6cc6ead7401e2c9`;
its evaluated runtime remains `c149f73`. The investigation checkout remains at
that implementation SHA with new, uncommitted diagnostic files only.

Added files:

- `scripts/evals/agents/investigate_pr7_timeouts.py`: read-only baseline/log comparison.
- `scripts/evals/agents/capture_pr7_timeout_support.py`: source, context and host-log support.
- `scripts/evals/agents/replay_pr7_timeout_packets.py`: narrowly bounded analyst-only replay.
- This directory: `REPORT.md`, `comparison.json`, `support.json`,
  `power_history.json`, `verification.json`.
- `analyst-replay-01/`: `provenance.json`, `per_case.jsonl`,
  `provider_requests.json`, `completion.json`.

Commands/checks performed in addition to the three scripts below:

- `git diff` / `git status` / `git rev-parse` for runtime, dependency files,
  baseline preservation and both checkout identities; `gh pr view 25` read-only.
- Python syntax compilation of all three diagnostic scripts.
- Offline assertions for six historical call mappings, token/packet/prompt
  equality, source hashes, sleep records and a provider-log parser smoke test.
- Exact replay-message/tool-schema/option/hash assertions and six cancellation
  correlations; no full regression suite or live gate was rerun.
- `pmset -g log`, `pmset -g batt`, `pmset -g custom`, `pmset -g therm` and a
  narrowly filtered `log show` query for historical power-mode transitions.
- Read-only local provider version/model inventory and SHA-256 evidence checks.

## Reproduction commands

Run from a separate checkout at `c149f73` using the existing `finsearch-arm`
environment. Supply local paths through arguments; do not put home-directory
paths into repository artifacts. All output paths below must be new.

```sh
PYTHONPATH=src python scripts/evals/agents/investigate_pr7_timeouts.py \
  --evidence-root "$PR7_EVIDENCE_ROOT" --provider-log "$PR7_PROVIDER_LOG" \
  --output artifacts/diagnostics/pr7-timeouts/c149f73/comparison.json
PYTHONPATH=src python scripts/evals/agents/capture_pr7_timeout_support.py \
  --evidence-root "$PR7_EVIDENCE_ROOT" --provider-log "$PR7_PROVIDER_LOG" \
  --output artifacts/diagnostics/pr7-timeouts/c149f73/support.json
PYTHONPATH=src python scripts/evals/agents/replay_pr7_timeout_packets.py \
  --per-query "$PR7_EVIDENCE_ROOT/c149f73/per_query.jsonl" \
  --provider-log "$PR7_PROVIDER_LOG" \
  --output-dir artifacts/diagnostics/pr7-timeouts/c149f73/analyst-replay-01
```

These commands document the completed/authorized diagnostic batch; they are not
authorization to rerun it, tune the release, or run another full gate.
