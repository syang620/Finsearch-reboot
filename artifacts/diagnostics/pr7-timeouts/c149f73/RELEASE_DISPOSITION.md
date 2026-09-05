# PR25 / PR7 owner-approved release disposition

Approved September 5, 2026. After being asked explicitly to approve both PR7
exceptions and commit/push/merge, the project owner replied:

> Approve. please proceed with commit, push, and merge.

This approval applies only to PR25 and unchanged evaluated implementation
`c149f73d073a306cf34d938955ae6cc739191528`. Diagnostic evidence was committed
at `59d3ac0ca6bedbbfb33f01d72c4f051cdaa0e39a`. It authorizes merging despite
the preserved failed gate, not relabeling the gate as passed.

## PR7-TIMEOUT-001

The owner accepts the six original unavailable analyst answers caused by the
120-second model-call limit in the original live measurement:

- `AGENT_V1_KB_002`
- `AGENT_V1_KB_003`
- `AGENT_V1_HYBRID_002`
- `AGENT_V1_HYBRID_003`
- `AGENT_V1_ANALYST_001`
- `AGENT_V1_ANALYST_003`

The [initial investigation](REPORT.md) established first-call timeouts, unchanged
analyst/model-client code, no input growth, substantially reduced generation
throughput, recorded host sleep, and an intervening battery/Low Power Mode
transition. Five prompts were byte-identical to PR24; KB_002 differed only in
retrieved-block order and remained a KB-only route.

Three separate diagnostic experiments are preserved:

| Experiment | Latency outcome | Qualification |
| --- | --- | --- |
| Captured six-case battery replay | 0/6 timeout-free | No resolver, retrieval or SEC execution |
| Captured six-case AC/awake replay | 5/6 timeout-free | Browser CPU bursts; not fully workload-isolated |
| Single isolated KB_002 replay | 1/1 timeout-free, valid answer | Browsers closed; AC/awake/LPM-off controls held |

The final isolated case took 109.629 seconds for the model call, generated 844
tokens at 10.613 tokens/s, and needed no retry. Input, tools, model digest,
settings and 120-second limit were unchanged. These results provide the
owner-accepted evidence for environmental contamination of the original timeout
measurement. They are **not** a new 6/6 or 15/15 release-gate run.

Residual risk: KB_002 retains roughly ten seconds of margin under 120 seconds;
performance remains sensitive to host load, cold/cache state and answer length.
The experiments do not isolate Low Power Mode as the sole cause, establish a
production latency guarantee, or prove the absence of all unsampled contention.
No timeout increase, provider adjustment or product change is approved here.

The AC diagnostic's HYBRID_003 and ANALYST_001 results were timeout-free but
still failed grounding, as in PR24. They are not described as successful
grounded answers and are not hidden by the latency summary.

## PR7-GROUNDING-001

The owner separately accepts the original gate's non-timeout failure:

- `AGENT_V1_HYBRID_001`: `ANALYST_GROUNDING_INVALID`, including
  `GROUNDING_EVIDENCE_TYPE_MISMATCH` and `GROUNDING_ROW_TEXT_MISMATCH`.

This is the same known failure class present in PR24, with byte-identical analyst
and grounding code. The runtime continues to reject the answer. No factual
correctness, successful answer, validator relaxation or deterministic prose
rewrite is claimed or authorized. This is a **new PR7-only owner decision**;
`PR24-GROUNDING-001` does not carry forward automatically.

## Preserved gate, invariants and next boundary

The original strict live gate remains **failed: 8/15 non-critical cases, seven
critical cases, score 0.859390, zero evaluator errors**. Its raw files under
`artifacts/evals/agents/v1/baselines/c149f73/` remain byte-for-byte unchanged.
The owner accepts the original answer-availability/quality risk for this merge.

The 170/170 resolver parity, 7/7 captured SEC execution/argument parity, unchanged
runtime's clean review, and measured offline results remain as recorded. There
were no resolver/routing/tool-argument mismatches to waive. Known full-suite
failures and semantic-quality limitations remain documented. Neither exception
is a blanket waiver for other defects or future changes.

Release changes consist only of diagnostic tools/evidence and this approval
documentation. Historical diagnostic `exception_granted: false` and blocked
statements remain intact as contemporaneous records; this disposition supersedes
their release status only. No new runtime freeze, full gate, or behavior tuning
was performed to close PR7.

After merge, PR8 planning starts from merged PR25. These exceptions do not apply
to PR8, authorize PR8 implementation, or silently change its release rules.
Filing/amendment semantics remain a separate planned scope.
