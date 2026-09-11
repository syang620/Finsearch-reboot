# Semantic v2 secondary-judge validation

**Decision: full-benchmark automated judging is disabled.** The one preregistered
candidate failed the frozen acceptance criteria. No candidate search, rubric
changes, parser repairs or retries followed these results. This is a judge
calibration result, not a production answer-quality score.

## Frozen identity and execution

- Reviewed construction: `d2abf079837809148d5a43050d957a61e4195783`.
- Evaluated implementation: `100ff6fc082cb28e971abb9c7a5ae84b3e0af79b`.
- Validation manifest SHA-256: `24da18f62d51c256670d25dd8018482d3dc579eea4112afcb2209559a6d7ebfc`.
- Fixture SHA-256: `4e42a4c8491057f305ca5c398b744fe1c1dd96eff8fa26a37bd4a8e16049e9e3`.
- Local Ollama `0.33.2`, `gemma4:e4b`, digest `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`.
- Temperature 0, context 32,768, prediction cap 4,096, thinking off, JSON output,
  240-second phase timeout, one attempt per phase. No production model calls.
- UTC 2026-09-08 04:14:01–04:31:46; AC power, Low Power Mode off, awake protection.
  Before/after power and process snapshots are preserved; these are endpoint
  observations, not proof of continuous workload isolation.
- 36 primary source-authored synthetic answers and 12 predetermined repeats,
  two phases each: 96 total calls. The primary gold contains 39 emitted claims
  and 65 required facets. This is not a sampled production-error prevalence set
  or independent human financial annotation.

## Preregistered gates

| Gate | Observed | Minimum | Result |
| --- | --- | --- | --- |
| Complete valid primary assessments | 29/36 (80.56%) | 95% | Fail |
| Claim-label agreement | 22/39 (56.41%) | 85% | Fail |
| Fully-supported precision | 17/29 (58.62%) | 90% | Fail |
| Fully-supported recall | 17/19 (89.47%) | 85% | Pass |
| Unsupported recall | 4/14 (28.57%) | 90% | Fail |
| Partial-support agreement | 1/6 (16.67%) | 80% | Fail |
| Whole-answer groundedness agreement | 14/36 (38.89%) | 90% | Fail |
| Whole-answer completeness agreement | 28/36 (77.78%) | 85% | Fail |
| Repeat agreement | 11/12 (91.67%) | 90% | Pass |
| Required-facet agreement | 42/65 (64.62%) | 85% | Fail |
| Partial-fulfillment recall | 0/3 | 80% | Fail |
| Off-topic recall | 1/2 | 90% | Fail |
| Unbound-prose recall | 0/2 | 90% | Fail |
| Generic cross-filing acceptance | 1/1 | 100% | Pass |
| Named-filing rejection | 0/1 | 100% | Fail |

Seven primary answer pairs and one repeat pair are invalid. Across the 96 phase
calls, eight errors comprise: four missing/unexpected-schema-key errors, one
claim-ID-coverage error, one invented/non-cited-evidence quotation and two
invented-answer/evidence-only-quotation errors. An invalid phase makes the paired
assessment unknown; valid content in the other phase is not silently salvaged.
Missing/invalid assessments remain in the fixed gold denominators. A pair of
invalid judgments does not count as repeat agreement. All eleven valid repeat pairs agree, but repeated
incorrect judgments do not establish semantic correctness.

Among 35 claims in valid primary assessments, the judge assigned 29 fully
supported, five unsupported and one partially supported labels. Only 17 of the
29 fully-supported predictions match the source labels. The problem is therefore
not merely JSON formatting; evidence-support judgment is insufficiently reliable.
This result says nothing about the quality of untested judge candidates.

## Evidence and reproduction

Raw packets/responses, individual errors, assessments, metrics, identities,
completion state and file hashes are immutable under
`artifacts/evals/semantic_answer/v2/judge_validation/100ff6fc082cb28e971abb9c7a5ae84b3e0af79b/`.
The artifact manifest SHA-256 is `66a3efc3ee943d7f1b3caeb1e05206ed6114a29ec4fa031f4c5e1a15ced643bf`;
the metrics SHA-256 is `ae6a613f781eb36de6ad47c533556cff65e0d0445d455b951c3cfe8aa1d0d3b0`.
The recorded decision is `data/evals/semantic_answer/v2/judge_decision.json`.
Reproduction verifies frozen inputs/code, raw artifact hashes, exact supplied
packets, parsers, assessments and gate arithmetic without another model call.

```sh
PYTHONPATH=src:. python scripts/evals/agents/freeze_semantic_validation_v2.py
PYTHONPATH=src:. python scripts/evals/agents/run_semantic_judge_validation_v2.py --out-root artifacts/evals/semantic_answer/v2/judge_validation
PYTHONPATH=src:. python scripts/evals/agents/freeze_semantic_benchmark_v2.py decision --validation artifacts/evals/semantic_answer/v2/judge_validation/100ff6fc082cb28e971abb9c7a5ae84b3e0af79b
```

These are the completed commands, not instructions to repeat the experiment.
The runners refuse an existing attempt. No local user paths or credentials were
found in the evidence privacy check.

## Consequence for the system baseline

After benchmark-quality review and optimization freeze, run the unchanged system
once on all 60 cases. Publish all-case execution and bounded deterministic
numeric/structural metrics. Broader support/completeness/groundedness metrics are
restricted to the preregistered 30-case source-audited subset, with eligible,
assessed and unknown denominators. No unattended full-corpus semantic score is
authorized by this calibration. No v1-to-v2 system-improvement claim is valid.
