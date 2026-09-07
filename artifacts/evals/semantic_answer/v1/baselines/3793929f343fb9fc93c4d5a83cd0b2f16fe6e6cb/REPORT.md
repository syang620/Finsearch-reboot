# Semantic answer benchmark v1: observed baseline

All 60 frozen questions were run once through the unchanged post-PR8 system. This is a **service/environment-affected observational baseline**, not a clean release gate or a controlled performance comparison. No runtime, prompt, retrieval, calculator, grounding, provider or retry tuning was performed.

The system delivered 12 substantive `ok` answers and 9 `insufficient_data` responses. The other 39 cases failed or stopped for clarification. Structural citation checks passed on all 20 emitted claims, but the source audit found unsupported assertions. **Neither these structural results nor PR6 grounding-37 constitute semantic correctness.**

## Frozen identities

| Item | Identity |
| --- | --- |
| Refreshed master / unchanged production | `21a792c9653e662964fee8fb550eba80fa0ad5de` (post-PR8, including merged PR27) |
| Dataset freeze before inference | `b7d30041a678cd9ef5395828daf49c4db634605c` |
| Evaluated implementation | `3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb` |
| Dataset manifest SHA-256 | `db6a8c7a985cfc95734e23a978bfbb923bcf145ce06b15886a4b44a3dbce3a49` |
| Queries SHA-256 | `63e085e940a98aa7aa6b6136f9a55a2ce5bdc810e64d8bc6663d4125fdba65d2` |
| Blind source-audit commit before judge | `ac394f35f39abbc8fcff48be9ba8d5e25ea63a86` |
| Corpus SHA-256 | `39c8d01ee5c71710e443e49698957d2aa991e71c16ad79905e7bd07d6a11fe4d` |
| Index fingerprint before = after | `641f5ee5c465daaa7106717eb4e4a8a4e145cdfd04e4e8afd202a892d2e53630` |

The read-only collection is `finsearch_benchmark_v2_39c8d01ee5c71710` (948 points, six filings). The historical index also remained unchanged. Original PR3–PR8 and PR27 datasets/artifacts, grounding-37, degradation-14, calculator-20, output-repair-20, resolver-170 and filing-period-49 files were not modified.

The 60 questions cover 86 required claims, three companies and six annual filings: AAPL 2024/2025, AMZN 2023/2024, MSFT 2024/2025. There are 54 answerable and 6 scope-insufficient cases. Gold numeric facts come from source HTML inline-XBRL elements, not `sec_metric` results; narrative and table gold has independently selected source/chunk/span/cell provenance. No current retriever output or generated answer supplied the gold labels.

## Models and execution

- Planner/analyst profile: local Ollama `qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
- Embedding: `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`.
- Hosted reranker: Dashscope `Qwen/Qwen3-Reranker-8B`; service weights have no independently recorded digest.
- Secondary judge: local Ollama `gemma4:e4b`, digest `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`; temperature 0, context 32768, output limit 4096, thinking off, JSON format, 240s timeout, one attempt. Rubric/config hashes are recorded.
- Ollama 0.33.2; Apple M4, 32 GiB memory, Darwin arm64; Python 3.11.14; environment-only pytest 9.0.2. No dependencies or package manifests were changed.
- Answer pass: 2026-09-07 00:52:21–12:31:33 UTC, fixed shuffled order, no harness retries or replies to runtime clarification. Runtime's own retries and 120s analyst-call timeout were unchanged.
- Evaluation-only resource selection used the frozen six-filing KB and table sidecars, fresh checkpoint/query cache, and the existing live SEC client without metric fixtures. This is not the historical PR8 15-case gate or its original index.

## Deterministic results

| Metric | Result | Interpretation |
| --- | --- | --- |
| Final responses emitted | 21/60 (35.0%) | Includes 9 insufficient-data responses; only 12 substantive `ok` answers |
| Claim citation coverage | 20/20 (100%) | Nonempty references, not support |
| Valid visible context IDs | 23/23 (100%) | References resolve to analyst-visible contexts |
| Structured numeric → structured evidence | 0/0 (not applicable) | No emitted structured-numeric claims; not a pass |
| Narrative/attribution/KB numeric → KB evidence | 17/17 (100%) | Evidence-type compatibility, not entailment |
| Strict typed-gold numeric reproduction | 0/56 (0%) | Required numeric claims with matching value and source/type bindings |
| Structural unsupported flags | 0 | Does not mean zero unsupported claims |
| Answerability status/claim contract | 17/60 (28.3%) | Does not certify semantic correctness |
| Expected insufficient-data cases correct | 5/6 (83.3%) | Four other refusals occurred on answerable cases |

The frozen output key `numeric_consistency` is the strict typed-gold reproduction rate. Correct equivalent KB facts or KB calculator operands may receive semantic credit while failing this stricter source/type check. Its number matching is not an exhaustive semantic quantity-role, negation or unit-format parser. **Do not relabel 0/56 as a general numeric-accuracy measurement.**

| Stratum | Cases | Emitted responses, including refusals | Emitted claims | Typed numeric matches |
| --- | ---: | ---: | ---: | ---: |
| Structured numeric | 12 | 0 | 0 | 0/12 |
| Narrative | 6 | 3 | 3 | N/A |
| Hybrid | 6 | 1 | 0 | 0/6 |
| Comparison | 6 | 2 | 3 | 0/12 |
| Calculator | 6 | 1 | 0 | 0/18 |
| Multiple claims | 6 | 5 | 12 | N/A |
| Insufficient data | 6 | 5 | 0 | N/A |
| Difficult attribution | 6 | 2 | 2 | N/A |
| Plausible wrong evidence | 6 | 2 | 0 | 0/8 |

Full per-stratum metrics, denominators and nulls are in [deterministic_summary.json](deterministic_summary.json) and [verification.json](verification.json).

## Source-inspected semantic audit

The predetermined 18 cases were reviewed against actual final answers and cited visible evidence before any secondary judge prediction. This is **assistant source inspection, not independent human adjudication**. All nine emitted claims in this subset were assessed; failed outputs were not given credit for rejected candidates or retained error-object prose.

| Audit metric | Result |
| --- | --- |
| Fully supported emitted claims | 6/9 (66.7%) |
| Partially supported emitted claims | 0/9 |
| Unsupported emitted claims | 3/9 (33.3%) |
| Fully grounded answers | 5/18 (27.8%): three substantive answers and two correct abstentions |
| Complete, relevant answers | 4/18 (22.2%) |
| Complete gold requirements | 4/27 (14.8%) |
| Structural-flag detection of audited unsupported claims | 0/3 recall; precision N/A (no positive flags) |

Concrete source-audited distinctions:

- `SEM1_AMZN_2024_05`: an accepted answer calls operating income ($68,593m / $36,852m) total revenue. Its table explicitly says Operating Income (Loss); required revenues are $637,959m / $574,785m. Both claims are unsupported despite valid citations and internally consistent rows.
- `SEM1_AMZN_2023_03`: an accepted unearned-revenue explanation overgeneralizes retail carrier-delivery timing. The required policy is payments received or due before service obligations, recognized over the service period. The bundled claim is unsupported, with partial gold completeness.
- `SEM1_AAPL_2025_07`: the emitted component-sourcing/risk claim is supported, but the required new-product-specific detail is omitted. Groundedness is not completeness.
- Several explicit cash questions stop for cash-versus-cash-flow clarification; Microsoft R&D questions can stop for company ambiguity. These remain unanswered cases, not correct insufficient-data results. No resolver or capability-policy repair was attempted.

Inspect all frozen reasons and exact cited quotations in [source_audit.jsonl](source_audit.jsonl). These subset results must not be extrapolated to all 60 cases or to general SEC-research accuracy.

## Secondary judge: low usable coverage

One pass produced **4 valid assessments, 17 judge errors and 39 unassessable system failures**. The 17 errors comprise 15 claim-coverage mismatches, one malformed claim/quotation structure, and one non-cited or invented evidence quotation. No judge retry, output repair or prompt/model adjustment was performed.

The judge assessed only **7/20 emitted claims (35%)**. It labeled all seven fully supported, zero partially supported and zero unsupported; all four valid assessed answers were labeled fully grounded. The machine-readable judged-only rates are therefore 100%, but **they are not overall semantic accuracy**. The judge-estimated count over all cases is 4/60 (6.7%), with the remaining assessments unavailable. Valid schema and verbatim quotes do not certify a judge's semantic conclusion.

On the overlapping source-audited cases, claim-label agreement is 6/6, all supported claims. The audited unsupported claims have no valid judge assessments, so this agreement says nothing about judge sensitivity to unsupported content. There is one recorded completeness disagreement: for `SEM1_AAPL_2025_07`, the judge credits a new-product detail present in the evidence but absent from the answer; the blind audit marks that requirement partial. Both judgments are preserved in [verification.json](verification.json), without adjudication after seeing predictions.

Full per-stratum secondary metrics and raw errors are in [judge/summary.json](judge/summary.json) and [judge/judgments.jsonl](judge/judgments.jsonl). This judge is not established as a reliable unattended scorer by this run.

## Failures, conditions and latency

Final outcome counts: 12 `ok`, 9 `insufficient_data`, 16 `error`, 7 `tool_error`, 4 `grounding_error`, 10 planner interruptions and 2 retrieval-stage failures without an analyst. Analyst error details include 19 occurrences of the unchanged 120s timeout, 4 grounding-invalid errors, 2 output-invalid errors, one calculation-result mismatch and one calculation ambiguity. Error-detail categories and top-level status categories are different views, not additive totals.

The captured retrieval failures include local embedding HTTP 500 and hosted reranker DNS errors. These are availability failures, not proof that an unproduced answer would have been semantically wrong. Power was AC at 59 case starts and non-AC at one; another case ended non-AC. Low Power Mode was recorded as off, but browsers and other workloads appeared during the run. The awake helper was used; this does not establish continuous workload isolation or explain all clock differences.

| Recorded clock | p50 | p95 |
| --- | ---: | ---: |
| Existing orchestration trace (`time.time` based) | 323.593s | 4064.140s |
| Outer harness interval (`perf_counter` based; raw key `wall_ms`) | 276.352s | 603.966s |

Eight cases have large disagreement between those clock domains. The UTC start/end span is about 11h39m while the sum of outer intervals is about 5h03m. Both are preserved; neither has been rewritten to conceal the discrepancy. The snapshots do not prove a specific sleep, clock, workload or hardware cause. **Do not use this run for controlled latency claims or attribute outcome changes solely to product code.** See [raw_answers.jsonl](raw_answers.jsonl), [runtime_observation.json](runtime_observation.json) and [system_observation_late.json](system_observation_late.json).

## Integrity, publication and tests

[verification.json](verification.json) verifies the local original evidence; [verification-publication.json](verification-publication.json) verifies the published copy. Dataset, source hashes, case order/coverage, index identity, deterministic scores, secondary aggregates, audit coverage and judge quotations all reproduce.

The published copy removes only an upstream model-build home prefix from `parent_model` metadata and rebinds the affected artifact checksums. Original files remain recoverable locally. [publication_lineage.json](publication_lineage.json) records original/published hashes and exact changed JSON pointers. Answers, judgments, audits, scores, model digests and source hashes are unchanged; all JSONL evidence is byte-identical. Publication helper and documentation commits after implementation freeze do not change evaluated behavior.

Final tests: **908 passed, 47 subtests passed, 2 pre-existing failures, 1 warning**. All 36 new evaluator/publication tests pass. The two unchanged failures are planner `alias_recognition/alias_002` routing and the retrieval no-tool-call attempt count. The warning tests table-render fallback without `tabulate`. This is not a clean full-suite pass. See [tests.txt](tests.txt).

Commands run (existing `finsearch-arm`; environment credentials omitted):

```sh
PYTHONPATH=src:. python -m pytest tests/evals/test_semantic_answer_v1.py -q
PYTHONPATH=src:. python -m pytest tests/evals/test_semantic_publication.py -q --import-mode=importlib
PYTHONPATH=src:. python -m pytest tests -q --import-mode=importlib
PYTHONPATH=src:. python scripts/evals/agents/run_semantic_v1.py answers
PYTHONPATH=src:. caffeinate -i python scripts/evals/agents/run_semantic_v1.py judge --baseline BASELINE --audit BASELINE/source_audit.jsonl
PYTHONPATH=src:. python scripts/evals/agents/verify_semantic_v1.py BASELINE --out BASELINE/verification.json
python scripts/evals/publish_semantic_evidence.py LOCAL_ORIGINAL BASELINE
PYTHONPATH=src:. python scripts/evals/agents/verify_semantic_v1.py BASELINE --out BASELINE/verification-publication.json
git diff --check
```

## What future claims are defensible

This work supports a claim that a source-first, frozen 60-question/86-claim benchmark and independent evaluation harness were built, all current-system outcomes were captured, and semantic citation failures invisible to structural checks were identified. It does not support general financial accuracy, human-reviewed accuracy, a reliable full-corpus LLM-judge score, or a clean 100% groundedness claim.

Future comparisons must retain the same question/gold/corpus/scoring identities, all cases and errors, disclose changed implementation/model/provider settings, and use comparable recorded execution conditions. Corrections require a new version, never relabeling v1. Given this run's availability and clock issues, a future causal improvement claim needs a separately authorized comparable baseline; do not silently substitute selected retries or this run's surviving answers. Repeated templates, only three large-cap technology companies, annual filings, revenue-heavy numeric questions and variable claim bundling further limit generalization. The initial runner deliberately pins production to the no-runtime-change base; future implementation variants need an explicit provenance extension, not removal of the guard in this PR.

No retrieval or answer-quality optimization and no merge are part of this work. Fresh Codex review is requested on the final evidence/documentation head.
