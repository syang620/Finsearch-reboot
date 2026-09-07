# Semantic answer-quality benchmark v1

This is a new evaluation-only benchmark. Production code is identical to merged post-PR8/PR27 master `21a792c9653e662964fee8fb550eba80fa0ad5de`. All historical datasets, gates and artifacts remain immutable.

## Frozen inputs

- 60 questions, 86 required claims, 54 answerable / 6 scope-insufficient.
- Six annual filings: AAPL 2024/2025, AMZN 2023/2024, MSFT 2024/2025.
- Nine strata: structured numeric (12), narrative, hybrid, comparisons, calculator, multiple claims, attribution, plausible wrong evidence and insufficient data (6 each).
- Numeric source catalog: 80 consolidated source-HTML facts; exact inline-XBRL concept/context/element/period/unit provenance. No tool results used as gold.
- Source-inspected narrative spans and table cells, with alternative chunks where equivalent. No ranked retrieval or generated answers used to choose gold.
- Eighteen source-audit cases selected before the run, two per stratum. Audit is assistant-reviewed, **not independent human adjudication**.

See [annotation rules](../../data/evals/semantic_answer/v1/ANNOTATION.md), [manifest](../../data/evals/semantic_answer/v1/manifest.json), [configuration](../../data/evals/semantic_answer/v1/evaluation_config.json) and [secondary judge rubric](../../data/evals/semantic_answer/v1/judge_rubric.txt). Dataset manifest SHA-256: `db6a8c7a985cfc95734e23a978bfbb923bcf145ce06b15886a4b44a3dbce3a49`.

## Procedure

1. Source-only builders establish gold and verify original HTML/chunk/cell correspondence. Commit frozen data before inference.
2. Test independent evaluator and preserve historical regressions; freeze a clean implementation.
3. Run all 60 questions once, sequentially in frozen shuffled order, through the unchanged orchestrator. Only the actual question enters runtime. No gold hints, harness retries or responses to clarification.
4. Save raw final outputs and analyst-visible context packets, deterministic rows, errors, existing timings and machine state immediately per case. Verify read-only corpus/index and model identities before/after.
5. Source-review the 18 predetermined cases without seeing secondary judge predictions; commit those annotations.
6. Run one secondary judge call per emitted final answer, preserving schema/quote errors. Report semantic coverage, unknowns, audit disagreements and both conditional rates and all-case lower bounds.
7. Commit immutable SHA-keyed evidence, request fresh Codex review and stop. Do not merge or tune retrieval/answers in this task.

Local execution uses the existing `finsearch-arm` environment; no package manifest or lockfile changes. `pytest==9.0.2` is environment-only and recorded in provenance. Existing SEC contact and hosted reranker credential are supplied via local environment, never in artifacts.

```sh
PYTHONPATH=src:. python -m pytest tests/evals/test_semantic_answer_v1.py -q
PYTHONPATH=src:. python scripts/evals/agents/run_semantic_v1.py answers
PYTHONPATH=src:. python scripts/evals/agents/run_semantic_v1.py judge --baseline artifacts/evals/semantic_answer/v1/baselines/IMPLEMENTATION_SHA --audit artifacts/evals/semantic_answer/v1/baselines/IMPLEMENTATION_SHA/source_audit.jsonl
```

## Interpretation

PR6 grounding-37 measures structural validity, not semantic correctness. This benchmark distinguishes citation presence, valid IDs, compatible evidence type, deterministic gold numeric consistency and secondary semantic entailment. The latter is judge-estimated and audited on only 18 source-reviewed cases. Correct answer values alone do not certify a correct cited attribution; structural flags do not detect all unsupported claims.

The output key `numeric_consistency` is a **strict typed-gold reproduction rate**, not general numerical or semantic accuracy. It requires the declared claim type and independently bound source metric/period as well as the displayed value. A correct equivalent KB answer to a structured-gold question (including KB calculator operands) may receive semantic credit while failing this deterministic check. Number matching is not an exhaustive unit-format, negation or quantity-role parser. Report this rate with its required-claim denominator and secondary/source-reviewed findings, never as a standalone financial-accuracy score.

Likewise, deterministic `answerability_correctness` checks the emitted status/claim contract; it does not certify that an accepted answer is correct. Semantic answerability and gold completeness are separate. Rejected candidates or factual prose retained inside failed output objects are preserved for inspection but are not credited as delivered grounded answers. Claim rates depend on how the runtime bundles assertions, so the number of claims is not an independent-sample count.

Publication may redact an upstream model-build home prefix from `parent_model` metadata using `scripts/evals/publish_semantic_evidence.py`. The publication lineage retains original and published file hashes; original bytes remain local. Answers, judgments, audits, scores, model digests and frozen source hashes are unchanged. This export-only step is separate from the evaluated implementation and does not justify an inference or scoring rerun.

This corpus/profile differs from the historical PR8 live gate; never relabel or compare it as a replacement gate. Paired question families, three large-cap technology companies, annual filings and revenue-heavy numeric cases restrict generality. Hosted service weights and live SEC data are not fully reproducible. Single-pass latency is observational. Future before/after claims must keep dataset/corpus/scoring fixed, retain failures and disclose implementation/model/provider changes. No broad financial-accuracy or independent-human agreement claims are justified.

The [frozen baseline report](../../artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/REPORT.md) records all 60 outcomes: 12 substantive answers, 9 insufficient-data responses, and 39 failures/clarification stops. The 18-case source audit found 3 unsupported claims despite perfect structural citation coverage. The secondary judge produced only 4 valid assessments and 17 errors; its judged-only rates are not overall accuracy. Service failures, power/workload changes and clock-domain discrepancies make this an observational baseline, not a controlled latency or causal-improvement comparison. No tuning or rerun was performed.

## Pre-inference checks

Dataset freeze: `b7d30041a678cd9ef5395828daf49c4db634605c`. New evaluator tests include malformed/missing gold, numeric units and periods, hidden/duplicate context IDs, valid-citation/wrong-number counterexamples, wrong evidence types, calculator provenance, judge schema/quote validation, empty denominators and missing-answer failures.

The full existing test run (`python -m pytest tests -q --import-mode=importlib`, before the final three new evaluator tests) produced **902 passed, 47 subtests passed, 2 known failures, 1 warning**. The failures match the prior PR27 record: planner `alias_recognition/alias_002` expects structured but routes KB, and retrieval no-tool-call test expects two attempts but gets one. Neither was changed. Plain pytest import mode initially hit an existing `tests` namespace collision; importlib mode collected the suite successfully. These results are not a clean full-suite pass and do not authorize tuning those unrelated behaviors.

The final post-evaluation regression run produced **908 passed, 47 subtests passed, the same 2 known failures, and 1 warning**. All 36 new evaluator/publication tests pass; the exact output and commands are in the baseline report. Original and published evidence both pass the independent verifier. Audit annotations were committed before judge predictions; no post-judge relabeling was performed.
