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

This corpus/profile differs from the historical PR8 live gate; never relabel or compare it as a replacement gate. Paired question families, three large-cap technology companies, annual filings and revenue-heavy numeric cases restrict generality. Hosted service weights and live SEC data are not fully reproducible. Single-pass latency is observational. Future before/after claims must keep dataset/corpus/scoring fixed, retain failures and disclose implementation/model/provider changes. No broad financial-accuracy or independent-human agreement claims are justified.

Baseline results will be linked here after the frozen run; absent results are not a pass.

## Pre-inference checks

Dataset freeze: `b7d30041a678cd9ef5395828daf49c4db634605c`. New evaluator tests include malformed/missing gold, numeric units and periods, hidden/duplicate context IDs, valid-citation/wrong-number counterexamples, wrong evidence types, calculator provenance, judge schema/quote validation, empty denominators and missing-answer failures.

The full existing test run (`python -m pytest tests -q --import-mode=importlib`, before the final three new evaluator tests) produced **902 passed, 47 subtests passed, 2 known failures, 1 warning**. The failures match the prior PR27 record: planner `alias_recognition/alias_002` expects structured but routes KB, and retrieval no-tool-call test expects two attempts but gets one. Neither was changed. Plain pytest import mode initially hit an existing `tests` namespace collision; importlib mode collected the suite successfully. These results are not a clean full-suite pass and do not authorize tuning those unrelated behaviors.
