# Frozen retrieval benchmark v2 — current-system baseline

## Identity and integrity

- Base: merged PR8 `9ec25f568402d700c88a5fa35b7e5e750bc36d5f`.
- Dataset freeze: `0cc20ce37c42d18412cfb518f8b06f775d07257b`, before any ranked comparison.
- Evaluated implementation: `2d50cfe0dc7b624676b472b7407aab7dc11f9648`, clean at start/end.
- Run: 2026-09-06, one pass, 480/480 pairs, 258.22 seconds measurement wall time.
- 120 answerable queries, six explicit exclusions, six filings, 948 chunks.
- Errors: 0; missing labels: 0; duplicate returned IDs: 0 in every mode.
- Benchmark and historical index payload/vector/config fingerprints unchanged.
- Embedding model digest unchanged; current Qwen3 API reranker used for all 120 reranked pairs, with no fallback.
- Offline verifier: 480 pairs, complete, hashes and recomputed metrics verified.
- No retrieval/prompt/provider tuning, selective rerun, or historical edits.

The accompanying `manifest.json` records exact source, dataset, model, package,
hardware, index and raw-output hashes. `per_query.jsonl` preserves all ranked
IDs, scores, errors, timings and individual metrics. `summary.json` contains
every requested metric overall and by all nine strata and all three companies.
`errors.jsonl` is intentionally empty, not missing.

## Overall results

All quality numbers are macro means over 120 queries per mode. Latencies are
milliseconds, nearest-rank percentiles, 120 observations per applicable stage.

| Mode | R@5 | R@10 | MRR@10 | nDCG@5 | nDCG@10 | Group coverage@10 | Total p50 | Total p95 | Reranker p50 | Reranker p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7215 | 0.8014 | 0.6014 | 0.6002 | 0.6310 | 0.8292 | 13.8 | 17.0 | n/a | n/a |
| Dense | 0.6861 | 0.8104 | 0.5934 | 0.5713 | 0.6170 | 0.8542 | 140.1 | 278.3 | n/a | n/a |
| Hybrid | 0.7368 | 0.8660 | 0.6883 | 0.6485 | 0.6989 | 0.8958 | 124.1 | 295.4 | n/a | n/a |
| Hybrid + Qwen3 | 0.7681 | 0.8660 | 0.7202 | 0.6778 | 0.7147 | 0.8958 | 1333.3 | 2127.7 | 1172 | 1873 |

The reranker only reorders the same ten candidates. It cannot improve their
Recall@10. Its aggregate ranking gain comes with additional latency and is not
consistent across strata. Dense's higher observed p50 than hybrid is not a
causal claim that adding BM25 makes retrieval faster: these are single,
interleaved measurements with shared model/process state.

## Stratum nDCG@10 (all other metrics and latencies in summary.json)

| Stratum | Queries | BM25 | Dense | Hybrid | Hybrid + Qwen3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Business/growth | 12 | 0.9583 | 0.9283 | 1.0000 | 0.9374 |
| Direct fact/table | 18 | 0.2866 | 0.4239 | 0.4295 | 0.5228 |
| Hard negative | 12 | 0.5476 | 0.5536 | 0.6197 | 0.6618 |
| MD&A | 12 | 0.6554 | 0.6709 | 0.7455 | 0.8016 |
| Multi-evidence | 12 | 0.5362 | 0.7015 | 0.6325 | 0.6383 |
| Narrative | 12 | 0.5554 | 0.2451 | 0.4117 | 0.4729 |
| Paraphrase | 12 | 0.7245 | 0.4823 | 0.7713 | 0.7623 |
| Risk factors | 18 | 0.6373 | 0.7150 | 0.8078 | 0.8333 |
| Section-specific | 12 | 0.9472 | 0.8796 | 0.9525 | 0.8382 |

## Execution controls and limitations

All 960 before/after-call samples show AC power, Low Power Mode off, and zero
Chrome/Safari processes. An owned awake assertion covered measurement. Every
embedding-enabled mode had 120 cache misses and zero hits in its separate,
initially empty cache namespace. The fixed shuffled query order and rotating
mode order put each mode first for 30 queries. No harness retries occurred.

This was not a perfectly idle laboratory machine. The preserved samples
include >50% CPU observations for ChatGPT (165), Docker Desktop Helper Renderer
(30), ecosystemd (23), Codex Renderer (16), git (4), python (3), and weather (2).
These are observations, not unique processes, durations or CPU attribution;
process IDs were not captured and the evaluator itself can appear as python.
Keep the observed latency tails and their limitations; do not silently rerun
or present them as an isolated performance bound. The API reranker lacks an
immutable service-side digest, limiting bit-for-bit future reproducibility.

Labels are source-first, AI-authored and source-inspected, independent of
retriever outputs, but not independently human adjudicated or exhaustive.
Unjudged top-ten returns total 1053/1043/1039/1039 for BM25/dense/hybrid/reranked;
they receive zero gain, not a claim of proven irrelevance. Recall is known-gold
chunk recall; group coverage separately handles alternative evidence chunks.
The three technology/commerce issuers and paired year/topic templates limit
generalization and effective sample size. Existing parser heading mistakes are
preserved. Raw table representation differs from PR20, invalidating direct
cross-benchmark improvement claims. Exclusions do not measure abstention.
No claims about financial calculations, answer correctness, routing or XBRL
execution are supported by this KB-only evaluation.

## Verification and commands

Environment: existing `finsearch-arm`, Python 3.11.14, pytest 9.0.2. No dependency
or lockfile changes. Before measurement, focused tests passed 55/55. The full
suite reported 860 passed and 47 subtests passed, with two existing failures:

- `test_planner_routes_match_expected_routes`: alias_002 expected structured_fact, got kb.
- `test_retrieval_no_tool_call_increments_attempts_and_stops_on_budget`: expected two attempts, got one.

One existing table-renderer warning also remains. This is not a green full-suite
claim. Commands below use portable environment paths; test namespace setup
avoids an unrelated installed package named `tests`.

```text
PYTHONPATH=src conda run -n finsearch-arm python -m pytest tests/evals/test_retrieval_benchmark_v2.py tests/evals/test_retrieval_ablation.py -q --tb=short
PYTHONPATH=.:src conda run -n finsearch-arm python -c 'import pathlib,sys,types;p=types.ModuleType("tests");p.__path__=[str(pathlib.Path.cwd()/"tests")];sys.modules["tests"]=p;import pytest;raise SystemExit(pytest.main(["-q","--tb=short"]))'
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/index_benchmark_v2.py --out-dir .cache/retrieval_benchmark_v2_index
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/run_benchmark_v2.py --index-manifest .cache/retrieval_benchmark_v2_index/completed.json
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/verify_benchmark_v2.py --baseline artifacts/evals/retrieval/benchmark_v2/baselines/2d50cfe0dc7b624676b472b7407aab7dc11f9648
git diff --check
```

The index and baseline commands are recorded history, not instructions to
overwrite this run. Only the verifier is read-only/model-free. Review is
requested after publication; no automatic merge or retrieval optimization.
