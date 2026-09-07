# PR27 post-review complete baseline

## Freeze, review and preservation

- Evaluated clean SHA: `54d31917c27263ea25ebf5d905caa0a4ee34f74f`.
- Unchanged dataset freeze: `0cc20ce37c42d18412cfb518f8b06f775d07257b`.
- The only evaluation-source change since the original run is the read-only
  verifier's finite-float tolerance (`1e-12` relative/absolute), committed at
  `4784e13`. Metric formulas, rankings, corpus, labels, models and settings
  are unchanged. Non-floating fields and file hashes remain exact.
- Codex review of `54d3191` completed with no new findings after the P2 fix.
- Original complete `2d50cfe` and interrupted 90-pair `4784e13` evidence are
  preserved unchanged. This is a new full pass, not continuation or selection
  of the interrupted run. It followed restored workload isolation, not tuning.

## Complete result

**480/480 pairs; zero retrieval errors, missing labels or duplicate returned IDs.**
The measurement took 257.81 seconds. The offline verifier checked raw hashes,
pair coverage and recomputed individual/aggregate metrics. Every ranked-ID list
and quality-metric dictionary exactly equals its counterpart in the original
480-pair run. This is an exact quality reproduction, not an optimization gain.

| Mode | R@5 | R@10 | MRR@10 | nDCG@5 | nDCG@10 | Total p50 ms | Total p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7215 | 0.8014 | 0.6014 | 0.6002 | 0.6310 | 13.5 | 16.1 |
| Dense | 0.6861 | 0.8104 | 0.5934 | 0.5713 | 0.6170 | 142.7 | 282.8 |
| Hybrid | 0.7368 | 0.8660 | 0.6883 | 0.6485 | 0.6989 | 123.8 | 293.1 |
| Hybrid + Qwen3 | 0.7681 | 0.8660 | 0.7202 | 0.6778 | 0.7147 | 1363.0 | 1884.6 |

Reranker-only p50/p95: **1193/1684 ms**, 120 observations. All overall,
stratum and company metrics, latency sample counts and unjudged-return counts
are in `summary.json`; all ranked results are in `per_query.jsonl`. No strata
or failing comparisons were removed. Empty `errors.jsonl` is intentional.

## Controls and limitations

All 960 before/after-call power/browser samples passed: AC power, Low Power Mode
off and no Chrome/Safari. The owned awake assertion covered measurement. Each
embedding-enabled mode recorded 120 cache misses and zero hits. Both benchmark
and historical index payload/vector/config fingerprints and the embedding
digest were unchanged. Package/model/index/source hashes are in `manifest.json`.

Background >50% CPU sample counts were ChatGPT 235, Docker Desktop Helper
Renderer 29, ecosystemd 27, Codex Renderer 15, git 4, python 3, sandboxd 1, and
weather 1. These are observations, not CPU-duration attribution or unique
process counts; the evaluator can appear as python. Latencies remain local
observations, not isolated laboratory bounds or production SLOs. No performance
claim is based on selecting the faster of the two complete runs.

All original limitations remain: AI-authored/source-inspected rather than
independently human-adjudicated labels; known-gold rather than exhaustive recall;
three technology/commerce issuers and correlated year/topic templates; retained
parser metadata problems; raw-table representation unlike PR20; no XBRL or
end-to-end answer-quality claims. Reranking reorders ten existing candidates,
so equal hybrid Recall@10 is expected. The API lacks an immutable service digest.

## Verification

At the code freeze: **67 focused tests passed**. Full suite: 872 passed, 47
subtests passed, two pre-existing failures (planner alias_002 and retrieval
no-tool-call attempt count), one existing table-renderer warning. Python
3.11.14 and pytest 9.0.2 in existing `finsearch-arm`; no dependency changes.
Only evidence/documentation changes follow this evaluated implementation.

```text
PYTHONPATH=src conda run -n finsearch-arm python -m pytest tests/evals/test_retrieval_benchmark_v2.py tests/evals/test_retrieval_ablation.py -q --tb=short
PYTHONPATH=.:src conda run -n finsearch-arm python -c 'import pathlib,sys,types;p=types.ModuleType("tests");p.__path__=[str(pathlib.Path.cwd()/"tests")];sys.modules["tests"]=p;import pytest;raise SystemExit(pytest.main(["-q","--tb=short"]))'
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/run_benchmark_v2.py --index-manifest .cache/retrieval_benchmark_v2_index/completed.json
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/verify_benchmark_v2.py --baseline artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f
git diff --check
```

The measurement command is recorded history and must not overwrite this run.
Only offline verification may be repeated against these frozen outputs.
