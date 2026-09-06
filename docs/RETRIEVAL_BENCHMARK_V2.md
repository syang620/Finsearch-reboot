# FinSearch source-first retrieval benchmark v2

This evaluation-only PR starts from merged PR8 master `9ec25f5`. It does not
change production retrieval, prompts, model/provider settings, ranking depths,
retry budgets, or historical datasets and baselines. PR20 remains immutable.

## Dataset freeze and scope

Dataset frozen **before any ranked comparison** at
`0cc20ce37c42d18412cfb518f8b06f775d07257b`:

- 120 answerable queries plus six explicit exclusions; 20 scored queries per
  filing across AAPL FY2024/2025, AMZN FY2023/2024 and MSFT FY2024/2025.
- 948 chunks, from all chunks extracted by the existing parser/splitter for the
  six tracked annual filings: 572 text chunks and 376 source tables.
- Nine primary strata: direct fact (18), narrative (12), risk factors (18),
  business/growth (12), MD&A (12), paraphrase (12), section-specific (12),
  hard negative (12), multi-evidence (12).
- 179 relevant, 24 partial and 16 explicitly irrelevant judgments over 122
  distinct chunks; 43 scored cases have multiple relevant chunks.
- Queries SHA-256:
  `e356eab5c78f027c570daa05945b4688e152ab3e54c79070862d9759fc2cc9d9`.

[Annotation rules](../data/evals/retrieval/benchmark_v2/ANNOTATION.md) define
grades, source spans, alternatives, required evidence groups, exclusions,
unjudged documents and missing-label failures. The
[manifest](../data/evals/retrieval/benchmark_v2/dataset_manifest.json) records
source hashes and counts; [inspectable queries](../data/evals/retrieval/benchmark_v2/queries.jsonl)
contain the evidence quotes and original extracted section metadata.

The historical 75-case text seed is unchanged. Its
[audit](../data/evals/retrieval/benchmark_v2/seed_compatibility.json) has seven
exact-anchor evidence mappings and 85 unresolved truncated evidence anchors
(counts are evidence entries, not query counts). Only one independently
source-inspected case is adopted with its original query semantics. There is no
silent ID renumbering, guessed mapping, label regeneration or retroactive PR20
reclassification.

## Comparison contract

All modes call existing `evals.retrieval_ablation.retrieve_ablation_points`.
The reranked mode delegates to unchanged production `retrieve_scored_points`.
BM25, dense, hybrid and hybrid+Qwen3 therefore retain their existing branch
limits (500), candidate depth (50), post-deduplication limit (10), equal-weight
RRF with k=60, embedding model and reranker. No XBRL/structured tool executes.
Every mode searches the same collection with the same company/year/form filter
and full document-type filter; gold strata do not restrict the search.

Quality is macro-averaged per query, both overall and by stratum/company:

- Recall@5/@10 and MRR@10 use grade 2 only.
- nDCG@5/@10 uses gain `2^grade - 1`, discount `log2(rank+1)`, and the ideal
  ranking across **all** positive labels, not only retrieved labels.
- Additional evidence-group recall@10 credits any valid alternative for each
  required facet. Ordinary chunk recall still counts all known relevant IDs.
- Errors stay in quality denominators with zero metrics. Explicit exclusions
  never enter quality/latency denominators. Missing labels fail validation.
- Duplicate returned IDs consume their original rank but gain no repeated
  credit. Unjudged returns are counted separately from explicit grade-zero
  negatives, though both have zero scoring gain.

Latency is wall-clock embedding + retrieval + fusion + enrichment + reranking,
as applicable. Reranker-only latency uses the existing stage timer. Both p50 and
p95 use nearest rank `ceil(p*n)`; absent/non-applicable timings are null with
sample counts, never fabricated zeros. Total retrieval timings include failed
calls; reranker timing coverage may be lower if failures prevent a stage result.

Query order uses fixed seed 20260906; mode order rotates so each mode runs first
30 times. Each mode has a separate initially empty query-cache directory, with
default caching enabled. Model/process warm state is shared; this is neither a
fully cold-start nor a production repeated-query cache benchmark. Each of the
480 query-mode pairs runs once, with no harness retry or outcome-based selection.

## Execution and reproducibility

Use the existing `finsearch-arm` environment; pytest 9.0.2 remains environment-
only. No dependencies are added. The corpus builder and freeze script refuse
existing outputs. The following build commands are **historical construction
steps**, not commands to run over this committed frozen dataset:

```text
PYTHONPATH=src python scripts/evals/retrieval/build_benchmark_v2_corpus.py --out-dir <NEW_VERSION_DIRECTORY>
PYTHONPATH=src python scripts/evals/retrieval/freeze_benchmark_v2.py
```

The new benchmark index has a corpus-SHA-derived name, never the production or
PR20 collection name. Index construction refuses an existing collection; dense
vectors use the existing Qwen3 embedding model/digest and sparse vectors use the
existing Qdrant BM25 ingestion helper. Full-corpus average document length is
recorded. No model-generated table summaries or row documents are introduced.
Local embedding outputs remain under ignored `.cache/`; their hash is recorded
in the index manifest rather than adding a large vector file to Git.

```text
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/index_benchmark_v2.py --out-dir .cache/retrieval_benchmark_v2_index
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/run_benchmark_v2.py --index-manifest .cache/retrieval_benchmark_v2_index/completed.json
PYTHONPATH=src conda run -n finsearch-arm python scripts/evals/retrieval/verify_benchmark_v2.py --baseline artifacts/evals/retrieval/benchmark_v2/baselines/<IMPLEMENTATION_SHA>
```

Only the final verifier is read-only and model-free. The measurement command
requires a clean frozen checkout, compatible source/index hashes, correct model
digest/settings, AC power, Low Power Mode off and browsers closed. It maintains
an awake assertion and records power/workload samples before/after each call.
Control loss preserves partial output and stops without rerunning. Existing
SHA-keyed output/cache directories cannot be overwritten. The source and
historical index fingerprints are checked before/after evaluation. Model and
code identity are also checked; reranker service identity remains externally
versioned without an immutable service-side digest.

No credentials or absolute local paths belong in committed artifacts. Each run
records raw ranked IDs/scores, per-case metrics/timings, errors, exact source
hashes, dataset/config/index provenance and aggregate strata. The offline
verifier recomputes quality/latency aggregates from the preserved raw rows.
Following PR27 review, recomputed floating-point values use `1e-12` relative
and absolute tolerances to accommodate Python/libm rounding differences.
Structure, counts, IDs, non-floating values and file hashes remain exact;
non-finite numbers are rejected. This changes verification acceptance only,
not ranking, metric formulas, labels or retrieval settings. The original
`2d50cfe` baseline remains immutable; review follow-up evidence is separately
SHA-keyed after the verifier/test implementation is re-frozen.

## Verification and release status

Before baseline measurement, focused benchmark + unchanged PR20 tests passed
55 tests. The full suite passed 860 tests and 47 subtests with the two unchanged
known failures (planner `alias_002`, retrieval no-tool-call attempt count), plus
one existing table-renderer warning. These are not a blanket full-suite pass.
The implementation was frozen at
`2d50cfe0dc7b624676b472b7407aab7dc11f9648` with a clean worktree. Its one-pass
baseline completed on 2026-09-06: **480/480 pairs, zero errors, zero missing
labels**, with unchanged benchmark/historical indexes and embedding identity.
The offline verifier confirmed raw hashes, pair coverage, individual metrics
and aggregate metrics. No evaluation rerun or optimization followed.

| Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Total p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7215 | 0.8014 | 0.6014 | 0.6002 | 0.6310 | 13.8 / 17.0 |
| Dense | 0.6861 | 0.8104 | 0.5934 | 0.5713 | 0.6170 | 140.1 / 278.3 |
| Hybrid | 0.7368 | 0.8660 | 0.6883 | 0.6485 | 0.6989 | 124.1 / 295.4 |
| Hybrid + Qwen3 | 0.7681 | 0.8660 | 0.7202 | 0.6778 | 0.7147 | 1333.3 / 2127.7 |

Reranker-only p50/p95: **1172 / 1873 ms** (120 calls). Each embedding-enabled
mode recorded 120 cache misses and zero hits. The reranker reorders the same
ten candidates, so identical hybrid Recall@10 is expected, not evidence of a
recall improvement. It improves aggregate ranking here but loses to hybrid in
business/growth, paraphrase and section-specific nDCG@10. Dense leads the
multi-evidence stratum; BM25 leads narrative. Do not claim universal superiority.

See the immutable [run report](../artifacts/evals/retrieval/benchmark_v2/baselines/2d50cfe0dc7b624676b472b7407aab7dc11f9648/REPORT.md),
[complete overall/stratum/company metrics](../artifacts/evals/retrieval/benchmark_v2/baselines/2d50cfe0dc7b624676b472b7407aab7dc11f9648/summary.json)
and [provenance](../artifacts/evals/retrieval/benchmark_v2/baselines/2d50cfe0dc7b624676b472b7407aab7dc11f9648/manifest.json).
All 960 per-call power/browser samples satisfied the controls. Background
desktop/system CPU bursts were still recorded, so these are observed local
latencies, not isolated-laboratory timings or production SLOs.

## Honest claims

This is a source-first, AI-annotated benchmark, not independently human-
adjudicated ground truth. It supports reproducible **known-gold** retrieval
comparisons, not exhaustive recall of every possible relevant passage. Three
large technology/commerce issuers and paired topics across years limit
generalization and effective sample size. Small per-stratum latency tails are
descriptive, not stable service SLOs or statistical significance claims.

Current parser metadata errors are retained. Raw table representation differs
from PR20's summarized table/row corpus, so cross-benchmark deltas are invalid.
No claims about answer correctness, financial calculations, unanswerable
detection, company/year resolution or non-annual filings follow from this run.
Any later tuning must use a separately declared development/evaluation protocol;
this PR neither tunes retrieval nor selects cases where hybrid wins. Request
fresh Codex review after recording results; do not merge automatically.
