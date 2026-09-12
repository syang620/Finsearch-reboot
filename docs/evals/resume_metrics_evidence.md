# Resume metrics evidence

**Verified offline on 2026-09-10 and reverified for handoff on 2026-09-11 America/New_York:** all 480 query/configuration pairs, all recorded per-query and grouped metrics, and 323 historical hashes passed the existing verifier. No output-writing option, benchmark run or model call was used. The measured implementation is `fc988918a0e4101196a21fb1642a7c9794f2e4fc`; semantic corrections were reviewed at `a8188d667002a746558a41b1b327111f3126749e` and integrated with workload qualification v3 at `554b7b5b5eea837931bfb0d68736e023de2b4b87`.

Sources: [full-precision summary](../../artifacts/evals/retrieval/benchmark_v3/baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/summary.json), [480 observations](../../artifacts/evals/retrieval/benchmark_v3/baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/per_query.jsonl), [run manifest](../../artifacts/evals/retrieval/benchmark_v3/baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/manifest.json), [corrected report](../../artifacts/evals/retrieval/benchmark_v3/report_corrections/77d95f9d140b98ab40946a902e97fed4abba6e29/REPORT.md), [offline verifier](../../scripts/evals/retrieval/verify_benchmark_v3.py). The corrected report supersedes the original report’s unsupported live Qdrant-version statement. Archived build provenance does not establish the live server binary version.

## Scope and identities

The frozen [v3 dataset](../../data/evals/retrieval/benchmark_v3/queries.jsonl) has 126 records: **120 scored queries × four configurations = 480 pairs**, plus six excluded questions outside every retrieval-score denominator. Each mode has 120 observations and zero retrieval errors. AAPL FY2024/25, AMZN FY2023/24 and MSFT FY2024/25 contribute 40 scored cases per issuer / 20 per filing. The corpus contains 948 chunks (572 text, 376 tables). Correct issuer/year/form metadata is supplied; routing, answer generation, abstention and XBRL execution are not evaluated. These scope counts are verified in the run summary and documented in the [corrected report](../../artifacts/evals/retrieval/benchmark_v3/report_corrections/77d95f9d140b98ab40946a902e97fed4abba6e29/REPORT.md).

The [frozen configuration](../../data/evals/retrieval/benchmark_v3/comparison_config.json) defines BM25-only, dense-only, equal-weight RRF hybrid (k=60), and hybrid plus `Qwen/Qwen3-Reranker-8B`. Dense embeddings use `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`. Retrieval top-k is 50, branch limit 500, rerank candidate limit and rerank top-k are ten. The DashScope reranker supplies no immutable model digest. Four existing configurations were compared; this is not evidence of personal implementation ownership.

Queries SHA-256: `308117369243451b0cdad9837beeecda541554bd7a8c8454df0886aa96d076c4`. Dataset manifest SHA-256: `db89aa15436a82b66ec9636ae452901f1047c732baf1ea03364ccd0a8097ed2a`. The [manifest](../../artifacts/evals/retrieval/benchmark_v3/baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/manifest.json) binds the frozen corpus, committed annotation approval, implementation sources, model and index fingerprints.

## Absolute scores

Values below retain the summary’s full serialized precision. Each metric is an equal-weight mean over 120 scored queries per mode; it is not pooled over 480 independent questions. Binary Recall/MRR use grade-2 labels; nDCG uses graded gain `2^grade - 1` and logarithmic rank discount. Group coverage counts required evidence groups with at least one retrieved grade-2 alternative. See the [metric contract](retrieval_benchmark_v3.md).

| Configuration | recall@5 | recall@10 | mrr@10 | ndcg@5 | ndcg@10 | evidence_group_recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 0.7215277777777778 | 0.8013888888888889 | 0.6111673280423281 | 0.6040971856882893 | 0.6349680176663811 | 0.8291666666666667 |
| dense_only | 0.6951388888888889 | 0.8215277777777777 | 0.6058630952380952 | 0.581812324402492 | 0.628790583609386 | 0.8791666666666667 |
| hybrid | 0.7395833333333334 | 0.875 | 0.6904927248677248 | 0.6510202349330931 | 0.7045069810437656 | 0.9208333333333333 |
| hybrid_reranker | 0.7770833333333333 | 0.875 | 0.7326951058201058 | 0.6853487611621379 | 0.7213075420040126 | 0.9208333333333333 |

## Configuration comparisons

Calculated from `summary.modes[mode].overall.metrics[metric]` before rounding. Absolute change is `candidate − comparator` in score units; multiply by 100 for percentage points. Relative change is `100 × (candidate − comparator) / comparator`. The denominator is the comparator’s full-precision score, never the candidate score or the 480 pair count. Independent Fraction arithmetic agreed with Decimal arithmetic for all 18 comparisons (absolute/relative tolerance 1e-12).

| Comparison | Metric | Absolute change | Relative change |
| --- | --- | ---: | ---: |
| bm25_only → hybrid | recall@5 | +0.018055556 | +2.502406% |
| bm25_only → hybrid | recall@10 | +0.073611111 | +9.185442% |
| bm25_only → hybrid | mrr@10 | +0.079325397 | +12.979325% |
| bm25_only → hybrid | ndcg@5 | +0.046923049 | +7.767467% |
| bm25_only → hybrid | ndcg@10 | +0.069538963 | +10.951569% |
| bm25_only → hybrid | evidence_group_recall@10 | +0.091666667 | +11.055276% |
| bm25_only → hybrid_reranker | recall@5 | +0.055555556 | +7.699711% |
| bm25_only → hybrid_reranker | recall@10 | +0.073611111 | +9.185442% |
| bm25_only → hybrid_reranker | mrr@10 | +0.121527778 | +19.884534% |
| bm25_only → hybrid_reranker | ndcg@5 | +0.081251575 | +13.450083% |
| bm25_only → hybrid_reranker | ndcg@10 | +0.086339524 | +13.597460% |
| bm25_only → hybrid_reranker | evidence_group_recall@10 | +0.091666667 | +11.055276% |
| hybrid → hybrid_reranker | recall@5 | +0.037500000 | +5.070423% |
| hybrid → hybrid_reranker | recall@10 | +0.000000000 | +0.000000% |
| hybrid → hybrid_reranker | mrr@10 | +0.042202381 | +6.111923% |
| hybrid → hybrid_reranker | ndcg@5 | +0.034328526 | +5.273035% |
| hybrid → hybrid_reranker | ndcg@10 | +0.016800561 | +2.384726% |
| hybrid → hybrid_reranker | evidence_group_recall@10 | +0.000000000 | +0.000000% |

The resume figure is **13.6% higher known-label nDCG@10 than BM25**: `0.7213075420040126 − 0.6349680176663811 = 0.0863395243376315`, or 8.633952 percentage points. It compares configurations within v3; **it is not a v2→v3 system improvement**. V3 corrected labels while retaining the retrieval stack.

The same [summary](../../artifacts/evals/retrieval/benchmark_v3/baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/summary.json) also reports correlation-aware groups. Use these to avoid presenting one query-weighted mean as universal superiority:

| Mode | Topic-family nDCG@10 (62 groups) | Correlation-component nDCG@10 (36 groups) |
| --- | ---: | ---: |
| bm25_only | 0.6386787267739172 | 0.6606247901170761 |
| dense_only | 0.6246360486542445 | 0.6170042066698914 |
| hybrid | 0.7004540532422387 | 0.7026197155555086 |
| hybrid_reranker | 0.7097206787045234 | 0.7293455473307124 |

The [corrected report](../../artifacts/evals/retrieval/benchmark_v3/report_corrections/77d95f9d140b98ab40946a902e97fed4abba6e29/REPORT.md) shows reranking gains vary by issuer: AAPL improves while AMZN and MSFT nDCG@10 decrease versus hybrid. It also records that all 120 hybrid/reranked top-ten ID sets are identical. Reranking changed order, with **zero Recall@10 and group-coverage change**; it did not increase candidate recall.

## Observed latency tradeoff

Milliseconds, nearest-rank p50/p95, 120 observations per retrieval column. Full-precision inputs come from the same verified summary; display rounded to three decimals.

| Mode | Retrieval p50 | Retrieval p95 | Reranker p50 | Reranker p95 |
| --- | ---: | ---: | ---: | ---: |
| bm25_only | 13.209 | 17.229 | — | — |
| dense_only | 126.049 | 260.942 | — | — |
| hybrid | 116.451 | 271.137 | — | — |
| hybrid_reranker | 1207.838 | 1667.310 | 1039.000 | 1477.000 |

Hybrid-plus-reranking median total retrieval was 10.372× hybrid (1091.387 ms higher; 937.206% increase). This is a ratio of observed medians, not the median paired slowdown. The [run report](../../artifacts/evals/retrieval/benchmark_v3/report_corrections/77d95f9d140b98ab40946a902e97fed4abba6e29/REPORT.md) records one-pass rotated modes, separate initially empty mode caches, shared warm model/process state, and background CPU bursts. These are observational timings, not workload-isolated estimates, production SLOs or statistical speed claims.

## Claim limits and wording

The [annotation contract](retrieval_benchmark_v3.md) describes source-inspecting assistant adjudication, not blinded human gold. There are 63 year-normalized question strings, 62 topic families, 102 filing-specific required evidence groups and 36 connected correlation components; these are descriptive counts, not independent sample sizes. Labels and topics were already exposed. Remaining unjudged results are unknown and receive zero gain only under known-label scoring. The benchmark does not establish exhaustive evidence recall, sector-wide financial accuracy, unseen-test generalization or statistical significance.

Evidence-scoped resume wording (select only the activity supported by separate work records):

> Compared four retrieval configurations on 120 fixed, metadata-filtered SEC-filing queries across three issuers and six filings; hybrid + Qwen3 reranking recorded 0.7213 known-label nDCG@10 versus 0.6350 for BM25 (+13.6% relative), with 0.8750 Recall@10 and 0.9208 evidence-group coverage@10.

All numbers in that sentence are backed by the [verified summary](../../artifacts/evals/retrieval/benchmark_v3/baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/summary.json) and the arithmetic above. “Compared” is appropriate only if the candidate performed that work; benchmark artifacts alone do not prove authorship. Do not substitute “implemented,” “built,” “optimized,” or claim personal ownership without commits or other contribution evidence. A result-only alternative is: “In the project’s fixed known-label retrieval benchmark, hybrid + Qwen3 reranking scored 13.6% higher nDCG@10 than BM25 across 120 metadata-filtered queries.”

**Grounded-answer improvement and unsupported-claim reduction: unavailable.** Retrieval rankings cannot establish answer entailment. The [semantic-v1 quality audit](benchmark_quality_audit_v1.md) documents scorer/gold/judge defects; the [semantic-v2 judge decision](semantic_answer_v2_judge_validation.md) disables full-benchmark automated judging. [Prior baseline](semantic_answer_v2_baseline_status.md) and [fresh-attempt](semantic_answer_v2_fresh_attempt_status.md) records remain invalid diagnostics, and the [controller candidate](semantic_v7_r2_status.md) is not execution authorization.

Missing evidence before proposing either improvement claim:

- Valid complete paired before/after answer captures on identical frozen cases, labels, evidence corpus and scoring rules, with implementation/model/configuration identities and disclosed conditions.
- A valid controlled baseline and the preregistered 30-case source audit, with eligible, assessed, failed and unknown denominators retained.
- Reliable source-grounded claim-support/completeness judgments: validated judging or an explicitly specified source-adjudication procedure; current calibration cannot support full-corpus semantic claims.
- Explicit grounded-answer and unsupported-claim definitions and denominators, paired absolute scores and relative-change arithmetic, uncertainty/correlation limitations, and separate contribution evidence for personal ownership.

No answer-quality evaluation was launched. See the [catalog](evaluation_catalog.md) for the exact offline command, local backup and preservation checks.
