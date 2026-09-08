# Retrieval Benchmark v3: corrected baseline report

## Publication correction (PR30 final-review P2)

This separately versioned report supersedes the reader-facing interpretation in
the [original frozen report](../../baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/REPORT.md).
It addresses [review finding 3953494401](https://github.com/syang620/Finsearch-reboot/pull/30#discussion_r3953494401)
on evidence head `77d95f9d140b98ab40946a902e97fed4abba6e29`.

The claim that a live pre-run service check establishes Qdrant 1.16.2 is withdrawn:
the frozen run artifacts do not capture a live server-version response. Version
1.16.2 appears only in archived build provenance. This baseline therefore does
**not independently establish the live Qdrant server version**. Collection
configuration and vector fingerprints remain verified; they do not prove server
binary identity.

No runtime, evaluator, dataset, raw observation, metric or latency changed.
No additional retrieval run was made. The original report and its artifact
inventory remain byte-identical historical evidence, including the now-withdrawn
claim. All raw filenames below refer to the original
`baselines/fc988918a0e4101196a21fb1642a7c9794f2e4fc/` directory. Its
`artifact_sha256.json` hashes the original report, not this correction; this
separately committed correction is identified by its review-head-keyed path and
Git history.

## Outcome and scope

Completed **480/480 unique query/mode pairs**, one pass, **zero retrieval errors**.
Six frozen exclusion cases are not scored. Offline verification recomputed all
per-query, stratum, issuer and grouped metrics, validated exact pair coverage,
and verified **323 unchanged historical dataset/artifact hashes**.

This is a **new known-label baseline**, not v2→v3 model improvement. The frozen
edition has 120 scored questions across AAPL FY2024/25, AMZN FY2023/24 and MSFT
FY2024/25: three technology/commerce issuers, six annual filings, 20 scored
questions per filing. Retrieval is given correct issuer/year/form metadata.
No XBRL execution, answer generation or semantic-answer evaluation was used.

## Freeze and provenance

- Evaluated implementation: `fc988918a0e4101196a21fb1642a7c9794f2e4fc` (clean worktree).
- Clean pre-comparison benchmark-quality review: `87073ab4fb50b9f99ddb1fbe9ab9cabf3ff1546a`;
  [actual review](https://github.com/syang620/Finsearch-reboot/pull/30#issuecomment-5577356419).
  Approval was committed before measurement and verified live against GitHub.
- Queries SHA-256: `308117369243451b0cdad9837beeecda541554bd7a8c8454df0886aa96d076c4`.
- Dataset-manifest SHA-256: `db89aa15436a82b66ec9636ae452901f1047c732baf1ea03364ccd0a8097ed2a`.
- Corpus SHA-256: `39c8d01ee5c71710e443e49698957d2aa991e71c16ad79905e7bd07d6a11fe4d`.
- Corpus: 948 documents (572 text, 376 tables). Dense embedding:
  `qwen3-embedding:8b`, local Ollama, digest
  `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`.
- Reranker: `Qwen/Qwen3-Reranker-8B`, DashScope API. The service supplies no
  immutable model digest; future service revision cannot be ruled out.
- Index: original read-only `finsearch_benchmark_v2_39c8d01ee5c71710`;
  original embedding-cache SHA-256
  `3bcd4ba8425d604973d694c3793eaae69f6e7c773c895504f5b4a7f57e4faea8`.
  Full served-vector/payload/config fingerprint matched the SHA-pinned earliest
  archived post-build v2 reference, including payload/vector digest
  `641f5ee5c465daaa7106717eb4e4a8a4e145cdfd04e4e8afd202a892d2e53630`.
  The old builder did not capture a served-vector digest at build completion:
  this proves identity to the historically evaluated index, not independent
  reconstruction of a pristine build.
- Both benchmark and historical production-index snapshots were unchanged
  before/after; embedding-model digest also unchanged.
- Run: 2026-09-08T00:54:12.764239+00:00 to 2026-09-08T00:58:13.019640+00:00;
  240.226 measured loop seconds.
- Host: Apple M4, 10 CPUs, 32 GiB, Darwin 25.6.0, arm64.
  Python 3.11.14; pytest 9.0.2;
  qdrant-client 1.16.1;
  requests 2.32.5;
  langchain-ollama 1.0.1.
  The archived build records Qdrant 1.16.2; live server version was not captured
  in the frozen run artifacts and is not established by this baseline.
  No dependencies were added for v3.

## Overall known-label quality

Values are on a 0–1 scale. Binary Recall/MRR use grade-2 evidence IDs; nDCG
uses graded gain. Group coverage counts required evidence groups with any
retrieved grade-2 alternative. Redundant valid chunks increase ordinary Recall's
denominator; group coverage must accompany evidence-sufficiency interpretations.

| Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Group coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7215 | 0.8014 | 0.6112 | 0.6041 | 0.6350 | 0.8292 |
| Dense | 0.6951 | 0.8215 | 0.6059 | 0.5818 | 0.6288 | 0.8792 |
| Hybrid | 0.7396 | 0.8750 | 0.6905 | 0.6510 | 0.7045 | 0.9208 |
| Hybrid + Qwen3 | 0.7771 | 0.8750 | 0.7327 | 0.6853 | 0.7213 | 0.9208 |

The reranker reorders the hybrid top ten; all **120/120** paired returned top-ten ID
sets were equal in this run. Identical Recall@10/group coverage is therefore not
evidence that reranking can improve candidate recall. Ranking at earlier positions
changed. These are observations of the four already-existing modes, not changes
or optimization made in this PR.

## Observational latency

Milliseconds; nearest-rank p50/p95, 120 observations per applicable column.
Total retrieval includes embedding/search/fusion/enrichment/reranking.

| Mode | Retrieval p50 | Retrieval p95 | Reranker p50 | Reranker p95 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 13.2 | 17.2 | — | — |
| Dense | 126.0 | 260.9 | — | — |
| Hybrid | 116.5 | 271.1 | — | — |
| Hybrid + Qwen3 | 1207.8 | 1667.3 | 1039.0 | 1477.0 |

All 960 before/after query samples met AC-power, Low Power Mode off and
zero-browser checks; the awake helper remained active for the run. Nevertheless,
recorded >=50%-CPU process samples included ChatGPT (193), biomesyncd (53),
Codex Renderer (21), Docker Desktop Renderer (20), DisplayLinkUserAgent (18), and
Python (3). Counts are process/sample occurrences, not durations or independent
workloads; the sampler cannot distinguish its own Python process. Thus this is
**not workload-isolated latency**, a production SLO, or statistically established
speed superiority. Preserve these observations without rerunning for nicer times.

The unchanged schedule rotates modes over a seeded shuffle. Each mode started
with its own empty query-cache namespace; all recorded query-cache hit counts
were zero. Model/process warm state and external service state are shared.
Candidate limit and rerank top-k both remain ten; branch limit 500, retrieval
top-k 50, equal-weight RRF k=60. No tuning or harness retry occurred.

## Correlation-aware descriptive summaries

There are 63 year-normalized question strings, 62 declared topic families,
36 family/shared-positive-evidence connected components, 102 filing-specific
required evidence groups, and 120 unique positive document IDs. These counts
are **not effective independent sample sizes**. The corpus/queries are already
visible; this is not an unseen holdout. Grouped values average within each group
then equally across groups, rather than letting repeated families dominate.
Issuer/filing macro-values equal query macro-values because their sizes are
balanced; detailed filing results remain in `summary.json`.

### Topic-family macro-average (62 families)

| Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Group coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7224 | 0.7997 | 0.6156 | 0.6088 | 0.6387 | 0.8266 |
| Dense | 0.6888 | 0.8112 | 0.6024 | 0.5792 | 0.6246 | 0.8669 |
| Hybrid | 0.7319 | 0.8710 | 0.6854 | 0.6461 | 0.7005 | 0.9153 |
| Hybrid + Qwen3 | 0.7762 | 0.8710 | 0.7167 | 0.6749 | 0.7097 | 0.9153 |

### Shared-evidence/family component macro-average (36 components)

| Mode | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Group coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7444 | 0.8345 | 0.6360 | 0.6280 | 0.6606 | 0.8681 |
| Dense | 0.6510 | 0.7930 | 0.5837 | 0.5654 | 0.6170 | 0.8253 |
| Hybrid | 0.7563 | 0.8796 | 0.6770 | 0.6569 | 0.7026 | 0.9028 |
| Hybrid + Qwen3 | 0.7928 | 0.8796 | 0.7301 | 0.6970 | 0.7293 | 0.9028 |

## Per-issuer quality

Each issuer has 40 scored questions. All six metrics and latency distributions
per issuer are also recorded in `summary.json`.

| Issuer / mode | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Group coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL / BM25 | 0.6792 | 0.7542 | 0.5125 | 0.5309 | 0.5602 | 0.7750 |
| AAPL / Dense | 0.7375 | 0.9042 | 0.6252 | 0.6209 | 0.6771 | 0.9500 |
| AAPL / Hybrid | 0.7479 | 0.9229 | 0.6592 | 0.6291 | 0.6988 | 0.9750 |
| AAPL / Hybrid + Qwen3 | 0.8833 | 0.9229 | 0.8653 | 0.8400 | 0.8529 | 0.9750 |
| AMZN / BM25 | 0.8458 | 0.9167 | 0.7389 | 0.7305 | 0.7546 | 0.9500 |
| AMZN / Dense | 0.7042 | 0.8042 | 0.6294 | 0.5993 | 0.6418 | 0.8250 |
| AMZN / Hybrid | 0.8125 | 0.9000 | 0.7873 | 0.7487 | 0.7866 | 0.9250 |
| AMZN / Hybrid + Qwen3 | 0.7792 | 0.9000 | 0.7253 | 0.6806 | 0.7258 | 0.9250 |
| MSFT / BM25 | 0.6396 | 0.7333 | 0.5821 | 0.5508 | 0.5901 | 0.7625 |
| MSFT / Dense | 0.6438 | 0.7562 | 0.5629 | 0.5252 | 0.5674 | 0.8625 |
| MSFT / Hybrid | 0.6583 | 0.8021 | 0.6250 | 0.5753 | 0.6281 | 0.8625 |
| MSFT / Hybrid + Qwen3 | 0.6687 | 0.8021 | 0.6075 | 0.5354 | 0.5852 | 0.8625 |

## Per-stratum quality

Direct-fact and risk-factor strata have 18 questions each; the other seven
scored strata have 12 each. Six unanswerable/exclusion cases are not included
in ranking means and do not establish abstention quality. The retained explicit
negative judgments are sparse (16 total); do not infer broad hard-negative
robustness from this deliberately limited set. Per-stratum latency and error
counts are in `summary.json`.

| Stratum / mode | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | Group coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| business_growth / BM25 | 1.0000 | 1.0000 | 0.9444 | 0.9583 | 0.9583 | 1.0000 |
| business_growth / Dense | 1.0000 | 1.0000 | 0.9167 | 0.9283 | 0.9283 | 1.0000 |
| business_growth / Hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| business_growth / Hybrid + Qwen3 | 0.9583 | 1.0000 | 0.9375 | 0.9203 | 0.9374 | 1.0000 |
| direct_fact / BM25 | 0.2870 | 0.4815 | 0.2179 | 0.2175 | 0.2866 | 0.5000 |
| direct_fact / Dense | 0.4861 | 0.7315 | 0.4454 | 0.3894 | 0.4807 | 0.8889 |
| direct_fact / Hybrid | 0.4259 | 0.6898 | 0.4005 | 0.3394 | 0.4500 | 0.8333 |
| direct_fact / Hybrid + Qwen3 | 0.6343 | 0.6898 | 0.5852 | 0.5493 | 0.5691 | 0.8333 |
| hard_negative / BM25 | 0.6667 | 0.6667 | 0.5000 | 0.5476 | 0.5476 | 0.6667 |
| hard_negative / Dense | 0.6250 | 0.8333 | 0.4667 | 0.4816 | 0.5536 | 0.8333 |
| hard_negative / Hybrid | 0.5833 | 0.9167 | 0.5316 | 0.5109 | 0.6197 | 0.9167 |
| hard_negative / Hybrid + Qwen3 | 0.8333 | 0.9167 | 0.5764 | 0.6321 | 0.6618 | 0.9167 |
| mda / BM25 | 0.9167 | 0.9167 | 0.7639 | 0.6404 | 0.6404 | 0.9167 |
| mda / Dense | 0.8333 | 1.0000 | 0.6910 | 0.5978 | 0.6697 | 1.0000 |
| mda / Hybrid | 0.9167 | 1.0000 | 0.8619 | 0.6793 | 0.7316 | 1.0000 |
| mda / Hybrid + Qwen3 | 1.0000 | 1.0000 | 0.9375 | 0.7665 | 0.7816 | 1.0000 |
| multi_evidence / BM25 | 0.5972 | 0.6667 | 0.6042 | 0.5605 | 0.5905 | 0.7083 |
| multi_evidence / Dense | 0.7222 | 0.7222 | 0.8750 | 0.7139 | 0.7139 | 0.8750 |
| multi_evidence / Hybrid | 0.5903 | 0.7153 | 0.7354 | 0.6008 | 0.6578 | 0.8750 |
| multi_evidence / Hybrid + Qwen3 | 0.5694 | 0.7153 | 0.8125 | 0.5798 | 0.6446 | 0.8750 |
| narrative / BM25 | 0.5833 | 0.7083 | 0.5104 | 0.5109 | 0.5554 | 0.7500 |
| narrative / Dense | 0.3333 | 0.5000 | 0.1682 | 0.2130 | 0.2670 | 0.5000 |
| narrative / Hybrid | 0.5000 | 0.6667 | 0.3322 | 0.3741 | 0.4255 | 0.6667 |
| narrative / Hybrid + Qwen3 | 0.5000 | 0.6667 | 0.4148 | 0.4385 | 0.4838 | 0.6667 |
| paraphrase / BM25 | 0.8750 | 0.9583 | 0.6536 | 0.6967 | 0.7245 | 1.0000 |
| paraphrase / Dense | 0.6250 | 0.7083 | 0.4181 | 0.4583 | 0.4823 | 0.7500 |
| paraphrase / Hybrid | 0.9583 | 0.9583 | 0.7181 | 0.7713 | 0.7713 | 1.0000 |
| paraphrase / Hybrid + Qwen3 | 0.8333 | 0.9583 | 0.7493 | 0.7178 | 0.7623 | 1.0000 |
| risk_factors / BM25 | 0.7778 | 0.9167 | 0.5759 | 0.5760 | 0.6373 | 1.0000 |
| risk_factors / Dense | 0.7500 | 0.9167 | 0.6765 | 0.6483 | 0.7150 | 1.0000 |
| risk_factors / Hybrid | 0.8333 | 0.9722 | 0.7778 | 0.7549 | 0.8078 | 1.0000 |
| risk_factors / Hybrid + Qwen3 | 0.8333 | 0.9722 | 0.8302 | 0.7889 | 0.8333 | 1.0000 |
| section_specific / BM25 | 0.9792 | 1.0000 | 0.9444 | 0.9364 | 0.9472 | 1.0000 |
| section_specific / Dense | 0.9583 | 0.9792 | 0.8403 | 0.8687 | 0.8796 | 1.0000 |
| section_specific / Hybrid | 0.9583 | 1.0000 | 0.9583 | 0.9325 | 0.9525 | 1.0000 |
| section_specific / Hybrid + Qwen3 | 0.8750 | 1.0000 | 0.7758 | 0.7912 | 0.8382 | 1.0000 |

## Errors, missing labels and unjudged evidence

| Mode | Explicitly irrelevant returned IDs | Unjudged returned IDs | Duplicate IDs | Missing corpus IDs | Wrong-filter IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 9 | 1049 | 0 | 0 | 0 |
| Dense | 2 | 1033 | 0 | 0 | 0 |
| Hybrid | 2 | 1032 | 0 | 0 | 0 |
| Hybrid + Qwen3 | 2 | 1032 | 0 | 0 | 0 |

All 480 retrieval calls succeeded. Zero missing labels means all frozen labeled
IDs resolve to this corpus, **not** that judgments are exhaustive. There were
4,146 unjudged returned-ID occurrences across 4,800 ranked slots (not unique
documents). They earn zero gain under known-label scoring but remain unknown,
not proven irrelevant. Full/partial/irrelevant label counts are 185/30/16
query-document judgments. Multi-evidence and equivalent-chunk handling is frozen;
a retrieved numeric table is not automatically a management explanation.

## Valid and invalid future claims

Defensible: built and source-adjudicated a versioned, multi-filing known-label
retrieval benchmark; measured the four unchanged modes on all 120 fixed questions;
report actual query-level and grouped metrics with the named corpus, filters,
model/provider, label and latency limitations.

A later before→after retrieval claim must use this same approved membership,
labels, corpus, scoring and all cases, record exact implementation/model identities,
and disclose conditions and any model/provider change. Any annotation correction
requires a separately versioned edition and symmetric comparisons. Do not tune
on only winning strata or remove inconvenient cases.

Do not claim exhaustive evidence recall, general SEC/financial accuracy,
semantic grounding or answer correctness, human-adjudicated gold, sector-wide
coverage, unseen-test generalization, 120 independent intents, abstention
performance, production latency/SLOs, or statistical significance from one run.
Do not call label-driven v2→v3 score movement model improvement. No production
retrieval tuning, prompt/model changes, threshold changes or automatic merge
belongs to this PR.

## Artifacts and verification

- `per_query.jsonl`: all 480 ordered observations, IDs/scores, timings,
  errors, controls and known-label classifications.
- `errors.jsonl`: empty (preserved, SHA-256 identifies the empty file).
- `started.json`: initial capture; its running status is historical, not current.
- `manifest.json`: final implementation/dataset/index/model/config provenance
  and raw-result hashes.
- `summary.json`: all overall, stratum, issuer, filing, family and component results.
- `verification.json`: successful offline recalculation/contract checks using the
  shared frozen scorer; separate arithmetic-oracle tests check 6,000 metric values.
- `artifact_sha256.json`: immutable file inventory including the original report
  (excludes itself to avoid a circular hash).

Focused verification: 65 tests passed. Full suite: 978 passed, 47 subtests passed,
two unchanged pre-existing failures (planner alias_002 routing and no-tool-call
attempt count), 25 existing warnings. Source/evaluator behavior did not change
after the clean review. Offline verifier passed on the complete new baseline.
Historical hash, diff and privacy checks are rerun before evidence publication.

Review history, including the requested fresh evidence-head review, is tracked
in [PR30](https://github.com/syang620/Finsearch-reboot/pull/30). This report records
the measured baseline; it does not authorize merge or retrieval optimization.
