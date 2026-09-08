# Retrieval benchmark v3 execution and freeze plan

Base: refreshed `origin/master` at
`25c15afbab31212a42c97e650e2418f6f82a8674`, after PR28 diagnostic evidence and
PR29 quality audit were merged. Work is on a new clean branch/worktree.

## Scope decision before annotation or measurement

Retain all 120 scored v2 queries and six exclusions over AAPL FY2024/25,
AMZN FY2023/24 and MSFT FY2024/25, with the identical 948-document corpus.
There will be no membership selection from retrieval outcomes. The defensible
scope is a metadata-filtered, multi-filing technology/commerce-company KB
benchmark, not a sector-diverse or representative SEC-research benchmark.
Expanding to 8–12 issuers requires source acquisition/normalization and a new
annotation campaign, beyond this validity-correction PR. Existing exposed cases
are not an unseen holdout. No question will be removed to inflate diversity.

## Ordered gates

1. Inspect source filings and candidate evidence independent of ranked outputs.
   Record adjudication for the four missing Microsoft table alternatives (two
   query cases, four evidence IDs), other justified alternatives, relevance
   grades, required facets, canonical source sections and correlated families.
2. Build new versioned dataset paths, exact parent-to-child membership/label diff,
   per-change reason codes, source/corpus hashes and historical integrity guard.
   Preserve every historical dataset, evaluator and artifact unchanged.
3. Add strict versioned scoring/validation, independent arithmetic/adversarial
   tests and group-aware summaries. The builder must never consume rankings.
4. Freeze annotation/membership with SHA-256 and commit. Obtain an explicit fresh
   benchmark-quality review of that exact candidate before any mode comparison.
   Findings require corrected candidate hashes and renewed review as necessary.
5. Freeze the tested implementation and unchanged comparison configuration.
   Run one complete four-mode comparison only after the review gate clears.
   Reuse the exact read-only corpus/index when compatible. No ranking parameters,
   embedding model, reranker, prompt, production settings or runtime changes.
6. Record all outcomes, query/stratum/issuer/family summaries, errors/unjudged
   coverage, observational latency and immutable SHA-keyed artifacts. Verify
   complete pair coverage and historical integrity. No v2→v3 model-gain claim.
7. Commit evidence/documentation, request fresh Codex review and address findings.
   Do not optimize or automatically merge v3.

## Annotation and independence limitations

The coding assistant is a source-inspecting annotator, not an independent human
financial reviewer. Prior v2 results are already visible in project history;
v3 adjudication must not use mode rankings or score deltas to assign labels.
Candidate discovery may use explicitly documented source terms; acceptance needs
query-specific source semantics, not global case-insensitive string matching.
Unjudged remains unknown, even where known-label metrics assign zero gain.

## Current state

The source-adjudicated candidate and harness passed renewed benchmark-quality
review on `87073ab4fb50b9f99ddb1fbe9ab9cabf3ff1546a`, after two P1 harness
guard fixes. Query SHA-256 remains
`308117369243451b0cdad9837beeecda541554bd7a8c8454df0886aa96d076c4`;
manifest SHA-256 is
`db89aa15436a82b66ec9636ae452901f1047c732baf1ea03364ccd0a8097ed2a`.
The verified review is recorded separately under
`artifacts/evals/retrieval/benchmark_v3/reviews/87073ab4fb50b9f99ddb1fbe9ab9cabf3ff1546a/annotation_approval.json`.
Implementation `fc988918a0e4101196a21fb1642a7c9794f2e4fc` completed one unchanged
four-mode baseline: 480/480 pairs, zero errors, successful offline verification.
Immutable evidence lives under the corresponding SHA in
`artifacts/evals/retrieval/benchmark_v3/baselines/`. Labels/corpus/index/config
were preserved. Latency is observational because background CPU bursts were
recorded despite all browser/power checks passing. Fresh evidence-head review is
tracked in PR30. No optimization or automatic merge.
