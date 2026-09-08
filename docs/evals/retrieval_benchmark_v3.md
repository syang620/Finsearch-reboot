# Retrieval benchmark v3: corrected known-label benchmark

## Scope and status

This is a new edition, not a relabeling of v2 in place. The parent is
`sec_retrieval_benchmark_v2`, queries SHA-256
`e356eab5c78f027c570daa05945b4688e152ab3e54c79070862d9759fc2cc9d9` and manifest
SHA-256 `83c31e4a307d68d77bb051658c4e45bec21764eb6b62e9974cfd151bab22adb3`.
The motivating audit is PR29, reviewed head `f190180c2e0e48d10a95f13d9d81082ca214d029`.

Retain all **120 scored queries and six exclusions**, across AAPL FY2024/25,
AMZN FY2023/24 and MSFT FY2024/25. Reuse the identical 948-chunk corpus and
read-only index; no chunking, production retrieval or model/configuration change.
Broadening to 8–12 issuers requires a separate source-annotation campaign. This
edition supports only a **metadata-filtered, multi-filing technology/commerce
company known-label retrieval benchmark**. It is not an unseen holdout or a
sector-diverse sample of SEC research.

At candidate construction, annotation review is pending and comparisons are
forbidden. Dataset files remain immutable after their commit. A separate,
committed review record must bind the exact dataset-manifest hash and reviewed
commit before the runner can measure anything. Later review/evidence commits do
not rewrite a dataset's candidate-status field to manufacture approval.

**Current status:** the renewed benchmark-quality review completed cleanly on
`87073ab4fb50b9f99ddb1fbe9ab9cabf3ff1546a` after both P1 guard fixes. The
[verified review](https://github.com/syang620/Finsearch-reboot/pull/30#issuecomment-5577356419)
and exact body hash are captured in
`artifacts/evals/retrieval/benchmark_v3/reviews/87073ab4fb50b9f99ddb1fbe9ab9cabf3ff1546a/annotation_approval.json`.
This clears only the narrow benchmark-quality contract above. Labels and
membership are frozen with their original v3 hashes. The one-pass baseline on
`fc988918a0e4101196a21fb1642a7c9794f2e4fc` completed **480/480 pairs with zero
retrieval errors** and passed offline verification. See the
[corrected baseline report](../../artifacts/evals/retrieval/benchmark_v3/report_corrections/77d95f9d140b98ab40946a902e97fed4abba6e29/REPORT.md)
for full metrics, grouping, hashes and limitations. Hybrid + Qwen3 recorded
known-label Recall@10 **0.8750**, MRR@10 **0.7327**, and nDCG@10 **0.7213**;
these are not v2→v3 model gains or semantic answer-quality metrics. Background
CPU bursts were recorded: latency remains observational. Fresh evidence-head
review is tracked in [PR30](https://github.com/syang620/Finsearch-reboot/pull/30);
no merge or retrieval optimization is authorized by recording this baseline.
The final review identified one unsupported live-server-version statement in the
original report. The separately versioned correction withdraws that statement;
the frozen artifacts establish archived build version and index fingerprints,
not live Qdrant binary identity. Original artifacts remain byte-identical, and no
runtime/evaluator/metric change or rerun was made.

## Source adjudication and v2→v3 changes

The new `data/evals/retrieval/benchmark_v3/` directory contains:

- `queries.jsonl`: explicit `KBV2_*` → `KBV3_*` identity mapping, unchanged
  questions/membership, enriched source and correlation metadata;
- `lineage.json`: all parent hashes, child query hash, exact ID mapping, zero
  membership additions/removals and reason codes;
- `label_changes.jsonl`: complete before/after judgment objects, including
  unchanged-grade provenance enrichments, not only changed scores;
- `adjudications.json`: source-inspected decisions, previous and corrected gold,
  reasons, equivalent evidence IDs and numeric-cell/quotation specifications;
- `source_sections.json`: original-filing section headings and normalized offsets;
- `corpus_ref.json`: explicit immutable parent corpus/source references;
- `historical_sha256.json`: preservation hashes for every tracked historical
  evaluation dataset/artifact at merged-audit master;
- `comparison_config.json`: byte-identical four-mode v2 configuration;
- `dataset_manifest.json`: SHA-256 of all dataset files, construction hashes,
  lineage and recomputed composition.

There are **12 newly explicit judgments**, six fully relevant and six partial.
No previously judged ID changes grade or evidence group. All six additions of
fully relevant query/document pairs concern IDs already present elsewhere in
the parent benchmark: the set of 120 unique positive corpus documents does not
grow. Annotation coverage grows, not corpus size or nominal question diversity.

| Parent cases | Evidence IDs newly judged | Grade | Source reason |
| --- | --- | ---: | --- |
| MSFT 2024 `_01` | tables 8, 72 | 2 | Original source reports consolidated revenue 245,122 and operating income 109,433, USD millions |
| MSFT 2025 `_01` | tables 8, 65 | 2 | Original source reports consolidated revenue 281,724 and operating income 128,528, USD millions |
| MSFT 2024/25 `_11` | same two tables per year | 1 | Numeric context, not management's causal explanation |
| AMZN 2023/24 `_20` | text 26 split 1 / split 2 respectively | 2 | Prepaid-Prime accounting facet: advance payment and recognition over subscription period |
| AMZN 2023/24 `_05` | same subscription passages | 1 | Recognition facet only; does not state the entire general received-or-due unearned-revenue rule |

The four Microsoft evidence-ID defects affect **two query cases**, not four
distinct questions. The omitted tables use `Operating Income`; v2's literal
anchor used `Operating income`. V3 names the four source-adjudicated IDs/cells
explicitly. It does not globally lowercase matching, infer labels from ranking,
or assume every similarly named financial measure is interchangeable.

Original Microsoft HTML tables were matched to their unchanged extracted
representations, with full normalized original-table spans and current-year
total rows recorded. All 376 table representations were checked against the
original HTML during source-provenance preparation. Amazon's additional policy
span also includes a separate explicit Prime-membership subject anchor.

Retained judgments are linked back to their original source passages/tables and
the stratum-specific review criteria in `adjudications_v3.py`. Canonical Item
locations come from source headings, not inherited chunker labels. All six
filings contain two occurrences of each inspected heading: table of contents
then body; preparation fails if that source-specific rule no longer holds.
This rule normalizes source typography (NFKC/whitespace) only, not relevance.

Some text chunks omit original page furniture, preventing whole-paragraph
identity. These records explicitly bind a literal source anchor, or an exact
80-character context shared by source and chunk, and disclose that narrower
binding. Repeated source passages can have multiple valid section locations;
section statistics count query/document/item memberships and disclose this.

This is **source-inspecting assistant adjudication**, not independently human-
blinded, dual-annotator financial gold. V2 results were previously exposed in the
project; v3 labels were not selected from mode rankings or score deltas. Source
inspection, not retriever output, supplied each correction. The changes are not
an exhaustive adjudication of all 120 × within-filing-document pairs. Remaining
unjudged results remain unknown; no exhaustive relevant-evidence-recall claim is
made. The quality review must assess this deliberately narrow contract.

## Relevance and alternatives

- **Relevant (2):** directly supplies the requested fact/explanation, or a
  complete independently necessary facet of a multi-evidence question.
- **Partial (1):** substantive related support that does not supply a complete
  requested facet, such as numeric context without the requested explanation.
- **Irrelevant (0):** explicitly inspected, similar-but-wrong evidence. A chunk
  containing both the correct and incorrect topic is not automatically negative.
- **Equivalent alternative:** a different source chunk that supplies the same
  complete required facet. Alternatives may share the same evidence group;
  they do not substitute for a different missing group.
- **Required evidence group:** a named necessary evidence need. Each needs at
  least one grade-2 alternative. Several groups may require several chunks.
- **Unjudged:** no frozen semantic judgment. Zero gain is only the known-label
  metric convention, not an assertion that the evidence is false or irrelevant.

An explicit table/statement or section restriction remains part of query scope.
Otherwise, genuinely equivalent table/text evidence is eligible regardless of
storage type. Do not manufacture a numeric-text alternative: source/corpus
inspection did not identify one for the audited consolidated totals. A numerical
table is not an alternative for causal narrative merely because numbers match.
Scope includes issuer, originating annual filing and fiscal year; later restated
segment comparatives are not silently interchangeable with earlier as-filed data.

## Correlation and effective diversity

Composition is balanced at 40 scored queries per issuer / 20 per filing. Strata
are unchanged: direct fact 18, risk factors 18, and 12 each narrative,
business/growth, MD&A, paraphrase, section-specific, hard-negative and multi-
evidence. Six future-audited-result questions are retained as exclusions, not
retrieval or abstention-score denominators.

There are **63 distinct year-normalized strings**, **62 declared topic families**,
**102 filing-specific required evidence groups**, and **36 connected correlation
components**. The Microsoft 2025 inventory→goodwill source-year overrides have
separate family identities; the Apple adopted-seed wording retains its matching
semantic family. These counts are descriptions, not estimated independent sample
sizes. The two years per issuer, shared passages and already exposed topics remain
correlated. Corpus composition stays 572 text chunks / 376 tables. Positive
query/document labels are 136 text / 49 table; grades total 185 relevant, 30
partial and 16 explicit negatives.

Report query-level metrics plus equal-group macro-averages by:

1. semantic topic family across years;
2. transitive connected component of same family **or shared positive evidence**;
3. filing;
4. issuer.

Each macro-average first averages all queries within a group, then gives each
group equal weight. It does not treat duplicated question templates as separate
independent statistical trials. No iid confidence interval, p-value, universal
significance claim or claim that 36 components are independent is made. The
component mapping is deterministic, uses labels/membership only and is frozen
before score observation; it is never regrouped based on wins or losses.

## Metric contract

Retain the independently tested v2 arithmetic: Recall@5/10 over known grade-2
chunk IDs; MRR@10 to first grade-2 ID; nDCG@5/10 with gain `2^grade - 1` and
rank discount `log2(rank + 1)`. Ideal DCG uses frozen judged grades only.
Duplicate returned IDs retain their rank positions but earn no repeated gain;
IDs outside the first ten cannot affect metrics. Partial labels get graded
nDCG credit, not binary hits. Evidence-group coverage@10 counts required groups
satisfied by any returned grade-2 alternative. Ordinary chunk Recall still
penalizes missing redundant alternative IDs: therefore never omit group coverage
when discussing evidence sufficiency. MRR does not imply multi-facet completeness.

Missing label IDs fail dataset validation. Returned missing-corpus IDs, filter-
incompatible IDs, explicit negatives, unjudged IDs and duplicate IDs are distinct
counts. Retrieval errors remain zero-quality observations, not omitted cases.
Full comparison requires all 480 unique query/mode pairs; partial capture is
explicitly incomplete. Empty aggregates are null, never perfect scores.

Latency uses the unchanged one-pass, rotated-mode schedule, initially empty
per-mode query caches and shared warm process/model state. Total retrieval
includes embedding/search/fusion/enrichment/reranking; reranker-only latency is
also reported. p50/p95 use nearest rank. Record power, awake assertion, background
workload, errors, model/service identity and index fingerprints. Latency remains
**observational**, even when sampled AC/Low Power Mode/browser checks pass;
these checks do not prove continuous laboratory isolation or service stability.

The index-origin guard is separate from the before/after mutation check. Before
any query, the completed build manifest must equal its immutable historical
copy, the sibling `embedded.jsonl` bytes must match that build's recorded hash,
and the live dense/sparse-vector, payload, point-ID and collection-configuration
fingerprint must equal the earliest archived v2 post-build snapshot. The reference
is `2d50cfe0dc7b624676b472b7407aab7dc11f9648/manifest.json`, SHA-256
`625047fc2cb5039ec0ee44af4979e7c2ee5bde32b6587a286d04658dce54219e`.
Its payload/vector digest is
`641f5ee5c465daaa7106717eb4e4a8a4e145cdfd04e4e8afd202a892d2e53630`.
This is a **historical post-build reference**, not a retroactively claimed
build-time snapshot: the old builder recorded the embedding-cache hash but no
served-vector fingerprint at completion. V3 proves identity to the previously
evaluated stack, not independent mathematical reconstruction of that index.
No current live fingerprint is promoted to expected gold or accepted merely
because it stays unchanged. These manifest fields are used only for index
provenance, never for relevance annotation or selecting benchmark cases.

## Claims after approval and baseline recording

Allowed: built a source-backed, versioned known-label KB benchmark; measured
four unchanged retrieval modes on 120 fixed questions, three named issuers and
six annual filings; report the actual known-label metrics with corpus/config,
grouping, error and latency qualifiers. Later ranking comparisons require the
same approved labels/membership/corpus/scoring contract, all cases and disclosed
implementation/model changes with comparable conditions.

Not allowed: v2→v3 score increases as model improvements; exhaustive evidence
recall; representative production traffic or sector-wide SEC performance;
unseen holdout, 120 independent intents, human-adjudicated labels, general
financial accuracy, end-to-end answer correctness, resolver/issuer-year routing
accuracy, abstention quality, semantic grounding, statistically significant
superiority from one run, or production latency/SLO claims.

The first v3 run is a **new baseline**, not a before→after gain. V2, PR20 and every
historical dataset/artifact remain unchanged. Any future annotation correction
needs a new edition and symmetric scoring of comparisons; approved v3 membership
or labels must not change during optimization. No optimization or automatic merge
belongs to this PR.

## Reproduction sequence

Use existing `finsearch-arm`; no dependencies were added. Run from repo root.
Source inspection and construction require unused output directories; never
rerun a builder over a frozen dataset.

```sh
PYTHONPATH=src:. python scripts/evals/retrieval/inspect_sources_v3.py --out NEW_INSPECTION_DIR
PYTHONPATH=src:. python scripts/evals/retrieval/build_benchmark_v3.py --inspection NEW_INSPECTION_DIR --out NEW_DATASET_DIR
PYTHONPATH=src:. python -m pytest tests/evals/test_retrieval_benchmark_v3.py -q --import-mode=importlib
PYTHONPATH=src:. python -m pytest tests -q --import-mode=importlib
```

Only after a matching benchmark-quality review record and clean implementation
freeze, run the unchanged stack once. The approval JSON must be committed at
the evaluated HEAD with byte-identical working-tree contents. It records
`status`, the full `reviewed_commit`, `dataset_manifest_sha256`, `pull_request`,
`review_comment_id`, `review_url`, and `review_body_sha256`. The runner uses the
authenticated GitHub CLI to verify the actual comment's repository, PR, author
(`chatgpt-codex-connector[bot]`), body hash and reviewed commit. The supported
protocol is the bot's “Didn't find any major issues” completion comment for
the open benchmark PR; a fabricated local status or URL is not approval.
Requested changes or inline Codex findings on that exact commit block execution;
all pages are checked. Prior-commit findings require a new clean reviewed commit,
not a local override. GitHub verification failure blocks comparison. The offline
verifier checks that the captured approval bytes were committed in the evaluated
implementation; it does not need to contact GitHub again.

```sh
PYTHONPATH=src:. python scripts/evals/retrieval/run_benchmark_v3.py --approval APPROVAL_JSON --index-manifest EXISTING_INDEX_MANIFEST --env-file LOCAL_CREDENTIAL_FILE
PYTHONPATH=src:. python scripts/evals/retrieval/verify_benchmark_v3.py --baseline SHA_KEYED_BASELINE --out NEW_VERIFICATION_JSON
```

Local absolute paths and credentials belong only in local execution notes, never
in committed documents or artifacts. Dataset hashes and immutable evidence links
will be recorded at the relevant freeze/publication stages.

## Annotation candidate submitted for review

Queries SHA-256:
`308117369243451b0cdad9837beeecda541554bd7a8c8454df0886aa96d076c4`.
Dataset manifest SHA-256:
`db89aa15436a82b66ec9636ae452901f1047c732baf1ea03364ccd0a8097ed2a`.

Initial submission: **31 new focused tests passed**. Full suite: **944 passed,
47 subtests passed, two pre-existing failures, 25 warnings**. Unchanged failures
are planner `alias_recognition/alias_002` and the retrieval no-tool-call attempt
count. Warnings are the existing table-render fallback and source-parser
dependency deprecations. **323 historical dataset/artifact hashes** verified
unchanged. The privacy scan found no local home paths or credentials; a token
pattern matched only the source phrase “risk-adjusted weighted average cost of
capital” after whitespace normalization, not a secret.

No v3 ranking results or baseline exist at this review stage. Passing tests and
a committed candidate do not themselves grant benchmark-quality approval.

The first review of `14121c5b7e0c9460885e93909c5ac72c3658b9c9` found one P1:
an uncommitted, fabricated approval JSON could bypass the pre-comparison gate.
The follow-up requires committed bytes and live GitHub review verification,
including paginated findings; offline evidence verification now binds approval
to the evaluated commit. Labels and their hashes are unchanged. Verification
after this fix: **53 focused tests passed; 966 full-suite tests and 47 subtests
passed, with the same two pre-existing failures and 25 warnings**. No ranking
comparison ran between candidates. A fresh exact-head review is required.

The next review of `208d86512c44e177b990f99b11859122d70e0cfd` found a second
P1: corpus IDs/payloads plus an unchanged-during-run index did not prove vector
identity to the historical build. The index-origin guard above addresses it
without rebuilding, querying, relabeling or altering any historical artifact.
The dataset hashes remain unchanged; source changes require another exact-head
review before comparison.

Verification after the index-origin fix: **65 focused tests passed; 978 full-
suite tests and 47 subtests passed**, with the same two pre-existing failures and
25 warnings. All 323 historical hashes remain unchanged; diff/privacy checks
passed. A read-only live snapshot and original embedding-cache hash matched the
frozen historical reference for all 948 points. This was provenance inspection,
not a query/ranking comparison.
