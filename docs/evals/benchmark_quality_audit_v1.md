# Independent evaluation-quality audit: retrieval v2 and semantic v1

Audit target: `df69f522c1fa65f9393fa5c7f585722d336210d1`, 2026-09-07.
**Decision: not approved as optimization holdouts or for broad résumé accuracy /
before→after claims.** Retain these releases as immutable diagnostic benchmarks.
The retrieval benchmark supports narrowly named known-label statistics; the
semantic benchmark demonstrates evaluation infrastructure and concrete failure
discovery, not a reliable aggregate semantic-accuracy estimate.

This is a fresh source/code/history audit by an assistant, not independent human
financial adjudication, and not blind to the published baseline. The GitHub code
review of the exact PR28 head reported no major issues; that is not evidence of
annotation validity or external validity. This audit independently recalculated
composition, used a separately written arithmetic oracle, constructed adversarial
scorer inputs, inspected source HTML for concrete label omissions, and read the
recorded answer/judge/source-audit evidence. It did not re-annotate every source
pair or establish absence of all leakage or label errors.

## 1. Blockers and evidence

### B1 — Confirmed valid retrieval alternatives receive zero credit

`KBV2_MSFT_2024_01` asks for reported FY2024 revenue and operating income, without
requiring one particular table. Gold contains tables 7 and 18. **Tables 8 and 72
also explicitly contain total revenue 245,122 and operating income 109,433, in
USD millions**, but are unjudged and receive zero Recall/MRR/nDCG gain. Likewise,
`KBV2_MSFT_2025_01` omits tables 8 and 65, which explicitly contain totals
281,724 and 128,528. These are totals, not an inferred sum of segments.

Evidence IDs are `MSFT_10-K_2024::table::8`, `::table::72` and
`MSFT_10-K_2025::table::8`, `::table::65`. The corresponding source HTML tables
were inspected in `data/html_filings/MSFT/10-K/10-K_2024.html` and
`10-K_2025.html`. Their year/unit/total rows corroborate the corpus content.
These examples were found by source/corpus inspection, not chosen from a
particular mode's rankings. This audit does not calculate which mode benefits
from correcting them.

Cause: the `revenue` annotation group in
`scripts/evals/retrieval/benchmark_v2_annotations.py` requires the literal
`Operating income`; `freeze_benchmark_v2.py` uses case-sensitive string matching.
The omitted tables say `Operating Income`. Source anchoring prevents fabricated
IDs, but it does not establish complete relevance judgments. Merely lowercasing
every anchor is not a sufficient fix: source-specific row, period, qualification
and facet semantics still require adjudication.

Between **86.6% and 87.8% of top-ten returned IDs are unjudged**, not explicitly
irrelevant. This does NOT imply that all those results are relevant; it means
the reported zero missing-label count is referential integrity, not exhaustive
annotation coverage. B1 demonstrates that at least some unknowns are genuine
missed positives. Corrected source-complete alternatives are required before
representing the scores as evidence-quality improvements.

### B2 — Numeric score is not even a sufficient accepted-answer check

`src/evals/semantic_answer_v1.py:numeric_requirement` accepts a matching number
anywhere in a declared claim plus compatible cited structured metadata. The
audit's fixed Apple FY2024 counterexamples all score **1/1**:

- correct control: revenue is 391,035 million USD;
- negation: revenue is **not** 391,035 million USD; it is 1 million;
- wrong issuer: **Microsoft** revenue is 391,035 million USD;
- wrong currency: Apple revenue is 391,035 million **EUR**;
- wrong quantity: **operating income** is 391,035 million; revenue is 1 million;
- a correct candidate retained inside `ok=False`, `status=grounding_error`.

The first five share the same valid Apple revenue context; changing metadata is
not required to produce the false positives. `deterministic_case` computes
numeric matches before checking final acceptance. This is a scorer defect for
accepted-answer quality, not a demonstrated runtime exploit. Existing answers
are not alleged to contain every synthetic counterexample.

Conversely, legitimate KB-supported numeric claims fail the strict
`structured_numeric`/structured-source rule, and calculator gold requires
structured operands even when equivalent KB table operands exist. Thus changes
in declared output type or evidence route can raise the score without improving
financial truth. Keep production evidence policy strict, but evaluate answer
truth and policy compatibility separately. Do not call `numeric_consistency`
numeric accuracy; even “strict typed-gold reproduction” needs the acceptance
qualification and explicit number-presence limitation. Historical 0/56 remains
unchanged, not retrospectively repaired into a new baseline.

### B3 — Semantic judge has unusable coverage and unproven sensitivity

Only **4/21 emitted responses** received valid judgments; 17 judge attempts
failed and 39 other cases were not assessable. Only **7/20 emitted claims** were
judged. The 100% fully-supported figure is conditional on those seven survivors,
not a semantic quality estimate for the benchmark. Temperature zero and a frozen
rubric do not demonstrate repeatability. No repeatability experiment was run.

The preselected 18-case assistant audit covers nine emitted claims: six supported
and three unsupported. The judge/audit overlap is six **supported** claims, with
no valid judgments on audited unsupported claims. Therefore 6/6 agreement says
nothing about false-positive grounding or unsupported-claim sensitivity. This
is not an independently human-reviewed agreement set.

The frozen packet exposes gold requirement text and all visible contexts. The
rubric correctly forbids uncited/gold-only rescue, but the recorded
`SEM1_AAPL_2025_07` completeness disagreement credits a detail in context rather
than in the answer. Schema and verbatim-quote checks do not verify entailment.
The name `fully_grounded_answers_over_all_cases_lower_bound` is also too strong:
unassessed cases receive no credit, but fallible positive judge verdicts do not
mathematically bound true accuracy. Call this a conservative **judge-positive
count over all cases**, not a proven ground-truth lower bound.

### B4 — Gold completeness and partial support need adjudication

The Apple custom-component question asks how some custom components are sourced
and what supply/pricing exposure results. Its gold additionally makes
**new-product-specific** sourcing mandatory. A source-supported answer about
custom single-source components and pricing risk can reasonably answer that
question without discussing product novelty. This materially affects the
existing Apple 2025 completeness verdict. This is an ambiguity to adjudicate
from question intent, not a reason to credit the observed answer automatically.

The Amazon unearned-revenue requirement bundles payment timing and subsequent
recognition. Distinguishing a substantively supported facet from a contradicted
central trigger is judgment-sensitive. Break gold into atomic requirements and
freeze precedence examples for partial versus contradicted claims. Variable
model-generated claim bundling changes the emitted-claim denominator; splitting
one bundle into many true clauses can improve a claim rate without changing
answer content. Require both fixed-gold coverage and whole-answer support.

Multiple valid sources must be accepted by equivalent fact/period/unit/section
scope, not only canonical chunk identity. Conversely, an annual filing's
restated segment comparatives must not silently substitute for originally
reported segment figures when the question fixes the originating filing.
The present deterministic source binding does not establish a complete policy
for all such alternatives.

### B5 — Correlated construction and overlapping benchmarks limit claims

Issuer balance is good, but both sets use the same three large technology /
commerce issuers, six annual filings and narrow topic families. All **43 unique
semantic KB gold IDs are already retrieval-v2 positive IDs**. This is not evidence
that system answers supplied gold; it means the semantic set is not an independent
holdout for optimization performed on retrieval v2. Both have now been exposed.

Retrieval contains 120 scored strings but only **63 year-normalized strings**
(the intended 60 issuer/topic families have a seed wording and two source-year
overrides). Semantic contains exactly **30 year-normalized strings** for 60
questions. There are no exact duplicate question strings, but “no duplicate IDs”
does not mean independent information needs. A random question split would leak
paired topics and repeated evidence across development/test boundaries.

Semantic numeric gold contains 56 requirements, **42 revenue/revenue-growth
requirements (75.0%)**. Its comparisons are current/prior revenue within an
issuer, not cross-company or broadly varied research comparisons. All six
insufficient-data cases use the same later-year-actuals-from-earlier-filing
archetype. Hard negatives predominantly name the wrong measure in the question
(“not sales”, “not gross-profit dollars”, “not the tax credit”), which supplies
an explicit disambiguation cue. Retrieval filters the correct issuer/year/form
in advance: it does not test resistance to wrong-issuer/year candidates.

### B6 — Baseline conditions cannot support causal answer-quality gains

Semantic v1 emitted 12 substantive answers and nine insufficient responses; 39
failed or stopped for clarification. It records 19 analyst timeouts, service
errors, workload/power changes and eight large clock discrepancies. Restoring
service availability or comparable machine conditions alone could improve the
score greatly. Do not compare a future healthy run against this contaminated
run and attribute the entire gain to retrieval or analyst changes. Preserve the
run as observed, and establish a separately authorized, preregistered comparable
untuned baseline before claiming causal improvements. No rerun was made here.

## 2. Composition statistics

Counts below were recomputed from raw records, not copied from summary claims.

| Property | Retrieval v2 | Semantic v1 |
| --- | --- | --- |
| Cases | 120 scored + 6 excluded | 60: 54 answerable + 6 insufficient |
| Issuers | AAPL / AMZN / MSFT: 40 scored each | 20 each |
| Filings | 6; 20 scored each | Same 6; 10 each |
| Filing years | AAPL 2024/25; AMZN 2023/24; MSFT 2024/25 | Same |
| Exact duplicate question strings | 0 | 0 |
| Distinct year-normalized strings | 63 | 30 |
| Gold | 179 grade-2, 24 grade-1, 16 grade-0 query/chunk judgments | 86 required claims |
| Explicit hard-negative cases | 12 | 6 |
| Multi-evidence / multi-claim primary cases | 12 | 6 |

Retrieval strata: direct fact **18**, narrative **12**, risk factors **18**,
business/growth **12**, MD&A **12**, paraphrase **12**, section-specific **12**,
hard-negative **12**, multi-evidence **12**. Six future-result exclusions are
not scored for retrieval quality, latency or abstention.

Semantic strata: structured numeric **12**, and **6 each** narrative, hybrid,
comparison, calculator, multiple claims, difficult attribution, plausible wrong
evidence and insufficient data. Gold claim types: structured numeric **42**,
narrative **24**, attribution **6**, calculation **6**, KB numeric **8**.

The shared corpus has **948 chunks: 572 text, 376 table**. Retrieval positives
are **134 text and 45 table query/chunk pairs**. At query level, **94 text-only,
22 table-only and 4 mixed**. This is not table-dominated, but only four scored
queries require both table and narrative gold. There are 120 unique positive
chunks and 102 distinct positive-ID sets: 16 repeated sets span 34 questions.
77 queries have one grade-2 ID; 30 have two; 10 have three; 3 have four.

None of the 12 declared multi-evidence cases can satisfy all annotated groups
with one labeled chunk. The implementation correctly distinguishes alternative
IDs from required groups, but ordinary chunk Recall penalizes not returning
redundant alternatives: retrieving one of two equivalent chunks plus the other
required facet gives group coverage 1.0 but chunk Recall 2/3. Prefer group
coverage for evidence sufficiency; MRR alone never certifies multi-facet success.

Stored section paths are unreliable: **40/179 positive pairs** carry
`Item 1 > Note About Forward-Looking Statements`, despite covering other topics.
Full raw metadata counts are in the observations file; they must not be passed
off as validated SEC-section composition. A corrected release needs
source-verified canonical section labels independent of parser headings.

Semantic numeric requirements: revenue **36**, revenue growth **6**, cash **6**,
Services margin **2**, leased AWS area **2**, owned AWS area **2**, R&D **2**.
Only 18 distinct inline-XBRL fact IDs appear in required-claim sources, repeatedly
reused across queries. The 80-record source fact catalog is not 80 independently
tested facts.

## 3. Leakage and provenance findings

**No direct answer-to-gold or ranking-to-gold leakage found in inspected tracked
construction.** Retrieval annotations froze at `0cc20ce` before measured harness
`2d50cfe`; semantic labels froze at `b7d3004` before inference implementation
`3793929`. Source matchers consume filing HTML/corpus and authored anchors, not
ranked outputs. Numeric gold comes from consolidated inline-XBRL source facts,
not current SEC tool executions. Query files have one creation commit each in
the inspected history. These are meaningful strengths, not proof about an
annotator's unrecorded exposure or pretrained-model memorization.

Historical seed handling is transparent: 75 cases preserved, 92 evidence mapping
entries (7 exact anchors, 85 unresolved truncated anchors), one query adopted.
Those are evidence-entry counts, not seven compatible cases. No silent repair or
claim that the original 75 cases now run was found. PR20's corpus representation
differs, so subtracting PR20 scores from these results is invalid.

Source-derived questions can repeat the vocabulary of their own evidence;
the 12 named paraphrase cases mitigate but do not eliminate that construction
bias. The same assistant authored gold and source audits. Generalization and
independent human-review claims require evidence not present here.

## 4. Which metrics and claims are defensible now?

| Measure | Defensible interpretation | Not defensible |
| --- | --- | --- |
| Retrieval Recall@5/10, MRR@10, nDCG@5/10 | Exact recovery/ranking of this frozen, incomplete known-label set | Exhaustive relevant-evidence recall or general retrieval superiority |
| Evidence-group coverage@10 | Recovery of annotated facets under fixed filters | Complete answerability or all valid alternatives covered |
| Retrieval latency p50/p95 | Recorded local single-pass timings; reranker separately disclosed | Production SLO, isolated causal speedup, universal tail latency |
| Citation coverage and visible-ID rate | 20/20 claims cited; 23/23 references resolve | 100% factual or semantic correctness |
| Evidence-type compatibility | Declared claim/evidence contract on applicable emitted claims | Entailment, correct company/period/value; 0/0 structured is not a pass |
| Numeric score | Historical typed-source/number-presence diagnostic with B2 defects | Numeric accuracy or sound accepted-answer reproduction |
| Insufficient-data contract | 5/6 expected cases met the status/no-claims check | General abstention quality or semantic insufficiency precision |
| Assistant source-audit counts | Nine inspected claims, six supported and three unsupported, with case limitations | Population unsupported rate, human-reviewed accuracy |
| Secondary judge | Coverage/errors and conditional verdicts only | 100% grounding, validated judge accuracy, true-accuracy lower bound |

The audit independently checked **1,000 synthetic rankings / 6,000 metric values**
against a direct-summation reference: no arithmetic discrepancies in retrieval
Recall/MRR/nDCG/group coverage under the stated convention. Duplicate IDs occupy
rank without gaining repeat credit; grade 1 contributes nDCG but not binary hits;
errors remain zero-quality observations. Correct arithmetic does not cure B1.
Reranking only reorders the same ten candidates, so equal hybrid/reranked
Recall@10 is expected, not independent corroboration.

A safe résumé claim now is: **Built source-backed SEC retrieval and claim-level
evaluation harnesses across six annual filings; recorded immutable four-mode
baselines and identified citation-valid but unsupported answers.** Counts may
be included with the three-issuer and assistant-annotation qualification.

Do not claim “120 independent research questions,” “60 independent held-out
questions,” “human-validated,” “100% grounded,” “0% hallucinations,” “general SEC
accuracy,” “state of the art,” or a causal before→after accuracy/latency gain
from these releases. Structural grounding-37 is still structural regression
coverage, never semantic correctness.

## 5. Required repairs before an optimization freeze

These are evaluation repairs, not ways to make the product score higher:

1. Create **new** retrieval-v3 and semantic-v2 candidates; never rewrite v2/v1,
   PR20, PR3–PR8 or any existing artifact. Carry lineage, per-case change reasons,
   source spans and a machine-readable old→new ID/label diff. The names here are
   proposed successor versions, not claims that corrected datasets exist.
2. Audit alternatives source-first across every relevant filing, beginning with
   B1. Freeze canonical SEC sections, fact/facet identity and partial relevance.
   Have a reviewer blinded to mode identity/rankings adjudicate ambiguous cases.
   If result pooling is used only to propose candidates, use all modes symmetrically
   and source-adjudicate them; never infer grades from rank. Do not drop hard cases.
3. Separate optional detail from required atomic claims. Adjudicate B4 and scope /
   equivalent-source rules before inspecting corrected scores. Preserve legitimate
   structured and KB evidence routes in semantic truth scoring while reporting
   PR6 compatibility separately; do not weaken PR6 or rewrite factual output.
4. Version the evaluator. Credit accepted answers only; distinguish numeric
   metadata/value presence from actual claim quantity-role/unit/period truth.
   Fail the five B2 non-control tests. If free-text truth cannot be determined
   deterministically, report unknown and use an adjudicated semantic channel,
   not guessed certainty. Keep fixed-gold and whole-answer denominators alongside
   emitted-claim rates. Rename the fallible-judge “lower bound” measure.
5. Calibrate the secondary judge on a separate fixed, independently reviewed
   set containing supported, partial, unsupported, refusals and alternative-source
   examples. Isolate cited-evidence support from gold-based completeness; measure
   schema coverage, class-wise sensitivity, disagreements and repeated-run
   stability before trusting aggregate judge metrics. This is evaluation-only
   future work, not permission to tune production or tune against baseline wins.
6. Group correlated question/evidence families and filings for analysis and
   development/test separation. Broader claims need prospectively chosen unseen
   issuers/sectors/filings, less answer-cued questions, cross-company comparisons,
   and varied insufficient-data conditions. Existing exposed data cannot be made
   genuinely unseen by assigning a new split name. Do not select membership by
   which current method wins; use a source-led coverage specification.
7. Freeze/hash approved membership, labels, rubric, scoring, corpus and execution
   protocol **before optimization**. Use clustered uncertainty analysis, fixed
   family/issuer weighting, complete-case reporting and a comparable untuned
   baseline. Versioned annotation repairs require rescoring both sides identically
   (or matched reruns if outputs/corpus are incompatible), not comparing a v2 old
   score with a v3 new score. Do not select a favorable run or replace failures.

**No corrected dataset is declared frozen or approved by this audit.** The
observations contain `optimization_freeze_approved: false`. Approval remains
blocked pending the source-adjudication and scorer/judge repairs above. This
avoids publishing four easy label additions as if they constituted a complete
new ground-truth audit. The historical frozen editions remain byte-for-byte
unchanged; a future correction must acquire its own manifest and SHA-256 before
being used for optimization. No evaluation rerun, merge, or external write was
performed by this audit.

## Reproduction and local changes

New files only:

- this report;
- `scripts/evals/audit_benchmark_quality_v1.py` (read-only audit);
- `tests/evals/test_benchmark_quality_audit_v1.py` (historical defect reproductions);
- `artifacts/evals/benchmark_quality_audit/v1/observations.json` (versioned input
  hashes, composition, scorer counterexamples and arithmetic results).

The script's input manifest includes dataset, corpus, scorer, baseline and audit
script SHA-256 values. Its output must exactly reproduce the observations file.
The tests intentionally characterize the immutable old evaluator; they are not
acceptance criteria for a corrected future scorer. No dependencies were added.

Commands (existing `finsearch-arm`; run from repository root):

```sh
PYTHONPATH=src python scripts/evals/audit_benchmark_quality_v1.py
PYTHONPATH=src:. python -m pytest tests/evals/test_benchmark_quality_audit_v1.py tests/evals/test_retrieval_benchmark_v2.py tests/evals/test_semantic_answer_v1.py -q --import-mode=importlib
PYTHONPATH=src python scripts/evals/retrieval/verify_benchmark_v2.py --baseline artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f
PYTHONPATH=src:. python scripts/evals/agents/verify_semantic_v1.py artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb
git diff --check
git diff df69f522c1fa65f9393fa5c7f585722d336210d1 -- src data artifacts/evals/retrieval artifacts/evals/semantic_answer
```

Additional read-only checks: git status/history, PR28 review status, original
Microsoft HTML tables, annotation builders/rubric, baseline reports and raw
records. The tracked-file diff against the audited head must remain empty.

Verification results: **78 tests passed** (audit + retrieval-v2 + semantic-v1);
retrieval verification reproduced all **480 pairs**; semantic verification
reproduced all **60 cases**, including 17 judge errors and four valid judgments.
The protected runtime/data/historical-artifact diff is empty. Only the four new
audit files above were added locally; no commit, push or merge was performed.

Audit observations SHA-256:
`6c9a2efb4f544ef358a79fe475f90da18ea2e5049012a71b3a4ce4ff288540da`.
This is an **audit artifact hash**, not a corrected-dataset freeze hash.

### Publication preflight

Before preparing the audit PR, the full suite was rerun: **913 passed, 47
subtests passed, two pre-existing failures, one warning**. The unchanged failures
are planner `alias_recognition/alias_002` and the retrieval no-tool-call attempt
count. The warning is the existing table-render fallback without `tabulate`.
No production fix or benchmark rerun was attempted. The command was
`PYTHONPATH=src:. python -m pytest tests -q --import-mode=importlib`.
