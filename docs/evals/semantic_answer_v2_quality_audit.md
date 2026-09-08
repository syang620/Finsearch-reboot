# Semantic v2 benchmark-quality audit before optimization freeze

Status: **quality review complete and optimization contract frozen**.
[Exact-head review](https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-5579446066)
found no major issues on `a73ce7535c00ab67d9b7c5ede65a18708f883840`.
The source/scorer candidate passed pre-calibration review at `d2abf07983`;
the subsequent frozen judge trial failed and full-population judging is disabled.
A subsequent baseline attempt captured two cases and was stopped by the user
after a workload violation; it is diagnostic-only, not a completed baseline.
See [baseline status](semantic_answer_v2_baseline_status.md).
This audit is a source/evaluation inspection,
not independent human certification of financial accuracy.

## Defects corrected and evidence boundaries

The review log traces the audit's numeric false positives, accepted-vs-error
channels, source-adjudicated atomic requirements, source-display/provenance
binding, fixed denominators, judge sensitivity and family reporting corrections.
All 60 v1 questions and 56 numeric targets/tolerances remain unchanged; 86 original
claims map to 119 required atomic facets. Two Microsoft records now retain the
complete source sentences supporting all three segment-growth facets. Generic
numeric questions admit nine independently re-extracted equivalent source facts;
explicit named-filing comparisons and calculations do not inherit those relaxations.

Gold uses original SEC HTML, independently extracted facts, canonical source
sections and source-linked KB chunks, never current answers or retriever rankings.
All six local filing byte hashes match the SEC primary documents. Listed evidence
is not an exhaustive semantic whitelist: equivalent unlisted evidence needs
logged source inspection under the frozen scope rule. Historical v1 data/artifacts
and all 343 pre-v2 evaluation files remain hash-unchanged. No production, provider,
prompt, retrieval, grounding, calculator, resolver, timeout or retry tuning occurs.

The 36-answer source-authored synthetic calibration is labeled before predictions
and explicitly described as assistant adjudication, not human annotation. The
candidate's 80.56% paired parse success, 56.41% claim agreement and 28.57%
unsupported recall fail preregistered gates. The disabled decision is required;
no full-population semantic metric may bypass it by selecting a channel label.

## Composition and residual limits

| Dimension | Composition / implication |
| --- | --- |
| Issuers and filings | AAPL, AMZN, MSFT; 20 cases per issuer, ten per each of six 10-Ks |
| Question families | 30 year-normalized strings across 60 questions; paired cases are correlated |
| Required claims | 119 facets with explicit v1 lineage; emitted-claim counts are a different denominator |
| Numeric concentration | 56 numeric requirements: revenue 36, growth six, cash six, margin two, leased area two, owned area two, R&D two |
| Strata | 12 structured-numeric cases; six each narrative, hybrid, comparison, calculator, multiple-claim, attribution, plausible-wrong-evidence and insufficient-data |
| Scope | 22 fact-period-equivalent numeric requirements, 34 named-filing-bound requirements |
| Source audit | 30 preselected baseline cases, ten per issuer; selection predates system answers |
| Abstention coverage | Future-actual questions under an earlier-filing restriction; not broad insufficient-data coverage |

Full requirement-family and shared-evidence-group membership is published in
`composition.json`. Sources overlap retrieval benchmarks. The sample is exposed,
technology/commerce-heavy and revenue-heavy; no independent holdout, sector-wide
generalization or confidence interval treating 60 questions as independent is
supported. Expansion was not selected using system success/failure.

The deterministic grammar is deliberately bounded. It can miss otherwise valid
free-form answers, so unknown/resolution coverage must accompany numeric rates.
Source-adjudicated semantics can resolve broader prose in the selected subset;
it cannot be extrapolated to the other 30 cases. Citation validity/type checks
and grounding-37 are not semantic entailment. Unsupported evidence is not proof
that a claim is factually false. A separate raw calculator-response ledger is not
exported by the unchanged runtime, limiting provenance claims to recorded calls,
bound operands, expressions and selected results.

## Metrics and future resume claims

Defensible after the controlled baseline and source audit are recorded:

- All-60 execution/substantive-answer, clarification/failure and detected service
  outcome counts, with unknown causes left unknown and failures retained.
- All-60 bounded deterministic numeric truth/verified-credit and structural
  citation/type metrics, explicitly paired with eligibility and resolution coverage.
- Source-audited subset support, partial/unsupported rates, required-facet
  completeness, grounded-answer and correct-abstention rates, with exact assessed,
  eligible and fixed-subset denominators. Do not call these all-60 accuracy.
- Detected explicit numeric-wrong answers with valid citations, labeled as bounded
  detection, not an exhaustive semantic wrong-answer or hallucination rate.
- Same-v2 paired before/after observations under controlled comparable conditions,
  with frozen questions, labels, scorer and disabled-judge policy; disclose changes
  in eligibility, grammar coverage, family concentration and environment.

Do not claim full-benchmark semantic accuracy, independent human validation,
unseen financial-QA generalization, causal v1-to-v2 system gains, production latency
SLAs or success inferred from structural grounding alone. The baseline runner
intentionally refuses runtime changes; a future authorized optimization experiment
needs its own implementation provenance while retaining the frozen evaluation
contract, not an in-place rewrite of this baseline or its scoring inputs.

## Remaining gate

Verification after calibration/decision publication: 1,139 tests pass, 47 subtests
pass, and the same two pre-existing planner/retrieval tests fail (25 warnings).
No source/script/test diff exists from the reviewed construction. Frozen input,
artifact reproduction, historical hash, privacy and whitespace checks pass.

The complete gold/scorer/calibration/decision evidence passed fresh review.
The optimization manifest and quality-approval record now bind that reviewed
candidate. The subsequent frozen implementation's attempted capture is now an
incomplete diagnostic, as described above. Failed power/workload,
model/index-integrity or capture/scoring completeness checks produce diagnostic-only evidence,
not an official baseline summary or automatic retry. Source-audit the fixed subset,
commit immutable evidence, complete fresh final review and stop without merging.
The original baseline and source-audit gates remain unmet; a replacement run
requires new explicit authorization and must preserve the diagnostic attempt.
