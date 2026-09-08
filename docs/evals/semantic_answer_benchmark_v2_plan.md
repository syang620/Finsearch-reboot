# Semantic Answer Benchmark v2: evaluation-correction execution plan

## Scope decided before new annotation or evaluation

Start from merged PR30 master `ef847550c80077bf9d785dc2694c1a9d6afb1ed3`.
Retain v1's 60 questions and six annual filings (AAPL 2024/25, AMZN 2023/24,
MSFT 2024/25) in a separately versioned v2. This PR repairs measurement validity;
a broader sector/issuer campaign requires additional source adjudication and is
not hidden inside these corrections. Preserve every existing dataset/artifact,
including semantic v1, the audit and retrieval v3. No production source, prompt,
model, retrieval, grounding, calculator, resolver, retry or timeout changes.

The resulting claims remain narrow, exposed-data, technology/commerce-company
sample claims. Sixty questions are not sixty independent information needs;
paired years, shared evidence and revenue concentration must remain visible.
Membership does not change based on current or historical system success.

## Ordered gates

1. Inspect the audit (B2–B6), v1 construction/scorer and original filings.
   Create explicit v1→v2 case/claim/source/scorer/judge lineage with old hashes.
2. Source-adjudicate all 60 cases / 86 original requirements. Separate atomic
   required facets from optional detail, especially Apple component sourcing
   and Amazon unearned-revenue triggers/recognition. Establish accepted paraphrase
   and equivalent-source rules, and valid scope-abstention conditions. Original
   filing scope and restatement distinctions remain binding.
3. Implement evaluation-only status-aware scoring. Positive numeric credit needs
   an accepted produced answer and an unambiguous entity/metric/period/value/
   unit/scale/sign/affirmation binding. Unresolved prose is explicitly unknown,
   never a guessed pass. Answer truth, evidence support, production evidence-type
   compatibility and calculator provenance are separate results. Error text is
   diagnostic only, never an answer channel.
4. Report execution outcomes separately from eligible-answer semantics. Publish
   unconditional fixed-case/fixed-gold rates and conditional rates with explicit
   denominators, unknown coverage and per-stratum/issuer/family summaries.
   No imaginary claims for timeouts; no removal of failed cases from E2E rates.
5. Freeze a source-adjudicated judge/scorer validation set before any judge calls.
   Use 36 prospectively authored answer fixtures: 12 numeric-focused, 18 narrative/
   attribution/completeness-focused, and six answerability fixtures (three valid
   abstentions, three incorrect refusals). Include at least six partial, ten
   unsupported and twelve fully supported emitted claims, calculator and equivalent
   evidence examples, incomplete grounded answers, and all B2 numeric adversaries.
   These are controlled validation fixtures, not sampled production answers.
6. Separate secondary support assessment (answer + only cited visible evidence;
   no gold rescue) from completeness assessment (question + atomic requirements
   + answer; no evidence-only fulfillment). Require exact claim-ID sets, typed
   schema, cited-evidence quotations for support and answer quotations for asserted
   fulfillment. Invalid responses are visible errors, never repaired silently.
7. Validate the one existing local judge candidate, `gemma4:e4b`, with its recorded
   digest and frozen evaluation-only settings/rubrics. One attempt per fixture/
   assessment phase, no retry or selecting favorable calls. Preselect 12 balanced
   fixtures for one identical repeat to assess stability; freeze their identities
   with the labels before the first call. Do not tune against final baseline output.
8. Apply the criteria below. If the candidate fails, keep full-benchmark automated
   judging disabled; report validation/manual subset evidence and limitations.
   Failure of this candidate does not prove no other model could work. Searching
   or tuning additional judge models is outside this initial candidate policy.
9. Obtain a fresh benchmark-quality review of gold/scorer/subset/judge decision
   before declaring the optimization contract frozen. Record dataset, source,
   corpus, scorer, rubric/config/model and manual-subset hashes. Any later gold,
   membership or scoring correction requires semantic v3, not in-place edits.
10. Freeze a clean implementation and capture all 60 current-system cases once:
    original planner/analyst models and settings, same frozen corpus/index/table
    sidecars, actual services healthy before launch, AC/LPM-off/awake protection,
    browser and heavy-workload controls. Record service identities directly, not
    infer live versions from historical index metadata. Preserve every failure;
    do not selectively rerun or raise runtime timeouts.
11. Source-adjudicate a preregistered 30-case baseline audit subset before any
    enabled full-baseline judge pass. Select three per ordinary stratum and six
    structured-numeric cases, balanced ten per issuer, before answers. Publish
    source-adjudicator provenance and explicit assessed-answer/claim denominators.
    If judge disabled, semantic estimates are restricted to this audited subset;
    no invented full-benchmark support rate.
12. Verify focused/full tests, denominator arithmetic, raw evidence/hash/privacy,
    immutable history and report values. Commit SHA-keyed evidence; request and
    complete fresh Codex review. Stop without production tuning or automatic merge.

## Predetermined judge acceptance criteria

All gates must pass on the fixed validation set. Report counts as well as rates.

- Complete schema/quotation/ID-valid case assessments: at least 95% of 36.
- Emitted-claim label agreement: at least 85% over all source-labeled claims;
  missing or malformed assessments count as non-agreement.
- Fully-supported precision at least 90%, recall at least 85%.
- Unsupported-claim recall at least 90%, including failed/missing judge labels
  as misses, not dropping them from the denominator.
- Partial-support agreement at least 80% over all gold-partial claims.
- Whole-answer groundedness agreement at least 90%; completeness agreement at
  least 85%, over all 36 answers. Invalid assessments never agree by default.
- Repeat label agreement at least 90% on the 12 preregistered repeat fixtures;
  report parse stability and case/claim coverage, including invalid results.

These are deployment gates for a **secondary** evaluator on a small adversarial
calibration set, not proof of human-equivalent judgment or broad generalization.
Source adjudication is performed by the coding assistant independently of judge
predictions; do not describe it as independent human financial annotation.

### Pre-calibration review amendment (before any judge predictions)

The independent review of draft `1c9c205` identified missing validation coverage.
Before calibration freeze, additionally require per-requirement fulfillment
agreement ≥85%, partial-fulfillment recall ≥80%, off-topic-answer recall ≥90%
and unbound-factual-prose recall ≥90%. Add three gold partial-fulfillment examples
and two positive examples for each answer-wide flag. These strengthen the gates
before observing results; they are not tuned to judge predictions. Preserve the
36 fixture count, 12 preselected repeats and all original acceptance gates.

A later pre-calibration review requires explicit cross-filing sensitivity. The
generic-revenue positive now cites the independently equivalent FY2025-filing
comparative fact; one redundant wrong-metric fixture becomes an explicit
FY2024-filing attribution with only FY2025 evidence (unsupported). The other
wrong-metric adversary remains. Require both cross-filing class gates to pass
100% in addition to overall agreement; aggregate agreement cannot mask either
scope error. The 36 answers, 39 emitted claims and 12 repeats are retained.

## Numeric and denominator contract to implement

Use Decimal-based normalization with explicit units and documented display
tolerances (retain v1 numeric target precision unless source adjudication proves
a validity correction). No magnitude-matching across currency, wrong metric,
wrong issuer/year, sign, negation or unspecified ambiguous scale. Canonical
equivalent USD/percentage/area formatting can pass only with correct semantics.
An accepted KB-table-supported fact may be financially correct even if its
declared route differs from structured gold; policy compatibility is reported
separately and PR6 itself is unchanged. Calculator-required claims must also
report bound operands and calculator-result/provenance checks.

Execution categories include substantive answer, abstention candidate/correct
abstention, clarification, planner/retrieval/tool failure, analyst timeout,
grounding fail-closed and unknown execution failure. Preserve multiple failure
signals with a documented primary-outcome precedence. Conditional numeric/support
rates exclude ineligible failures but disclose unknown assessments. Unconditional
rates retain all expected cases/requirements and are not relabeled accuracy when
truth remains unassessed. Fixed-gold coverage and whole-answer groundedness must
accompany emitted-claim rates to resist gains from claim splitting.

## Current state

PR30 merged; clean v2 branch/worktree created. Audit and v1 contracts inspected.
At this initial planning checkpoint, no v2 judge validation or current-system
baseline had run. This plan sets scope
and judge acceptance criteria before constructing labels or observing v2 results.
