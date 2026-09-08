# Semantic Answer Benchmark v2

Status: **judge calibration completed and failed; full-benchmark judge disabled;
frozen for optimization; no v2 production baseline run yet**.
Built from PR30 merge `ef847550c80077bf9d785dc2694c1a9d6afb1ed3` on a
separate clean branch. No production behavior or historical artifact changes.

The original launcher stopped during preflight without starting a benchmark case.
A [separately versioned launcher correction](semantic_answer_v2_launcher_correction.md)
is awaiting its own review/freeze; the benchmark optimization contract stays unchanged.

## What changes, and why

The v1 benchmark-quality audit identified numeric substring false positives,
error-channel leakage, weak judge validation, ambiguous bundled requirements,
correlated coverage and contaminated execution conditions (B2–B6). V2 corrects
measurement, not the system. **V1→v2 scores cannot be called system improvement.**

- The same 60 questions retain exact wording, issuer, filing and answerability.
  Case IDs change deterministically from `SEM1_` to `SEM2_`.
- All 86 original requirements have explicit lineage into 119 required facets.
  The 56 numeric targets and tolerances are unchanged. Original consolidated
  source facts re-extract exactly from all six original filing HTML files.
- Apple custom-component sourcing does not require the extra new-product detail.
  Amazon receipt/due-payment triggers and revenue recognition are separate.
  Multi-driver, fulfillment-method and oversight/cadence requirements are split
  so omissions do not become imaginary false emitted claims.
- Original source IDs remain inspectable. Adjudicated narrative spans repair
  v1 sentence truncation at decimal points and ambiguous anchor offsets. Canonical
  SEC Item locations are recorded separately from faulty historical chunk headers.
- A new 225-link source catalog connects all 80 original numeric catalog facts
  to KB tables through original inline-XBRL element IDs and immutable table
  sidecars, never retriever ranking. This is annotation, not XBRL tool execution.
- A separate 68-table presentation catalog derives the unchanged runtime's
  hydrated display forms from those same source sidecars. Full source-table
  evidence must actually be visible; a correct stable payload hash alone cannot
  rescue altered or invisible displayed evidence. Prefix row text is allowed.

`claim_lineage.jsonl` preserves the original case/claim identity, canonical
UTF-8 JSON SHA-256 (`sort_keys=True`, `ensure_ascii=False`), new requirement IDs
and reasons. `source_references.json` and `historical_sha256.json` preserve
original source, corpus, v1 and historical artifact identities. No labels are
derived from new system answers or judge predictions. The coding assistant had
access to the v1 audit; this is not blind or independent human annotation.

## Scoring boundaries and denominators

Execution is classified before semantic scoring. A failed outer run, rejected
analyst result, grounding error or retained error text cannot become an eligible
answer. Completed degraded answers can remain eligible. Terminal precedence is
clarification → planner failure → analyst timeout → grounding fail-closed →
retrieval failure → tool failure → unknown failure. All detected signals remain
visible; detected service errors are an additional diagnostic flag, not causal
attribution from guessed free text. Abstention status is only a candidate until
its filing scope and answer content are adjudicated.

Numeric scoring separates:

1. Explicit financial meaning in the answer: entity, metric, fiscal year,
   value/tolerance, currency/unit, scale, sign and affirmative statement.
2. Evidence support and source/period binding.
3. Declared structured/KB evidence-type compatibility.
4. Calculator call, operands, expression and selected-result provenance.

The deterministic parser intentionally supports only a bounded single-assertion
grammar with explicit financial fields and an exact standalone answer assertion.
Negation, missing units, complex prose and unresolved source provenance remain
**unknown**, not guessed correct or wrong. It is not general semantic entailment.
Structured contexts lacking independently established original-filing provenance
can be numerically consistent but evidence-unknown; source assessment can accept
genuinely equivalent evidence under the frozen scope rules. Unlisted valid
evidence is not automatically irrelevant. Financially correct KB-table evidence
can pass truth/support while failing the separate structured-route policy metric.

USD targets retain ±500,000 dollars for amounts requested in whole USD millions;
percentage/area source targets retain ±0.000001 in their stated units; calculated
growth retains ±0.005001 percentage points for two-decimal display. Decimal
normalization accepts equivalent explicit scales, not wrong-currency magnitudes.

Calculator evaluation checks source-bound current/prior operands, a recorded
matching call, a bounded growth expression and selected result. The unchanged
runtime does not export an independent raw calculator-response ledger. Report
this as recorded trace provenance, not independent observation of tool execution.

Every rate must expose numerator/denominator. Fixed numeric-gold denominators
include all expected numeric requirements for unconditional verified-credit
rates. Conditional denominators include requirements in eligible answers, with
missing/unknown outcomes disclosed. Resolved-only correctness must accompany its
resolution coverage. Execution losses are unassessed requirements, not wrong
claims. Emitted-claim support rates must be paired with fixed-required-facet
coverage and whole-answer measures to resist claim-splitting gains. Citation-ID
validity, type compatibility and PR6 structural grounding are not semantic truth.

Structured source binding uses the independently verified six-filing identity
catalog: original local HTML bytes match the SEC primary-document SHA-256;
accession, URL, filing report date and filed date come from the SEC filing index.
The scorer checks these runtime-exposed fields, plus the fact's start date and
financial meaning. It never requires a nonexistent runtime `source_sha256`.
Filing report date denotes the anchor filing, not the end of a comparative fact.
Missing or unadjudicated filing identity remains unknown; a matching value or
invented local source hash cannot substitute for provenance.

Source-only scope adjudication distinguishes 22 generic numeric requirements
from 34 explicitly filing-bound requirements. Nine equivalent inline-fact
alternatives were re-extracted from the six original filings for generic
questions. Generic questions do not silently acquire a filing-year restriction
absent from the question. Explicit comparisons/calculations and named-filing
disclosures remain filing-bound. Unlisted equivalent sources require logged
source inspection; they are not automatically irrelevant or automatically valid.

The full-population semantic channel requires an enabled judge decision bound
by the optimization manifest. A caller-provided channel name is not permission
to report a disabled judge. Baseline completion likewise requires complete
unique capture/scoring, no control violations, and successful unchanged-model
and index checks. Any failed check preserves the raw run as `invalid_diagnostic`
and withholds the official summary; it does not authorize another attempt.

## Judge calibration, not judge certification

The preregistered plan defines acceptance gates before predictions. The draft
calibration set has 36 source-authored synthetic answers, including 19 fully
supported, 6 partially supported and 14 unsupported emitted claims, 2 grounded
but incomplete answers, 3 correct abstentions and 3 incorrect refusals. Twelve
identical repeats are selected before calls. This is not a random production
sample or an estimate of production error prevalence.

Pre-calibration review added three partial-fulfillment labels, two off-topic
answers and two answers with unbound factual prose. Requirement-level agreement
and partial-fulfillment recall are independently gated, as are recall on both
non-default answer-wide flags. Whole-answer booleans cannot mask the wrong
missing-facet decision. This amendment occurred before any judge predictions.

The generic cross-filing positive and named-filing rejection are explicitly
tested and individually gated. A high overall agreement score cannot hide a
failure of either scope rule. This replaces one redundant wrong-metric example;
all required numeric adversary classes remain represented before prediction.

Support packets contain only cited visible evidence, not gold requirements or
uncited-context rescue. Completeness packets contain atomic requirements and
the answer, not source evidence. Strict schema checks enforce exact unique IDs,
real booleans and verbatim quotations from the correct supplied channel.
Completeness quotations must occur in final-answer prose, not merely claim
metadata. Quotation validity does not itself establish entailment.

The only preregistered candidate is existing local `gemma4:e4b`, exact digest in
`judge_config.json`, unchanged candidate settings, one attempt per phase. Any
failed gate disables unattended full-benchmark judging under this candidate
policy. No candidate/rubric search against the final baseline. Failure does not
prove that every available judge would fail. Calibration labels/rubrics are
separately hashed before predictions; optimization freeze occurs later, after
judge decision and benchmark-quality audit.

## Composition and claims

There are 20 cases per issuer (AAPL, AMZN, MSFT), ten per filing across six
10-Ks, 12 structured-numeric cases and six in each of eight other strata.
Only 30 year-normalized question strings exist. Revenue (36) and revenue-growth
(6) account for 42/56 numeric requirements. Shared evidence groups and all
requirement-family, metric, stratum and filing counts are published in
`composition.json`. Technology/commerce issuers and future-actual abstention
questions dominate their respective categories; fiscal calendars differ, but
this is not broad sector/calendar coverage. Sources overlap retrieval benchmarks.
The official deterministic summary includes all 30 question-family breakdowns
and all requirement-family counts, case-level execution and family-restricted
numeric checks. Non-numeric fulfillment remains a separate semantic assessment;
overlapping family counts must not be summed into independent-trial claims.

Scope expansion was declined before any v2 results to keep this correction PR
bounded, not to select where the system wins. Future resume claims may describe
exact observed performance on this frozen exposed SEC-research sample, separating
availability, conditional quality, assessment coverage and same-version paired
before→after changes. Do not claim unseen generalization, broad financial QA
accuracy, production latency SLAs, independent human validation or causal gains
from v1→v2 evaluator/environment changes. Final metric recommendations await the
controlled baseline and audit; no current performance claim is approved yet.

## Calibration freeze checkpoint

The [clean pre-calibration review](https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-5579057991)
inspected exact construction `d2abf079837809148d5a43050d957a61e4195783`.
Calibration inputs are now frozen before any predictions in
`validation_manifest.json`, SHA-256
`24da18f62d51c256670d25dd8018482d3dc579eea4112afcb2209559a6d7ebfc`.
Dataset SHA-256 is `f9148a29cb5b0f2da30b6ecf5017a9d6bc4dc9fd31523643557b93906646ad7c`.
That calibration freeze was not optimization approval. The subsequent
[clean quality review](https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-5579446066)
inspected exact candidate `a73ce7535c00ab67d9b7c5ede65a18708f883840` after
the failed judge decision. The separate `optimization_manifest.json` now freezes
labels, scoring and disabled-judge policy, with SHA-256
`653a65e778a5633c4d1adc2d90ea7a56a578b81695e69d679f9d69915e54bec0`.
Any future correction requires v3; optimization must retain this contract.

## Remaining release gates

The single judge validation is complete and failed; see
`semantic_answer_v2_judge_validation.md` for all gates and immutable evidence.
Full-benchmark judging is disabled. Quality review and optimization freeze are
complete. Remaining: one controlled 60-case unchanged-system baseline;
source assessment of the predetermined 30-case subset; denominator/privacy/hash checks;
immutable evidence and fresh final Codex review. Do not merge automatically.

Initial checkpoint verification: 102 focused tests passed. Full suite: 1,080 passed, 47 subtests
pass, 2 unchanged pre-existing failures (`alias_002` planner route and retrieval
no-tool-call attempt count), 25 warnings. No dependency changes. These are draft
development checks, not a release gate or evidence that semantic accuracy passes.
See the review-fix log for subsequent checks and pending gates.
