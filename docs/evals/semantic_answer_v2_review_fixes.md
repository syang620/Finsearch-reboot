# Semantic v2 pre-calibration review log

Current post-calibration state is documented in `semantic_answer_v2_judge_validation.md`:
the one fixed trial completed and failed, full-benchmark judging is disabled,
and the production baseline has not run. The entries below describe earlier
development checkpoints, not a claim that calibration is still pending.

The post-calibration review of `2cac500` found one
[P2 stale top-level status](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954387619).
The current status now explicitly distinguishes completed failed calibration
from the not-yet-run system baseline. Earlier plan/checkpoint statements are
marked historical. The quality-audit wording also clarifies that incomplete
capture/scoring invalidates a run; poor semantic answer completeness does not.
These are documentation-only corrections; frozen evidence and evaluation code
are unchanged and no evaluation rerun is required.

The final pre-calibration candidate `d2abf079837809148d5a43050d957a61e4195783`
received a [clean Codex review](https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-5579057991).
Calibration inputs were then hash-frozen before the first judge call. The
development chronology below remains historical; no optimization freeze or
baseline approval is implied by this pre-calibration review.

The independent Codex review of draft `1c9c205` completed before any judge call,
calibration freeze or production baseline. The four findings are addressed in
evaluation-only code/data; renewed review is required. No score motivated these
changes because no v2 model results exist.

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| [P1 verified credit counted truth-only matches](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3953858276) | Verified-credit numerators use `credit`; parsed truth and resolved-only numeric correctness remain separately named. | Correct parsed number with unknown evidence earns zero verified credit. |
| [P1 canonical and hydrated table bytes differ](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3953858279) | Source-derived display catalog for 68 tables; visible full representation plus source identity, independent of rank/answer outputs. Synthetic KB fixtures use hydrated presentation with a row prefix. | All 68 forms match the unchanged renderer; mutated/absent displayed source and wrong issuer metadata cannot pass. |
| [P1 whole-answer completeness hid facet errors](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3953858282) | Per-requirement agreement ≥85% and partial-fulfillment recall ≥80%; three partial gold requirements added prospectively. | A judge with correct whole-answer booleans but wrong facet labels fails. |
| [P2 constant answer-wide flags were untested](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3953858286) | Two off-topic answers and two unbound-prose answers; ≥90% recall required for each non-default class. | Always-relevant/always-bound predictions fail the new gates. |

Other pre-calibration hardening: exact raw source values (display tolerance is
for answers only), explicit additional wrong-metric parsing, invalid extra
citation rejection, missing-metric compatibility rejection, outer-error answer
isolation, judge context/generation-capacity checks, fixed-population semantic
summaries and a baseline runner blocked on external review plus optimization
freeze. Neither source nor evaluator has been declared frozen for optimization.

The 36-fixture size, 12 preselected repeats, original support-label counts
(19 full / 6 partial / 14 unsupported), two grounded-but-incomplete answers and
six answerability fixtures are retained. New labels are source/rubric-validity
corrections before prediction, not labels changed to improve a measured score.
All 60 benchmark questions and all numeric targets remain unchanged.

## Files and checks

### Fourth review of `5c4d717`

[P1 complete quotes behind fragment anchors](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954189608)
identified missing source-adjudicated spans for the Productivity and Personal
Computing facets in Microsoft's FY2024 and FY2025 hybrid questions. The builder
now retains complete stored sentences behind fragment anchors, while rejecting
mid-sentence tails and decimal-truncated quotes. Duplicate sentence extensions
are deduplicated by exact offsets. Both revised source records contain all three
segment statements, re-corroborated against original filing HTML and canonical
SEC Item offsets. Reinspection of all original narrative anchor/quote pairs
found only these two case records change under the corrected construction rule.

Regressions require the specific supporting segment sentence for each facet and
reject the historical decimal/bare-word fragments. The 14-test dataset suite
passes, including byte-rebuild/source/history checks. No target, question,
membership or calibration-fixture changes. Current dataset SHA-256:
`f9148a29cb5b0f2da30b6ecf5017a9d6bc4dc9fd31523643557b93906646ad7c`.
The calibration fixture hash remains `4e42a4c8491057f305ca5c398b744fe1c1dd96eff8fa26a37bd4a8e16049e9e3`.
Full verification after this fix: **161 new tests pass** within the full suite;
**1,139 tests and 47 subtests pass**, with the same two pre-existing failures
and 25 warnings. Diff/privacy checks pass; no production/dependency changes.

### Third review of `43c9a1c`

- [P1 cross-filing judge coverage](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954119215): the generic numeric positive now uses its accepted comparative fact from the other filing. One redundant wrong-metric fixture becomes an explicit named-filing attribution with only the other filing's evidence. It is unsupported despite matching financial digits; current-year completeness and missing prior-year completeness are labeled independently. Both scope classes have individual 100% acceptance gates before predictions. Existing wrong-issuer/currency/scale/sign/negation/metric/period examples remain. The 36-answer / 39-claim / 12-repeat counts and support-label distribution remain unchanged.
- [P2 missing family summaries](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954119220): the baseline writer now emits the preregistered question-family and requirement-family breakdowns. Requirement-family numeric denominators include only that family's requirements; case execution stays explicitly case-level. Overlapping families are not independent samples.

The prospective support rubric explicitly distinguishes a comparative fact period
from filing identity, and scoped abstention explanations from additional unbound
factual assertions. No judge prediction informed these clarifications. Current
fixture SHA-256: `4e42a4c8491057f305ca5c398b744fe1c1dd96eff8fa26a37bd4a8e16049e9e3`.
Verification: **158 new tests pass** within the full suite; **1,136 tests and
47 subtests pass**, with the same two pre-existing failures and 25 warnings.

### Second review of `5ebd169`

This review also completed before calibration or any v2 model result. All three
findings are corrected in the unfrozen evaluation draft:

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| [P1 runtime cannot expose required source hash](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954002960) | New six-filing identity catalog binds source HTML hashes to independently inspected SEC accession, primary-document URL, report date and filed date. All six SEC primary-document downloads match the immutable local bytes. Structured scoring uses runtime-exposed provenance; synthetic fixtures now conform to the real structured contract. | Correct runtime-shaped evidence passes; missing/wrong accession, URL, dates and fabricated hashes cannot pass. Calculator source operands use the same binding. |
| [P1 disabled judge channel could be selected](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954002966) | Full-population semantic reporting verifies the enabled decision and hash-bound optimization policy. | Missing, disabled and modified policies reject; enabled policy hashes are reported. |
| [P1 invalid controlled run could publish summary](https://github.com/syang620/Finsearch-reboot/pull/31#discussion_r3954002970) | Final validity gate requires complete unique capture/scoring, no control violations and successful unchanged model/index verification. | Power/workload violations, failed/missing integrity checks and incomplete scoring become `invalid_diagnostic`; no official summary, raw evidence retained, no automatic retry. |

The filing identity catalog records direct SEC index/document URLs and the
independent byte-match verification date. Gold question text, numeric targets,
source facts and all 36 calibration labels remain unchanged. Only ten synthetic
fixture context records change to expose realistic structured provenance.
The new fixture SHA-256 is
`85e6ff7dd07ed4555a91b9b4f833b848df4f7afcec3e60920e01e3554130188b`.
The second-round focused suite has **154 passing tests** (153 in the first
focused run plus the subsequently added all-fixture runtime-contract regression,
which passes within the ten-test dataset suite). The full suite before that
last added test has **1,131 passing tests**, 47 passing subtests and the same two
pre-existing failures. No production/dependency diff or local-path leak exists.

### Change inventory

A subsequent source-only ambiguity check clarifies generic numeric scope before
calibration. Twenty-two numeric requirements ask for a fact period without an
explicit filing-year restriction; 34 remain named-filing-bound. Nine equivalent
inline facts from other original filings are explicitly cataloged for the generic
requirements. The same source cannot substitute in an explicit named-filing
comparison. All numeric targets, question text, membership and calibration labels
are retained. Case/claim lineage and shared-evidence composition are regenerated
deterministically; calibration fixture case metadata follows the corrected gold.
Current fixture SHA-256:
`744b71c2f3e2e0a877d142f1d77f0d5b1d05932391c5d0a424ea6f135e427ae9`.
This supersedes the draft fixture hash above before any predictions.
Verification after scope clarification: **155 new tests pass** within the full
suite; **1,133 tests and 47 subtests pass**, with the same two pre-existing
failures and 25 warnings. The regenerated draft is byte-reproducible. A temporary
patch-placement error in composition metadata was caught by the hash/rebuild
tests and corrected before commit; it did not change labels or any model run.

New benchmark data live only under `data/evals/semantic_answer/v2/`: queries,
claim lineage, composition, source references, historical hashes, numeric-source
links, source display forms, validation fixtures, two judge rubrics and config.
New evaluation modules: `semantic_numeric_v2.py`, `semantic_outcomes_v2.py`,
`semantic_answer_v2.py`, `semantic_dataset_v2.py`, `semantic_judge_v2.py` and
`semantic_metrics_v2.py`. Source-only builders, the calibration freeze/runner and
the guarded baseline runner live in `scripts/evals/agents/`. Matching focused
tests live in `tests/evals/`. Documentation is under `docs/evals/`.

Verification after fixes: **129 new focused tests pass** within the full suite;
**1,107 full-suite tests pass**, 47 subtests pass, and the same two pre-existing
planner/retrieval tests fail (25 warnings). The source-only draft rebuild is
byte-identical, including the historical snapshot's original path ordering.
Production and dependency-file diffs against merged PR30 are empty.

Commands run with the existing `finsearch-arm` environment (no dependency edits):

```sh
PYTHONPATH=src:. python scripts/evals/agents/build_semantic_dataset_v2.py --out-dir data/evals/semantic_answer/v2
PYTHONPATH=src:. python scripts/evals/agents/build_semantic_numeric_evidence_v2.py
PYTHONPATH=src:. python scripts/evals/agents/build_semantic_displays_v2.py
PYTHONPATH=src:. python scripts/evals/agents/build_semantic_validation_v2.py
PYTHONPATH=src:. python -m pytest tests -q --import-mode=importlib
git diff --check
```

During pre-freeze editing, revised fixture drafts were generated in temporary
directories and applied as explicit patches; the builders refuse overwriting an
existing dataset. Historical hash validation and source/corpus compatibility
checks run in the focused tests. At that pre-calibration checkpoint, no judge-run
or baseline-run command had been executed. Full changed-file inventory/diff is in PR31; production paths and
all pre-existing datasets/artifacts remain unchanged.
