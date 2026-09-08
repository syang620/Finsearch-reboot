# Semantic v2 pre-calibration review log

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
checks run in the focused tests. No judge-run or baseline-run command has yet
been executed. Full changed-file inventory/diff is in PR31; production paths and
all pre-existing datasets/artifacts remain unchanged.
