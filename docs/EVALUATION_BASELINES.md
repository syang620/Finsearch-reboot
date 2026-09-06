# FinSearch Evaluation Baselines

## PR8 release exception — owner approved 2026-09-06

The project owner explicitly authorized **PR8-GROUNDING-001** for
`AGENT_V1_ANALYST_001` and instructed final review of evidence/documentation head
`4a55cff`, merge of PR26 and refresh of `master`. Final Codex review of that
exact head completed on 2026-09-06 at 13:35:35 UTC. Its sole finding required
this superseding exception record before merging; this documentation addresses
that finding without changing runtime or evaluation results.

**The strict live gate failed and remains failed: 14/15 non-critical rows, one
critical row, score 0.981481 and zero evaluator errors.** This new exception
accepts only the unrelated analyst claim/row grounding failure
`ANALYST_GROUNDING_INVALID` / `GROUNDING_ROW_TEXT_MISMATCH` for the PR26 merge.
The answer remains unavailable and fail-closed; the exception does not make it a
correct calculation answer or convert the gate into a pass.

The exception does not cover filing/period selection, provenance, resolver,
routing, admission, lane semantics, timeouts or any other regression. It does not
weaken PR6 validation, extend retries or carry forward any PR7/PR24 waiver. It
expires with this PR8 merge and does not apply to PR9 or subsequent releases.

Runtime remains frozen at `99cf1b20e2796741df735365e78e241ea6667c2d`. Closure
changes only release documentation; no evaluation rerun is needed or performed.
The original raw evidence, manifest, controls and contemporaneous blocked
disposition below remain unchanged. Their historical "no exception/merge
authorized" statements are superseded only by this explicit owner approval.
See the [release disposition](../artifacts/diagnostics/pr8-live-controls/99cf1b2/RELEASE_DISPOSITION.md)
for scope, residual risk and final-review tracking.

## PR8 controlled live gate — 99cf1b2, 2026-09-06

Historical measurement and pre-exception stop disposition; see the subsequent
owner-approved PR8-only exception above. The failed measurement is unchanged.

**Release blocked: the unchanged strict gate failed, with 14/15 non-critical
rows.** The single run evaluated exact implementation
`99cf1b20e2796741df735365e78e241ea6667c2d` after fresh review, from a clean tracked
checkout. Deterministic score was **0.981481** against 0.90, but one critical row
violated the unchanged zero-critical-failure threshold. All 15 rows were valid;
there were zero evaluator errors and no terminal analyst timeouts. RAGAS was
disabled; no semantic-quality pass is claimed.

The sole critical case, `AGENT_V1_ANALYST_001`, failed closed with
`ANALYST_GROUNDING_INVALID` / `GROUNDING_ROW_TEXT_MISMATCH`. The calculator was
used, but the claim's full calculation sentence and its bound comparison row
remained inconsistent through the initial grounding decision and both retries.
The analyst returned no accepted answer/citations. This is not a recurrence of
`CALCULATION_RESULT_AMBIGUOUS`, nor a passing calculator answer: the live
compute-match rate is zero for this case. The failure class has prior PR24/PR7
history, but no prior exception applies and causal equivalence is not assumed.

Route, metadata, KB/structured lane matching, structured metric/status, lane
truthfulness, degradation and grounding consistency were **100%**. All seven
successful structured results became typed, analyst-visible evidence with full
measured provenance coverage; no synthetic structured-text contexts were used.
The dedicated 49-case PR8 matrix below remains the authority for annual-period
and amendment semantics, rather than this integration gate.

The run lasted **3,063,972 ms** (about 51 minutes); controller wall time was
3,070.766 seconds. All **188** samples reported AC power, Low Power Mode off and
no Chrome/Safari processes, with an owned awake assertion maintained throughout.
No control violation occurred; maximum sampling gap was 16.399 seconds. Workload
isolation was sampled, not exclusive: **40** samples contained transient heavy
non-model processes, but none reached the three-consecutive-sample abort rule.
Those samples are retained, not excluded. Model digests, dataset and corpus
matched preflight; the same 582-point index fingerprint matched before/after:
`6c72627b05dbec1e7f65d2132b8572b9fa12c47af711f2a12b8b776967994cee`.
The hosted reranker still exposes no immutable service-side digest.

Fresh review of `99cf1b2` completed before the run. The first review's P1
start-date renderer omission was fixed at that SHA; the subsequent P2 missing
verification-artifact reference was addressed by evidence commit `2803adf` and
verified remotely. Both review threads were resolved before launch. No runtime,
provider, retry, timeout, grounding or threshold change accompanied this run.

- [Live manifest and provenance](../artifacts/evals/agents/v1/baselines/99cf1b2/manifest.json)
- [Unchanged raw summary](../artifacts/evals/agents/v1/baselines/99cf1b2/summary.json)
- [Per-query traces and failed candidates](../artifacts/evals/agents/v1/baselines/99cf1b2/per_query.jsonl)
- [Execution controls and disposition](../artifacts/diagnostics/pr8-live-controls/99cf1b2/REPORT.md)

PR26 remains draft. **No rerun, scope expansion, exception or merge is
authorized by this result.** Preserve the original failed gate and stop pending
owner direction. Earlier SHA-keyed evidence, including PR7's 8/15 gate, is
unchanged.

## PR8 review correction — 99cf1b2, offline verification

Current evaluated implementation:
`99cf1b20e2796741df735365e78e241ea6667c2d` in draft
[PR26](https://github.com/syang620/Finsearch-reboot/pull/26).
The reviewed renderer omission is corrected with an additive start-date field;
the independent evaluator now checks actual analyst-visible provenance and has
a mutation test proving dropped starts fail the gate. Prompt instructions,
grounding rules, providers, timeouts and retry budgets remain unchanged.

- PR8 **49/49**, including analyst visibility, provenance, approved changes,
  unchanged parity and order invariance **100%**; unexpected differences **0%**.
- Resolver **170/170**, calculator **20/20**, output repair **20/20**,
  grounding **37/37**, degradation **14/14**, applicable consistency **100%**.
- Full suite: **832 passed**, **47 subtests**, **two unchanged known failures**,
  one known warning. All **113** focused PR8 tests pass.
- Python 3.11.14 and environment-only pytest 9.0.2; dependency files untouched.

[Current verification](../artifacts/evals/agents/v1/filing_period/baselines/99cf1b2/verification.json)
records source hashes, complete commands, changed files, test results and the
review correction. Previous 2933a5f/562e01e measurements are retained separately.
The fixture source and all frozen old/expected tool outputs remain unchanged.

At this offline measurement, fresh review and the live gate were still pending.
Their subsequent completion and failed-live-gate disposition are recorded above.
PR26 remains draft and is not approved to merge; no PR7/PR24 release exception
applies. The offline verification file remains the original measurement record.

## PR8 original-as-filed selection — pre-review offline verification, 2026-09-05

[PR26](https://github.com/syang620/Finsearch-reboot/pull/26) implements ADR 005
from merged PR25 `f09e884`. Evaluated implementation is
`562e01ebc4b387fc7a38bd3a2131e8ae11c01a51`. This measurement predates the review
correction: review of `2933a5f` found P1 missing start-date rendering in the actual
analyst prompt. The typed field alone was insufficient. A failing prompt regression
was reproduced before the additive renderer correction; the independent matrix
now also checks analyst visibility. **These results are historical, not final
release evidence. No PR8 live gate has run and no exception applies.**

The 49-case oracle was frozen at `e5ac8cf` before runtime edits. Two independently
specified capex labels were corrected to the unchanged registry at `434012c`,
also before runtime changes; no old observed results were altered. Exact expected
responses and allowed field paths distinguish 31 intentional core semantic
changes from 18 unchanged core outcomes. All responses receive explicitly
specified trace/start-date provenance additions; unchanged parity does not claim
byte equality for those approved additions.

At `562e01e`:

- PR8 matrix **49/49**: approved-change accuracy, unchanged-case parity,
  five-seed order invariance, independently checked provenance and network-call
  contract rates **100%**; unexpected differences **0%**.
- Resolver/execution snapshot parity **170/170**; calculator **20/20**;
  output repair **20/20**; grounding **37/37**; degradation **14/14**, consistency
  **100%**. The PR8 matrix, not the live gate, establishes period semantics.
- Full suite **831 passed**, **47 subtests passed**, **two known baseline
  failures**, one known table-renderer warning. Failures remain planner
  `alias_recognition/alias_002` and retrieval no-tool-call attempt count. No new
  failing tests and no blanket full-suite pass are claimed.
- Python 3.11.14 / pytest 9.0.2 in `finsearch-arm`; dependencies unchanged.
  The documented PR7 test-namespace bootstrap was required again; initial direct
  pytest collection errors were invocation/environment issues, not passing tests.

Source code is unchanged between initial implementation `2933a5f` and `562e01e`;
the latter strengthens independent evaluator checks and adds three tests. Both
sets of SHA-keyed offline evidence are preserved. The
[verification record](../artifacts/evals/agents/v1/filing_period/baselines/562e01e/verification.json)
contains hashes, commands, changed files and measured results. Other suites live
under their existing `artifacts/evals/agents/v1/<suite>/baselines/562e01e/` paths.

Live preflight restored the existing stopped evaluation Qdrant container and
verified the same 582-point index, model digests, corpus and dataset as PR7.
No index rebuild or new discovery call was added. Browser/workload isolation and
fresh review must be confirmed before the one live run. PR7-TIMEOUT-001,
PR7-GROUNDING-001 and PR24-GROUNDING-001 do not carry forward. PR26 is not ready
to merge while review/live evaluation remain outstanding.

## PR7 release exceptions — owner approved 2026-09-05

The project owner explicitly approved both PR7 exceptions and instructed:
"Approve. please proceed with commit, push, and merge."
This authorizes closure of [PR25](https://github.com/syang620/Finsearch-reboot/pull/25)
only, with frozen runtime `c149f73d073a306cf34d938955ae6cc739191528`.
Diagnostic evidence is committed at `59d3ac0ca6bedbbfb33f01d72c4f051cdaa0e39a`.

**The strict live gate remains failed: 8/15 non-critical rows, seven critical
rows, score 0.859390, zero evaluator errors.** No raw baseline, gate result,
threshold, runtime behavior or historical stop disposition is rewritten.

- **PR7-TIMEOUT-001:** accepts the six original analyst timeouts for this merge.
  The battery analyst-only replay reproduced 0/6 timeout-free cases; the AC/awake
  batch recovered 5/6, with browser-workload caveats preserved. A subsequent
  single browser-closed KB_002 run succeeded with a valid grounded answer in
  109.629 seconds at 10.613 tokens/s, under the unchanged 120-second limit.
  Its power/browser controls held, and no competing >=50%-of-one-core process
  was sampled. This supports environmental contamination, not a resolver bug.
  These separate diagnostics are not a replacement six- or fifteen-case gate.
- **PR7-GROUNDING-001:** accepts the original `AGENT_V1_HYBRID_001`
  `ANALYST_GROUNDING_INVALID`, with evidence-type and row-text mismatches, for
  this merge. It is the known PR24 failure class in unchanged analyst/validation
  code. The answer remains unavailable and fail-closed; no correct answer is
  claimed. This is a new, explicitly approved PR7 exception, not an inherited
  `PR24-GROUNDING-001` waiver.

Exact resolver parity (170/170), captured SEC execution/argument parity (7/7),
grounding (37/37), degradation (14/14) and applicable consistency checks remain
as originally measured. The two known full-suite failures and semantic-quality
limitations remain visible; neither exception waives an unobserved regression.
Any resolver, routing or tool-argument mismatch would remain a real blocker.

The [release disposition](../artifacts/diagnostics/pr7-timeouts/c149f73/RELEASE_DISPOSITION.md)
defines the case-level scope and residual risk. The [isolated follow-up](../artifacts/diagnostics/pr7-timeouts/c149f73/kb002-isolated-01/REPORT.md)
and preceding [AC comparison](../artifacts/diagnostics/pr7-timeouts/c149f73/AC_COMPARISON.md)
retain their complete measurements and limitations. Their contemporaneous
"blocked / no exception granted" statements are historical and are superseded
only by this owner-approved disposition.

No runtime change or full-gate rerun accompanied release closure. PR8 planning
may start from the merged PR25 baseline; neither exception carries forward to
PR8 or authorizes changes to grounding, timeouts, retry budgets or provider
settings. PR8 implementation is not part of this closure.

## PR7 resolver extraction — 2026-09-05 UTC

Historical measurement and pre-exception stop disposition; see the subsequent
owner-approved exceptions above. The failed measurement below is unchanged.

PR7 is implemented in [PR25](https://github.com/syang620/Finsearch-reboot/pull/25)
at frozen runtime `c149f73d073a306cf34d938955ae6cc739191528`. **Release is blocked;
the PR remains draft and no PR7 exception has been granted.** The unchanged
15-case live gate ran once and failed: **8/15 non-critical rows**, seven critical
rows, deterministic score `0.859390` against `0.90`, and zero evaluator errors.
Six failures are analyst model timeouts, so the approved plan's grounding-only
exception scenario is insufficient. No further live run or behavior tuning was
performed. Further investigation, rerun or release disposition needs new owner
direction; `PR24-GROUNDING-001` does not carry forward.

The characterization oracle was frozen first at `fae5bbf`, from actual merged-PR24
runtime `2860c430036f3fa9e9488663fed9d086a03820af`. All **170/170** cases replay
exactly before and after extraction, including status, metric ID, ticker, year,
selected target, exact reason, capability decisions, complete legacy execution
results and ordered SEC arguments. The capture script verifies old source bytes
and refuses replacement. The current harness reads static expected observations;
it never generates expected decisions from the new resolver.

The pure resolver owns existing metric/target matching and ticker/year precedence.
Selected-target metadata is returned unchanged; filing-form interpretation and
execution permission remain in orchestration. First-match and single-target
fallback behavior, dictionary compatibility, capability aliases, evidence admission,
lane semantics and PR6 validation are preserved.

Offline evidence at the clean tracked implementation passes:

- Resolver/execution parity: **170/170**.
- Calculator: **20/20**; analyst output repair: **20/20**.
- Grounding: **37/37**; degradation: **14/14**, with applicable consistency rates
  **100%**. These are behavior/contract checks, not semantic-quality guarantees.
- Full pytest suite: **719 passed**, 47 subtests passed, two known PR24 failures,
  one expected table-renderer warning. All 174 new resolver tests pass. The full
  suite ran before freeze against identical runtime/test source bytes. The existing
  failures remain planner `alias_recognition/alias_002` and retrieval no-tool-call
  attempt count; no blanket full-suite success is claimed.

[Fresh review](https://github.com/syang620/Finsearch-reboot/pull/25#issuecomment-5546911177)
completed on `c149f73` before the live gate, with no major issues or inline findings.
Python is 3.11.14 and pytest 9.0.2 remains environment-only; no dependency or
lockfile changes were made.

The live failures are preserved in full:

- `ANALYST_MODEL_TIMEOUT after 120.0s`: `AGENT_V1_KB_002`, `AGENT_V1_KB_003`,
  `AGENT_V1_HYBRID_002`, `AGENT_V1_HYBRID_003`, `AGENT_V1_ANALYST_001`, and
  `AGENT_V1_ANALYST_003`. The model-call limit is unchanged and the analyst source
  is byte-identical to merged PR24. The underlying timeout cause is not established.
  In particular, the live calculation case timed out; no successful calculation
  answer is claimed from this run.
- `AGENT_V1_HYBRID_001`: `ANALYST_GROUNDING_INVALID`, with
  `GROUNDING_EVIDENCE_TYPE_MISMATCH` and `GROUNDING_ROW_TEXT_MISMATCH`. Validation
  continues to fail closed. This is the same failure class present in PR24, but
  it does not account for or waive the six timeout failures.

Routing, metadata, lane expectations, structured metric/status matching and all
applicable status/degradation/grounding consistency checks remain **100%**. All
seven successful structured executions produce analyst-visible typed evidence with
complete measured provenance. A separate replay of the captured live planner
inputs through the actual merged-PR24 executor, injecting captured SEC responses
without external calls, reproduces **7/7 complete results and exact ordered SEC
arguments**. No resolver or tool-argument mismatch was found.

Model digests, corpus hashes, dataset and execution driver match PR24. The
582-point index fingerprint is unchanged before/after. Execution timestamps contain
a substantial wall-clock gap; its cause was not established, and timing is not
used for a performance comparison. The single-run change from PR24's 12/15 to
8/15 is not evidence of a causal resolver regression or a semantic-quality trend.

New immutable evidence (all earlier PR5/PR6/PR24 evidence remains unchanged):

- `artifacts/evals/agents/v1/resolver/baselines/c149f73/`: snapshot parity,
  manifest and `verification.json` containing source diff/file list, exact commands,
  test outcomes, hashes, live old-path replay and release disposition.
- `artifacts/evals/agents/v1/calculator/baselines/c149f73/`
- `artifacts/evals/agents/v1/output_repair/baselines/c149f73/`
- `artifacts/evals/agents/v1/grounding/baselines/c149f73/`
- `artifacts/evals/agents/v1/degradation/baselines/c149f73/`
- `artifacts/evals/agents/v1/baselines/c149f73/`: all 15 raw rows, summary,
  errors file and full live provenance/failed-gate manifest.

## PR24 release exception PR24-GROUNDING-001 — 2026-09-04

After reviewing the failed 12/15 live-gate result, the project owner explicitly
authorized merging PR24 and proceeding to PR7 planning. This is a new exception
for PR24, not an extension of `PR6-CALC-001`. It applies to implementation
`7062f48c9d929f2925ffa6820cfd08e18c41ecac` and its evidence recorded at
`aebd4bede06df105e469bab69bea52fc3a5b3fc0`.

The strict live gate **failed**: three critical rows, score `0.946394`, zero
evaluator errors. The owner accepts the following known unavailable answers for
this merge only:

- `AGENT_V1_HYBRID_001`: incompatible attribution evidence type and row-text
  mismatches.
- `AGENT_V1_HYBRID_003`: persistent row-text mismatches.
- `AGENT_V1_ANALYST_001`: persistent calculation-row text mismatch; this is not
  a successful calculator answer.

No claim is made that all three failures predate PR24. The runtime continues to
fail closed; no unsupported answer, threshold change, validation relaxation,
selective rerun or rewritten evidence is approved. The 20/20 calculator and
output-repair matrices, 37/37 grounding, 14/14 degradation and 100% applicable
runtime/shadow consistency remain as measured. The two documented pre-existing
full-suite failures and semantic-quality limitations remain visible; this is not
a blanket waiver for other regressions or a claim of full-suite/semantic success.

The immutable failed live measurement remains under
`artifacts/evals/agents/v1/baselines/7062f48/`; its contemporaneous draft/stop
disposition is historical and is not rewritten. This owner-approved exception
supersedes that release disposition only. PR7 is authorized for planning, not
implementation. PR7 must preserve behavior and record its own evidence; this
exception does not automatically waive any PR7 gate failure.

## PR24 output-contract repair — 2026-09-04 UTC

Historical measurement and pre-exception disposition; see the release exception
above for the subsequent owner-approved merge decision. All raw evidence remains
unchanged.

Current frozen implementation:
`7062f48c9d929f2925ffa6820cfd08e18c41ecac`. The owner approved a narrow
extension to schema and claim/row retry feedback: **repair model-output contract
violations, not evidence truth**. PR6's validator and factual prose assembly,
retrieval, lanes, resolver, provider settings, retry budgets and thresholds are
unchanged. Only analyst diagnostic feedback changed after fixture freeze.

The four failed live packets and final candidates from `6f4387a`, 20 new
scenarios, and an independent expectation-based harness were frozen at
`33b57b6eeed8abc99bdcfa4817cd447c02ab3f86` before changing runtime behavior.
The matched before/after output-repair contracts improve **14/20 → 20/20**;
precise-feedback coverage improves **0/6 → 6/6**, including the four captured
live failure shapes **0/4 → 4/4**. Dataset, source, harness, independent oracle
and PR6 validator hashes match exactly before/after.

Terminal expectations were already **20/20 before** and remain 20/20: the stub
injects corrected candidates unconditionally, not in response to interpreting
the feedback. These results establish more precise diagnostics and preserved
fail-closed behavior, not newly achieved terminal repair or live-model efficacy.
All 20 inputs/histories are preserved, all seven successes pass independent
grounding checks, and there are zero unexpected successes. No fields or factual
claims are silently filled or rewritten by the new repair code.

The original calculator matrix remains **20/20**, preserving its original
4/20 → 20/20 comparison. Unchanged grounding passes **37/37** and degradation
passes **14/14**. Focused tests pass **362**. Full tests pass **545**, with 47
passing subtests and the same two pre-existing planner-alias/retrieval-retry
failures; there is one expected table-renderer warning. Python is 3.11.14 and
pytest 9.0.2 remains environment-only; dependency files are unchanged.

The unchanged 15-case live gate ran **once** on clean tracked implementation
`7062f48` and **failed: 12/15 non-critical rows, three critical rows**, score
`0.946394`, zero evaluator errors. The zero-critical threshold is unchanged.
Remaining failures, all correctly reported as analyst grounding errors:

- `AGENT_V1_HYBRID_001`: `GROUNDING_EVIDENCE_TYPE_MISMATCH` and
  `GROUNDING_ROW_TEXT_MISMATCH`. An attribution claim cites only structured
  evidence; row labels also disagree with their bound claim text.
- `AGENT_V1_HYBRID_003`: `GROUNDING_ROW_TEXT_MISMATCH`. Capitalization and numeric
  formatting still differ between the rows and their claims after bounded retry.
- `AGENT_V1_ANALYST_001`: `GROUNDING_ROW_TEXT_MISMATCH`. Required fields and the
  explicit calculation are present, but the row label is not a verbatim substring
  of the bound claim. This is **not a successful calculator answer**.

The previous `AGENT_V1_SF_004` and `AGENT_V1_HYBRID_002` failures are non-critical
in this run. There are no terminal `ANALYST_OUTPUT_INVALID` errors. The change
from 11/15 to 12/15 non-critical rows is a single stochastic observation, not a
controlled causal or semantic-quality improvement claim. All applicable lane,
effective-status, failure-stage, degradation and grounding consistency rates
remain 100%; seven structured executions yield seven visible typed contexts.

The run took 2,799,423 ms (about 47 minutes). It exited zero and wrote all rows,
despite the known MCP/AnyIO shutdown warnings. The 582-point Qdrant fingerprint
remains `6c72627b05dbec1e7f65d2132b8572b9fa12c47af711f2a12b8b776967994cee`
before and after, matching `0f8e477` and `6f4387a`. Model digests, source/sidecar
hashes and redacted execution settings are recorded in the new live manifest.

Fresh review of `7062f48` raised one P2: the release records still referred to
older implementation evidence. This entry and the new SHA-keyed artifacts
address that documentation/provenance gap; no runtime changes were made after
freeze. Review references and exact commands are in the verification record.

**Disposition: PR24 remains draft, not release ready.** As instructed, stop
behavior changes and further evaluations after this failed live gate. Do not
broaden PR24 again, selectively rerun failures, weaken validation, carry forward
the PR6 exception, or start PR7. All earlier evidence remains unchanged.

New immutable evidence:

- `artifacts/evals/agents/v1/output_repair/baselines/33b57b6/` (before).
- `artifacts/evals/agents/v1/output_repair/baselines/7062f48/` (after and
  `verification.json`, including commands, matched hashes and review).
- `artifacts/evals/agents/v1/calculator/baselines/7062f48/`.
- `artifacts/evals/agents/v1/grounding/baselines/7062f48/`.
- `artifacts/evals/agents/v1/degradation/baselines/7062f48/`.
- `artifacts/evals/agents/v1/baselines/7062f48/` (failed live measurement and
  `manifest.json` with provenance and case-level diagnosis).

## Calculator follow-up verification — 2026-09-03

Historical pre-extension snapshot; superseded for PR24 release assessment by the
`7062f48` entry above. Its raw measurements are preserved unchanged.

PR 23 is merged at `6155034e4c1864e187a3318619524d9192024a78` under the
PR6-only exception below. Separate calculator PR 24 remains **draft, not release
ready**, with implementation frozen at
`6f4387aa1a30f5a3fac7f88cadd44f4b4c3b1b54`. PR7 has not started.

The matched 20-case replay improves full behavior contracts from **4/20 to
20/20** and terminal status/error expectations from **10/20 to 20/20**. The
dataset, harness and independent grounding-oracle hashes are identical before
and after. All 20 raw histories are preserved, zero unexpected successes occur,
and all 11 successful after outputs have valid grounding. The minimized replay
isolates computation equivalence and selection; it is not an exact live replay.
The unchanged grounding matrix passes 37/37 and degradation passes 14/14.
Focused tests pass 342; full tests pass 525 with 47 passing subtests and the same
two pre-existing planner-alias/retrieval-retry failures. pytest 9.0.2 remains
environment-only. Fresh review of the frozen implementation found no major
issues; that review does not override the subsequent failed live gate.

The unchanged 15-case live gate ran once on the clean tracked implementation and
**failed: 11/15 non-critical rows, four critical rows**, score `0.926786`, zero
evaluator errors. The zero-critical threshold was not relaxed. Failures are:

- `AGENT_V1_SF_004`: `ANALYST_OUTPUT_INVALID`; final answer omits required
  `compare_rows`.
- `AGENT_V1_HYBRID_002` and `AGENT_V1_HYBRID_003`:
  `ANALYST_GROUNDING_INVALID` / `GROUNDING_ROW_TEXT_MISMATCH`.
- `AGENT_V1_ANALYST_001`: final answer retains an explicit calculation selection
  but omits required `status`, producing `ANALYST_OUTPUT_INVALID`. The earlier
  ambiguity error is absent, but this is **not a successful calculator answer**.

Lane/effective-status/failure-stage/degradation/grounding consistency remains
100%: the runtime truthfully reports these failures. The 582-point index has the
same before/after fingerprint as PR6 `0f8e477`; model, corpus, provider settings,
datasets and thresholds are recorded unchanged. The run took 3,138,636 ms (about
52 minutes) and exited zero despite known MCP/AnyIO shutdown warnings. Exit zero
does not mean gate pass. This stochastic comparison alone cannot attribute the
additional failures to the prompt change or establish a semantic-quality effect.

No PR6 validation was weakened, no failed cases were selectively rerun, and no
release exception applies to PR 24. Reaching 15/15 now requires a separately
approved scope for output-schema and claim/row repair, preserving strict
validation. Freeze those observed failure fixtures before any such behavior
change; do not start PR7 or rerun the expensive gate speculatively.

Immutable after evidence and exact execution provenance:

- `artifacts/evals/agents/v1/calculator/baselines/6f4387a/` (including
  `verification.json` with test commands and review link).
- `artifacts/evals/agents/v1/grounding/baselines/6f4387a/`.
- `artifacts/evals/agents/v1/degradation/baselines/6f4387a/`.
- `artifacts/evals/agents/v1/baselines/6f4387a/` (raw failed live measurement,
  hashes, redacted command, provenance and diagnosis in `manifest.json`).

All earlier PR5/PR6 and calculator-before measurements remain unchanged.

## Calculator reliability before baseline — 2026-09-03

PR6 merged at `6155034e4c1864e187a3318619524d9192024a78` under exception
`PR6-CALC-001`. The separate calculator follow-up froze 20 focused fixtures and
their independent expectation-based replay harness at
`4e81d66acf6c478c015b6a5ea7ca1cc2bf149253`, before runtime behavior changed.
The runtime source at that measurement is byte-identical to merged PR6.

The before run satisfies 4/20 full behavior contracts. This strict count includes
bounded retry/candidate-count and explicit selection-feedback expectations, not
just terminal status: some already fail-closed cases fail only the new repair
expectation. Both successful outputs retain valid grounding, all 20 histories are
preserved, and there are zero unexpected successes. Repeated identical successful
calculations are falsely ambiguous; missing final selection with distinct results
fails without a dedicated selection repair.

Evidence is in `artifacts/evals/agents/v1/calculator/baselines/4e81d66/`.
Its manifest records source/dataset/harness hashes, environment-only pytest 9.0.2,
Python version and the exact command. Derived fixtures are deliberately minimal,
not exact live transcript replays; see
`data/evals/agents/v1/calculator_regression_v1.md`. Grounding, lane policy and
the historical PR6/PR5 evidence remain unchanged. At this before measurement,
the expensive live gate had not been rerun; its later frozen-implementation
result is recorded above.

## PR6 release exception PR6-CALC-001 — 2026-09-03

The project owner explicitly approved merging PR6 with the single known
pre-existing live failure `AGENT_V1_ANALYST_001` /
`CALCULATION_RESULT_AMBIGUOUS`. This exception applies only to PR 23 and its
reviewed implementation `0f8e477c47b7f11a8e069720b3b089e81f25cf9d`.

The unchanged live gate **failed**, with 14/15 non-critical rows, one critical
calculator-provenance failure, zero evaluator errors and score `0.980392`.
The exception accepts that result for PR6 closure; it does not change the gate,
its thresholds, the raw artifacts, or the runtime's fail-closed behavior. The
same error is preserved in the PR5 `de0142c` baseline. PR6-specific grounding
passed 37/37, degradation passed 14/14, applicable runtime/evaluator consistency
was 100%, and fresh review found no major issues.

The remaining risk is an unavailable calculation answer when successful tool
results cannot be uniquely matched. No ambiguous calculation is accepted as a
successful answer. The separately documented semantic-quality limitations and
pre-existing full-suite failures remain visible; this exception is not a claim
of semantic correctness or a blanket waiver for other regressions.

Required follow-up before PR7: open a separate, narrow calculator-reliability PR,
freeze a focused regression fixture against unchanged PR6 before changing
behavior, measure before/after, then rerun the unchanged 15-case gate once the
calculator fix is frozen. Do not change PR6 grounding policy to obtain a pass.
This exception does not automatically apply to that follow-up or future PRs.

Supporting immutable evidence:
`artifacts/evals/agents/v1/baselines/0f8e477/manifest.json` and the grounding,
degradation and review records linked below. Merge authorization was provided
explicitly by the project owner after reviewing these results.

## PR6 live gate — 2026-09-03

The unchanged 15-case live gate ran exactly once on clean, detached implementation
`0f8e477c47b7f11a8e069720b3b089e81f25cf9d`. All 15 rows were valid and there
were zero evaluator errors. The deterministic score was `0.980392`, but one
critical row (`1/15`, `0.066667`) means the strict zero-critical-row gate **failed**.
No expectations, thresholds, provider settings or retry budgets were changed,
and no cases were selectively rerun. PR 23 remains draft; release closure needs
the calculator failure resolved or an explicit release exception.

The critical case is `AGENT_V1_ANALYST_001`: the final calculation could not be
uniquely matched to a successful calculator result, producing
`CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and analyst-stage failure.
This same failure is recorded in the prior PR5 baseline `de0142c`. The lower
critical count in this stochastic run is not a controlled performance-uplift claim.

Runtime/shadow lane status, effective status, failure stage, degradation and
grounding consistency were all 100%. The independent grounding assessment covered
12 successful outputs, including one no-claim insufficient-data response. Eleven
claim-bearing outputs contained 18 claims and 18 context references, with no
grounding inconsistencies. Seven successful structured executions yielded seven
typed, visible contexts with complete provenance-field coverage and no synthetic
structured-text contexts. The frozen PR6 matrix remains the authoritative
PR-specific gate: 37/37 passed; the unchanged PR5 matrix passed 14/14.

Structural validity does not establish answer completeness or semantic quality.
The preserved live traces show that `AGENT_V1_HYBRID_001` omits the requested
explanation, `AGENT_V1_HYBRID_002` points to Item 7 rather than explaining cash
flow, and `AGENT_V1_KB_002` mixes English and Thai. These are qualitative
observations, not newly scored metrics or evidence of semantic uplift. The frozen
secondary judge results remain separately reported below.

Execution used Python 3.11.14, environment-only pytest 9.0.2, the recorded
qwen2.5:14b-instruct and qwen3-embedding:8b digests, hosted Qwen3 reranking, a fresh
checkpointer and Qdrant 1.16.2. Real SEC contact information was supplied only in
the local environment. The 582-point index fingerprint was
`6c72627b05dbec1e7f65d2132b8572b9fa12c47af711f2a12b8b776967994cee`
before and after, using sorted Qdrant Record JSON including payloads and vectors.
Source and sidecar hashes match PR5; equality to historical fingerprints produced
by other methods is not claimed. The hosted reranker has no immutable service-side
digest. RAGAS was disabled.

The run took 2,583,004 ms (about 43 minutes) and exited zero. Known MCP/AnyIO
shutdown warnings did not prevent artifact generation. A fresh focused suite on
the same implementation passed 318 tests with one expected warning. The earlier
full-suite result and fresh no-major-issues review remain unchanged.

New immutable outputs under `artifacts/evals/agents/v1/baselines/0f8e477/`:

- `summary.json`, SHA-256 `d83eaa7aef10e729adee61418cd58564fc3fcb627f28df633446b3ea514e7131`.
- `per_query.jsonl`, SHA-256 `4840d2f646d9577d658bdb47ba7527f560bc284052c1357cffa3f757cb842836`.
- Empty `errors.jsonl`, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `manifest.json`: redacted effective command, environment/model/corpus provenance,
  source hashes, critical-case details, focused-test command and review reference.

Earlier measurements, including the pre-live verification snapshot, all PR5
baselines and the original failing `25e359d` evidence, remain unchanged.

## PR6 implementation evidence — 2026-09-03

Implementation `0f8e477c47b7f11a8e069720b3b089e81f25cf9d` enforces explicit
claim bindings, claim-derived filing prose and comparison rows, bounded repair,
and analyst-stage fail-closed behavior. Calculation claims activate the existing
calculator checks even on extraction tasks; positive calculation claims do not
gain an insufficient-data exemption. Per-variable evidence mapping and semantic
entailment remain secondary evaluation concerns, as explicitly scoped in ADR 004.

The frozen 37-case grounding matrix passes 37/37, compared with 2/37 on the
canonical unchanged-PR5 runtime. All applicable deterministic rates are 100%,
with denominators recorded: 18/18 claim coverage, 40/40 valid references, 20/20
source preservation and completeness, 3/3 structured-metric compatibility,
11/11 narrative-type compatibility, 1/1 mixed hybrid binding, 6/6 invalid-ID
detection, and 18/18 fail-closed handling. Dataset, harness and independent oracle
hashes match exactly between the canonical before and after measurements. These
are structural enforcement results, not claimed semantic improvements.

The unchanged 14-case PR5 degradation matrix also passes all six rates at 100%.
Full tests: 501 passed, 47 subtests passed, and the same two pre-existing failures
in planner alias routing and retrieval retry-count expectations. pytest 9.0.2
remains environment-only; repository dependency files were not changed.

The frozen secondary judge assessed 18 observed claims: 3 fully supported,
14 partially supported and 1 unsupported, with zero judge errors and 100%
assessment coverage. Full-support precision and grounded-claim rate are 0.166667;
unsupported rate is 0.055556. The unsupported attribution fixture is intentionally
retained. PR5 emitted no bound claims, so no before/after semantic uplift is
claimed. The judge was qwen2.5:14b-instruct at temperature 0, digest
`7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.

Artifacts and command/provenance records:

- `artifacts/evals/agents/v1/grounding/baselines/0f8e477/`
- `artifacts/evals/agents/v1/grounding/baselines/0f8e477/judge/`
- `artifacts/evals/agents/v1/degradation/baselines/0f8e477/`

All prior measurements, including initial implementation `9851ffd`, remain
unchanged. The fresh GitHub review on `0f8e477` reported no major issues after the
no-calculator claim bypass was fixed and the agreed semantic boundary documented.

**Pre-live status at the time of this entry:** the unchanged 15-case live gate had not run because
SEC_USER_AGENT is unset and real contact information is required. No placeholder
contact, modified expectation, weakened threshold or synthetic replacement was
used. PR 23 remains draft pending that measurement. A temporary corpus was
prepared from the same hashed PR5 sidecars, but no live-run corpus equivalence or
gate result is claimed before execution.

## PR6 pre-implementation grounding baseline — 2026-09-03

The canonical before measurement is
`artifacts/evals/agents/v1/grounding/baselines/4ba5553-canonical`, using unchanged
PR5 runtime `4ba5553737a2e788da1f4e5a01519f65685b15bd` in an isolated worktree
and frozen harness/oracle commit `9c2e8fa`. The 37 candidate-injection scenarios
run the real analyst graph without a live model. This is a structural enforcement
measurement, not a semantic-quality benchmark.

The original prototype measurement under `grounding/baselines/4ba5553` remains
unchanged but is superseded: its calculator stub used an MCP envelope where the
injected-tool interface expects a direct payload, and its KB fixture text was in
an unrecognized field. Both harness issues were corrected and measured against
unchanged PR5 code before the matched comparison. The frozen dataset itself is
unchanged. Intermediate harness-development measurements were diagnostic only.

Canonical baseline: 2/37 scenarios satisfy the new contract; 36 outputs report
success, but only 1/36 passes full finalized-output integrity. Grounding fail-closed
behavior is 1/18; known visible citation IDs and source preservation are each
31/32. Legacy outputs contain no explicit claims, so claim-type/coverage metrics
have zero denominators and are unavailable, not inferred or labeled perfect.
No semantic improvement can be inferred from these missing bindings.

The environment remains Python 3.11.14 with pytest 9.0.2 installed environment-only
in finsearch-arm. Historical PR5 evidence and repository dependency files are
unchanged. Final implementation, after evidence and fresh review are recorded
separately once available.

This document records measured evaluation results. It is intentionally separate from
`docs/ARCHITECTURE.md`, which describes current system behavior without mutable model
scores.

Add new baseline entries rather than rewriting older results. Every entry must name
the evaluated commit, model identity, dataset version and filter, exact command, raw
artifact, measured result, and known coverage limitations.

## Planner Routing P0 — 2026-08-15

### Run identity

| Field | Value |
|---|---|
| Status | Canonical planner-routing baseline |
| Evaluated commit | `a8583845bf415beb6819a5c76ee8a8d682168a73` |
| Branch | `agent/document-current-architecture` |
| Completed | `2026-08-15T19:35:51-04:00` (`America/New_York`) |
| Planner model | `ollama/qwen2.5:14b-instruct` |
| Ollama model digest | `7cdf5a0187d5` |
| Dataset | `data/evals/agents/planner/planner_routing_core.v1.json` |
| Dataset version | `1.0` |
| Dataset filter | `P0` |
| Evaluated cases | 25 |
| Runner exit code | `0` |

### Command

```bash
PYTHONPATH=src python scripts/evals/agents/eval_planner_routes.py \
  --cases-path data/evals/agents/planner/planner_routing_core.v1.json \
  --priority P0 \
  --model ollama/qwen2.5:14b-instruct \
  --out-path artifacts/evals/agents/planner/planner_routing_core.v1/runs/a858384_p0_ollama_qwen2.5_14b-instruct.json
```

### Artifact

```text
artifacts/evals/agents/planner/planner_routing_core.v1/runs/a858384_p0_ollama_qwen2.5_14b-instruct.json
```

SHA-256:

```text
c7770d34afa9f214eed8911ace8d4e3a2e711d57abf6f867d95dc4d6f0250c16
```

### Result

Overall: **25/25 passed (100.0%)**.

| Category | Passed | Failed | Total |
|---|---:|---:|---:|
| `structured_fact` | 5 | 0 | 5 |
| `kb_narrative` | 5 | 0 | 5 |
| `hybrid` | 5 | 0 | 5 |
| `unsupported_metrics` | 5 | 0 | 5 |
| `multi_company_comparison` | 5 | 0 | 5 |
| **Overall** | **25** | **0** | **25** |

The artifact case IDs exactly match the 25 cases marked `P0` in dataset version
`1.0`. The summary counts were independently recomputed from the serialized case
results.

### Coverage limitations

This baseline measures planner routing behavior and structured-fact request counts.
It does not validate:

- KB retrieval quality or recall.
- SEC structured-fact resolution or numeric correctness.
- Analyst answer quality, calculations, or citations.
- Hybrid answer synthesis or partial-degradation behavior.
- End-to-end latency or failure-stage accuracy.

The older tracked planner artifacts remain historical evidence. They are not the
canonical baseline for commit `a858384` because they were generated before that
commit and do not embed commit provenance.

## Route-Aware Agent E2E v1 — 2026-08-26

### Run identity

| Field | Value |
|---|---|
| Status | Canonical route-aware v1 release baseline |
| PR2 implementation merge | `ca3566688df78de2a7054ce88798bacb4e6e49fe` |
| Evaluated commit | `6aae2651518e4495c947e393ed78978b515bd482` |
| Completed | `2026-08-26T21:38:20-04:00` (`America/New_York`) |
| Evaluator | Route-aware agent E2E v1 |
| Dataset | `data/evals/agents/v1/agent_eval_routing_v1.jsonl` |
| Dataset SHA-256 | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` |
| Evaluated cases | 15 |
| Python | `3.12.12` |
| Planner / analyst / retrieval fallback | `ollama/qwen2.5:14b-instruct` |
| Planner / analyst model digest | `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Embedding model | `qwen3-embedding:8b` |
| Embedding model digest | `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Reranker | `Qwen/Qwen3-Reranker-8B` via DashScope |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 points |
| Chunking dependency | `sec-parser==0.58.1` in the isolated evaluation environment |
| RAGAS | Disabled |

### Exact executed gate command

The following is the recovered successful command. Only the credential assignment,
SEC contact, virtualenv path, and temporary output paths have been replaced with
explicit placeholders; its CLI arguments are unchanged.

```bash
set -o pipefail
SEC_USER_AGENT='FinSearch-Reboot/1.0 <SEC_CONTACT>' \
<RERANK_CREDENTIAL_ENV>='<RERANK_CREDENTIAL>' \
QWEN3_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
SEC_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
QWEN3_EMBED_API_URL='http://127.0.0.1:11434/api/embed' \
QWEN3_EMBED_MODEL='qwen3-embedding:8b' \
QDRANT_URL='http://127.0.0.1:6333' \
QDRANT_COLLECTION_NAME='sec_docs_dense_bm25_pr2_63dcec0' \
FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH='<EVAL_OUTPUT_DIR>/orchestrator_checkpointer.sqlite' \
PYTHONPATH=.:src \
<EVAL_VENV>/bin/python scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir '<EVAL_OUTPUT_DIR>/artifacts' \
  --planner-model 'ollama/qwen2.5:14b-instruct' \
  --analyst-model 'ollama/qwen2.5:14b-instruct' \
  --deterministic-threshold 0.90 \
  --max-critical-failure-rate 0.0 \
  2>&1 | tee '<EVAL_OUTPUT_DIR>/gate.log'
exit ${pipestatus[1]}
```

No RAGAS flag was supplied, and the serialized configuration confirms
`enable_ragas=false`.

### Artifacts

| Artifact | SHA-256 |
|---|---|
| `artifacts/evals/agents/v1/baselines/6aae265/summary.json` | `c56ec817665231280a037d9fb4ddf1c691e1da0ec81c38a929fe73b60f8433a5` |
| `artifacts/evals/agents/v1/baselines/6aae265/per_query.jsonl` | `a7032d13214f501ec60882454501e165adbbeff1899147748d9d02b175657aa0` |
| `artifacts/evals/agents/v1/baselines/6aae265/errors.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

### Result

| Metric | Result |
|---|---:|
| Valid queries | 15 / 15 |
| Evaluator errors | 0 |
| Deterministic score | 1.0 |
| Critical failure rate | 0.0 |
| Citation match rate | 1.0 |
| `gate.overall_pass` | `true` |
| Runtime | 1,787,765 ms (about 29m48s) |

`AGENT_V1_ANALYST_001` correctly computed the FY2024 Services net-sales increase
as `12.874413145539906%` from 96,169 and 85,200, used the financial evaluator,
and cited the matching AAPL FY2024 filing table.

### Corpus provenance

The gate used the pinned AAPL FY2024 10-K corpus in Qdrant collection
`sec_docs_dense_bm25_pr2_63dcec0`, which contained 582 points. The tracked source
HTML was `data/html_filings/AAPL/10-K/10-K_2024.html` with SHA-256
`24a830a0f1256e371d36a1f7f72e5e85a38037d1de2f6f966eb8457db42ff6d6`.

Fresh chunk sidecars generated from that HTML were:

| Sidecar | Rows | SHA-256 |
|---|---:|---|
| `data/chunked/AAPL/10-K/10-K_2024.text.jsonl` | 101 | `a2035e3048c9d69945a6851e9f92b6eb64d97ea15ef1ab8fe1812b25e8cd08e0` |
| `data/chunked/AAPL/10-K/10-K_2024.tables.jsonl` | 48 | `a60a91dd358656c3667b9700a4e3a2fde5558a67900f62c0a22893fdaf2a6b1d` |
| `data/chunked/AAPL/10-K/10-K_2024.text.split.jsonl` | 107 | `856f6aec30636750141a695d864eb08da182160e25143b393b49bc7769436f71` |

The available 48-row frozen table-summary input is preserved at
`artifacts/evals/agents/v1/baselines/6aae265/corpus_inputs/AAPL_10-K_2024.tables.summaries.jsonl`
with SHA-256
`63a8e65659787ec3e32ca0a8b0e7773193108529a35c32377396ef474368050f`.

The exact sidecar-generation command recovered from the canonical-run setup was:

```bash
SEC_USER_AGENT='FinSearch-Reboot/1.0 <SEC_CONTACT>' \
PYTHONPATH=.:src \
<EVAL_VENV>/bin/python scripts/ingestion_cli.py run-all \
  --tickers AAPL \
  --forms 10-K \
  --years 2024 \
  --skip-download \
  --skip-summarize \
  --skip-embed \
  --skip-ingest
```

This command regenerated the ignored chunk sidecars needed for table hydration;
because both `--skip-embed` and `--skip-ingest` were present, it did not build the
Qdrant corpus. The collection came from an earlier corpus embed/ingest operation.
That original command was not preserved, so no command is reconstructed here.
Raw Qdrant storage, container state, model data, and the temporary isolated
environment (where `sec-parser==0.58.1` was installed) are not committed.

### Closeout provenance

PR2 introduced the route-aware v1 evaluator. Its early live runs exposed focused
integration defects in checkpointer lifecycle, retrieval serialization and model
defaults, Ollama tool arguments, form/period routing, and analyst terminal and
calculation provenance. Those defects landed as focused follow-ups, and the first
canonical passing gate was recorded at PR #16 merge SHA `6aae2651518e4495c947e393ed78978b515bd482`.
The post-artifact MCP/AnyIO cleanup hang is non-gating follow-up issue #17.

### Coverage limitations

This 15-case gate validates route selection across `kb`, `structured_fact`, and
`hybrid`; expected lane execution; selected structured metric outcomes; planner
metadata; clarification control flow; analyst status; calculator use; citation
minimums/matching; and selected end-to-end failure semantics.

It does not establish exhaustive retrieval recall, coverage of all SEC issuers,
forms, and filings, amendment precedence correctness, production SLOs,
load/concurrency behavior, full claim-level citation faithfulness, or complete
structured-fact capability coverage. RAGAS was disabled, so this baseline also
does not measure RAGAS semantic quality.

## Structured-Fact Capability Policy v1 — 2026-08-30

### Run identity

| Field | Value |
|---|---|
| Status | Canonical PR3 capability-policy baseline |
| Pre-PR3 commit | `586fe24cdeeb87040ad241a1cac39e71648d907b` |
| PR3 implementation evaluated commit | `6b6b6173bac9045b03ad2292910b5acfd51740c8` |
| First review-fix evaluated commit | `5a93171562dce106d1dd45cfe0c80aa5c01628ac` |
| Second review-fix evaluated commit | `af15c3c0acc301790d74187dcfacdc11c3dabe98` |
| Third review-fix evaluated commit | `a3af6f08cf1b28c6a53ca52f4c58747ad62c4a99` |
| Fourth review-fix evaluated commit | `4c94cffbb8fda68196ce9a0101cc07c9b0876c9d` |
| Fifth review-fix evaluated commit | `ced41d0718f46bc4e24dd018c830492d3db96ca4` |
| Sixth review-fix evaluated commit | `6adf1e259434f3b5034c8bac4d3122d6873a7796` |
| Seventh review-fix evaluated commit | `79bcb8e0c6d6e3199bc99b326ada3084e27d3986` |
| Eighth review-fix evaluated commit | `f9a07e8b02ba3dabfd881bcada14139077e22614` |
| Ninth review-fix evaluated commit | `5d0dc4647efff81e07d576d430ac45a8b6c910c9` |
| Tenth review-fix evaluated commit | `3ef13f3599bab9552d698c6d2270eec05cc0bb45` |
| Eleventh review-fix evaluated commit | `e7dae4662006a2e0cef0d9adc33c1b1857b7e93a` |
| Twelfth review-fix evaluated commit | `b1f9ebcbbcb004de1b7901c60ac88e7c641c12e0` |
| Thirteenth review-fix evaluated commit | `eb1bc4b3cea26640604cdd6b6f5dac02cce5fe20` |
| Fourteenth review-fix evaluated commit | `95a0efa085a54a884c8d43442b914fe59c3502de` |
| Fifteenth review-fix evaluated commit | `011eb6ae5c6f16f6d5a8662c4a3c4d3f96bfc10c` |
| Sixteenth review-fix evaluated commit | `992e715c85b7cbb8ba0f5bd772c5d6974135eb76` |
| Seventeenth review-fix evaluated commit | `8b9f8ec4a32b22976a2a9ccd0de09d0b2674712d` |
| Eighteenth review-fix evaluated commit | `f301f32742b85faba079031c4d3381ec15a2e51f` |
| Nineteenth review-fix evaluated commit | `f47576bfdf775ddc21ef90267bea03318ef4e397` |
| Twentieth review-fix evaluated commit | `089763d702020fda627303a09004b937dbfdbbbe` |
| Twenty-first review-fix evaluated commit | `21dd490e287f208bff92c8d4613db0009a93d8c0` |
| Twenty-second review-fix evaluated commit | `dfd3ffe997fbdb55173ff5878ddd7cd8f0160f2a` |
| Twenty-third review-fix evaluated commit | `7791e64c586a4bacfb6e3a46e8f7b4c8aac31ae1` |
| Twenty-fourth review-fix evaluated commit | `efb5e4251dd97af5165b24850aee06a4b18c542b` |
| Twenty-fifth review-fix evaluated commit | `5df0d19771015908abdca2dc949189418bc26016` |
| Final review-fix evaluated commit | `8ee0d1e0c7e3ec840ff7ccf149a535433d6ca9d4` |
| PR3 evidence commit | This documentation/artifact-only follow-up commit |
| Dataset | `data/evals/agents/planner/structured_fact_capability_adversarial.v1.jsonl` |
| Dataset SHA-256 | `cb683920ac5f4f99ce6ec603c11f80be8c2e2b36598536e480781e28d8133273` |
| Evaluator SHA-256 | `ee7daa52e50c8c7c7793ca1a7e30d6e63faf55933109bac28fb0f3563f7628f7` |
| Cases / proposed requests | 40 / 44 |

The dataset and evaluator were frozen before implementation. The pre-PR3 run used
the same bytes from outside the baseline checkout, so neither the policy nor its
tests could generate the expected cases.

### Commands

Pre-PR3 checkout:

```bash
PYTHONPATH=src <EVAL_VENV>/bin/python <FROZEN_EVALUATOR> \
  --dataset <FROZEN_DATASET> \
  --output <BEFORE_OUTPUT>
```

Evaluated PR3 implementation:

```bash
PYTHONPATH=src <EVAL_VENV>/bin/python \
  scripts/evals/agents/eval_structured_fact_capabilities.py \
  --dataset data/evals/agents/planner/structured_fact_capability_adversarial.v1.jsonl \
  --output <AFTER_OUTPUT>
```

Live planner P0 regression gate:

```bash
PYTHONPATH=src <EVAL_VENV>/bin/python scripts/evals/agents/eval_planner_routes.py \
  --cases-path data/evals/agents/planner/planner_routing_core.v1.1.json \
  --priority P0 \
  --model ollama/qwen2.5:14b-instruct \
  --out-path <PLANNER_P0_OUTPUT>
```

### Artifacts

| Artifact | SHA-256 |
|---|---|
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/baselines/586fe24_before.json` | `d0bd5eae53864c2ed72938011dc64b838274517d0bcd6ce39edac35a21232db5` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/6b6b617_after.json` | `e7ed34f0be56ba41a3b636f3093a930e70aed7fc866295105638d14be12dc41a` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/5a93171_post_review.json` | `12486d8ec4497fe2c6367b477bc83d5c27ea1cb2ab71e61e24c12f56f82262d8` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/af15c3c_post_review_2.json` | `0dcecc9b25ddc7ec636945f681801c1b7baa2088913b97e22e6b5bb060dd8b89` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/a3af6f0_post_review_3.json` | `1b37eaf2f822e71e2838a74f824b76b35707d3f1f9bd19ea98af36d5e70ab360` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/4c94cff_post_review_4.json` | `fcab6e630f4152ae8ef3d0b42ed4f732a3e3da337d502552d0b02ca79fd1d418` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/ced41d0_post_review_5.json` | `96f21d8dc6d9e7046974af3d29f24b2a5e2d3b40ce0dd99a63fb5c01a5e992fd` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/6adf1e2_post_review_6.json` | `8a22fafe9a58e70997cdec8970f67b507f643449cac4e901421a9b8b2280eeee` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/79bcb8e_post_review_7.json` | `b62717a4cbbdc9bc58fde9761766c321182f3f7ed18a7a7f951da593afb2f698` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/f9a07e8_post_review_8.json` | `5201be0ee413439019e211cc097405c265675764079696b43bd56bdf56d86a28` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/5d0dc46_post_review_9.json` | `7c9f41f27cba1f6ee4f2f71565cef5e56df6d8699e87b0b9b4850f409d7244b9` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/3ef13f3_post_review_10.json` | `df411f1b2e740b8eaadf36144cdf813d019474d8c334c54534ac18d452f3603e` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/e7dae46_post_review_11.json` | `1a12199a979f5d104417a6810c35965589525a0eb7f13a03ba35bc7db1324589` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/b1f9ebc_post_review_12.json` | `c54bd9ba6ffcc345078a402500e3b91cbad107e2fc96a59387bde1b6e1045975` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/eb1bc4b_post_review_13.json` | `11312d471f14f306abd3ff948be4258576d2b1cf231b8066fc5c5bf0af60d7cd` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/95a0efa_post_review_14.json` | `95aaff2fa134e752c09bc70ea8f74813b2375a29f27fbfcaee3085e90066a0b9` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/011eb6a_post_review_15.json` | `cc2b509e5bfacf9f9b6f71da29cbec44121700020129bc607b1af56f0398fad6` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/992e715_post_review_16.json` | `ab5e83e8f4bef6a67b8b2188ad1dbd583bddcb79d90a3c979fc90b4edeafa623` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/8b9f8ec_post_review_17.json` | `14e140d2c6249daafd5f0ea1548ce3bd94ce0af9fb40b658a02b0a61346a4fb0` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/f301f32_post_review_18.json` | `8650e15bd7d3b679df45fe00368e8ddd7688cf419eea56ef07bd1c5b669a2f60` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/f47576b_post_review_19.json` | `5b73d0f17a7a9e324793514c24a299dde54e4c74cd6b67cc65b4a45e30e3d69d` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/089763d_post_review_20.json` | `3aa9804008224368c9e85758ecb20340bebfe96ef20a5e93bebc1208e950c3e8` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/21dd490_post_review_21.json` | `e8d2b525ed4ee63de1e1feb21a8d5cf8b5d6ac0d2b435ba6438aa5bf270377f6` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/dfd3ffe_post_review_22.json` | `571c88669134a75b31a6636d9cd97dce1de034613c3b9a6cc288ddb935fbfc72` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/7791e64_post_review_23.json` | `dba0a1d06ae0a3b38cb6db03a27dadba1560adc55e9f271adb85b77c4e36579f` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/efb5e42_post_review_24.json` | `b928783350627d8aa04c150ed73509c420e7bed7f09af49012eb78c9c1441ac4` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/5df0d19_post_review_25.json` | `b1d38a2dff6d276e6440c1fb65700a9d6a9a906f8c72d1be367bf1339773c3ca` |
| `artifacts/evals/agents/planner/structured_fact_capability_adversarial.v1/runs/8ee0d1e_post_review_26.json` | `3f5e6ea4475e9da68283a66fb70dce13c589c73f9d8e1668e20809ab93fb4fdb` |
| `artifacts/evals/agents/planner/planner_routing_core.v1.1/runs/6b6b617_p0_ollama_qwen2.5_14b-instruct.json` | `e3cc95fe8a56db9596a4bcfa27ad0d99267352b732d56b176932df49da1c31a7` |

### Before / after result

| Metric | Pre-PR3 `586fe24` | PR3 `6b6b617` |
|---|---:|---:|
| Structured-route precision | 19 / 36 (52.78%) | 19 / 19 (100%) |
| Supported-query recall | 18 / 19 (94.74%) | 19 / 19 (100%) |
| Unsupported-query rejection rate | 7 / 20 (35%) | 20 / 20 (100%) |
| Ambiguous-query clarification accuracy | 0 / 4 (0%) | 4 / 4 (100%) |
| False structured-routing rate | 17 / 24 (70.83%) | 0 / 24 (0%) |
| Effective-route accuracy | 23 / 40 (57.5%) | 40 / 40 (100%) |

PR3 therefore reduced invalid structured routing from 70.83% to 0% on the frozen
adversarial set while raising supported-query recall from 94.74% to 100%.

Structured-route precision counts supported requests among all retained structured
requests. Supported-query recall counts retained supported requests. Rejection rate
counts rejected unsupported and unknown requests. Ambiguity accuracy requires the
whole case to request clarification. False structured-routing counts retained
unsupported, ambiguous, or unknown requests. Effective-route accuracy requires the
normalized route and retained request set to match the independently authored case.

The live planner P0 gate used dataset version `1.1`, SHA-256
`64103d8ad9e130ef9a62b6208abcc0f7ebb7e309f39503e15390d037631e931b`,
and `ollama/qwen2.5:14b-instruct`. It passed 25/25: 5/5 each for
`structured_fact`, `kb_narrative`, `hybrid`, `unsupported_metrics`, and
`multi_company_comparison`.

### Post-review verification

Codex review fixes tightened alias matching, enforced the original-query guard for
partially rejected proposals, recognized independent noun-phrase conjunctions, and
built rejected fallback goals from the sanitized subquestion. Commit
`5a93171562dce106d1dd45cfe0c80aa5c01628ac` was evaluated against the unchanged
dataset and evaluator. All 40 cases and 44 requests passed with the same post-PR3
metrics recorded above: 100% precision, recall, rejection, clarification accuracy,
and effective-route accuracy, with 0% false structured routing.

The fresh review then identified supported metric hints that contradicted unknown
subquestions. Commit `af15c3c0acc301790d74187dcfacdc11c3dabe98` added deterministic
hint/subquestion agreement validation and was rerun against the same unchanged
dataset and evaluator. Its 40-case/44-request result preserved all six metrics.

The next review tightened exact metric-phrase boundaries and target-identifier
sanitization. Commit `a3af6f08cf1b28c6a53ca52f4c58747ad62c4a99` was rerun against
the same unchanged dataset and evaluator and again preserved all six metrics.

The final review required direct hints when structured proposals omit them and
recognized independent clauses separated by `as well as`, `plus`, or semicolons.
Commit `4c94cffbb8fda68196ce9a0101cc07c9b0876c9d` was rerun against the same
unchanged dataset and evaluator and preserved all six metrics.

The final resumed-execution review aligned orchestrator validation with the
planner's clarified metric query and recognized `compared to` comparisons and
spaced fiscal periods. Commit `eb1bc4b3cea26640604cdd6b6f5dac02cce5fe20` was
rerun against the unchanged dataset and evaluator. All 40 cases and 44 requests
again preserved the six post-PR3 metrics recorded above.

The final annual-boundary review preserved every repeated metric occurrence during
clause mapping and rejected quarterly periods from the annual-only structured
executor. Commit `95a0efa085a54a884c8d43442b914fe59c3502de` was rerun against the
unchanged dataset and evaluator and again preserved all six metrics.

The final metadata and omitted-sibling review rejected nonannual fiscal-period
metadata, covered additional named ratios, and retained retrieval for independently
unknown sibling concepts omitted from structured proposals. Commit
`011eb6ae5c6f16f6d5a8662c4a3c4d3f96bfc10c` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final clarified-sibling review rejected explicit nonannual filing targets in
both planner and orchestrator defenses and preserved unaffected clauses across
metric clarification. Commit `21dd490e287f208bff92c8d4613db0009a93d8c0` was
rerun against the unchanged dataset and evaluator and again preserved all six
metrics.

The final occurrence/period review preserved precise sibling metrics during generic
clarification, matched repeated metric requests by fiscal year, and accepted the
annual phrase `fiscal year ended`. Commit
`dfd3ffe997fbdb55173ff5878ddd7cd8f0160f2a` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final scoped-fallback review matched omitted repeated metrics by fiscal year,
recognized comma-separated sibling requests, and limited rejected-request KB jobs
to the applicable target year and entity. Commit
`7791e64c586a4bacfb6e3a46e8f7b4c8aac31ae1` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final original-query agreement review rejected structured proposals that
rewrote a single supported or unknown request to a different metric, and based the
KB fallback on the original request. Commit
`efb5e4251dd97af5165b24850aee06a4b18c542b` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final deterministic-context review rejected unclaimed metric/period proposals
across independent clauses and applied deterministic nonannual filing metadata
before capability routing. Commit `5df0d19771015908abdca2dc949189418bc26016`
was rerun against the unchanged dataset and evaluator and again preserved all six
metrics.

The final period-boundary review rejected YTD and trailing-twelve-month requests,
preserved comma-plus-conjunction clause boundaries, and accepted dated annual
fiscal-period metadata. Commit `8ee0d1e0c7e3ec840ff7ccf149a535433d6ca9d4`
was rerun against the unchanged dataset and evaluator and again preserved all six
metrics.

The final omitted-supported-clause review routed supported clauses missing complete
structured inputs to KB fallback and rejected half-year period notation. Commit
`089763d702020fda627303a09004b937dbfdbbbe` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final uncovered-clause review retained omitted explicit rejections, recognized
leading-quarter notation, preserved compound unknown concepts, and attached
ambiguity to the clause that caused it. Commit
`f47576bfdf775ddc21ef90267bea03318ef4e397` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final narrative-boundary review routed explanation requests to retrieval,
preserved sentence-separated omitted siblings, and removed recognized issuer names
before semantic deny-pattern matching. Commit
`f301f32742b85faba079031c4d3381ec15a2e51f` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final concept-clause review rejected percentage-symbol ratios and preserved
conjunctions inside unknown financial concepts while building KB fallbacks. Commit
`8b9f8ec4a32b22976a2a9ccd0de09d0b2674712d` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

The final semantic-coverage review added percent-of ratios and trend language,
retained omitted unknown clauses in mixed rejected proposals, accepted annual
year-ended wording, and preserved leading-article issuer names. Commit
`992e715c85b7cbb8ba0f5bd772c5d6974135eb76` was rerun against the unchanged
dataset and evaluator and again preserved all six metrics.

### Route-aware live-gate diagnostic

A 15-case route-aware v1 run against the sealed PR3 implementation completed all
cases with perfect intent, route, metadata, lane, structured-metric, and
structured-status match rates. It was not promoted to a canonical passing baseline:
two analyst calls reached the existing 120-second local-model deadline and one
retrieval call reached the existing 30-second MCP deadline, yielding a 0.9172 mean
score and 20% critical-failure rate.

Fresh-checkpointer retries cleared the risk-factors and insufficient-data cases.
The calculation case then returned the correct 12.8744% result and used the
financial evaluator, but was rejected because the model emitted multiple competing
calculation calls. An isolated run of that identical frozen case on untouched
pre-PR3 commit `586fe24cdeeb87040ad241a1cac39e71648d907b` reproduced the same
`CALCULATION_RESULT_AMBIGUOUS` outcome and 0.6154 score. This is therefore recorded
as a pre-existing live-model/tool-call instability, not a PR3 behavior regression.
The post-run MCP/AnyIO asynchronous-generator cleanup warnings remain tracked by
issue #17 and are out of scope for this policy PR. RAGAS was disabled.

### Freeze and coverage limitations

Commit `6b6b6173bac9045b03ad2292910b5acfd51740c8` remains the sealed initial
implementation measured by the original artifacts. The review-fix implementation
at `5a93171562dce106d1dd45cfe0c80aa5c01628ac` and the final review fix at
`af15c3c0acc301790d74187dcfacdc11c3dabe98`, and the final review fix at
`a3af6f08cf1b28c6a53ca52f4c58747ad62c4a99`, followed by final verification at
`4c94cffbb8fda68196ce9a0101cc07c9b0876c9d`, and final resumed-execution
verification at `eb1bc4b3cea26640604cdd6b6f5dac02cce5fe20`, followed by final
annual-boundary verification at `95a0efa085a54a884c8d43442b914fe59c3502de`,
followed by final metadata and omitted-sibling verification at
`011eb6ae5c6f16f6d5a8662c4a3c4d3f96bfc10c` and final semantic-coverage
verification at `992e715c85b7cbb8ba0f5bd772c5d6974135eb76` and final concept-clause
verification at `8b9f8ec4a32b22976a2a9ccd0de09d0b2674712d` and final narrative-boundary
verification at `f301f32742b85faba079031c4d3381ec15a2e51f` and final uncovered-clause
verification at `f47576bfdf775ddc21ef90267bea03318ef4e397` and final
omitted-supported-clause verification at `089763d702020fda627303a09004b937dbfdbbbe`,
followed by final clarified-sibling verification at
`21dd490e287f208bff92c8d4613db0009a93d8c0` and final occurrence/period
verification at `dfd3ffe997fbdb55173ff5878ddd7cd8f0160f2a`, followed by final scoped-fallback
verification at `7791e64c586a4bacfb6e3a46e8f7b4c8aac31ae1` and final original-query agreement
verification at `efb5e4251dd97af5165b24850aee06a4b18c542b`, followed by final deterministic-context
verification at `5df0d19771015908abdca2dc949189418bc26016` and final period-boundary
verification at `8ee0d1e0c7e3ec840ff7ccf149a535433d6ca9d4`, are measured separately by their
post-review artifacts. Changes after the final SHA are limited to evaluation
artifacts and documentation; any later source, test, prompt, policy, evaluator, or
dataset change invalidates the final post-review measurement and requires a rerun.

The adversarial baseline measures routing permission and normalization, not SEC fact
numeric correctness, filing-specific metric resolution, retrieval relevance,
analyst synthesis, citations, latency, load, or production distribution shift. Its
40 cases intentionally emphasize known boundary classes and hostile proposals; it
is not a frequency-weighted sample of user traffic.

## Retrieval Ablation Baseline — 2026-08-31 (canonical rerun)

The original run at evaluated SHA
`5c9ed8ff19255a4f07f93410ea1723f5aa57b501` is superseded and non-canonical.
Its harness could fingerprint a different Qdrant endpoint from the runtime client,
verify a different embedding endpoint from the one used by dense modes, and run
from a dirty tracked checkout. Its artifacts remain only in Git history and must
not be used as baseline evidence.

### Run identity and integrity

- Evaluated implementation SHA: `b61200968f62537af85c8812376b9790a45dfb5f`.
- Evidence commit SHA: `c31bc00b8e591de15690c8f8445fb5b54a106229`.
- Frozen definition: `data/evals/retrieval/retrieval_ablation_v1.json`.
- Table dataset: 12 unique cases at
  `data/evals/retrieval/table/table_eval_v1.jsonl`, SHA-256
  `e20c375fef3bd88553e22e6210849c9127b1cfb3d13ca2dd1ae866754f4b592e`.
- Candidate text dataset: 75 unique cases at
  `data/evals/retrieval/text/aapl_2024_10k_text_retrieval_eval_split_with_uids.jsonl`,
  SHA-256 `a44b51bc2cf733e757116a3ca79c4ef0fca7dc5c3ea2e7e6a7f320be78a1b888`.
  It was frozen but excluded before results because IDs 99 and 100 are absent
  from the PR2 collection, its stable `chunk_uid` labels are not stored in that
  collection, and its chunk numbering is from a different lineage. No label was
  remapped or regenerated; text metrics are therefore not available.
- Qdrant endpoint `http://localhost:6333`, version 1.16.2, collection
  `sec_docs_dense_bm25_pr2_63dcec0`: 582 points.
  The full payload/vector fingerprint was
  `103a1131b373fb651727048b6a875e4bd96b64754ebafd41c42dc6dcccfc5df8`
  in the frozen definition and both before and after the run.
- Corpus provenance reused the PR2 AAPL FY2024 10-K corpus: tracked HTML SHA-256
  `24a830a0f1256e371d36a1f7f72e5e85a38037d1de2f6f966eb8457db42ff6d6`;
  101-row text SHA-256
  `a2035e3048c9d69945a6851e9f92b6eb64d97ea15ef1ab8fe1812b25e8cd08e0`;
  48-row table SHA-256
  `a60a91dd358656c3667b9700a4e3a2fde5558a67900f62c0a22893fdaf2a6b1d`;
  and 107-row split-text SHA-256
  `856f6aec30636750141a695d864eb08da182160e25143b393b49bc7769436f71`.
  The unpreserved original ingestion command was not reconstructed.
- Embedding endpoint `http://localhost:11434/api/embed`, model
  `qwen3-embedding:8b`, Ollama digest
  `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`.
- Reranker: `Qwen/Qwen3-Reranker-8B` through the Qwen3 API backend. The hosted
  service exposes no immutable model digest, which limits exact reranker
  reproducibility. All full-stack rows recorded `qwen3_api` with no fallback.
- RAGAS was disabled. The harness required `HEAD` to equal the evaluated SHA,
  rejected staged or unstaged tracked changes and runtime-affecting untracked
  files, and threaded the verified Qdrant client and embedding endpoint through
  every enabled mode. Conflicting runtime environment configuration fails closed.
  Nonzero `min_total_score` is rejected because BM25, dense, fused, and reranked
  scores are not comparable. Dataset hashes, case IDs, component traces, raw
  JSONL parsing, zero-byte error files, and the frozen collection fingerprint
  were verified.

The executed command was:

```bash
QDRANT_HOST=localhost QDRANT_PORT=6333 \
QDRANT_COLLECTION_NAME=sec_docs_dense_bm25_pr2_63dcec0 \
QWEN3_EMBED_API_URL=http://localhost:11434/api/embed \
QWEN3_EMBED_MODEL=qwen3-embedding:8b \
SEC_RETRIEVAL_TOP_K=50 SEC_RERANK_CANDIDATE_LIMIT=10 \
SEC_RERANK_TOP_K=10 SEC_RERANK_MODEL=Qwen/Qwen3-Reranker-8B \
PYTHONPATH=src <EVAL_PYTHON> \
scripts/evals/retrieval/run_retrieval_ablation_v1.py \
  --evaluated-sha b61200968f62537af85c8812376b9790a45dfb5f \
  --out-root artifacts/evals/retrieval/ablations/b61200968f62537af85c8812376b9790a45dfb5f \
  --collection sec_docs_dense_bm25_pr2_63dcec0 \
  --expected-points 582 \
  --expected-qdrant-version 1.16.2 \
  --expected-embedding-digest 64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b
```

`<EVAL_PYTHON>` replaces only the temporary virtualenv interpreter path. The
reranker credential was inherited from the environment and was neither printed
nor preserved.

### Frozen configurations

All modes used identical queries, binary relevance labels, AAPL/2024/10-K/table
filters, top 10 scoring, K = 1/3/5/10, collection, embedding model, and evaluator.
Dense and BM25 branch depths were 500; candidate top K was 50; table-parent
deduplication retained 10. Hybrid fusion used RRF K = 60 with 1.0/1.0 weights.

- BM25 only: BM25 vector `bm25`; dense and reranker disabled.
- Dense only: dense vector `dense`; BM25 and reranker disabled.
- Hybrid: both branches and RRF; reranker disabled.
- Hybrid + reranker: unchanged production candidate generation, table-summary
  enrichment, and Qwen3 reranking of the 10 deduplicated candidates.

### Table retrieval results

Metrics are macro averages over all 12 cases. nDCG uses the complete number of
gold labels for IDCG; a retrieved table row is grouped with its parent table.

| Metric | BM25 only | Dense only | Hybrid | Hybrid + reranker |
| --- | ---: | ---: | ---: | ---: |
| hit@1 | 0.9167 | 0.7500 | 0.9167 | 0.8333 |
| hit@3 | 0.9167 | 1.0000 | 0.9167 | 0.9167 |
| hit@5 | 0.9167 | 1.0000 | 0.9167 | 0.9167 |
| hit@10 | 0.9167 | 1.0000 | 0.9167 | 0.9167 |
| recall@1 | 0.4333 | 0.3778 | 0.4333 | 0.3917 |
| recall@3 | 0.8556 | 0.8139 | 0.8139 | 0.7444 |
| recall@5 | 0.9000 | 0.8861 | 0.8583 | 0.9167 |
| recall@10 | 0.9167 | 0.9583 | 0.9167 | 0.9167 |
| mrr@1 | 0.9167 | 0.7500 | 0.9167 | 0.8333 |
| mrr@3 | 0.9167 | 0.8611 | 0.9167 | 0.8611 |
| mrr@5 | 0.9167 | 0.8611 | 0.9167 | 0.8611 |
| mrr@10 | 0.9167 | 0.8611 | 0.9167 | 0.8611 |
| ndcg@1 | 0.9167 | 0.7500 | 0.9167 | 0.8333 |
| ndcg@3 | 0.8837 | 0.7954 | 0.8582 | 0.7808 |
| ndcg@5 | 0.8884 | 0.8164 | 0.8624 | 0.8546 |
| ndcg@10 | 0.8985 | 0.8500 | 0.8906 | 0.8546 |

### Headline deltas

Absolute values are score-point changes; relative values divide that change by
the source score.

| Comparison | Metric | Absolute | Relative |
| --- | --- | ---: | ---: |
| Dense → hybrid | recall@5 | -0.0278 | -3.13% |
| Dense → hybrid | recall@10 | -0.0417 | -4.35% |
| Dense → hybrid | mrr@10 | +0.0556 | +6.45% |
| Dense → hybrid | ndcg@5 | +0.0459 | +5.63% |
| Dense → hybrid | ndcg@10 | +0.0407 | +4.79% |
| Hybrid → full | recall@5 | +0.0583 | +6.80% |
| Hybrid → full | recall@10 | +0.0000 | +0.00% |
| Hybrid → full | mrr@10 | -0.0556 | -6.06% |
| Hybrid → full | ndcg@5 | -0.0078 | -0.90% |
| Hybrid → full | ndcg@10 | -0.0361 | -4.05% |
| Dense → full | recall@5 | +0.0306 | +3.45% |
| Dense → full | recall@10 | -0.0417 | -4.35% |
| Dense → full | mrr@10 | +0.0000 | +0.00% |
| Dense → full | ndcg@5 | +0.0382 | +4.67% |
| Dense → full | ndcg@10 | +0.0046 | +0.54% |

### Latency

Times are per-query, serial, single-pass observations in milliseconds. Candidate
time includes embedding, search/fusion, dedupe, and (for full stack) enrichment;
reranking is isolated. These are not concurrent-load measurements.

| Mode | Candidate mean / p50 / p95 | Rerank mean / p50 / p95 | Total mean / p50 / p95 |
| --- | ---: | ---: | ---: |
| BM25 only | 9.5 / 5.5 / 53 | 0 / 0 / 0 | 9.5 / 5.5 / 53 |
| Dense only | 364.2 / 225.5 / 1314 | 0 / 0 / 0 | 364.2 / 225.5 / 1314 |
| Hybrid | 224.6 / 221.5 / 246 | 0 / 0 / 0 | 224.6 / 221.5 / 246 |
| Hybrid + reranker | 252.9 / 251.0 / 274 | 1110.1 / 902.5 / 2941 | 1363.0 / 1161.0 / 3191 |

### Artifacts and conclusion

Canonical artifacts are under
`artifacts/evals/retrieval/ablations/b61200968f62537af85c8812376b9790a45dfb5f/`.
`comparison.json` SHA-256 is
`2292fca3436ee757236f4d8cc0cd7a88241f8c06a3104e2ef41c09bfd0c79a21`;
`manifest.sha256` SHA-256 is
`1585a47a1e85fbdee5c7caa7b79ff929fb6a6bc2205d76ad36b4dcf1b21875a8`.
The manifest records each raw summary, per-query file, empty error file, and the
comparison artifact.

The canonical rerun reproduced the superseded diagnostic run's deterministic
quality metrics exactly. Latency observations changed, as expected for a
single-pass local/service measurement; only the provenance-hardened rerun is
canonical.

This benchmark does not support the proposed resume statement. Dense → full
improved Recall@5 by 0.0306 absolute (3.45% relative) and nDCG@10 by only 0.0046
absolute (0.54% relative), while Recall@10 regressed by 0.0417 absolute (4.35%
relative). Hybrid without reranking also regressed Recall@5 and Recall@10 versus
dense-only, and reranking traded higher Recall@5 for lower MRR@10 and nDCG@10
versus hybrid. With only 12 AAPL table queries and no compatible text measurement,
no broad retrieval-quality or resume claim is technically defensible. The result
is a baseline for future dataset repair and tuning, not evidence of improvement.

## Native Structured Evidence and Provenance — 2026-09-01 (superseded)

This baseline is superseded because its runtime and evaluator could silently coerce
a malformed `missing_component_groups` string into individual characters. The raw
artifacts and hashes below remain unchanged as historical evidence but are not the
canonical PR4 baseline. The corrected rerun is recorded in the following section.

PR4 replaces synthetic structured-fact text with typed analyst evidence. The
implementation was frozen before measurement; the evidence commit contains only
raw evaluator output and documentation.

### Run identity and provenance

- Evaluated implementation SHA:
  `523ade1a17dce890a35f20acdd0b5ead537aed89`.
- Evidence artifact commit SHA: `4e4dcc0f4b9c01528135c8edb1b5c101af828be7`.
- Dataset: 15 cases at
  `data/evals/agents/v1/agent_eval_routing_v1.jsonl`, SHA-256
  `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd`.
- Planner and analyst: `qwen2.5:14b-instruct`, Ollama digest
  `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
- Dense embedding: `qwen3-embedding:8b` at the local Ollama embed endpoint,
  digest
  `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`.
- Reranker: `Qwen/Qwen3-Reranker-8B` through the configured Qwen3 API backend.
- Qdrant endpoint: `http://127.0.0.1:6334`, version 1.16.2; collection
  `sec_docs_dense_bm25_pr2_63dcec0`, 582 green points. The full collection
  fingerprint was
  `103a1131b373fb651727048b6a875e4bd96b64754ebafd41c42dc6dcccfc5df8`
  after the run.
- Corpus provenance is the pinned PR2 AAPL FY2024 10-K corpus: tracked HTML
  SHA-256
  `24a830a0f1256e371d36a1f7f72e5e85a38037d1de2f6f966eb8457db42ff6d6`;
  101-row text SHA-256
  `a2035e3048c9d69945a6851e9f92b6eb64d97ea15ef1ab8fe1812b25e8cd08e0`;
  48-row table SHA-256
  `a60a91dd358656c3667b9700a4e3a2fde5558a67900f62c0a22893fdaf2a6b1d`;
  107-row split-text SHA-256
  `856f6aec30636750141a695d864eb08da182160e25143b393b49bc7769436f71`.
  Corpus construction used `sec-parser==0.58.1`; the original ingestion command
  remains unpreserved and was not reconstructed.
- Runtime: Python 3.12.12. RAGAS was disabled.

The executed command was:

```bash
QWEN3_RERANK_MODEL=Qwen/Qwen3-Reranker-8B \
SEC_RERANK_MODEL=Qwen/Qwen3-Reranker-8B \
QWEN3_EMBED_API_URL=http://127.0.0.1:11434/api/embed \
QWEN3_EMBED_MODEL=qwen3-embedding:8b \
QDRANT_URL=http://127.0.0.1:6334 \
QDRANT_HOST=127.0.0.1 QDRANT_PORT=6334 \
QDRANT_COLLECTION_NAME=sec_docs_dense_bm25_pr2_63dcec0 \
SEC_USER_AGENT='<SEC_CONTACT>' \
FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH='<CHECKPOINTER_DB>' \
PYTHONPATH=src <EVAL_PYTHON> scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/baselines/523ade1 \
  --analyst-model ollama/qwen2.5:14b-instruct \
  --planner-model ollama/qwen2.5:14b-instruct \
  --deterministic-threshold 0.90 \
  --max-critical-failure-rate 0
```

`<SEC_CONTACT>` replaces the SEC contact value, `<CHECKPOINTER_DB>` replaces the
temporary SQLite path, and `<EVAL_PYTHON>` replaces the temporary virtualenv
interpreter. Reranker credentials and endpoint were inherited from the environment
and were neither printed nor preserved. An earlier configuration diagnostic used
an invalid placeholder-style SEC contact and received SEC 403 responses. It was
discarded outside the repository and is not canonical because it did not exercise
successful structured execution.

### Results

The evaluator parsed all 15 cases, produced 15 valid rows and zero evaluator errors.
The deterministic score mean was 0.950549. The strict release gate did not pass:
two cases were critical, for a 0.133333 critical-failure rate against the required
zero; `deterministic_gate_pass=false` and `overall_pass=false`. The empty RAGAS
object and `ragas_gate_pass=true` mean only that RAGAS was disabled, not that an
LLM-judge gate ran.

| Diagnostic | Result |
| --- | ---: |
| Successful structured execution results | 7 |
| Typed structured-fact contexts | 7 |
| Analyst-visible typed contexts | 7 |
| Synthetic structured-text contexts | 0 |
| Typed representation rate | 100% |
| Analyst visibility rate | 100% |
| Metric ID coverage | 100% |
| Finite numeric-value coverage | 100% |
| Accession coverage | 100% |
| Report-date coverage | 100% |
| Filed-date coverage | 100% |
| Source-URL coverage | 100% |

Route, intent, metadata, KB-lane, structured-lane, structured-metric, and
structured-status match rates were all 100%. Reported-status, effective-status,
and failure-stage match rates were 86.67%; analyst-status match was 84.62%; citation
match was 83.33%; tool-use match was 100%; compute match was 0%.

The two critical rows were:

- `AGENT_V1_HYBRID_001`, score 0.642857: the local analyst returned a schema-invalid
  final answer and the run failed at the analyst stage. Its successful structured
  fact was nevertheless typed, visible, and fully provenanced.
- `AGENT_V1_ANALYST_001`, score 0.615385: the analyst returned
  `insufficient_data`, did not produce the required computation, and did not meet
  the citation minimum. The same case had already exposed local-model/calculator
  instability in the PR3 baseline.

These failures are preserved rather than tuned away. The PR4 evidence-contract
claim is narrow: every admitted structured fact was typed, analyst-visible, finite,
and fully covered by the measured filing provenance fields. This run is not evidence
that the overall route-aware release gate passed.

Mean stage timings were 35,925 ms planner, 59,082 ms retrieval, 250 ms structured
execution, 50,618 ms analyst, and 145,943 ms total per query. The complete serial
run took 2,189,750 ms. These are single-run local/service observations, not a
concurrency benchmark.

### Artifacts and limitations

Canonical raw artifacts are under
`artifacts/evals/agents/v1/baselines/523ade1/`:

- `summary.json` SHA-256
  `39aac83ea13328a5e89e729ac092a2efdbdbaaf51bda59c9f1b8a820ca5b8b10`.
- `per_query.jsonl` SHA-256
  `84238d212d250d9b1006b473d1f0855cd5d830b59f41433a235a9be85b81660b`.
- zero-byte `errors.jsonl` SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The raw per-query trace is preserved byte-for-byte. Two analyst table-hydration
diagnostics include the ephemeral absolute evaluator worktree path in an
`attempted_file` field; no credential, SEC contact, authorization header,
environment snapshot, user-home path, or API-key value is present. The missing table
sidecar also limits interpretation of the two analyst-oriented cases. Post-run
AnyIO/MCP async-generator shutdown warnings remain the non-gating cleanup tracked
in issue #17; PR4 does not address them.

## Native Structured Evidence and Provenance — 2026-09-02

This canonical rerun uses one strict shared predicate for structured-fact
`missing_component_groups`: the raw value must be a list containing only non-empty
strings, which are trimmed without container or element coercion. Runtime admission
maps malformed metadata to `STRUCTURED_FACT_INVALID_EVIDENCE`, and evaluator
diagnostics exclude the same result from successful-execution counts.

### Run identity and unchanged inputs

- Evaluated implementation SHA:
  `b3d49134038adde61d160677cfc79d6bb8b06ea5`.
- Evidence artifact commit SHA: `ae10ca3b22ae1cc1833e1d0ee88fbb1798450787`.
- Dataset: the same 15 cases at
  `data/evals/agents/v1/agent_eval_routing_v1.jsonl`, SHA-256
  `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd`
  before and after the run.
- Planner and analyst: `qwen2.5:14b-instruct`, Ollama digest
  `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
- Dense embedding: `qwen3-embedding:8b` at the same local Ollama endpoint,
  digest
  `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b`.
- Reranker: unchanged `Qwen/Qwen3-Reranker-8B` through the configured Qwen3 API
  backend.
- Qdrant endpoint: isolated `http://127.0.0.1:6334`, version 1.16.2;
  collection `sec_docs_dense_bm25_pr2_63dcec0`, 582 green points. The full
  collection fingerprint before and after the run was
  `103a1131b373fb651727048b6a875e4bd96b64754ebafd41c42dc6dcccfc5df8`.
- Corpus provenance, retrieval configuration, evaluator formulas, thresholds, and
  SEC tool behavior were unchanged from the superseded run. RAGAS was disabled.

The executed command was:

```bash
QWEN3_RERANK_MODEL=Qwen/Qwen3-Reranker-8B \
SEC_RERANK_MODEL=Qwen/Qwen3-Reranker-8B \
QWEN3_EMBED_API_URL=http://127.0.0.1:11434/api/embed \
QWEN3_EMBED_MODEL=qwen3-embedding:8b \
QDRANT_URL=http://127.0.0.1:6334 \
QDRANT_HOST=127.0.0.1 QDRANT_PORT=6334 \
QDRANT_COLLECTION_NAME=sec_docs_dense_bm25_pr2_63dcec0 \
SEC_USER_AGENT='<SEC_CONTACT>' \
FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH='<CHECKPOINTER_DB>' \
PYTHONPATH=src <EVAL_PYTHON> scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/baselines/b3d4913 \
  --analyst-model ollama/qwen2.5:14b-instruct \
  --planner-model ollama/qwen2.5:14b-instruct \
  --deterministic-threshold 0.90 \
  --max-critical-failure-rate 0
```

The placeholders have the same narrow meanings as in the superseded run. The
implementation checkout was clean and exactly matched the evaluated SHA before the
single complete run. No case was retried selectively and no result was tuned.

### Results

All 15 dataset rows parsed and produced valid evaluator rows, with zero evaluator
errors. The deterministic score mean was 0.922772. The strict gate did not pass:
three cases were critical, for a 0.20 critical-failure rate against the required
zero; `deterministic_gate_pass=false` and `overall_pass=false`. RAGAS was disabled,
so `ragas_gate_pass=true` does not represent an LLM-judge measurement.

| Diagnostic | Result |
| --- | ---: |
| Successful structured execution results | 7 |
| Typed structured-fact contexts | 7 |
| Analyst-visible typed contexts | 7 |
| Synthetic structured-text contexts | 0 |
| Typed representation rate | 100% |
| Analyst visibility rate | 100% |
| Metric ID coverage | 100% |
| Finite numeric-value coverage | 100% |
| Accession coverage | 100% |
| Report-date coverage | 100% |
| Filed-date coverage | 100% |
| Source-URL coverage | 100% |

Route, intent, metadata, KB-lane, structured-lane, structured-metric, and
structured-status match rates remained 100%. Reported-status, effective-status, and
failure-stage match rates were 80%; analyst-status match was 76.92%; citation match
was 75%; tool-use match was 100%; compute match was 0%.

The two prior analyst-stage failures reproduced unchanged:

- `AGENT_V1_HYBRID_001`, score 0.642857: analyst schema error, zero citations;
  its structured fact remained typed, visible, and fully provenanced.
- `AGENT_V1_ANALYST_001`, score 0.615385: `insufficient_data`, with the required
  computation and citation minimum still unmet.

One additional failure was preserved without rerunning:

- `AGENT_V1_ANALYST_003`, score 0.583333: the Qwen3 hosted reranker endpoint failed
  DNS resolution, so retrieval failed before analyst execution. This is an external
  service failure, not a structured-evidence admission failure.

The complete serial run took 1,889,040 ms. Mean measured stage times were 32,532 ms
planner, 47,888 ms retrieval, 248 ms structured execution, and 45,172 ms analyst.
Per-row total timings exceeded wall-clock duration for several rows and are not
reliable for latency claims; the raw values are preserved unchanged.

### Artifacts and limitations

Canonical raw artifacts are under
`artifacts/evals/agents/v1/baselines/b3d4913/`:

- `summary.json` SHA-256
  `610ea7b6df03bedfd7a193c47f3d2a68689204e06c424a2710ec90a82cc4f37c`.
- `per_query.jsonl` SHA-256
  `5c25ed820c886a2dfe594e5b9eefca3e903bfdf17b9924f87d36de29ce232e14`.
- zero-byte `errors.jsonl` SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The raw artifacts preserve the hosted-reranker DNS failure and the same ephemeral
worktree paths in table-hydration diagnostics as the superseded run. They contain no
credential, SEC contact, authorization header, user-home path, or API-key value.
The 15 case IDs exactly match the frozen dataset; all successful tool results carry
list-shaped string-only missing-component metadata, and all seven analyst evidence
items satisfy the strict typed contract. Post-run AnyIO/MCP shutdown warnings remain
tracked in issue #17 and are out of PR4 scope.

## Per-Lane Status and Degradation — 2026-09-02

### Run identity

| Field | Value |
|---|---|
| Status | Canonical PR5 degradation baseline; live regression gate failed and was preserved |
| Evaluated implementation SHA | `25e359d85776dc422e0d4d72b47152b407862d30` |
| Dataset | `data/evals/agents/v1/agent_eval_degradation_v1.jsonl` |
| Dataset SHA-256 | `89bae38e5439a561a6241ee3d53ac96d1b19d43de81a8ffd7368890289999900` |
| Deterministic cases | 11 |
| Live regression dataset | unchanged `data/evals/agents/v1/agent_eval_routing_v1.jsonl` |
| Live dataset SHA-256 | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` |
| Live cases | 15 |
| Python | `3.12.12` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 points |
| RAGAS | Disabled |

The implementation commit contains runtime behavior, prompts, evaluator logic,
tests, and the frozen degradation dataset. Only immutable artifacts and
documentation were changed after that SHA.

### Deterministic degradation matrix

Command:

```bash
PYTHONPATH=.:src python scripts/evals/agents/eval_agent_degradation_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_degradation_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/degradation/baselines/25e359d
```

| Metric | Result |
|---|---:|
| Degradation classification accuracy | 100% |
| Failure-containment rate | 100% |
| All-lanes-unusable fail-closed rate | 100% |
| Degradation-disclosure rate | 100% |
| Runtime/evaluator consistency rate | 100% |
| Overall graceful-degradation behavior accuracy | 100% |

The matrix covers both lanes succeeding; either lane failing while the sibling
succeeds; both failing; usable KB partial; unusable structured partial;
structured-only and KB-only unusable execution; degraded evidence followed by an
analyst failure; legitimate non-filing zero-lane execution; and filing-task
zero-evidence rejection.

Artifacts:

- `artifacts/evals/agents/v1/degradation/baselines/25e359d/summary.json`, SHA-256
  `811de3c7be68f72297202d0ece04015fe30d8013a6f58f22ee3d09524f51d743`.
- `artifacts/evals/agents/v1/degradation/baselines/25e359d/per_case.jsonl`, SHA-256
  `c59984a087fcbf4113d7d7a9a8ceea65de97411ac4fadfffb755c669748f9a38`.
- `artifacts/evals/agents/v1/degradation/baselines/25e359d/manifest.json` records
  both evaluation commands, identities, hashes, and cleanup limitation.

### Unchanged route-aware live gate

The 15-case RAGAS-disabled gate was run once after freezing the implementation.
Environment values for the SEC contact and reranker credential were supplied but
are not recorded. The effective command was:

```bash
QWEN3_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
SEC_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
QWEN3_EMBED_API_URL='http://127.0.0.1:11434/api/embed' \
QWEN3_EMBED_MODEL='qwen3-embedding:8b' \
QDRANT_URL='http://127.0.0.1:6333' \
QDRANT_COLLECTION_NAME='sec_docs_dense_bm25_pr2_63dcec0' \
SEC_USER_AGENT='<SEC_CONTACT>' \
FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH='<CHECKPOINTER_DB>' \
PYTHONPATH=.:src python scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/baselines/25e359d \
  --planner-model 'ollama/qwen2.5:14b-instruct' \
  --analyst-model 'ollama/qwen2.5:14b-instruct' \
  --deterministic-threshold 0.90 \
  --max-critical-failure-rate 0
```

All 15 rows were valid and the evaluator recorded zero errors. The deterministic
score was `0.926100`; five rows had critical failures, for a critical-failure rate
of `0.333333`. The strict gate therefore failed (`overall_pass=false`). This result
is preserved without case reruns or tuning.

Runtime/evaluator lane-status, effective-status, failure-stage, and degradation
consistency rates were all 100%. Seven successful structured executions produced
seven typed, analyst-visible contexts with 100% coverage for metric ID, finite
numeric value, accession, report date, filed date, and source URL. No synthetic
structured-text contexts were observed.

The five critical rows were:

- `AGENT_V1_HYBRID_001` and `AGENT_V1_HYBRID_002`: runtime evidence degradation
  was correctly disclosed, but the frozen dataset expects unqualified completion
  and does not allow degradation.
- `AGENT_V1_HYBRID_003`: analyst `insufficient_data` caused an analyst-stage
  failure instead of the frozen successful expectation.
- `AGENT_V1_ANALYST_001`: KB retrieval supplied no usable evidence, so PR5 failed
  closed at retrieval and skipped the required computation and citation.
- `AGENT_V1_ANALYST_002`: runtime evidence degradation was correctly disclosed,
  while the frozen dataset expects unqualified completion.

Artifacts:

- `artifacts/evals/agents/v1/baselines/25e359d/summary.json`, SHA-256
  `0823393a9d6279e5c0be6079bf08b13ce496100f5b9ed33d6dffb5ed546f3bcc`.
- `artifacts/evals/agents/v1/baselines/25e359d/per_query.jsonl`, SHA-256
  `563b4ad95a3edb5ed7967e3c450dbd83a1804198d0a227e947d3b98d995a31bf`.
- zero-byte `artifacts/evals/agents/v1/baselines/25e359d/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The frozen live cases do not intentionally exercise degradation; their PR5 role is
regression observation. The run took 2,431,747 ms (about 40m32s). After every
artifact was written, the previously documented MCP/AnyIO asynchronous-generator
cleanup hang recurred. The completed wrapper was terminated with SIGTERM; no case
was rerun, and the original Qdrant container state was restored.

## PR5 Admission-Loss Correction — 2026-09-03

This correction closes the two PR5 admission-loss gaps while preserving the
original `25e359d` implementation and evidence artifacts unchanged.

### Run identity

| Field | Value |
|---|---|
| Status | Canonical admission-loss correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `06835feaf10237aa79bd494d9fef0f4fc1aa0c0c` |
| Degradation dataset | `data/evals/agents/v1/agent_eval_degradation_v1.jsonl` |
| Degradation dataset SHA-256 | `0c7fa622902bf00a89a93120e8a0ad8866e6195f54810adf68f75b51f2cc0ca6` |
| Deterministic cases | 13 |
| Live regression dataset | unchanged `data/evals/agents/v1/agent_eval_routing_v1.jsonl` |
| Live dataset SHA-256 | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` |
| Live cases | 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Reranker | `Qwen/Qwen3-Reranker-8B` through the configured Qwen3 API backend |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 green points; fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744` before and after |
| RAGAS | Disabled |

The repository did not pin or recommend a pytest version. `pytest==9.0.2` was
installed only in the `finsearch-arm` environment; no requirements file, project
metadata, or lockfile changed. The tracked worktree was clean and exactly matched
the evaluated implementation SHA before measurement.

The corrected runtime counts retrieved KB candidates against analyst-visible KB
evidence. Some admitted evidence plus any expected admission loss is usable
`partial`; zero admitted KB evidence is unusable `failed`; complete admission is
`ok`. Structured `ok` requires every successfully executed requested fact expected
to yield evidence to survive PR4 admission. A successful-but-rejected result is
usable `partial` when another fact survives and unusable `failed` otherwise.

The shadow evaluator independently derives these outcomes from raw execution
results plus analyst-visible contexts and issues. It does not import or call the
runtime lane helper. Compact retrieval output carries the exact final
`retrieved_candidate_count`, allowing count-only losses at the five-item analyst
visibility boundary to be checked independently.

### Deterministic degradation matrix

Command:

```bash
conda run -n finsearch-arm --no-capture-output \
  env PYTHONPATH=.:src python \
  scripts/evals/agents/eval_agent_degradation_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_degradation_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/degradation/baselines/06835fe
```

| Metric | Result |
|---|---:|
| Degradation classification accuracy | 100% |
| Failure-containment rate | 100% |
| All-lanes-unusable fail-closed rate | 100% |
| Degradation-disclosure rate | 100% |
| Runtime/evaluator consistency rate | 100% |
| Overall graceful-degradation behavior accuracy | 100% |

The two added cases exercise real admission behavior:

- `PR5_KB_ADMISSION_PARTIAL`: two retrieved candidates, one real hydration
  rejection, and one admitted analyst-visible text item produce usable `partial`,
  top-level `degraded`, and analyst execution.
- `PR5_STRUCTURED_MIXED_ADMISSION`: two raw tool-`ok` results, one PR4 rejection,
  and one admitted typed fact produce usable `partial`, top-level `degraded`, and
  analyst execution.

Artifacts:

- `artifacts/evals/agents/v1/degradation/baselines/06835fe/summary.json`, SHA-256
  `709d04f659981a76db054cd363c9310efb9efa3d1b200110bdce80bdaf088015`.
- `artifacts/evals/agents/v1/degradation/baselines/06835fe/per_case.jsonl`, SHA-256
  `c25bbe4a07e3c2780374e144f9fa9357b7e68df943e970a668cf4f9c027e08f6`.
- `artifacts/evals/agents/v1/degradation/baselines/06835fe/manifest.json` records
  both commands, identities, hashes, environment provenance, and limitations.

### Fresh route-aware live gate

The same 15-case RAGAS-disabled dataset was run once. Environment values for the
SEC contact and reranker credential were supplied but are not recorded. The
effective command was:

```bash
DASHSCOPE_API_KEY='<RERANKER_CREDENTIAL>' \
QWEN3_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
SEC_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
QWEN3_EMBED_API_URL='http://127.0.0.1:11434/api/embed' \
QWEN3_EMBED_MODEL='qwen3-embedding:8b' \
QDRANT_URL='http://127.0.0.1:6333' \
QDRANT_HOST='127.0.0.1' \
QDRANT_PORT='6333' \
QDRANT_COLLECTION_NAME='sec_docs_dense_bm25_pr2_63dcec0' \
SEC_USER_AGENT='<SEC_CONTACT>' \
FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH='<CHECKPOINTER_DB>' \
PYTHONPATH=.:src conda run -n finsearch-arm --no-capture-output python \
  scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/baselines/06835fe \
  --planner-model 'ollama/qwen2.5:14b-instruct' \
  --analyst-model 'ollama/qwen2.5:14b-instruct' \
  --deterministic-threshold 0.90 \
  --max-critical-failure-rate 0
```

All 15 rows were valid and the evaluator recorded zero errors. The deterministic
score was `0.948540`; three rows had critical failures, for a critical-failure rate
of `0.20`. The strict gate therefore failed (`overall_pass=false`) and the result
is preserved without selective reruns or tuning.

Runtime/evaluator lane-status, effective-status, failure-stage, and degradation
consistency rates were all 100%. Seven successful structured executions produced
seven typed, analyst-visible contexts with 100% coverage for metric ID, finite
numeric value, accession, report date, filed date, and source URL. No synthetic
structured-text contexts were observed.

The three critical rows were:

- `AGENT_V1_HYBRID_001`: the analyst exceeded its tool-loop budget and its final
  fallback output failed the analyst contract, producing an analyst-stage failure
  and no accepted citation instead of the expected successful answer.
- `AGENT_V1_ANALYST_001`: multiple successful calculator calls could not be
  uniquely matched to the final calculation, producing
  `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted citation.
- `AGENT_V1_ANALYST_002`: one retrieval job timed out while three KB contexts
  survived. Runtime and shadow both classified the KB lane as usable `partial` and
  the run as `degraded`; the frozen row expects unqualified completion and does not
  allow degradation.

Artifacts:

- `artifacts/evals/agents/v1/baselines/06835fe/summary.json`, SHA-256
  `af36479d90980fcf47e4d9756e75378563ccebbd1e185efe504f9eb640432b6d`.
- `artifacts/evals/agents/v1/baselines/06835fe/per_query.jsonl`, SHA-256
  `489ad032d4d46c4ca61c11b493bd4dce156ee3e91159576b39578b409c39f1a5`.
- zero-byte `artifacts/evals/agents/v1/baselines/06835fe/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The live run took 2,352,473 ms (about 39m12s). The known MCP/AnyIO
asynchronous-generator warnings appeared during final interpreter shutdown, after
all artifacts were written. The evaluator exited with status zero; no manual
termination or case rerun was needed.

### Corpus rebuild and reproducibility limitation

The tracked AAPL FY2024 10-K HTML retained SHA-256
`24a830a0f1256e371d36a1f7f72e5e85a38037d1de2f6f966eb8457db42ff6d6`.
Freshly generated 101-row text, 48-row table, and 107-row split-text sidecars
retained their historical SHA-256 values
`a2035e3048c9d69945a6851e9f92b6eb64d97ea15ef1ab8fe1812b25e8cd08e0`,
`a60a91dd358656c3667b9700a4e3a2fde5558a67900f62c0a22893fdaf2a6b1d`, and
`856f6aec30636750141a695d864eb08da182160e25143b393b49bc7769436f71`.

The rebuilt 582-point collection fingerprint was
`0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744`
before and after the live run, but differs from the historical PR5 fingerprint
`103a1131b373fb651727048b6a875e4bd96b64754ebafd41c42dc6dcccfc5df8`.
The exact historical vector snapshot therefore was not reproduced; the preserved
model digests, source and sidecar hashes, point count, and stable within-run
fingerprint bound this fresh result without claiming collection equivalence.

### Local test verification

The focused admission-loss, compact-retrieval, degradation-matrix, and route-aware
evaluator suite passed 67 tests. The full suite completed with 394 passing tests,
47 passing subtests, and two failures that were reproduced unchanged against the
pre-fix branch head:

- `tests/agents/planner/test_planner_routes.py::test_planner_routes_match_expected_routes`
  (`alias_002` expected `structured_fact`, observed `kb`).
- `tests/agents/retrieval/test_retrieval_workflow_no_tool_call.py::test_retrieval_no_tool_call_increments_attempts_and_stops_on_budget`
  (expected two attempts, observed one).

The installed environment also contains a top-level `tests` package, so the full
run used an in-process namespace shim to select this repository's namespace-only
test directory. Neither pre-existing failure touches the admission-loss changes.

## PR5 Post-Review Integration Correction — 2026-09-03

This correction addresses the three findings from the fresh review of the
admission-loss implementation. The `25e359d` and `06835fe` implementations and
evidence directories remain immutable.

### Run identity

| Field | Value |
|---|---|
| Status | Canonical post-review correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `7f258a378b75b3781d5ebf45edf70c597b381511` |
| Degradation dataset SHA-256 / cases | `0c7fa622902bf00a89a93120e8a0ad8866e6195f54810adf68f75b51f2cc0ca6` / 13 |
| Live dataset SHA-256 / cases | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` / 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 green points; fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744` before and after |
| RAGAS | Disabled |

`pytest==9.0.2` remains an environment-only installation in `finsearch-arm`.
Repository dependency files were not changed, and the tracked worktree exactly
matched the evaluated implementation SHA before measurement.

The correction makes three integration guarantees:

- Batch consumers count successful `degraded` results as completed and do not
  attach a false `non-completed-status` error.
- All-ambiguous defensive structured rejections resolve to `ambiguous`, only sets
  composed entirely of explicit unsupported classes resolve to `unsupported`, and
  mixed or unknown sets fail closed as `failed`.
- The independent shadow evaluator reconstructs the exact typed degradation
  summary from raw results and analyst-visible evidence. It compares the full
  notice at the runtime boundary and verifies that the analyst packet received the
  same summary; it does not import the runtime lane helper.

### Deterministic degradation matrix

Command:

```bash
conda run -n finsearch-arm --no-capture-output \
  env PYTHONPATH=.:src python \
  scripts/evals/agents/eval_agent_degradation_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_degradation_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/degradation/baselines/7f258a3
```

All 13 cases passed. Degradation classification, failure containment,
all-lanes-unusable fail-closed behavior, disclosure, runtime/evaluator
consistency, and overall graceful-degradation accuracy were each 100%.

Artifacts:

- `artifacts/evals/agents/v1/degradation/baselines/7f258a3/summary.json`, SHA-256
  `709d04f659981a76db054cd363c9310efb9efa3d1b200110bdce80bdaf088015`.
- `artifacts/evals/agents/v1/degradation/baselines/7f258a3/per_case.jsonl`, SHA-256
  `c25bbe4a07e3c2780374e144f9fa9357b7e68df943e970a668cf4f9c027e08f6`.
- `artifacts/evals/agents/v1/degradation/baselines/7f258a3/manifest.json` records
  commands, identities, hashes, test results, and limitations.

### Fresh route-aware live gate

The unchanged 15-case dataset was run once with a fresh checkpointer and output
directory. Environment values for the SEC contact and reranker credential were
supplied but are not recorded. The effective command was:

```bash
DASHSCOPE_API_KEY='<RERANKER_CREDENTIAL>' \
QWEN3_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
SEC_RERANK_MODEL='Qwen/Qwen3-Reranker-8B' \
QWEN3_EMBED_API_URL='http://127.0.0.1:11434/api/embed' \
QWEN3_EMBED_MODEL='qwen3-embedding:8b' \
QDRANT_URL='http://127.0.0.1:6333' \
QDRANT_HOST='127.0.0.1' \
QDRANT_PORT='6333' \
QDRANT_COLLECTION_NAME='sec_docs_dense_bm25_pr2_63dcec0' \
SEC_USER_AGENT='<SEC_CONTACT>' \
FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH='<CHECKPOINTER_DB>' \
PYTHONPATH=.:src conda run -n finsearch-arm --no-capture-output python \
  scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir artifacts/evals/agents/v1/baselines/7f258a3 \
  --planner-model 'ollama/qwen2.5:14b-instruct' \
  --analyst-model 'ollama/qwen2.5:14b-instruct' \
  --deterministic-threshold 0.90 \
  --max-critical-failure-rate 0
```

All 15 rows were valid and the evaluator recorded zero errors. The deterministic
score was `0.961874`; two rows had critical failures, for a critical-failure rate
of `0.133333`. The zero-critical-row gate therefore remained false and the
single-pass result was preserved without tuning or selective reruns.

Runtime/shadow lane status, effective status, failure stage, and exact degradation
summary consistency were all 100%. Seven successful structured executions produced
seven typed, analyst-visible contexts with 100% representation, visibility, and
provenance-field coverage. No synthetic structured-text contexts were observed.

The two critical rows were:

- `AGENT_V1_HYBRID_001`: the analyst exceeded its tool-loop budget and its final
  fallback output failed the analyst contract, so it produced an analyst-stage
  failure with no accepted citation.
- `AGENT_V1_ANALYST_001`: multiple successful calculator calls could not be
  uniquely matched to the final calculation, producing
  `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted citation.

Artifacts:

- `artifacts/evals/agents/v1/baselines/7f258a3/summary.json`, SHA-256
  `60515de3af25c92de646b9ee922962a27afa93a614b2c868e1a75777f39b8246`.
- `artifacts/evals/agents/v1/baselines/7f258a3/per_query.jsonl`, SHA-256
  `04b690f6230226815742859677ab762cac52994b9d911c2c672a63ea548a8179`.
- Zero-byte `artifacts/evals/agents/v1/baselines/7f258a3/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The run took 2,219,597 ms (about 37 minutes). The known MCP/AnyIO
asynchronous-generator warnings appeared during final interpreter shutdown after
all artifacts were written; the process exited successfully.

### Provenance and local verification

The source HTML and three rebuilt sidecars retained the hashes recorded in the
`06835fe` baseline. The Qdrant fingerprint was identical before and after this run,
but still differs from the historical PR5 fingerprint, so exact historical
collection equivalence is not claimed. The hosted reranker exposes no immutable
service-side model digest.

The focused batch, lane-classification, degradation-matrix, and route-aware
evaluator suite passed 72 tests. The full suite completed with 402 passing tests,
47 passing subtests, and the same two pre-existing failures documented in the
`06835fe` baseline. Neither failure touches this correction.

## PR5 Final Review Closure — 2026-09-03

This final correction closes the two findings from the review of `7ad524c`. The
`25e359d`, `06835fe`, and `7f258a3` evidence remains immutable.

### Run identity

| Field | Value |
|---|---|
| Status | Canonical final PR5 correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `cbff8ce06cdbfc13f9c19c445c59965d326d61b9` |
| Degradation dataset SHA-256 / cases | `0c7fa622902bf00a89a93120e8a0ad8866e6195f54810adf68f75b51f2cc0ca6` / 13 |
| Live dataset SHA-256 / cases | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` / 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 green points; fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744` before and after |
| RAGAS | Disabled |

The primary CLI now treats a usable `degraded` result as successful, displays the
degraded status and canonical notice before the analyst answer, and exits zero.
The runtime and independent shadow evaluator also count `EMPTY_TEXT_CONTEXT` as KB
admission loss when later candidates backfill the three visible KB slots; the lane
remains usable but becomes `partial` and the top-level result is `degraded`.

`pytest==9.0.2` remained environment-only, repository dependency files were
unchanged, and the tracked worktree matched the implementation SHA before both
measurements.

### Deterministic and live results

The unchanged 13-case deterministic command wrote to
`artifacts/evals/agents/v1/degradation/baselines/cbff8ce`. All six matrix rates
were 100%. Its summary SHA-256 is
`709d04f659981a76db054cd363c9310efb9efa3d1b200110bdce80bdaf088015` and its
per-case SHA-256 is
`c25bbe4a07e3c2780374e144f9fa9357b7e68df943e970a668cf4f9c027e08f6`.

The unchanged 15-case route-aware live command used the same redacted environment
and thresholds as the `7f258a3` run, a fresh checkpointer, and output directory
`artifacts/evals/agents/v1/baselines/cbff8ce`. It ran exactly once. All 15 rows
were valid with zero evaluator errors. The deterministic score was `0.961874` and
two rows were critical, for a `0.133333` critical-failure rate; the strict
zero-critical-row gate therefore remained false.

Runtime/shadow lane status, effective status, failure stage, and exact degradation
summary consistency were all 100%. Seven successful structured executions yielded
seven typed, analyst-visible contexts with complete provenance-field coverage and
no synthetic structured-text contexts.

The critical rows were unchanged:

- `AGENT_V1_HYBRID_001`: analyst tool-loop exhaustion and an invalid fallback
  output caused analyst-stage failure with no accepted citation.
- `AGENT_V1_ANALYST_001`: ambiguous matching across successful calculator calls
  produced `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted
  citation.

Live artifacts:

- `artifacts/evals/agents/v1/baselines/cbff8ce/summary.json`, SHA-256
  `8d00c760ae69b85b80426ca0537fe885479888fad68ede6f361ef96d677a4709`.
- `artifacts/evals/agents/v1/baselines/cbff8ce/per_query.jsonl`, SHA-256
  `c013a54667addc3c14dc402ade692b7c3d1400276b1b407a77024e941a4f067b`.
- Zero-byte `artifacts/evals/agents/v1/baselines/cbff8ce/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The live run took 2,178,744 ms (about 36 minutes). Known MCP/AnyIO shutdown
warnings appeared only after artifact generation; the evaluator exited
successfully. The post-run collection fingerprint and all dataset/source/sidecar
hashes matched their pre-run values. Exact historical collection equivalence is
still not claimed, and the hosted reranker exposes no immutable service-side
digest.

The focused CLI, batch, lane, degradation, and route-aware evaluator suite passed
75 tests. The full suite completed with 405 passing tests, 47 passing subtests, and
the same two pre-existing failures documented above. Neither failure touches PR5.
Full commands, redacted environment, hashes, provenance, and limitations are in
`artifacts/evals/agents/v1/degradation/baselines/cbff8ce/manifest.json`.

## PR5 Admission Truthfulness Audit Closure — 2026-09-03

This correction closes the findings from the final review of `4fc9082`. All
evidence keyed by `25e359d`, `06835fe`, `7f258a3`, and `cbff8ce` remains
unchanged.

### Run identity

| Field | Value |
|---|---|
| Status | Final reviewed PR5 correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `29d22268a27e51a62157eeae53649858d74e5242` |
| Degradation dataset SHA-256 / cases | `0c7fa622902bf00a89a93120e8a0ad8866e6195f54810adf68f75b51f2cc0ca6` / 13 |
| Live dataset SHA-256 / cases | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` / 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 green points; fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744` before and after |
| RAGAS | Disabled |

The runtime now emits an analyst-visible issue whenever a retrieved candidate
cannot be admitted as supported table or text evidence. If later candidates fill
the three visible KB slots, the surviving evidence remains usable but the KB lane
is `partial`; no surviving KB evidence still produces `failed` and unusable.

Defensive structured-fact capability rejections retain per-request identity even
when their messages match. Runtime classification requires complete request-index
coverage, while the independent shadow evaluator derives the same truth from raw
results plus analyst-visible contexts and issue metadata without importing the
runtime lane helper.

`pytest==9.0.2` remained environment-only, repository dependency files were
unchanged, and the tracked worktree matched the implementation SHA before both
measurements. The final independent code review found no actionable P0–P2 issue.

### Deterministic and live results

The unchanged 13-case deterministic command wrote to
`artifacts/evals/agents/v1/degradation/baselines/29d2226`. All six matrix rates
were 100%. Its summary SHA-256 is
`709d04f659981a76db054cd363c9310efb9efa3d1b200110bdce80bdaf088015` and its
per-case SHA-256 is
`c25bbe4a07e3c2780374e144f9fa9357b7e68df943e970a668cf4f9c027e08f6`.

The unchanged 15-case route-aware live command used the same redacted environment
and thresholds as the earlier PR5 runs, a fresh checkpointer, and output directory
`artifacts/evals/agents/v1/baselines/29d2226`. It ran exactly once. All 15 rows
were valid with zero evaluator errors. The deterministic score was `0.948540` and
three rows were critical, for a `0.200000` critical-failure rate; the strict
zero-critical-row gate therefore remained false.

Runtime/shadow lane status, effective status, failure stage, and exact degradation
summary consistency were all 100%. Seven successful structured executions yielded
seven typed, analyst-visible contexts with complete provenance-field coverage and
no synthetic structured-text contexts.

The critical rows were:

- `AGENT_V1_HYBRID_001`: analyst tool-loop exhaustion and an invalid fallback
  output caused analyst-stage failure with no accepted citation.
- `AGENT_V1_ANALYST_001`: ambiguous matching across successful calculator calls
  produced `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted
  citation.
- `AGENT_V1_ANALYST_002`: the analyst correctly returned `insufficient_data`
  because `market_price_per_share` was unavailable, while the frozen expectation
  required completion; this produced reported-status, effective-status, and
  failure-stage mismatches.

Live artifacts:

- `artifacts/evals/agents/v1/baselines/29d2226/summary.json`, SHA-256
  `fc387ebdda545d51704d3943b57091dbddbdfc78a7484f0ed86f0dec837c064b`.
- `artifacts/evals/agents/v1/baselines/29d2226/per_query.jsonl`, SHA-256
  `53b667754e259a5e2eda7494f074f7a6c6e140c180ebb758744a4e59728ca913`.
- Zero-byte `artifacts/evals/agents/v1/baselines/29d2226/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The live run took 2,183,886 ms (about 36 minutes). Known MCP/AnyIO shutdown
warnings appeared only after artifact generation; the evaluator exited
successfully. The post-run collection fingerprint and all dataset/source/sidecar
hashes matched their pre-run values. Exact historical collection equivalence is
still not claimed, and the hosted reranker exposes no immutable service-side
digest.

The focused runtime, packet-flow, deduplication, CLI, batch, degradation, and
route-aware evaluator suite passed 188 tests. The full suite completed with 418
passing tests, 47 passing subtests, and the same two pre-existing failures
documented above. Neither failure touches PR5. Full commands, redacted environment,
hashes, provenance, and limitations are in
`artifacts/evals/agents/v1/degradation/baselines/29d2226/manifest.json`.

## PR5 Executed-Retrieval Trace Closure — 2026-09-03

This correction closes the final finding from the review of `88d7fcb`. All
earlier SHA-keyed PR5 evidence, including `29d2226`, remains unchanged.

### Run identity

| Field | Value |
|---|---|
| Status | Final reviewed PR5 correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `65296dc6253e3dd1c8285e7f1fcaf553c4a85180` |
| Degradation dataset SHA-256 / cases | `0c7fa622902bf00a89a93120e8a0ad8866e6195f54810adf68f75b51f2cc0ca6` / 13 |
| Live dataset SHA-256 / cases | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` / 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 green points; fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744` before and after |
| RAGAS | Disabled |

Compact retrieval output now distinguishes retrieval that never ran from an
executed retrieval that returned an empty object. The former exposes
`attempted=false` and remains `skipped`; the latter exposes `attempted=true` and
remains `failed`. The independent shadow evaluator consumes that public trace
fact while retaining its older attempts/runs/ok/target inference for historical
artifacts.

`pytest==9.0.2` remained environment-only, repository dependency files were
unchanged, and the tracked worktree matched the implementation SHA before both
measurements. An independent review of the correction found no actionable P0–P2
issue.

### Deterministic and live results

The unchanged 13-case deterministic command wrote to
`artifacts/evals/agents/v1/degradation/baselines/65296dc`. All six matrix rates
were 100%. Its summary SHA-256 is
`709d04f659981a76db054cd363c9310efb9efa3d1b200110bdce80bdaf088015` and its
per-case SHA-256 is
`4e661abdc6c1c78f0937c86d0123ef9c2b3a2858ab77ded929dc58c9b36cd84c`.
The per-case hash changed because the additive public `retrieval.attempted` trace
field is now serialized; the dataset and aggregate matrix results are unchanged.

The unchanged 15-case route-aware live command used the same redacted environment
and thresholds as the earlier PR5 runs, a fresh checkpointer, and output directory
`artifacts/evals/agents/v1/baselines/65296dc`. It ran exactly once. All 15 rows
were valid with zero evaluator errors. The deterministic score was `0.961874` and
two rows were critical, for a `0.133333` critical-failure rate; the strict
zero-critical-row gate therefore remained false.

Runtime/shadow lane status, effective status, failure stage, and exact degradation
summary consistency were all 100%. Seven successful structured executions yielded
seven typed, analyst-visible contexts with complete provenance-field coverage and
no synthetic structured-text contexts.

The two critical rows were unchanged from the strongest earlier PR5 runs:

- `AGENT_V1_HYBRID_001`: analyst tool-loop exhaustion and an invalid fallback
  output caused analyst-stage failure with no accepted citation.
- `AGENT_V1_ANALYST_001`: ambiguous matching across successful calculator calls
  produced `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted
  citation.

Live artifacts:

- `artifacts/evals/agents/v1/baselines/65296dc/summary.json`, SHA-256
  `10122a372f657be94f89754e7c50b6ed83846e7bb0f2134ab156cdf057079b51`.
- `artifacts/evals/agents/v1/baselines/65296dc/per_query.jsonl`, SHA-256
  `71b1389304269e099a3f5ad2e532efda4a4562c8acb6aa005c347653462e58ef`.
- Zero-byte `artifacts/evals/agents/v1/baselines/65296dc/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The live run took 2,187,035 ms (about 36 minutes). Known MCP/AnyIO shutdown
warnings appeared only after artifact generation; the evaluator exited
successfully. The post-run collection fingerprint and all dataset/source/sidecar
hashes matched their pre-run values. Exact historical collection equivalence is
still not claimed, and the hosted reranker exposes no immutable service-side
digest.

The focused runtime, compaction, packet-flow, deduplication, CLI, batch,
degradation, and route-aware evaluator suite passed 194 tests. The full suite
completed with 421 passing tests, 47 passing subtests, and the same two pre-existing
failures documented above. Neither failure touches PR5. Full commands, redacted
environment, hashes, provenance, and limitations are in
`artifacts/evals/agents/v1/degradation/baselines/65296dc/manifest.json`.

## PR5 Structured Outcome Completeness Closure — 2026-09-03

This correction closes the mixed-unusable structured-lane finding from the fresh
review of `5d7820f`, plus exact request/result coverage and malformed-result gaps
found during the independent pre-commit audit. All earlier SHA-keyed PR5 evidence,
including the original failing `25e359d` baseline, remains unchanged.

### Run identity

| Field | Value |
|---|---|
| Status | Reviewed PR5 correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `5762f51668016376746438c6dd5503e04b9c50c1` |
| Degradation dataset SHA-256 / cases | `a8c54aad42061b99bf114a60818cb817e5f15e76279dfa59b72464fdb5ed7012` / 14 |
| Live dataset SHA-256 / cases | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` / 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; pre-run 582 green points; post-run fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744`, matching the prior verified PR5 fingerprint |
| RAGAS | Disabled |

The structured lane now reserves unusable `partial` for a wholly partial outcome.
Any partial mixed with an error, ambiguity, unsupported outcome, missing result, or
malformed result fails closed when no evidence survives. Every whole-lane status
requires exact requested/result coverage. If valid admitted evidence survives an
incomplete or malformed result set, the lane remains usable but becomes `partial`.

The independent shadow evaluator derives the same result from raw structured
outputs and analyst-visible contexts/issues without importing the runtime lane
helper. Invalid result containers and entries are represented safely and produce
normalized admission issues, so malformed output cannot masquerade as skipped or
complete.

`pytest==9.0.2` remained environment-only, repository dependency files were
unchanged, and the tracked worktree matched the implementation SHA before both
measurements. Independent pre-commit review found no actionable P0–P2 issue after
the final corrections.

### Deterministic and live results

The expanded 14-case deterministic command wrote to
`artifacts/evals/agents/v1/degradation/baselines/5762f51`. All six matrix rates
were 100%. Its summary SHA-256 is
`df12eddaf88b8a9dc22a471537b0f91477072c3fa5c6e1961368f2842591fe11` and its
per-case SHA-256 is
`1fcce481c35acd2d6428abadc8bb3753fd2dff1987ff0bd5db8e0df97cb0a370`.

The new `PR5_STRUCTURED_MIXED_UNUSABLE` case executes two requested structured
facts that return `partial` and `error`, admits zero evidence, classifies both
runtime and shadow lanes as `failed` and unusable, fails overall at the structured
stage, and skips the analyst. The previously added KB and structured admission-loss
cases remain usable `partial`, report `degraded`, and run the analyst.

The unchanged 15-case route-aware live command used the same redacted environment
and thresholds as earlier PR5 runs, a fresh checkpointer, and output directory
`artifacts/evals/agents/v1/baselines/5762f51`. It ran exactly once. All 15 rows
were valid with zero evaluator errors. The deterministic score was `0.961874` and
two rows were critical, for a `0.133333` critical-failure rate; the strict
zero-critical-row gate therefore remained false.

Runtime/shadow lane status, effective status, failure stage, and exact degradation
summary consistency were all 100%. Seven successful structured executions yielded
seven typed, analyst-visible contexts with complete provenance-field coverage and
no synthetic structured-text contexts.

The two critical rows matched the strongest earlier PR5 runs:

- `AGENT_V1_HYBRID_001`: analyst tool-loop exhaustion and an invalid fallback
  output caused analyst-stage failure with no accepted citation.
- `AGENT_V1_ANALYST_001`: ambiguous matching across successful calculator calls
  produced `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted
  citation.

Live artifacts:

- `artifacts/evals/agents/v1/baselines/5762f51/summary.json`, SHA-256
  `9e3c335355ae4e47de623e155c9280e29d64a6676d2151e042c6f36b6d538481`.
- `artifacts/evals/agents/v1/baselines/5762f51/per_query.jsonl`, SHA-256
  `fa776d121db124dd6626b25c6ab951214dc8b7e756cf220e07de177ffae78572`.
- Zero-byte `artifacts/evals/agents/v1/baselines/5762f51/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The live run took 2,238,643 ms (about 37 minutes). Known MCP/AnyIO shutdown
warnings appeared only after artifact generation; the evaluator exited
successfully. Pre-run collection info reported the expected 582 green points. The
read-only evaluator performed no collection writes, and the post-run fingerprint
matched the prior independently verified PR5 fingerprint. Exact historical
collection equivalence is still not claimed, and the hosted reranker exposes no
immutable service-side digest.

The focused runtime, packet-flow, independent evaluator, and degradation-matrix
suite passed 177 tests. The full suite completed with 449 passing tests, 47 passing
subtests, and the same two pre-existing failures documented above. Neither failure
touches PR5. Full commands, redacted environment, hashes, provenance, and
limitations are in
`artifacts/evals/agents/v1/degradation/baselines/5762f51/manifest.json`.

## PR5 Analyst Packet Lane Consistency Closure — 2026-09-03

This correction closes the fresh-review finding on `95ce251`: the evaluator now
verifies the analyst packet's lane summaries as well as its degradation summary.
All earlier SHA-keyed evidence, including `25e359d` and `5762f51`, remains
unchanged.

### Run identity

| Field | Value |
|---|---|
| Status | Fresh-review correction; deterministic matrix passed; live regression gate failed and was preserved |
| Evaluated implementation SHA | `de0142ca89384aee9c1bad014879e7e83e94fc86` |
| Degradation dataset SHA-256 / cases | `a8c54aad42061b99bf114a60818cb817e5f15e76279dfa59b72464fdb5ed7012` / 14 |
| Live dataset SHA-256 / cases | `c0fbdd097d7fa1b3eb22953a3ba4e8c0e84aa70a8786e08ba78a3b305cabe6cd` / 15 |
| Python / pytest | `3.11.14` / `9.0.2` |
| Planner / analyst | `ollama/qwen2.5:14b-instruct`, digest `7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6` |
| Dense embedding | `qwen3-embedding:8b`, digest `64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b` |
| Qdrant | `1.16.2`; collection `sec_docs_dense_bm25_pr2_63dcec0`; 582 green points; fingerprint `0e60c99368eae07b4f00cb86a3d6e58d49b7e6bbd2c1df84e98204e8be75a744` before and after |
| RAGAS | Disabled |

When caller-visible runtime lanes are correct but the analyst packet carries a
stale `requested`, `attempted`, or `status` value, degradation consistency is now
false and the existing critical inconsistency gate fires. The comparison uses the
independent shadow lanes; it does not import or call the runtime lane helper.

`pytest==9.0.2` remained environment-only, repository dependency files were
unchanged, and the tracked worktree matched the implementation SHA before both
measurements.

### Deterministic and live results

The 14-case deterministic command wrote to
`artifacts/evals/agents/v1/degradation/baselines/de0142c`. All six matrix rates
were 100%. Its summary SHA-256 is
`df12eddaf88b8a9dc22a471537b0f91477072c3fa5c6e1961368f2842591fe11` and its
per-case SHA-256 is
`1fcce481c35acd2d6428abadc8bb3753fd2dff1987ff0bd5db8e0df97cb0a370`.

The unchanged 15-case route-aware live command used a fresh checkpointer and
output directory `artifacts/evals/agents/v1/baselines/de0142c`. It ran exactly
once. All 15 rows were valid with zero evaluator errors. The deterministic score
was `0.948540` and three rows were critical, for a `0.200000` critical-failure
rate; the strict zero-critical-row gate therefore remained false.

Runtime/shadow lane status, effective status, failure stage, degradation summary,
and analyst-packet lane consistency were all 100%. Seven successful structured
executions yielded seven typed, analyst-visible contexts with complete
provenance-field coverage and no synthetic structured-text contexts.

The critical rows were:

- `AGENT_V1_HYBRID_001`: analyst tool-loop exhaustion and an invalid fallback
  output caused analyst-stage failure with no accepted citation.
- `AGENT_V1_ANALYST_001`: ambiguous matching across successful calculator calls
  produced `CALCULATION_RESULT_AMBIGUOUS`, analyst `tool_error`, and no accepted
  citation.
- `AGENT_V1_ANALYST_002`: the analyst correctly returned `insufficient_data`
  because `market_price_per_share` was unavailable, while the frozen expectation
  required completion; this produced status and failure-stage mismatches.

Live artifacts:

- `artifacts/evals/agents/v1/baselines/de0142c/summary.json`, SHA-256
  `88f6ba76feb0827256107b379cdd9edae77768b70ecff1c5deb4f4d47f75eb59`.
- `artifacts/evals/agents/v1/baselines/de0142c/per_query.jsonl`, SHA-256
  `7c7d0864ec3968b700ae0632db39e1e39db7b5d33b09208122cebb7299a0a885`.
- Zero-byte `artifacts/evals/agents/v1/baselines/de0142c/errors.jsonl`, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The live run took 2,229,159 ms (about 37 minutes). Known MCP/AnyIO shutdown
warnings appeared only after artifact generation; the evaluator exited
successfully. The collection fingerprint and all dataset/source/sidecar hashes
matched before and after. Exact historical collection equivalence is not claimed,
and the hosted reranker exposes no immutable service-side digest.

The focused runtime, packet-flow, independent evaluator, and degradation-matrix
suite passed 178 tests. The full suite completed with 450 passing tests, 47 passing
subtests, and the same two pre-existing failures documented above. Neither failure
touches PR5. Full commands, redacted environment, hashes, provenance, and
limitations are in
`artifacts/evals/agents/v1/degradation/baselines/de0142c/manifest.json`.
