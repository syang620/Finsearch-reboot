# PR8 release exception: PR8-GROUNDING-001

Owner approval: 2026-09-06. Applies only to PR26 / roadmap PR8.
Evaluated runtime: `99cf1b20e2796741df735365e78e241ea6667c2d`.
Final evidence/documentation review target:
`4a55cff7533a68ccd866a0f65954b4eb091c7b51`.

## Explicit acceptance and scope

The owner instructed recording a new explicit exception for
`AGENT_V1_ANALYST_001`, final Codex review of the current evidence/documentation
head if not already inspected, merge of PR26 and refresh of `master`.

**The strict live gate failed.** Its result remains 14/15 non-critical rows,
one critical row, score 0.981481 and zero evaluator errors. No gate output,
threshold or historical blocked disposition is rewritten.

PR8-GROUNDING-001 accepts only the unrelated analyst grounding failure in
`AGENT_V1_ANALYST_001`: `ANALYST_GROUNDING_INVALID` with
`GROUNDING_ROW_TEXT_MISMATCH`, persisting through all three grounding decisions.
The analyst correctly fails closed; calculator tool use did not yield an
accepted final answer. This exception is risk acceptance for this one failed
case, not proof of evidence truth, a passing calculation or a green strict gate.

It does not waive any filing/period selection, SEC provenance, resolver/routing,
tool-argument, admission/lane, timeout or other defect. PR8's deterministic
49-case correctness matrix and 170-case resolver parity remain as measured,
alongside calculator/output repair 20/20 each, grounding 37/37 and degradation
14/14. The two documented full-suite baseline failures remain disclosed, not
newly covered by this exception. No PR7/PR24 exception is inherited; this new
exception does not carry forward to PR9 or future releases.

## Evidence preservation and residual risk

The original `REPORT.md`, control files, live manifest, raw summary and per-query
traces are unchanged. Their contemporaneous blocked/no-exception statements are
historical and superseded only by this approval. Model-output grounding remains
a known residual reliability risk, with no change to PR6 validation, factual
prose, prompts, providers, budgets, timeouts or gate thresholds.

The controlled run recorded no terminal analyst timeouts and all 188 power and
browser samples met the controls. Forty transient heavy-workload samples remain
disclosed; sampled controls do not establish exclusive hardware. RAGAS was off,
so no additional semantic-quality pass is claimed. No evaluation rerun is
requested or performed: only evidence/release documentation changed after the
evaluated implementation.

## Final review and merge conditions

The latest completed pre-closure review inspected `99cf1b2`, not `4a55cff`.
One final review of exact `4a55cff` was requested:
https://github.com/syang620/Finsearch-reboot/pull/26#issuecomment-5559552864

Final review completed on 2026-09-06 at 13:35:35 UTC for exact `4a55cff`:
https://github.com/syang620/Finsearch-reboot/pull/26#issuecomment-5552377529

Its sole new finding was P1 "Record the authorized PR8 exception before merging":
https://github.com/syang620/Finsearch-reboot/pull/26#discussion_r3944094924

This superseding exception record and the updated evaluation/roadmap status
address that finding. No runtime correction was requested. Earlier renderer and
artifact-reference findings were already resolved. The closure commit changes
only these three documents, so the reviewed evidence and frozen runtime remain
identical; no evaluation rerun is performed. The review thread will be resolved
with the closure commit before the authorized merge.

No blanket exception for other review findings is granted. Runtime changes would
require renewed evaluation/freeze, not this documentation-only closure.

## Closure checks

Files changed: this new disposition, `docs/EVALUATION_BASELINES.md`, and
`docs/IMPLEMENTATION_PLAN.md`. Original evidence files are unchanged.

```text
git diff --check
git diff 99cf1b2 -- src scripts tests data/evals/agents/v1
git diff 4a55cff -- artifacts/evals/agents/v1/baselines/99cf1b2
git diff --cached --stat
git push origin codex/pr8-filing-period-semantics
```

The first check validates patch whitespace; both scoped parity diffs must remain
empty. The final merge uses an exact expected head and does not bypass repository
protections. Local master will be fast-forwarded to the merged remote master.
