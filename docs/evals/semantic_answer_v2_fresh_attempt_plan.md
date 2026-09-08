# One explicitly authorized fresh semantic-v2 attempt

Status: authorization registered; no new attempt started. The user approved
one fresh controlled pass, followed by the frozen 30-case source audit, evidence
verification, fresh review, conditional PR31 merge, and a short cross-benchmark
audit before declaring optimization readiness. This supersedes the earlier
no-rerun/no-merge instruction only within those conditions.

## Preserve the benchmark, change only operational bookkeeping

The benchmark/scorer contract, original launcher, separately reviewed v2.1
launcher, judge calibration/disabled decision and first invalid diagnostic stay
byte-unchanged. The authorization JSON binds their hashes. No new launcher code,
dependency, model call, threshold or runtime behavior is introduced by this plan.

Use the existing v2.1 launcher's supported output-root option with the single
new root registered in `semantic_answer_v2_fresh_attempt_20260908.json`. The
original diagnostic remains exactly as `invalid_diagnostic`, not “baseline 1.”
The launcher still refuses any subsequent run once a `started.json` exists in
the newly registered root, even under another implementation SHA. This is one
explicitly authorized attempt, not a general bypass of its single-pass guard.

To reduce live UI/log activity, redirect console output to an ignored local log
and stage runner files under the registered ignored cache root. Monitor compact
case counts, process status and recorded workload violations; do not display
large answers or run tests/audits concurrently. The UI process remains subject
to the same 50% CPU threshold. This isolation is an operational precaution, not
proof of what caused the earlier CPU burst or either analyst timeout.

After completion, preserve staging and exclusively create a SHA-keyed published
directory containing byte-identical copies of every generated file. Check full
file-set equality and hashes before committing. No edited manifests, filtered
cases, suppressed failures or relabeled outputs are permitted. Failed/incomplete
capture is published as diagnostic-only, regardless of directory naming.

## Ordered gates

1. Fresh review of this execution plan; record that exact-head review separately,
   commit the approval, and verify a clean launch SHA. Verify all frozen hashes,
   preserved diagnostic hashes, unused staging root and existing model/index
   provenance. No re-calibration or benchmark editing.
2. One sequential 60-case run on AC, Low Power Mode off, awake, no browsers/heavy
   workloads and healthy services. Keep the original 120-second analyst timeout,
   runtime retries, model settings, schedule and cache policy. Preserve poor
   answers honestly. If operational validity fails, stop and preserve diagnostic
   evidence; another attempt requires new direction.
3. Only after a valid full capture, source-audit all 30 preregistered IDs. Freeze
   the audit against raw-output hashes. For eligible answers assess each emitted
   claim against its cited visible evidence, and each required facet against the
   final answer and source gold. Record source references, exact quotes and
   reasoning. Reject evidence-only rescue of omitted answer facts. Record
   ineligible answers as execution losses with no semantic credit or replacement
   cases. This is assistant source adjudication, not independent human labeling.
4. Use the unchanged semantic-summary validators and fixed subset membership.
   Publish all-60 execution/deterministic metrics separately from 30-case-scope
   semantics, with eligible/assessed/unknown/fixed denominators and family
   concentration. Do not extrapolate subset accuracy to 60. The full-corpus judge
   stays disabled. Verify raw hashes, exact scorer reproduction, evidence,
   historical immutability, privacy and relevant tests.
5. Obtain clean fresh final review, then merge PR31 as now conditionally authorized
   and refresh master safely. Any review blocker prevents merge; this permission
   does not authorize changing frozen labels/scoring to resolve it.
6. Briefly audit retrieval v3 and semantic v2 together: canonical hashes and
   source/label independence, known-label/equivalence limits, correlated coverage,
   metric denominators, judge-disabled policy, baseline validity and permissible
   future paired claims. Only then publish joint optimization readiness; preserve
   the existing dataset freezes rather than rewriting them. Do not optimize yet.

The final cross-benchmark audit is a release-readiness declaration, not permission
to mutate either benchmark or a claim of independent external certification.
