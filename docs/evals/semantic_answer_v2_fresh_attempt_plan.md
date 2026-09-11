# One explicitly authorized fresh semantic-v2 attempt

Current status: the single permission was consumed and the attempt stopped on a
recorded workload violation. See the [separate preserved disposition](semantic_answer_v2_fresh_attempt_status.md).
No new attempt or merge is authorized by this exhausted permission.

The following is the registered pre-run plan. At registration no attempt had
started. The user approved
one fresh controlled pass, followed by the frozen 30-case source audit, evidence
verification, fresh review, conditional PR31 merge, and a short cross-benchmark
audit before declaring optimization readiness. This supersedes the earlier
no-rerun/no-merge instruction only within those conditions.

## Preserve the benchmark, change only operational bookkeeping

The benchmark/scorer contract, original launcher, separately reviewed v2.1
launcher, judge calibration/disabled decision and first invalid diagnostic stay
byte-unchanged. The authorization JSON binds their hashes. An external operation
wrapper records permission consumption; it does not implement or patch the
launcher, scoring or runtime. No dependency, model/configuration or threshold change.

Use the existing v2.1 launcher's supported output-root option with the single
new root registered in `semantic_answer_v2_fresh_attempt_20260908.json`. The
original diagnostic remains exactly as `invalid_diagnostic`, not “baseline 1.”
The original launcher writes `started.json` only after fallible preflight, so that
file alone cannot enforce this authorization. Invoke only the registered operation
wrapper: after local committed-registration checks it exclusively creates the
fixed consumption marker, flushes/fsyncs its bytes and parent directory, and only
then imports project code, verifies remote review or launches the frozen child.
Even a review outage, import failure or zero-case preflight refusal consumes the
permission. A second invocation fails without changing marker/outcome evidence,
irrespective of a new implementation SHA. Never delete/reset the marker or invoke
the underlying launcher directly for this permission. The original root-local
`started.json` guard remains an additional safeguard. This is one explicitly
authorized attempt, not permission for preflight retries or arbitrary new roots.

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
Preserve and publish consumption/outcome records separately with their own hashes,
including when preflight produces no runner directory. Console logs remain local;
any public preflight failure explanation must avoid credentials and private paths.

## Ordered gates

1. Fresh review of this execution plan and external wrapper; record that exact-head
   review plus authorization/wrapper hashes separately, commit the approval, and
   verify a clean launch SHA. Verify all frozen hashes,
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

## Pre-launch review correction and verification

Review of `7a3a74a` identified that `started.json` alone left preflight failures
outside the one-attempt boundary. The external wrapper fixes only that bookkeeping:
exclusive durable consumption precedes fallible operational work, failures retain
an outcome record, interruption is forwarded to the exact child, and the original
launcher is invoked unchanged. Registration, duplicate invocation, review failure,
child launch/exit and interruption paths have 12 new regression tests.

Focused consumption/launcher/control/semantic-summary checks: 62 passed. Full
suite: 1,173 passed, 47 subtests passed, the same two pre-existing planner-route
and retrieval-attempt failures, and 25 warnings. Original `src`, `scripts/evals`
and dataset files remain unchanged; privacy/hash/whitespace checks pass. The
consumption marker and new staging root do not yet exist. Fresh review of this
corrected execution contract is required before allocating the one invocation.

The follow-up review on `72dc799` found that an abbreviated reviewed SHA could
miss full-SHA GitHub findings. Registration now requires exactly 40 lowercase
hexadecimal characters before ancestry or remote checks. Five additional cases
reject abbreviated, uppercase, non-hex, missing and non-string identities; all
17 operation tests pass. The full-suite counts above describe the preceding
wrapper revision. No frozen benchmark/launcher file or run permission changed.

The corrected wrapper received clean exact-head review at `26993ea6f6`, recorded
in the separate hash-bound approval JSON. Full verification on that revision:
1,178 passed, 47 subtests passed, the same two pre-existing failures and 25
warnings. The approval commit is bookkeeping only; no baseline has started at
its creation. Capture validity and the later final review remain separate gates.
