# Workload-control v2 hard-control-gated calibration report

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED`**

Selected policy: **`B_CONSECUTIVE_10`**

This separately keyed comparison supersedes the release use of the `7f90a5e`
comparison after exact-head review found that awake protection and the S3 frozen
service identity were recorded as requirements but did not gate selection. The
analyzer at `baa2dd2` now rejects every candidate unless both controls are proved.
The registered candidates, ordering, thresholds, scenarios, and acceptance rules
did not change. No semantic case, retrieval, reranker, or model inference ran.

## Results

- Supervised idle: 0 B10 episodes / 900 samples; maximum streak 4.
- Terminal-only idle: 0 episodes / 900 samples; maximum streak 6.
- Required service activity: 0 episodes / 180 samples; maximum streak 1.
- Sustained interference: 3/3 detected at 9.9999, 10.0000, and 10.0049 seconds.
- Six 1.5-second bursts: 0 B10 episodes; maximum streak 6.
- Real Chrome: the independent hard rule detected it in the first sampled interval.
- Awake protection: active in every sample of all six scenarios and still active
  immediately before controlled cleanup.
- S3 identity: all required steps succeeded; model digests, Qdrant identity,
  current index, historical index, SEC health, and tracked repository state matched
  the frozen semantic provenance.
- Process evidence: 100% classified or retained as unknown.

The valid terminal capture retained every known supervision executable. It found
zero active ChatGPT/Codex UI, renderer, or service records and zero browser
processes. Four launchd-owned `browser_crashpad_handler` processes were retained at
0.0% CPU in all 900 samples. This supports terminal-only control monitoring with
that explicit inert-handler limitation; semantic execution autonomy remains
untested.

## Invalid attempts preserved

One supervised batch was invalidated after user Chrome reopened during S1; all
five affected files remain under `invalid_protocol_attempts`. Two S2 attempts are
also preserved: one retained active ChatGPT/Codex and browser workloads, and one
started sampling 20 seconds before ChatGPT fully quit. None contributes to the
selected comparison.

## Review correction and failure path

Every candidate now includes `awake_protection` and `required_service_identity`
acceptance gates. Awake evidence requires a matching monitor-owned PID to be active
in every sample and alive immediately before cleanup. S3 evidence requires the
complete registered step set, no errors, a clean tracked tree, exact frozen model
digests, exact Qdrant identity, healthy SEC service, and exact current/historical
index snapshots. Unit regressions prove either gate blocks an otherwise passing
candidate. Historical PR32 replay remains a lower bound and historical statuses
remain unchanged.

## Limitations and release status

The clean calibration covers 33 minutes, not a multi-hour semantic run. It shows
that B10 met the preregistered control-only scenarios on this machine/session; it
does not establish timeout causality, semantic quality, benchmark validity, or a
statistical false-invalidation rate. PR31 remains frozen, draft, blocked, and
unmerged. This PR is draft and must not merge automatically.

## Verification

Focused diagnostics passed **33/33**. The full repository suite is recorded in the
final provenance. The comparison and all raw evidence are immutable SHA-keyed
artifacts.
