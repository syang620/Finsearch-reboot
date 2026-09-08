# ADR: Semantic Benchmark v2 workload-control contract

Status: **validated control candidate; frozen pending independent review**

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED` — `B_CONSECUTIVE_10`**

## Boundary

This control-design change is stacked on reviewed PR32 evidence. PR31 remains at
`7667af52169fb12965adf5680bae58a74c7c977e`, draft, release-blocked, and
unmerged. No semantic case ran. No benchmark membership/gold, scorer, production
runtime, planner, retrieval, analyst, model/provider, timeout, retry, or historical
artifact changed.

The preregistration was committed at `974c381e9147a444eaf4c574f01079d240cb67d6`
before calibration. Its SHA-256 is
`0c6c4ac93cee1b23e33ea2ea93ded0a9b56092cd9898045b44a0bb7595463483`.
The registered candidate set, parameters, scenario order, and acceptance criteria
were not changed after observing results.

Before artifact freeze, the publication scan found a personal email address in a
transient process command line. The evidence publisher now replaces email
addresses with `$EMAIL`, in addition to the existing `$USER_HOME` substitution.
The affected raw strings were deterministically sanitized and the comparison was
regenerated at `7d68a8c96edf2d5f2f2a4996b88e384946973670`. This changed file hashes but no
process identity, classification, CPU value, policy result, or selection. The
pre-sanitization comparison is retained and explicitly excluded from release use.

## Old and new rules

The v1 reference invalidates on one sample containing a real-or-substring-matched
browser or any non-exempt process at 50% CPU. PR32 showed 235/1,200 idle/read-only
samples failing that rule without a ten-sample sustained process.

V2 retains hard AC, Low Power Mode, awake, actual-browser, and model/index/service
identity checks. Its CPU rule continuously samples once per second and invalidates
only when the same CPU-scored stable process group reaches 50% aggregate CPU for
10 consecutive samples. A below-threshold or absent sample resets that group's
streak. Unknowns remain scored. Directly evidenced model/service/monitor processes
are retained but excluded from external-interference scoring; application-wide
Docker, ChatGPT, or Codex exemptions do not exist.

The real-browser hard rule uses the actual Chrome/Safari browser executable. During
calibration, the v1 substring rule counted Safari's `CacheDeleteExtension`; v2
correctly classifies that extension as OS background activity while continuing to
hard-fail the actual Chrome/Safari binary.

## Preregistered calibration result

Values are CPU invalidation episodes / violating samples. Browser hard-rule samples
are reported separately in the comparison artifact.

| Candidate | Supervised idle | Terminal idle | Service activity | Sustained load | Short bursts | Browser load | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A instantaneous v1 | 156/199 | 6/6 | 14/20 | 12/145 | 20/21 | 7/35 | reference only |
| B consecutive 3 | 12/32 | 0/0 | 1/1 | 4/128 | 1/2 | 1/27 | no |
| B consecutive 5 | 5/13 | 0/0 | 0/0 | 3/120 | 0/0 | 1/25 | no |
| **B consecutive 10** | **0/0** | **0/0** | **0/0** | **3/105** | **0/0** | **1/20** | **yes** |
| C occupancy 5/30 | 3/885 | 0/0 | 1/171 | 1/204 | 1/111 | 1/54 | no |
| C occupancy 12/60 | 3/845 | 0/0 | 1/157 | 1/192 | 1/97 | 1/44 | no |
| D burden 25/30 | 1/894 | 2/62 | 1/175 | 1/204 | 1/115 | 1/54 | no |
| D burden 20/60 | 1/890 | 2/132 | 1/171 | 1/201 | 1/112 | 1/51 | no |

B10 passed every registered clean scenario with zero CPU or hard-rule episodes.
It detected all three 45-second one-core workloads at approximately 10 seconds,
inside the 15-second bound. The controlled processes had median CPU of 99.6–99.7%.
It ignored all six 1.5-second bursts and detected the real Chrome workload in the
first browser sample. Process records were classified or retained as unknown at
100%.

B3 failed supervised idle, service activity, and short-burst acceptance. B5 failed
supervised idle. The occupancy and burden candidates frequently stayed active
during clean or transient conditions and could not receive sustained-detection
credit when already invalid before the controlled workload began.

The selected rule has a narrow observed margin: supervised idle reached nine
consecutive samples for one scored group. The registered rule is still selected
because the clean requirement was zero invalidation episodes and B10 detected all
sustained workloads within the fixed latency bound. This limitation must accompany
any future claim.

## Process identity and terminal viability

Classification uses executable path, command, PID/PPID, process group, start time,
cwd when available, and ancestry. Docker's directly evidenced backend/Qdrant host
is distinct from Docker Desktop UI/renderers. ChatGPT/Codex UI remains scored.
Unknown processes remain visible and scored.

Terminal-only execution is **viable with limitations**. A notification-gated
Terminal launch produced 900/900 one-second samples with valid AC/LPM state and
zero ChatGPT/Codex or actual-browser records. Two prior procedural captures are
preserved: one retained ChatGPT throughout, one was reopened at sample 891. A
wrapper attempt produced no samples because it waited on inert crash-report
handlers. Only the final capture satisfies the registered scenario.

This proves detached control monitoring and autonomous evidence writing. It does
not prove terminal-only semantic execution because semantic cases were forbidden,
and the read-only preflight was calibrated separately under supervision. Therefore
the official run should prefer terminal-only execution, but integration still
requires review.

## Historical replay

Selection occurred before replay. Applying B10 offline to PR32's 1,200 samples
produces zero v2 CPU episodes, compared with 188 v1 episodes / 235 v1 samples.
This does **not** relabel PR32 or any prior run. PR32 retains threshold-crossing
identity needed for consecutive and occupancy replay; burden replay is only a lower
bound because sub-threshold CPU was not retained.

The two earlier invalid diagnostics are indeterminate under v2 because their sparse
records lack consecutive cadence and executable/ancestry identity. Their
`invalid_diagnostic` status is permanent.

## Exact PR31 continuation

1. Keep PR31 frozen and do not run the 60-case baseline while this control PR is
   unreviewed or unmerged.
2. Obtain an independent clean review of the preregistration, raw evidence,
   classifier, comparison, and this contract. Do not merge automatically.
3. After an explicit merge decision, create a separate evaluation-infrastructure
   integration that makes PR31 opt into v2; do not rewrite its historical v1
   launcher or artifacts.
4. The integration must continuously sample at one second from settled preflight
   through artifact closure, use the reviewed classifier/B10 rule, preserve hard
   requirements, and record the contract and implementation hashes.
5. Freeze and independently review that integration. Do not change semantic
   runtime, data, scorer, models, timeouts, or retries.
6. Launch one notification-gated terminal-only 60-case attempt. No ChatGPT/Codex
   supervision is required during cases. Any hard failure or B10 streak invalidates
   the attempt and preserves its raw evidence.
7. Only a valid complete 60-case capture may proceed to the predetermined 30-case
   source audit. Historical diagnostics remain invalid forever.

## Claim limits

The evidence supports only that B10 met this preregistered control-only calibration
on this machine/session. It does not prove multi-hour false-invalidation rates,
semantic quality, timeout causality, or benchmark validity. It does not authorize
a baseline until the reviewed v2 integration explicitly opts in.
