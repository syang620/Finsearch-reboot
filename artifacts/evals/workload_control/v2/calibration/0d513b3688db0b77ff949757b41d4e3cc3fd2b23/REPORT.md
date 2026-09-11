# Workload-control v2 final hard-control-gated calibration report

Decision: **`WORKLOAD_CONTROL_V2_VALIDATED`**

Selected policy: **`B_CONSECUTIVE_10`**

This comparison supersedes `baa2dd2` for release use after exact-head review found
that the frozen Ollama service version was not included in the S3 identity gate.
The final analyzer requires it in addition to the already gated model digests,
Qdrant identity, SEC health, index snapshots, tracked state, complete steps, and
absence of errors. The registered candidates, ordering, thresholds, scenarios,
and acceptance criteria did not change. The complete `baa2dd2` raw inputs already
contained the required Ollama evidence and were re-evaluated without recapture.
No semantic case, retrieval, reranker, or model inference ran.

## Final results

- Clean scenarios: 0 B10 CPU/hard episodes in 900 supervised-idle, 900
  terminal-only, and 180 required-service samples.
- Awake protection: active in every one of 2,370 samples and alive immediately
  before cleanup in all six scenarios.
- S3 frozen identity: all 9 checks pass, including exact Ollama version.
- Sustained interference: 3/3 detected at 9.9999, 10.0000, and 10.0049 seconds.
- Six short bursts: 0 B10 episodes.
- Real Chrome: detected by the hard rule in the first sampled interval.
- Terminal-only: zero active ChatGPT/Codex or browser records; four inert,
  launchd-owned crash handlers retained at 0.0% CPU across 900/900 samples.
- Process evidence: 100% classified or retained as unknown.

The browser-contaminated supervised batch and two invalid terminal attempts remain
preserved under the raw-capture SHA and are excluded. Historical PR32 replay is a
lower bound, and no historical status changes.

## Claim limits

The evidence supports only that B10 met this preregistered control-only calibration
on this machine/session. It does not establish multi-hour false-invalidation
rates, timeout causality, semantic quality, or benchmark validity. PR31 remains
frozen, draft, blocked, and unmerged. This PR remains draft and must not merge
automatically.

## Verification

Focused diagnostics pass **34/34**. The full suite records **1,212 passed**, **47
subtests passed**, the same **2 known pre-existing failures**, and **25 warnings**.
All final artifact hashes, JSON, privacy, and whitespace checks pass.
