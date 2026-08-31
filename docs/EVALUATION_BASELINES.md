# FinSearch Evaluation Baselines

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
| Final review-fix evaluated commit | `f301f32742b85faba079031c4d3381ec15a2e51f` |
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
verification at `f301f32742b85faba079031c4d3381ec15a2e51f`, are measured separately by
their post-review artifacts. Changes after the final SHA are limited to evaluation
artifacts and documentation; any later source, test, prompt, policy, evaluator, or
dataset change invalidates the final post-review measurement and requires a rerun.

The adversarial baseline measures routing permission and normalization, not SEC fact
numeric correctness, filing-specific metric resolution, retrieval relevance,
analyst synthesis, citations, latency, load, or production distribution shift. Its
40 cases intentionally emphasize known boundary classes and hostile proposals; it
is not a frequency-weighted sample of user traffic.
