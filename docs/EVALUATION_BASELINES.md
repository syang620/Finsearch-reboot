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
