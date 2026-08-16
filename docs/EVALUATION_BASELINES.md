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
