# FinSearch Evaluation

This document is the entry point for how FinSearch quality is measured and which
checks can block a release. Detailed evaluator usage stays in the component guides,
and measured results stay in [Evaluation Baselines](EVALUATION_BASELINES.md).

## Evaluation Layers

| Layer | What it measures | Current role | Details |
|---|---|---|---|
| Deterministic tests | Contracts, routing, execution, tools, and failure behavior | Required for affected changes | Repository test suite |
| Planner routing | Route selection and structured-fact request shape | Release gate for planner and routing changes; every selected P0 case must pass | `scripts/evals/agents/eval_planner_routes.py` |
| Retrieval | Filing evidence ranking and recall | Deterministic retrieval metrics gate retrieval changes; RAGAS metrics are supplemental | [Retrieval Evaluation](EVALUATION_RETRIEVAL.md) |
| End-to-end agents v0 | Planner, legacy KB-oriented filing flow, and analyst behavior | Diagnostic only | [Agent Evaluation v0 (Legacy)](EVALUATION_AGENTS.md) |
| Route-aware multi-lane E2E | `kb`, `structured_fact`, and `hybrid` behavior | Planned for PR 2; not yet a release gate | [Implementation Plan](IMPLEMENTATION_PLAN.md) |

## Release-Gate Policy

- Run deterministic tests for every affected component.
- Planner or route-contract changes must pass the selected P0 planner-routing suite.
- Retrieval changes must pass the deterministic criteria described in the retrieval
  evaluation guide. RAGAS results provide supporting evidence but are not a gate.
- Do not use the current end-to-end v0 overall score as a release gate. Its route
  logic is incompatible with valid `structured_fact` behavior, and its semantic
  context extraction excludes structured-fact evidence.
- Route-aware E2E evaluation becomes the multi-lane release gate only after the PR 2
  evaluator and acceptance criteria are implemented.

## Results and Artifacts

Evaluation output belongs under `artifacts/evals/`. Commit-specific measured results,
including the model, dataset version, command, artifact path, and limitations, belong
in [Evaluation Baselines](EVALUATION_BASELINES.md). Do not copy mutable pass counts or
scores into the canonical architecture document.

## Component Guides

- [Planner and agent E2E evaluation](EVALUATION_AGENTS.md)
- [Retrieval evaluation](EVALUATION_RETRIEVAL.md)
- [Recorded evaluation baselines](EVALUATION_BASELINES.md)
