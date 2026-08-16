# Agent Evaluation v0 (Legacy)

This evaluator measures end-to-end multi-agent performance:

- Planner (intent/routing/metadata)
- Retrieval handoff quality
- Analyst behavior (tool use, computation, citations)
- Optional RAGAS LLM-judge metrics

> **Legacy evaluator warning:** This evaluator was designed around the legacy
> KB-oriented filing path and is not route-aware.
>
> - **Route logic:** It classifies `filing_fact` or `filing_calc` with
>   `retrieval_needed=False` as `RETRIEVAL_ROUTING_INVALID`, so valid
>   `structured_fact` behavior fails by definition.
> - **Semantic context:** It derives RAGAS contexts only from KB retrieval output and
>   excludes structured-fact evidence.
>
> Consequently, its overall score is diagnostic only and is not a valid multi-lane
> release gate until the route-aware evaluator in PR 2 lands.

## Inputs

Default eval set:

- `data/evals/agents/v0/agent_eval_v0.jsonl`

Expected row shape (simplified):

```json
{
  "id": "AGENT_E2E_001",
  "user_query": "What was Apple's total debt ... 2024?",
  "gold_answer": "Apple's total debt ...",
  "expected": {
    "intent": "filing_calc",
    "retrieval_needed": true,
    "ticker": "AAPL",
    "fiscal_year": 2024,
    "form_type": "10-K",
    "must_use_financial_evaluator": true,
    "expected_metric": "total debt",
    "expected_min_citations": 1
  }
}
```

## Run

Deterministic + RAGAS:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --eval-path data/evals/agents/v0/agent_eval_v0.jsonl \
  --out-dir artifacts/evals/agents/v0 \
  --analyst-model qwen3:14b
```

Deterministic only (skip RAGAS):

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --disable-ragas
```

Enable context RAGAS metrics explicitly:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --enable-context-precision \
  --enable-context-recall
```

## Outputs

- `artifacts/evals/agents/v0/per_query.jsonl`
- `artifacts/evals/agents/v0/summary.json`
- `artifacts/evals/agents/v0/errors.jsonl`

## Deterministic checks

Per query:

- contract validity
- intent match
- retrieval_needed match
- metadata match (ticker/year/form where specified)
- financial_evaluator tool-use match
- compute metric match
- citation minimum match

A weighted deterministic score is produced per query and aggregated.

## RAGAS checks

When enabled, the evaluator computes:

- `answer_relevancy`
- `faithfulness`
- optional: `context_precision`, `context_recall`

RAGAS errors are captured in `errors.jsonl` and reflected in gate decisions.

## Environment

Agent runtime dependencies:

- Qdrant + indexed corpus for retrieval path
- Ollama models for planner/analyst

RAGAS/Ollama judge:

- `RAGAS_OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `RAGAS_JUDGE_MODEL` (default `llama3.1:8b-instruct`)
- `RAGAS_EMBED_MODEL` (default `nomic-embed-text`)

## Notes

- Use deterministic metrics for fast iteration.
- Use RAGAS runs for diagnostic semantic quality checks only; do not use v0 scores
  for release decisions until PR 2 provides route-aware evaluation.
- Keep `gold_answer` non-empty in dataset rows; RAGAS quality degrades otherwise.
