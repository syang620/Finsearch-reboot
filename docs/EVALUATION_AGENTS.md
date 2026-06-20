# Agent Evaluation v0

This evaluator measures end-to-end multi-agent performance:

- Planner (intent/routing/metadata)
- Retrieval handoff quality
- Analyst behavior (tool use, computation, citations)
- Optional RAGAS LLM-judge metrics

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
- Use RAGAS runs for semantic quality checks and release decisions.
- Keep `gold_answer` non-empty in dataset rows; RAGAS quality degrades otherwise.
