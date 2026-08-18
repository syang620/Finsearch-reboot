# FinSearch Reboot

Route-aware research system for SEC filings (10-K / 10-Q) using KB retrieval,
deterministic structured facts, and a grounded analyst.

## Overview

Runtime flow:

1. Planner extracts metadata and selects `kb`, `structured_fact`, or `hybrid`.
2. Orchestrator handles clarification/resume and executes the selected evidence lanes.
3. KB retrieval searches and reranks filing context; structured facts resolve supported SEC metrics.
4. Hybrid runs KB retrieval first and structured-fact execution second.
5. Orchestrator builds a normalized evidence packet for the grounded analyst.

Core runtime entrypoint:

- `agents.orchestrator.run_multi_agent_orchestration`

## Quickstart (Runtime)

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Set Python path

```bash
export PYTHONPATH=src
```

### 3. Start MCP server

```bash
PYTHONPATH=src python -m mcp_server.server
```

### 4. Run orchestration

```bash
PYTHONPATH=src python - <<'PY'
import asyncio
from agents.orchestrator import run_multi_agent_orchestration

async def main():
    result = await run_multi_agent_orchestration(
        "What was Apple's total debt (short-term plus long-term) at year-end 2024?",
        debug=True,
    )
    print(result)

asyncio.run(main())
PY
```

## Quickstart (Ingestion)

Use the unified ingestion CLI:

```bash
PYTHONPATH=src python scripts/ingestion_cli.py --help
PYTHONPATH=src python scripts/ingestion_cli.py run-all --help
```

See `scripts/README.md` for ingestion details.

## Retrieval Evaluation (v0)

Run retrieval-only evaluation (deterministic metrics + optional RAGAS context metrics):

```bash
PYTHONPATH=src python scripts/evals/retrieval/eval_retrieval_v0.py \
  --eval-mode table \
  --eval-path data/evals/retrieval/table/table_eval_v1.jsonl \
  --top-k 10 \
  --k-values 1,3,5,10 \
  --out-dir artifacts/evals/retrieval/v0
```

Detailed guide: `docs/EVALUATION_RETRIEVAL.md`

## Agent Evaluation (route-aware v1)

Run the current multi-lane end-to-end evaluation:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v1.py \
  --eval-path data/evals/agents/v1/agent_eval_routing_v1.jsonl \
  --out-dir artifacts/evals/agents/v1 \
  --analyst-model ollama/qwen2.5:14b-instruct
```

The v1 harness validates `kb`, `structured_fact`, and `hybrid` routes, lane outcomes,
reported versus evaluation-derived status, analyst behavior, citations, and stage
latency. RAGAS diagnostics are opt-in with `--enable-ragas`.

## Agent Evaluation (legacy v0)

Run the existing end-to-end agent evaluation:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --eval-path data/evals/agents/v0/agent_eval_v0.jsonl \
  --out-dir artifacts/evals/agents/v0 \
  --analyst-model qwen3:14b
```

Detailed guide: `docs/EVALUATION_AGENTS.md`

This v0 harness covers the legacy planner -> retrieval -> analyst path and remains
available only for historical reproducibility. It intentionally retains its
KB-oriented routing assumptions.

## Repository Map

- `src/agents` - planner/retrieval/analyst/orchestrator
- `src/retrieval` - retrieval pipeline and query expansion
- `src/analysis` - answer synthesis helpers
- `src/ingestion` - download/chunk/summarize/embed/ingest pipeline
- `src/mcp_server` - MCP server and tools
- `scripts` - operator scripts (ingestion-focused)
- `notebooks` - development/testing notebooks

## Configuration

Common environment variables:

- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `QDRANT_COLLECTION_NAME` (default `sec_docs_dense_bm25`)
- `sec_docs_dense_bm25` is the default retrieval and ingestion target in this repo.
- `TABLES_DIR` (default resolves to `data/chunked`)
- `SEC_API_KEY` (required only for `scripts/download_xbrl.py`)
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

Optional LLM model aliases for easy provider switching:

- `LITELLM_GPT_MODEL` (e.g. `openai/gpt-4o-mini`)
- `LITELLM_CLAUDE_MODEL` (e.g. `anthropic/claude-3-5-sonnet-latest`)
- `LITELLM_GEMINI_MODEL` (e.g. `gemini/gemini-2.5-flash`)

Optional fallback chain for chat model construction:

- `LLM_FALLBACK_MODELS` (comma-separated or JSON list)
  - Example: `gpt,openai/gpt-4o-mini`
  - Use when one provider is unavailable; the system will try configured backups in order.

Pass full model names with provider prefixes (`openai/...`, `anthropic/...`, `gemini/...`) or
short aliases (`gpt`, `claude`, `gemini`) to `analyst_model` / `planner_model`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — what exists
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — what we're changing
- [Evaluation](docs/EVALUATION.md) — how quality is measured
- [Runbook](docs/RUNBOOK.md) — how to run/debug it
