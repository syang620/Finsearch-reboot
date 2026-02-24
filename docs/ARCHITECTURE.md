# FinSearch Architecture

This document is the detailed engineering map for the current agentic SEC-filing RAG system.

## 1. End-to-End Runtime Flow

Primary runtime flow:

1. `PlannerAgent` generates `PlannerOutput`.
2. If retrieval is needed and metadata is sufficient, orchestrator calls retrieval MCP tool.
3. Retrieval output is transformed into `AnalystPacket` with hydrated context.
4. `AnalystAgent` answers (and calls `financial_evaluator` for compute tasks).
5. Orchestrator returns planner/retrieval/analyst outputs plus timing traces.

Main entrypoint:

- `agents.orchestrator.run_multi_agent_orchestration`

Implementation:

- `src/agents/orchestrator/agent_orchestrator.py`

## 2. Core Modules

### 2.1 Agents (`src/agents`)

- `src/agents/contracts.py`
  - Shared Pydantic schemas for planner/retrieval/analyst handoffs.

- `src/agents/planner/agent.py`
  - Deterministic pre-extraction (ticker/year/form hints).
  - Intent classification and compute cue overrides.
  - Query expansion via `retrieval.query_expansion`.
  - LLM plan generation + schema validation.
  - Fallback plan path and timing trace output.

- `src/agents/retrieval/mcp_client.py`
  - Async stdio MCP client wrapper.
  - Invokes `sec_retrieve_tables` on MCP server.

- `src/agents/retrieval/agent.py`
  - Thin retrieval node wrapper used by orchestrator.

- `src/agents/analyst/agent.py`
  - Builds analyst prompts from `AnalystPacket` context.
  - Executes ReAct-style loop with tool access.
  - Parses tool calls and emits `AnalystRunResult` trace.

- `src/agents/analyst/table_loader.py`
  - Hydrates retrieved table references into full table dictionaries from chunk files.

### 2.2 Retrieval (`src/retrieval`)

- `src/retrieval/pipeline.py`
  - Hybrid retrieval orchestration.
  - Dedupe + enrichment + reranking pipeline.

- `src/retrieval/evaluator.py`
  - Retrieval utilities and scoring helpers.
  - BM25/dense/sparse helpers and selection utilities.

- `src/retrieval/rerank_enricher.py`
  - Candidate enrichment for reranking context.

- `src/retrieval/query_expansion.py`
  - Query expansion prompt and expansion execution.

- `src/retrieval/ollama_client.py`
  - Shared Ollama invocation and list-output parsing.

- `src/retrieval/accounting_terms.py`
  - Accounting term digest utilities for planner/query expansion prompts.

### 2.3 Analysis (`src/analysis`)

- `src/analysis/answer_synthesis.py`
  - Table text loading and answer-synthesis helper functions.
  - Used by notebook-side analysis workflows.

### 2.4 MCP Server (`src/mcp_server`)

- `src/mcp_server/server.py`
  - Registers and runs MCP tools over stdio.

- `src/mcp_server/tools/sec_retrieval.py`
  - Retrieval MCP tool (`sec_retrieve_tables`).
  - Uses `retrieval.pipeline` + lexical scoring.

- `src/mcp_server/tools/financial_evaluator.py`
  - Safe arithmetic tool used by analyst compute tasks.

### 2.5 Ingestion (`src/ingestion`)

- `src/ingestion/sec_html_fetcher.py`
- `src/ingestion/sec_chunker.py`
- `src/ingestion/chunk_splitter.py`
- `src/ingestion/tables_summarizer.py`
- `src/ingestion/sec_embedder.py`
- `src/ingestion/qdrant_ingester.py`

These components are build-time/indexing infrastructure (not request-time serving).

## 3. Contracts and Data Interfaces

All cross-agent payloads should be driven by `src/agents/contracts.py`.

Primary models:

- `PlannerOutput`
- `RetrieveTablesResponse`
- `AnalystPacket`
- `AnalystRunResult`

Contract-first rule:

1. Update contract schema.
2. Update planner output and retrieval packet adapters.
3. Update analyst input/result handling.

## 4. Orchestration Details

Orchestrator graph (LangGraph) includes these phases:

1. Initialize run state/timing.
2. Planner node.
3. Route:
   - retrieval path if `retrieval_needed=True` and metadata exists.
   - analyst-only path otherwise.
4. Retrieval node (MCP tool call).
5. Packet-building node (with retrieval quality gate issue tagging).
6. Analyst node.
7. Finalize timing and return structured result.

Returned top-level object contains:

- planner payload
- retrieval payload (if attempted)
- analyst payload
- orchestrator timing traces

## 5. Runtime Configuration

Common environment variables:

- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `QDRANT_COLLECTION_NAME` (default `sec_docs_hybrid`)
- `TABLES_DIR` (default resolves to `data/chunked`)

For XBRL helper script only:

- `SEC_API_KEY`

## 6. Running the System

### 6.1 Start MCP server

```bash
PYTHONPATH=src python -m mcp_server.server
```

### 6.2 Run orchestration

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

## 7. Ingestion Pipeline

Use the unified ingestion CLI:

```bash
PYTHONPATH=src python scripts/ingestion_cli.py --help
PYTHONPATH=src python scripts/ingestion_cli.py run-all --help
```

Reference:

- `scripts/README.md`

## 8. Troubleshooting

### 8.1 Retrieval returns empty/low quality

- Verify planner extracted `ticker` and `fiscal_year`.
- Confirm `sec_docs_hybrid` exists and is populated.
- Confirm `TABLES_DIR` matches chunk files used for lexical scoring.

### 8.2 Analyst does not call `financial_evaluator`

- Ensure planner intent/task is compute-oriented.
- Ensure context includes required numeric line items.

### 8.3 MCP timeout on first request

- First request can include model loading overhead.
- Retry once before assuming failure.

## 9. Migration Notes

Legacy package `src/rag10kq` has been removed.

Key path migrations:

- `rag10kq.query_expansion_helper` -> `retrieval.query_expansion`
- `rag10kq.retrieval_evaluator` -> `retrieval.evaluator`
- `rag10kq.rerank_context_enricher` -> `retrieval.rerank_enricher`
- `rag10kq.pipeline` -> `retrieval.pipeline`
- `rag10kq.sec_*` + `rag10kq.qdrant_ingester` + `rag10kq.tables_summarizer` -> `ingestion.*`
- `rag10kq.context_generator` helpers -> `analysis.answer_synthesis`
- `mcp_backend` -> `mcp_server`

## 10. Compatibility Layer

`src/tools` remains as a compatibility shim forwarding to `src/mcp_server` modules.

Prefer direct `mcp_server` imports for all new code.
