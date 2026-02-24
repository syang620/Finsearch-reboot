# FinSearch Reboot

Agentic RAG system for SEC filings (10-K / 10-Q) using a planner -> retrieval -> analyst workflow.

## Overview

Runtime flow:

1. Planner extracts metadata, intent, and retrieval queries.
2. Retrieval fetches/reranks filing context via MCP tools.
3. Analyst computes metrics and produces the final grounded answer.

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
- `QDRANT_COLLECTION_NAME` (default `sec_docs_hybrid`)
- `TABLES_DIR` (default resolves to `data/chunked`)
- `SEC_API_KEY` (required only for `scripts/download_xbrl.py`)

## Documentation

- Detailed architecture and runbook: `docs/ARCHITECTURE.md`
- Ingestion script guide: `scripts/README.md`
- Change history: `docs/CHANGELOG.md`

## Compatibility Note

`src/tools` is a compatibility shim layer forwarding to `src/mcp_server`. Prefer `mcp_server` paths in new code.
