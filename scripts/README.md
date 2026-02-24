# Scripts

This directory contains operator/developer scripts for data ingestion and prep.

## Scope
- These scripts are not part of the agent serving runtime.
- They are used to build/update the retrieval knowledge base.

## Primary Entry Point
Use the unified ingestion CLI:

```bash
PYTHONPATH=src python scripts/ingestion_cli.py --help
PYTHONPATH=src python scripts/ingestion_cli.py run-all --help
```

`run-all` executes:
1. HTML download
2. HTML chunking
3. Table summarization
4. Embedding generation
5. Qdrant ingestion

## Key Scripts
- `scripts/ingestion_cli.py`: unified CLI (recommended)
- `scripts/orchestrate_html_downloads.py`: SEC filing HTML download
- `scripts/orchestrate_chunk_html_filings.py`: filing chunk generation
- `scripts/orchestrate_table_summaries.py`: table summary generation
- `scripts/orchestrate_embeddings.py`: embedding generation
- `scripts/orchestrate_ingestion.py`: Qdrant ingestion
- `scripts/download_xbrl.py`: XBRL download helper (requires `SEC_API_KEY`)

## Notes
- Default Qdrant collection for ingestion is `sec_docs_hybrid`.
- Use environment variables for secrets (for example `SEC_API_KEY`).
- Runtime usage audit and rationale are captured in `artifacts/scripts_runtime_usage_audit.md`.
