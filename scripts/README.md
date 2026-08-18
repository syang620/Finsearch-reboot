# Scripts

This directory contains operator/developer scripts for ingestion and evaluation.

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
- Default Qdrant collection for ingestion is `sec_docs_dense_bm25`.
- Default ingest schema for that collection is dense + BM25 only.
- Use environment variables for secrets (for example `SEC_API_KEY`).
- Runtime usage audit and rationale are captured in `artifacts/scripts_runtime_usage_audit.md`.

## Evaluation Scripts
- `scripts/evals/retrieval/eval_retrieval_v0.py`: retrieval-only evaluation CLI (deterministic + optional RAGAS metrics).
- `scripts/evals/agents/eval_agents_v1.py`: route-aware multi-lane end-to-end evaluation CLI.
- `scripts/evals/agents/eval_agents_v0.py`: legacy KB-oriented end-to-end evaluation CLI.
- `scripts/evals/agents/eval_planner_routes.py`: planner routing evaluation CLI.
- `scripts/evals/llm_judge/eval_llm_judge_rag_answers.py`: LLM-as-judge evaluation CLI.
- `scripts/evals/chunking/`: chunking strategy and table-summary benchmark scripts.
- Backward-compatible wrappers: `scripts/eval_retrieval_v0.py`, `scripts/eval_planner_routes.py`, and the legacy files directly under `scripts/evals/`.
