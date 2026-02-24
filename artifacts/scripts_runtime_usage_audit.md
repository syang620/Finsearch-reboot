# Scripts Runtime Usage Audit

Date: 2026-02-23
Repo: `FinSearch-reboot`

## Scope
Verify whether `scripts/*.py` are used by the current agentic RAG serving runtime, and identify consolidation opportunities given `src/ingestion/*`.

## What Was Checked
1. Runtime code paths (`src/agents`, `src/mcp_server`, `src/retrieval`, `src/tools`) for direct references to `scripts/*`.
2. Packaging/entrypoints in `pyproject.toml`.
3. Current ingestion script inventory and overlap.

## Evidence
### 1) No runtime imports/calls to `scripts/*`
Command:
```bash
rg -n "orchestrate_|ingestion_cli|scripts/" src/agents src/mcp_server src/retrieval src/tools -g '!**/.ipynb_checkpoints/**'
```
Result: no matches.

Interpretation: current serving/orchestration runtime does not call `scripts/*` directly.

### 2) Runtime uses package modules, not script wrappers
Command:
```bash
rg -n "from rag10kq|import rag10kq|from retrieval|from ingestion|import ingestion" src/agents src/mcp_server src/retrieval src/tools -g '!**/.ipynb_checkpoints/**'
```
Representative matches:
- `src/mcp_server/tools/sec_retrieval.py` imports `retrieval.pipeline` and `retrieval.evaluator`.
- `src/agents/planner/agent.py` imports `retrieval.query_expansion`.
- No references to `scripts/*`.

### 3) No packaged script entrypoints
Command:
```bash
rg -n "\[project\.scripts\]|entry_points|console_scripts|scripts/" pyproject.toml
```
Result: no matches.

Interpretation: `scripts/*.py` are developer/operator utilities, not installed runtime commands.

### 4) Script inventory is ingestion-oriented
`scripts/` currently contains ingestion/data-prep utilities (download/chunk/summarize/embed/ingest) plus helper CLI.

## Consolidation Implemented
1. Added shared helper module: `scripts/_common.py`
- Centralizes ticker loading and quarter parsing.

2. Added unified CLI: `scripts/ingestion_cli.py`
- Subcommands: `download-html`, `chunk-html`, `summarize-tables`, `embed`, `ingest-qdrant`, `run-all`.
- Uses existing script modules as underlying executors.

3. Reduced duplicate logic in orchestrators
- Updated:
  - `scripts/orchestrate_html_downloads.py`
  - `scripts/orchestrate_chunk_html_filings.py`
  - `scripts/orchestrate_embeddings.py`
  - `scripts/orchestrate_table_summaries.py`
- All now reuse `_common` for ticker parsing.

4. Aligned ingestion default collection with retrieval backend
- `scripts/orchestrate_ingestion.py` default changed to `sec_docs_hybrid`.

5. Removed hardcoded secret in script
- `scripts/download_xbrl.py` now requires `SEC_API_KEY` environment variable.

## Conclusion
- `scripts/*.py` are not part of the current agentic serving runtime path.
- They are ingestion/operator tooling and can be consolidated independently.
- The implemented consolidation (`ingestion_cli` + shared helpers) is safe with respect to runtime serving behavior.

## Suggested Follow-up (Optional)
1. Add a dedicated `.gitignore` policy for `__pycache__/` and `.ipynb_checkpoints/` to keep script diffs clean.
2. If desired, add `[project.scripts]` entrypoints for ingestion commands to standardize operator usage.
