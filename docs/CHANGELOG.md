# Changelog

All notable repository updates can be tracked here.

## 2026-08-15
### Architecture documentation and governance
- Documented the current multi-lane architecture and operating procedures in `docs/ARCHITECTURE.md` and `docs/RUNBOOK.md`.
- Added evaluation policy and baseline guidance in `docs/EVALUATION.md` and `docs/EVALUATION_BASELINES.md`.
- Added the implementation roadmap and ADR framework under `docs/IMPLEMENTATION_PLAN.md` and `docs/adr/`.

## 2026-06-20
### Repository organization and hygiene
- Organized tests and evaluation scripts by domain under `tests/` and `scripts/evals/`.
- Updated repository documentation and compatibility entrypoints for the new layout.
- Added `.gitignore` coverage and removed tracked cache and notebook-checkpoint files.

## 2026-06-14
### Planner evaluation
- Added the planner-routing evaluation harness and core dataset and baseline artifact under `scripts/evals/agents/`, `data/evals/agents/planner/`, and `artifacts/evals/agents/planner/`.
- Added planner route, schema, and safety tests under `tests/agents/planner/`.
- Added Ollama planner support through `src/agents/planner/` and `src/llm_client.py`.

## 2026-06-03
### Structured-fact routing
- Extended the planner contract and orchestrator to route structured financial fact requests through the appropriate retrieval tooling.
- Added schema and planner-error coverage under `tests/agents/`.

## 2026-04-12
### SEC structured metrics
- Added SEC metric MCP tooling under `src/mcp_server/tools/`, including filing-period anchoring and metric computation support.
- Added fixture-backed computation, filing-anchor, and tool tests under `tests/tools/sec_metric/`.
- Removed stale generated data and obsolete notebooks from version control.

## 2026-03-21
### Runtime and retrieval cleanup
- Consolidated runtime behavior around the canonical planner and retrieval paths and removed obsolete compatibility modules.
- Made dense-plus-BM25 ingestion the default and added LLM provider fallback configuration.
- Added reliability coverage for planner, orchestrator, analyst, retrieval, and evaluator workflows under `tests/`.

## 2026-03-09
### Runtime and ingestion workflows
- Updated agent and retrieval workflows around the shared `src/llm_client.py` and canonical query-planning paths.
- Added shared chunk-path handling in `src/ingestion/chunk_paths.py`.
- Expanded `scripts/` with ingestion orchestration, diagnostics, validation, and retrieval benchmark utilities.

## 2026-03-01
### Evaluation pipelines
- Added agent and retrieval evaluation contracts, runners, and RAGAS metrics under `src/evals/`.
- Added seed datasets under `data/evals/` and evaluation entrypoints under `scripts/evals/`.
- Added agent and retrieval evaluation guidance in `docs/EVALUATION_AGENTS.md` and `docs/EVALUATION_RETRIEVAL.md`.

## 2026-02-24
### Documentation structure
- Added root `README.md` as a concise project entrypoint.
- Added `docs/ARCHITECTURE.md` with detailed architecture/runbook content.
- Replaced `code.md` with a legacy pointer to current docs.
- Preserved links to `scripts/README.md` and `docs/CHANGELOG.md` for operational and historical context.

## 2026-02-23
### Scripts and ingestion tooling
- Added `scripts/ingestion_cli.py` as a unified ingestion entrypoint.
- Added `scripts/_common.py` to centralize ticker/quarter parsing.
- Updated orchestrator scripts to reuse shared helpers and `ingestion.*` modules.
- Updated ingestion default collection to `sec_docs_dense_bm25` in `scripts/orchestrate_ingestion.py`.
- Removed hardcoded API key usage from `scripts/download_xbrl.py`; now requires `SEC_API_KEY`.
- Added runtime-boundary audit in `artifacts/scripts_runtime_usage_audit.md`.

### Tracking guidance
- Add one dated section per meaningful change batch.
- Keep entries concise: what changed, why, and impact.
- Reference affected paths for fast navigation.
