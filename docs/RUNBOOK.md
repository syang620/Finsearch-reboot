# FinSearch Runbook

This is the operational entry point for setting up, running, and debugging the
current FinSearch runtime. For system structure and ownership boundaries, see
[Architecture](ARCHITECTURE.md).

## Setup

Create a virtual environment, install the pinned dependencies, and expose `src` on
the Python path:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Before running a KB or hybrid query, ensure Qdrant is reachable and the configured
collection has been ingested. Ensure the selected local model is available in
Ollama, or configure credentials for the selected hosted provider.

Common environment variables:

- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `QDRANT_COLLECTION_NAME` (default `sec_docs_dense_bm25`)
- `TABLES_DIR` (default `data/chunked`)
- `SEC_USER_AGENT` (required for live SEC structured-fact requests; include real
  contact information)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` when
  using the corresponding provider

The optional model aliases `LITELLM_GPT_MODEL`, `LITELLM_CLAUDE_MODEL`, and
`LITELLM_GEMINI_MODEL` select provider-specific models. `LLM_FALLBACK_MODELS` accepts
a comma-separated or JSON fallback list.

## Run a Query

Use the orchestration CLI and request full JSON while debugging:

```bash
PYTHONPATH=src python scripts/run_multi_agent_orchestration.py \
  --query "What was Apple's total debt at year-end 2024?" \
  --json
```

The CLI prompts for clarification when run interactively. To resume a previously
interrupted run, reuse its `run_id` and provide the answer:

```bash
PYTHONPATH=src python scripts/run_multi_agent_orchestration.py \
  --resume-run-id run-12345678 \
  --answer "Apple Inc. for fiscal year 2024" \
  --json
```

Use `--analyst-model` to select the analyst model, `--tables-dir` to override the
chunk directory, and `--no-debug` to suppress detailed analyst debug output.

## MCP Tools

The runtime launches its retrieval and financial-evaluator MCP processes as needed;
a separately managed MCP server is not required for the orchestration CLI. To verify
that the combined stdio server starts, run:

```bash
PYTHONPATH=src python -m mcp_server.server
```

The command waits for MCP input and normally prints nothing. Stop it with `Ctrl-C`
after confirming that startup does not produce an import or configuration error.

## Build or Refresh the Knowledge Base

Use the unified ingestion CLI:

```bash
PYTHONPATH=src python scripts/ingestion_cli.py --help
PYTHONPATH=src python scripts/ingestion_cli.py run-all --help
```

`run-all` downloads filings, chunks HTML, summarizes tables, generates embeddings,
and ingests Qdrant. See the [scripts guide](../scripts/README.md) for individual
stages and required credentials.

## Run Evaluations

Use [Evaluation](EVALUATION.md) to select the applicable gate and component guide.
Raw outputs belong under `artifacts/evals/`; durable measured results belong in
[Evaluation Baselines](EVALUATION_BASELINES.md).

## Debugging

Start with the orchestration CLI's `--json` output. Inspect:

- `status`, `ok`, and `failure_stage` for the failing stage.
- `open_issues` for normalized planner, retrieval, structured-fact, or analyst errors.
- `planner` and `route` for target resolution and lane selection.
- `retrieval` and `structured_fact_results` for lane-specific evidence or failures.
- `analyst` for answer status, citations, calculations, and tool failures.
- `orchestrator_trace` for planner, retrieval, structured-fact, and total timings.

### Retrieval returns no evidence

- Confirm Qdrant is running at `QDRANT_HOST:QDRANT_PORT`.
- Confirm `QDRANT_COLLECTION_NAME` matches the ingested collection.
- Confirm the requested ticker, fiscal year, and form type exist in that collection.
- Confirm `TABLES_DIR` points to the chunk files used by the retrieval result payloads.

### Model calls fail

- Confirm the requested model is installed and visible with `ollama list`, or that
  the selected hosted provider key is set.
- Expect the first local request to take longer while the model loads.
- Check the planner and analyst errors in the JSON output before changing routes or
  retrieval configuration.

### Structured facts fail

- Set `SEC_USER_AGENT` to a descriptive value with real contact information.
- Inspect each item in `structured_fact_results`; unsupported, ambiguous, missing,
  and tool-error statuses are distinct outcomes.
- Use `open_issues` and `failure_stage` to distinguish metric resolution from live
  SEC request failures.

### MCP startup fails

- Run the combined MCP server directly to expose import and startup errors.
- Confirm `PYTHONPATH=src` and that dependencies from `requirements.txt` are installed.
- For analyst calculations, inspect `analyst.open_issues` for financial-evaluator
  startup or connection failures.

### Clarification cannot resume

- Resume with the exact `run_id` returned by the interrupted result.
- Supply one `--answer` per clarification question, or pass a JSON array with
  `--answers-json`.
- Check that the checkpoint path is writable and that the run has not expired under
  the configured checkpoint TTL.
