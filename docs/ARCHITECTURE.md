# FinSearch Architecture

This document describes the FinSearch runtime as it is implemented today. It is the
current-system reference, not a roadmap, implementation plan, runbook, or record of
evaluation results.

## 1. Runtime Overview

FinSearch is a route-aware SEC-filing research system. The planner describes the
work, the orchestrator owns execution, and the analyst answers from a normalized
evidence packet.

```text
User query
    |
    v
Planner
    |-- needs clarification --> interrupt --> checkpoint --> resume planner
    |
    `-- completed: kb | structured_fact | hybrid
                         |
                         v
                    Orchestrator
                    |-- KB retrieval evidence
                    |-- structured SEC fact evidence
                    `-- sequential combination for hybrid
                         |
                         v
                    AnalystPacket
                         |
                         v
                       Analyst
```

The runtime entrypoint is:

- `agents.orchestrator.run_multi_agent_orchestration`
- Implementation: `src/agents/orchestrator/agent_orchestrator.py`

### Route contract

| Route | Current execution |
|---|---|
| `kb` | Use filing retrieval for narrative, comparative, table-heavy, interpretive, or unsupported structured-metric questions. |
| `structured_fact` | Skip KB retrieval and execute supported scalar SEC metric requests. |
| `hybrid` | Run KB retrieval first, then structured facts, and provide both evidence sets to the analyst. |

The planner recommends a route. The orchestrator interprets that route and owns
sequencing, interruption, failure handling, packet construction, and finalization.

## 2. Planner Routing

`InteractivePlannerAgent` combines deterministic extraction with an LLM planning
step. It extracts or resolves:

- Company and ticker.
- Fiscal year and form type.
- Task class and analysis instructions.
- Retrieval targets and jobs.
- Human-readable structured metric hints.
- Clarification requirements and open issues.

Planner output is normalized after the model call. The shared structured-fact
capability policy classifies every proposed structured request independently. It
retains supported requests, moves rejected portions to KB work, and requests
clarification for materially ambiguous metrics. If the planner model is unavailable
or its output is invalid, the current fallback produces a conservative KB-oriented
plan from deterministically extracted metadata.

A planner turn has one of three effective states:

- `completed`: execution can continue using the selected route.
- `needs_clarification`: blocking metadata or intent is unresolved.
- `error`: the orchestrator builds an error packet rather than running an evidence lane.

Implementation:

- `src/agents/planner/interactive_target_resolution.py`

## 3. Clarification and Resume

Clarification is part of the orchestrator graph rather than an out-of-band caller
loop.

1. The planner returns `needs_clarification` with one or more questions.
2. The orchestrator invokes a LangGraph interrupt.
3. Graph state is persisted by the SQLite checkpointer under the run's thread ID.
4. The caller resumes the same run with answers.
5. Answers are appended to clarification history and the planner runs again.

The planner treats recorded answers as authoritative and should not repeat a resolved
question. A run remains interrupted until it is resumed or otherwise terminated by
the caller.

## 4. Orchestration and Route Execution

The LangGraph orchestrator contains nodes for initialization, planning, interruption,
resume handling, retrieval checks, KB retrieval, structured-fact execution, packet
construction, open-issue handling, analysis, and finalization.

### KB route

For a KB plan that requires retrieval, the orchestrator validates required target
metadata, executes the retrieval workflow, converts the returned evidence into
analyst context, and records retrieval issues. A KB plan that does not require
retrieval proceeds to packet construction without a retrieval call.

### Structured-fact route

The orchestrator skips KB retrieval, revalidates each request against the shared
capability policy, resolves permitted metric hints against the supported metric
registry, calls the SEC metric tool, and converts each result into analyst context.

### Hybrid route

Hybrid execution is intentionally sequential in the current implementation:

```text
KB retrieval --> structured-fact execution --> combined AnalystPacket
```

The two lanes do not mutate graph state concurrently. Their evidence becomes sibling
context in the same packet. A lane can return errors or unresolved results without
silently disappearing; those results and associated open issues remain visible to
the analyst and in the final run output.

### Final run output

The orchestrator returns a structured object containing:

- `run_id`, `thread_id`, route, status, and `ok`.
- A derived failure stage and accumulated open issues.
- Planner and planner-turn payloads.
- Compact KB retrieval output.
- Structured-fact resolver and tool results.
- Analyst output and pending interrupt data.
- Total, planner, retrieval, and structured-fact timing traces.

## 5. KB Retrieval Lane

The planner produces targets and retrieval jobs. The retrieval workflow executes the
applicable jobs, invokes the retrieval MCP tool, reviews the returned evidence, and
can retry within a fixed attempt limit.

`sec_retrieve_tables` performs the filing search against the configured Qdrant
collection. The current retrieval path includes:

- Target-specific queries and metadata filters.
- Dense and BM25 retrieval combined with reciprocal-rank fusion.
- Candidate enrichment and reranking.
- Normalized text, row, and table evidence.
- Per-attempt review, timing, and error information.

Multiple retrieval jobs or targets may execute concurrently inside the retrieval
workflow. This is separate from hybrid lane sequencing, which remains sequential.

Primary implementations:

- `src/agents/retrieval/query_planner_v2.py`
- `src/agents/retrieval/mcp_client.py`
- `src/mcp_server/tools/sec_retrieval.py`

## 6. Structured-Fact Lane

The planner emits human-readable requests such as `revenue` or `total debt`; it does
not emit SEC concept names. The pure resolver in `src/structured_facts/resolver.py`
maps those hints to the supported metric registry and resolves ticker and fiscal
year. It returns the existing selected target as opaque metadata. Orchestration
owns capability permission, filing-form eligibility and calls to `sec_get_metric`.

`src/structured_facts/capabilities.py` answers whether structured execution is
permitted and classifies unsupported, ambiguous, and unknown requests. It also
generates the planner prompt appendix and provides the orchestrator's defensive
execution check. It may report supported or candidate metric IDs for policy and
diagnostics, but it does not perform filing-specific metric resolution or own SEC
fact execution semantics.

Classification precedence is explicit: unsupported semantics, exact supported
phrases, ambiguous generic metrics, supported aliases, then unknown concepts. Mixed
supported and rejected proposals become hybrid plans; supported requests remain in
the structured lane while KB retrieval covers rejected portions. A materially
ambiguous metric pauses the plan for clarification. The original user request is
also checked so an unsupported top-level derivation cannot bypass policy through
supported-looking component decomposition.

The metric tool uses SEC submissions and company-facts data to:

- Resolve the company CIK.
- Anchor the request to an annual `10-K` or `10-K/A` filing.
- Select facts associated with the requested filing and fiscal year.
- Return direct metrics or explicitly registered derived metrics.
- Preserve result status, value, unit, filing identifiers, components, and source URL
  in the tool result.

Unsupported, ambiguous, missing, and error results use explicit statuses rather than
inventing a value. The client applies SEC request identification, rate limiting,
retry behavior, and local caching.

Primary implementations:

- `src/mcp_server/tools/sec_metric_registry.py`
- `src/mcp_server/tools/sec_metric.py`
- `src/mcp_server/tools/sec_metric_client.py`

## 7. Analyst Packet and Native Structured Evidence

The orchestrator normalizes available evidence into `AnalystPacket`. KB results are
represented as analyst-visible context items with stable context IDs and available
filing provenance.

Successful SEC metric results are represented as native `STRUCTURED_FACT` context
items with a validated `StructuredFactEvidence` payload. Admission requires a
resolved metric identity, a successful tool result, and a finite numeric value that
is not a Boolean. Malformed, partial, ambiguous, unsupported, unresolved, or failed
results remain execution results and open issues; they do not become analyst
evidence.

The SEC tool result is authoritative for accession number, report date, filed date,
source URL, and returned form. Packet metadata may fill only missing ticker, fiscal
year, and requested form. The metric label comes from registry metadata, falling
back to the authoritative metric ID. Structured facts are ordered ahead of KB
contexts so successful facts remain inside the analyst's bounded five-context
window. The analyst and evaluator use the same deterministic typed renderer; no
successful structured fact is flattened into synthetic text.

Shared packet models live in:

- `src/agents/contracts.py`

## 8. Analyst

`AnalystAgent` receives the user query, analysis task, normalized context, open
issues, and packet metadata. It is instructed to reason only from supplied context.

The analyst:

- Produces a structured `AnalystRunResult`.
- Uses `financial_evaluator` for arithmetic instead of performing unsupported
  free-form calculations.
- Runs within bounded model-attempt and tool-round limits.
- Returns answer status, narrative answer, metric/computation data, citations, and a
  trace.
- Cites analyst-visible context IDs; unknown IDs are excluded from resolved citation
  metadata and recorded as warnings.

Implementation:

- `src/agents/analyst/agent.py`

## 9. MCP Tool Boundary

The stdio MCP server registers three runtime tools:

| Tool | Responsibility |
|---|---|
| `sec_retrieve_tables` | Search, combine, rerank, and return SEC filing evidence from the KB. |
| `sec_get_metric` | Return a supported, filing-anchored SEC metric or an explicit non-OK status. |
| `financial_evaluator` | Safely evaluate arithmetic expressions from named variables. |

The server is assembled in `src/mcp_server/server.py`. The orchestrator and agents
use MCP clients rather than importing tool implementations as their runtime
interface.

## 10. Current Implementation Boundaries

The following are current architectural limitations, not descriptions of planned
behavior:

- The shared `PlannerOutput` contract does not contain every field used by the
  orchestrator. Runtime planner payloads and LangGraph state still contain normalized
  dictionaries alongside Pydantic models.
- Structured metric resolution is separate from execution coordination. Capability
  and resolver aliases remain distinct; resolution preserves existing precedence
  and fallback behavior, while orchestration owns permission and execution.
- Structured results are flattened into synthetic `TEXT` context. Native numeric
  types, metric status, components, and source URLs are not fully represented by the
  analyst context contract.
- Hybrid lanes execute sequentially rather than concurrently.
- The top-level failure-stage model has no dedicated structured-fact stage and no
  complete per-lane status model.
- The planner fallback is KB-oriented and does not reconstruct structured or hybrid
  requests when the planner model fails.
- Structured facts are limited to the registered annual metrics and filing-anchor
  behavior. Other ratios, comparisons, quarterly questions, and broad filing
  interpretation remain KB work.
- The existing end-to-end agent evaluator primarily models the legacy
  planner/retrieval/analyst path and does not yet provide complete structured-fact or
  hybrid validation.

These boundaries are important when interpreting runtime status, citations, and
evaluation results. Proposed changes and measured model baselines intentionally do
not belong in this document.
