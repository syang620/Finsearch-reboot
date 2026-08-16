# ADR 001: Routing and Lane Ownership

- **Status:** `Accepted`
- **Date:** `2026-08-15`
- **Supersedes:** None
- **Superseded by:** None

## Context

FinSearch can answer filing questions through KB retrieval, deterministic structured
SEC facts, or both. Without an explicit ownership boundary, the planner could become
coupled to metric registry identifiers and MCP execution details, while sequencing
and partial-failure behavior could be distributed across the planner, tools, and
analyst.

The implemented runtime already separates planning from execution through a
route-aware orchestrator. This ADR records that existing boundary.

## Decision

- The planner proposes one of `kb`, `structured_fact`, or `hybrid` and emits
  execution intent and requests.
- The planner does not own downstream metric resolution or tool execution. It emits
  human-readable metric hints rather than registry or SEC concept identifiers.
- The orchestrator owns graph branching, clarification and resume, evidence-lane
  sequencing, failure and degradation handling, and analyst-packet construction.
- KB retrieval and structured facts are sibling evidence lanes. Their results are
  normalized into the analyst packet rather than one lane being treated as an
  implementation detail of the other.
- The `hybrid` route currently executes KB retrieval first and structured-fact
  execution second. The lanes do not mutate shared graph state concurrently.
- The analyst consumes the normalized evidence packet and does not select or execute
  upstream evidence lanes.

This ADR assigns ownership only. It does not define the structured-fact capability
set, degradation statuses, citation requirements, or filing-amendment precedence.
Those decisions will receive separate ADRs when their implementation PRs make them
concrete.

## Consequences

### Positive

- The planner remains decoupled from registry identifiers, SEC concepts, and MCP
  invocation details.
- Execution sequencing and partial-failure coordination have one owner.
- KB and structured evidence reach the analyst through a uniform packet boundary.
- Route behavior can be evaluated independently from downstream tool internals.

### Negative and Trade-offs

- The orchestrator carries additional branching, adaptation, and failure-handling
  responsibility.
- Hybrid latency is currently additive because KB retrieval and structured-fact
  execution run sequentially.
- Changing hybrid execution to concurrent lanes will require explicit state-merge,
  cancellation, and degradation behavior rather than a local scheduling change.

## Alternatives Considered

- **Planner-owned tool execution:** Rejected because it would couple routing prompts
  and planner output to downstream registries, tool schemas, and execution failures.
- **Analyst-selected evidence lanes:** Rejected because the analyst should reason
  over supplied evidence rather than control retrieval topology.
- **Concurrent hybrid execution:** Not selected for the current runtime because the
  graph uses shared state and does not yet define the required merge and degradation
  semantics. This may be reconsidered in a future ADR.

## References

- [Current runtime architecture](../ARCHITECTURE.md)
- `src/agents/planner/interactive_target_resolution.py`
- `src/agents/orchestrator/agent_orchestrator.py`
- `src/agents/contracts.py`
