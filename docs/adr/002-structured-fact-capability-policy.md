# ADR 002: Structured-Fact Capability Policy

- **Status:** `Accepted`
- **Date:** `2026-08-30`
- **Supersedes:** None
- **Superseded by:** None

## Context

Structured-fact eligibility was previously repeated in the planner prompt, a small
post-model denylist, registry-adjacent aliases, and behavior tests. The duplicated
rules could drift, and a hostile or mistaken planner proposal could reach metric
resolution without a complete capability check. The old safeguard also downgraded
an entire plan when any one structured request was rejected, discarding supported
requests that could still provide useful evidence.

The metric registry remains the authority for executable metric definitions. A
separate policy is needed to decide whether the semantics of a proposed request may
enter that execution path.

## Decision

- `src/structured_facts/capabilities.py` is the deterministic authority for
  structured-fact eligibility and question classification.
- Classification precedence is: explicit unsupported semantics; exact supported
  metric phrases; ambiguous generic metrics; supported aliases; unknown concepts.
- Ratios, margins, yields, per-share metrics, unsupported derivations, and
  comparisons requiring derivation are rejected from structured execution.
- Generic `cash`, `profit`, and `profitability` requests require clarification.
  Precise phrases such as `cash and cash equivalents` and `gross profit` remain
  supported.
- Capability is evaluated independently for every proposed structured request. A
  mixed supported/rejected plan becomes `hybrid`: supported requests stay in the
  structured lane and rejected portions receive KB retrieval work. If every request
  is rejected, the plan becomes `kb`. Any materially ambiguous metric pauses the
  plan for clarification.
- The original request remains a defensive semantic boundary: an unsupported
  top-level ratio, derivation, or narrative change question cannot be converted into
  structured execution merely by decomposing it into supported-looking components.
- Registry-defined derived metrics such as `total_debt` and `capex` are supported
  because their derivation semantics are owned by the registry and metric tool.
- Planner normalization enforces the policy after model output. The orchestrator
  applies the same policy as a defensive boundary before metric resolution or MCP
  execution.
- Rejections emit `STRUCTURED_FACT_CAPABILITY_REJECTED` through the existing
  `OpenIssue` contract with sanitized request context, classification, candidate
  metric IDs, and fallback outcome.

The capability policy may identify a supported or candidate registry metric ID for
policy and diagnostic decisions. It does not own filing-specific phrase resolution,
ticker/year resolution, SEC concept selection, filing anchoring, fact selection, or
metric execution semantics. Phrase and ticker/year resolution live in
`src/structured_facts/resolver.py`; filing eligibility remains in orchestration,
and SEC anchoring, selection and execution remain with the structured metric tool.
The resolver returns selected-target metadata unchanged and does not interpret
its filing form. A resolved identity is not permission to execute a request.

## Consequences

### Positive

- Planner prompts, deterministic guards, runtime validation, and diagnostics share
  one explicit capability set.
- A rejected request no longer removes supported sibling requests.
- Ambiguous generic metrics cannot silently resolve to an unintended SEC fact.
- Manually constructed or hostile planner outputs cannot bypass capability policy at
  the execution boundary.
- Adding a registry metric requires an explicit capability-policy decision.

### Negative and Trade-offs

- The policy and registry must change together when executable capabilities change.
- Conservative unknown classification sends unrecognized synonyms to KB retrieval
  until an explicit alias is approved.
- Any material metric ambiguity pauses the entire plan, including otherwise
  supported sibling requests, because executing before clarification could answer a
  different question than the user intended.

## Alternatives Considered

- **Keep prompt-only guidance:** Rejected because model compliance is not a runtime
  authorization boundary.
- **Use one plan-wide denylist:** Rejected because it discards valid structured
  requests in mixed questions.
- **Move phrase resolution into the capability module:** Rejected because capability
  authorization and filing-specific execution resolution are separate concerns.
- **Treat unknown metrics as clarification:** Not selected. Unknown concepts fall
  back to KB retrieval; clarification is reserved for known material ambiguity.

## References

- [Current runtime architecture](../ARCHITECTURE.md)
- [ADR 001: Routing and Lane Ownership](001-routing-and-lane-ownership.md)
- `src/structured_facts/capabilities.py`
- `src/agents/planner/interactive_target_resolution.py`
- `src/agents/orchestrator/agent_orchestrator.py`
- `src/mcp_server/tools/sec_metric_registry.py`
