# ADR 003: Per-Lane Status and Degradation Semantics

- **Status:** `Accepted`
- **Date:** `2026-09-02`
- **Supersedes:** None
- **Superseded by:** None

## Context

The route-aware runtime can execute KB retrieval and structured facts independently,
but previously returned only `completed`, `failed`, or `interrupted`. The evaluator
reconstructed lane outcomes after execution, so callers could receive an
unqualified success even when one hybrid lane failed. A tool status also did not
prove that evidence survived admission into analyst context.

## Decision

The runtime is authoritative for serialized `lanes`, `degradation`, top-level
`status`, and `failure_stage`. The evaluator independently derives a shadow result
and reports inconsistencies; it never replaces runtime values.

Each lane reports `requested`, `attempted`, normalized `issues`, and one status:
`not_requested`, `ok`, `partial`, `failed`, `ambiguous`, `unsupported`, or
`skipped`.

Usability is an internal derivation, not a public status synonym:

- KB is usable only when at least one real non-structured context item survives.
- Structured fact is usable only when at least one PR4-valid typed fact enters the
  analyst packet.
- A structured tool result may be `partial` while remaining unusable.

Lane normalization follows this truth table:

| Runtime outcome | Lane status | Usable |
|---|---|---:|
| Lane not requested by the effective plan | `not_requested` | No |
| Requested but never processed | `skipped` | No |
| All operations succeeded and admitted evidence exists | `ok` | Yes |
| Admitted evidence exists and any operation was unsuccessful | `partial` | Yes |
| Structured tool returned partial without admitted evidence | `partial` | No |
| Every requested structured operation was ambiguous | `ambiguous` | No |
| Every requested structured operation was unsupported at the runtime boundary | `unsupported` | No |
| Any other or mixed unusable outcome | `failed` | No |

PR3 capability fallback is proposal history. When planning normalizes an unsupported
structured proposal to effective KB-only routing, the structured lane is
`not_requested`; `STRUCTURED_FACT_CAPABILITY_REJECTED` remains observable in its
issues. `unsupported` is reserved for a structured lane that was actually requested
and then rejected by the orchestrator's defensive boundary.

Top-level outcomes are:

| Evidence and analyst outcome | Status | `ok` |
|---|---|---:|
| At least one usable lane; every requested lane `ok`; analyst succeeds | `completed` | Yes |
| At least one usable lane; any requested lane non-OK; analyst succeeds | `degraded` | Yes |
| Filing task has zero usable requested lanes; analyst is skipped | `failed` | No |
| Analyst fails after any evidence outcome | `failed` | No |
| Graph awaits clarification | `interrupted` | No |
| Explicitly non-filing task succeeds without evidence lanes | `completed` | Yes |

`degradation.active` is independent of final success. It is true whenever an
actually requested lane is non-OK, including total evidence failure and a later
analyst failure. `affected_lanes` uses canonical order (`kb`, then
`structured_fact`) and excludes `not_requested` proposal history. The deterministic
notice contains only lane names and normalized statuses; raw error text is never
interpolated. The identical typed summary and notice are supplied to the analyst,
which is instructed not to claim unavailable coverage.

Fatal evidence-stage selection uses the terminal attempted lane that leaves a
filing task with no usable evidence. With the current execution order, total hybrid
failure resolves to `structured_fact`; structured-only failure resolves to
`structured_fact`; KB-only failure resolves to `retrieval`. If no evidence lane was
attempted because the plan was invalid or empty, the stage is `planner`. A later
analyst failure always resolves to `analyst` while retaining evidence degradation.

## Consequences

### Positive

- Callers and analysts receive the same explicit coverage boundary.
- Filing tasks fail closed when no admitted evidence remains.
- Hybrid partial success remains usable without hiding sibling-lane failure.
- Evaluation can detect runtime/status drift independently.

### Negative and Trade-offs

- Runtime output and analyst packets are larger.
- Existing consumers must accept `degraded` as a successful top-level status.
- Lane derivation remains coupled to the current two-lane execution order; adding a
  lane requires extending status ordering, notices, tests, and failure-stage rules.

## Alternatives Considered

- Evaluator-only degradation was rejected because it cannot protect production
  callers or analyst claims.
- Tool-status-based usability was rejected because malformed or inadmissible
  results could be treated as evidence.
- Treating all partial outcomes as fatal was rejected because a valid sibling lane
  can still support a disclosed answer.

## References

- [ADR 001](001-routing-and-lane-ownership.md)
- [ADR 002](002-structured-fact-capability-policy.md)
- [Agent evaluation guide](../EVALUATION_AGENTS.md)
- [Implementation roadmap](../IMPLEMENTATION_PLAN.md)
