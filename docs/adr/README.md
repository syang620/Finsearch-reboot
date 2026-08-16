# Architecture Decision Records

Architecture Decision Records (ADRs) capture consequential architectural choices,
the reasons behind them, and their trade-offs. `docs/ARCHITECTURE.md` describes what
the current system does; ADRs explain why durable boundaries and behaviors were
chosen.

Create an ADR when a decision becomes concrete enough to implement or enforce. Do
not create speculative ADRs for unresolved roadmap work.

## Statuses

| Status | Meaning |
|---|---|
| `Proposed` | The decision is under review and is not yet authoritative. |
| `Accepted` | The decision is approved and applies to the current system or its committed implementation. |
| `Superseded` | A later ADR replaces this decision; retain the original record and link both ADRs. |

## Naming and Lifecycle

- Name ADRs `NNN-kebab-case.md` using the next unused three-digit number.
- Never renumber an ADR or reuse its identifier.
- Start from `TEMPLATE.md` and default the status to `Proposed`.
- Mark an ADR `Accepted` only when its decision is approved and concrete.
- Do not silently rewrite an accepted decision. Create a new ADR, mark the old one
  `Superseded`, and add reciprocal links.
- Minor corrections that do not change the decision or its consequences may be made
  in place.

## Records

| ADR | Status | Decision |
|---|---|---|
| [001](001-routing-and-lane-ownership.md) | `Accepted` | Routing and lane ownership |

## Backlog

These records should be created only with the PR that makes the corresponding
decision concrete. Their outcomes are intentionally unspecified here.

| Planned ADR | Topic |
|---|---|
| `002-structured-fact-capability-policy.md` | Structured-fact capability policy |
| `003-degradation-semantics.md` | Per-lane and overall degradation semantics |
| `004-citation-policy.md` | Citation and grounding policy |
| `005-filing-anchor-and-amendments.md` | Filing anchor and amendment precedence |
