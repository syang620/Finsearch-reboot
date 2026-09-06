# ADR 005: Original-as-filed identity and annual period eligibility

- **Status:** `Accepted`
- **Date:** `2026-09-05`
- **Supersedes:** None
- **Superseded by:** None

## Context

The owner approved original-as-filed PR8 semantics before implementation.
Merged PR25 (`f09e884`) prefers original filings but admits facts by accession OR
end date, discards start dates, and uses filing metadata to rank period candidates.
Those rules can admit a quarter, a comparative period or an amendment's value.

## Decision

The SEC metric tool owns selection. Resolver, routing, capability decisions and
tool arguments stay unchanged. Select exactly one original 10-K in the supplied
submissions for the requested report-date year. Duplicate identical filing rows
collapse. Multiple distinct original identities for that year are unavailable
(`FISCAL_YEAR_UNRESOLVED`), not arbitrarily ranked. Missing/invalid identifying
metadata is unavailable. No issuer fiscal-calendar inference or history fetch
is added. Known incompatible fiscal labels on exact-period original facts fail
closed with `FISCAL_YEAR_UNRESOLVED`; `fy` never establishes annuality.

Original-as-filed is authoritative, not latest-restated. A 10-K/A never replaces
or supplies missing original facts. Amendment-only supplied history returns
`not_found` / `AMENDMENT_ONLY`. Report-date coincidence is not filing identity.
An accession must be present and a report date must be a valid ISO date.

Amendment observations identify supplied filing metadata and/or supplied fact
candidates as sources. Coverage is **supplied inputs only**, never complete;
absence is `unknown`, never a claim that no amendment exists. No additional SEC
request, historical submissions pagination or amendment discovery is authorized.

An eligible fact requires exact original accession, exact intended end and
original 10-K form. Units follow the existing metric registry. Non-finite or
non-numeric values are ineligible. Instant facts must not carry a duration start.
Duration facts require strict ISO dates, start < end, and inclusive day count:

`duration_days = (end - start).days + 1`

Accepted bands are discrete, zero-tolerance lengths:

| Annual kind | Inclusive days | Supplied filing context |
| --- | --- | --- |
| 52-week | 364 | AAPL 2023-10-01 through 2024-09-28 |
| Standard | 365 | MSFT 2022-07-01 through 2023-06-30 |
| Leap | 366 | MSFT 2023-07-01 through 2024-06-30 |
| 53-week | 371 | AAPL 2022-09-25 through 2023-09-30 |

These periods were inspected in supplied original filing HTML before freeze.
The capture manifest records the source file hashes and extracted contexts.
No 330–400 or SEC Frames tolerance is used. Quarter/YTD/transition periods,
invalid/missing starts and adjacent unsupported lengths fail closed. `fp`, `fy`,
filed date, frame presence and input order do not override the predicate. Frame
presence alone no longer excludes an otherwise exact eligible companyfact.

For one registry concept, identical eligible duplicates collapse deterministically.
Different values OR different eligible starts represent competing interpretations:
return `ambiguous` / `CONFLICTING_ELIGIBLE_FACTS`. Do not round away conflicts.
Preserve existing registry concept priority; a conflicting preferred concept
cannot be bypassed via a lower-priority alias. Stable lexical metadata tie-breaks
are allowed only after value/period identity agrees.

Derived metrics require one accession, form, end and period type across selected
components; duration starts must be identical. Incompatible components return
`ambiguous` / `INCOMPATIBLE_COMPONENT_PERIODS` with no usable value. A conflict in
any consulted component blocks success even if another component survived.
Missing required components retain existing `partial`/`not_found` behavior.

Public statuses remain unchanged. Stable trace reasons distinguish:

- `AMENDMENT_ONLY`: original unavailable (`not_found`).
- `NO_ELIGIBLE_ANNUAL_PERIOD`: duration facts exist but none eligible (`not_found`).
- `CONFLICTING_ELIGIBLE_FACTS`: competing eligible facts (`ambiguous`).
- `FISCAL_YEAR_UNRESOLVED`, `NO_ORIGINAL_FILING`, `INVALID_FILING_METADATA`:
  unavailable original identity (`not_found`).
- `NO_ELIGIBLE_FACT`, `MISSING_COMPONENTS`, `OVERLAPPING_COMPONENTS`,
  `INCOMPATIBLE_COMPONENT_PERIODS`, `SELECTED_ORIGINAL_ANNUAL_FACT` and
  `SELECTED_ORIGINAL_INSTANT_FACT` distinguish remaining outcomes.

Selection trace records candidate provenance, eligibility/rejection reasons and
amendment observation scope. Fact/component result objects preserve `start_date`.
Successful atomic duration evidence also carries start date across PR4 admission;
legacy evidence without this additive field remains compatible. No grounding or
lane validation is weakened.

## Verification

Freeze complete old outputs, independent expected PR8 outputs, per-case allowed
field paths and reasons before changing runtime. The independent evaluator must
not import runtime selection helpers. Expected values, not merely allowlists,
are authoritative. Require approved-change accuracy, unchanged-case parity,
order invariance and provenance consistency of 100%, unexpected differences 0%.
Preserve old fixtures and failed gates. Then resolver 170, calculator 20, output
repair 20, grounding 37, degradation 14, SEC tests and full suite; clean SHA,
fresh review and one controlled unchanged live gate. No earlier exception carries.

## Consequences and alternatives

Strict original identity reduces availability when supplied history is incomplete.
Zero tolerance rejects unusual reporting calendars until explicitly supported.
Fiscal-label mapping, latest-restated views, history discovery and decimal-number
transport are deferred. Broad date tolerance, report-date fallback and automatic
amendment supersession were rejected because they can change evidence meaning.

## References

- [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Apple 2023 original 10-K](https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm)
- [Implementation roadmap](../IMPLEMENTATION_PLAN.md)
- [PR8 frozen cases](../../data/evals/agents/v1/filing_period_pr8.json)
