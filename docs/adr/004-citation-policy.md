# ADR 004: Claim-level citation integrity

- Status: Accepted
- Date: 2026-09-03
- Extends: ADR 003 without changing evidence-lane degradation semantics

## Decision

Filing answers contain explicit analyst-internal claims. Each claim has a unique
claim_id, a claim_type, text, context_ids and optional metric_id. Types are
structured_numeric, kb_numeric, calculation, narrative and attribution. Numeric
KB extraction has its own type rather than pretending all numbers originate in
structured facts. The model never supplies citation provenance.

Every substantive claim needs usable analyst-visible evidence. Structured numeric
claims require a typed structured fact with matching metric_id; KB numeric,
narrative and attribution claims require text/table evidence. Calculation claims
cite input evidence while the existing calculator-result checks remain in force.
This checks declared bindings and compatibility, not natural-language entailment.

Public filing answer prose is composed from validated claim text, not an
independent unvalidated answer field. Comparison rows reference claim_id; their
label and value must occur verbatim in its text, and an explicit target must match
cited evidence. Row citations are derived from that claim. An insufficient_data
response with no claims gets a fixed non-factual limitation message; positive
claims in an insufficient-data response still require validation.

Missing/invalid bindings share the existing analyst retry budget. Successful
calculator history is preserved across citation repairs. After exhaustion,
analyst.ok is false, status is grounding_error, and the orchestrator reports
failed with failure_stage=analyst. Existing evidence lanes and degradation notices
are not rewritten. Extra invalid references can be removed with an explicit issue
only if every claim retains compatible support; an unsupported claim is never
silently dropped to make the answer succeed.

Public used_context_ids and citations remain, derived from validated claims.
Citation SourceRef is copied from the corresponding context. Original candidates,
repair decisions, the exact context limit and visible IDs remain in analyst trace.

## Compatibility and scope

Legacy FinalAnswer payloads still parse, but filing status=ok without claims now
retries and fails closed. Existing API fields remain; claims, claim_id on rows and
grounding trace fields are additive. Non-filing answers do not require claims.
Consumers must recognize analyst grounding_error; top-level statuses and exit-code
semantics are unchanged. No retrieval, capability, resolver, period or provider
policy is changed. A rollback reverts runtime/enforcement commits, never deletes
historical evidence; rolling back restores the explicitly weaker PR5 contract.

## Evaluation

The frozen 37-case candidate-injection matrix runs the real analyst graph without
a live model. Its oracle is independently implemented and never imports runtime
grounding helpers. Route-aware evaluation also compares final claims, citations,
provenance and prose against visible context and original candidates. Structural
violations in successful outputs are critical and shadow-fail at analyst stage.

An independent frozen rubric measures semantic support secondarily. Correct type
and metric identity do not prove entailment. Missing legacy bindings remain absent;
flat citation lists cannot be upgraded into observed claim bindings. Semantic
annotations are not judge measurements. Report denominators and assessment coverage
and do not market structural improvements as measured semantic improvements.

The PR6 matrix is the PR-specific gate. The unchanged 14-case PR5 matrix and
15-case live route gate remain separately required and reported; inherited live
failures are never relabeled as passes.
