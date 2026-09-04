# Analyst output-contract repair v1

Freeze these sources, 20 scenarios and harness before extending calculator
implementation `6f4387aa1a30f5a3fac7f88cadd44f4b4c3b1b54` (evidence head
`f1f1225c5b8d4420cbbe1cfc325d1822291b4ea5`). Repair model-output contract
violations, not evidence truth. PR6 validation remains unchanged.

`analyst_output_repair_sources_v1.json` copies the analyst packets and last raw
FinalAnswer arguments from the four critical rows in
`artifacts/evals/agents/v1/baselines/6f4387a/per_query.jsonl`:

- `AGENT_V1_SF_004`: missing required `compare_rows`.
- `AGENT_V1_ANALYST_001`: missing required `status`, with calculation retained.
- `AGENT_V1_HYBRID_002`: row label case differs from its bound claim text.
- `AGENT_V1_HYBRID_003`: two row labels differ from their bound claim text.

The original raw measurement is immutable. Each calculation-bearing source uses
one synthetic successful calculator response with the captured final expression,
inputs and deterministic result; these are minimized replays, not full original
interleaved model/tool transcripts. Corrected candidates add the missing field or
change only row labels to existing verbatim substrings; they do not rewrite
factual prose. Semantic entailment and answer completeness are not measured.

The model stub emits the frozen next candidate unconditionally. It does not
interpret feedback, import a runtime repair helper, or decide success using the
implementation. Expectations separately measure:

- Terminal status/error, candidate counts and fail-closed controls.
- Precise field/row/claim feedback before a corrected candidate is injected.
- Original model claims, calculator result and raw history preservation.
- Independent PR6 finalized-output integrity on every success.

Existing retries can already accept injected valid corrections. Therefore a
feedback-contract improvement must not be reported as newly achieved terminal
repair or evidence of live-model reliability. The unchanged full live gate is
required after implementation freeze, exactly once.

Each live shape has corrected and persistent-invalid cases. Additional controls
cover required nested context IDs, all-unknown references, wrong evidence type,
wrong metric, unknown claim type, unbound rows, wrong targets, sequential schema
then row repairs within the existing budget, budget exhaustion, a valid unchanged
answer, and schema repair followed by unsupported evidence. No fields are filled
silently; persistent violations must remain errors. No retrieval, lane, resolver,
provider, retry-budget, threshold or grounding-policy changes are authorized.
