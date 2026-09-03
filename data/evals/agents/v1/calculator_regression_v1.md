# Calculator provenance regression v1

Freeze before changing post-PR6 runtime `6155034e4c1864e187a3318619524d9192024a78`.
The 20-case dataset and replay harness are the focused follow-up gate. Expected
results are fixture constants; the harness does not import the runtime selector
or equivalence helpers. Successful outputs also pass the independent PR6 oracle.

The source failure is `AGENT_V1_ANALYST_001` in
`artifacts/evals/agents/v1/baselines/0f8e477/per_query.jsonl`, SHA-256
`4840d2f646d9577d658bdb47ba7527f560bc284052c1357cffa3f757cb842836`.
The preserved trace includes repeated growth calculations, intermediate values,
and a final candidate that omits its calculation. Earlier candidates also contain
independent grounding errors. The live artifact retains call arguments, not the
complete raw calculator response transcript.

`CALC_LIVE_HISTORY_EXPLICIT` and `CALC_LIVE_OMISSION_REPAIR` are **minimal derived
reproductions**, not exact transcript replays: they preserve captured expression,
input and omission shapes, inject the known deterministic results, and use valid
claim bindings to isolate calculation selection. The growth value is
`((96169 - 85200) / 85200) * 100 = 12.874413145539906`; the intermediate delta is
`10969`. Original live evidence is unchanged. A focused pass does not establish
that the original model transcript would pass unchanged or guarantee 15/15 live.

Desired policy:

- Repeated executions with the same parsed expression, exact normalized referenced
  inputs and exact finite result constitute one logical computation. Preserve raw
  call history and return one original trusted representative.
- Distinct expressions/inputs and conflicting results are not equivalent merely
  because numeric answers coincide or fall within the existing final-result tolerance.
- A missing/ambiguous final selection may use the existing bounded answer retry,
  with successful history visible, without re-executing tools. It must identify
  the selected expression, variables and result; do not choose the last result or
  infer a selection from prose.
- Unrepaired ambiguity still fails closed. Insufficient-data exemptions, tool
  failures, numeric mismatch checks and PR6 grounding requirements remain intact.

Coverage includes duplicate/normalized/zero results, harmless unused variables,
explicit and omitted selection, repair and exhaustion, equal-result collisions,
conflicting results, mismatched inputs, invalid expressions, no tool execution,
insufficient data, calculation claims on extraction tasks, and grounding failure.
Candidate/tool-call counts and explicit repair feedback are part of expectations.
No provider tuning, retrieval changes, resolver work, new dependencies or changes
to the frozen grounding/degradation/live datasets belong in this follow-up.
