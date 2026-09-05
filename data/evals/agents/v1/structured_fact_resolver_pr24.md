# PR7 resolver characterization

`structured_fact_resolver_pr24.json` contains observed outputs from merged PR24
`2860c430036f3fa9e9488663fed9d086a03820af`, captured before resolver extraction.
The capture script verifies the original source bytes against that commit and
refuses to replace the snapshot. It never imports the new resolver.

Every case records the complete isolated resolution (status, metric ID, ticker,
year, selected target, exact reason and eventual SEC metric arguments), followed
by the actual capability decisions, legacy execution results and ordered SEC
calls with the real capability gate enabled. A recording client returns a marked
synthetic response; this measures resolution and orchestration, not SEC data.
The isolated observation explicitly permits each request so missing-input and
ambiguity behavior remains visible even when capability policy rejects it.

Coverage includes every registry ID, label and resolver alias, exact/phrase
matching, normalization, precedence, ambiguity, missing inputs, defensive partial
dictionaries, multiple and conflicting targets, first-match and single-target
fallback, nonannual rejection, unsupported queries, mixed request order and tool
errors. Expected outputs come from execution of the old path, not handwritten
expected decisions or shared runtime resolver helpers.

PR7 owns existing target selection and returns the selected target unchanged as
opaque metadata. It does not interpret `form_type`; filing-form eligibility stays
with the orchestrator/capability guard and filing semantics remain PR8 work.
Already-validated contract objects may be accepted by the new API, but the existing
dictionary path must not gain validation or normalization during extraction.

Release rule: exact frozen parity is required. Any resolver, routing or tool-call
argument mismatch blocks PR7. After regression suites, freeze the implementation,
review it and run the unchanged 15-case live gate once. Preserve any failures;
unrelated analyst grounding failures require a new explicit PR7 exception before
merge. `PR24-GROUNDING-001` does not apply, and failures must not prompt stochastic
grounding tuning or expanded PR7 scope.

Capture command (from the merged runtime; output is an apply_patch document):

```sh
PYTHONPATH=.:src conda run -n finsearch-arm --no-capture-output python scripts/evals/agents/capture_resolver_pr24.py
```

Python and environment-only pytest versions and source hashes are embedded in the
snapshot. Existing dependency manifests and historical evidence remain unchanged.
