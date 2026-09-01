# Agent Evaluation

The route-aware v1 evaluator is the end-to-end architecture gate for the current
`kb`, `structured_fact`, and `hybrid` runtime. The original v0 evaluator remains
available unchanged for historical reproducibility.

## Route-aware v1

Default dataset:

- `data/evals/agents/v1/agent_eval_routing_v1.jsonl`

Canonical release baseline: evaluated commit
`6aae2651518e4495c947e393ed78978b515bd482`; immutable artifacts and full run
provenance are recorded in `docs/EVALUATION_BASELINES.md` under
“Route-Aware Agent E2E v1 — 2026-08-26.”

The PR4 native-structured-evidence diagnostic baseline was evaluated at
`523ade1a17dce890a35f20acdd0b5ead537aed89`. Its immutable artifacts, typed-evidence
coverage, live-gate result, and limitations are recorded in
`docs/EVALUATION_BASELINES.md` under “Native Structured Evidence and Provenance —
2026-09-01.”

The initial PR3 structured-fact capability baseline was evaluated independently at
`6b6b6173bac9045b03ad2292910b5acfd51740c8`; review fixes were verified at
`5a93171562dce106d1dd45cfe0c80aa5c01628ac` and
`af15c3c0acc301790d74187dcfacdc11c3dabe98`, with final verification at
`a3af6f08cf1b28c6a53ca52f4c58747ad62c4a99` and
`4c94cffbb8fda68196ce9a0101cc07c9b0876c9d`; the latest review fixes were verified at
`ced41d0718f46bc4e24dd018c830492d3db96ca4` and
`6adf1e259434f3b5034c8bac4d3122d6873a7796`, with final verification at
`79bcb8e0c6d6e3199bc99b326ada3084e27d3986` and
`f9a07e8b02ba3dabfd881bcada14139077e22614`, with canonical concept verification at
`5d0dc4647efff81e07d576d430ac45a8b6c910c9` and final boundary verification at
`3ef13f3599bab9552d698c6d2270eec05cc0bb45`, with clarification/adjacent-clause
verification at `e7dae4662006a2e0cef0d9adc33c1b1857b7e93a`, against the same frozen dataset. The
final semantic-context verification was recorded at
`b1f9ebcbbcb004de1b7901c60ac88e7c641c12e0`, followed by resumed-execution
verification at `eb1bc4b3cea26640604cdd6b6f5dac02cce5fe20` and final annual-boundary
verification at `95a0efa085a54a884c8d43442b914fe59c3502de`, followed by final metadata and
omitted-sibling verification at `011eb6ae5c6f16f6d5a8662c4a3c4d3f96bfc10c` and final semantic-coverage
verification at `992e715c85b7cbb8ba0f5bd772c5d6974135eb76`, followed by final concept-clause
verification at `8b9f8ec4a32b22976a2a9ccd0de09d0b2674712d` and final narrative-boundary
verification at `f301f32742b85faba079031c4d3381ec15a2e51f`, followed by final uncovered-clause
verification at `f47576bfdf775ddc21ef90267bea03318ef4e397` and final omitted-supported-clause
verification at `089763d702020fda627303a09004b937dbfdbbbe`, followed by final clarified-sibling
verification at `21dd490e287f208bff92c8d4613db0009a93d8c0` and final occurrence/period
verification at `dfd3ffe997fbdb55173ff5878ddd7cd8f0160f2a`, followed by final scoped-fallback
verification at `7791e64c586a4bacfb6e3a46e8f7b4c8aac31ae1` and final original-query agreement
verification at `efb5e4251dd97af5165b24850aee06a4b18c542b`, followed by final deterministic-context
verification at `5df0d19771015908abdca2dc949189418bc26016` and final period-boundary
verification at `8ee0d1e0c7e3ec840ff7ccf149a535433d6ca9d4`.
The adversarial before/after artifacts, metric definitions, and planner P0 gate are recorded in
`docs/EVALUATION_BASELINES.md` under “Structured-Fact Capability Policy v1 —
2026-08-30.”

Run deterministic architectural checks:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v1.py \
  --analyst-model ollama/qwen2.5:14b-instruct
```

Add optional RAGAS diagnostics:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v1.py \
  --enable-ragas \
  --enable-context-precision \
  --enable-context-recall
```

V1 checks the reported runtime status separately from the evaluation-only effective
status:

```json
{
  "reported_status": "completed",
  "effective_status": "degraded",
  "effective_status_source": "evaluator_derived"
}
```

Until the runtime exposes first-class lane summaries, v1 derives KB and structured
lane outcomes from the planner, retrieval result, structured-fact results, and
analyst result. Once runtime lane summaries are available, they become authoritative
while the evaluator continues deriving a shadow result and reports consistency.

The v1 evaluator requests the opt-in orchestration evidence trace. Its only payload
is the final serialized `AnalystPacket`; semantic contexts are derived from that
packet in analyst-visible order. The analyst and evaluator share the same current
five-context visibility limit and the same deterministic renderer for native
structured facts. Semantic context kinds distinguish `structured_fact` from KB
`table` and `text` evidence.

PR4 diagnostics count successful structured executions, typed and analyst-visible
typed contexts, synthetic structured-text contexts, and coverage of metric ID,
finite numeric value, accession number, report date, filed date, and source URL.
These are evidence-contract diagnostics; they do not change the deterministic score
formula or release thresholds.

V1 outputs:

- `artifacts/evals/agents/v1/per_query.jsonl`
- `artifacts/evals/agents/v1/summary.json`
- `artifacts/evals/agents/v1/errors.jsonl`

The per-query artifact preserves expected behavior, raw stage outputs, reported and
effective status, runtime and derived lane summaries, structured outcomes, analyst
evidence, timings, deterministic checks, and errors. Any explicitly required route,
lane, structured outcome, status, failure stage, analyst status, calculator use, or
positive citation minimum is release-critical.

Clarification cases stop at the initial `interrupted` result. V1 does not resume the
run or evaluate a multi-turn conversation.

The live v1 dataset favors stable success, control-flow, and analyst-behavior cases.
Forced KB/structured degradation combinations are authoritative deterministic unit
fixtures so SEC data and capability-policy changes do not make the release gate
artificially brittle.

## Legacy v0

This evaluator measures end-to-end multi-agent performance:

- Planner (intent/routing/metadata)
- Retrieval handoff quality
- Analyst behavior (tool use, computation, citations)
- Optional RAGAS LLM-judge metrics

> **Legacy evaluator warning:** This evaluator was designed around the legacy
> KB-oriented filing path and is not route-aware.
>
> - **Route logic:** It classifies `filing_fact` or `filing_calc` with
>   `retrieval_needed=False` as `RETRIEVAL_ROUTING_INVALID`, so valid
>   `structured_fact` behavior fails by definition.
> - **Semantic context:** It derives RAGAS contexts only from KB retrieval output and
>   excludes structured-fact evidence.
>
> Consequently, its overall score is diagnostic only and is not a valid multi-lane
> release gate. Use route-aware v1 for current architecture decisions.

### Inputs

Default eval set:

- `data/evals/agents/v0/agent_eval_v0.jsonl`

Expected row shape (simplified):

```json
{
  "id": "AGENT_E2E_001",
  "user_query": "What was Apple's total debt ... 2024?",
  "gold_answer": "Apple's total debt ...",
  "expected": {
    "intent": "filing_calc",
    "retrieval_needed": true,
    "ticker": "AAPL",
    "fiscal_year": 2024,
    "form_type": "10-K",
    "must_use_financial_evaluator": true,
    "expected_metric": "total debt",
    "expected_min_citations": 1
  }
}
```

### Run

Deterministic + RAGAS:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --eval-path data/evals/agents/v0/agent_eval_v0.jsonl \
  --out-dir artifacts/evals/agents/v0 \
  --analyst-model qwen3:14b
```

Deterministic only (skip RAGAS):

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --disable-ragas
```

Enable context RAGAS metrics explicitly:

```bash
PYTHONPATH=src python scripts/evals/agents/eval_agents_v0.py \
  --enable-context-precision \
  --enable-context-recall
```

### Outputs

- `artifacts/evals/agents/v0/per_query.jsonl`
- `artifacts/evals/agents/v0/summary.json`
- `artifacts/evals/agents/v0/errors.jsonl`

### Deterministic checks

Per query:

- contract validity
- intent match
- retrieval_needed match
- metadata match (ticker/year/form where specified)
- financial_evaluator tool-use match
- compute metric match
- citation minimum match

A weighted deterministic score is produced per query and aggregated.

### RAGAS checks

When enabled, the evaluator computes:

- `answer_relevancy`
- `faithfulness`
- optional: `context_precision`, `context_recall`

RAGAS errors are captured in `errors.jsonl` and reflected in gate decisions.

### Environment

Agent runtime dependencies:

- Qdrant + indexed corpus for retrieval path
- Ollama models for planner/analyst

RAGAS/Ollama judge:

- `RAGAS_OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `RAGAS_JUDGE_MODEL` (default `llama3.1:8b-instruct`)
- `RAGAS_EMBED_MODEL` (default `nomic-embed-text`)

### Notes

- Use deterministic metrics for architecture gating and fast iteration.
- Use RAGAS runs as diagnostic semantic quality checks.
- Do not use v0 scores for multi-lane release decisions.
- Keep `gold_answer` non-empty in dataset rows; RAGAS quality degrades otherwise.
