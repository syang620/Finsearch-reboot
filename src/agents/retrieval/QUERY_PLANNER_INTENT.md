# Query Planner Design Intent

## What this module is trying to be

Despite the current filename `query_planner.py`, this module is **not** meant to be a generic query planner that emits standalone planner JSON and stops.

It should be treated as a **bounded retrieval controller** (or a small retrieval subgraph) for SEC-filings RAG.

Its job is to:

1. take a precomputed retrieval plan plus resolved filing targets,
2. generate good retrieval queries for **one job + one target** at a time,
3. call the SEC retrieval MCP tool,
4. review whether the retrieval is good enough,
5. optionally retry **once** with improved queries,
6. return an auditable retrieval trace for downstream answer generation.

In other words:

> This module is the layer between planner output and raw retrieval.
> It is responsible for query formation, retrieval execution, lightweight review, and one bounded retry.

---

## What this module is **not** supposed to do

This module should **not**:

- answer the final user question,
- change deterministic filing metadata like `ticker`, `fiscal_year`, or `form_type`,
- act like a broad autonomous agent,
- perform unlimited self-reflection loops,
- own business-answer synthesis,
- depend on complex domain-specific deterministic heuristics.

This module is intentionally narrow and controlled.

---

## Core mental model

Think of this file as a **hybrid RAG retrieval workflow**:

- deterministic metadata comes from upstream
- the LLM helps with query generation / query revision
- the retrieval tool does evidence lookup
- a review step decides whether the first retrieval is sufficient
- at most one retry is allowed

So the design is closer to:

**planner handoff -> retrieval controller -> retrieved evidence trace**

than to:

**general agent that can do anything**

---

## Inputs this module expects

This module assumes upstream components have already produced:

- `original_user_query`
- `clarification_history` (optional)
- `retrieval_plan.jobs`
- `targets`

Each run inside this module is effectively over:

- one retrieval **job**
- one filing **target**

The target metadata should already be resolved upstream and treated as fixed.

---

## High-level workflow

For each applicable **job x target** pair:

1. **Normalize the job**
   - Determine the job type and goal.
   - Determine deterministic doc-type hints if needed.

2. **Build the first retrieval prompt input**
   - Use the original user query
   - Use the clarified goal
   - Use fixed target metadata
   - Include suggested query cues
   - Include deterministic doc-type hints if appropriate

3. **Run the retrieval agent (initial phase)**
   - The retrieval agent does **not** answer the question.
   - It calls `sec_retrieve_tables` with:
     - `queries`
     - `doc_types` (if relevant)
     - fixed target metadata
     - retrieval parameters like `top_k` and `min_total_score`

4. **Capture retrieval output and compact it**
   - Save the exact request used
   - Save the retrieval result
   - Build a compact summary of the top returned evidence
   - Record the attempt in traceable form

5. **Review the retrieval result**
   - A reviewer LLM decides whether the retrieval is good enough to stop,
     or whether one retry is worth attempting.
   - The reviewer should return structured control fields such as:
     - `action` = `accept` or `retry`
     - `reason`
     - `rewrite_notes`
   - The reviewer should **not** own the actual retry-query generation.

6. **If retry is needed, run the retrieval agent again (review/retry phase)**
   - The retrieval agent receives:
     - the previous request
     - the previous retrieval result
     - the reviewer feedback
   - The retrieval agent rewrites the queries using the reviewer guidance.
   - It calls `sec_retrieve_tables` one more time.

7. **Stop after one retry at most**
   - Return the final run payload with attempts, prompts, raw outputs, compact results, and retrieved evidence.

---

## Intended control flow

Conceptually, the workflow should look like this:

```text
Human / upstream state
    -> retrieval job + fixed target
    -> retrieval agent (phase = initial)
    -> sec_retrieve_tables
    -> retrieval result
    -> reviewer LLM
        -> accept  -> stop
        -> retry   -> retrieval agent (phase = review)
                     -> sec_retrieve_tables
                     -> final result
```

If implemented in LangGraph, the retrieval step can still use the standard pattern:

```text
call_model
  -> AIMessage (maybe tool call)
  -> ToolNode executes retrieval tool
  -> ToolMessage added to state
  -> call_model again
```

But architecturally, this should still be viewed as a **custom retrieval workflow**, not a broad autonomous agent.

---

## Prompt architecture the code should converge to

### 1) Retrieval agent prompt

Use **one unified retrieval-agent prompt** with an explicit `phase` field.

The retrieval agent should:

- own query generation for the initial pass,
- own query revision on retry,
- place retrieval strings directly into the tool-call `queries` field,
- keep target metadata fixed,
- respect deterministic doc-type hints when applicable,
- never answer the user question.

Phases:

- `phase = initial`
  - generate first-pass queries and call retrieval tool
- `phase = review`
  - inspect reviewer feedback
  - if reviewer says `accept`, do not call tool again
  - if reviewer says `retry`, revise queries and call tool once more

### 2) Reviewer prompt

Use a separate reviewer prompt.

The reviewer should:

- judge whether the retrieval is good enough for downstream use,
- return structured control output,
- provide flexible `rewrite_notes`,
- avoid owning the retry-query rewrite logic.

Recommended reviewer output shape:

```json
{
  "action": "accept" | "retry",
  "reason": "short explanation",
  "rewrite_notes": "free-form guidance for how the retrieval agent should improve the next query set"
}
```

This keeps the control interface structured while allowing flexible guidance.

---

## Why the retry-query rewrite belongs to the retrieval agent

The retrieval agent should own query rewriting because:

- it already owns query generation,
- it already knows the target metadata and retrieval constraints,
- it already knows how to package arguments for the retrieval MCP tool,
- it keeps the reviewer simpler and more reusable,
- it avoids turning the reviewer into both judge and retriever.

So the reviewer should diagnose.
The retrieval agent should revise.

---

## Minimal deterministic logic is enough for v1

Do **not** overbuild domain-specific heuristics right now.

For v1, deterministic logic should stay minimal:

- retriever error
- zero results
- max retries reached
- malformed or missing tool call

Everything else can be handled by the reviewer + retrieval-agent retry loop.

---

## What success looks like

A successful version of this module should:

- preserve deterministic filing metadata,
- generate cleaner retrieval queries than the raw user wording,
- retrieve evidence for one job-target pair at a time,
- make the retry decision in a lightweight structured way,
- allow one improved retry when helpful,
- produce an auditable retrieval trace,
- hand downstream components a better evidence package.

---

## Practical summary for future editors / Codex

If you are modifying this module, keep the following intent in mind:

1. This is a **retrieval controller**, not a final-answer agent.
2. Upstream planning resolves **what evidence to retrieve**.
3. This module resolves **how to retrieve it well**.
4. The retrieval agent owns **query generation and query revision**.
5. The reviewer owns **accept vs retry** plus **rewrite guidance**.
6. Target metadata remains fixed.
7. At most one retry.
8. Favor clarity, auditability, and bounded behavior over cleverness.

---

## Pseudocode version

```python
for each job in retrieval_plan:
    for each applicable target in targets:
        # first pass
        first_input = {
            "phase": "initial",
            "original_user_query": ...,
            "job": ...,
            "target": ...,
            "suggested_query_cues": ...,
            "required_doc_types": ...,
        }

        attempt_1 = retrieval_agent(first_input)   # tool call -> sec_retrieve_tables

        review = reviewer({
            "job": ...,
            "target": ...,
            "request_used": attempt_1.request,
            "retrieval_result": attempt_1.compact_result,
            "attempts_remaining": 1,
        })

        if review.action == "accept":
            final_attempt = attempt_1
        else:
            retry_input = {
                "phase": "review",
                "job": ...,
                "target": ...,
                "request_used": attempt_1.request,
                "retrieval_result": attempt_1.compact_result,
                "review_feedback": review,
            }
            final_attempt = retrieval_agent(retry_input)

        store_trace(job, target, final_attempt)
```

---

## Recommended interpretation of the current file

The current `query_planner.py` should be refactored toward this design and mentally understood as:

**`retrieval_controller.py` hidden inside a file still named `query_planner.py`**

That framing is the most important thing future edits should preserve.
