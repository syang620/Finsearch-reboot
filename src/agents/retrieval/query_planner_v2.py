from __future__ import annotations

"""
query_planner_v2.py

A bounded SEC-filings retrieval workflow that uses:
- one retrieval-agent LLM to generate tool-call args for retrieval
- one reviewer LLM to decide accept vs retry
- one SEC retrieval tool wrapped as a LangChain tool
- LangGraph to orchestrate the flow

Why this exists:
- keep ticker / fiscal_year / form_type deterministic
- let the retrieval agent focus only on retrieval, not answering
- allow one reviewer-guided retry without building a brittle finance-specific heuristic layer
- return an auditable per-run trace that downstream steps can consume

Expected input state shape:
{
    "original_user_query": str,
    "clarification_history": list[dict] | None,
    "retrieval_plan": {
        "jobs": [
            {
                "job_type": "metric_extract" | "narrative_extract" | "component_extract" | "fact_lookup",
                "goal": str,
                "target_id": int | None,
                "target_ids": list[int] | None,
            },
            ...
        ]
    },
    "targets": [
        {
            "target_id": 1,
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "form_type": "10-K",
            ...
        },
        ...
    ]
}

Expected retrieval client interface:
    await client.retrieve_tables(
        queries=[...],
        ticker="AAPL",
        fiscal_year=2024,
        form_type="10-K",
        doc_types=["text_chunk", "table", "table_row"] | None,
        top_k=3,
        min_total_score=0.0,
        timeout_s=30,
    )
"""

import json
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


_DEFAULT_FORM_TYPE = "10-K"
_MAX_QUERIES = 4
_ALLOWED_DOC_TYPES = {"text_chunk", "table", "table_row"}
_JOB_TYPES = {"fact_lookup", "metric_extract", "component_extract", "narrative_extract"}


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(raw)):
        char = raw[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start : index + 1])
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_reviewer_feedback(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, RetrievalReview):
        return raw.model_dump()

    if isinstance(raw, BaseMessage):
        raw = raw.content
    elif isinstance(raw, BaseModel):
        try:
            return raw.model_dump()
        except Exception:
            return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    if isinstance(raw, dict):
        candidate = raw
    elif isinstance(raw, str):
        candidate = _extract_first_json_object(raw)
    else:
        return None

    if not isinstance(candidate, dict):
        return None
    try:
        return RetrievalReview.model_validate(candidate).model_dump()
    except Exception:
        return None



DEFAULT_RETRIEVAL_AGENT_PROMPT_TEMPLATE = """
You are an SEC filings retrieval agent for exactly one retrieval job and exactly one filing target.

Your job is to retrieve evidence by calling `sec_retrieve_tables`.
You do not answer the user's question.
You do not change the target metadata.
You may make at most one retrieval tool call in the current step.

You will receive structured input containing:
- phase
- original_user_query
- clarification_history
- job
- target
- suggested_query_cues
- required_doc_types
- and, on retry steps, request_used, retrieval_result, and review_feedback

Core rules:
1. Never change ticker, fiscal_year, or form_type.
2. Preserve the retrieval intent of the job exactly.
3. Build 1 to 4 short retrieval-oriented queries.
4. Queries should be search-friendly phrases, not full natural-language questions.
5. Prefer filing-native terminology, note titles, line-item wording, section phrases, and close synonyms when helpful.
6. Do not include ticker, company name, fiscal year, or form type in the queries unless essential to the financial concept.
7. Apply required_doc_types as `doc_types` when provided.
8. Use `top_k=3` and `min_total_score=0`.
9. Pass the retrieval strings directly in the tool argument field `queries`.

How to behave by phase:

When phase == "initial":
- Call `sec_retrieve_tables` exactly once.
- Use the job goal, original user query, and suggested_query_cues to produce the best first-pass queries.
- Bias query wording by job_type:
  - metric_extract: prefer exact metric, line-item, statement, note, or accounting wording.
  - narrative_extract: prefer short topical phrases that surface explanatory text.
  - component_extract: prefer component or breakdown terminology.
  - fact_lookup: prefer one concise exact phrase for the target fact.

When phase == "review":
- First inspect review_feedback.
- If review_feedback.action == "accept", do not call the tool again. Return a short completion note only.
- If review_feedback.action == "retry" and attempts_remaining > 0, call `sec_retrieve_tables` exactly once with revised queries.
- Use review_feedback.rewrite_notes as guidance for how to improve the next query set.
- Treat rewrite_notes as hints for query revision, not as permission to change the retrieval goal.
- Keep the same target metadata and overall evidence need.

How to revise queries on retry:
- Start from the previous request_used.queries.
- Keep any strong useful anchor term when appropriate.
- Remove wording that appears too generic, too broad, or poorly aligned with the requested evidence.
- Incorporate the reviewer's rewrite_notes to make the new queries more retrieval-effective.
- Improve the queries by using sharper filing-native wording, likely section phrases, note titles, line-item language, or better topical anchors.
- Do not merely paraphrase the same weak query set.
- Keep the revised queries compact and retrieval-oriented.
- Preserve doc_types unless the reviewer or the input clearly indicates they were mismatched.

Output behavior:
- On retrieval steps, issue a tool call to `sec_retrieve_tables`.
- On non-retry review steps, do not call the tool and return only a short completion note.
""".strip()


DEFAULT_REVIEWER_PROMPT_TEMPLATE = """
You are the Retrieval Review Judge for an SEC-filings retrieval system.

Your job is NOT to answer the user’s question.
Your job is ONLY to decide whether the current retrieval attempt is clearly bad enough to justify one retry.

You have strong background knowledge in accounting and finance.
You understand how answers to financial questions are often supported by:
- financial statement tables
- note tables
- footnotes
- rollforwards
- breakdowns and component disclosures
- narrative accounting discussion
- MD&A
- risk factor discussion
- policy disclosures
- segment or geographic disclosures
- debt, liquidity, cash flow, revenue, expense, tax, and equity-related text or tables

Given retrieved tables or text, you can judge whether the retrieved content is likely sufficient to answer the question, or likely contains the right supporting evidence, even if the final answer is not explicitly written in the short summary.

Core principle:
- Default to ACCEPT.
- Retry is an exceptional action.
- If there is any plausible evidence that at least one retrieved item is the right source, close to the right source, or likely contains the needed supporting information, ACCEPT.
- Do NOT ask for retry just because you personally cannot derive the final answer from the compact summaries alone.
- Do NOT confuse “I am not fully sure” with “retrieval failed.”
- The retrieval only needs to be good enough for downstream reasoning or extraction. It does not need to be perfect, self-contained, or fully explicit in the short review artifact.

You are reviewing exactly one retrieval attempt for exactly one filing target.

You will receive structured input containing fields such as:
- original_user_query
- job
- target
- attempt_index
- attempts_remaining
- request_used
- retrieval_result

How to think about the evidence:
- The retrieval_result is only a compact review artifact.
- top_items summaries may be lossy, abbreviated, or incomplete.
- A summary that does not explicitly state the answer does NOT mean the underlying chunk or table is wrong.
- A result can be good enough even if the answer would require downstream reading, extraction, arithmetic, aggregation, comparison, or combining multiple rows/items.
- Your role is to judge retrieval adequacy, not final answer derivation.

Domain-aware review standard:
- Use your accounting and finance knowledge to judge whether the retrieved table/text is the kind of evidence that could answer the question.
- Recognize that the right evidence may appear indirectly:
  - as a note table rather than a face statement line
  - as a component breakdown rather than a final rolled-up total
  - as narrative disclosure rather than a table
  - as a policy note or risk discussion rather than an explicit answer sentence
- If the retrieved material looks like the right evidence source, ACCEPT.
- Do not require that the compact summary already contain the final answer in plain language.

High-bar retry policy:
Choose RETRY only if there is clear evidence that the retrieval attempt is unusable or materially off-target, such as:
- retriever error
- zero results
- all top items are clearly unrelated to the requested evidence
- the evidence type is clearly mismatched to the job in a way that makes downstream use unlikely
- the retrieval is so broad or generic that none of the top items appears plausibly usable
- the retrieved items are clearly about the wrong concept, wrong section, or wrong type of evidence

Choose ACCEPT in all of these situations:
- at least one top item appears plausibly on-topic
- a table/text looks like the likely source even if the answer is not explicit in the summary
- the retrieval seems partially relevant but still likely useful downstream
- the result is ambiguous
- you are uncertain whether retry would materially improve quality
- the evidence appears likely to answer the question with downstream extraction or reasoning

Important bias:
- Prefer a false ACCEPT over a false RETRY.
- If unsure, ACCEPT.
- Only request RETRY when the current retrieval is clearly poor and retry has a strong chance of improving it.

How to judge by job type:
- metric_extract:
  ACCEPT if at least one item looks like a plausible source of the metric or ingredients needed to derive it, such as a relevant table, statement row, note disclosure, or accounting discussion.
  Do not require the exact value to be visible in the compact summary.
- narrative_extract:
  ACCEPT if at least one item appears to discuss the relevant topic, risk, policy, operation, or section.
- component_extract:
  ACCEPT if at least one item appears likely to contain the relevant breakdown, supporting components, or sub-items, even if the full decomposition is not obvious from the summary.
- fact_lookup:
  ACCEPT if at least one item appears likely to contain the fact or direct supporting evidence.

Important review rules:
- Do not change ticker, fiscal_year, or form_type.
- Keep the retrieval intent aligned to the original job and target.
- Use request_used and retrieval_result.top_items to judge whether the attempt is sufficient.
- If attempts_remaining is 0, choose ACCEPT unless there is a hard failure such as retriever error or zero results.
- Do not invent missing evidence.
- Do not recommend retry merely because a different query might be marginally better.
- Recommend retry only when the current result is clearly inadequate.

When action = "retry":
- Explain, in free-form language, why the current retrieval is clearly inadequate.
- Put that guidance in `rewrite_notes`.
- Focus on what is missing or clearly wrong.
- Keep the guidance short and concrete.
- Do not provide the final answer.

Return ONLY one JSON object with exactly this schema:
{
  "action": "accept" | "retry",
  "reason": "short explanation of the decision",
  "rewrite_notes": "free-form guidance for improving the next retrieval attempt; empty string if action is accept",
  "revised_doc_types": ["text_chunk" | "table" | "table_row"] | null
}
""".strip()


class RetrievalReview(BaseModel):
    action: Literal["accept", "retry"]
    reason: str
    rewrite_notes: str = ""
    revised_doc_types: Optional[List[Literal["text_chunk", "table", "table_row"]]] = None


class _RunGraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    review_feedback: Dict[str, Any] | None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        cleaned = _normalize_text(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def deterministic_doc_types_for_job(job_type: str) -> Optional[List[str]]:
    normalized = _normalize_text(job_type).lower()
    if normalized == "metric_extract":
        return ["table", "table_row", "text_chunk"]
    if normalized == "component_extract":
        return ["table", "table_row", "text_chunk"]
    if normalized == "narrative_extract":
        return ["text_chunk"]
    if normalized == "fact_lookup":
        return None
    return None


def _coerce_doc_types(raw: Any, fallback: Optional[List[str]] = None) -> Optional[List[str]]:
    if raw is None:
        return list(fallback) if fallback else None
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Sequence):
        items = list(raw)
    else:
        items = []

    cleaned = [
        _normalize_text(item).lower().replace(" ", "_")
        for item in items
        if _normalize_text(item)
    ]
    valid = [item for item in cleaned if item in _ALLOWED_DOC_TYPES]
    valid = _dedupe_keep_order(valid)
    if valid:
        return valid
    return list(fallback) if fallback else None


def _extract_summary(item: Dict[str, Any]) -> str:
    candidates = [
        item.get("summary"),
        item.get("text"),
        item.get("content"),
        item.get("snippet"),
        item.get("value"),
    ]
    for candidate in candidates:
        text = _normalize_text(candidate)
        if text:
            return text[:500]
    return ""


def _compact_retrieval_result(result: Dict[str, Any], *, top_n: int = 3) -> Dict[str, Any]:
    rows = list(result.get("top_tables") or result.get("results") or [])
    compact_rows: List[Dict[str, Any]] = []
    for row in rows[:top_n]:
        compact_rows.append(
            {
                "doc_id": row.get("doc_id"),
                "section_path": row.get("section_path") or row.get("section") or row.get("path"),
                "doc_type": row.get("doc_type"),
                "total_score": row.get("total_score") or row.get("score"),
                "summary": _extract_summary(row),
            }
        )

    return {
        "ok": bool(result.get("ok", False)),
        "error": _normalize_text(result.get("error")) or None,
        "queries_used": list(result.get("queries_used") or []),
        "num_results": len(rows),
        "max_total_score": result.get("max_total_score"),
        "metadata_used": dict(result.get("metadata_used") or {}),
        "top_items": compact_rows,
    }


def _build_attempt_log(*, attempt_index: int, request: Dict[str, Any], retrieval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "attempt_index": int(attempt_index),
        "request": dict(request),
        "retrieval": dict(retrieval),
        "retrieval_compact": _compact_retrieval_result(retrieval),
    }


def _seed_queries_for_job(
    *,
    goal: str,
    original_user_query: str,
    job_type: str,
    target: Dict[str, Any],
) -> List[str]:
    text = f"{goal} {original_user_query}".lower()
    seeds: List[str] = []

    if goal:
        seeds.append(goal)
    if original_user_query and original_user_query != goal:
        seeds.append(original_user_query)

    if "revenue" in text or "sales" in text:
        seeds.extend(["revenue", "net sales", "total net sales"])
    if "debt" in text:
        seeds.extend(["debt", "long-term debt", "current portion of long-term debt"])
    if "cash flow" in text:
        seeds.extend(["cash flow", "operating activities", "cash and cash equivalents"])
    if "supply chain" in text:
        seeds.extend(["supply chain", "component shortages", "manufacturing partners"])
    if "risk" in text and job_type == "narrative_extract":
        seeds.extend(["risk factors", "supply chain risk", "operations risk"])

    # remove target metadata from raw fallback seeds because retrieval filters are deterministic
    target_tokens = {
        _normalize_text(target.get("ticker")).lower(),
        _normalize_text(target.get("form_type")).lower(),
        str(target.get("fiscal_year")),
    }
    normalized = []
    for seed in _dedupe_keep_order(seeds):
        words = [w for w in seed.split() if w.lower() not in target_tokens]
        cleaned = " ".join(words).strip()
        if cleaned:
            normalized.append(cleaned)

    return _dedupe_keep_order(normalized)[:_MAX_QUERIES]


def _coerce_queries(raw: Any, *, fallback: List[str]) -> List[str]:
    if raw is None:
        return list(fallback)
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, Sequence):
        items = list(raw)
    else:
        items = []
    cleaned = _dedupe_keep_order([_normalize_text(item) for item in items])
    return (cleaned or list(fallback))[:_MAX_QUERIES]


def _render_prompt(system_prompt: str, prompt_input: Dict[str, Any]) -> str:
    return system_prompt.strip() + "\n\nActual input:\n" + json.dumps(prompt_input, indent=2, ensure_ascii=False)


def _select_targets_for_job(job: Dict[str, Any], targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {target.get("target_id"): target for target in targets if target.get("target_id") is not None}
    raw_ids = []
    if job.get("target_id") is not None:
        raw_ids.append(job.get("target_id"))
    if isinstance(job.get("target_ids"), Sequence):
        raw_ids.extend(job.get("target_ids") or [])
    if isinstance(job.get("applies_to_target_ids"), Sequence):
        raw_ids.extend(job.get("applies_to_target_ids") or [])

    selected: List[Dict[str, Any]] = []
    for raw_id in raw_ids:
        if raw_id in by_id:
            selected.append(by_id[raw_id])
    return selected or list(targets)


def _normalize_job(job: Dict[str, Any], *, original_user_query: str) -> Dict[str, Any]:
    job_type = _normalize_text(job.get("job_type")).lower() or "fact_lookup"
    if job_type not in _JOB_TYPES:
        job_type = "fact_lookup"
    goal = _normalize_text(job.get("goal")) or original_user_query
    return {
        **job,
        "job_type": job_type,
        "goal": goal,
    }


def _build_retrieval_output(*, state: Dict[str, Any], runs_payload: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    all_tables: List[Dict[str, Any]] = []
    all_queries: List[str] = []
    errors: List[str] = []
    max_total_score = None

    for run in runs_payload:
        final_retrieval = dict(run.get("final_retrieval") or {})
        all_tables.extend(list(final_retrieval.get("top_tables") or final_retrieval.get("results") or []))
        all_queries.extend(list(final_retrieval.get("queries_used") or []))
        error_text = _normalize_text(final_retrieval.get("error"))
        if error_text:
            errors.append(error_text)
        current_score = final_retrieval.get("max_total_score")
        if isinstance(current_score, (int, float)):
            score_value = float(current_score)
            if max_total_score is None or score_value > max_total_score:
                max_total_score = score_value

    targets = [dict(target) for target in (state.get("targets") or []) if isinstance(target, dict)]
    primary_target = next(
        (
            target
            for target in targets
            if _normalize_text(target.get("ticker")) and _normalize_int(target.get("fiscal_year")) is not None
        ),
        {},
    )
    deduped_queries = _dedupe_keep_order(all_queries)
    deduped_targets = [dict(target) for target in targets]

    return {
        "type": "retrieval",
        "ok": bool(all_tables) or not errors,
        "rerank_query": _normalize_text(state.get("original_user_query")) or (deduped_queries[0] if deduped_queries else ""),
        "original_user_query": _normalize_text(state.get("original_user_query")),
        "targets": deduped_targets,
        "metadata_used": {
            "ticker": _normalize_text(primary_target.get("ticker")) or None,
            "fiscal_year": _normalize_int(primary_target.get("fiscal_year")),
            "form_type": _normalize_text(primary_target.get("form_type")) or None,
            "original_user_query": _normalize_text(state.get("original_user_query")),
            "clarification_history": [
                dict(turn)
                for turn in (state.get("clarification_history") or [])
                if isinstance(turn, dict)
            ],
            "targets": deduped_targets,
            "retrieval_plan": dict(state.get("retrieval_plan") or {}),
            "planner_hints": dict(state.get("planner_hints") or {}),
            "job_runs": runs_payload,
            "retrieval_agent_model": model_name,
            "max_attempts": 2,
            "top_k": 3,
            "min_total_score": 0.0,
        },
        "retrieval_plan": dict(state.get("retrieval_plan") or {}),
        "retrieval_agent_model": model_name,
        "max_attempts": 2,
        "top_k": 3,
        "min_total_score": 0.0,
        "max_total_score": max_total_score,
        "queries_used": deduped_queries,
        "top_tables": all_tables,
        "job_runs": runs_payload,
        "error": "; ".join(_dedupe_keep_order(errors)) if errors else None,
        "trace": {"runs": runs_payload},
    }


class RetrievalWorkflowAgent:
    """Bounded retrieval controller for one retrieval plan over one set of targets."""

    def __init__(
        self,
        *,
        retrieval_llm: BaseChatModel,
        reviewer_llm: BaseChatModel,
        retrieval_agent_prompt: str = DEFAULT_RETRIEVAL_AGENT_PROMPT_TEMPLATE,
        reviewer_prompt: str = DEFAULT_REVIEWER_PROMPT_TEMPLATE,
        top_k: int = 3,
        min_total_score: float = 0.0,
        max_attempts: int = 2,
        timeout_s: int = 30,
    ) -> None:
        self.retrieval_llm = retrieval_llm
        self.reviewer_llm = reviewer_llm
        self.retrieval_agent_prompt = retrieval_agent_prompt
        self.reviewer_prompt = reviewer_prompt
        self.top_k = top_k
        self.min_total_score = min_total_score
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s

    async def _retrieve_with_client(
        self,
        *,
        client: Any,
        request: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            result = await client.retrieve_tables(
                queries=request.get("queries") or [],
                ticker=target["ticker"],
                fiscal_year=target["fiscal_year"],
                form_type=target["form_type"],
                doc_types=request.get("doc_types"),
                top_k=self.top_k,
                min_total_score=self.min_total_score,
                timeout_s=self.timeout_s,
            )
            if "queries_used" not in result:
                result["queries_used"] = list(request.get("queries") or [])
            if "metadata_used" not in result:
                result["metadata_used"] = {
                    "ticker": target["ticker"],
                    "fiscal_year": target["fiscal_year"],
                    "form_type": target["form_type"],
                    "doc_types": request.get("doc_types"),
                }
            return result
        except Exception as exc:
            return {
                "ok": False,
                "error": f"RETRIEVER_CALL_FAILED: {exc}",
                "queries_used": list(request.get("queries") or []),
                "metadata_used": {
                    "ticker": target["ticker"],
                    "fiscal_year": target["fiscal_year"],
                    "form_type": target["form_type"],
                    "doc_types": request.get("doc_types"),
                },
                "top_tables": [],
                "max_total_score": None,
            }

    def _build_first_prompt_input(
        self,
        *,
        state: Dict[str, Any],
        job_plan: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        suggested_queries = _seed_queries_for_job(
            goal=job_plan["goal"],
            original_user_query=_normalize_text(state.get("original_user_query")),
            job_type=job_plan["job_type"],
            target=target,
        )
        return {
            "phase": "initial",
            "original_user_query": _normalize_text(state.get("original_user_query")),
            "clarification_history": [
                dict(turn)
                for turn in (state.get("clarification_history") or [])
                if isinstance(turn, dict)
            ],
            "job": {
                "job_type": job_plan["job_type"],
                "goal": job_plan["goal"],
            },
            "target": target,
            "suggested_query_cues": suggested_queries,
            "required_doc_types": deterministic_doc_types_for_job(job_plan["job_type"]),
            "review_feedback": None,
        }

    def _build_retry_prompt_input(
        self,
        *,
        state: Dict[str, Any],
        job_plan: Dict[str, Any],
        target: Dict[str, Any],
        request: Dict[str, Any],
        result: Dict[str, Any],
        review_feedback: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        return {
            "phase": "review",
            "original_user_query": _normalize_text(state.get("original_user_query")),
            "clarification_history": [
                dict(turn)
                for turn in (state.get("clarification_history") or [])
                if isinstance(turn, dict)
            ],
            "job": {
                "job_type": job_plan["job_type"],
                "goal": job_plan["goal"],
            },
            "target": target,
            "suggested_query_cues": [],
            "required_doc_types": deterministic_doc_types_for_job(job_plan["job_type"]),
            "attempt_index": int(attempt_index),
            "attempts_remaining": max(self.max_attempts - int(attempt_index), 0),
            "request_used": {
                "queries": list(request.get("queries") or []),
                "doc_types": request.get("doc_types"),
                "top_k": self.top_k,
                "min_total_score": self.min_total_score,
            },
            "retrieval_result": _compact_retrieval_result(result),
            "review_feedback": review_feedback,
        }

    def _build_review_input(
        self,
        *,
        state: Dict[str, Any],
        job_plan: Dict[str, Any],
        target: Dict[str, Any],
        request: Dict[str, Any],
        result: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        return {
            "original_user_query": _normalize_text(state.get("original_user_query")),
            "job": {
                "job_type": job_plan["job_type"],
                "goal": job_plan["goal"],
            },
            "target": target,
            "attempt_index": int(attempt_index),
            "attempts_remaining": max(self.max_attempts - int(attempt_index), 0),
            "request_used": {
                "queries": list(request.get("queries") or []),
                "doc_types": request.get("doc_types"),
                "top_k": self.top_k,
                "min_total_score": self.min_total_score,
            },
            "retrieval_result": _compact_retrieval_result(result),
        }

    async def _run_single_target(
        self,
        *,
        state: Dict[str, Any],
        client: Any,
        job_plan: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        model_turns: List[Dict[str, Any]] = []
        reviewer_turns: List[Dict[str, Any]] = []
        required_doc_types = deterministic_doc_types_for_job(job_plan["job_type"])
        seed_queries = _seed_queries_for_job(
            goal=job_plan["goal"],
            original_user_query=_normalize_text(state.get("original_user_query")),
            job_type=job_plan["job_type"],
            target=target,
        )
        first_pass_input = self._build_first_prompt_input(state=state, job_plan=job_plan, target=target)

        @tool
        async def sec_retrieve_tables(
            queries: List[str],
            doc_types: Optional[List[str]] = None,
            top_k: int = 3,
            min_total_score: float = 0.0,
            ticker: Optional[str] = None,
            fiscal_year: Optional[int] = None,
            form_type: Optional[str] = None,
            reason: str = "",
        ) -> str:
            """Retrieve SEC filing evidence for the fixed job and target."""
            del top_k, min_total_score, ticker, fiscal_year, form_type
            attempt_index = len(attempts) + 1
            request = {
                "queries": _coerce_queries(queries, fallback=seed_queries),
                "doc_types": _coerce_doc_types(doc_types, fallback=required_doc_types),
                "reason": _normalize_text(reason),
            }
            retrieval = await self._retrieve_with_client(client=client, request=request, target=target)
            attempt_log = _build_attempt_log(
                attempt_index=attempt_index,
                request=request,
                retrieval=retrieval,
            )
            attempts.append(attempt_log)
            return json.dumps(
                {
                    **attempt_log["retrieval_compact"],
                    "attempt_index": attempt_index,
                    "attempts_remaining": max(self.max_attempts - attempt_index, 0),
                    "request_used": {
                        "queries": list(request.get("queries") or []),
                        "doc_types": request.get("doc_types"),
                        "top_k": self.top_k,
                        "min_total_score": self.min_total_score,
                    },
                    "reason": request.get("reason") or None,
                    "metadata_fixed": {
                        "ticker": target["ticker"],
                        "fiscal_year": target["fiscal_year"],
                        "form_type": target["form_type"],
                    },
                },
                ensure_ascii=False,
            )

        async def call_retrieval_agent(graph_state: _RunGraphState) -> Dict[str, Any]:
            attempt_index = len(attempts)
            if attempt_index == 0:
                prompt_input = first_pass_input
            else:
                last_attempt = attempts[-1]
                prompt_input = self._build_retry_prompt_input(
                    state=state,
                    job_plan=job_plan,
                    target=target,
                    request=last_attempt["request"],
                    result=last_attempt["retrieval"],
                    review_feedback=dict(graph_state.get("review_feedback") or {}),
                    attempt_index=attempt_index,
                )

            prompt = _render_prompt(self.retrieval_agent_prompt, prompt_input)
            messages_for_model = list(graph_state.get("messages") or [])
            prompt_msg = HumanMessage(content=prompt)
            messages_for_model.append(prompt_msg)
            try:
                response = await self.retrieval_llm.bind_tools(
                    [sec_retrieve_tables],
                    tool_choice="any",
                ).ainvoke(messages_for_model)
            except Exception as exc:
                error_msg = f"RETRIEVAL_LLM_CALL_FAILED: {type(exc).__name__}: {exc}"
                attempts.append(
                    _build_attempt_log(
                        attempt_index=attempt_index + 1,
                        request={
                            "queries": list(_coerce_queries(seed_queries, fallback=seed_queries)),
                            "doc_types": list(required_doc_types) if required_doc_types else None,
                            "reason": "retrieval workflow model call failed",
                        },
                        retrieval={
                            "ok": False,
                            "error": error_msg,
                            "queries_used": list(seed_queries),
                            "top_tables": [],
                            "metadata_used": {
                                "ticker": target.get("ticker"),
                                "fiscal_year": target.get("fiscal_year"),
                                "form_type": target.get("form_type"),
                                "doc_types": list(required_doc_types) if required_doc_types else None,
                            },
                            "max_total_score": None,
                        },
                    )
                )
                model_turns.append(
                    {
                        "attempt_index": attempt_index,
                        "phase": prompt_input.get("phase"),
                        "prompt_input": prompt_input,
                        "prompt": prompt,
                        "message": None,
                        "error": error_msg,
                    }
                )
                return {
                    "messages": [AIMessage(content=error_msg)]
                }
            model_turns.append(
                {
                    "attempt_index": attempt_index,
                    "phase": prompt_input.get("phase"),
                    "prompt_input": prompt_input,
                    "prompt": prompt,
                    "message": response,
                    "raw_output": getattr(response, "content", None),
                }
            )
            return {"messages": [prompt_msg, response]}

        async def fallback_seed_retrieve(_: _RunGraphState) -> Dict[str, Any]:
            request = {
                "queries": list(seed_queries),
                "doc_types": list(required_doc_types) if required_doc_types else None,
                "reason": "deterministic fallback because the retrieval agent produced no tool call",
            }
            retrieval = await self._retrieve_with_client(client=client, request=request, target=target)
            attempts.append(
                _build_attempt_log(
                    attempt_index=1,
                    request=request,
                    retrieval=retrieval,
                )
            )
            return {
                "messages": [
                    HumanMessage(content="Fallback retrieval executed because no retrieval tool call was emitted.")
                ]
            }

        async def review_last_attempt(_: _RunGraphState) -> Dict[str, Any]:
            last_attempt = attempts[-1]
            review_input = self._build_review_input(
                state=state,
                job_plan=job_plan,
                target=target,
                request=last_attempt["request"],
                result=last_attempt["retrieval"],
                attempt_index=len(attempts),
            )
            prompt = _render_prompt(self.reviewer_prompt, review_input)
            review_raw_output = None
            review_dict: Optional[Dict[str, Any]] = None

            try:
                structured_reviewer = self.reviewer_llm.with_structured_output(RetrievalReview)
                review_dict = _coerce_reviewer_feedback(await structured_reviewer.ainvoke(prompt))
            except Exception as structured_exc:
                try:
                    raw_review = await self.reviewer_llm.ainvoke(prompt)
                    review_raw_output = getattr(raw_review, "content", raw_review)
                    review_dict = _coerce_reviewer_feedback(raw_review)
                except Exception as fallback_exc:
                    review_raw_output = f"{review_raw_output or ''} [fallback_parse_error: {fallback_exc}]"
                    compact = _compact_retrieval_result(last_attempt["retrieval"])
                    hard_failure = bool(compact.get("error")) or int(compact.get("num_results") or 0) == 0
                    review_dict = RetrievalReview(
                        action="retry" if hard_failure and len(attempts) < self.max_attempts else "accept",
                        reason=f"reviewer_failed: {structured_exc}",
                        rewrite_notes=(
                            "Focus more tightly on the core evidence need and use filing-native terminology."
                            if hard_failure and len(attempts) < self.max_attempts
                            else ""
                        ),
                        revised_doc_types=None,
                    ).model_dump()

            if review_dict is None:
                compact = _compact_retrieval_result(last_attempt["retrieval"])
                hard_failure = bool(compact.get("error")) or int(compact.get("num_results") or 0) == 0
                review_dict = RetrievalReview(
                    action="retry" if hard_failure and len(attempts) < self.max_attempts else "accept",
                    reason="reviewer_failed: unable to parse reviewer output",
                    rewrite_notes=(
                        "Focus more tightly on the core evidence need and use filing-native terminology."
                        if hard_failure and len(attempts) < self.max_attempts
                        else ""
                    ),
                    revised_doc_types=None,
                ).model_dump()

            if review_dict is not None and "reason" in review_dict and review_dict["reason"] == "":
                review_dict["reason"] = "ok"

            if review_dict is None:
                review_dict = {
                    "action": "accept",
                    "reason": "reviewer_failed: no_review_output",
                    "rewrite_notes": "",
                    "revised_doc_types": None,
                }

            if isinstance(review_dict, dict) and review_dict.get("reason", "").startswith("reviewer_failed:"):
                if review_raw_output is not None:
                    review_dict["reason"] = f"{review_dict['reason']} | raw={str(review_raw_output)[:1800]}"

            reviewer_turns.append(
                {
                    "attempt_index": len(attempts),
                    "prompt_input": review_input,
                    "prompt": prompt,
                    "raw_output": review_raw_output,
                    "review": review_dict,
                }
            )
            return {"review_feedback": review_dict}

        def route_after_retrieval_agent(graph_state: _RunGraphState) -> str:
            messages = list(graph_state.get("messages") or [])
            last_message = messages[-1] if messages else None
            if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                return "tools"
            if not attempts:
                return "fallback_seed_retrieve"
            return "review_last_attempt"

        def route_after_review(graph_state: _RunGraphState) -> str:
            feedback = dict(graph_state.get("review_feedback") or {})
            if (
                feedback.get("action") == "retry"
                and len(attempts) < self.max_attempts
            ):
                return "call_retrieval_agent"
            return END

        graph = StateGraph(_RunGraphState)
        graph.add_node("call_retrieval_agent", call_retrieval_agent)
        graph.add_node("tools", ToolNode([sec_retrieve_tables]))
        graph.add_node("fallback_seed_retrieve", fallback_seed_retrieve)
        graph.add_node("review_last_attempt", review_last_attempt)

        graph.add_edge(START, "call_retrieval_agent")
        graph.add_conditional_edges(
            "call_retrieval_agent",
            route_after_retrieval_agent,
            {
                "tools": "tools",
                "fallback_seed_retrieve": "fallback_seed_retrieve",
                "review_last_attempt": "review_last_attempt",
                END: END,
            },
        )
        graph.add_edge("tools", "review_last_attempt")
        graph.add_edge("fallback_seed_retrieve", "review_last_attempt")
        graph.add_conditional_edges(
            "review_last_attempt",
            route_after_review,
            {
                "call_retrieval_agent": "call_retrieval_agent",
                END: END,
            },
        )

        compiled = graph.compile()
        try:
            final_state = await compiled.ainvoke({"messages": [], "review_feedback": None})
        except Exception as exc:
            workflow_error = f"RETRIEVER_GRAPH_FAILED: {type(exc).__name__}: {exc}"
            failure_request = {
                "queries": list(seed_queries),
                "doc_types": list(required_doc_types) if required_doc_types else None,
                "reason": "retrieval workflow execution failed",
            }
            attempts.append(
                _build_attempt_log(
                    attempt_index=1,
                    request=failure_request,
                    retrieval={
                        "ok": False,
                        "error": workflow_error,
                        "queries_used": list(failure_request["queries"]),
                        "top_tables": [],
                        "metadata_used": {
                            "ticker": target.get("ticker"),
                            "fiscal_year": target.get("fiscal_year"),
                            "form_type": target.get("form_type"),
                            "doc_types": failure_request.get("doc_types"),
                        },
                        "max_total_score": None,
                    },
                )
            )
            return {
                "job": job_plan,
                "target": target,
                "attempts": attempts,
                "model_turns": [
                    {
                        **turn,
                        "message": None,
                        "error": turn.get("error") or str(exc),
                    } for turn in model_turns
                ],
                "reviewer_turns": [
                    {
                        "attempt_index": 1,
                        "prompt_input": {
                            "reason": workflow_error,
                        },
                        "prompt": "",
                        "raw_output": workflow_error,
                        "review": {
                            "action": "accept",
                            "reason": workflow_error,
                            "rewrite_notes": "",
                            "revised_doc_types": None,
                        },
                    }
                ],
                "review_feedback": {
                    "action": "accept",
                    "reason": workflow_error,
                    "rewrite_notes": "",
                    "revised_doc_types": None,
                },
                "final_retrieval": {
                    "ok": False,
                    "error": workflow_error,
                    "queries_used": list(failure_request["queries"]),
                    "top_tables": [],
                    "max_total_score": None,
                },
            }

        final_retrieval = dict(attempts[-1]["retrieval"]) if attempts else {
            "ok": False,
            "error": "NO_RETRIEVAL_ATTEMPT_COMPLETED",
            "queries_used": [],
            "top_tables": [],
        }

        return {
            "job": job_plan,
            "target": target,
            "attempts": attempts,
            "model_turns": [
                {
                    **turn,
                    "message": None,  # avoid returning raw model objects in final payload
                }
                for turn in model_turns
            ],
            "reviewer_turns": reviewer_turns,
            "review_feedback": dict(final_state.get("review_feedback") or {}),
            "final_retrieval": final_retrieval,
        }

    async def run(self, state: Dict[str, Any], client: Any) -> List[Dict[str, Any]]:
        targets = [
            {
                **dict(target),
                "form_type": _normalize_text(target.get("form_type")) or _DEFAULT_FORM_TYPE,
            }
            for target in (state.get("targets") or [])
            if isinstance(target, dict) and _normalize_text(target.get("ticker"))
        ]
        if not targets:
            raise ValueError("state['targets'] must contain at least one resolved target")

        retrieval_plan = dict(state.get("retrieval_plan") or {})
        jobs = list(retrieval_plan.get("jobs") or [])
        if not jobs:
            jobs = [{
                "job_type": "fact_lookup",
                "goal": _normalize_text(state.get("original_user_query")),
            }]

        runs: List[Dict[str, Any]] = []
        for raw_job in jobs:
            if not isinstance(raw_job, dict):
                continue
            job = _normalize_job(raw_job, original_user_query=_normalize_text(state.get("original_user_query")))
            for target in _select_targets_for_job(job, targets):
                run_payload = await self._run_single_target(
                    state=state,
                    client=client,
                    job_plan=job,
                    target=target,
                )
                runs.append(run_payload)
        return runs


async def retrieval_agent_v2(
    state: Dict[str, Any],
    *,
    client: Any,
    retrieval_llm: BaseChatModel,
    reviewer_llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Convenience wrapper that mirrors the shape of the existing retrieval_agent entrypoint.
    """
    workflow = RetrievalWorkflowAgent(
        retrieval_llm=retrieval_llm,
        reviewer_llm=reviewer_llm,
        top_k=3,
        min_total_score=0.0,
        max_attempts=2,
    )
    runs_payload = await workflow.run(state, client)
    retrieval_output = _build_retrieval_output(
        state=state,
        runs_payload=runs_payload,
        model_name=getattr(retrieval_llm, "model_name", retrieval_llm.__class__.__name__),
    )
    return {**state, "retrieval": retrieval_output}


__all__ = [
    "DEFAULT_RETRIEVAL_AGENT_PROMPT_TEMPLATE",
    "DEFAULT_REVIEWER_PROMPT_TEMPLATE",
    "RetrievalReview",
    "RetrievalWorkflowAgent",
    "deterministic_doc_types_for_job",
    "retrieval_agent_v2",
]
