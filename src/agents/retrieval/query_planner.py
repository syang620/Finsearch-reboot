from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm_client import build_chat_model


DEFAULT_RETRIEVAL_QUERY_PLANNER_PROMPT_TEMPLATE = """
You are a query planner for SEC-filings retrieval.

Your job is to generate a compact retrieval query intent for hybrid search over a SINGLE retrieval intent.
The downstream retriever already receives deterministic metadata filters such as ticker, fiscal_year, and form_type.
Do NOT invent or modify those filters.
Do NOT answer the user's question.


Rules:
1. Preserve the user's retrieval intent exactly.
2. `canonical_query` should be a single stable normalized statement of what evidence to retrieve from one filing.
3. `search_queries` should contain 1 to 4 short retrieval-oriented queries.
4. Keep `search_queries` short and search-friendly, not full natural-language questions.
5. Prefer filing/accounting terminology and close synonyms when helpful.
6. Avoid broad or vague search queries such as "financial performance" or "important metrics".
7. If `job_type` is `metric_extract`, bias toward exact metric/line-item wording.
8. If `job_type` is `narrative_extract`, bias toward short topical phrases that would surface explanatory text.
9. If the input is weak or underspecified, still produce the best compact retrieval intent possible from the provided `goal`.
10. Do not include ticker, company name, filing form or fiscal year inside the search queries unless they are essential to the financial concept itself.


Input:
{
  "job_type": "...",
  "goal": "...",
  "original_user_query": "...",
  "clarification_history": [...],
  "form_type": "10-K"
}

Output schema:
{
  "canonical_query": "string",
  "search_queries": ["string"]
}

Return exactly one JSON object matching the schema.
"""


class _QueryPlannerGraphState(MessagesState):
    messages: Annotated[List[Any], add_messages]

DEFAULT_RETRIEVAL_TOOL_CALLING_SYSTEM_PROMPT = """You are an SEC filings retrieval agent.
Your job is to retrieve evidence for exactly one retrieval job and exactly one filing target.
You interact with the SEC retrieval tool `sec_retrieve_tables` using tool calls.
You do not answer the user question.
Do not change ticker, fiscal_year, or form_type.
Stop after one retry at most.
"""

DEFAULT_RETRIEVAL_TOOL_CALLING_PROMPT_TEMPLATE = """
You must issue a tool call for SEC filings evidence lookup.

Task:
1. Make at most 1 call to `sec_retrieve_tables`.
2. Use the provided target metadata exactly as given.
3. Never change ticker, fiscal_year, or form_type.
4. Build 1 to 4 short retrieval queries.
5. Apply required deterministic doc filter by job type if provided.
6. Use `top_k=3` and `min_total_score=0`.
7. Pass this input as tool arguments.
Input:
{
  "original_user_query": "...",
  "clarification_history": [...],
  "job": {
    "job_type": "...",
    "goal": "..."
  },
  "target": {
    "ticker": "...",
    "fiscal_year": 2024,
    "form_type": "10-K"
  },
  "suggested_query_cues": ["..."],
  "required_doc_types": ["..."] | null
}
"""

DEFAULT_RETRIEVAL_REVIEW_TOOL_CALLING_PROMPT_TEMPLATE = """
You have just received the result of one retrieval tool call for the same job/target.

Rules:
1. Keep target metadata fixed (ticker, fiscal_year, form_type).
2. Return the current result if it looks acceptable.
3. If it is weak, you may issue one retry by calling `sec_retrieve_tables` again with revised queries.
4. Use the same target metadata for any retry.
5. If retrying is unnecessary or attempts are exhausted, stop and provide a short completion note.

Input:
{
  "attempt_index": 1,
  "attempts_remaining": 1,
  "request_used": {...},
    "retrieval_result": {...},
  "target": {...}
}

If you need a retry, call the tool once. Otherwise, respond with a short completion message.
"""

_DEFAULT_MODEL = "qwen2.5-14b-instruct-1m"
_DEFAULT_FORM_TYPE = "10-K"
_MAX_QUERIES = 4
_DOC_TYPE_ALIASES = {
    "text": "text_chunk",
    "text_chunk": "text_chunk",
    "text chunk": "text_chunk",
    "chunk": "text_chunk",
    "table": "table",
    "tables": "table",
    "row": "table_row",
    "rows": "table_row",
    "table_row": "table_row",
    "table row": "table_row",
}
_ALLOWED_DOC_TYPES = {"text_chunk", "table", "table_row"}
_JOB_TYPES = {"fact_lookup", "metric_extract", "component_extract", "narrative_extract"}


def _sec_retrieve_tables_tool(
    *,
    queries: List[str],
    ticker: str,
    fiscal_year: int,
    form_type: str = "10-K",
    doc_types: Optional[List[str]] = None,
    top_k: int = 3,
    min_total_score: int = 0,
) -> Dict[str, Any]:
    return {
        "queries": queries,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "form_type": form_type,
        "doc_types": doc_types,
        "top_k": top_k,
        "min_total_score": min_total_score,
    }


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def deterministic_doc_types_for_job(job_type: Any) -> Optional[List[str]]:
    normalized = _normalize_text(job_type).lower()
    if normalized == "metric_extract":
        return ["table"]
    if normalized == "narrative_extract":
        return ["text_chunk"]
    return None


def _dedupe_strings(values: Sequence[Any], *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _normalize_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _dedupe_ints(values: Sequence[Any], *, limit: Optional[int] = None) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        number = _normalize_int(value)
        if number is None or number in seen:
            continue
        seen.add(number)
        out.append(number)
        if limit is not None and len(out) >= limit:
            break
    return out


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(text)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
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
                    candidate = json.loads(raw[start : index + 1])
                except Exception:
                    return None
                if isinstance(candidate, dict):
                    return candidate
                return None
    return None


def _clean_goal_phrase(goal: str) -> str:
    cleaned = _normalize_text(goal)
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^(extract|find|locate|retrieve|look up|get|show|identify|report)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned


def _canonical_metric_phrase(goal: str, original_user_query: str) -> str:
    combined = " ".join(piece for piece in (_clean_goal_phrase(goal), _normalize_text(original_user_query)) if piece)
    lower = combined.lower()

    if "total debt" in lower:
        return "total debt"
    if "net debt" in lower:
        return "net debt"
    if "revenue" in lower or "net sales" in lower:
        return "revenue"
    if "operating cash flow" in lower or "cash flow from operations" in lower:
        return "net cash provided by operating activities"
    if "free cash flow" in lower or re.search(r"\bfcf\b", lower):
        return "free cash flow"
    if "capital expenditures" in lower or "capital expenditure" in lower or "capex" in lower:
        return "capital expenditures"
    if "earnings per share" in lower or re.search(r"\beps\b", lower):
        return "earnings per share"

    cleaned_goal = _clean_goal_phrase(goal)
    return cleaned_goal or _normalize_text(original_user_query) or "filing evidence"


def _infer_section_hints(metric_phrase: str, job_type: str) -> List[str]:
    metric = metric_phrase.lower()
    if "debt" in metric or "liabilit" in metric or "borrow" in metric:
        return ["Consolidated Balance Sheets", "Debt"]
    if "revenue" in metric or "sales" in metric:
        return ["Consolidated Statements of Operations", "Revenue Recognition"]
    if "cash" in metric or "capex" in metric or "capital expenditures" in metric:
        return ["Consolidated Statements of Cash Flows"]
    if "earnings per share" in metric:
        return ["Consolidated Statements of Operations", "Note 2. Earnings Per Share"]
    if job_type == "narrative_extract":
        return ["Management's Discussion and Analysis"]
    return []


def _infer_must_include(metric_phrase: str) -> List[str]:
    metric = metric_phrase.lower()
    if "total debt" in metric:
        return ["total debt", "long-term debt", "current portion of long-term debt"]
    if "net debt" in metric:
        return ["total debt", "cash and cash equivalents"]
    if "revenue" in metric or "sales" in metric:
        return ["revenue", "net sales"]
    if "free cash flow" in metric:
        return ["net cash provided by operating activities", "capital expenditures"]
    if "cash" in metric or "capital expenditures" in metric:
        return ["net cash provided by operating activities", "capital expenditures"]
    if "earnings per share" in metric:
        return ["earnings per share", "basic", "diluted"]
    return _dedupe_strings([metric_phrase], limit=4)


def _infer_nice_to_include(metric_phrase: str) -> List[str]:
    metric = metric_phrase.lower()
    if "total debt" in metric:
        return ["debt", "borrowings", "notes payable"]
    if "revenue" in metric or "sales" in metric:
        return ["sales", "total net sales"]
    if "cash" in metric:
        return ["cash flows", "operating activities"]
    return []


def _infer_doc_types(job_type: str, metric_phrase: str) -> List[str]:
    deterministic_doc_types = deterministic_doc_types_for_job(job_type)
    if deterministic_doc_types is not None:
        return deterministic_doc_types

    metric = metric_phrase.lower()
    if job_type == "narrative_extract":
        return ["text_chunk"]
    if job_type in {"metric_extract", "component_extract"}:
        return ["table", "table_row"]
    if "debt" in metric or "revenue" in metric or "cash" in metric or "earnings per share" in metric:
        return ["table", "table_row"]
    return ["text_chunk", "table", "table_row"]


def _build_fallback_queries(metric_phrase: str, section_hints: Sequence[str]) -> List[str]:
    metric = metric_phrase.lower()
    queries: List[str] = [metric_phrase]

    if "total debt" in metric:
        queries.extend(
            [
                "long-term debt",
                "current portion of long-term debt",
            ]
        )
    elif "net debt" in metric:
        queries.extend(
            [
                "total debt",
                "cash and cash equivalents",
            ]
        )
    elif "revenue" in metric or "sales" in metric:
        queries.extend(
            [
                "revenue",
                "net sales",
            ]
        )
    elif "free cash flow" in metric:
        queries.extend(
            [
                "net cash provided by operating activities",
                "capital expenditures",
            ]
        )
    elif "cash" in metric or "capital expenditures" in metric:
        queries.extend(
            [
                "net cash provided by operating activities",
                "capital expenditures",
            ]
        )
    elif "earnings per share" in metric:
        queries.extend(
            [
                "earnings per share",
                "diluted earnings per share",
            ]
        )

    if section_hints:
        queries.append(f"{metric_phrase} {section_hints[0]}")

    return _dedupe_strings(queries, limit=_MAX_QUERIES)


def _normalize_doc_types(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _normalize_text(value).lower().replace("-", "_")
        text = text.replace(" ", "_")
        text = _DOC_TYPE_ALIASES.get(text, text)
        if text not in _ALLOWED_DOC_TYPES or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _collapse_spacing(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    return cleaned.strip(" ,;:-")


def _sanitize_query(text: str, *, tickers: Sequence[str], years: Sequence[int]) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ""

    for ticker in tickers:
        if ticker:
            cleaned = re.sub(rf"\b{re.escape(ticker)}\b", " ", cleaned, flags=re.IGNORECASE)

    for year in years:
        year_text = str(year)
        cleaned = re.sub(rf"\bFY\s*{re.escape(year_text)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\bfiscal year\s*{re.escape(year_text)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\bfiscal\s*{re.escape(year_text)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\b{re.escape(year_text)}\b", " ", cleaned)

    return _collapse_spacing(cleaned)


def _seed_queries_for_job(
    *,
    goal: str,
    original_user_query: str,
    job_type: str,
    target: Dict[str, Any],
) -> List[str]:
    metric_phrase = _canonical_metric_phrase(goal, original_user_query)
    section_hints = _infer_section_hints(metric_phrase, job_type)
    tickers, years = _banned_filters(target=target)
    seed_queries = [
        _sanitize_query(value, tickers=tickers, years=years)
        for value in _build_fallback_queries(metric_phrase, section_hints)
    ]
    return _dedupe_strings(seed_queries, limit=_MAX_QUERIES)


def _targets_for_job(
    *,
    targets: Sequence[Dict[str, Any]],
    job: Dict[str, Any],
) -> List[Dict[str, Any]]:
    target_map = {
        int(target_id): dict(target)
        for target in targets
        if isinstance(target, dict)
        and (target_id := _normalize_int(target.get("target_id"))) is not None
    }
    applies_to_target_ids = [
        int(target_id)
        for target_id in _as_list(job.get("applies_to_target_ids"))
        if _normalize_int(target_id) is not None
    ]
    matched = [target_map[target_id] for target_id in applies_to_target_ids if target_id in target_map]
    return matched or [dict(target) for target in targets if isinstance(target, dict)]


def _banned_filters(*, target: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    tickers = _dedupe_strings([target.get("ticker")])
    years = [
        year
        for year in [_normalize_int(target.get("fiscal_year"))]
        if year is not None
    ]
    return tickers, years


def _fallback_request(
    *,
    original_user_query: str,
    job: Dict[str, Any],
    target: Dict[str, Any],
) -> Dict[str, Any]:
    job_type = _normalize_text(job.get("job_type")).lower() or "fact_lookup"
    goal = _normalize_text(job.get("goal")) or original_user_query
    metric_phrase = _canonical_metric_phrase(goal, original_user_query)
    section_hints = _infer_section_hints(metric_phrase, job_type)

    tickers, years = _banned_filters(target=target)
    queries = [
        _sanitize_query(value, tickers=tickers, years=years)
        for value in _build_fallback_queries(metric_phrase, section_hints)
    ]
    queries = _dedupe_strings(queries, limit=_MAX_QUERIES)
    if not queries:
        queries = _build_fallback_queries(metric_phrase, section_hints)

    doc_types = _infer_doc_types(job_type, metric_phrase)
    if deterministic_doc_types_for_job(job_type) is not None:
        doc_types = deterministic_doc_types_for_job(job_type)

    return {
        "queries": queries,
        "doc_types": doc_types,
        "reason": "deterministic fallback",
    }


def _normalize_initial_request(
    raw_request: Optional[Dict[str, Any]],
    *,
    original_user_query: str,
    job: Dict[str, Any],
    target: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = _fallback_request(
        original_user_query=original_user_query,
        job=job,
        target=target,
    )
    raw_request = raw_request or {}

    tickers, years = _banned_filters(target=target)
    required_doc_types = deterministic_doc_types_for_job(job.get("job_type"))

    queries = [
        _sanitize_query(value, tickers=tickers, years=years)
        for value in _as_list(raw_request.get("queries"))
    ]
    queries = _dedupe_strings(queries, limit=_MAX_QUERIES)
    if not queries:
        queries = fallback["queries"]

    normalized_doc_types = _normalize_doc_types(_as_list(raw_request.get("doc_types")))
    if required_doc_types is not None:
        doc_types = required_doc_types
    else:
        doc_types = normalized_doc_types or fallback["doc_types"]

    return {
        "queries": queries,
        "doc_types": doc_types,
        "reason": _normalize_text(raw_request.get("reason")) or fallback["reason"],
    }


def render_retrieval_query_planner_prompt(
    *,
    prompt_template: str,
    prompt_input: Dict[str, Any],
) -> str:
    return (
        prompt_template.strip()
        + "\n\nActual input:\n"
        + json.dumps(prompt_input, indent=2, ensure_ascii=False)
    )


def render_retrieval_tool_calling_prompt(
    *,
    prompt_template: str,
    prompt_input: Dict[str, Any],
) -> str:
    return (
        prompt_template.strip()
        + "\n\nActual input:\n"
        + json.dumps(prompt_input, indent=2, ensure_ascii=False)
    )


def _extract_tool_call(message: Any) -> Optional[Dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None

    first = tool_calls[0]
    if isinstance(first, dict):
        return first
    name = getattr(first, "name", None)
    args = getattr(first, "args", None)
    tool_id = getattr(first, "id", None)
    return {"name": name, "args": args, "id": tool_id}


def _coerce_tool_args(
    raw: Optional[Dict[str, Any]],
    *,
    state: Dict[str, Any],
    job: Dict[str, Any],
    target: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    if not isinstance(raw, dict):
        request = _normalize_initial_request(
            None,
            original_user_query=_normalize_text(state.get("original_user_query")),
            job=job,
            target=target,
        )
        return request, "FALLBACK_TOOL_ARGS_MISSING"

    request = _normalize_initial_request(
        raw,
        original_user_query=_normalize_text(state.get("original_user_query")),
        job=job,
        target=target,
    )
    return request, "TOOL_ARGS_COERCED"


def _extract_result_blob(result: Dict[str, Any]) -> str:
    compact = _compact_retrieval_result(result)
    pieces: List[str] = []
    for item in compact.get("top_items") or []:
        pieces.extend(
            [
                _normalize_text(item.get("doc_id")),
                _normalize_text(item.get("section_path")),
                _normalize_text(item.get("summary")),
            ]
        )
    return "\n".join(part for part in pieces if part).lower()


def _extract_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    table_obj = candidate.get("table")
    if isinstance(table_obj, dict):
        payload = table_obj.get("payload")
        if isinstance(payload, dict):
            return payload
        return table_obj
    return candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}


def _candidate_doc_id(candidate: Dict[str, Any]) -> str:
    payload = _extract_payload(candidate)
    return _normalize_text(
        candidate.get("doc_id")
        or payload.get("doc_id")
        or payload.get("docId")
        or candidate.get("table_id")
    )


def _candidate_section_path(candidate: Dict[str, Any]) -> str:
    payload = _extract_payload(candidate)
    return _normalize_text(candidate.get("section_path") or payload.get("section_path") or payload.get("sectionPath"))


def _candidate_summary(candidate: Dict[str, Any]) -> str:
    payload = _extract_payload(candidate)
    for key in ("rerank_table_summary", "content", "rerank_original_content", "text", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
    return ""


def _compact_retrieval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    top_tables = [item for item in _as_list(result.get("top_tables")) if isinstance(item, dict)]
    compact_items: List[Dict[str, Any]] = []
    for index, item in enumerate(top_tables[:3], start=1):
        compact_items.append(
            {
                "rank": index,
                "doc_id": _candidate_doc_id(item),
                "section_path": _candidate_section_path(item),
                "total_score": item.get("total_score"),
                "summary": _candidate_summary(item),
            }
        )

    return {
        "ok": bool(result.get("ok", True)),
        "error": _normalize_text(result.get("error")),
        "queries_used": _dedupe_strings(_as_list(result.get("queries_used")), limit=_MAX_QUERIES),
        "num_results": len(top_tables),
        "max_total_score": result.get("max_total_score"),
        "metadata_used": dict(result.get("metadata_used") or {}),
        "top_items": compact_items,
        "trace": dict(result.get("trace") or {}),
    }


def _extract_expected_terms(
    *,
    job: Dict[str, Any],
    original_user_query: str,
) -> List[str]:
    goal = _normalize_text(job.get("goal")) or original_user_query or "filing evidence"
    metric_phrase = _canonical_metric_phrase(goal, original_user_query)
    return _infer_must_include(metric_phrase)


def _looks_good_enough(
    result: Dict[str, Any],
    *,
    job: Dict[str, Any],
    state: Dict[str, Any],
    min_results: int = 1,
    min_total_score: float = 0.0,
) -> bool:
    compact = _compact_retrieval_result(result)
    if not compact.get("ok"):
        return False

    if int(compact.get("num_results") or 0) < min_results:
        return False

    max_total_score = compact.get("max_total_score")
    if max_total_score is not None:
        try:
            if float(max_total_score) < min_total_score:
                return False
        except Exception:
            pass

    anchors = _extract_expected_terms(
        job=job,
        original_user_query=_normalize_text(state.get("original_user_query")) or "",
    )
    anchors = _dedupe_strings(anchors, limit=6)
    if anchors:
        blob = _extract_result_blob(result)
        if not any(anchor.lower() in blob for anchor in anchors):
            return False

    return True


def _fallback_review_decision(
    *,
    request: Dict[str, Any],
    result: Dict[str, Any],
    job: Dict[str, Any],
    state: Dict[str, Any],
    seed_queries: List[str],
    attempts_remaining: int,
) -> Dict[str, Any]:
    if _looks_good_enough(
        result,
        job=job,
        state=state,
        min_results=1,
        min_total_score=0.0,
    ):
        return {
            "action": "return",
            "reason": "heuristic accept",
            "queries": [],
            "doc_types": request.get("doc_types"),
        }

    if attempts_remaining <= 0:
        return {
            "action": "return",
            "reason": "no attempts remaining",
            "queries": [],
            "doc_types": request.get("doc_types"),
        }

    retry_queries = _dedupe_strings(_as_list(seed_queries), limit=_MAX_QUERIES)
    if not retry_queries:
        retry_queries = request.get("queries") or []

    return {
        "action": "retry",
        "reason": "heuristic retry",
        "queries": retry_queries[:_MAX_QUERIES],
        "doc_types": request.get("doc_types"),
    }


def build_retrieval_tool_calling_prompt(
    *,
    original_user_query: str,
    clarification_history: List[Dict[str, Any]],
    job: Dict[str, Any],
    target: Dict[str, Any],
    suggested_queries: List[str],
    prompt_template: str = DEFAULT_RETRIEVAL_TOOL_CALLING_PROMPT_TEMPLATE,
) -> str:
    job_type = _normalize_text(job.get("job_type")) or "fact_lookup"
    prompt_input = {
        "original_user_query": original_user_query,
        "clarification_history": clarification_history,
        "job": job,
        "target": target,
        "suggested_query_cues": suggested_queries,
        "required_doc_types": deterministic_doc_types_for_job(job_type),
    }
    return render_retrieval_tool_calling_prompt(
        prompt_template=prompt_template,
        prompt_input=prompt_input,
    )


class RetrievalQueryPlannerAgent:
    """Tool-calling runtime agent for planner handoff to MCP retrieval."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        llm: Any | None = None,
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_RETRIEVAL_QUERY_PLANNER_PROMPT_TEMPLATE,
        top_k: int = 3,
        min_total_score: float = 0.0,
        max_attempts: int = 2,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.top_k = int(top_k)
        self.min_total_score = float(min_total_score)
        self.max_attempts = max(1, int(max_attempts))
        self.timeout_s = float(timeout_s)
        self.llm = llm or build_chat_model(model=model, temperature=temperature)

    def _normalize_job(self, *, job: Dict[str, Any]) -> Dict[str, Any]:
        job_type = _normalize_text(job.get("job_type")) or "fact_lookup"
        if job_type not in _JOB_TYPES:
            job_type = "fact_lookup"
        return {
            "job_type": job_type,
            "goal": _normalize_text(job.get("goal")) or _normalize_text(job.get("original_user_query")) or "filing evidence",
            "applies_to_target_ids": _dedupe_ints(_as_list(job.get("applies_to_target_ids")), limit=64),
        }

    def _resolve_target(self, *, target: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target_id": target.get("target_id"),
            "ticker": _normalize_text(target.get("ticker")),
            "fiscal_year": _normalize_int(target.get("fiscal_year")),
            "form_type": _normalize_text(target.get("form_type")) or _DEFAULT_FORM_TYPE,
        }

    def _build_first_prompt_input(self, *, state: Dict[str, Any], job: Dict[str, Any], target: Dict[str, Any], job_plan: Dict[str, Any]) -> Dict[str, Any]:
        suggested_queries = _seed_queries_for_job(
            goal=job_plan["goal"],
            original_user_query=_normalize_text(state.get("original_user_query")),
            job_type=job_plan["job_type"],
            target=target,
        )

        return {
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
        }

    def _build_review_prompt_input(
        self,
        *,
        attempt_index: int,
        request: Dict[str, Any],
        result: Dict[str, Any],
        target: Dict[str, Any],
        job_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "attempt_index": int(attempt_index),
            "attempts_remaining": max(self.max_attempts - int(attempt_index), 0),
            "request_used": {
                "queries": list(request.get("queries") or []),
                "doc_types": request.get("doc_types"),
                "top_k": self.top_k,
                "min_total_score": self.min_total_score,
            },
            "retrieval_result": _compact_retrieval_result(result),
            "job": {
                "job_type": job_plan["job_type"],
                "goal": job_plan["goal"],
            },
            "target": target,
        }

    def _build_prompt(self, *, prompt_input: Dict[str, Any]) -> str:
        return (
            self.system_prompt.strip()
            + "\n\nActual input:\n"
            + json.dumps(prompt_input, indent=2, ensure_ascii=False)
        )

    async def _retrieve_with_client(
        self,
        *,
        client: Any,
        request: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return await client.retrieve_tables(
                queries=request.get("queries") or [],
                ticker=target["ticker"],
                fiscal_year=target["fiscal_year"],
                form_type=target["form_type"],
                doc_types=request.get("doc_types"),
                top_k=self.top_k,
                min_total_score=self.min_total_score,
                timeout_s=self.timeout_s,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"RETRIEVER_CALL_FAILED: {exc}",
                "queries_used": request.get("queries") or [],
                "metadata_used": {
                    "ticker": target["ticker"],
                    "fiscal_year": target["fiscal_year"],
                    "form_type": target["form_type"],
                },
            }

    def _build_attempt_log(
        self,
        *,
        attempt_index: int,
        request: Dict[str, Any],
        retrieval: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "attempt_index": int(attempt_index),
            "request": {
                "queries": list(request.get("queries") or []),
                "doc_types": request.get("doc_types"),
                "top_k": self.top_k,
                "min_total_score": self.min_total_score,
                "reason": request.get("reason"),
            },
            "retrieval": dict(retrieval),
            "retrieval_compact": _compact_retrieval_result(retrieval),
        }

    def _coerce_review_decision(
        self,
        *,
        review_request: Optional[Any],
        request: Dict[str, Any],
        result: Dict[str, Any],
        state: Dict[str, Any],
        job_plan: Dict[str, Any],
        job: Dict[str, Any],
        seed_queries: List[str],
        target: Dict[str, Any],
        attempts_remaining: int,
    ) -> Dict[str, Any]:
        # Keep deterministic behaviour when model output is malformed.
        decision = _fallback_review_decision(
            request=request,
            result=result,
            job=job,
            state=state,
            seed_queries=seed_queries,
            attempts_remaining=attempts_remaining,
        )

        tool_call = _extract_tool_call(review_request)
        if not tool_call or tool_call.get("name") != "sec_retrieve_tables":
            return decision

        attempted_request, _ = _coerce_tool_args(
                tool_call.get("args"),
                state=state,
                job={"job_type": job_plan["job_type"]},
                target=target,
            )
        if not attempted_request.get("queries"):
            return decision

        return {
            "action": "retry",
            "reason": _normalize_text(tool_call.get("id")) or "tool_call_retry",
            "queries": attempted_request.get("queries") or request.get("queries", []),
            "doc_types": attempted_request.get("doc_types"),
        }

    async def run_single_target(
        self,
        *,
        state: Dict[str, Any],
        job: Dict[str, Any],
        target: Dict[str, Any],
        client: Any,
    ) -> Dict[str, Any]:
        job_plan = self._normalize_job(job=job)
        first_pass_input = self._build_first_prompt_input(
            state=state,
            job=job,
            target=target,
            job_plan=job_plan,
        )
        seed_queries = _dedupe_strings(
            list(first_pass_input.get("suggested_query_cues") or []),
            limit=_MAX_QUERIES,
        )

        attempts: List[Dict[str, Any]] = []
        model_turns: List[Dict[str, Any]] = []
        review_decision = {
            "action": "return",
            "reason": "fallback_decision",
            "queries": [],
            "doc_types": None,
        }

        review_input: Optional[Dict[str, Any]] = None
        review_request_for_decision: Optional[Dict[str, Any]] = None
        review_result_for_decision: Optional[Dict[str, Any]] = None

        # Debug flag can be toggled for a single troubleshooting run:
        # python -m environment var "RETRIEVAL_DEBUG_TOOL_CALLS=1"
        debug_tool_calls = False
        if str(__import__("os").environ.get("RETRIEVAL_DEBUG_TOOL_CALLS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            debug_tool_calls = True

        @tool
        async def sec_retrieve_tables(
            queries: List[str],
            doc_types: Optional[List[str]] = None,
            ticker: str = "",
            fiscal_year: Optional[int] = None,
            form_type: str = _DEFAULT_FORM_TYPE,
            top_k: int = 3,
            min_total_score: float = 0,
        ) -> str:
            """Retrieve SEC filing chunks for one retrieval attempt."""
            attempt_index = len(attempts) + 1
            if attempt_index > self.max_attempts:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "MAX_RETRIEVAL_ATTEMPTS_EXCEEDED",
                        "attempt_index": attempt_index,
                        "attempts_remaining": 0,
                        "queries_used": _as_list(queries),
                    },
                    ensure_ascii=False,
                )

            request, reason = _coerce_tool_args(
                {
                    "queries": queries,
                    "doc_types": doc_types,
                    "top_k": top_k,
                    "min_total_score": min_total_score,
                    "ticker": _normalize_text(ticker) or target["ticker"],
                    "fiscal_year": fiscal_year if fiscal_year is not None else target["fiscal_year"],
                    "form_type": _normalize_text(form_type) or target["form_type"],
                },
                state=state,
                job={"job_type": job_plan["job_type"]},
                target=target,
            )
            if debug_tool_calls:
                print(
                    json.dumps(
                        {
                            "event": "query_planner_tool_request",
                            "target": {"ticker": target["ticker"], "fiscal_year": target["fiscal_year"], "form_type": target["form_type"]},
                            "job_type": job_plan["job_type"],
                            "attempt_index": attempt_index,
                            "tool_input": {
                                "queries": list(queries or []),
                                "doc_types": list(doc_types or []) if doc_types else None,
                                "top_k": top_k,
                                "min_total_score": min_total_score,
                                "ticker": _normalize_text(ticker) or target["ticker"],
                                "fiscal_year": fiscal_year if fiscal_year is not None else target["fiscal_year"],
                                "form_type": _normalize_text(form_type) or target["form_type"],
                            },
                            "normalized_request": request,
                            "reason": reason,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            request["reason"] = reason

            retrieval = await self._retrieve_with_client(
                client=client,
                request=request,
                target=target,
            )
            attempt_log = self._build_attempt_log(
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
                    "reason": _normalize_text(request.get("reason")) or reason,
                    "metadata_fixed": {
                        "ticker": target["ticker"],
                        "fiscal_year": target["fiscal_year"],
                        "form_type": target["form_type"],
                    },
                },
                ensure_ascii=False,
            )

        async def call_model(step_state: _QueryPlannerGraphState) -> Dict[str, Any]:
            nonlocal review_input, review_request_for_decision, review_result_for_decision
            attempt_index = len(attempts)
            if attempt_index == 0:
                prompt_input = first_pass_input
            else:
                if not attempts:
                    return {
                        "messages": [HumanMessage(content="No retrieval attempt completed; retrying deterministically is required.")]
                    }

                last_attempt = attempts[-1]
                prompt_input = self._build_review_prompt_input(
                    attempt_index=attempt_index,
                    request=last_attempt["request"],
                    result=last_attempt["retrieval"],
                    target=target,
                    job_plan=job_plan,
                )
                review_input = prompt_input
                review_request_for_decision = dict(last_attempt.get("request") or {})
                review_result_for_decision = dict(last_attempt.get("retrieval") or {})

            prompt = self._build_prompt(prompt_input=prompt_input)
            messages_for_model = list(step_state["messages"])
            prompt_msg = HumanMessage(content=prompt)
            messages_for_model.append(prompt_msg)

            try:
                response = await self.llm.bind_tools(
                    [sec_retrieve_tables],
                    tool_choice="any" if attempt_index == 0 else None,
                ).ainvoke(messages_for_model)

                model_turns.append(
                    {
                        "attempt_index": attempt_index,
                        "prompt": prompt,
                        "raw_output": response.content if hasattr(response, "content") else str(response),
                        "message": response,
                    }
                )
                return {"messages": [prompt_msg, response]}
            except Exception as exc:
                error_text = f"LLM_CALL_FAILED: {exc}"
                model_turns.append(
                    {
                        "attempt_index": attempt_index,
                        "prompt": prompt,
                        "raw_output": "",
                        "error": error_text,
                        "message": None,
                    }
                )
                return {"messages": [prompt_msg, HumanMessage(content=error_text)]}

        graph = StateGraph(_QueryPlannerGraphState)
        graph.add_node("call_model", call_model)
        graph.add_node("tools", ToolNode([sec_retrieve_tables]))
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges("call_model", tools_condition)
        graph.add_edge("tools", "call_model")
        await graph.compile().ainvoke({"messages": []})

        if not attempts:
            fallback_request, fallback_reason = _coerce_tool_args(
                None,
                state=state,
                job={"job_type": job_plan["job_type"]},
                target=target,
            )
            fallback_request["reason"] = fallback_reason
            fallback_result = await self._retrieve_with_client(
                client=client,
                request=fallback_request,
                target=target,
            )
            attempts.append(
                self._build_attempt_log(
                    attempt_index=1,
                    request=fallback_request,
                    retrieval=fallback_result,
                )
            )

        if attempts:
            if len(attempts) >= 2 and review_request_for_decision is None:
                review_request_for_decision = dict(attempts[-2].get("request") or {})
                review_result_for_decision = dict(attempts[-2].get("retrieval") or {})

            if len(model_turns) >= 2 and model_turns[1].get("message") is not None:
                reviewed_request = review_request_for_decision or attempts[-1].get("request") or {}
                reviewed_result = review_result_for_decision or attempts[-1].get("retrieval") or {}
                review_decision = self._coerce_review_decision(
                    review_request=model_turns[1].get("message"),
                    request=reviewed_request,
                    result=reviewed_result,
                    state=state,
                    job_plan=job_plan,
                    job=job_plan,
                    seed_queries=seed_queries,
                    target=target,
                    attempts_remaining=max(self.max_attempts - len(attempts), 0),
                )
            else:
                review_decision = _fallback_review_decision(
                    request=attempts[-1]["request"],
                    result=attempts[-1]["retrieval"],
                    job=job_plan,
                    state=state,
                    seed_queries=seed_queries,
                    attempts_remaining=max(self.max_attempts - len(attempts), 0),
                )

            if review_decision.get("action") == "retry" and len(attempts) < self.max_attempts:
                retry_args = {
                    "queries": list(review_decision.get("queries") or attempts[-1]["request"].get("queries")),
                    "doc_types": review_decision.get("doc_types"),
                }
                retry_request, _ = _coerce_tool_args(
                    retry_args,
                    state=state,
                    job={"job_type": job_plan["job_type"]},
                    target=target,
                )
                retry_request["reason"] = review_decision.get("reason") or "DETERMINISTIC_RETRY"
                if not review_input:
                    review_input = self._build_review_prompt_input(
                        attempt_index=len(attempts),
                        request=attempts[-1]["request"],
                        result=attempts[-1]["retrieval"],
                        target=target,
                        job_plan=job_plan,
                    )

                retry_result = await self._retrieve_with_client(
                    client=client,
                    request=retry_request,
                    target=target,
                )
                attempts.append(
                    self._build_attempt_log(
                        attempt_index=len(attempts) + 1,
                        request=retry_request,
                        retrieval=retry_result,
                    )
                )
                review_decision["action"] = "return"
                review_decision["reason"] = review_decision.get("reason") or "DETERMINISTIC_RETRY_EXECUTED"
                review_decision["queries"] = []
                review_decision["doc_types"] = retry_request.get("doc_types")

        final_retrieval = attempts[-1]["retrieval"] if attempts else {
            "ok": False,
            "queries_used": [],
            "rerank_query": "",
            "top_tables": [],
            "max_total_score": None,
            "metadata_used": {
                "ticker": target["ticker"],
                "fiscal_year": target["fiscal_year"],
                "form_type": target["form_type"],
            },
            "error": "NO_ATTEMPTS_EXECUTED",
        }
        final_action = "return_after_retry" if len(attempts) > 1 else "return"

        first_turn = model_turns[0] if len(model_turns) >= 1 else {}
        second_turn = model_turns[1] if len(model_turns) >= 2 else {}
        if second_turn:
            reviewed_request = review_request_for_decision or attempts[-1].get("request") or {}
            reviewed_result = review_result_for_decision or attempts[-1].get("retrieval") or {}
            review_decision = self._coerce_review_decision(
                review_request=second_turn.get("message"),
                request=reviewed_request,
                result=reviewed_result,
                state=state,
                job=job_plan,
                seed_queries=seed_queries,
                job_plan=job_plan,
                target=target,
                attempts_remaining=max(self.max_attempts - 1, 0),
            )

        first_pass_prompt = first_turn.get("prompt") or self._build_prompt(prompt_input=first_pass_input)
        review_prompt = second_turn.get("prompt") or (self._build_prompt(prompt_input=review_input) if review_input else "")

        return {
            "job_type": job_plan["job_type"],
            "goal": job_plan["goal"],
            "target": target,
            "first_pass_prompt_input": first_pass_input,
            "first_pass_prompt": first_pass_prompt,
            "first_pass_raw_output": first_turn.get("raw_output") or "",
            "first_pass_error": first_turn.get("error"),
            "review_prompt_input": review_input if review_input is not None else None,
            "review_prompt": review_prompt,
            "review_raw_output": second_turn.get("raw_output") or "",
            "review_error": second_turn.get("error"),
            "review_decision": review_decision,
            "attempts": attempts,
            "final_action": final_action,
            "retrieval": final_retrieval,
        }

    async def run(self, state: Dict[str, Any], client: Any) -> Dict[str, Any]:
        retrieval_state = dict(state or {})
        retrieval_plan = dict(retrieval_state.get("retrieval_plan") or {})
        jobs = [dict(job) for job in (retrieval_plan.get("jobs") or []) if isinstance(job, dict)]
        targets = [dict(target) for target in (retrieval_state.get("targets") or []) if isinstance(target, dict)]

        runs: List[Dict[str, Any]] = []
        for job_index, job in enumerate(jobs, start=1):
            normalized_job = self._normalize_job(job=job)
            for target in _targets_for_job(targets=targets, job=job):
                selected_target = self._resolve_target(target=target)
                if not selected_target["ticker"] or selected_target["fiscal_year"] is None:
                    runs.append(
                        {
                            "job_index": job_index,
                            "applies_to_target_ids": normalized_job["applies_to_target_ids"],
                            "job_type": normalized_job["job_type"],
                            "goal": normalized_job["goal"],
                            "target": selected_target,
                            "first_pass_prompt_input": None,
                            "first_pass_prompt": "",
                            "first_pass_raw_output": "",
                            "first_pass_error": "MISSING_TARGET_METADATA",
                            "review_prompt_input": None,
                            "review_prompt": "",
                            "review_raw_output": "",
                            "review_error": None,
                            "review_decision": {
                                "action": "return",
                                "reason": "missing target metadata",
                                "queries": [],
                                "doc_types": None,
                            },
                            "attempts": [],
                            "final_action": "return",
                            "retrieval": {
                                "ok": False,
                                "queries_used": [],
                                "rerank_query": "",
                                "top_tables": [],
                                "max_total_score": None,
                                "metadata_used": {
                                    "ticker": selected_target.get("ticker"),
                                    "fiscal_year": selected_target.get("fiscal_year"),
                                    "form_type": selected_target.get("form_type"),
                                },
                                "error": "MISSING_TARGET_METADATA",
                            },
                        }
                    )
                    continue

                target_run = await self.run_single_target(
                    state=retrieval_state,
                    job=job,
                    target=selected_target,
                    client=client,
                )
                runs.append(
                    {
                        "job_index": job_index,
                        "applies_to_target_ids": normalized_job["applies_to_target_ids"],
                        **target_run,
                    }
                )

        return {
            "original_user_query": _normalize_text(retrieval_state.get("original_user_query")),
            "runs": runs,
        }

    def _requires_client(self, state: Dict[str, Any]) -> bool:
        retrieval_state = dict(state or {})
        retrieval_plan = dict(retrieval_state.get("retrieval_plan") or {})
        jobs = [dict(job) for job in (retrieval_plan.get("jobs") or []) if isinstance(job, dict)]
        targets = [dict(target) for target in (retrieval_state.get("targets") or []) if isinstance(target, dict)]
        if not jobs or not targets:
            return False

        for job in jobs:
            for target in _targets_for_job(targets=targets, job=job):
                resolved = self._resolve_target(target=target)
                if _normalize_text(resolved.get("ticker")) and _normalize_int(resolved.get("fiscal_year")) is not None:
                    return True
        return False

    async def _run_with_default_client(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.retrieval.mcp_client import SecRetrievalMCPClient

        async with SecRetrievalMCPClient() as client:
            return await self.run(state, client)

    def run_sync(self, state: Dict[str, Any], client: Any | None = None) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if client is None:
                if not self._requires_client(state):
                    return asyncio.run(self.run(state, None))
                return asyncio.run(self._run_with_default_client(state))
            return asyncio.run(self.run(state, client))
        raise RuntimeError("run_sync() cannot be used inside an existing event loop; use await run(...)")

    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility entrypoint name for older notebook calls."""
        return {**dict(state), "runs_payload": self.run_sync(state=state)}


def _score(entry: Any) -> float:
    if isinstance(entry, dict):
        value = entry.get("total_score")
    else:
        value = getattr(entry, "total_score", None)
    try:
        return float(value)
    except Exception:
        return float("-inf")


def _build_retrieval_output(
    *,
    state: Dict[str, Any],
    runs_payload: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    runs = [dict(run) for run in (runs_payload.get("runs") or []) if isinstance(run, dict)]
    top_tables: List[Dict[str, Any]] = []
    queries_used: List[str] = []
    errors: List[str] = []
    max_total_score = None

    for run in runs:
        result = dict(run.get("retrieval") or {})
        top_tables.extend([item for item in _as_list(result.get("top_tables")) if isinstance(item, dict)])
        queries_used.extend(_as_list(result.get("queries_used")))

        current_score = result.get("max_total_score")
        if isinstance(current_score, (int, float)):
            if max_total_score is None or float(current_score) > max_total_score:
                max_total_score = float(current_score)

        error = _normalize_text(result.get("error"))
        if error:
            errors.append(error)

    top_tables = sorted(top_tables, key=_score, reverse=True)
    queries_used = _dedupe_strings(queries_used, limit=32)

    targets = [dict(target) for target in (state.get("targets") or []) if isinstance(target, dict)]
    primary_target = next(
        (
            target
            for target in targets
            if _normalize_text(target.get("ticker")) and _normalize_int(target.get("fiscal_year")) is not None
        ),
        {},
    )
    retrieval_plan = dict(state.get("retrieval_plan") or {})

    return {
        "ok": bool(top_tables) or not errors,
        "queries_used": queries_used,
        "rerank_query": _normalize_text(state.get("original_user_query")) or (queries_used[0] if queries_used else ""),
        "top_tables": top_tables,
        "max_total_score": max_total_score,
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
            "targets": targets,
            "retrieval_plan": retrieval_plan,
            "job_runs": runs,
            "retrieval_agent_model": model,
            "max_attempts": 2,
            "top_k": 3,
            "min_total_score": 0.0,
        },
        "error": "; ".join(errors) if errors else None,
        "trace": {
            "runs": runs,
        },
    }


async def retrieval_agent(
    state: Dict[str, Any],
    client: Any | None = None,
    agent: RetrievalQueryPlannerAgent | None = None,
) -> Dict[str, Any]:
    retrieval_model = (
        _normalize_text(state.get("retrieval_query_planner_model"))
        or _normalize_text(state.get("retrieval_agent_model"))
        or _DEFAULT_MODEL
    )
    reviewer_model = _normalize_text(state.get("retrieval_reviewer_model")) or retrieval_model

    async def _run_with_default_client(active_agent: RetrievalQueryPlannerAgent) -> Dict[str, Any]:
        from agents.retrieval.mcp_client import SecRetrievalMCPClient

        async with SecRetrievalMCPClient() as created_client:
            return {**state, "retrieval": _build_retrieval_output(
                state=state,
                runs_payload=await active_agent.run(state, created_client),
                model=retrieval_model,
            )}

    if agent is not None:
        if client is None:
            return await _run_with_default_client(agent)
        runs_payload = await agent.run(state, client)
        retrieval_output = _build_retrieval_output(
            state=state,
            runs_payload=runs_payload,
            model=retrieval_model,
        )
        return {**state, "retrieval": retrieval_output}

    from agents.retrieval.query_planner_v2 import retrieval_agent_v2

    same_model_for_reviewer = _normalize_text(
        state.get("retrieval_query_planner_use_same_model_for_reviewer")
    ).lower() in {"1", "true", "yes", "on"}
    tool_model = retrieval_model
    review_model = tool_model if same_model_for_reviewer else reviewer_model

    from llm_client import build_chat_model

    retrieval_llm = build_chat_model(model=tool_model, temperature=0.0)
    reviewer_llm = build_chat_model(model=review_model, temperature=0.0)

    if client is not None:
        return await retrieval_agent_v2(
            state=state,
            client=client,
            retrieval_llm=retrieval_llm,
            reviewer_llm=reviewer_llm,
        )

    async def _run_with_default_client_v2() -> Dict[str, Any]:
        from agents.retrieval.mcp_client import SecRetrievalMCPClient

        async with SecRetrievalMCPClient() as created_client:
            return await retrieval_agent_v2(
                state=state,
                client=created_client,
                retrieval_llm=retrieval_llm,
                reviewer_llm=reviewer_llm,
            )

    return await _run_with_default_client_v2()


__all__ = [
    "build_retrieval_tool_calling_prompt",
    "DEFAULT_RETRIEVAL_QUERY_PLANNER_PROMPT_TEMPLATE",
    "DEFAULT_RETRIEVAL_TOOL_CALLING_PROMPT_TEMPLATE",
    "DEFAULT_RETRIEVAL_TOOL_CALLING_SYSTEM_PROMPT",
    "RetrievalQueryPlannerAgent",
    "deterministic_doc_types_for_job",
    "render_retrieval_query_planner_prompt",
    "render_retrieval_tool_calling_prompt",
]
