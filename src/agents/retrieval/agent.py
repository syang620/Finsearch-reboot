from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_client import build_chat_model


DEFAULT_RETRIEVAL_AGENT_SYSTEM_PROMPT = """
You are the retrieval agent for an SEC-filings RAG system.

You operate on exactly one retrieval job and exactly one filing target at a time.
The planner has already determined the metadata filters.
You must never alter ticker, fiscal_year, or form_type.
You do not answer the user question.

Your only responsibilities are:
1. produce an initial retrieval request (queries + optional doc_types)
2. inspect the retrieval result
3. decide whether to return the result or retry once with revised queries

Always be conservative:
- preserve the planner's intent
- keep queries short and search-friendly
- prefer filing/accounting terminology
- never broaden beyond the same target filing
- never ask clarifying questions here
- never invent facts from the filing

Return exactly one JSON object for each step.
""".strip()


DEFAULT_RETRIEVAL_FIRST_PASS_PROMPT_TEMPLATE = """
Step: INITIAL_REQUEST

Produce the first retrieval request for this single job / target pair.

Rules:
1. Use the target metadata exactly as given elsewhere in the runtime. Do not repeat them in the queries unless essential to the financial concept.
2. Return 1 to 4 short retrieval queries.
3. Prefer planner hints from `analysis_task` when available.
4. Refine only lightly. This is not a second planner.
5. If `required_doc_types` is non-null, use exactly that list.
6. If `required_doc_types` is null, you may return null for `doc_types` unless a narrower filter is clearly beneficial.
7. For `metric_extract`, bias toward exact line items / statements.
8. For `narrative_extract`, bias toward short topic / section phrases.

Output schema:
{
  "queries": ["string"],
  "doc_types": ["string"] | null,
  "reason": "string"
}
""".strip()


DEFAULT_RETRIEVAL_REVIEW_PROMPT_TEMPLATE = """
Step: REVIEW_RESULT

You are reviewing the first retrieval attempt for this same single job / target pair.

Decide whether to:
- return: keep the current retrieval result
- retry: issue one revised retrieval request and try one more time

Rules:
1. Only choose `retry` if the first result looks clearly weak, missing, or off-target.
2. If you choose `retry`, keep the same target metadata and return 1 to 4 revised queries.
3. If you choose `retry`, keep `doc_types` the same unless changing them is clearly beneficial.
4. If you choose `return`, set `queries` to an empty list.
5. You may use planner hints such as `must_include`, `section_hints`, and `analysis_task.metric`.
6. Never ask clarifying questions.
7. Never answer the user question.

Output schema:
{
  "action": "return" | "retry",
  "reason": "string",
  "queries": ["string"],
  "doc_types": ["string"] | null
}
""".strip()

_DEFAULT_MODEL = "qwen2.5-14b-instruct-1m"
_DEFAULT_FORM_TYPE = "10-K"
_MAX_QUERIES = 4
_ALLOWED_ACTIONS = {"return", "retry"}
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


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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


def deterministic_doc_types_for_job(job_type: Any) -> Optional[List[str]]:
    normalized = _normalize_text(job_type).lower()
    if normalized == "metric_extract":
        return ["table"]
    if normalized == "narrative_extract":
        return ["text_chunk"]
    return None


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
    combined = " ".join(
        piece for piece in (_clean_goal_phrase(goal), _normalize_text(original_user_query)) if piece
    )
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
        return ["Consolidated Statements of Operations", "Earnings Per Share"]
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


def _build_default_queries(metric_phrase: str, section_hints: Sequence[str]) -> List[str]:
    metric = metric_phrase.lower()
    queries: List[str] = [metric_phrase]

    if "total debt" in metric:
        queries.extend(["long-term debt", "current portion of long-term debt"])
    elif "net debt" in metric:
        queries.extend(["total debt", "cash and cash equivalents"])
    elif "revenue" in metric or "sales" in metric:
        queries.extend(["revenue", "net sales"])
    elif "free cash flow" in metric:
        queries.extend(["net cash provided by operating activities", "capital expenditures"])
    elif "cash" in metric or "capital expenditures" in metric:
        queries.extend(["net cash provided by operating activities", "capital expenditures"])
    elif "earnings per share" in metric:
        queries.extend(["earnings per share", "diluted earnings per share"])

    if section_hints:
        queries.append(f"{metric_phrase} {section_hints[0]}")

    return _dedupe_strings(queries, limit=_MAX_QUERIES)


def _targets_for_job(*, targets: Sequence[Dict[str, Any]], job: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_map = {
        int(target_id): dict(target)
        for target in targets
        if isinstance(target, dict)
        and (target_id := _normalize_int((target or {}).get("target_id"))) is not None
    }
    applies_to_target_ids = [
        int(target_id)
        for target_id in _as_list(job.get("applies_to_target_ids"))
        if _normalize_int(target_id) is not None
    ]
    matched_targets = [target_map[target_id] for target_id in applies_to_target_ids if target_id in target_map]
    return matched_targets or [dict(target) for target in targets if isinstance(target, dict)]


def _banned_filters(*, target: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    tickers = _dedupe_strings([target.get("ticker")])
    years = [year for year in [_normalize_int(target.get("fiscal_year"))] if year is not None]
    return tickers, years


def _planner_hints_from_state(*, state: Dict[str, Any], job: Dict[str, Any], goal: str) -> Dict[str, Any]:
    planner_hints: Dict[str, Any] = {
        "metric": _normalize_text((job.get("analysis_task") or {}).get("metric"))
        or _normalize_text((state.get("analysis_task") or {}).get("metric"))
        or "",
        "queries": [],
        "doc_types": [],
        "must_include": [],
        "nice_to_include": [],
        "section_hints": [],
    }

    metric = _canonical_metric_phrase(goal, _normalize_text(state.get("original_user_query")))
    if metric and not planner_hints["metric"]:
        planner_hints["metric"] = metric

    if not planner_hints["section_hints"]:
        planner_hints["section_hints"] = _infer_section_hints(
            planner_hints["metric"],
            _normalize_text(job.get("job_type")).lower() or "fact_lookup",
        )
    if not planner_hints["must_include"]:
        planner_hints["must_include"] = _infer_must_include(planner_hints["metric"]) if planner_hints["metric"] else []

    return planner_hints


def _fallback_request(
    *,
    original_user_query: str,
    job: Dict[str, Any],
    target: Dict[str, Any],
    planner_hints: Dict[str, Any],
) -> Dict[str, Any]:
    job_type = _normalize_text(job.get("job_type")).lower() or "fact_lookup"
    goal = _normalize_text(job.get("goal")) or original_user_query
    metric_phrase = (
        _normalize_text(planner_hints.get("metric"))
        or _canonical_metric_phrase(goal, original_user_query)
    )
    section_hints = _dedupe_strings(
        _as_list(planner_hints.get("section_hints"))
        or _infer_section_hints(metric_phrase, job_type),
        limit=6,
    )

    tickers, years = _banned_filters(target=target)
    queries = [
        _sanitize_query(value, tickers=tickers, years=years)
        for value in (
            _as_list(planner_hints.get("queries"))
            or _build_default_queries(metric_phrase, section_hints)
        )
    ]
    queries = _dedupe_strings(queries, limit=_MAX_QUERIES)
    if not queries:
        queries = _build_default_queries(metric_phrase, section_hints)

    doc_types = (
        _normalize_doc_types(_as_list(planner_hints.get("doc_types")))
        or deterministic_doc_types_for_job(job_type)
        or None
    )

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
    planner_hints: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = _fallback_request(
        original_user_query=original_user_query,
        job=job,
        target=target,
        planner_hints=planner_hints,
    )
    raw_request = raw_request or {}

    tickers, years = _banned_filters(target=target)
    queries = [
        _sanitize_query(value, tickers=tickers, years=years)
        for value in _as_list(raw_request.get("queries"))
    ]
    queries = _dedupe_strings(queries, limit=_MAX_QUERIES) or fallback["queries"]

    required_doc_types = deterministic_doc_types_for_job(job.get("job_type"))
    if required_doc_types is not None:
        doc_types = required_doc_types
    else:
        normalized_doc_types = _normalize_doc_types(_as_list(raw_request.get("doc_types")))
        doc_types = normalized_doc_types or fallback["doc_types"]

    return {
        "queries": queries,
        "doc_types": doc_types,
        "reason": _normalize_text(raw_request.get("reason")) or fallback["reason"],
    }


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


def _retrieval_requirements(job: Dict[str, Any], planner_hints: Dict[str, Any]) -> Dict[str, Any]:
    job_type = _normalize_text(job.get("job_type")).lower() or "fact_lookup"
    metric = _normalize_text(planner_hints.get("metric"))
    return {
        "min_results": 1 if job_type in {"metric_extract", "component_extract", "narrative_extract"} else 2,
        "min_total_score": 0.0,
        "accept_if_contains_any": _dedupe_strings(
            _as_list(planner_hints.get("must_include")) or ([metric] if metric else []),
            limit=6,
        ),
    }


def _retrieval_text_blob(result: Dict[str, Any]) -> str:
    compact = _compact_retrieval_result(result)
    pieces: List[str] = []
    for item in compact.get("top_items", []):
        pieces.extend([
            _normalize_text(item.get("doc_id")),
            _normalize_text(item.get("section_path")),
            _normalize_text(item.get("summary")),
        ])
    return " \n ".join(piece for piece in pieces if piece).lower()


def _looks_good_enough(
    result: Dict[str, Any],
    *,
    retrieval_requirements: Dict[str, Any],
) -> bool:
    compact = _compact_retrieval_result(result)
    if not compact.get("ok"):
        return False

    min_results = int(retrieval_requirements.get("min_results") or 1)
    min_total_score = float(retrieval_requirements.get("min_total_score") or 0.0)
    if int(compact.get("num_results") or 0) < min_results:
        return False

    max_total_score = compact.get("max_total_score")
    if max_total_score is not None:
        try:
            if float(max_total_score) < min_total_score:
                return False
        except Exception:
            pass

    anchors = _dedupe_strings(_as_list(retrieval_requirements.get("accept_if_contains_any")), limit=6)
    if anchors:
        blob = _retrieval_text_blob(result)
        if not any(anchor.lower() in blob for anchor in anchors):
            return False

    return True


def _fallback_review_decision(
    *,
    request: Dict[str, Any],
    result: Dict[str, Any],
    retrieval_requirements: Dict[str, Any],
    planner_hints: Dict[str, Any],
    attempts_remaining: int,
) -> Dict[str, Any]:
    if _looks_good_enough(result, retrieval_requirements=retrieval_requirements):
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

    retry_queries = _dedupe_strings(
        _as_list(planner_hints.get("queries"))
        + _as_list(planner_hints.get("must_include"))
        + _as_list(planner_hints.get("section_hints")),
        limit=_MAX_QUERIES,
    )
    if not retry_queries:
        retry_queries = request.get("queries") or []

    return {
        "action": "retry",
        "reason": "heuristic retry",
        "queries": retry_queries[:_MAX_QUERIES],
        "doc_types": request.get("doc_types"),
    }


def _normalize_review_decision(
    raw_decision: Optional[Dict[str, Any]],
    *,
    request: Dict[str, Any],
    result: Dict[str, Any],
    target: Dict[str, Any],
    job_type: str,
    retrieval_requirements: Dict[str, Any],
    planner_hints: Dict[str, Any],
    attempts_remaining: int,
) -> Dict[str, Any]:
    fallback = _fallback_review_decision(
        request=request,
        result=result,
        retrieval_requirements=retrieval_requirements,
        planner_hints=planner_hints,
        attempts_remaining=attempts_remaining,
    )
    raw_decision = raw_decision or {}

    action = _normalize_text(raw_decision.get("action")).lower()
    if action not in _ALLOWED_ACTIONS:
        action = fallback["action"]

    required_doc_types = deterministic_doc_types_for_job(job_type)
    doc_types = required_doc_types or request.get("doc_types")
    if action == "retry":
        normalized_doc_types = _normalize_doc_types(_as_list(raw_decision.get("doc_types")))
        if required_doc_types is None and normalized_doc_types:
            doc_types = normalized_doc_types

    queries = []
    if action == "retry":
        tickers, years = _banned_filters(target=target)
        raw_queries = _as_list(raw_decision.get("queries")) or fallback["queries"]
        queries = _dedupe_strings(
            [_sanitize_query(value, tickers=tickers, years=years) for value in raw_queries],
            limit=_MAX_QUERIES,
        )
        if not queries:
            queries = request.get("queries") or fallback["queries"]

    return {
        "action": action,
        "reason": _normalize_text(raw_decision.get("reason")) or fallback["reason"],
        "queries": queries,
        "doc_types": doc_types,
    }


def _render_prompt(*, system_prompt: str, prompt_template: str, prompt_input: Dict[str, Any]) -> str:
    return (
        system_prompt.strip()
        + "\n\n"
        + prompt_template.strip()
        + "\n\nActual input:\n"
        + json.dumps(prompt_input, indent=2, ensure_ascii=False)
    )


class RetrievalToolCallingAgent:
    """
    Single-job, single-target retrieval loop.

    planner payload -> initial request -> MCP retrieval -> review -> optional retry -> final result
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        llm: Any | None = None,
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_RETRIEVAL_AGENT_SYSTEM_PROMPT,
        first_pass_prompt_template: str = DEFAULT_RETRIEVAL_FIRST_PASS_PROMPT_TEMPLATE,
        review_prompt_template: str = DEFAULT_RETRIEVAL_REVIEW_PROMPT_TEMPLATE,
        top_k: int = 3,
        min_total_score: float = 0.0,
        max_attempts: int = 2,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.first_pass_prompt_template = first_pass_prompt_template
        self.review_prompt_template = review_prompt_template
        self.top_k = int(top_k)
        self.min_total_score = float(min_total_score)
        self.max_attempts = max(1, int(max_attempts))
        self.timeout_s = float(timeout_s)
        self.llm = llm or build_chat_model(model=model, temperature=temperature)

    def _build_first_pass_input(
        self,
        *,
        state: Dict[str, Any],
        job: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        job_type = _normalize_text(job.get("job_type")).lower() or "fact_lookup"
        if job_type not in _JOB_TYPES:
            job_type = "fact_lookup"
        goal = _normalize_text(job.get("goal")) or _normalize_text(state.get("original_user_query"))
        planner_hints = _planner_hints_from_state(state=state, job=job, goal=goal)

        return {
            "original_user_query": _normalize_text(state.get("original_user_query")),
            "clarification_history": [
                dict(turn)
                for turn in (state.get("clarification_history") or [])
                if isinstance(turn, dict)
            ],
            "job": {
                "job_type": job_type,
                "goal": goal,
            },
            "target": {
                "target_id": target.get("target_id"),
                "ticker": _normalize_text(target.get("ticker")),
                "fiscal_year": _normalize_int(target.get("fiscal_year")),
                "form_type": _normalize_text(target.get("form_type")) or _DEFAULT_FORM_TYPE,
            },
            "planner_hints": planner_hints,
            "required_doc_types": deterministic_doc_types_for_job(job_type),
        }

    def _build_review_input(
        self,
        *,
        first_pass_input: Dict[str, Any],
        request: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        attempt_index: int,
    ) -> Dict[str, Any]:
        planner_hints = dict(first_pass_input.get("planner_hints") or {})
        retrieval_requirements = _retrieval_requirements(
            first_pass_input.get("job") or {},
            planner_hints,
        )
        return {
            **first_pass_input,
            "attempt_index": int(attempt_index),
            "attempts_remaining": max(self.max_attempts - int(attempt_index), 0),
            "request_used": {
                "queries": list(request.get("queries") or []),
                "doc_types": request.get("doc_types"),
                "top_k": self.top_k,
                "min_total_score": self.min_total_score,
            },
            "retrieval_requirements": retrieval_requirements,
            "retrieval_result": _compact_retrieval_result(retrieval_result),
        }

    def _llm_json(self, prompt: str) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        raw_output = ""
        error = None
        parsed = None
        try:
            response = self.llm.invoke(prompt)
            raw_output = response.content if hasattr(response, "content") else str(response)
            parsed = _extract_first_json_object(raw_output)
            if parsed is None:
                error = "PARSE_FAILED"
        except Exception as exc:
            error = f"LLM_CALL_FAILED: {exc}"
        return parsed, raw_output, error

    async def run_single_target(
        self,
        *,
        state: Dict[str, Any],
        job: Dict[str, Any],
        target: Dict[str, Any],
        client: Any,
    ) -> Dict[str, Any]:
        first_pass_input = self._build_first_pass_input(state=state, job=job, target=target)
        planner_hints = dict(first_pass_input.get("planner_hints") or {})
        review_retrieval_requirements = _retrieval_requirements(first_pass_input["job"], planner_hints)

        first_prompt = _render_prompt(
            system_prompt=self.system_prompt,
            prompt_template=self.first_pass_prompt_template,
            prompt_input=first_pass_input,
        )
        parsed_first, raw_first, first_error = self._llm_json(first_prompt)
        request = _normalize_initial_request(
            parsed_first,
            original_user_query=first_pass_input["original_user_query"],
            job=first_pass_input["job"],
            target=first_pass_input["target"],
            planner_hints=planner_hints,
        )

        attempts: List[Dict[str, Any]] = []
        first_result = await client.retrieve_tables(
            queries=request["queries"],
            ticker=first_pass_input["target"]["ticker"],
            fiscal_year=first_pass_input["target"]["fiscal_year"],
            form_type=first_pass_input["target"]["form_type"],
            doc_types=request.get("doc_types"),
            top_k=self.top_k,
            min_total_score=self.min_total_score,
            timeout_s=self.timeout_s,
        )
        attempts.append(
            {
                "attempt_index": 1,
                "request": request,
                "retrieval": first_result,
                "retrieval_compact": _compact_retrieval_result(first_result),
            }
        )

        review_input = self._build_review_input(
            first_pass_input=first_pass_input,
            request=request,
            retrieval_result=first_result,
            attempt_index=1,
        )
        review_prompt = _render_prompt(
            system_prompt=self.system_prompt,
            prompt_template=self.review_prompt_template,
            prompt_input=review_input,
        )
        parsed_review, raw_review, review_error = self._llm_json(review_prompt)
        review = _normalize_review_decision(
            parsed_review,
            request=request,
            result=first_result,
            target=first_pass_input["target"],
            job_type=first_pass_input["job"]["job_type"],
            retrieval_requirements=review_retrieval_requirements,
            planner_hints=planner_hints,
            attempts_remaining=max(self.max_attempts - 1, 0),
        )

        final_result = first_result
        final_action = "return"

        if review["action"] == "retry" and self.max_attempts > 1:
            retry_request = {
                "queries": _dedupe_strings(review.get("queries") or request["queries"], limit=_MAX_QUERIES),
                "doc_types": review.get("doc_types"),
                "reason": review.get("reason") or "retry",
            }
            retry_result = await client.retrieve_tables(
                queries=retry_request["queries"],
                ticker=first_pass_input["target"]["ticker"],
                fiscal_year=first_pass_input["target"]["fiscal_year"],
                form_type=first_pass_input["target"]["form_type"],
                doc_types=retry_request.get("doc_types"),
                top_k=self.top_k,
                min_total_score=self.min_total_score,
                timeout_s=self.timeout_s,
            )
            attempts.append(
                {
                    "attempt_index": 2,
                    "request": retry_request,
                    "retrieval": retry_result,
                    "retrieval_compact": _compact_retrieval_result(retry_result),
                }
            )
            final_result = retry_result
            final_action = "return_after_retry"

        return {
            "job_type": first_pass_input["job"]["job_type"],
            "goal": first_pass_input["job"]["goal"],
            "target": first_pass_input["target"],
            "planner_hints": planner_hints,
            "first_pass_prompt_input": first_pass_input,
            "first_pass_prompt": first_prompt,
            "first_pass_raw_output": raw_first,
            "first_pass_error": first_error,
            "review_prompt_input": review_input,
            "review_prompt": review_prompt,
            "review_raw_output": raw_review,
            "review_error": review_error,
            "review_decision": review,
            "attempts": attempts,
            "final_action": final_action,
            "retrieval": final_result,
        }

    async def run(self, state: Dict[str, Any], client: Any) -> Dict[str, Any]:
        retrieval_state = dict(state or {})
        retrieval_plan = dict(retrieval_state.get("retrieval_plan") or {})
        jobs = [dict(job) for job in (retrieval_plan.get("jobs") or []) if isinstance(job, dict)]
        targets = [dict(target) for target in (retrieval_state.get("targets") or []) if isinstance(target, dict)]

        runs: List[Dict[str, Any]] = []
        for job_index, job in enumerate(jobs, start=1):
            matched_targets = _targets_for_job(targets=targets, job=job)
            for target in matched_targets:
                ticker = _normalize_text(target.get("ticker"))
                fiscal_year = _normalize_int(target.get("fiscal_year"))
                form_type = _normalize_text(target.get("form_type")) or _DEFAULT_FORM_TYPE
                if not ticker or fiscal_year is None:
                    runs.append(
                        {
                            "job_index": job_index,
                            "applies_to_target_ids": _dedupe_ints(_as_list(job.get("applies_to_target_ids")), limit=64),
                            "job_type": _normalize_text(job.get("job_type")) or "fact_lookup",
                            "goal": _normalize_text(job.get("goal")) or _normalize_text(retrieval_state.get("original_user_query")),
                            "target": {
                                "target_id": target.get("target_id"),
                                "ticker": ticker,
                                "fiscal_year": fiscal_year,
                                "form_type": form_type,
                            },
                            "planner_hints": _planner_hints_from_state(
                                state=retrieval_state,
                                job=job,
                                goal=_normalize_text(job.get("goal")) or _normalize_text(retrieval_state.get("original_user_query")),
                            ),
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
                                    "ticker": ticker,
                                    "fiscal_year": fiscal_year,
                                    "form_type": form_type,
                                },
                                "error": "MISSING_TARGET_METADATA",
                            },
                        }
                    )
                    continue

                target_run = await self.run_single_target(
                    state=retrieval_state,
                    job=job,
                    target=target,
                    client=client,
                )
                runs.append(
                    {
                        "job_index": job_index,
                        "applies_to_target_ids": _dedupe_ints(_as_list(job.get("applies_to_target_ids")), limit=64),
                        **target_run,
                    }
                )

        return {
            "original_user_query": _normalize_text(retrieval_state.get("original_user_query")),
            "runs": runs,
        }

    def run_sync(self, state: Dict[str, Any], client: Any) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(state, client))
        raise RuntimeError("run_sync() cannot be used inside an existing event loop; use await run(...)")


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
            "retrieval_plan": dict(state.get("retrieval_plan") or {}),
            "planner_hints": dict(state.get("planner_hints") or {}),
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
    agent: RetrievalToolCallingAgent | None = None,
) -> Dict[str, Any]:
    retrieval_model = (
        _normalize_text(state.get("retrieval_agent_model"))
        or _normalize_text(state.get("retrieval_query_planner_model"))
        or _DEFAULT_MODEL
    )
    tool_agent = agent or RetrievalToolCallingAgent(
        model=retrieval_model,
        top_k=3,
        min_total_score=0.0,
        max_attempts=2,
    )

    if client is None:
        from agents.retrieval.mcp_client import SecRetrievalMCPClient

        async with SecRetrievalMCPClient() as created_client:
            runs_payload = await tool_agent.run(state, created_client)
    else:
        runs_payload = await tool_agent.run(state, client)

    retrieval_output = _build_retrieval_output(
        state=state,
        runs_payload=runs_payload,
        model=retrieval_model,
    )
    return {**state, "retrieval": retrieval_output}


__all__ = [
    "DEFAULT_RETRIEVAL_AGENT_SYSTEM_PROMPT",
    "DEFAULT_RETRIEVAL_FIRST_PASS_PROMPT_TEMPLATE",
    "DEFAULT_RETRIEVAL_REVIEW_PROMPT_TEMPLATE",
    "RetrievalToolCallingAgent",
    "deterministic_doc_types_for_job",
    "retrieval_agent",
]
