from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from agents.contracts import AnalysisTask, FilingMetadata
from agents.planner.agent import (
    _DEFAULT_COMPANY_TICKER_MAP,
    _extract_first_json_object,
    _extract_metadata_hints_and_issues,
    _guess_metric,
    _resolve_ticker_from_company_name,
    _intent_hint_from_query,
)
from llm_client import build_chat_model


DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE = """
You are the Target Resolver Planner for an SEC-filings Agentic RAG system.

Very important system design:
Before you are called, a deterministic resolver has already attempted to extract:
- ticker
- fiscal_year
- form_type

Those deterministic results are provided to you as input.

Your role is:
- trust deterministic values when they are resolved
- never overwrite a resolved deterministic ticker, fiscal_year, or form_type
- only use language reasoning to fill gaps when a blocking field could not be resolved deterministically
- if the meaning is still too ambiguous to resolve correctly, ask for clarification through the JSON fields below
- produce a compact retrieval handoff for the retriever

Because this is a programmatic setting:
- NEVER ask the user questions outside the JSON
- If clarification is needed, set `needs_clarification=true` and populate `clarification_questions`
- Always still return one valid JSON object

Return EXACTLY ONE valid JSON object.
Do not return markdown.
Do not return code fences.
Do not return commentary.
Do not include any keys outside the schema.

Rules:
1. Never invent a company, ticker, fiscal year, or form type.
2. If deterministic extraction resolved ticker, fiscal_year, or form_type, copy those values exactly.
3. Do not "improve", reinterpret, or replace resolved deterministic values.
4. Only infer ticker when deterministic_ticker is null and the query strongly implies a single canonical company.
5. Only infer fiscal_year when deterministic_fiscal_year is null and the query explicitly or strongly implies one year.
6. If the user references a supported group alias, expand it to the underlying canonical entities.
7. If form_type is unresolved and not clearly implied, return form_type=null rather than guessing.
8. When multiple fiscal years apply to the same resolved company set, emit one target per (company, fiscal_year) pair.
9. If clarification is needed, DO NOT return anything to the retrieval plan.
10. `goal` must be a short, atomic, retrieval-only evidence request.
11. The `goal` field is a retrieval-level instruction dispatched to individual filings.
    It must NOT reference specific company names, tickers, or fiscal years.
    It should describe WHAT to extract from a single filing (e.g., "extract annual revenue").
    Comparison, ranking, and aggregation happen downstream — not in the goal.
12. When all targets require the same extraction, emit exactly ONE job whose
    `applies_to_target_ids` lists every target_id. Do NOT create separate jobs
    per company or per company-year pair unless the extraction goal differs.
13. `goal` should be generic and reusable without company name or fiscal year.
14. Do not mention company name(s) or year-to-year comparison language inside `goal`.
15. If `needs_clarification=true`, return `targets=[]`.
16. Do not emit partial, provisional, or guessed targets before clarification is resolved.
17. `target_id` must be an integer. Use sequential integers starting at 1.
18. `target_key` must be a stable readable string key:
    - if ticker and fiscal_year are known: "{TICKER}_FY{YEAR}"
    - if ticker is known but fiscal_year is unknown: "{TICKER}_UNKNOWN_YEAR"
    - if ticker is unknown: "TARGET_{N}"

How to use the deterministic input:
- deterministic_targets: authoritative target candidates from rules/regex/maps
- deterministic_ticker: authoritative if non-null
- deterministic_fiscal_year: authoritative if non-null
- deterministic_form_type: authoritative if non-null
- unresolved_blockers: tells you which blocking fields remain unresolved after deterministic extraction
- clarification_history contains prior clarification questions and the user's answers; treat the answers as authoritative user input
- If unresolved_blockers is empty, do not perform gap filling for ticker or fiscal_year.
- If clarification_history resolves a blocker, do not ask the same question again.
- If unresolved_blockers contains "ticker" or "fiscal_year", you may use the user query plus alias maps to try to fill only those missing fields.

Output schema:
{
  "retrieval_needed": boolean,
  "task_class": "single_target_fact" | "multi_target_compare" | "multi_target_screen" | "other",
  "targets": [
    {
      "target_id": integer,
      "target_key": string,
      "company_name": string | null,
      "ticker": string | null,
      "fiscal_year": integer | null,
      "form_type": "10-K" | "10-Q" | null
    }
  ],
  "retrieval_plan": {
    "fanout_mode": "single_target" | "per_target",
    "jobs": [
      {
        "applies_to_target_ids": [integer],
        "goal": string,
        "job_type": "metric_extract" | "narrative_extract"
      }
    ]
  },
  "needs_clarification": boolean,
  "clarification_reason": string | null,
  "clarification_questions": [string],
  "open_issues": [
    {
      "code": string,
      "message": string,
      "severity": "info" | "warning" | "error"
    }
  ]
}


Return exactly one JSON object matching the schema.
"""


_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_ALLOWED_FORM_TYPES = {"10-K", "10-Q"}
_ALLOWED_TASK_CLASSES = {
    "single_target_fact",
    "multi_target_compare",
    "multi_target_screen",
    "other",
}
_ALLOWED_SEVERITIES = {"info", "warning", "error"}
_MULTI_TARGET_TASK_CLASSES = {"multi_target_compare", "multi_target_screen"}
_ALLOWED_JOB_TYPES = {
    "fact_lookup",
    "metric_extract",
    "component_extract",
    "narrative_extract",
}


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_form_type(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    if text is None:
        return None
    upper = text.upper()
    return upper if upper in _ALLOWED_FORM_TYPES else None


def _build_target_key(
    *,
    ticker: Optional[str],
    fiscal_year: Optional[int],
    index: int,
) -> str:
    if ticker and fiscal_year is not None:
        return f"{ticker}_FY{fiscal_year}"
    if ticker:
        return f"{ticker}_UNKNOWN_YEAR"
    return f"TARGET_{index}"


def _dedupe_ints(values: Sequence[Any], *, limit: Optional[int] = None) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        number = _normalize_int(value)
        if number is None:
            continue
        if number in seen:
            continue
        seen.add(number)
        out.append(number)
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_open_issue(issue: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(issue, dict):
        return None
    code = _normalize_text(issue.get("code")) or "UNSPECIFIED"
    message = _normalize_text(issue.get("message")) or ""
    severity = (_normalize_text(issue.get("severity")) or "warning").lower()
    if severity not in _ALLOWED_SEVERITIES:
        severity = "warning"
    return {
        "code": code,
        "message": message,
        "severity": severity,
    }


def _dedupe_strings(values: Sequence[Any], *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _normalize_text(value)
        if text is None:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_target(target: Any, *, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(target, dict):
        return None

    ticker = _normalize_text(target.get("ticker"))
    company_name = _normalize_text(target.get("company_name"))
    fiscal_year = _normalize_int(target.get("fiscal_year"))
    form_type = _normalize_form_type(target.get("form_type"))
    target_id = _normalize_int(target.get("target_id"))
    if target_id is None:
        target_id = index
    target_key = _normalize_text(target.get("target_key")) or _build_target_key(
        ticker=ticker,
        fiscal_year=fiscal_year,
        index=index,
    )

    return {
        "target_id": target_id,
        "target_key": target_key,
        "company_name": company_name,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "form_type": form_type,
    }


def _normalize_retrieval_job(
    job: Any,
    *,
    target_ids: Sequence[int],
) -> Optional[Dict[str, Any]]:
    if not isinstance(job, dict):
        return None

    goal = _normalize_text(job.get("goal"))
    if goal is None:
        return None

    applies_to_target_ids = _dedupe_ints(job.get("applies_to_target_ids") or [])
    valid_target_ids = set(target_ids)
    applies_to_target_ids = [target_id for target_id in applies_to_target_ids if target_id in valid_target_ids]
    if not applies_to_target_ids:
        applies_to_target_ids = list(target_ids)

    job_type = (_normalize_text(job.get("job_type")) or "fact_lookup").strip().lower()
    if job_type not in _ALLOWED_JOB_TYPES:
        job_type = "fact_lookup"

    return {
        "applies_to_target_ids": applies_to_target_ids,
        "goal": goal,
        "job_type": job_type,
    }


def _normalize_retrieval_plan(
    retrieval_plan: Any,
    *,
    targets: Sequence[Dict[str, Any]],
    needs_clarification: bool,
) -> Optional[Dict[str, Any]]:
    if needs_clarification or not isinstance(retrieval_plan, dict):
        return None

    target_ids = [int(target["target_id"]) for target in targets if _normalize_int(target.get("target_id")) is not None]
    if not target_ids:
        return None

    fanout_mode = (_normalize_text(retrieval_plan.get("fanout_mode")) or "").strip().lower()
    if fanout_mode not in {"single_target", "per_target"}:
        fanout_mode = "single_target" if len(target_ids) == 1 else "per_target"

    jobs = [
        normalized
        for normalized in (
            _normalize_retrieval_job(job, target_ids=target_ids)
            for job in (retrieval_plan.get("jobs") or [])
        )
        if normalized is not None
    ]
    if not jobs:
        return None

    return {
        "fanout_mode": fanout_mode,
        "jobs": jobs,
    }


def _normalize_clarification_turns(
    clarification_turns: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for turn in clarification_turns or []:
        if not isinstance(turn, dict):
            continue
        question = _normalize_text(turn.get("question"))
        answer = _normalize_text(turn.get("answer"))
        if not question:
            continue
        out.append(
            {
                "question": question,
                "answer": answer or "",
            }
        )
    return out


def _format_clarification_context(clarification_turns: List[Dict[str, str]]) -> str:
    if not clarification_turns:
        return ""
    lines = ["Clarification history:"]
    for turn in clarification_turns:
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)


def _build_deterministic_targets(hints: Any) -> List[Dict[str, Any]]:
    ticker = _normalize_text(getattr(hints, "ticker", None))
    company_name = _normalize_text(getattr(hints, "company_name", None))
    fiscal_year = _normalize_int(getattr(hints, "fiscal_year", None))
    form_type = _normalize_form_type(getattr(hints, "form_type", None))

    if not any([ticker, company_name, fiscal_year is not None, form_type]):
        return []

    return [
        {
            "target_id": 1,
            "target_key": _build_target_key(ticker=ticker, fiscal_year=fiscal_year, index=1),
            "company_name": company_name,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "form_type": form_type,
        }
    ]


def _build_planner_state(
    *,
    user_query: str,
    effective_user_query: str,
    clarification_history: List[Dict[str, str]],
    deterministic_targets: List[Dict[str, Any]],
    deterministic_ticker: Optional[str],
    deterministic_fiscal_year: Optional[int],
    deterministic_form_type: Optional[str],
    unresolved_blockers: List[str],
) -> Dict[str, Any]:
    return {
        "original_user_query": user_query,
        "clarification_history": clarification_history,
        "deterministic_targets": deterministic_targets,
        "deterministic_ticker": deterministic_ticker,
        "deterministic_fiscal_year": deterministic_fiscal_year,
        "deterministic_form_type": deterministic_form_type,
        "unresolved_blockers": unresolved_blockers,
        "effective_user_query": effective_user_query,
    }


def build_target_resolution_payload(
    *,
    planner: Any,
    user_query: str,
    clarification_turns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    clarification_history = _normalize_clarification_turns(clarification_turns)
    clarification_context = _format_clarification_context(clarification_history)

    effective_user_query = user_query.strip()
    if clarification_context:
        effective_user_query = f"{effective_user_query}\n\n{clarification_context}"

    hints, issues = _extract_metadata_hints_and_issues(
        effective_user_query,
        company_ticker_map=planner.company_ticker_map,
    )
    metric_guess = _guess_metric(effective_user_query)
    intent_hint, task_type_hint, retrieval_needed_hint, calc_cues = _intent_hint_from_query(
        effective_user_query,
        metric_guess,
    )

    unresolved_blockers: List[str] = []
    if getattr(hints, "ticker", None) is None:
        unresolved_blockers.append("ticker")
    if getattr(hints, "fiscal_year", None) is None:
        unresolved_blockers.append("fiscal_year")
    deterministic_targets = _build_deterministic_targets(hints)
    deterministic_ticker = getattr(hints, "ticker", None)
    deterministic_fiscal_year = getattr(hints, "fiscal_year", None)
    deterministic_form_type = _normalize_form_type(getattr(hints, "form_type", None))

    payload = {
        "user_query": user_query,
        "effective_user_query": effective_user_query,
        "clarification_history": clarification_history,
        "deterministic_targets": deterministic_targets,
        "deterministic_ticker": deterministic_ticker,
        "deterministic_fiscal_year": deterministic_fiscal_year,
        "deterministic_form_type": deterministic_form_type,
        "deterministic_hints": hints.model_dump(mode="json"),
        "deterministic_open_issues": [issue.model_dump(mode="json") for issue in issues],
        "unresolved_blockers": unresolved_blockers,
        "metric_guess": metric_guess,
        "deterministic_intent_hint": intent_hint.value,
        "deterministic_task_type_hint": task_type_hint,
        "deterministic_retrieval_needed_hint": retrieval_needed_hint,
        "deterministic_calc_cues": calc_cues,
    }

    planner_state = _build_planner_state(
        user_query=user_query,
        effective_user_query=effective_user_query,
        clarification_history=clarification_history,
        deterministic_targets=deterministic_targets,
        deterministic_ticker=deterministic_ticker,
        deterministic_fiscal_year=deterministic_fiscal_year,
        deterministic_form_type=deterministic_form_type,
        unresolved_blockers=unresolved_blockers,
    )

    return {
        "hints": hints,
        "issues": issues,
        "metric_guess": metric_guess,
        "payload": payload,
        "planner_state": planner_state,
        "effective_user_query": effective_user_query,
        "clarification_history": clarification_history,
    }


def render_target_resolution_prompt(
    prompt_template: str,
    *,
    user_query: str,
    payload: Dict[str, Any],
) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    prompt = str(prompt_template or "")
    prompt = prompt.replace("{{USER_QUERY}}", user_query)
    if "{{PLANNER_PAYLOAD_JSON}}" in prompt:
        prompt = prompt.replace("{{PLANNER_PAYLOAD_JSON}}", payload_json)
    else:
        prompt = prompt + "\n\n" + payload_json
    return prompt


def _normalize_resolution_output(parsed_output: Any) -> Dict[str, Any]:
    if not isinstance(parsed_output, dict):
        raise ValueError("Parsed output must be a JSON object.")

    task_class = _normalize_text(parsed_output.get("task_class")) or "other"
    if task_class not in _ALLOWED_TASK_CLASSES:
        task_class = "other"

    targets: List[Dict[str, Any]] = []
    for index, target in enumerate(parsed_output.get("targets") or [], start=1):
        normalized = _normalize_target(target, index=index)
        if normalized is not None:
            targets.append(normalized)

    clarification_questions = [
        question
        for question in (
            _normalize_text(question) for question in (parsed_output.get("clarification_questions") or [])
        )
        if question
    ]

    needs_clarification = _normalize_bool(parsed_output.get("needs_clarification"))
    if not needs_clarification:
        clarification_questions = []
    else:
        targets = []

    retrieval_plan = _normalize_retrieval_plan(
        parsed_output.get("retrieval_plan"),
        targets=targets,
        needs_clarification=needs_clarification,
    )

    open_issues = [
        issue
        for issue in (_normalize_open_issue(issue) for issue in (parsed_output.get("open_issues") or []))
        if issue is not None
    ]

    return {
        "retrieval_needed": _normalize_bool(parsed_output.get("retrieval_needed")),
        "task_class": task_class,
        "targets": targets,
        "retrieval_plan": retrieval_plan,
        "needs_clarification": needs_clarification,
        "clarification_reason": _normalize_text(parsed_output.get("clarification_reason")),
        "clarification_questions": clarification_questions,
        "open_issues": open_issues,
    }


def _build_default_retrieval_plan(
    *,
    targets: Sequence[Dict[str, Any]],
    metric_guess: str,
    user_query: str,
) -> Optional[Dict[str, Any]]:
    target_ids = [int(target["target_id"]) for target in targets if _normalize_int(target.get("target_id")) is not None]
    if not target_ids:
        return None

    goal = _normalize_text(metric_guess) or _normalize_text(user_query) or "extract relevant filing evidence"
    return {
        "fanout_mode": "single_target" if len(target_ids) == 1 else "per_target",
        "jobs": [
            {
                "applies_to_target_ids": target_ids,
                "goal": goal,
                "job_type": "fact_lookup",
            }
        ],
    }


def _build_fallback_target_resolution(*, target_run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    planner_state = dict(target_run.get("planner_state") or {})
    deterministic_targets = [
        dict(target) for target in (planner_state.get("deterministic_targets") or [])
        if isinstance(target, dict)
    ]
    if not deterministic_targets:
        return None

    for target in deterministic_targets:
        if not _normalize_text(target.get("ticker")) and target.get("company_name"):
            resolved = _resolve_ticker_from_company_name(
                target.get("company_name"),
                _DEFAULT_COMPANY_TICKER_MAP,
            )
            if resolved is not None:
                target["ticker"] = resolved

    intent_hint = _normalize_text(target_run.get("deterministic_intent_hint"))
    retrieval_needed_hint = target_run.get("deterministic_retrieval_needed_hint")
    metric_guess = _normalize_text(target_run.get("metric_guess")) or "filing evidence"
    query_text = str(
        planner_state.get("original_user_query")
        or planner_state.get("effective_user_query")
        or ""
    ).lower()
    filing_signal = any(
        token in query_text
        for token in [
            "10-k",
            "10-k",
            "10 k",
            "annual report",
            "fiscal",
            "filing",
            "revenue",
            "profit",
            "segment",
            "debt",
            "loan",
            "asset",
            "liquid",
            "risk",
            "credit",
        ]
    ) and bool(planner_state.get("deterministic_targets") or [])
    needs_filing_retrieval = intent_hint in {"filing_fact", "filing_calc"} or filing_signal
    retrieval_needed = (
        bool(retrieval_needed_hint)
        if retrieval_needed_hint is not None
        else bool(needs_filing_retrieval)
    )
    if not retrieval_needed and needs_filing_retrieval:
        retrieval_needed = True

    retrieval_plan = None
    if retrieval_needed:
        retrieval_plan = _build_default_retrieval_plan(
            targets=deterministic_targets,
            metric_guess=metric_guess,
            user_query=str(
                planner_state.get("original_user_query")
                or planner_state.get("effective_user_query")
                or ""
            ).strip(),
        )

    return {
        "retrieval_needed": bool(retrieval_needed),
        "task_class": "single_target_fact" if len(deterministic_targets) == 1 else "other",
        "targets": deterministic_targets,
        "retrieval_plan": retrieval_plan,
        "needs_clarification": False,
        "clarification_reason": None,
        "clarification_questions": [],
        "open_issues": [
            {
                "code": "PLANNER_LLM_FALLBACK",
                "message": (
                    "Planner LLM output could not be produced or validated; "
                    "using deterministic target metadata fallback."
                ),
                "severity": "warning",
            }
        ],
    }


def _build_metadata(
    *,
    targets: Sequence[Dict[str, Any]],
    deterministic_hints: Dict[str, Any],
) -> Dict[str, Any]:
    first_target = dict(targets[0]) if targets else {}
    fiscal_years = sorted(
        {
            int(year)
            for year in (
                _normalize_int(target.get("fiscal_year"))
                for target in targets
            )
            if year is not None
        }
    )
    form_types = _dedupe_strings(
        _normalize_form_type(target.get("form_type"))
        for target in targets
    )

    metadata = FilingMetadata(
        ticker=(
            first_target.get("ticker")
            if len(targets) == 1
            else _normalize_text(deterministic_hints.get("ticker"))
        ),
        company_name=(
            first_target.get("company_name")
            if len(targets) == 1
            else _normalize_text(deterministic_hints.get("company_name"))
        ),
        fiscal_year=fiscal_years[0] if len(fiscal_years) == 1 else _normalize_int(deterministic_hints.get("fiscal_year")),
        form_type=_normalize_form_type(
            first_target.get("form_type")
            if len(targets) == 1
            else (form_types[0] if len(form_types) == 1 else deterministic_hints.get("form_type"))
        )
        or "10-K",
        doc_types=deterministic_hints.get("doc_types"),
        fiscal_quarter=deterministic_hints.get("fiscal_quarter"),
    )
    return metadata.model_dump(mode="json", exclude_none=False)


def _build_analysis_task(
    *,
    task_class: str,
    metric_guess: str,
    retrieval_plan: Optional[Dict[str, Any]],
    task_type_hint: str,
) -> Dict[str, Any]:
    metric = _normalize_text(metric_guess)
    if metric is None:
        metric = _normalize_text((((retrieval_plan or {}).get("jobs") or [{}])[0]).get("goal")) or "filing evidence"

    task_type = (_normalize_text(task_type_hint) or "extract").lower()
    if task_class in _MULTI_TARGET_TASK_CLASSES:
        task_type = "compare"
    elif task_type not in {"extract", "compute", "compare", "trend"}:
        task_type = "extract"

    analysis_task = AnalysisTask(
        task_type=task_type,
        metric=metric,
        definition_notes=[],
        expected_artifacts=["table"],
        output_format="table" if task_class in _MULTI_TARGET_TASK_CLASSES else "short_answer",
    )
    return analysis_task.model_dump(mode="json")


def _merge_open_issues(*issue_lists: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for issue_list in issue_lists:
        for issue in issue_list or []:
            normalized = _normalize_open_issue(issue)
            if normalized is None:
                continue
            key = (
                normalized["code"],
                normalized["message"],
                normalized["severity"],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
    return merged


def _build_planner_output(
    *,
    status: str,
    target_run: Dict[str, Any],
    target_resolution: Optional[Dict[str, Any]],
    clarification_request: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    planner_state = dict(target_run.get("planner_state") or {})
    deterministic_hints = dict(target_run.get("deterministic_hints") or {})
    metric_guess = _normalize_text(target_run.get("metric_guess")) or "filing evidence"
    task_class = _normalize_text((target_resolution or {}).get("task_class")) or "other"
    targets = list((target_resolution or {}).get("targets") or [])

    retrieval_needed = bool((target_resolution or {}).get("retrieval_needed")) and status == "completed"
    retrieval_plan = (target_resolution or {}).get("retrieval_plan")
    if retrieval_needed and retrieval_plan is None:
        retrieval_plan = _build_default_retrieval_plan(
            targets=targets,
            metric_guess=metric_guess,
            user_query=str(planner_state.get("original_user_query") or target_run.get("user_query") or "").strip(),
        )
    if status != "completed":
        retrieval_plan = None

    intent = _normalize_text(target_run.get("deterministic_intent_hint")) or "filing_fact"
    if intent not in {"filing_fact", "filing_calc", "definition", "other"}:
        intent = "other"

    analysis_task = _build_analysis_task(
        task_class=task_class,
        metric_guess=metric_guess,
        retrieval_plan=retrieval_plan,
        task_type_hint=_normalize_text(target_run.get("deterministic_task_type_hint")) or "extract",
    )
    metadata = _build_metadata(targets=targets, deterministic_hints=deterministic_hints)
    open_issues = _merge_open_issues(
        target_run.get("deterministic_open_issues") or [],
        (target_resolution or {}).get("open_issues") or [],
    )

    return {
        "status": status,
        "retrieval_needed": retrieval_needed,
        "intent": intent,
        "metadata": metadata,
        "analysis_task": analysis_task,
        "open_issues": open_issues,
        "task_class": task_class,
        "targets": targets,
        "retrieval_plan": retrieval_plan,
        "original_user_query": str(planner_state.get("original_user_query") or target_run.get("user_query") or "").strip(),
        "clarification_history": list(planner_state.get("clarification_history") or []),
        "clarification_request": clarification_request,
        "effective_user_query": str(planner_state.get("effective_user_query") or "").strip(),
    }


def run_target_resolution_prompt(
    *,
    planner: Any,
    user_query: str,
    prompt_template: str,
    clarification_turns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pre_llm = build_target_resolution_payload(
        planner=planner,
        user_query=user_query,
        clarification_turns=clarification_turns,
    )
    prompt = render_target_resolution_prompt(
        prompt_template,
        user_query=user_query,
        payload=pre_llm["payload"],
    )

    raw_output = ""
    parsed_output: Optional[Dict[str, Any]] = None
    llm_error: Optional[str] = None
    validation_error: Optional[str] = None
    final_resolution: Optional[Dict[str, Any]] = None

    try:
        response = planner.llm.invoke(prompt)
        raw_output = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        llm_error = f"LLM_CALL_FAILED: {exc}"

    if llm_error is None:
        parsed_output = _extract_first_json_object(raw_output)
        if parsed_output is None:
            validation_error = "PARSE_FAILED"
        else:
            try:
                final_resolution = _normalize_resolution_output(parsed_output)
            except Exception as exc:
                validation_error = f"VALIDATION_FAILED: {exc}"

    return {
        "user_query": user_query,
        "planner_state": pre_llm["planner_state"],
        "effective_user_query": pre_llm["effective_user_query"],
        "clarification_history": pre_llm["clarification_history"],
        "deterministic_hints": pre_llm["hints"].model_dump(mode="json", exclude_none=True),
        "deterministic_open_issues": [issue.model_dump(mode="json") for issue in pre_llm["issues"]],
        "metric_guess": pre_llm["metric_guess"],
        "deterministic_intent_hint": pre_llm["payload"].get("deterministic_intent_hint"),
        "deterministic_task_type_hint": pre_llm["payload"].get("deterministic_task_type_hint"),
        "deterministic_retrieval_needed_hint": pre_llm["payload"].get("deterministic_retrieval_needed_hint"),
        "deterministic_calc_cues": list(pre_llm["payload"].get("deterministic_calc_cues") or []),
        "expanded_queries": [],
        "expansion_error": None,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "llm_error": llm_error,
        "validation_error": validation_error,
        "used_fallback": False,
        "needs_clarification": bool(final_resolution and final_resolution.get("needs_clarification")),
        "clarification_questions": list((final_resolution or {}).get("clarification_questions") or []),
        "final_resolution": final_resolution,
        "final_plan": final_resolution,
    }


def _coerce_answers(
    answers: Any,
    *,
    questions: Sequence[str],
) -> List[str]:
    if isinstance(answers, str):
        if len(questions) != 1:
            raise ValueError("Expected one answer per clarification question.")
        return [answers.strip()]

    if isinstance(answers, Sequence):
        answer_list = [str(answer).strip() for answer in answers]
        if len(answer_list) != len(questions):
            raise ValueError("Number of answers must match number of clarification questions.")
        return answer_list

    raise TypeError("answers must be a string or a sequence of strings.")


class InteractivePlannerAgent:
    """
    Primary planner implementation with a structured clarification loop.
    """

    def __init__(
        self,
        planner: Optional[Any] = None,
        *,
        llm: Optional[Any] = None,
        model: str = "qwen2.5-14b-instruct-1m",
        temperature: float = 0.0,
        target_resolution_prompt_template: str = DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE,
        default_doc_types: Optional[List[str]] = None,
        company_ticker_map: Optional[Dict[str, str]] = None,
        auto_run_full_planner: bool = False,
        full_planner_include_trace: bool = False,
        log_timing: bool = True,
        **planner_kwargs: Any,
    ) -> None:
        del planner_kwargs
        inherited_doc_types = getattr(planner, "default_doc_types", None)
        inherited_company_ticker_map = getattr(planner, "company_ticker_map", None) or {}
        self.llm = llm or getattr(planner, "llm", None) or build_chat_model(
            model=model,
            temperature=temperature,
        )
        self.default_doc_types = (
            list(default_doc_types)
            if default_doc_types is not None
            else (list(inherited_doc_types) if inherited_doc_types is not None else None)
        )
        self.company_ticker_map = {
            **_DEFAULT_COMPANY_TICKER_MAP,
            **dict(inherited_company_ticker_map),
            **(company_ticker_map or {}),
        }
        self.target_resolution_prompt_template = str(target_resolution_prompt_template or "")
        self.auto_run_full_planner = bool(auto_run_full_planner)
        self.full_planner_include_trace = bool(full_planner_include_trace)
        self.log_timing = bool(log_timing)
        self.last_timing_ms: Dict[str, int] = {}

    def start(self, user_query: str) -> Dict[str, Any]:
        target_run = run_target_resolution_prompt(
            planner=self,
            user_query=user_query,
            prompt_template=self.target_resolution_prompt_template,
            clarification_turns=[],
        )
        return self._package_turn(target_run)

    def resume(self, prior_turn: Dict[str, Any], answers: Any) -> Dict[str, Any]:
        state = dict(prior_turn.get("planner_state") or {})
        questions = list(((prior_turn.get("clarification_request") or {}).get("questions")) or [])
        if not questions:
            raise ValueError("prior_turn does not contain pending clarification questions.")

        answer_list = _coerce_answers(answers, questions=questions)
        clarification_history = _normalize_clarification_turns(state.get("clarification_history"))
        for question, answer in zip(questions, answer_list):
            clarification_history.append({"question": str(question), "answer": answer})

        target_run = run_target_resolution_prompt(
            planner=self,
            user_query=str(state.get("original_user_query") or prior_turn.get("user_query") or "").strip(),
            prompt_template=self.target_resolution_prompt_template,
            clarification_turns=clarification_history,
        )
        return self._package_turn(target_run)

    def plan(self, user_query: str, *, include_trace: bool = True) -> Any:
        t0 = time.perf_counter()
        turn = self.start(user_query)
        timing_ms = {"plan_total_ms": int((time.perf_counter() - t0) * 1000)}
        self.last_timing_ms = timing_ms
        if self.log_timing:
            ordered = " ".join(f"{k}={timing_ms[k]}" for k in sorted(timing_ms))
            print(f"[planner_timing_ms] {ordered}")

        plan_payload = dict(turn.get("planner_output") or {})
        if not include_trace:
            return plan_payload

        trace: Dict[str, Any] = {
            "timing_ms": timing_ms,
            "status": turn.get("status"),
            "clarification_request": turn.get("clarification_request"),
        }
        if turn.get("llm_error"):
            trace["error"] = str(turn["llm_error"])
        if turn.get("validation_error"):
            trace["validation_error"] = str(turn["validation_error"])
        return {"plan": plan_payload, "trace": trace}

    def chat(
        self,
        user_query: str,
        *,
        max_rounds: int = 3,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
    ) -> Dict[str, Any]:
        turn = self.start(user_query)
        remaining_rounds = max(1, int(max_rounds))

        while turn.get("status") == "needs_clarification" and remaining_rounds > 0:
            clarification_request = turn.get("clarification_request") or {}
            reason = _normalize_text(clarification_request.get("reason"))
            if reason:
                print_fn(f"Clarification needed: {reason}")

            questions = list(clarification_request.get("questions") or [])
            answers: List[str] = []
            for question in questions:
                answers.append(str(input_fn(f"{question}\n> ") or "").strip())

            turn = self.resume(turn, answers)
            remaining_rounds -= 1

        return turn

    def _package_turn(self, target_run: Dict[str, Any]) -> Dict[str, Any]:
        planner_state = dict(target_run.get("planner_state") or {})
        target_resolution = target_run.get("final_resolution")
        clarification_request = None
        fallback_used = False

        if target_run.get("llm_error") or target_run.get("validation_error"):
            target_resolution = _build_fallback_target_resolution(target_run=target_run)
            status = "completed" if target_resolution is not None else "error"
            fallback_used = True
        elif target_run.get("needs_clarification"):
            status = "needs_clarification"
            clarification_request = {
                "reason": (target_resolution or {}).get("clarification_reason"),
                "questions": list((target_resolution or {}).get("clarification_questions") or []),
            }
        else:
            status = "completed"

        planner_output = _build_planner_output(
            status=status,
            target_run=target_run,
            target_resolution=target_resolution,
            clarification_request=clarification_request,
        )

        return {
            "status": status,
            "user_query": target_run.get("user_query"),
            "planner_state": planner_state,
            "target_resolution": target_resolution,
            "planner_output": planner_output,
            "clarification_request": clarification_request,
            "prompt": target_run.get("prompt"),
            "raw_output": target_run.get("raw_output"),
            "parsed_output": target_run.get("parsed_output"),
            "llm_error": target_run.get("llm_error"),
            "validation_error": target_run.get("validation_error"),
            "fallback_used": fallback_used,
            "full_plan": planner_output if status == "completed" else None,
            "full_plan_trace": None,
            "downstream_skipped_reason": None,
        }


def run_interactive_target_resolution(
    *,
    planner: Any,
    user_query: str,
    prompt_template: str,
    max_rounds: int = 3,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    agent = InteractivePlannerAgent(
        planner=planner,
        target_resolution_prompt_template=prompt_template,
        auto_run_full_planner=False,
    )
    turn = agent.start(user_query)
    rounds: List[Dict[str, Any]] = [turn]
    remaining_rounds = max(1, int(max_rounds))

    while turn.get("status") == "needs_clarification" and remaining_rounds > 0:
        clarification_request = turn.get("clarification_request") or {}
        reason = _normalize_text(clarification_request.get("reason"))
        if reason:
            print_fn(f"Clarification needed: {reason}")

        questions = list(clarification_request.get("questions") or [])
        answers: List[str] = []
        for question in questions:
            answers.append(str(input_fn(f"{question}\n> ") or "").strip())

        turn = agent.resume(turn, answers)
        rounds.append(turn)
        remaining_rounds -= 1

    planner_state = dict(turn.get("planner_state") or {})
    return {
        "user_query": user_query,
        "clarification_turns": list(planner_state.get("clarification_history") or []),
        "rounds": rounds,
        "round_count": len(rounds),
        "stopped_due_to_max_rounds": bool(turn.get("status") == "needs_clarification" and remaining_rounds <= 0),
        "final_resolution": turn.get("target_resolution"),
        "last_run": turn,
        "status": turn.get("status"),
    }


__all__ = [
    "DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE",
    "InteractivePlannerAgent",
    "build_target_resolution_payload",
    "render_target_resolution_prompt",
    "run_target_resolution_prompt",
    "run_interactive_target_resolution",
]
