import asyncio
import json
import re
from typing import Any, Dict, List

import pytest

from agents.planner import InteractivePlannerAgent
from agents.planner.evaluation import PlannerEvalCase, load_planner_eval_cases, run_planner_cases


_COMPANY_TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Google": "GOOGL",
}


def _extract_payload(prompt: str) -> Dict[str, Any]:
    marker = "Deterministic extraction results:\n"
    start = prompt.index(marker) + len(marker)
    end_marker = "\n\nReturn exactly one JSON object"
    end = prompt.index(end_marker, start)
    return json.loads(prompt[start:end].strip())


def _year_from_query(query: str) -> int | None:
    match = re.search(r"\b(?:FY\s*)?(19\d{2}|20\d{2})\b", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _companies_from_query(query: str) -> List[str]:
    out: List[str] = []
    for company in _COMPANY_TICKERS:
        if re.search(rf"\b{re.escape(company)}\b", query, re.IGNORECASE):
            out.append(company)
    return out


def _metric_hint(query: str) -> str:
    q = query.lower()
    if "operating cash flow" in q:
        return "operating cash flow"
    if "cash" in q:
        return "cash and cash equivalents"
    if "total debt" in q or "debt" in q or "borrowings" in q:
        return "total debt"
    if "stockholders equity" in q:
        return "stockholders equity"
    if "total liabilities" in q:
        return "total liabilities"
    if "total assets" in q:
        return "total assets"
    if "gross profit" in q:
        return "gross profit"
    if "operating income" in q:
        return "operating income"
    if "net income" in q:
        return "net income"
    if "sales" in q or "revenue" in q:
        return "revenue"
    return "filing facts"


def _is_unsupported_or_derived(query: str) -> bool:
    q = query.lower()
    return any(
        token in q
        for token in [
            "gross margin",
            "eps",
            "earnings per share",
            "debt-to-equity",
            "debt to equity",
            "return on equity",
            "return on assets",
            "free cash flow yield",
            "ev/ebitda",
            "ebitda margin",
            "operating margin",
            "revenue growth",
            "year-over-year",
            "net debt",
            "free cash flow",
            "leverage ratio",
            "percentage of assets",
            "change in",
        ]
    )


def _is_comparison(query: str) -> bool:
    q = query.lower()
    companies = _companies_from_query(query)
    return len(companies) > 1 and any(token in q for token in ["compare", "higher", "between", " or "])


def _is_hybrid(query: str) -> bool:
    q = query.lower()
    return " and " in q and any(
        token in q
        for token in ["drove", "explain", "factors", "contributed", "used", "strategy", "discuss"]
    )


def _is_narrative(query: str) -> bool:
    q = query.lower()
    return any(
        token in q
        for token in [
            "what drove",
            "why did",
            "describe",
            "risks",
            "challenges",
            "summarize",
            "management say",
            "risk factors",
            "key takeaways",
        ]
    )


def _targets_for_query(payload: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    deterministic_targets = [
        dict(target)
        for target in payload.get("deterministic_targets") or []
        if isinstance(target, dict)
    ]
    if deterministic_targets:
        return deterministic_targets

    year = _year_from_query(query) or payload.get("deterministic_fiscal_year")
    if year is None:
        return []

    targets: List[Dict[str, Any]] = []
    for index, company in enumerate(_companies_from_query(query), start=1):
        ticker = _COMPANY_TICKERS[company]
        targets.append(
            {
                "target_id": index,
                "target_key": f"{ticker}_FY{year}",
                "company_name": company,
                "ticker": ticker,
                "fiscal_year": year,
                "form_type": payload.get("deterministic_form_type"),
            }
        )
    return targets


def _retrieval_plan(route: str, targets: List[Dict[str, Any]], query: str) -> Dict[str, Any] | None:
    if route == "structured_fact" or not targets:
        return None
    target_ids = [int(target["target_id"]) for target in targets]
    job_type = "narrative_extract" if route == "kb" and _is_narrative(query) else "metric_extract"
    return {
        "fanout_mode": "single_target" if len(target_ids) == 1 else "per_target",
        "jobs": [
            {
                "applies_to_target_ids": target_ids,
                "goal": _metric_hint(query),
                "job_type": job_type,
            }
        ],
    }


class FakePlannerRoutingLLM:
    async def ainvoke(self, prompt: str) -> str:
        payload = _extract_payload(prompt)
        query = str(payload.get("effective_user_query") or payload.get("user_query") or "")
        targets = _targets_for_query(payload, query)
        unresolved_blockers = list(payload.get("unresolved_blockers") or [])
        if unresolved_blockers and not targets:
            return json.dumps(
                {
                    "retrieval_needed": False,
                    "route": "kb",
                    "structured_fact_requests": [],
                    "task_class": "other",
                    "targets": [],
                    "retrieval_plan": None,
                    "needs_clarification": True,
                    "clarification_reason": "Missing required target metadata.",
                    "clarification_questions": ["Which ticker and fiscal year should be used?"],
                    "open_issues": [],
                }
            )

        route = "structured_fact"
        if _is_unsupported_or_derived(query) or _is_comparison(query):
            route = "kb"
        elif _is_hybrid(query):
            route = "hybrid"
        elif _is_narrative(query):
            route = "kb"

        structured_fact_requests: List[Dict[str, Any]] = []
        if route in {"structured_fact", "hybrid"}:
            first_target = targets[0] if targets else {}
            structured_fact_requests.append(
                {
                    "subquestion": query,
                    "metric_hint": _metric_hint(query),
                    "entity_hint": first_target.get("company_name"),
                    "fiscal_year": first_target.get("fiscal_year"),
                    "fiscal_period": "FY",
                }
            )

        return json.dumps(
            {
                "retrieval_needed": route != "structured_fact",
                "route": route,
                "structured_fact_requests": structured_fact_requests,
                "task_class": "multi_target_compare" if _is_comparison(query) else "single_target_fact",
                "targets": targets,
                "retrieval_plan": _retrieval_plan(route, targets, query),
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            }
        )


@pytest.fixture(scope="session")
def planner_eval_cases() -> List[PlannerEvalCase]:
    return load_planner_eval_cases()


@pytest.fixture(scope="session")
def planner_eval_results(planner_eval_cases: List[PlannerEvalCase]):
    planner = InteractivePlannerAgent(llm=FakePlannerRoutingLLM(), log_timing=False)
    return asyncio.run(run_planner_cases(planner, planner_eval_cases))
