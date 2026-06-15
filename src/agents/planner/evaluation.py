from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_ENTITY_RE = re.compile(
    r"\b(Apple|Microsoft|Alphabet|Google|Amazon|Meta|Nvidia|Tesla|AAPL|MSFT|GOOGL|GOOG)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:FY\s*)?(19\d{2}|20\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class PlannerEvalCase:
    id: str
    query: str
    category: str
    expected_route: Optional[str] = None
    expected_behavior: Optional[str] = None
    expected_structured_fact_count: Optional[int] = None
    priority: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerEvalResult:
    case: PlannerEvalCase
    ok: bool
    failures: List[str]
    actual_route: Optional[str]
    actual_structured_fact_count: int
    planner_status: Optional[str]
    planner_output: Dict[str, Any]
    planner_turn: Dict[str, Any] = field(default_factory=dict)


def default_cases_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    preferred = repo_root / "data" / "evals" / "agents" / "planner_eval_cases.json"
    if preferred.exists():
        return preferred
    return repo_root / "data" / "evals" / "agents" / "planner_routing_core.v1.json"


def normalize_route(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text or None


def _normalize_priority(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    return text or None


def load_planner_eval_cases(
    path: Optional[str | Path] = None,
    *,
    categories: Optional[Sequence[str]] = None,
    priorities: Optional[Sequence[str]] = None,
) -> List[PlannerEvalCase]:
    cases_path = Path(path) if path is not None else default_cases_path()
    with cases_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    allowed_categories = {str(item) for item in categories or []}
    allowed_priorities = {_normalize_priority(item) for item in priorities or []}
    allowed_priorities.discard(None)

    cases: List[PlannerEvalCase] = []
    for category_payload in payload.get("categories") or []:
        if not isinstance(category_payload, dict):
            continue
        category = str(category_payload.get("category") or "").strip()
        if allowed_categories and category not in allowed_categories:
            continue

        for raw_case in category_payload.get("cases") or []:
            if not isinstance(raw_case, dict):
                continue
            priority = _normalize_priority(raw_case.get("priority"))
            if allowed_priorities and priority not in allowed_priorities:
                continue
            case_id = str(raw_case.get("id") or "").strip()
            query = str(raw_case.get("query") or "").strip()
            if not case_id or not query:
                continue
            cases.append(
                PlannerEvalCase(
                    id=case_id,
                    query=query,
                    category=category,
                    expected_route=normalize_route(raw_case.get("expected_route")),
                    expected_behavior=str(raw_case.get("expected_behavior") or "").strip() or None,
                    expected_structured_fact_count=(
                        int(raw_case["expected_structured_fact_count"])
                        if raw_case.get("expected_structured_fact_count") is not None
                        else None
                    ),
                    priority=priority,
                    raw=dict(raw_case),
                )
            )
    return cases


def extract_planner_output(turn: Any) -> Dict[str, Any]:
    if hasattr(turn, "model_dump"):
        turn = turn.model_dump(mode="json")
    if not isinstance(turn, dict):
        return {}
    planner_output = turn.get("planner_output")
    if isinstance(planner_output, dict):
        return dict(planner_output)
    plan = turn.get("plan")
    if isinstance(plan, dict):
        return dict(plan)
    return dict(turn)


def query_has_entity_and_year(query: str) -> bool:
    return bool(_ENTITY_RE.search(query or "") and _YEAR_RE.search(query or ""))


def _structured_fact_requests(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    requests = plan.get("structured_fact_requests") or []
    return [dict(item) for item in requests if isinstance(item, dict)]


def evaluate_planner_output(
    case: PlannerEvalCase,
    planner_output: Dict[str, Any],
    *,
    planner_turn: Optional[Dict[str, Any]] = None,
) -> PlannerEvalResult:
    route = normalize_route(planner_output.get("route"))
    status = str(planner_output.get("status") or "").strip().lower() or None
    structured_fact_requests = _structured_fact_requests(planner_output)
    failures: List[str] = []

    if case.expected_route and route != case.expected_route:
        failures.append(f"route expected {case.expected_route!r}, got {route!r}")

    if case.expected_behavior == "clarify_or_kb":
        if status != "needs_clarification" and route != "kb":
            failures.append(
                "expected ambiguous query to either request clarification or stay on kb"
            )

    if case.expected_route in {"structured_fact", "hybrid"}:
        expected_count = case.expected_structured_fact_count
        if expected_count is not None and len(structured_fact_requests) < expected_count:
            failures.append(
                "structured_fact_requests expected at least "
                f"{expected_count}, got {len(structured_fact_requests)}"
            )
        for index, request in enumerate(structured_fact_requests, start=1):
            if not str(request.get("subquestion") or "").strip():
                failures.append(f"structured_fact_request #{index} missing subquestion")
            if not str(request.get("metric_hint") or "").strip():
                failures.append(f"structured_fact_request #{index} missing metric_hint")
            if query_has_entity_and_year(case.query):
                if not str(request.get("entity_hint") or "").strip():
                    failures.append(f"structured_fact_request #{index} missing entity_hint")
                if request.get("fiscal_year") is None:
                    failures.append(f"structured_fact_request #{index} missing fiscal_year")

    if case.category in {"unsupported_metrics", "multi_company_comparison"}:
        if route != "kb":
            failures.append(f"{case.category} must route to kb, got {route!r}")
        if structured_fact_requests:
            failures.append(
                f"{case.category} must not emit structured_fact_requests, "
                f"got {len(structured_fact_requests)}"
            )

    return PlannerEvalResult(
        case=case,
        ok=not failures,
        failures=failures,
        actual_route=route,
        actual_structured_fact_count=len(structured_fact_requests),
        planner_status=status,
        planner_output=dict(planner_output),
        planner_turn=dict(planner_turn or {}),
    )


async def run_planner_case(planner: Any, case: PlannerEvalCase) -> PlannerEvalResult:
    if hasattr(planner, "aplan_turn"):
        turn = await planner.aplan_turn(user_query=case.query)
    elif hasattr(planner, "start"):
        turn = await asyncio.to_thread(planner.start, case.query)
    else:
        raise TypeError("planner must expose aplan_turn() or start().")
    turn_payload = turn.model_dump(mode="json") if hasattr(turn, "model_dump") else turn
    return evaluate_planner_output(
        case,
        extract_planner_output(turn),
        planner_turn=dict(turn_payload) if isinstance(turn_payload, dict) else {},
    )


async def run_planner_cases(
    planner: Any,
    cases: Sequence[PlannerEvalCase],
) -> List[PlannerEvalResult]:
    results: List[PlannerEvalResult] = []
    for case in cases:
        results.append(await run_planner_case(planner, case))
    return results


def summarize_results(results: Iterable[PlannerEvalResult]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "accuracy": 0.0,
        "categories": {},
    }
    for result in results:
        summary["total"] += 1
        if result.ok:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
        category_summary = summary["categories"].setdefault(
            result.case.category,
            {"total": 0, "passed": 0, "failed": 0, "accuracy": 0.0},
        )
        category_summary["total"] += 1
        if result.ok:
            category_summary["passed"] += 1
        else:
            category_summary["failed"] += 1

    if summary["total"]:
        summary["accuracy"] = summary["passed"] / summary["total"]
    for category_summary in summary["categories"].values():
        if category_summary["total"]:
            category_summary["accuracy"] = (
                category_summary["passed"] / category_summary["total"]
            )
    return summary


def serialize_results(results: Sequence[PlannerEvalResult]) -> Dict[str, Any]:
    summary = summarize_results(results)
    return {
        "summary": summary,
        "results": [
            {
                "id": result.case.id,
                "category": result.case.category,
                "priority": result.case.priority,
                "query": result.case.query,
                "expected_route": result.case.expected_route,
                "expected_behavior": result.case.expected_behavior,
                "expected_structured_fact_count": result.case.expected_structured_fact_count,
                "ok": result.ok,
                "failures": list(result.failures),
                "actual_route": result.actual_route,
                "actual_structured_fact_count": result.actual_structured_fact_count,
                "planner_status": result.planner_status,
                "planner_output": dict(result.planner_output),
                "planner_turn": dict(result.planner_turn),
            }
            for result in results
        ],
    }
