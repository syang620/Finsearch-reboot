from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from agents.planner.interactive_target_resolution import _normalize_resolution_output


SUPPORTED_CLASS = "supported_direct_metric"
AMBIGUOUS_CLASS = "ambiguous"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            cases.append(value)
    return cases


def _request_key(request: dict[str, Any]) -> tuple[str, str]:
    return (
        str(request.get("subquestion") or "").strip(),
        str(request.get("metric_hint") or "").strip(),
    )


def _proposal(case: dict[str, Any]) -> dict[str, Any]:
    open_issues = []
    if case.get("multi_company_query"):
        open_issues.append(
            {
                "code": "MULTI_COMPANY_QUERY",
                "message": "Multiple company entities detected.",
                "severity": "warning",
            }
        )
    return {
        "retrieval_needed": case["proposed_route"] != "structured_fact",
        "route": case["proposed_route"],
        "structured_fact_requests": [
            {
                "subquestion": request["subquestion"],
                "metric_hint": request.get("metric_hint"),
                "entity_hint": "Apple",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
            }
            for request in case["requests"]
        ],
        "task_class": "multi_target_compare" if case.get("multi_company_query") else "single_target_fact",
        "targets": [
            {
                "target_id": 1,
                "target_key": "AAPL_FY2025",
                "company_name": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "form_type": "10-K",
            }
        ],
        "retrieval_plan": (
            {
                "fanout_mode": "single_target",
                "jobs": [
                    {
                        "applies_to_target_ids": [1],
                        "goal": "retrieve filing evidence",
                        "job_type": "metric_extract",
                    }
                ],
            }
            if case["proposed_route"] == "hybrid"
            else None
        ),
        "needs_clarification": False,
        "clarification_reason": None,
        "clarification_questions": [],
        "open_issues": open_issues,
    }


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": (numerator / denominator) if denominator else None,
    }


def evaluate(dataset_path: Path) -> dict[str, Any]:
    cases = _load_cases(dataset_path)
    supported_total = 0
    supported_retained = 0
    unsupported_unknown_total = 0
    unsupported_unknown_removed = 0
    non_supported_total = 0
    non_supported_retained = 0
    retained_total = 0
    retained_supported = 0
    ambiguous_cases = 0
    ambiguous_correct = 0
    route_correct = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        normalized = _normalize_resolution_output(_proposal(case))
        retained = Counter(
            _request_key(request)
            for request in normalized.get("structured_fact_requests") or []
            if isinstance(request, dict)
        )
        expected_indices = {int(index) for index in case["expected_retained_indices"]}
        actual_retained_indices: list[int] = []
        request_results: list[dict[str, Any]] = []

        for index, request in enumerate(case["requests"]):
            key = _request_key(request)
            is_retained = retained[key] > 0
            if is_retained:
                retained[key] -= 1
                actual_retained_indices.append(index)
            expected_class = request["expected_class"]
            is_supported = expected_class == SUPPORTED_CLASS
            is_ambiguous = expected_class == AMBIGUOUS_CLASS

            if is_supported:
                if index in expected_indices:
                    supported_total += 1
                    supported_retained += int(is_retained)
            else:
                non_supported_total += 1
                non_supported_retained += int(is_retained)
                if not is_ambiguous:
                    unsupported_unknown_total += 1
                    unsupported_unknown_removed += int(not is_retained)
            retained_total += int(is_retained)
            retained_supported += int(is_retained and is_supported)
            request_results.append(
                {
                    "index": index,
                    "expected_class": expected_class,
                    "expected_retained": index in expected_indices,
                    "actual_retained": is_retained,
                }
            )

        actual_route = "clarification" if normalized.get("needs_clarification") else normalized.get("route")
        case_route_correct = actual_route == case["expected_route"]
        route_correct += int(case_route_correct)
        has_ambiguity = any(
            request["expected_class"] == AMBIGUOUS_CLASS
            for request in case["requests"]
        )
        if has_ambiguity:
            ambiguous_cases += 1
            ambiguous_correct += int(actual_route == "clarification")

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected_route": case["expected_route"],
                "actual_route": actual_route,
                "route_correct": case_route_correct,
                "expected_retained_indices": sorted(expected_indices),
                "actual_retained_indices": actual_retained_indices,
                "request_results": request_results,
            }
        )

    try:
        evaluated_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        evaluated_sha = None

    return {
        "schema_version": "structured_fact_capability_adversarial.v1",
        "evaluated_sha": evaluated_sha,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "case_count": len(cases),
        "request_count": sum(len(case["requests"]) for case in cases),
        "metrics": {
            "structured_route_precision": _metric(retained_supported, retained_total),
            "supported_query_recall": _metric(supported_retained, supported_total),
            "unsupported_query_rejection_rate": _metric(
                unsupported_unknown_removed, unsupported_unknown_total
            ),
            "ambiguous_query_clarification_accuracy": _metric(
                ambiguous_correct, ambiguous_cases
            ),
            "false_structured_routing_rate": _metric(
                non_supported_retained, non_supported_total
            ),
            "effective_route_accuracy": _metric(route_correct, len(cases)),
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.dataset)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
