from agents.planner.evaluation import PlannerEvalResult, query_has_entity_and_year


def test_structured_fact_and_hybrid_requests_have_required_shape(
    planner_eval_results: list[PlannerEvalResult],
) -> None:
    failures = []
    for result in planner_eval_results:
        if result.case.expected_route not in {"structured_fact", "hybrid"}:
            continue
        expected_count = result.case.expected_structured_fact_count
        requests = result.planner_output.get("structured_fact_requests") or []

        if expected_count is not None and len(requests) < expected_count:
            failures.append(
                f"{result.case.category}/{result.case.id}: expected at least "
                f"{expected_count} structured fact request(s), got {len(requests)}"
            )

        for index, request in enumerate(requests, start=1):
            if not str(request.get("subquestion") or "").strip():
                failures.append(
                    f"{result.case.category}/{result.case.id} request #{index}: "
                    "missing subquestion"
                )
            if not str(request.get("metric_hint") or "").strip():
                failures.append(
                    f"{result.case.category}/{result.case.id} request #{index}: "
                    "missing metric_hint"
                )
            if query_has_entity_and_year(result.case.query):
                if not str(request.get("entity_hint") or "").strip():
                    failures.append(
                        f"{result.case.category}/{result.case.id} request #{index}: "
                        "missing entity_hint"
                    )
                if request.get("fiscal_year") is None:
                    failures.append(
                        f"{result.case.category}/{result.case.id} request #{index}: "
                        "missing fiscal_year"
                    )

    assert not failures, "\n".join(failures)
