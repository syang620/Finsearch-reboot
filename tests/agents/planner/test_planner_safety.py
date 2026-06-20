from agents.planner.evaluation import PlannerEvalResult


def test_unsupported_metrics_stay_kb_without_structured_fact_requests(
    planner_eval_results: list[PlannerEvalResult],
) -> None:
    failures = []
    for result in planner_eval_results:
        if result.case.category != "unsupported_metrics":
            continue
        if result.actual_route != "kb":
            failures.append(
                f"{result.case.id}: expected kb, got {result.actual_route!r}"
            )
        if result.actual_structured_fact_count:
            failures.append(
                f"{result.case.id}: expected no structured fact requests, "
                f"got {result.actual_structured_fact_count}"
            )

    assert not failures, "\n".join(failures)


def test_multi_company_comparisons_stay_kb_without_structured_fact_requests(
    planner_eval_results: list[PlannerEvalResult],
) -> None:
    failures = []
    for result in planner_eval_results:
        if result.case.category != "multi_company_comparison":
            continue
        if result.actual_route != "kb":
            failures.append(
                f"{result.case.id}: expected kb, got {result.actual_route!r}"
            )
        if result.actual_structured_fact_count:
            failures.append(
                f"{result.case.id}: expected no structured fact requests, "
                f"got {result.actual_structured_fact_count}"
            )

    assert not failures, "\n".join(failures)
