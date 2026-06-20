from agents.planner.evaluation import PlannerEvalResult


def test_planner_routes_match_expected_routes(planner_eval_results: list[PlannerEvalResult]) -> None:
    failures = []
    for result in planner_eval_results:
        if result.case.expected_route is None:
            continue
        if result.actual_route != result.case.expected_route:
            failures.append(
                f"{result.case.category}/{result.case.id}: "
                f"expected route {result.case.expected_route!r}, got {result.actual_route!r}"
            )

    assert not failures, "\n".join(failures)


def test_planner_ambiguous_queries_clarify_or_stay_kb(
    planner_eval_results: list[PlannerEvalResult],
) -> None:
    failures = []
    for result in planner_eval_results:
        if result.case.expected_behavior != "clarify_or_kb":
            continue
        if result.planner_status != "needs_clarification" and result.actual_route != "kb":
            failures.append(
                f"{result.case.category}/{result.case.id}: "
                "expected clarification or kb route"
            )

    assert not failures, "\n".join(failures)
