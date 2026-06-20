from agents.planner.interactive_target_resolution import _build_analysis_task


def test_metric_compare_marks_requires_calculation():
    analysis_task = _build_analysis_task(
        task_class="multi_target_compare",
        metric_guess="net debt per share",
        retrieval_plan={
            "jobs": [
                {"job_type": "metric_extract", "goal": "net debt"},
            ]
        },
        task_type_hint="compute",
    )

    assert analysis_task["task_type"] == "compare"
    assert analysis_task["requires_calculation"] is False
    assert analysis_task["expected_artifacts"] == ["table", "row", "text"]


def test_metric_extract_does_not_require_calculation_by_default():
    analysis_task = _build_analysis_task(
        task_class="single_target_fact",
        metric_guess="revenue",
        retrieval_plan={
            "jobs": [
                {"job_type": "metric_extract", "goal": "revenue"},
            ]
        },
        task_type_hint="extract",
    )

    assert analysis_task["task_type"] == "extract"
    assert analysis_task["requires_calculation"] is False
    assert analysis_task["expected_artifacts"] == ["table", "row", "text"]


def test_narrative_extract_prefers_text_artifacts():
    analysis_task = _build_analysis_task(
        task_class="single_target_fact",
        metric_guess="risk factors",
        retrieval_plan={
            "jobs": [
                {"job_type": "narrative_extract", "goal": "risk factors"},
            ]
        },
        task_type_hint="extract",
    )

    assert analysis_task["task_type"] == "extract"
    assert analysis_task["requires_calculation"] is False
    assert analysis_task["expected_artifacts"] == ["text"]
