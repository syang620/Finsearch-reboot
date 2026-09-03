from __future__ import annotations

from pathlib import Path

from evals.agent_eval_degradation_v1 import (
    evaluate_degradation_matrix,
    load_degradation_cases,
)


DATASET = Path("data/evals/agents/v1/agent_eval_degradation_v1.jsonl")


def test_frozen_degradation_matrix_is_complete_and_consistent() -> None:
    cases = load_degradation_cases(DATASET)
    result = evaluate_degradation_matrix(cases)

    assert len(cases) == 14
    assert set(result["summary"].values()) == {14, 1.0}
    assert all(row["checks"]["overall_correct"] for row in result["rows"])


def test_degradation_matrix_covers_admission_loss() -> None:
    rows = {
        row["id"]: row
        for row in evaluate_degradation_matrix(
            load_degradation_cases(DATASET)
        )["rows"]
    }

    kb = rows["PR5_KB_ADMISSION_PARTIAL"]["output"]
    kb_packet = kb["evaluation_trace"]["analyst_packet"]
    assert kb["retrieval"]["targets"][0]["tables_retrieved"] == 2
    assert len(kb_packet["context_items"]) == 1
    assert {issue["code"] for issue in kb_packet["open_issues"]} == {
        "TABLE_HYDRATION_FAILED"
    }
    assert kb["lanes"]["kb"]["status"] == "partial"
    assert kb["status"] == "degraded"
    assert kb["analyst"] is not None

    structured = rows["PR5_STRUCTURED_MIXED_ADMISSION"]["output"]
    structured_packet = structured["evaluation_trace"]["analyst_packet"]
    assert len(structured["planner"]["structured_fact_requests"]) == 2
    assert [
        result["tool_result"]["status"]
        for result in structured["structured_fact_results"]
    ] == ["ok", "ok"]
    assert len(structured_packet["context_items"]) == 1
    assert {issue["code"] for issue in structured_packet["open_issues"]} == {
        "STRUCTURED_FACT_INVALID_EVIDENCE"
    }
    assert structured["lanes"]["structured_fact"]["status"] == "partial"
    assert structured["status"] == "degraded"
    assert structured["analyst"] is not None


def test_degradation_matrix_preserves_analyst_failure_diagnostics() -> None:
    rows = {
        row["id"]: row
        for row in evaluate_degradation_matrix(
            load_degradation_cases(DATASET)
        )["rows"]
    }
    row = rows["PR5_DEGRADED_THEN_ANALYST_FAILED"]

    assert row["output"]["status"] == "failed"
    assert row["output"]["failure_stage"] == "analyst"
    assert row["output"]["degradation"]["active"] is True
    assert row["output"]["degradation"]["affected_lanes"] == [
        "structured_fact"
    ]


def test_degradation_matrix_fails_closed_for_mixed_unusable_structured_results() -> None:
    rows = {
        row["id"]: row
        for row in evaluate_degradation_matrix(
            load_degradation_cases(DATASET)
        )["rows"]
    }
    row = rows["PR5_STRUCTURED_MIXED_UNUSABLE"]
    output = row["output"]
    packet = output["evaluation_trace"]["analyst_packet"]

    assert len(output["planner"]["structured_fact_requests"]) == 2
    assert [
        result["tool_result"]["status"]
        for result in output["structured_fact_results"]
    ] == ["partial", "error"]
    assert packet["context_items"] == []
    assert output["lanes"]["structured_fact"]["status"] == "failed"
    assert row["shadow"]["lanes"]["structured_fact"]["usable"] is False
    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"
    assert output["analyst"] is None
