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

    assert len(cases) == 11
    assert set(result["summary"].values()) == {11, 1.0}
    assert all(row["checks"]["overall_correct"] for row in result["rows"])


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
