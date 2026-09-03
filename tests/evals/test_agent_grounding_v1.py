from __future__ import annotations

import asyncio
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.analyst import AnalystRunResult
from agents.analyst.grounding import validate_grounding
from agents.orchestrator.agent_orchestrator import _format_run_output
from evals import grounding_oracle
from evals.agent_eval_route_aware_v1 import evaluate_run_output
from scripts.evals.agents.eval_agent_grounding_v1 import evaluate, packet_for
from tests.evals.test_agent_eval_route_aware_v1 import _example, _run_output


DATASET = Path(__file__).resolve().parents[2] / "data/evals/agents/v1/agent_eval_grounding_v1.jsonl"
CASES = [json.loads(line) for line in DATASET.read_text().splitlines()]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_frozen_grounding_scenario(case):
    row = asyncio.run(evaluate(case))
    assert row["behavior_pass"], row
    assert row["candidate_calls"] <= case.get("max_attempts", 1) + 1
    if case.get("calculator"):
        # Citation retries must reuse successful calculator results.
        assert row["model_calls"] == row["candidate_calls"] + 1


def test_independent_oracle_does_not_import_runtime_validator():
    source = inspect.getsource(grounding_oracle)
    assert "agents.analyst" not in source
    assert "validate_grounding" not in source


@pytest.mark.parametrize("mutation", ["missing_claims", "unknown_id", "wrong_metric", "provenance", "unbound_answer", "visibility"])
def test_route_gate_rejects_successful_but_inconsistent_grounding(mutation):
    output = _run_output("structured_fact")
    result = output["analyst"]
    if mutation == "missing_claims":
        result["claims"] = []
    elif mutation == "unknown_id":
        result["claims"][0]["context_ids"] = ["ghost"]
    elif mutation == "wrong_metric":
        result["claims"][0]["metric_id"] = "net_income"
    elif mutation == "provenance":
        result["citations"][0]["source"]["ticker"] = "MSFT"
    elif mutation == "unbound_answer":
        result["answer"] += " An unbound assertion."
    else:
        result["trace"]["context_item_limit"] = 1
        result["trace"]["analyst_visible_context_ids"] = ["ghost"]
    row, errors, _ = evaluate_run_output(_example("structured_fact"), output)
    assert errors == []
    assert row.grounding["consistent"] is False
    assert "GROUNDING_INCONSISTENT" in row.deterministic.critical_failures
    assert row.derived_effective_status == "failed"
    assert row.derived_failure_stage == "analyst"


def test_grounding_error_is_analyst_failure_without_mutating_lanes():
    row = asyncio.run(evaluate(CASES[1]))
    output = _run_output("structured_fact")
    packet = packet_for(CASES[1])
    state = {
        "plan_obj": output["planner"], "planner_dump": output["planner"],
        "packet": packet, "structured_fact_results": output["structured_fact_results"],
        "analyst_result": AnalystRunResult.model_validate(row["analyst"]),
    }
    formatted = _format_run_output(run_id="grounding-failure", state_snapshot=SimpleNamespace(values=state, interrupts=()))
    assert formatted["status"] == "failed"
    assert formatted["failure_stage"] == "analyst"


def test_empty_kb_context_cannot_support_a_claim():
    case = CASES[4]
    packet = packet_for(case)
    packet.context_items[2].payload = {}
    candidate = case["outputs"][0]
    decision = validate_grounding(packet, candidate, limit=4)
    assert not decision.valid
    assert "GROUNDING_CONTEXT_UNUSABLE" in decision.issue_codes
    assert not grounding_oracle.inspect_claims(packet.model_dump(mode="json"), candidate, 4)["valid"]


def test_semantic_negative_passes_structural_policy_only():
    row = asyncio.run(evaluate(CASES[14]))
    assert row["behavior_pass"]
    assert row["semantic_annotation"] == "unsupported"
    # Correct evidence kind/ID is not proof of entailment.


def test_sanitization_is_claim_local():
    case = copy.deepcopy(CASES[9])
    case["outputs"][0]["claims"].append({
        "claim_id": "c2", "claim_type": "narrative", "text": "Costs fell.", "context_ids": ["ghost"],
    })
    case["expected_ok"] = False
    case.pop("expected_sanitized")
    assert asyncio.run(evaluate(case))["behavior_pass"]


@pytest.mark.parametrize("status", ["ok", "insufficient_data"])
def test_calculation_claim_cannot_bypass_calculator_on_extract_task(status):
    case = copy.deepcopy(CASES[8])
    case.pop("calculator")
    case["outputs"][0]["status"] = status
    row = asyncio.run(evaluate(case))
    assert not row["analyst"]["ok"]
    assert row["analyst"]["error"] in {"COMPUTE_RESULT_MISSING", "CALCULATION_RESULT_MISMATCH"}


def test_shadow_checks_calculation_claims_even_without_planner_compute_flag():
    output = _run_output("kb")
    output["analyst"]["claims"][0]["claim_type"] = "calculation"
    row, errors, _ = evaluate_run_output(_example("kb"), output)
    assert errors == []
    assert "GROUNDING_CALCULATOR_MISSING" in row.grounding["errors"]
    assert row.derived_failure_stage == "analyst"
