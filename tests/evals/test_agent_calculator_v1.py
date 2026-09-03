import asyncio
import copy
import json
from pathlib import Path

import pytest

from scripts.evals.agents.eval_agent_calculator_v1 import evaluate


DATASET = Path(__file__).resolve().parents[2] / "data/evals/agents/v1/agent_eval_calculator_v1.jsonl"
CASES = [json.loads(line) for line in DATASET.read_text().splitlines()]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_frozen_calculator_regression(case):
    row = asyncio.run(evaluate(case))
    assert row["behavior_pass"], row
    assert row["candidate_calls"] <= case["max_attempts"] + 1
    assert row["calculator_calls"] == len(case["history"])


@pytest.mark.parametrize("bad_input", ["not-a-number", "NaN", "Infinity"])
def test_duplicate_invalid_inputs_are_not_collapsed(bad_input):
    case = copy.deepcopy(CASES[0])
    for item in case["history"]:
        item["args"]["variables"]["services_2023"] = bad_input
        item["result"]["variables"]["services_2023"] = bad_input
    case["outputs"][0]["calculation"]["variables"]["services_2023"] = bad_input
    row = asyncio.run(evaluate(case))
    assert not row["analyst"]["ok"]
    assert row["analyst"]["error"] == "CALCULATION_RESULT_AMBIGUOUS"


def test_ambiguous_repair_preserves_claims_and_history():
    row = asyncio.run(evaluate(CASES[3]))
    assert row["behavior_pass"]
    assert row["analyst"]["claims"] == [
        {"metric_id": None, **claim} for claim in CASES[3]["outputs"][1]["claims"]
    ]
    assert len(row["analyst"]["trace"]["grounding_attempts"]) == 2
    assert row["calculator_calls"] == 3
