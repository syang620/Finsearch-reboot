import asyncio
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
