import asyncio
import json
from pathlib import Path

import pytest

from scripts.evals.agents.eval_agent_output_repair_v1 import evaluate


DATASET = Path(__file__).resolve().parents[2] / "data/evals/agents/v1/agent_eval_output_repair_v1.jsonl"
CASES = [json.loads(line) for line in DATASET.read_text().splitlines()]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_frozen_output_repair_contract(case):
    row = asyncio.run(evaluate(case))
    assert row["behavior_pass"], row
    assert row["candidate_calls"] <= case.get("max_attempts", 2) + 1
    assert row["inputs_unchanged"]
    assert row["history_preserved"]

