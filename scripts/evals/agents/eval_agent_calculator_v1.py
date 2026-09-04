"""Frozen calculator-provenance replay; no model, retrieval, or runtime selector oracle."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agents.analyst import AnalystAgent
from agents.contracts import AnalystPacket
from evals.grounding_oracle import inspect_final


def packet_for(case):
    return AnalystPacket.model_validate({
        "plan_id": "calculator-regression-v1",
        "user_query": "Using Apple's FY2024 filing, what was the percentage increase in Services net sales from 2023 to 2024?",
        "intent": "filing_calc",
        "metadata": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
        "analysis_task": {"task_type": case.get("task_type", "compute"), "metric": "Services net sales"},
        "context_quality": "high",
        "context_items": [{
            "context_id": "ctx_1", "kind": "text",
            "source": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
            "payload": {"content": "Services net sales: 2023 = 85,200; 2024 = 96,169 (millions). Synthetic collision controls: revenue = expense = 100; x varies by fixture."},
        }],
    })


class ReplayModel:
    def __init__(self, case):
        self.case = case
        self.calls = 0
        self.candidates = 0
        self.feedback = []
        self.emitted = []

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1 and self.case["history"]:
            calls = [{"name": "financial_evaluator", "id": f"calc-{i}", "args": item["args"]}
                     for i, item in enumerate(self.case["history"])]
        else:
            self.feedback = [message.content for message in messages if isinstance(message, HumanMessage)]
            outputs = self.case["outputs"]
            candidate = outputs[min(self.candidates, len(outputs) - 1)]
            self.candidates += 1
            self.emitted.append(candidate)
            calls = [{"name": "FinalAnswer", "id": f"final-{self.candidates}", "args": candidate}]
        return AIMessage(content="", tool_calls=calls)


class ReplayCalculator:
    def __init__(self, history):
        self.history = history
        self.calls = 0

    async def ainvoke(self, args):
        item = self.history[self.calls]
        assert args == item["args"]
        self.calls += 1
        return item["result"]


async def evaluate(case):
    packet = packet_for(case)
    model, calculator = ReplayModel(case), ReplayCalculator(case["history"])
    agent = AnalystAgent(max_attempts=case["max_attempts"])
    agent._bound_model_override = model
    agent._tool_map = {"financial_evaluator": calculator}
    result = await agent.arun(packet)
    await agent.aclose()
    output = result.model_dump(mode="json")
    grounding = inspect_final(packet.model_dump(mode="json"), output, agent.max_context_items)
    selection_feedback = any("CALCULATION_RESULT_AMBIGUOUS" in str(item) for item in model.feedback)
    actual_computation = output.get("computation")
    computation_ok = (not case["expected_ok"] or case["expected_status"] != "ok"
                      or actual_computation == case["expected_computation"])
    history_preserved = (calculator.calls == len(case["history"])
                         and sum(call["name"] == "financial_evaluator" for call in output["trace"]["tool_calls"]) == len(case["history"]))
    behavior = (output["ok"] == case["expected_ok"] and output["status"] == case["expected_status"]
                and (case["expected_ok"] or output["error"] == case["expected_error"])
                and computation_ok and history_preserved
                and model.candidates == case["expected_candidate_calls"]
                and (not case.get("requires_selection_feedback") or selection_feedback)
                and (not output["ok"] or grounding["valid"]))
    return {
        "id": case["id"], "behavior_pass": behavior,
        "expected_ok": case["expected_ok"], "expected_error": case.get("expected_error"),
        "candidate_calls": model.candidates, "model_calls": model.calls,
        "calculator_calls": calculator.calls, "history_preserved": history_preserved,
        "selection_feedback": selection_feedback, "feedback": model.feedback,
        "emitted_candidates": model.emitted, "analyst": output,
        "computation_matches_fixture": computation_ok, "grounding_oracle": grounding,
    }


async def main(args):
    dataset = Path(args.eval_path)
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    # Refuse stale/dirty tracked source and existing evidence before running.
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    out = Path(args.out_dir)
    assert not out.exists()
    rows = [await evaluate(case) for case in cases]
    summary = {
        "cases": len(rows), "behavior_passes": sum(row["behavior_pass"] for row in rows),
        "gate_pass": all(row["behavior_pass"] for row in rows),
        "history_preserved": sum(row["history_preserved"] for row in rows),
        "unexpected_successes": sum(row["analyst"]["ok"] and not row["expected_ok"] for row in rows),
        "successful_outputs": sum(row["analyst"]["ok"] for row in rows),
        "successful_grounding_valid": sum(row["analyst"]["ok"] and row["grounding_oracle"]["valid"] for row in rows),
    }
    import pytest
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "evaluated_commit": sha, "tracked_worktree_clean": True,
        "dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grounding_oracle_sha256": hashlib.sha256(Path("src/evals/grounding_oracle.py").read_bytes()).hexdigest(),
        "runtime_source_sha256": hashlib.sha256(Path("src/agents/analyst/agent.py").read_bytes()).hexdigest(),
        "python": platform.python_version(), "pytest": pytest.__version__,
        "measurement": "Frozen synthetic calculator-message and FinalAnswer replay through the real analyst graph; not an exact live transcript or a semantic benchmark.",
        "command": f"PYTHONPATH=.:src conda run -n finsearch-arm python scripts/evals/agents/eval_agent_calculator_v1.py --eval-path {dataset} --out-dir {out}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "per_case.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="data/evals/agents/v1/agent_eval_calculator_v1.jsonl")
    parser.add_argument("--out-dir", required=True)
    asyncio.run(main(parser.parse_args()))
