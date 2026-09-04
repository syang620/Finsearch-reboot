"""Frozen analyst contract-repair replay; feedback checks are fixture expectations.

Candidate responses are injected unconditionally, not generated from the feedback.
Report terminal outcomes separately from diagnostic-feedback contract coverage.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
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


SOURCES = Path("data/evals/agents/v1/analyst_output_repair_sources_v1.json")


class CandidateModel:
    def __init__(self, case, source):
        self.case, self.source = case, source
        self.calls = self.candidates = 0
        self.feedback, self.emitted = [], []

    async def ainvoke(self, messages):
        self.calls += 1
        self.feedback = [str(message.content) for message in messages if isinstance(message, HumanMessage)][1:]
        if self.source["calculator"] and self.calls == 1:
            call = {"name": "financial_evaluator", "id": "calc-1", "args": self.source["calculator"]["args"]}
        else:
            outputs = self.case["outputs"]
            payload = copy.deepcopy(outputs[min(self.candidates, len(outputs) - 1)])
            self.candidates += 1
            self.emitted.append(copy.deepcopy(payload))
            call = {"name": "FinalAnswer", "id": f"final-{self.candidates}", "args": payload}
        return AIMessage(content="", tool_calls=[call])


class Calculator:
    def __init__(self, source):
        self.source, self.calls = source, 0

    async def ainvoke(self, args):
        assert args == self.source["args"]
        self.calls += 1
        return copy.deepcopy(self.source["result"])


async def evaluate(case, sources=None):
    sources = sources if sources is not None else json.loads(SOURCES.read_text())
    source = sources[case["source"]]
    original = copy.deepcopy((case, source))
    packet = AnalystPacket.model_validate(source["packet"])
    model, calculator = CandidateModel(case, source), Calculator(source["calculator"])
    agent = AnalystAgent(max_attempts=case.get("max_attempts", 2))
    agent._bound_model_override = model
    if source["calculator"]:
        agent._tool_map = {"financial_evaluator": calculator}
    try:
        result = await agent.arun(packet)
    finally:
        await agent.aclose()
    output = result.model_dump(mode="json")
    oracle = inspect_final(packet.model_dump(mode="json"), output, agent.max_context_items)
    terminal = (output["ok"] == case["expected_ok"] and output["status"] == case["expected_status"]
                and (case["expected_ok"] or output["error"] == case["expected_error"]))
    feedback_checks = [
        check["retry"] < len(model.feedback)
        and all(token in model.feedback[check["retry"]] for token in check["contains"])
        for check in case.get("feedback_checks", [])
    ]
    issue_codes = [item["code"] for item in output["open_issues"]]
    issue_ok = not case.get("expected_issue") or case["expected_issue"] in issue_codes
    # Verify only the model-emitted factual claims survive, with no repair-time
    # insertion or rewriting. PR6's existing normalization/assembly is unchanged.
    factual_keys = ("claim_id", "claim_type", "text", "metric_id")
    prose_preserved = not output["ok"] or [
        {key: claim.get(key) for key in factual_keys} for claim in output["claims"]
    ] == [{key: claim.get(key) for key in factual_keys} for claim in model.emitted[-1].get("claims", [])]
    computation_preserved = not output["ok"] or not source["calculator"] or output["computation"] == source["calculator"]["result"]
    raw_calls = output["trace"]["tool_calls"]
    history_preserved = (
        calculator.calls == int(bool(source["calculator"]))
        and sum(item["name"] == "financial_evaluator" for item in raw_calls) == calculator.calls
        and [item["args"] for item in raw_calls if item["name"] == "FinalAnswer"] == model.emitted
    )
    inputs_unchanged = (case, source) == original
    behavior = (terminal and all(feedback_checks) and issue_ok and prose_preserved
                and computation_preserved and history_preserved and inputs_unchanged
                and model.candidates == case["expected_candidates"]
                and (not output["ok"] or oracle["valid"]))
    return {
        "id": case["id"], "source": case["source"], "behavior_pass": behavior,
        "terminal_pass": terminal, "feedback_contract_pass": all(feedback_checks),
        "feedback_checks": feedback_checks, "feedback": model.feedback,
        "candidate_calls": model.candidates, "model_calls": model.calls,
        "calculator_calls": calculator.calls, "emitted_candidates": model.emitted,
        "prose_preserved": prose_preserved, "computation_preserved": computation_preserved,
        "history_preserved": history_preserved, "inputs_unchanged": inputs_unchanged,
        "expected_ok": case["expected_ok"], "analyst": output, "oracle": oracle,
    }


async def main(args):
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    out, dataset = Path(args.out_dir), Path(args.eval_path)
    assert not out.exists()
    sources = json.loads(SOURCES.read_text())
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    rows = [await evaluate(case, sources) for case in cases]
    feedback_rows = [row for row in rows if row["feedback_checks"]]
    live_repairs = [row for row in rows if row["id"].startswith("AGENT_V1_") and row["id"].endswith("_REPAIR")]
    summary = {
        "cases": len(rows), "behavior_passes": sum(row["behavior_pass"] for row in rows),
        "terminal_passes": sum(row["terminal_pass"] for row in rows),
        "feedback_contract_passes": sum(row["feedback_contract_pass"] for row in feedback_rows),
        "feedback_contract_cases": len(feedback_rows),
        "live_shape_repair_contract_passes": sum(row["behavior_pass"] for row in live_repairs),
        "live_shape_repair_cases": len(live_repairs),
        "successful_outputs": sum(row["analyst"]["ok"] for row in rows),
        "successful_grounding_valid": sum(row["analyst"]["ok"] and row["oracle"]["valid"] for row in rows),
        "unexpected_successes": sum(row["analyst"]["ok"] and not row["expected_ok"] for row in rows),
        "history_preserved": sum(row["history_preserved"] for row in rows),
        "inputs_unchanged": sum(row["inputs_unchanged"] for row in rows),
        "gate_pass": all(row["behavior_pass"] for row in rows),
    }
    import pytest
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = {
        "evaluated_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tracked_worktree_clean": True, "dataset": str(dataset), "dataset_sha256": digest(dataset),
        "sources": str(SOURCES), "sources_sha256": digest(SOURCES),
        "harness_sha256": digest(__file__), "oracle_sha256": digest("src/evals/grounding_oracle.py"),
        "grounding_validator_sha256": digest("src/agents/analyst/grounding.py"),
        "runtime_sha256": digest("src/agents/analyst/agent.py"),
        "python": platform.python_version(), "pytest": pytest.__version__,
        "measurement": "Unconditional candidate injection with separate diagnostic-feedback and terminal expectations; not evidence that a live model follows feedback.",
        "command": f"PYTHONPATH=.:src conda run -n finsearch-arm python scripts/evals/agents/eval_agent_output_repair_v1.py --eval-path {dataset} --out-dir {out}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "per_case.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="data/evals/agents/v1/agent_eval_output_repair_v1.jsonl")
    parser.add_argument("--out-dir", required=True)
    asyncio.run(main(parser.parse_args()))
