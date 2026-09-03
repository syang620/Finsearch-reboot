"""Frozen synthetic candidate-injection evaluation of the real analyst workflow.

No live model is used by this deterministic gate. Semantic fixture labels are
annotations, never reported as judge observations or semantic improvements.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage

from agents.analyst import AnalystAgent
from agents.contracts import AnalystPacket
from evals.grounding_oracle import inspect_final


def packet_for(case: dict) -> AnalystPacket:
    source = {
        "ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K",
        "accession_no": "0000320193-24-000123", "report_date": "2024-09-28",
        "filing_date": "2024-11-01",
        "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
    }
    contexts = []
    for cid, kind, metric, value in [
        ("rev", "structured_fact", "revenue", 100.0),
        ("income", "structured_fact", "net_income", 20.0),
        ("kb", "text", None, None), ("table", "table", None, None),
        ("hidden", "text", None, None),
    ]:
        context = {
            "context_id": cid, "kind": kind, "source": source,
            "target_id": "AAPL:2024:10-K", "text": "Revenue was $100; net income was $20. Management attributed growth to demand.",
        }
        if metric:
            context["structured_fact"] = {
                "metric_id": metric, "metric_label": metric, "value": value, "unit": "USD",
                "ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K",
                "accession_number": source["accession_no"], "report_date": source["report_date"],
                "filed_date": source["filing_date"], "source_url": source["source_url"],
            }
        contexts.append(context)
    if case.get("survivor") == "kb":
        contexts = [item for item in contexts if item["kind"] != "structured_fact"]
    elif case.get("survivor") == "structured_fact":
        contexts = contexts[:2]
    if case.get("duplicate_visible_id"):
        contexts[1]["context_id"] = contexts[0]["context_id"]
    data = {
        "plan_id": "grounding-fixture", "user_query": case["query"],
        "intent": case.get("intent", "filing_fact"), "metadata": source,
        "analysis_task": {"task_type": "compute" if case.get("calculator") else "extract", "metric": "revenue"},
        "context_quality": "high", "context_items": contexts,
    }
    return AnalystPacket.model_validate(data)


class CandidateModel:
    def __init__(self, case):
        self.case, self.calls, self.candidates = case, 0, 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.case.get("calculator") and self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "financial_evaluator", "id": "calc-1",
                "args": {"expression": "revenue-income", "variables": {"revenue": "100", "income": "20"}},
            }])
        payloads = self.case["outputs"]
        payload = payloads[min(self.candidates, len(payloads) - 1)]
        self.candidates += 1
        return AIMessage(content="", tool_calls=[{"name": "FinalAnswer", "id": f"final-{self.calls}", "args": payload}])


class Calculator:
    async def ainvoke(self, args):
        # Injected tools return their payload directly (not an MCP envelope).
        return {"result": 80.0, **args}


async def evaluate(case):
    packet = packet_for(case)
    agent = AnalystAgent(max_attempts=case.get("max_attempts", 1), max_context_items=case.get("visible_limit", 4))
    model = CandidateModel(case)
    agent._bound_model_override = model
    if case.get("calculator"):
        agent._tool_map = {"financial_evaluator": Calculator()}
    result = await agent.arun(packet)
    await agent.aclose()
    output = result.model_dump(mode="json")
    oracle = inspect_final(packet.model_dump(mode="json"), output, agent.max_context_items)
    codes = [issue["code"] for issue in output.get("open_issues", [])]
    behavior = output["ok"] == case["expected_ok"]
    if case["expected_ok"]:
        behavior = behavior and oracle["valid"]
    else:
        behavior = behavior and output["status"] in {"grounding_error", "error"}
    if case.get("expected_sanitized"):
        behavior = behavior and "GROUNDING_UNKNOWN_CONTEXT_ID" in codes
    if case.get("expected_repair"):
        behavior = behavior and model.candidates > 1
    return {
        "id": case["id"], "expected_ok": case["expected_ok"], "behavior_pass": behavior,
        "candidate_calls": model.candidates, "model_calls": model.calls,
        "injected_candidates": case["outputs"], "analyst_packet": packet.model_dump(mode="json"),
        "analyst": output, "oracle": oracle,
        "semantic_annotation": case.get("semantic_annotation"),
    }


async def main(args):
    dataset = Path(args.eval_path)
    cases = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    rows = [await evaluate(case) for case in cases]
    success = [row for row in rows if row["oracle"]["successful"]]
    summary = {
        "cases": len(rows), "behavior_passes": sum(row["behavior_pass"] for row in rows),
        "behavior_accuracy": sum(row["behavior_pass"] for row in rows) / len(rows),
        "successful_outputs": len(success),
        "successful_output_integrity_rate": sum(row["oracle"]["valid"] for row in success) / len(success) if success else None,
        "gate_pass": all(row["behavior_pass"] for row in rows),
        "semantic_judge": {"status": "not_run", "note": "Fixture annotations are not judge scores."},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "per_case.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import pytest
    manifest = {
        "runtime_commit": args.runtime_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(Path("src/evals/grounding_oracle.py").read_bytes()).hexdigest(),
        "dataset": str(dataset), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "python": platform.python_version(), "pytest": pytest.__version__,
        "measurement": "deterministic synthetic FinalAnswer injection through AnalystAgent.arun; no live model",
        "legacy_binding_policy": "Absent finalized claims remain absent; never infer from flat citations.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="data/evals/agents/v1/agent_eval_grounding_v1.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runtime-commit")
    asyncio.run(main(parser.parse_args()))
