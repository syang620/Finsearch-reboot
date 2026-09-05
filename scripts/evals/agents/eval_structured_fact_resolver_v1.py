"""Compare current resolution/execution with immutable observed PR24 outputs."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import pytest

from agents.contracts import FilingMetadata
from agents.orchestrator import agent_orchestrator as runtime
from structured_facts.capabilities import StructuredFactCapabilityDecision, StructuredFactQuestionClass
from structured_facts.resolver import resolve_structured_fact_request

DATASET = Path("data/evals/agents/v1/structured_fact_resolver_pr24.json")


class RecordingClient:
    def __init__(self, error=None):
        self.arguments = []
        self.error = error

    async def get_metric(self, **kwargs):
        self.arguments.append(kwargs)
        if self.error:
            raise ValueError(self.error)
        return {"status": "ok", "snapshot_stub": True, "arguments": kwargs}


async def observe(case):
    plan = deepcopy(case["plan"])
    requests = [item for item in plan.get("structured_fact_requests", []) if isinstance(item, dict)]
    isolated = []
    allow = StructuredFactCapabilityDecision(
        question_class=StructuredFactQuestionClass.SUPPORTED_DIRECT_METRIC,
        permitted=True, matched_metric_ids=(), reason="Snapshot isolation only",
    )
    for request in requests:
        resolution = resolve_structured_fact_request(
            request=request, targets=plan.get("targets") or [],
            metadata=FilingMetadata.model_validate(plan.get("metadata") or {}),
        )
        client = RecordingClient()
        with patch.object(runtime, "_structured_fact_capability_decisions", return_value=(allow,)):
            result, = await runtime._execute_structured_fact_requests(
                plan_obj={**plan, "structured_fact_requests": [request]}, client=client,
            )
        # Check the adapter as well as the resolver; neither constructs expected outputs.
        assert (result["resolver_status"], result["resolved_metric_id"], result["resolved_ticker"],
                result["resolved_fiscal_year"], result["resolver_reason"]) == (
                    resolution.status, resolution.metric_id, resolution.ticker,
                    resolution.fiscal_year, resolution.reason,
                )
        isolated.append({**resolution.model_dump(mode="json"), "sec_metric_arguments": client.arguments})
    decisions = runtime._structured_fact_capability_decisions(plan_obj=plan, requests=requests)
    client = RecordingClient(case.get("tool_error"))
    results = await runtime._execute_structured_fact_requests(plan_obj=plan, client=client)
    assert plan == case["plan"], "Current path mutated input"
    return json.loads(json.dumps({
        "resolutions": isolated, "capability_decisions": [asdict(item) for item in decisions],
        "execution_results": results, "sec_metric_arguments": client.arguments,
    }))


async def evaluate():
    document = json.loads(DATASET.read_text())
    rows = []
    for case in document["cases"]:
        actual = await observe(case)
        mismatches = [key for key in case["expected"] if actual.get(key) != case["expected"][key]]
        rows.append({"id": case["id"], "passed": not mismatches,
                     "mismatched_sections": mismatches, "actual": actual})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = asyncio.run(evaluate())
    summary = {"cases": len(rows), "passed": sum(row["passed"] for row in rows),
               "exact_parity": all(row["passed"] for row in rows)}
    implementation = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = [str(DATASET), __file__, "src/structured_facts/resolver.py", "src/structured_facts/models.py",
               "src/agents/orchestrator/agent_orchestrator.py", "src/structured_facts/capabilities.py"]
    root = Path.cwd()
    hashes = {str(Path(path).resolve().relative_to(root)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in sources}
    manifest = {
        "evaluated_commit": implementation, "snapshot_source_commit": json.loads(DATASET.read_text())["source_commit"],
        "tracked_worktree_clean": not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "pytest": pytest.__version__, "sha256": hashes,
        "oracle": "Frozen observed merged-PR24 outputs; no expected values computed by current runtime.",
    }
    for name, value in (("summary", summary), ("manifest", manifest)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    (out / "per_case.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(json.dumps(summary))
    raise SystemExit(0 if summary["exact_parity"] else 1)


if __name__ == "__main__":
    main()
