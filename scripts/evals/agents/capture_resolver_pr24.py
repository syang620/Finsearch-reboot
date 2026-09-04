"""Print an apply_patch snapshot captured ONLY from the merged PR24 runtime.

Run from the repository root with PYTHONPATH=.:src. No network calls are made.
The isolated path allows resolution to run; the execution path retains the real
capability policy. Expected outputs are observations, never resolver reimplementations.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import pytest

from agents.orchestrator import agent_orchestrator as old
from structured_facts.capabilities import (
    StructuredFactCapabilityDecision,
    StructuredFactQuestionClass,
)

BASE = "2860c430036f3fa9e9488663fed9d086a03820af"
OUTPUT = "data/evals/agents/v1/structured_fact_resolver_pr24.json"


class RecordingClient:
    def __init__(self, error=None):
        self.arguments = []
        self.error = error

    async def get_metric(self, **kwargs):
        self.arguments.append(kwargs)
        if self.error:
            raise ValueError(self.error)
        return {"status": "ok", "snapshot_stub": True, "arguments": kwargs}


def cases():
    rows = []
    default = {"ticker": "AAPL", "fiscal_year": 2024}

    def add(name, request=None, **plan):
        plan.setdefault("metadata", default)
        plan.setdefault("structured_fact_requests", [request or {}])
        rows.append({"id": name, "plan": deepcopy(plan)})

    for metric_id, definition in old.METRIC_REGISTRY.items():
        terms = [metric_id, definition.label, *old._STRUCTURED_FACT_ALIAS_MAP.get(metric_id, ())]
        for index, term in enumerate(terms):
            add(f"{metric_id}-term-{index}", {"metric_hint": term})
            add(f"{metric_id}-phrase-{index}", {"subquestion": f"What was the {term} reported?"})
        add(f"{metric_id}-normalized", {"metric_hint": f"  {metric_id.upper().replace('_', '-')}  "})

    for index, request in enumerate([
        {}, {"metric_hint": "cash"}, {"metric_hint": "profit"},
        {"metric_hint": "assets"}, {"metric_hint": "equity"},
        {"metric_hint": "revenue and total debt"},
        {"metric_hint": "gross profit", "subquestion": "revenue"},
        {"metric_hint": "unknown", "subquestion": "total debt"},
        {"metric_hint": " \n ", "subquestion": "total debt"},
        {"metric_hint": "revenue."}, {"metric_hint": "revenue!"},
        {"metric_hint": "salesforce"}, {"metric_hint": "net sales"},
        {"metric_hint": "cash flow from operations"},
        {"metric_hint": 123}, {"metric_hint": False},
        {"metric_hint": ["revenue"]},
        {"metric_hint": "revenue", "fiscal_period": "Q1"},
        {"metric_hint": "gross margin"},
        {"metric_hint": "revenue", "subquestion": "revenue per share"},
    ]):
        add(f"metric-edge-{index}", request)

    apple = {"target_id": 1, "target_key": "aapl-2024", "ticker": "AAPL",
             "company_name": "Apple Inc.", "fiscal_year": 2024, "form_type": "10-K"}
    microsoft = {"target_id": 2, "target_key": "msft-2023", "ticker": "MSFT",
                 "company_name": "Microsoft", "fiscal_year": 2023, "form_type": "10-K/A"}
    for index, (request, metadata, targets) in enumerate([
        ({}, {}, []), ({"entity_hint": "AAPL"}, {}, []),
        ({"fiscal_year": 2024}, {}, []),
        ({"entity_hint": "MSFT", "fiscal_year": 2023}, default, [apple]),
        ({"entity_hint": "Apple Inc."}, {"ticker": "MSFT", "fiscal_year": 2022}, [apple]),
        ({"entity_hint": "Apple Inc."}, {}, [apple, microsoft]),
        ({"entity_hint": "Apple Inc"}, {}, [apple, microsoft]),
        ({"entity_hint": "aapl"}, {}, [apple, microsoft]),
        ({"entity_hint": "AAPL"}, {}, [apple, {**apple, "target_id": 3, "fiscal_year": 2025}]),
        ({"entity_hint": "AAPL", "fiscal_year": 2025}, {}, [apple, {**apple, "target_id": 3, "fiscal_year": 2025}]),
        ({"entity_hint": "UNKNOWN", "fiscal_year": 2022}, {}, [apple]),
        ({"entity_hint": "Unknown company", "fiscal_year": 2022}, {}, [apple, microsoft]),
        ({}, {}, [apple]), ({}, {}, [apple, microsoft]),
        ({"entity_hint": "AAPL", "fiscal_year": "2024"}, {}, [apple]),
        ({"fiscal_year": "invalid"}, {}, [apple]),
        ({"fiscal_year": 0}, {}, [apple]),
        ({"fiscal_year": False}, {}, [apple]),
        ({"fiscal_year": 2024.9}, {}, [apple]),
        ({"fiscal_year": -1}, {}, [apple]),
        ({"entity_hint": "BRK.B"}, {}, [{"ticker": "BRK.B", "fiscal_year": 2024}]),
        ({"entity_hint": "BRK-B"}, {}, [{"ticker": "BRK-B", "fiscal_year": 2024}]),
        ({"entity_hint": "TOOLONGTICKER"}, {}, [apple, microsoft]),
        ({"entity_hint": "Apple Inc."}, {}, [None, "invalid", {**apple, "ticker": "aapl"}]),
        ({"entity_hint": "AAPL"}, {}, [{**apple, "fiscal_year": None}, apple]),
        ({"entity_hint": "AAPL"}, {}, [{**apple, "form_type": "10-Q"}]),
        ({"entity_hint": "AAPL"}, {}, [{**apple, "form_type": "10-K/A"}]),
        ({}, {"ticker": " aapl ", "fiscal_year": "2024"}, []),
    ]):
        add(f"inputs-{index}", {"metric_hint": "revenue", **request}, metadata=metadata, targets=targets)
    for hint in ("cash", "unknown", "profit", "revenue"):
        add(f"missing-precedence-{hint}", {"metric_hint": hint}, metadata={})
    add("nonannual-issue", {"metric_hint": "revenue"}, open_issues=[{"code": "FORM_NOT_10K_DATASET"}])
    add("hostile-original", {"metric_hint": "revenue"}, original_user_query="What is Apple's revenue per share?")
    add("mixed-requests", structured_fact_requests=[
        {"metric_hint": "revenue"}, None, "ignored",
        {"metric_hint": "gross margin"}, {"metric_hint": "cash"},
        {"metric_hint": "total debt", "entity_hint": "MSFT", "fiscal_year": 2023},
        {"metric_hint": "revenue"},
    ], targets=[apple, microsoft])
    add("empty-requests", structured_fact_requests=[])
    add("tool-error", {"metric_hint": "revenue"})
    rows[-1]["tool_error"] = "frozen metric execution failure"
    return rows


async def observe(case):
    plan = deepcopy(case["plan"])
    requests = [item for item in plan.get("structured_fact_requests", []) if isinstance(item, dict)]
    isolated = []
    allow = StructuredFactCapabilityDecision(
        question_class=StructuredFactQuestionClass.SUPPORTED_DIRECT_METRIC,
        permitted=True, matched_metric_ids=(), reason="Snapshot isolation only",
    )
    for request in requests:
        ticker, year, selected = old._resolve_structured_fact_inputs(plan_obj=plan, request=request)
        client = RecordingClient()
        with patch.object(old, "_structured_fact_capability_decisions", return_value=(allow,)):
            result, = await old._execute_structured_fact_requests(
                plan_obj={**plan, "structured_fact_requests": [request]}, client=client,
            )
        isolated.append({
            "status": result["resolver_status"], "metric_id": result["resolved_metric_id"],
            "ticker": ticker, "fiscal_year": year, "selected_target": selected,
            "reason": result["resolver_reason"], "sec_metric_arguments": client.arguments,
        })
    decisions = old._structured_fact_capability_decisions(plan_obj=plan, requests=requests)
    client = RecordingClient(case.get("tool_error"))
    results = await old._execute_structured_fact_requests(plan_obj=plan, client=client)
    assert plan == case["plan"], "Old path mutated input"
    return {"resolutions": isolated, "capability_decisions": [asdict(item) for item in decisions],
            "execution_results": results, "sec_metric_arguments": client.arguments}


async def main():
    paths = ["src/agents/orchestrator/agent_orchestrator.py", "src/structured_facts/capabilities.py",
             "src/mcp_server/tools/sec_metric_registry.py", "src/agents/contracts.py", "src/agents/text_utils.py"]
    hashes = {}
    for path in paths:
        original = subprocess.check_output(["git", "show", f"{BASE}:{path}"])
        assert Path(path).read_bytes() == original, f"Refuse capture from changed source: {path}"
        hashes[path] = hashlib.sha256(original).hexdigest()
    rows = cases()
    for case in rows:
        case["expected"] = await observe(case)
    document = {"source_commit": BASE, "source_sha256": hashes,
                "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "pytest": pytest.__version__,
                "measurement": "Actual merged-PR24 outputs; isolated resolution plus unchanged capability-gated execution; recording SEC client, no network.",
                "cases": rows}
    header = json.dumps({key: value for key, value in document.items() if key != "cases"}, indent=2)
    payload = header[:-2] + ',\n  "cases": [\n'
    payload += ",\n".join("    " + json.dumps(row, ensure_ascii=False) for row in rows)
    payload += "\n  ]\n}\n"
    assert not Path(OUTPUT).exists(), "Frozen snapshot already exists"
    print("*** Begin Patch\n*** Add File: " + OUTPUT)
    print("\n".join("+" + line for line in payload.splitlines()))
    print("*** End Patch")


if __name__ == "__main__":
    asyncio.run(main())
