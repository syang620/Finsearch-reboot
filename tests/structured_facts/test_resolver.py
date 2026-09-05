from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agents.contracts import FilingMetadata, PlannerTarget, StructuredFactRequest
from scripts.evals.agents.eval_structured_fact_resolver_v1 import RecordingClient, observe
from agents.orchestrator.agent_orchestrator import _execute_structured_fact_requests
from structured_facts.resolver import resolve_structured_fact_request

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = json.loads((ROOT / "data/evals/agents/v1/structured_fact_resolver_pr24.json").read_text())


@pytest.mark.parametrize("case", SNAPSHOT["cases"], ids=lambda case: case["id"])
def test_exact_merged_pr24_parity(case):
    assert asyncio.run(observe(case)) == case["expected"]


def test_validated_contracts_and_dictionaries_share_resolution():
    request = StructuredFactRequest(subquestion="What was revenue?", entity_hint="Apple Inc.")
    target = PlannerTarget(target_id=1, target_key="aapl-2024", company_name="Apple Inc.",
                           ticker="AAPL", fiscal_year=2024, form_type="10-K")
    metadata = FilingMetadata()
    typed = resolve_structured_fact_request(request, [target], metadata)
    raw = resolve_structured_fact_request(request.model_dump(mode="json"), [target.model_dump(mode="json")], metadata)
    assert typed == raw
    assert typed.status == "resolved"
    assert typed.selected_target == target.model_dump(mode="json")


def test_form_metadata_is_opaque_to_resolver():
    target = {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "future-unknown-form",
              "target_id": 91, "extra_metadata": {"preserve": True}}
    before = deepcopy(target)
    resolution = resolve_structured_fact_request({"metric_hint": "revenue"}, [target], FilingMetadata())
    assert resolution.status == "resolved"
    assert resolution.selected_target == before
    assert target == before


def test_legacy_orchestrator_still_ignores_non_dictionary_targets():
    # Direct typed resolver calls are supported, but PR24's raw-plan boundary
    # ignored model objects. Do not let the adapter broaden execution eligibility.
    target = PlannerTarget(target_id=1, target_key="aapl-2024", company_name="Apple",
                           ticker="AAPL", fiscal_year=2024, form_type="10-Q")
    client = RecordingClient()
    result, = asyncio.run(_execute_structured_fact_requests(
        plan_obj={"targets": [target], "structured_fact_requests": [{"metric_hint": "revenue"}]},
        client=client,
    ))
    assert result["resolver_status"] == "missing_inputs"
    assert result["resolved_ticker"] is None
    assert result["resolved_fiscal_year"] is None
    assert client.arguments == []


def test_resolver_import_does_not_load_orchestration_or_network_clients():
    code = """
import sys
from structured_facts.resolver import resolve_structured_fact_request
for prefix in ('agents.orchestrator', 'agents.planner', 'agents.retrieval',
               'agents.analyst', 'langgraph', 'qdrant_client', 'httpx'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
