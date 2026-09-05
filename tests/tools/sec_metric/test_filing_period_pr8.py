"""Frozen PR8 semantics, input-order invariance and analyst admission regressions."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from mcp_server.tools.sec_metric import get_metric

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pr8_eval", ROOT / "scripts/evals/agents/eval_filing_period_pr8.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
CASES = json.loads((ROOT / oracle.DATASET).read_text())["cases"]


def run(c):
    client = oracle.FixtureClient(c)
    return asyncio.run(get_metric(**c["request"], client=client)).model_dump(mode="json")


@pytest.mark.parametrize("c", CASES, ids=lambda c: c["id"])
def test_frozen_expected_and_order(c):
    actual = run(c)
    assert actual == c["expected_pr8_result"]
    assert not set(oracle.differences(c["old_result"], actual)) - set(c["allowed_fields"])
    assert oracle.provenance_consistent(c, actual)
    for seed in range(5):
        assert run(oracle.shuffled(c, seed)) == actual


@pytest.mark.parametrize("c", CASES, ids=lambda c: c["id"])
def test_actual_tool_results_admitted_iff_usable(c):
    from agents.contracts import AnalystPacket, AnalysisTask, FilingMetadata, PlannerIntent
    from agents.orchestrator.agent_orchestrator import _structured_fact_evidence_from_result
    packet = AnalystPacket(plan_id="pr8", user_query="What was the annual metric?",
                           intent=PlannerIntent.FILING_FACT,
                           metadata=FilingMetadata(ticker="AAPL", fiscal_year=2025, form_type="10-K"),
                           analysis_task=AnalysisTask(metric=c["request"]["metric_id"]))
    actual = run(c)
    evidence, issue = _structured_fact_evidence_from_result(packet=packet, result={
        "resolver_status": "resolved", "resolved_metric_id": c["request"]["metric_id"],
        "resolved_ticker": "AAPL", "resolved_fiscal_year": 2025, "tool_result": actual,
    })
    if actual["ok"]:
        assert evidence is not None and issue is None
        assert evidence.accession_number == actual["accession_number"]
        assert evidence.components == actual["components"]
        if actual["primary_fact"]:
            assert evidence.start_date == actual["primary_fact"]["start_date"]
        from agents.analyst import build_analyst_prompt
        from agents.contracts import ContextItem, ContextItemKind
        packet.context_items = [ContextItem(context_id="pr8_evidence", kind=ContextItemKind.STRUCTURED_FACT,
                                            structured_fact=evidence)]
        prompt = build_analyst_prompt(packet)
        if actual["primary_fact"] and actual["primary_fact"]["start_date"]:
            assert f'"start_date":"{actual["primary_fact"]["start_date"]}"' in prompt
        for component in actual["components"]:
            if component["start_date"]:
                assert f'"start_date":"{component["start_date"]}"' in prompt
    else:
        assert evidence is None and issue is not None


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "invalid", None])
def test_invalid_values_never_admitted(value):
    c = deepcopy(CASES[0])
    records = next(iter(c["companyfacts"]["facts"]["us-gaap"].values()))["units"]["USD"]
    records[0]["val"] = value
    result = run(c)
    assert result["status"] == "not_found" and result["value"] is None
    assert result["trace"]["selection"][0]["reason"] == "INVALID_NUMERIC_VALUE"


def test_wrong_units_rejected_with_trace():
    c = deepcopy(CASES[0])
    units = next(iter(c["companyfacts"]["facts"]["us-gaap"].values()))["units"]
    units["EUR"] = units.pop("USD")
    result = run(c)
    assert result["status"] == "not_found"
    assert result["trace"]["selection"][0]["reason"] == "UNIT_MISMATCH"


def test_no_rounding_away_conflicts():
    c = deepcopy(CASES[0])
    records = next(iter(c["companyfacts"]["facts"]["us-gaap"].values()))["units"]["USD"]
    records[0]["val"] = 1.0
    records.append({**records[0], "val": 1.000000001})
    assert run(c)["status"] == "ambiguous"


def test_truncated_metadata_is_unavailable_not_index_error():
    c = deepcopy(CASES[0])
    c["submissions"]["filings"]["recent"]["accessionNumber"] = []
    result = run(c)
    assert result["status"] == "not_found"
    assert result["trace"]["reason"] == "INVALID_FILING_METADATA"


def test_shadow_rejects_forged_period_even_if_flags_claim_match():
    c = CASES[0]
    forged = deepcopy(c["expected_pr8_result"])
    forged["primary_fact"]["start_date"] = "2025-07-01"
    assert not oracle.provenance_consistent(c, forged)


@pytest.mark.parametrize("field", ["source_url", "filed_date"])
def test_shadow_checks_anchor_provenance_against_submissions(field):
    c = CASES[0]
    forged = deepcopy(c["expected_pr8_result"])
    forged[field] = "forged"
    assert not oracle.provenance_consistent(c, forged)


def test_shadow_gate_requires_unchanged_parity_independently():
    row = {"exact_expected": True, "unexpected_differences": [], "order_invariant": True,
           "provenance_consistent": True, "calls_correct": True, "unchanged_case": True,
           "unchanged_parity": False, "semantic_change_expected": False, "analyst_visibility": True}
    assert not oracle.summarize([row])["passed"]


def test_shadow_gate_detects_renderer_dropping_start(monkeypatch):
    from agents.analyst import agent
    original = agent.render_structured_fact_evidence
    monkeypatch.setattr(agent, "render_structured_fact_evidence", lambda evidence:
                        original(evidence.model_copy(update={"start_date": None})))
    assert not oracle.analyst_visibility(CASES[0], CASES[0]["expected_pr8_result"])


def test_unsupported_metric_preserves_old_contract_without_calls():
    c = deepcopy(CASES[0])
    client = oracle.FixtureClient(c)
    result = asyncio.run(get_metric(ticker="AAPL", fiscal_year=2025, metric_id="unknown", client=client))
    assert result.status == "unsupported_metric" and result.trace == {}
    assert client.calls == []


def test_real_filing_period_evidence_covers_exact_supported_lengths():
    data = json.loads((ROOT / oracle.DATASET).read_text())
    from datetime import date
    days = set()
    for source in data["period_sources"]:
        for p in source["annual_contexts"]:
            assert (date.fromisoformat(p["end"])-date.fromisoformat(p["start"])).days+1 == p["inclusive_days"]
            days.add(p["inclusive_days"])
    assert days == {364, 365, 366, 371}
