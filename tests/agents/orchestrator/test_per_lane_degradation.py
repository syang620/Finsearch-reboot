from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agents.analyst import AnalystRunResult, build_analyst_prompt
from agents.contracts import (
    AnalysisTask,
    AnalystPacket,
    ContextItem,
    ContextItemKind,
    DegradationSummary,
    EvidenceLaneStatus,
    EvidenceLaneStatusSet,
    EvidenceLaneSummary,
    FilingMetadata,
    OpenIssue,
    PlannerIntent,
    SourceRef,
    StructuredFactEvidence,
)
from agents.orchestrator.agent_orchestrator import (
    _format_run_output,
    _packet_with_evidence_status,
    _route_after_retrieval_attach_open_issues,
)


def _plan(route: str, *, intent: str = "filing_fact") -> dict:
    return {
        "status": "completed",
        "intent": intent,
        "route": route,
        "retrieval_needed": route in {"kb", "hybrid"},
        "structured_fact_requests": (
            [{"subquestion": "What was revenue?", "metric_hint": "revenue"}]
            if route in {"structured_fact", "hybrid"}
            else []
        ),
        "metadata": {"ticker": "AAPL", "fiscal_year": 2024},
        "analysis_task": {"metric": "revenue"},
        "open_issues": [],
    }


def _kb_item() -> ContextItem:
    return ContextItem(
        context_id="ctx_kb",
        kind=ContextItemKind.TEXT,
        source=SourceRef(ticker="AAPL", fiscal_year=2024),
        payload={"content": "Revenue evidence from the filing."},
    )


def _structured_item() -> ContextItem:
    return ContextItem(
        context_id="ctx_structured",
        kind=ContextItemKind.STRUCTURED_FACT,
        source=SourceRef(ticker="AAPL", fiscal_year=2024),
        structured_fact=StructuredFactEvidence(
            metric_id="revenue",
            metric_label="Revenue",
            value=391_000_000_000.0,
            ticker="AAPL",
            fiscal_year=2024,
        ),
    )


def _packet(*items: ContextItem, intent: PlannerIntent = PlannerIntent.FILING_FACT) -> AnalystPacket:
    return AnalystPacket(
        plan_id="run-pr5",
        user_query="What was revenue?",
        intent=intent,
        metadata=FilingMetadata(ticker="AAPL", fiscal_year=2024),
        analysis_task=AnalysisTask(metric="revenue"),
        context_items=list(items),
    )


def _structured_result(status: str) -> dict:
    return {
        "subquestion": "What was revenue?",
        "resolved_metric_id": "revenue",
        "resolver_status": "resolved",
        "tool_result": {
            "ok": status == "ok",
            "status": status,
            "metric_id": "revenue",
            "missing_component_groups": [],
        },
    }


def _analyst(*, ok: bool = True) -> AnalystRunResult:
    return AnalystRunResult(
        ok=ok,
        status="ok" if ok else "error",
        answer="Revenue was reported." if ok else "Analysis failed.",
        intent=PlannerIntent.FILING_FACT,
        metric="revenue",
        error=None if ok else "analyst failure",
    )


def _output(values: dict) -> dict:
    return _format_run_output(
        run_id="run-pr5",
        state_snapshot=SimpleNamespace(values=values, interrupts=()),
    )


def test_lane_contracts_serialize_and_reject_noncanonical_degradation() -> None:
    lanes = EvidenceLaneStatusSet(
        kb=EvidenceLaneSummary(
            requested=True,
            attempted=True,
            status=EvidenceLaneStatus.PARTIAL,
            issues=[OpenIssue(code="RETRIEVAL_PARTIAL_FAILURE", message="one job failed")],
        )
    )

    assert lanes.model_dump(mode="json")["kb"]["status"] == "partial"
    with pytest.raises(ValidationError):
        DegradationSummary(
            active=True,
            affected_lanes=["structured_fact", "kb"],
            notice="degraded",
        )


def test_usable_kb_with_failed_operation_is_partial_and_usable() -> None:
    plan = _plan("kb")
    packet = _packet(_kb_item())
    state = {
        "plan_obj": plan,
        "retrieval_state": {"targets": [{}]},
        "retrieval_output": {
            "ok": True,
            "partial_failures": [{"error": "redacted by notice"}],
        },
        "packet": packet,
        "analyst_result": _analyst(),
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet

    output = _output(state)

    assert output["status"] == "degraded"
    assert output["ok"] is True
    assert output["lanes"]["kb"]["status"] == "partial"
    assert output["degradation"]["affected_lanes"] == ["kb"]
    assert "redacted by notice" not in output["degradation"]["notice"]


def test_structured_tool_partial_without_admitted_evidence_is_unusable() -> None:
    output = _output(
        {
            "plan_obj": _plan("structured_fact"),
            "structured_fact_results": [_structured_result("partial")],
            "packet": _packet(),
        }
    )

    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"
    assert output["lanes"]["structured_fact"]["status"] == "partial"
    assert output["degradation"]["active"] is True


def test_hybrid_total_failure_uses_terminal_structured_stage() -> None:
    output = _output(
        {
            "plan_obj": _plan("hybrid"),
            "retrieval_state": {"targets": [{}]},
            "retrieval_output": {"ok": False},
            "structured_fact_results": [_structured_result("error")],
            "packet": _packet(),
        }
    )

    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"
    assert output["degradation"]["affected_lanes"] == ["kb", "structured_fact"]


def test_filing_zero_lane_plan_fails_at_planner_but_nonfiling_may_complete() -> None:
    filing = _output(
        {
            "plan_obj": _plan("kb"),
            "packet": _packet(),
        }
    )
    nonfiling_plan = _plan("kb", intent="definition")
    nonfiling_plan.update({"retrieval_needed": False})
    nonfiling = _output(
        {
            "plan_obj": nonfiling_plan,
            "packet": _packet(intent=PlannerIntent.DEFINITION),
            "analyst_result": AnalystRunResult(
                ok=True,
                status="ok",
                answer="A definition.",
                intent=PlannerIntent.DEFINITION,
                metric="revenue",
            ),
        }
    )

    assert filing["status"] == "failed"
    assert filing["failure_stage"] == "planner"
    assert nonfiling["status"] == "completed"
    assert nonfiling["failure_stage"] == "none"


def test_pr3_kb_fallback_keeps_structured_issue_but_not_requested() -> None:
    plan = _plan("kb")
    issue = OpenIssue(
        code="STRUCTURED_FACT_CAPABILITY_REJECTED",
        message="Proposal was normalized to KB-only routing.",
    )
    plan["open_issues"] = [issue.model_dump(mode="json")]
    packet = _packet(_kb_item()).model_copy(update={"open_issues": [issue]})
    output = _output(
        {
            "plan_obj": plan,
            "retrieval_output": {"ok": True},
            "packet": packet,
            "analyst_result": _analyst(),
        }
    )

    assert output["lanes"]["structured_fact"]["status"] == "not_requested"
    assert output["lanes"]["structured_fact"]["issues"][0]["code"] == (
        "STRUCTURED_FACT_CAPABILITY_REJECTED"
    )
    assert output["degradation"]["active"] is False


def test_defensive_runtime_capability_rejection_is_unsupported() -> None:
    issue = OpenIssue(
        code="STRUCTURED_FACT_CAPABILITY_REJECTED",
        message="The runtime boundary rejected this request.",
        metadata={
            "outcome": "defensive_rejection",
            "question_class": "unsupported_derived_metric",
        },
    )
    packet = _packet().model_copy(update={"open_issues": [issue]})
    output = _output(
        {
            "plan_obj": _plan("structured_fact"),
            "structured_fact_results": [
                {
                    "resolver_status": "unresolved",
                    "resolver_reason": "Structured-fact capability rejected.",
                    "tool_result": None,
                }
            ],
            "packet": packet,
        }
    )

    assert output["lanes"]["structured_fact"]["status"] == "unsupported"
    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"


def test_analyst_failure_overrides_status_and_preserves_evidence_degradation() -> None:
    output = _output(
        {
            "plan_obj": _plan("hybrid"),
            "retrieval_output": {"ok": True},
            "structured_fact_results": [_structured_result("error")],
            "packet": _packet(_kb_item()),
            "analyst_result": _analyst(ok=False),
        }
    )

    assert output["status"] == "failed"
    assert output["failure_stage"] == "analyst"
    assert output["degradation"]["active"] is True
    assert output["degradation"]["affected_lanes"] == ["structured_fact"]


def test_fail_closed_route_skips_analyst_and_prompt_discloses_same_notice() -> None:
    plan = _plan("hybrid")
    packet = _packet(_kb_item())
    state = {
        "plan_obj": plan,
        "retrieval_output": {"ok": True},
        "structured_fact_results": [_structured_result("partial")],
        "packet": packet,
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet

    assert _route_after_retrieval_attach_open_issues(state) == "analyst"
    prompt = build_analyst_prompt(packet)
    assert packet.degradation.notice in prompt
    assert "do not claim coverage from unavailable lanes" in prompt.lower()

    state["packet"] = _packet_with_evidence_status(
        state={**state, "packet": _packet()},
        packet=_packet(),
    )
    assert _route_after_retrieval_attach_open_issues(state) == "finalize"
