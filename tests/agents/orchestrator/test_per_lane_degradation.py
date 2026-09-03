from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agents.analyst import (
    AnalystRunResult,
    build_analyst_prompt,
    build_packet_from_retrieval_output,
)
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
    _append_structured_fact_context_items,
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


def _successful_structured_result(
    *, subquestion: str, metric_id: str, value: object
) -> dict:
    result = _structured_result("ok")
    result.update(
        {
            "subquestion": subquestion,
            "resolved_metric_id": metric_id,
        }
    )
    result["tool_result"].update(
        {
            "metric_id": metric_id,
            "value": value,
        }
    )
    return result


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


@pytest.mark.parametrize(
    "issue_code",
    ["TABLE_HYDRATION_FAILED", "TABLE_MARKDOWN_EMPTY"],
)
def test_kb_admission_loss_with_surviving_evidence_is_partial_and_usable(
    issue_code: str,
) -> None:
    issue = OpenIssue(
        code=issue_code,
        message="raw table admission detail",
    )
    packet = _packet(_kb_item()).model_copy(update={"open_issues": [issue]})
    state = {
        "plan_obj": _plan("kb"),
        "retrieval_output": {
            "ok": True,
            "top_tables": [
                {"doc_id": "AAPL_10-K_2024::table::1"},
                {"doc_id": "AAPL_10-K_2024::text::2"},
            ],
            "partial_failures": [],
        },
        "packet": packet,
        "analyst_result": _analyst(),
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet

    assert _route_after_retrieval_attach_open_issues(state) == "analyst"
    output = _output(state)

    assert output["status"] == "degraded"
    assert output["ok"] is True
    assert output["lanes"]["kb"]["status"] == "partial"
    assert output["lanes"]["kb"]["issues"][0]["code"] == issue_code
    assert output["degradation"]["affected_lanes"] == ["kb"]
    assert "raw table admission detail" not in output["degradation"]["notice"]


def test_kb_admission_loss_without_surviving_evidence_is_failed_and_unusable() -> None:
    issue = OpenIssue(
        code="TABLE_HYDRATION_FAILED",
        message="raw table admission detail",
    )
    packet = _packet().model_copy(update={"open_issues": [issue]})
    state = {
        "plan_obj": _plan("kb"),
        "retrieval_output": {
            "ok": True,
            "top_tables": [{"doc_id": "AAPL_10-K_2024::table::1"}],
        },
        "packet": packet,
    }

    assert _route_after_retrieval_attach_open_issues(state) == "finalize"
    output = _output(state)

    assert output["status"] == "failed"
    assert output["failure_stage"] == "retrieval"
    assert output["lanes"]["kb"]["status"] == "failed"


def test_kb_candidate_count_above_packet_cap_requires_admission_issue_for_partial() -> None:
    packet = _packet(
        _kb_item(),
        _kb_item().model_copy(update={"context_id": "ctx_kb_2"}),
        _kb_item().model_copy(update={"context_id": "ctx_kb_3"}),
    )
    state = {
        "plan_obj": _plan("kb"),
        "retrieval_output": {
            "ok": True,
            "top_tables": [
                {"doc_id": f"AAPL_10-K_2024::text::{index}"}
                for index in range(4)
            ],
        },
        "packet": packet,
        "analyst_result": _analyst(),
    }

    assert _output(state)["lanes"]["kb"]["status"] == "ok"

    issue = OpenIssue(
        code="TABLE_HYDRATION_FAILED",
        message="one candidate was rejected before a later candidate backfilled the cap",
    )
    state["packet"] = packet.model_copy(update={"open_issues": [issue]})
    assert _output(state)["lanes"]["kb"]["status"] == "partial"


def test_empty_text_loss_is_partial_when_later_candidates_backfill_packet_cap() -> None:
    retrieval_output = {
        "ok": True,
        "top_tables": [
            {
                "table_name": f"text-{index}",
                "total_score": 10 - index,
                "table": {
                    "payload": {
                        "doc_type": "text",
                        "doc_id": f"AAPL_10-K_2024::text::{index}",
                        "content": "" if index == 0 else f"usable evidence {index}",
                        "ticker": "AAPL",
                        "fiscal_year": 2024,
                        "form_type": "10-K",
                    }
                },
            }
            for index in range(4)
        ],
        "metadata_used": {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "form_type": "10-K",
        },
    }
    packet = build_packet_from_retrieval_output(
        user_query="What was revenue?",
        retrieval_output=retrieval_output,
        intent=PlannerIntent.FILING_FACT,
        analysis_task={"task_type": "extract", "metric": "revenue"},
    )
    state = {
        "plan_obj": _plan("kb"),
        "retrieval_output": retrieval_output,
        "packet": packet,
        "analyst_result": _analyst(),
    }

    assert len(packet.context_items) == 3
    assert [issue.code for issue in packet.open_issues] == ["EMPTY_TEXT_CONTEXT"]

    output = _output(state)

    assert output["status"] == "degraded"
    assert output["lanes"]["kb"]["status"] == "partial"
    assert [issue["code"] for issue in output["lanes"]["kb"]["issues"]] == [
        "EMPTY_TEXT_CONTEXT"
    ]


def test_unsupported_candidate_loss_is_partial_when_valid_candidates_backfill_packet_cap() -> None:
    retrieval_output = {
        "ok": True,
        "top_tables": [
            {
                "table_name": "unrecognized",
                "total_score": 50,
                "table": {
                    "payload": {
                        "doc_type": "chart",
                        "doc_id": "AAPL_10-K_2024::chart::0",
                    }
                },
            },
            *[
                {
                    "table_name": f"text-{index}",
                    "total_score": 40 - index,
                    "table": {
                        "payload": {
                            "doc_type": "text",
                            "doc_id": f"AAPL_10-K_2024::text::{index}",
                            "content": f"usable evidence {index}",
                            "ticker": "AAPL",
                            "fiscal_year": 2024,
                            "form_type": "10-K",
                        }
                    },
                }
                for index in range(1, 4)
            ],
        ],
        "metadata_used": {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "form_type": "10-K",
        },
    }
    packet = build_packet_from_retrieval_output(
        user_query="What was revenue?",
        retrieval_output=retrieval_output,
        intent=PlannerIntent.FILING_FACT,
        analysis_task={"task_type": "extract", "metric": "revenue"},
    )
    state = {
        "plan_obj": _plan("kb"),
        "retrieval_output": retrieval_output,
        "packet": packet,
        "analyst_result": _analyst(),
    }

    assert len(packet.context_items) == 3
    assert [issue.code for issue in packet.open_issues] == [
        "RETRIEVAL_CANDIDATE_UNSUPPORTED"
    ]
    assert packet.open_issues[0].message == (
        "Skipping retrieved candidate because it could not be identified as a "
        "supported table or text evidence type."
    )
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet
    assert packet.lanes.kb.status == EvidenceLaneStatus.PARTIAL
    assert packet.degradation.active
    assert _route_after_retrieval_attach_open_issues(state) == "analyst"

    output = _output(state)

    assert output["status"] == "degraded"
    assert output["ok"] is True
    assert output["lanes"]["kb"]["status"] == "partial"
    assert output["lanes"]["kb"]["issues"][0]["code"] == (
        "RETRIEVAL_CANDIDATE_UNSUPPORTED"
    )


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


@pytest.mark.parametrize(
    ("operation_statuses", "expected_lane_status"),
    [
        (["partial", "partial"], "partial"),
        (["partial", "error"], "failed"),
        (["partial", "ambiguous"], "failed"),
        (["partial", "unsupported_metric"], "failed"),
    ],
)
def test_structured_without_admitted_evidence_is_partial_only_when_all_operations_are_partial(
    operation_statuses: list[str],
    expected_lane_status: str,
) -> None:
    plan = _plan("structured_fact")
    plan["structured_fact_requests"] = [
        {
            "subquestion": f"What was metric {index}?",
            "metric_hint": f"metric_{index}",
        }
        for index, _status in enumerate(operation_statuses)
    ]
    results = []
    for index, status in enumerate(operation_statuses):
        result = _structured_result(status)
        result.update(
            {
                "subquestion": f"What was metric {index}?",
                "resolved_metric_id": f"metric_{index}",
            }
        )
        if status == "ambiguous":
            result.update(
                {
                    "resolved_metric_id": None,
                    "resolver_status": "ambiguous",
                    "resolver_reason": "Multiple registry metrics matched.",
                    "tool_result": None,
                }
            )
        elif status == "partial":
            result["tool_result"].update(
                {
                    "error": "A required component was unavailable.",
                    "missing_component_groups": ["required_component"],
                }
            )
        else:
            result["tool_result"]["error"] = "Structured fact evidence is unavailable."
        results.append(result)

    packet = _append_structured_fact_context_items(
        packet=_packet(),
        structured_fact_results=results,
    )
    state = {
        "plan_obj": plan,
        "structured_fact_results": results,
        "packet": packet,
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet

    assert packet.context_items == []
    assert packet.open_issues
    assert _route_after_retrieval_attach_open_issues(state) == "finalize"

    output = _output(state)

    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"
    assert output["lanes"]["structured_fact"]["status"] == expected_lane_status
    assert output["degradation"]["active"] is True


@pytest.mark.parametrize(
    "operation_status",
    ["partial", "ambiguous", "unsupported_metric"],
)
def test_incomplete_structured_results_without_admitted_evidence_fail_closed(
    operation_status: str,
) -> None:
    plan = _plan("structured_fact")
    plan["structured_fact_requests"].append(
        {"subquestion": "What was net income?", "metric_hint": "net_income"}
    )

    output = _output(
        {
            "plan_obj": plan,
            "structured_fact_results": [_structured_result(operation_status)],
            "packet": _packet(),
        }
    )

    assert output["lanes"]["structured_fact"]["status"] == "failed"
    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"


def test_incomplete_structured_results_with_admitted_evidence_are_partial() -> None:
    plan = _plan("structured_fact")
    plan["structured_fact_requests"].append(
        {"subquestion": "What was net income?", "metric_hint": "net_income"}
    )
    result = _successful_structured_result(
        subquestion="What was revenue?",
        metric_id="revenue",
        value=391_000_000_000.0,
    )
    packet = _append_structured_fact_context_items(
        packet=_packet(),
        structured_fact_results=[result],
    )
    state = {
        "plan_obj": plan,
        "structured_fact_results": [result],
        "packet": packet,
        "analyst_result": _analyst(),
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet

    output = _output(state)

    assert len(packet.context_items) == 1
    assert output["lanes"]["structured_fact"]["status"] == "partial"
    assert output["status"] == "degraded"
    assert output["analyst"] is not None


def test_malformed_structured_results_are_attempted_and_cannot_be_ok() -> None:
    valid_result = _successful_structured_result(
        subquestion="What was revenue?",
        metric_id="revenue",
        value=391_000_000_000.0,
    )
    usable_output = _output(
        {
            "plan_obj": _plan("structured_fact"),
            "structured_fact_results": [valid_result, "malformed"],
            "packet": _packet(_structured_item()),
            "analyst_result": _analyst(),
        }
    )
    unusable_output = _output(
        {
            "plan_obj": _plan("structured_fact"),
            "structured_fact_results": ["malformed"],
            "packet": _packet(),
        }
    )

    assert usable_output["lanes"]["structured_fact"]["status"] == "partial"
    assert usable_output["status"] == "degraded"
    assert unusable_output["lanes"]["structured_fact"]["attempted"] is True
    assert unusable_output["lanes"]["structured_fact"]["status"] == "failed"
    assert unusable_output["status"] == "failed"


def test_structured_admission_records_malformed_result_shapes() -> None:
    valid_result = _successful_structured_result(
        subquestion="What was revenue?",
        metric_id="revenue",
        value=391_000_000_000.0,
    )
    packet = _append_structured_fact_context_items(
        packet=_packet(),
        structured_fact_results=[valid_result, "malformed"],
    )
    invalid_container_packet = _append_structured_fact_context_items(
        packet=_packet(),
        structured_fact_results=42,
    )

    assert len(packet.context_items) == 1
    assert [issue.code for issue in packet.open_issues] == [
        "STRUCTURED_FACT_INVALID_EVIDENCE"
    ]
    assert invalid_container_packet.context_items == []
    assert [issue.code for issue in invalid_container_packet.open_issues] == [
        "STRUCTURED_FACT_INVALID_EVIDENCE"
    ]


@pytest.mark.parametrize(
    "raw_results",
    [42, "malformed", {"tool_result": {"status": "ok"}}],
)
def test_invalid_structured_result_containers_fail_closed(raw_results: object) -> None:
    output = _output(
        {
            "plan_obj": _plan("structured_fact"),
            "structured_fact_results": raw_results,
            "packet": _packet(),
        }
    )

    assert output["structured_fact_results"] == [None]
    assert output["lanes"]["structured_fact"]["attempted"] is True
    assert output["lanes"]["structured_fact"]["status"] == "failed"
    assert output["status"] == "failed"


def test_structured_success_rejected_during_admission_is_partial_when_evidence_survives() -> None:
    plan = _plan("structured_fact")
    plan["structured_fact_requests"] = [
        {"subquestion": "What was revenue?", "metric_hint": "revenue"},
        {"subquestion": "What was net income?", "metric_hint": "net income"},
    ]
    results = [
        _successful_structured_result(
            subquestion="What was revenue?",
            metric_id="revenue",
            value=391_000_000_000.0,
        ),
        _successful_structured_result(
            subquestion="What was net income?",
            metric_id="net_income",
            value="invalid numeric evidence",
        ),
    ]
    results[1]["tool_result"]["error"] = "raw structured admission detail"
    packet = _append_structured_fact_context_items(
        packet=_packet(),
        structured_fact_results=results,
    )
    state = {
        "plan_obj": plan,
        "structured_fact_results": results,
        "packet": packet,
        "analyst_result": _analyst(),
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet

    assert len(packet.context_items) == 1
    assert _route_after_retrieval_attach_open_issues(state) == "analyst"
    output = _output(state)

    assert output["status"] == "degraded"
    assert output["ok"] is True
    assert output["lanes"]["structured_fact"]["status"] == "partial"
    assert output["lanes"]["structured_fact"]["issues"][0]["code"] == (
        "STRUCTURED_FACT_INVALID_EVIDENCE"
    )
    assert output["degradation"]["affected_lanes"] == ["structured_fact"]
    assert "raw structured admission detail" not in output["degradation"]["notice"]


def test_structured_success_rejected_without_surviving_evidence_is_failed() -> None:
    result = _successful_structured_result(
        subquestion="What was revenue?",
        metric_id="revenue",
        value="invalid numeric evidence",
    )
    packet = _append_structured_fact_context_items(
        packet=_packet(),
        structured_fact_results=[result],
    )
    state = {
        "plan_obj": _plan("structured_fact"),
        "structured_fact_results": [result],
        "packet": packet,
    }

    assert _route_after_retrieval_attach_open_issues(state) == "finalize"
    output = _output(state)

    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"
    assert output["lanes"]["structured_fact"]["status"] == "failed"


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


@pytest.mark.parametrize(
    ("question_classes", "expected_lane_status"),
    [
        (["ambiguous", "ambiguous"], "ambiguous"),
        (["unsupported_derived_metric", "unsupported_ratio"], "unsupported"),
        (["ambiguous", "unsupported_ratio"], "failed"),
        (["ambiguous", "unknown"], "failed"),
        (["unknown", "unknown"], "failed"),
    ],
)
def test_defensive_runtime_capability_rejection_status(
    question_classes: list[str],
    expected_lane_status: str,
) -> None:
    issues = [
        OpenIssue(
            code="STRUCTURED_FACT_CAPABILITY_REJECTED",
            message=f"The runtime boundary rejected request {index}.",
            metadata={
                "outcome": "defensive_rejection",
                "question_class": question_class,
            },
        )
        for index, question_class in enumerate(question_classes)
    ]
    packet = _packet().model_copy(update={"open_issues": issues})
    plan = _plan("structured_fact")
    plan["structured_fact_requests"] = [
        {
            "subquestion": f"Rejected request {index}",
            "metric_hint": "revenue",
        }
        for index, _question_class in enumerate(question_classes)
    ]
    output = _output(
        {
            "plan_obj": plan,
            "structured_fact_results": [
                {
                    "resolver_status": (
                        "ambiguous"
                        if question_class == "ambiguous"
                        else "unresolved"
                    ),
                    "resolver_reason": "Structured-fact capability rejected.",
                    "tool_result": None,
                }
                for question_class in question_classes
            ],
            "packet": packet,
        }
    )

    assert output["lanes"]["structured_fact"]["status"] == expected_lane_status
    assert output["status"] == "failed"
    assert output["failure_stage"] == "structured_fact"


@pytest.mark.parametrize("request_indexes", [[0, 0], [0, None]])
def test_defensive_runtime_rejection_requires_complete_request_index_coverage(
    request_indexes: list[int | None],
) -> None:
    issues = []
    for request_index in request_indexes:
        metadata = {
            "outcome": "defensive_rejection",
            "question_class": "unsupported_ratio",
        }
        if request_index is not None:
            metadata["request_index"] = request_index
        issues.append(
            OpenIssue(
                code="STRUCTURED_FACT_CAPABILITY_REJECTED",
                message="The runtime boundary rejected this request.",
                metadata=metadata,
            )
        )
    packet = _packet().model_copy(update={"open_issues": issues})
    plan = _plan("structured_fact")
    plan["structured_fact_requests"] = [
        {
            "subquestion": f"Rejected request {index}",
            "metric_hint": "revenue",
        }
        for index in range(2)
    ]

    output = _output(
        {
            "plan_obj": plan,
            "structured_fact_results": [
                {
                    "resolver_status": "unresolved",
                    "resolver_reason": "Structured-fact capability rejected.",
                    "tool_result": None,
                }
                for _index in range(2)
            ],
            "packet": packet,
        }
    )

    assert output["lanes"]["structured_fact"]["status"] == "failed"


def test_defensive_runtime_rejection_requires_result_coverage() -> None:
    issue = OpenIssue(
        code="STRUCTURED_FACT_CAPABILITY_REJECTED",
        message="The runtime boundary rejected this request.",
        metadata={
            "outcome": "defensive_rejection",
            "question_class": "unsupported_ratio",
        },
    )
    packet = _packet().model_copy(update={"open_issues": [issue]})
    plan = _plan("structured_fact")
    plan["structured_fact_requests"].append(
        {"subquestion": "Rejected request 2", "metric_hint": "revenue"}
    )

    output = _output(
        {
            "plan_obj": plan,
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

    assert output["lanes"]["structured_fact"]["status"] == "failed"


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
