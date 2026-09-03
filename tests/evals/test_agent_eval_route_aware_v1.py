from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agents.analyst import AnalystRunResult
from agents.contracts import (
    AnalystPacket,
    AnalysisTask,
    ContextItem,
    ContextItemKind,
    FilingMetadata,
    FormType,
    PlannerIntent,
    PlannerRuntimeOutput,
    SourceRef,
    StructuredFactEvidence,
)
from agents.orchestrator.agent_orchestrator import (
    _format_run_output,
    _structured_fact_evidence_from_result,
)
from evals.agent_eval_contracts import AgentEvalExample
from evals.agent_eval_legacy_v0 import build_deterministic_checks as build_legacy_checks
from evals.agent_eval_route_aware_v1 import (
    derive_lane_status,
    deterministic_summary,
    evaluate_run_output,
)
from evals.agent_eval_runner import run_agent_eval
from evals.agent_eval_v1_contracts import AgentEvalExampleV1, AgentExpectedV1


def _structured_request() -> dict:
    return {
        "subquestion": "What was Apple revenue in FY2024?",
        "metric_hint": "revenue",
        "entity_hint": "Apple",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
    }


def _planner_payload(route: str = "kb") -> dict:
    retrieval_needed = route in {"kb", "hybrid"}
    structured_requests = (
        [_structured_request()] if route in {"structured_fact", "hybrid"} else []
    )
    return {
        "status": "completed",
        "retrieval_needed": retrieval_needed,
        "intent": "filing_fact",
        "route": route,
        "structured_fact_requests": structured_requests,
        "metadata": {
            "ticker": "AAPL",
            "company_name": "Apple",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "doc_types": None,
            "fiscal_quarter": None,
        },
        "analysis_task": {
            "task_type": "extract",
            "metric": "revenue",
            "definition_notes": [],
            "requires_calculation": False,
            "expected_artifacts": ["text"],
            "output_format": "short_answer",
        },
        "task_class": "single_target_fact",
        "targets": [
            {
                "target_id": 1,
                "target_key": "AAPL_FY2024",
                "company_name": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "form_type": "10-K",
            }
        ],
        "retrieval_plan": (
            {
                "fanout_mode": "single_target",
                "jobs": [
                    {
                        "applies_to_target_ids": [1],
                        "goal": "Find filing evidence",
                        "job_type": "narrative_extract",
                    }
                ],
            }
            if retrieval_needed
            else None
        ),
        "open_issues": [],
        "original_user_query": "What was Apple revenue in FY2024?",
        "effective_user_query": "What was Apple revenue in FY2024?",
        "clarification_history": [],
        "clarification_request": None,
    }


def _packet(context_count: int = 1) -> AnalystPacket:
    return AnalystPacket(
        plan_id="run-1",
        user_query="What was Apple revenue in FY2024?",
        intent=PlannerIntent.FILING_FACT,
        metadata=FilingMetadata(
            ticker="AAPL",
            company_name="Apple",
            fiscal_year=2024,
            form_type="10-K",
        ),
        analysis_task=AnalysisTask(metric="revenue"),
        context_items=[
            ContextItem(
                context_id=f"ctx_{index}",
                kind=ContextItemKind.TEXT,
                source=SourceRef(ticker="AAPL", fiscal_year=2024),
                payload={"content": f"evidence {index}"},
            )
            for index in range(1, context_count + 1)
        ],
    )


def _analyst_result(packet=None) -> dict:
    packet = packet or _packet()
    item = packet.context_items[0] if packet.context_items else None
    claims = [{"claim_id": "c1", "claim_type": "structured_numeric" if item.structured_fact else "narrative",
               "text": "Apple revenue was reported in the filing.", "context_ids": [item.context_id],
               "metric_id": item.structured_fact.metric_id if item.structured_fact else None}] if item else []
    return AnalystRunResult(
        ok=True,
        status="ok",
        answer="Apple revenue was reported in the filing.",
        intent=PlannerIntent.FILING_FACT,
        metric="revenue",
        used_context_ids=["ctx_1"],
        claims=claims,
        citations=[{"context_id": item.context_id, "source": item.source}] if item else [],
    ).model_dump(mode="json")


def _structured_result(status: str = "ok") -> dict:
    return {
        "subquestion": "What was Apple revenue in FY2024?",
        "metric_hint": "revenue",
        "resolved_ticker": "AAPL",
        "resolved_fiscal_year": 2024,
        "resolved_metric_id": "revenue",
        "resolver_status": "resolved",
        "resolver_reason": "Supported metric.",
        "tool_result": {
            "ok": status == "ok",
            "status": status,
            "metric_id": "revenue",
            "missing_component_groups": [],
        },
    }


def _run_output(
    route: str,
    *,
    kb_ok: bool = True,
    structured_status: str = "ok",
    reported_status: str = "completed",
    include_analyst: bool = True,
    context_count: int = 1,
) -> dict:
    planner = _planner_payload(route)
    retrieval_requested = planner["retrieval_needed"]
    packet = _packet(context_count if retrieval_requested and kb_ok else 0)
    if route in {"structured_fact", "hybrid"} and structured_status == "ok":
        structured_item = ContextItem(
            context_id="ctx_1",
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
        packet.context_items = [
            structured_item,
            *[
                item.model_copy(update={"context_id": f"ctx_{index}"})
                for index, item in enumerate(packet.context_items, start=2)
            ],
        ]
    retrieval = {
        "type": "retrieval",
        "ok": bool(kb_ok and retrieval_requested),
        "attempts": ([{"attempt_index": 1}] if retrieval_requested else []),
        "targets": (
            [
                {
                    "runs": 1,
                    "successful_runs": 1 if kb_ok else 0,
                    "failed_runs": 0 if kb_ok else 1,
                }
            ]
            if retrieval_requested
            else []
        ),
    }
    return {
        "run_id": "run-1",
        "route": route,
        "status": reported_status,
        "ok": reported_status in {"completed", "degraded"},
        "failure_stage": "none" if reported_status in {"completed", "degraded"} else "retrieval",
        "planner": planner,
        "retrieval": retrieval,
        "structured_fact_results": (
            [_structured_result(structured_status)]
            if route in {"structured_fact", "hybrid"}
            else []
        ),
        "analyst": _analyst_result(packet) if include_analyst else None,
        "evaluation_trace": {
            "analyst_packet": packet.model_dump(mode="json")
            if include_analyst
            else None
        },
        "orchestrator_trace": {
            "total_ms": 20,
            "planner_timing_ms": {"planner_start_ms": 2},
            "retrieval_timing_ms": {"retrieve_ms": 3},
            "structured_fact_timing_ms": {"structured_facts_ms": 4},
        },
    }


def _clear_packet_context(output: dict) -> None:
    output["evaluation_trace"]["analyst_packet"]["context_items"] = []


def _add_consistent_degraded_runtime_summaries(output: dict) -> None:
    lanes = {
        "kb": {"requested": True, "attempted": True, "status": "ok"},
        "structured_fact": {
            "requested": True,
            "attempted": True,
            "status": "partial",
        },
    }
    degradation = {
        "active": True,
        "affected_lanes": ["structured_fact"],
        "notice": (
            "Evidence coverage is degraded (structured_fact=partial). Do not "
            "claim coverage from unavailable or incomplete evidence lanes."
        ),
    }
    output["lanes"] = lanes
    output["degradation"] = degradation
    packet = output["evaluation_trace"]["analyst_packet"]
    packet["lanes"] = lanes
    packet["degradation"] = dict(degradation)


def _example(route: str, **expected_overrides) -> AgentEvalExampleV1:
    expected = {
        "intent": "filing_fact",
        "route": route,
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "form_type": "10-K",
        "expect_kb_lane": route in {"kb", "hybrid"},
        "expect_structured_fact_lane": route in {"structured_fact", "hybrid"},
        "expected_reported_status": "completed",
        "expected_effective_status": "completed",
        "expected_failure_stage": "none",
        "expected_analyst_status": "ok",
    }
    expected.update(expected_overrides)
    return AgentEvalExampleV1(
        id="case-1",
        user_query="What was Apple revenue in FY2024?",
        gold_answer="Apple reported revenue.",
        expected=expected,
    )


def test_structured_route_passes_without_kb_retrieval() -> None:
    example = _example(
        "structured_fact",
        expected_structured_metric="revenue",
        expected_structured_status="ok",
    )

    row, errors, _sample = evaluate_run_output(
        example,
        _run_output("structured_fact"),
    )

    assert errors == []
    assert row.lane_status.kb.status == "not_requested"
    assert row.lane_status.structured_fact.status == "ok"
    assert row.deterministic.critical_failures == []
    assert "RETRIEVAL_ROUTING_INVALID" not in row.deterministic.critical_failures


@pytest.mark.parametrize(
    (
        "kb_ok",
        "structured_status",
        "kb_status",
        "structured_lane_status",
        "effective_status",
    ),
    [
        (False, "ok", "failed", "ok", "degraded"),
        (True, "error", "ok", "failed", "degraded"),
        (False, "error", "failed", "failed", "failed"),
        (True, "partial", "ok", "partial", "degraded"),
    ],
)
def test_degradation_truth_table(
    kb_ok: bool,
    structured_status: str,
    kb_status: str,
    structured_lane_status: str,
    effective_status: str,
) -> None:
    output = _run_output(
        "hybrid",
        kb_ok=kb_ok,
        structured_status=structured_status,
    )
    example = _example(
        "hybrid",
        expected_effective_status="degraded",
        allow_degraded=True,
    )

    row, _errors, _sample = evaluate_run_output(example, output)

    assert row.reported_status == "completed"
    assert row.effective_status == effective_status
    assert row.effective_status_source == "evaluator_derived"
    assert row.lane_status.kb.status == kb_status
    assert row.lane_status.structured_fact.status == structured_lane_status
    assert "DEGRADATION_NOT_ALLOWED" not in row.deterministic.critical_failures


def test_unauthorized_degradation_is_critical() -> None:
    example = _example("hybrid")

    row, _errors, _sample = evaluate_run_output(
        example,
        _run_output("hybrid", structured_status="partial"),
    )

    assert "DEGRADATION_NOT_ALLOWED" in row.deterministic.critical_failures
    assert "EFFECTIVE_STATUS_MISMATCH" in row.deterministic.critical_failures


def test_runtime_lanes_are_authoritative_and_shadow_checked() -> None:
    output = _run_output(
        "hybrid",
        structured_status="partial",
        reported_status="degraded",
    )
    output["lanes"] = {
        "kb": {"requested": True, "attempted": True, "status": "ok"},
        "structured_fact": {
            "requested": True,
            "attempted": True,
            "status": "ok",
        },
    }
    output["degradation"] = {
        "active": True,
        "affected_lanes": ["structured_fact"],
        "notice": "Evidence coverage is degraded (structured_fact=partial).",
    }
    example = _example(
        "hybrid",
        expected_reported_status=None,
        expected_effective_status="degraded",
        allow_degraded=True,
    )

    row, _errors, _sample = evaluate_run_output(example, output)

    assert row.lane_status_source == "runtime"
    assert row.runtime_lane_status is not None
    assert row.derived_lane_status.structured_fact.status == "partial"
    assert row.lane_status.structured_fact.status == "ok"
    assert row.lane_status_consistent is False
    assert "LANE_STATUS_INCONSISTENT" in row.deterministic.critical_failures


def test_consistent_runtime_lanes_are_authoritative() -> None:
    output = _run_output("hybrid")
    output["lanes"] = {
        "kb": {"requested": True, "attempted": True, "status": "ok"},
        "structured_fact": {
            "requested": True,
            "attempted": True,
            "status": "ok",
        },
    }
    output["degradation"] = {
        "active": False,
        "affected_lanes": [],
        "notice": "",
    }
    packet = output["evaluation_trace"]["analyst_packet"]
    packet["lanes"] = output["lanes"]
    packet["degradation"] = output["degradation"]

    row, errors, _sample = evaluate_run_output(_example("hybrid"), output)

    assert errors == []
    assert row.lane_status_source == "runtime"
    assert row.effective_status_source == "runtime"
    assert row.lane_status_consistent is True
    assert row.effective_status_consistent is True
    assert row.deterministic.critical_failures == []


def test_degradation_consistency_requires_exact_notice_content() -> None:
    output = _run_output(
        "hybrid",
        structured_status="partial",
        reported_status="degraded",
    )
    _add_consistent_degraded_runtime_summaries(output)
    stale_notice = "Evidence coverage is degraded."
    output["degradation"]["notice"] = stale_notice
    output["evaluation_trace"]["analyst_packet"]["degradation"][
        "notice"
    ] = stale_notice

    row, errors, _sample = evaluate_run_output(
        _example(
            "hybrid",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.degradation_consistent is False
    assert "DEGRADATION_INCONSISTENT" in row.deterministic.critical_failures


def test_degradation_consistency_requires_packet_summary_match() -> None:
    output = _run_output(
        "hybrid",
        structured_status="partial",
        reported_status="degraded",
    )
    _add_consistent_degraded_runtime_summaries(output)
    output["evaluation_trace"]["analyst_packet"]["degradation"][
        "notice"
    ] = "Analyst received a stale degradation notice."

    row, errors, _sample = evaluate_run_output(
        _example(
            "hybrid",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.degradation_consistent is False
    assert "DEGRADATION_INCONSISTENT" in row.deterministic.critical_failures


def test_degradation_consistency_requires_packet_lane_summary_match() -> None:
    output = _run_output(
        "hybrid",
        structured_status="partial",
        reported_status="degraded",
    )
    _add_consistent_degraded_runtime_summaries(output)
    output["evaluation_trace"]["analyst_packet"]["lanes"] = {
        lane_name: dict(lane_summary)
        for lane_name, lane_summary in output["lanes"].items()
    }
    output["evaluation_trace"]["analyst_packet"]["lanes"]["structured_fact"][
        "status"
    ] = "ok"

    row, errors, _sample = evaluate_run_output(
        _example(
            "hybrid",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.lane_status_consistent is True
    assert row.degradation_consistent is False
    assert "DEGRADATION_INCONSISTENT" in row.deterministic.critical_failures


def test_exact_degradation_summary_reaches_analyst_packet() -> None:
    output = _run_output(
        "hybrid",
        structured_status="partial",
        reported_status="degraded",
    )
    _add_consistent_degraded_runtime_summaries(output)

    row, errors, _sample = evaluate_run_output(
        _example(
            "hybrid",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.degradation_consistent is True
    assert "DEGRADATION_INCONSISTENT" not in row.deterministic.critical_failures


def test_packet_evidence_is_ordered_and_limited_to_analyst_visible_items() -> None:
    row, errors, sample = evaluate_run_output(
        _example("kb"),
        _run_output("kb", context_count=7),
    )

    assert errors == []
    assert row.semantic_contexts == [f"evidence {index}" for index in range(1, 6)]
    assert sample is not None
    assert sample["contexts"] == row.semantic_contexts


def test_compact_kb_count_keeps_shadow_aligned_at_context_budget() -> None:
    planner = _planner_payload("hybrid")
    metrics = ["revenue", "net_income", "operating_income"]
    planner["structured_fact_requests"] = [
        {
            **_structured_request(),
            "subquestion": f"What was Apple {metric} in FY2024?",
            "metric_hint": metric,
        }
        for metric in metrics
    ]
    structured_items = [
        ContextItem(
            context_id=f"ctx_structured_{index}",
            kind=ContextItemKind.STRUCTURED_FACT,
            source=SourceRef(ticker="AAPL", fiscal_year=2024),
            structured_fact=StructuredFactEvidence(
                metric_id=metric,
                metric_label=metric.replace("_", " ").title(),
                value=float(index),
                ticker="AAPL",
                fiscal_year=2024,
            ),
        )
        for index, metric in enumerate(metrics, start=1)
    ]
    kb_items = [
        item.model_copy(update={"context_id": f"ctx_kb_{index}"})
        for index, item in enumerate(_packet(context_count=3).context_items, start=1)
    ]
    packet = _packet(context_count=0)
    packet.context_items = [*structured_items, *kb_items]
    structured_results = []
    for metric in metrics:
        result = _structured_result()
        result.update(
            {
                "subquestion": f"What was Apple {metric} in FY2024?",
                "metric_hint": metric,
                "resolved_metric_id": metric,
            }
        )
        result["tool_result"]["metric_id"] = metric
        structured_results.append(result)

    output = _format_run_output(
        run_id="run-1",
        state_snapshot=SimpleNamespace(
            values={
                "plan_obj": planner,
                "planner_dump": planner,
                "retrieval_output": {
                    "ok": True,
                    "top_tables": [
                        {"doc_id": f"kb-candidate-{index}"}
                        for index in range(1, 4)
                    ],
                },
                "structured_fact_results": structured_results,
                "packet": packet,
                "analyst_result": AnalystRunResult.model_validate(_analyst_result()),
                "include_evidence_trace": True,
            },
            interrupts=(),
        ),
    )

    assert output["retrieval"]["retrieved_candidate_count"] == 3
    assert output["lanes"]["kb"]["status"] == "partial"
    row, errors, _sample = evaluate_run_output(
        _example(
            "hybrid",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.derived_lane_status.kb.status == "partial"
    assert row.derived_lane_status.kb.usable_evidence_count == 2
    assert row.lane_status_consistent is True
    assert "LANE_STATUS_INCONSISTENT" not in row.deterministic.critical_failures


def test_typed_structured_evidence_is_visible_and_measured() -> None:
    output = _run_output("hybrid")
    output["structured_fact_results"][0]["tool_result"].update(
        {
            "value": 391_000_000_000.0,
            "accession_number": "0000320193-24-000123",
            "report_date": "2024-09-28",
            "filed_date": "2024-11-01",
            "source_url": "https://www.sec.gov/example",
        }
    )
    packet = _packet(context_count=5)
    structured_item = ContextItem(
        context_id="ctx_1",
        kind=ContextItemKind.STRUCTURED_FACT,
        source=SourceRef(
            ticker="AAPL",
            fiscal_year=2024,
            form_type=FormType.TEN_K,
            accession_no="0000320193-24-000123",
            report_date="2024-09-28",
            filing_date="2024-11-01",
            source_url="https://www.sec.gov/example",
        ),
        structured_fact=StructuredFactEvidence(
            metric_id="revenue",
            metric_label="Revenue",
            value=391_000_000_000.0,
            unit="USD",
            ticker="AAPL",
            fiscal_year=2024,
            form_type=FormType.TEN_K,
            accession_number="0000320193-24-000123",
            report_date="2024-09-28",
            filed_date="2024-11-01",
            source_url="https://www.sec.gov/example",
        ),
    )
    packet.context_items = [
        structured_item,
        *[
            item.model_copy(update={"context_id": f"ctx_{index}"})
            for index, item in enumerate(packet.context_items, start=2)
        ],
    ]
    output["evaluation_trace"]["analyst_packet"] = packet.model_dump(mode="json")

    row, errors, sample = evaluate_run_output(_example("hybrid"), output)
    summary = deterministic_summary([row])

    assert errors == []
    assert row.semantic_context_kinds == [
        "structured_fact",
        "text",
        "text",
        "text",
        "text",
    ]
    assert row.semantic_contexts[0].startswith("structured_fact:\n")
    assert sample is not None
    assert sample["contexts"] == row.semantic_contexts
    assert row.structured_evidence == {
        "successful_execution_results": 1,
        "typed_structured_fact_contexts": 1,
        "analyst_visible_typed_structured_fact_contexts": 1,
        "synthetic_structured_text_contexts": 0,
        "provenance_coverage_counts": {
            "metric_id": 1,
            "numeric_value": 1,
            "accession": 1,
            "report_date": 1,
            "filed_date": 1,
            "source_url": 1,
        },
    }
    assert summary["structured_typed_representation_rate"] == 1.0
    assert summary["structured_analyst_visibility_rate"] == 1.0
    assert summary["structured_source_url_coverage_rate"] == 1.0
    assert summary["structured_synthetic_text_contexts"] == 0.0


def test_conflicting_tool_metric_is_not_counted_as_successful_execution() -> None:
    output = _run_output("structured_fact")
    output["structured_fact_results"][0]["tool_result"].update(
        {
            "metric_id": "total_debt",
            "value": 391_000_000_000.0,
        }
    )

    row, errors, _sample = evaluate_run_output(
        _example("structured_fact"),
        output,
    )

    assert errors == []
    assert row.structured_evidence["successful_execution_results"] == 0


@pytest.mark.parametrize(
    ("raw_groups", "expected_admitted"),
    [
        ([], True),
        (["current_debt"], True),
        ([" current_debt ", "noncurrent_debt"], True),
        ("current_debt", False),
        ({"current_debt"}, False),
        (("current_debt",), False),
        ([123], False),
        ([None], False),
        (["current_debt", 123], False),
        (["   "], False),
        (None, False),
    ],
)
def test_missing_group_admission_matches_runtime_and_evaluator(
    raw_groups,
    expected_admitted,
) -> None:
    output = _run_output("structured_fact")
    result = output["structured_fact_results"][0]
    result["tool_result"].update(
        {
            "value": 391_000_000_000.0,
            "missing_component_groups": raw_groups,
        }
    )
    packet = AnalystPacket.model_validate(
        output["evaluation_trace"]["analyst_packet"]
    )
    evidence, _issue = _structured_fact_evidence_from_result(
        packet=packet,
        result=result,
    )

    row, errors, _sample = evaluate_run_output(
        _example("structured_fact"),
        output,
    )

    assert errors == []
    assert (evidence is not None) is expected_admitted
    assert row.structured_evidence["successful_execution_results"] == int(
        expected_admitted
    )


def test_malformed_missing_groups_do_not_increase_evidence_coverage() -> None:
    output = _run_output("structured_fact")
    _clear_packet_context(output)
    output["structured_fact_results"][0]["tool_result"].update(
        {
            "value": 391_000_000_000.0,
            "missing_component_groups": "current_debt",
        }
    )

    row, errors, _sample = evaluate_run_output(
        _example("structured_fact"),
        output,
    )
    summary = deterministic_summary([row])

    assert errors == []
    assert row.structured_evidence["successful_execution_results"] == 0
    assert row.structured_evidence["typed_structured_fact_contexts"] == 0
    assert row.structured_evidence["synthetic_structured_text_contexts"] == 0
    assert set(
        row.structured_evidence["provenance_coverage_counts"].values()
    ) == {0}
    assert summary["structured_typed_representation_rate"] == 0.0
    assert summary["structured_analyst_visibility_rate"] == 0.0
    assert summary["structured_metric_id_coverage_rate"] == 0.0
    assert summary["structured_source_url_coverage_rate"] == 0.0


@pytest.mark.parametrize(
    "tool_overrides",
    [
        {"ticker": "MSFT"},
        {"fiscal_year": 2023},
        {"form_type": "10-Q"},
        {"components": 1},
        {"components": ["not-a-component"]},
    ],
)
def test_invalid_provenance_is_not_counted_as_successful_execution(
    tool_overrides,
) -> None:
    output = _run_output("structured_fact")
    output["structured_fact_results"][0]["tool_result"].update(
        {
            "metric_id": "revenue",
            "value": 391_000_000_000.0,
            "ticker": "AAPL",
            "fiscal_year": 2024,
            **tool_overrides,
        }
    )

    row, errors, _sample = evaluate_run_output(
        _example("structured_fact"),
        output,
    )

    assert errors == []
    assert row.structured_evidence["successful_execution_results"] == 0


def test_interrupted_run_does_not_require_analyst_or_packet() -> None:
    planner = _planner_payload("kb")
    planner.update(
        {
            "status": "needs_clarification",
            "retrieval_needed": False,
            "retrieval_plan": None,
            "clarification_request": {
                "reason": "Missing year",
                "questions": ["Which fiscal year?"],
            },
        }
    )
    output = _run_output("kb", reported_status="interrupted", include_analyst=False)
    output.update(
        {
            "planner": planner,
            "failure_stage": "interrupted",
            "interrupt": [{"id": "i-1", "value": {"questions": ["Which year?"]}}],
        }
    )
    example = _example(
        "kb",
        expect_kb_lane=False,
        expected_reported_status="interrupted",
        expected_effective_status="interrupted",
        expected_failure_stage="interrupted",
        expected_analyst_status=None,
    )

    row, errors, _sample = evaluate_run_output(example, output)

    assert errors == []
    assert row.deterministic.contract_valid
    assert row.analyst_status is None


def test_structured_status_mismatch_is_critical() -> None:
    example = _example(
        "structured_fact",
        expected_structured_status="ok",
        expected_effective_status="degraded",
        allow_degraded=True,
    )

    row, _errors, _sample = evaluate_run_output(
        example,
        _run_output("structured_fact", structured_status="not_found"),
    )

    assert "STRUCTURED_STATUS_MISMATCH" in row.deterministic.critical_failures


def test_route_mismatch_is_critical() -> None:
    output = _run_output("structured_fact")
    output["route"] = "kb"

    row, errors, _sample = evaluate_run_output(
        _example("structured_fact"),
        output,
    )

    assert any(error["stage"] == "route_trace_contract" for error in errors)
    assert "ROUTE_MISMATCH" in row.deterministic.critical_failures


@pytest.mark.parametrize(
    ("resolver_status", "expected_lane_status"),
    [("ambiguous", "ambiguous"), ("unresolved", "failed"), ("missing_inputs", "failed")],
)
def test_non_resolved_structured_statuses_are_normalized(
    resolver_status: str,
    expected_lane_status: str,
) -> None:
    output = _run_output("structured_fact")
    _clear_packet_context(output)
    output["structured_fact_results"][0].update(
        {
            "resolver_status": resolver_status,
            "resolved_metric_id": None,
            "tool_result": None,
        }
    )

    lanes, metric_ids, statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == expected_lane_status
    assert metric_ids == []
    assert statuses == [resolver_status]


def test_requested_structured_lane_without_results_is_skipped() -> None:
    output = _run_output("structured_fact")
    output["structured_fact_results"] = []

    lanes, _metric_ids, statuses = derive_lane_status(output)

    assert lanes.structured_fact.requested
    assert not lanes.structured_fact.attempted
    assert lanes.structured_fact.status == "skipped"
    assert statuses == []


def test_requested_kb_lane_distinguishes_empty_execution_from_no_execution() -> None:
    output = _run_output(
        "kb",
        kb_ok=False,
        reported_status="failed",
        include_analyst=False,
        context_count=0,
    )
    output["retrieval"].update(
        {
            "attempted": True,
            "attempts": [],
            "targets": [],
            "target": {},
            "retrieved_candidate_count": 0,
        }
    )

    executed_lanes, _metric_ids, _statuses = derive_lane_status(output)
    output["retrieval"]["attempted"] = False
    skipped_lanes, _metric_ids, _statuses = derive_lane_status(output)

    assert executed_lanes.kb.attempted
    assert executed_lanes.kb.status == "failed"
    assert not skipped_lanes.kb.attempted
    assert skipped_lanes.kb.status == "skipped"


def test_explicit_analyst_calculator_and_citation_requirements_are_critical() -> None:
    example = _example(
        "kb",
        expected_analyst_status="insufficient_data",
        must_use_financial_evaluator=True,
        expected_min_citations=1,
    )

    output = _run_output("kb")
    output["analyst"]["citations"] = []
    row, _errors, _sample = evaluate_run_output(example, output)

    assert {
        "ANALYST_STATUS_MISMATCH",
        "COMPUTE_TOOL_NOT_USED",
        "CITATION_MINIMUM_NOT_MET",
    }.issubset(row.deterministic.critical_failures)


def test_expected_degraded_status_requires_explicit_permission() -> None:
    with pytest.raises(ValidationError):
        AgentExpectedV1(expected_effective_status="degraded")


def test_legacy_v0_fixture_keeps_retrieval_routing_failure_and_score_shape() -> None:
    planner = PlannerRuntimeOutput.model_validate(_planner_payload("structured_fact"))
    analyst = AnalystRunResult.model_validate(_analyst_result())
    example = AgentEvalExample(
        id="legacy-1",
        user_query="What was Apple revenue in FY2024?",
        gold_answer="Apple reported revenue.",
        expected={
            "intent": "filing_fact",
            "retrieval_needed": False,
            "must_use_financial_evaluator": False,
            "expected_min_citations": 0,
        },
    )

    checks = build_legacy_checks(
        ex=example,
        planner_obj=planner,
        analyst_obj=analyst,
        contract_valid=True,
    )

    assert checks.critical_failures == ["RETRIEVAL_ROUTING_INVALID"]
    assert checks.score == 1.0
    assert checks.weighted_components == {
        "intent_match": 1.0,
        "retrieval_needed_match": 1.0,
        "tool_use_match": 1.0,
        "citation_match": 1.0,
        "contract_valid": 1.0,
    }


def test_shared_runner_defaults_to_legacy_v0_without_artifact_drift(
    tmp_path,
    monkeypatch,
) -> None:
    example = AgentEvalExample(
        id="legacy-1",
        user_query="What was Apple revenue in FY2024?",
        gold_answer="Apple reported revenue.",
        expected={
            "intent": "filing_fact",
            "retrieval_needed": False,
            "must_use_financial_evaluator": False,
            "expected_min_citations": 0,
        },
    )
    eval_path = tmp_path / "legacy.jsonl"
    eval_path.write_text(example.model_dump_json() + "\n", encoding="utf-8")
    output = _run_output("structured_fact")
    output["retrieval"] = None
    calls = []

    async def fake_run(query, **kwargs):
        calls.append((query, kwargs))
        return output

    monkeypatch.setattr(
        "evals.agent_eval_runner.run_multi_agent_orchestration",
        fake_run,
    )
    monkeypatch.setattr(
        "evals.agent_eval_runner.InteractivePlannerAgent",
        lambda **_kwargs: object(),
    )

    summary, rows, errors = run_agent_eval(
        eval_path=str(eval_path),
        out_dir=str(tmp_path / "out"),
        enable_ragas=False,
    )

    assert errors == []
    assert "mode" not in summary.config
    assert rows[0].deterministic.critical_failures == [
        "RETRIEVAL_ROUTING_INVALID"
    ]
    assert rows[0].deterministic.score == 1.0
    assert calls[0][1]["include_evidence_trace"] is False


def test_evidence_trace_is_opt_in_and_contains_only_packet() -> None:
    packet = _packet()
    values = {
        "plan_obj": _planner_payload("kb"),
        "planner_dump": _planner_payload("kb"),
        "packet": packet,
        "retrieval_output": {},
        "structured_fact_results": [],
    }

    without_trace = _format_run_output(
        run_id="run-1",
        state_snapshot=SimpleNamespace(values=values, interrupts=()),
    )
    with_trace = _format_run_output(
        run_id="run-1",
        state_snapshot=SimpleNamespace(
            values={**values, "include_evidence_trace": True},
            interrupts=(),
        ),
    )

    assert "evaluation_trace" not in without_trace
    assert set(with_trace["evaluation_trace"]) == {"analyst_packet"}
    assert with_trace["evaluation_trace"]["analyst_packet"] == packet.model_dump(
        mode="json"
    )


def test_lane_derivation_reports_partial_multi_target_kb() -> None:
    output = _run_output("kb")
    output["retrieval"]["targets"] = [
        {"runs": 1, "successful_runs": 1, "failed_runs": 0},
        {"runs": 1, "successful_runs": 0, "failed_runs": 1},
    ]

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.kb.status == "partial"


@pytest.mark.parametrize(
    "issue_code",
    ["TABLE_HYDRATION_FAILED", "TABLE_MARKDOWN_EMPTY"],
)
def test_shadow_kb_lane_detects_partial_admission_loss(issue_code: str) -> None:
    output = _run_output("kb", reported_status="degraded")
    output["retrieval"]["targets"][0]["tables_retrieved"] = 2
    output["retrieval"]["attempts"][0]["results"] = [
        {"doc_id": "missing::table::0", "score": 1.0},
        {"doc_id": "usable::text::1", "score": 0.9},
    ]
    hydration_issue = {
        "code": issue_code,
        "message": "One retrieved table could not be admitted.",
        "severity": "warning",
    }
    output["open_issues"] = [hydration_issue]
    output["evaluation_trace"]["analyst_packet"]["open_issues"] = [
        hydration_issue
    ]

    row, errors, _sample = evaluate_run_output(
        _example(
            "kb",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.derived_lane_status.kb.status == "partial"
    assert row.derived_lane_status.kb.usable
    assert row.derived_lane_status.kb.usable_evidence_count == 1
    assert row.effective_status == "degraded"
    assert row.analyst_status == "ok"

    _clear_packet_context(output)
    output["retrieval"]["targets"][0]["tables_retrieved"] = 1
    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.kb.status == "failed"
    assert not lanes.kb.usable


def test_shadow_kb_lane_counts_empty_text_loss_after_packet_cap_backfill() -> None:
    output = _run_output("kb", reported_status="degraded", context_count=3)
    output["retrieval"]["retrieved_candidate_count"] = 4
    output["retrieval"]["attempts"][0]["results"] = [
        {"doc_id": f"AAPL_10-K_2024::text::{index}", "score": 1.0}
        for index in range(4)
    ]
    empty_text_issue = {
        "code": "EMPTY_TEXT_CONTEXT",
        "message": "One retrieved text candidate could not be admitted.",
        "severity": "warning",
    }
    output["open_issues"] = [empty_text_issue]
    output["evaluation_trace"]["analyst_packet"]["open_issues"] = [
        empty_text_issue
    ]

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.kb.status == "partial"
    assert lanes.kb.usable
    assert lanes.kb.usable_evidence_count == 3


def test_shadow_kb_lane_counts_unsupported_candidate_loss_after_packet_cap_backfill() -> None:
    output = _run_output("kb", reported_status="degraded", context_count=3)
    output["retrieval"]["retrieved_candidate_count"] = 4
    output["retrieval"]["attempts"][0]["results"] = [
        {"doc_id": "AAPL_10-K_2024::chart::0", "score": 1.0},
        *[
            {
                "doc_id": f"AAPL_10-K_2024::text::{index}",
                "score": 1.0 - index / 10,
            }
            for index in range(1, 4)
        ],
    ]
    unsupported_issue = {
        "code": "RETRIEVAL_CANDIDATE_UNSUPPORTED",
        "message": "One retrieved candidate had an unrecognized evidence type.",
        "severity": "warning",
    }
    output["open_issues"] = [unsupported_issue]
    output["evaluation_trace"]["analyst_packet"]["open_issues"] = [
        unsupported_issue
    ]

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.kb.status == "partial"
    assert lanes.kb.usable
    assert lanes.kb.usable_evidence_count == 3


def test_shadow_structured_lane_detects_tool_ok_admission_rejection() -> None:
    output = _run_output("structured_fact", reported_status="degraded")
    output["planner"]["structured_fact_requests"].append(
        {
            **_structured_request(),
            "subquestion": "What was Apple net income in FY2024?",
            "metric_hint": "net_income",
        }
    )
    output["structured_fact_results"][0]["tool_result"]["value"] = (
        391_000_000_000.0
    )
    rejected_result = _structured_result()
    rejected_result.update(
        {
            "subquestion": "What was Apple net income in FY2024?",
            "metric_hint": "net_income",
            "resolved_metric_id": "net_income",
        }
    )
    rejected_result["tool_result"].update(
        {"metric_id": "net_income", "value": "not-numeric"}
    )
    output["structured_fact_results"].append(rejected_result)
    invalid_issue = {
        "code": "STRUCTURED_FACT_INVALID_EVIDENCE",
        "message": "One successful result failed PR4 admission.",
        "severity": "error",
    }
    output["open_issues"] = [invalid_issue]
    output["evaluation_trace"]["analyst_packet"]["open_issues"] = [invalid_issue]

    row, errors, _sample = evaluate_run_output(
        _example(
            "structured_fact",
            expected_reported_status="degraded",
            expected_effective_status="degraded",
            allow_degraded=True,
        ),
        output,
    )

    assert errors == []
    assert row.structured_statuses == ["ok", "ok"]
    assert row.derived_lane_status.structured_fact.status == "partial"
    assert row.derived_lane_status.structured_fact.usable
    assert row.derived_lane_status.structured_fact.usable_evidence_count == 1
    assert row.effective_status == "degraded"
    assert row.analyst_status == "ok"

    _clear_packet_context(output)
    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == "failed"
    assert not lanes.structured_fact.usable


@pytest.mark.parametrize(
    ("operation_statuses", "expected_lane_status"),
    [
        (["partial", "partial"], "partial"),
        (["partial", "error"], "failed"),
        (["partial", "ambiguous"], "failed"),
        (["partial", "unsupported_metric"], "failed"),
    ],
)
def test_shadow_structured_lane_requires_wholly_partial_unusable_outcomes(
    operation_statuses: list[str],
    expected_lane_status: str,
) -> None:
    output = _run_output(
        "structured_fact",
        structured_status="partial",
        reported_status="failed",
    )
    output["planner"]["structured_fact_requests"] = [
        {
            **_structured_request(),
            "subquestion": f"What was metric {index}?",
            "metric_hint": f"metric_{index}",
        }
        for index, _status in enumerate(operation_statuses)
    ]
    results = []
    issue_codes = {
        "partial": "STRUCTURED_FACT_PARTIAL",
        "error": "STRUCTURED_FACT_ERROR",
        "ambiguous": "STRUCTURED_FACT_AMBIGUOUS",
        "unsupported_metric": "STRUCTURED_FACT_UNSUPPORTED",
    }
    issues = []
    for index, status in enumerate(operation_statuses):
        result = _structured_result(status)
        result.update(
            {
                "subquestion": f"What was metric {index}?",
                "metric_hint": f"metric_{index}",
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
        issues.append(
            {
                "code": issue_codes[status],
                "message": "Structured fact evidence is unavailable.",
                "severity": "error" if status == "error" else "warning",
            }
        )

    output["structured_fact_results"] = results
    output["analyst"] = None
    output["failure_stage"] = "structured_fact"
    output["open_issues"] = issues
    packet = output["evaluation_trace"]["analyst_packet"]
    packet["context_items"] = []
    packet["open_issues"] = issues
    lanes = {
        "kb": {
            "requested": False,
            "attempted": False,
            "status": "not_requested",
        },
        "structured_fact": {
            "requested": True,
            "attempted": True,
            "status": expected_lane_status,
            "issues": issues,
        },
    }
    degradation = {
        "active": True,
        "affected_lanes": ["structured_fact"],
        "notice": (
            "Evidence coverage is degraded "
            f"(structured_fact={expected_lane_status}). Do not claim coverage "
            "from unavailable or incomplete evidence lanes."
        ),
    }
    output["lanes"] = lanes
    output["degradation"] = degradation
    packet["lanes"] = lanes
    packet["degradation"] = degradation

    row, errors, _sample = evaluate_run_output(
        _example(
            "structured_fact",
            expected_reported_status="failed",
            expected_effective_status="failed",
            expected_failure_stage="structured_fact",
            expected_analyst_status=None,
        ),
        output,
    )

    assert errors == []
    assert row.structured_statuses == operation_statuses
    assert row.derived_lane_status.structured_fact.status == expected_lane_status
    assert not row.derived_lane_status.structured_fact.usable
    assert row.lane_status_consistent
    assert row.degradation_consistent
    assert row.deterministic.critical_failures == []


@pytest.mark.parametrize(
    "operation_status",
    ["partial", "ambiguous", "unsupported_metric"],
)
def test_shadow_structured_lane_requires_complete_result_coverage_without_evidence(
    operation_status: str,
) -> None:
    output = _run_output(
        "structured_fact",
        structured_status=operation_status,
        reported_status="failed",
    )
    output["planner"]["structured_fact_requests"].append(
        {
            **_structured_request(),
            "subquestion": "What was Apple net income in FY2024?",
            "metric_hint": "net_income",
        }
    )
    _clear_packet_context(output)

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == "failed"
    assert not lanes.structured_fact.usable


def test_shadow_structured_lane_marks_incomplete_admitted_results_partial() -> None:
    output = _run_output("structured_fact", reported_status="degraded")
    output["planner"]["structured_fact_requests"].append(
        {
            **_structured_request(),
            "subquestion": "What was Apple net income in FY2024?",
            "metric_hint": "net_income",
        }
    )

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == "partial"
    assert lanes.structured_fact.usable
    assert lanes.structured_fact.usable_evidence_count == 1


def test_shadow_structured_lane_fails_closed_for_malformed_results() -> None:
    usable_output = _run_output("structured_fact", reported_status="degraded")
    usable_output["structured_fact_results"].append("malformed")
    unusable_output = _run_output(
        "structured_fact",
        reported_status="failed",
        include_analyst=False,
    )
    unusable_output["structured_fact_results"] = ["malformed"]

    usable_lanes, _metrics, _statuses = derive_lane_status(usable_output)
    unusable_lanes, _metrics, _statuses = derive_lane_status(unusable_output)

    assert usable_lanes.structured_fact.status == "partial"
    assert usable_lanes.structured_fact.usable
    assert unusable_lanes.structured_fact.attempted
    assert unusable_lanes.structured_fact.status == "failed"
    assert not unusable_lanes.structured_fact.usable


@pytest.mark.parametrize(
    "raw_results",
    [42, "malformed", {"tool_result": {"status": "ok"}}],
)
def test_shadow_structured_lane_fails_closed_for_invalid_result_containers(
    raw_results: object,
) -> None:
    output = _run_output(
        "structured_fact",
        reported_status="failed",
        include_analyst=False,
    )
    output["structured_fact_results"] = raw_results

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.structured_fact.attempted
    assert lanes.structured_fact.status == "failed"
    assert not lanes.structured_fact.usable


@pytest.mark.parametrize(
    ("question_classes", "expected_lane_status"),
    [
        (["ambiguous", "ambiguous"], "ambiguous"),
        (
            [
                "unsupported_derived_metric",
                "unsupported_ratio",
                "unsupported_per_share",
                "unsupported_comparison",
            ],
            "unsupported",
        ),
        (["ambiguous", "unsupported_ratio"], "failed"),
        (["ambiguous", "unknown"], "failed"),
        (["unknown", "unknown"], "failed"),
    ],
)
def test_shadow_structured_lane_classifies_defensive_capability_rejections(
    question_classes: list[str],
    expected_lane_status: str,
) -> None:
    output = _run_output("structured_fact", reported_status="failed")
    output["planner"]["structured_fact_requests"] = [
        {
            **_structured_request(),
            "subquestion": f"Rejected request {index}",
        }
        for index, _question_class in enumerate(question_classes)
    ]
    output["structured_fact_results"] = [
        {
            "subquestion": f"Rejected request {index}",
            "metric_hint": "revenue",
            "resolved_ticker": "AAPL",
            "resolved_fiscal_year": 2024,
            "resolved_metric_id": None,
            "resolver_status": (
                "ambiguous" if question_class == "ambiguous" else "unresolved"
            ),
            "resolver_reason": "Structured-fact capability rejected.",
            "tool_result": None,
        }
        for index, question_class in enumerate(question_classes)
    ]
    issues = [
        {
            "code": "STRUCTURED_FACT_CAPABILITY_REJECTED",
            "message": "The runtime boundary rejected this request.",
            "severity": "warning",
            "metadata": {
                "request_index": index,
                "question_class": question_class,
                "metric_hint": "revenue",
                "subquestion": f"Rejected request {index}",
                "original_route": "structured_fact",
                "effective_route": None,
                "outcome": "defensive_rejection",
                "reason": "Unsupported at the runtime boundary.",
                "candidate_metric_ids": [],
            },
        }
        for index, question_class in enumerate(question_classes)
    ]
    output["open_issues"] = issues
    packet = output["evaluation_trace"]["analyst_packet"]
    packet["context_items"] = []
    packet["open_issues"] = issues
    output["analyst"] = None
    output["failure_stage"] = "structured_fact"
    runtime_lanes = {
        "kb": {
            "requested": False,
            "attempted": False,
            "status": "not_requested",
        },
        "structured_fact": {
            "requested": True,
            "attempted": True,
            "status": expected_lane_status,
            "issues": issues,
        },
    }
    degradation = {
        "active": True,
        "affected_lanes": ["structured_fact"],
        "notice": (
            "Evidence coverage is degraded "
            f"(structured_fact={expected_lane_status}). Do not claim coverage "
            "from unavailable or incomplete evidence lanes."
        ),
    }
    output["lanes"] = runtime_lanes
    output["degradation"] = degradation
    packet["lanes"] = runtime_lanes
    packet["degradation"] = degradation

    lanes, _metrics, statuses = derive_lane_status(output)

    assert statuses == [
        "ambiguous" if question_class == "ambiguous" else "unresolved"
        for question_class in question_classes
    ]
    assert lanes.structured_fact.status == expected_lane_status
    assert not lanes.structured_fact.usable

    row, errors, _sample = evaluate_run_output(
        _example(
            "structured_fact",
            expected_reported_status="failed",
            expected_effective_status="failed",
            expected_failure_stage="structured_fact",
            expected_analyst_status=None,
        ),
        output,
    )

    assert errors == []
    assert row.lane_status_consistent
    assert row.degradation_consistent
    assert row.deterministic.critical_failures == []


@pytest.mark.parametrize(
    ("request_indexes", "expected_lane_status"),
    [
        ([0, 0], "failed"),
        ([0, None], "failed"),
        ([None, None], "unsupported"),
    ],
)
def test_shadow_structured_lane_uses_defensive_request_index_coverage(
    request_indexes: list[int | None],
    expected_lane_status: str,
) -> None:
    output = _run_output("structured_fact", reported_status="failed")
    output["planner"]["structured_fact_requests"] = [
        {
            **_structured_request(),
            "subquestion": f"Rejected request {index}",
        }
        for index in range(2)
    ]
    output["structured_fact_results"] = [
        {
            "subquestion": f"Rejected request {index}",
            "metric_hint": "revenue",
            "resolved_metric_id": None,
            "resolver_status": "unresolved",
            "resolver_reason": "Structured-fact capability rejected.",
            "tool_result": None,
        }
        for index in range(2)
    ]
    issues = []
    for request_index in request_indexes:
        metadata = {
            "outcome": "defensive_rejection",
            "question_class": "unsupported_ratio",
        }
        if request_index is not None:
            metadata["request_index"] = request_index
        issues.append(
            {
                "code": "STRUCTURED_FACT_CAPABILITY_REJECTED",
                "message": "The runtime boundary rejected this request.",
                "severity": "warning",
                "metadata": metadata,
            }
        )
    output["open_issues"] = issues
    packet = output["evaluation_trace"]["analyst_packet"]
    packet["context_items"] = []
    packet["open_issues"] = issues

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == expected_lane_status
    assert not lanes.structured_fact.usable


def test_shadow_structured_lane_requires_defensive_result_coverage() -> None:
    output = _run_output("structured_fact", reported_status="failed")
    output["planner"]["structured_fact_requests"].append(
        {
            **_structured_request(),
            "subquestion": "Rejected request 2",
        }
    )
    output["structured_fact_results"] = [
        {
            "subquestion": "Rejected request 1",
            "metric_hint": "revenue",
            "resolved_ticker": "AAPL",
            "resolved_fiscal_year": 2024,
            "resolved_metric_id": None,
            "resolver_status": "unresolved",
            "resolver_reason": "Structured-fact capability rejected.",
            "tool_result": None,
        }
    ]
    issue = {
        "code": "STRUCTURED_FACT_CAPABILITY_REJECTED",
        "message": "The runtime boundary rejected this request.",
        "severity": "warning",
        "metadata": {
            "question_class": "unsupported_ratio",
            "outcome": "defensive_rejection",
        },
    }
    output["open_issues"] = [issue]
    packet = output["evaluation_trace"]["analyst_packet"]
    packet["context_items"] = []
    packet["open_issues"] = [issue]

    lanes, _metrics, _statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == "failed"
    assert not lanes.structured_fact.usable


def test_shared_runner_selects_v1_and_requests_evidence_trace(
    tmp_path,
    monkeypatch,
) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        _example(
            "structured_fact",
            expected_structured_metric="revenue",
            expected_structured_status="ok",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    calls = []

    async def fake_run(query, **kwargs):
        calls.append((query, kwargs))
        return _run_output("structured_fact")

    monkeypatch.setattr(
        "evals.agent_eval_runner.run_multi_agent_orchestration",
        fake_run,
    )
    monkeypatch.setattr(
        "evals.agent_eval_runner.InteractivePlannerAgent",
        lambda **_kwargs: object(),
    )

    summary, rows, errors = run_agent_eval(
        eval_path=str(eval_path),
        out_dir=str(tmp_path / "out"),
        mode="route_aware_v1",
    )

    assert errors == []
    assert summary.gate.overall_pass
    assert summary.config["mode"] == "route_aware_v1"
    assert summary.config["enable_ragas"] is False
    assert rows[0].deterministic.score == 1.0
    assert calls[0][1]["include_evidence_trace"] is True
