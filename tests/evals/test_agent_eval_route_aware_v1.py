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
from agents.orchestrator.agent_orchestrator import _format_run_output
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


def _analyst_result() -> dict:
    return AnalystRunResult(
        ok=True,
        status="ok",
        answer="Apple revenue was reported in the filing.",
        intent=PlannerIntent.FILING_FACT,
        metric="revenue",
        used_context_ids=["ctx_1"],
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
        "ok": reported_status == "completed",
        "failure_stage": "none" if reported_status in {"completed", "degraded"} else "retrieval",
        "planner": planner,
        "retrieval": retrieval,
        "structured_fact_results": (
            [_structured_result(structured_status)]
            if route in {"structured_fact", "hybrid"}
            else []
        ),
        "analyst": _analyst_result() if include_analyst else None,
        "evaluation_trace": {
            "analyst_packet": _packet(context_count).model_dump(mode="json")
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
    ("kb_ok", "structured_status", "kb_status", "structured_lane_status"),
    [
        (False, "ok", "failed", "ok"),
        (True, "error", "ok", "error"),
        (False, "error", "failed", "error"),
        (True, "partial", "ok", "partial"),
    ],
)
def test_degradation_truth_table(
    kb_ok: bool,
    structured_status: str,
    kb_status: str,
    structured_lane_status: str,
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
    assert row.effective_status == "degraded"
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

    row, errors, _sample = evaluate_run_output(_example("hybrid"), output)

    assert errors == []
    assert row.lane_status_source == "runtime"
    assert row.effective_status_source == "runtime"
    assert row.lane_status_consistent is True
    assert row.effective_status_consistent is True
    assert row.deterministic.critical_failures == []


def test_packet_evidence_is_ordered_and_limited_to_analyst_visible_items() -> None:
    row, errors, sample = evaluate_run_output(
        _example("kb"),
        _run_output("kb", context_count=7),
    )

    assert errors == []
    assert row.semantic_contexts == [f"evidence {index}" for index in range(1, 6)]
    assert sample is not None
    assert sample["contexts"] == row.semantic_contexts


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


@pytest.mark.parametrize("resolver_status", ["ambiguous", "unresolved", "missing_inputs"])
def test_non_resolved_structured_statuses_are_preserved(resolver_status: str) -> None:
    output = _run_output("structured_fact")
    output["structured_fact_results"][0].update(
        {
            "resolver_status": resolver_status,
            "resolved_metric_id": None,
            "tool_result": None,
        }
    )

    lanes, metric_ids, statuses = derive_lane_status(output)

    assert lanes.structured_fact.status == resolver_status
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


def test_explicit_analyst_calculator_and_citation_requirements_are_critical() -> None:
    example = _example(
        "kb",
        expected_analyst_status="insufficient_data",
        must_use_financial_evaluator=True,
        expected_min_citations=1,
    )

    row, _errors, _sample = evaluate_run_output(example, _run_output("kb"))

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
