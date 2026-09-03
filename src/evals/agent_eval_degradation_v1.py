from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.analyst import AnalystRunResult, build_packet_from_retrieval_output
from agents.contracts import (
    AnalysisTask,
    AnalystPacket,
    ContextItem,
    ContextItemKind,
    FilingMetadata,
    OpenIssue,
    PlannerRuntimeOutput,
    PlannerIntent,
    SourceRef,
    StructuredFactEvidence,
)
from agents.orchestrator.agent_orchestrator import (
    _append_structured_fact_context_items,
    _format_run_output,
    _packet_with_evidence_status,
)
from evals.agent_eval_route_aware_v1 import (
    _derive_effective_status,
    _derive_failure_stage_shadow,
    derive_lane_status,
)


LaneFixture = Literal[
    "not_requested",
    "ok",
    "partial_usable",
    "kb_admission_partial",
    "structured_mixed_admission",
    "structured_mixed_unusable",
    "partial",
    "failed",
    "skipped",
]
AnalystFixture = Literal["ok", "failed", "skipped"]


class DegradationExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_status: str
    kb_usable: bool
    structured_fact_status: str
    structured_fact_usable: bool
    status: Literal["completed", "degraded", "failed"]
    failure_stage: Literal["none", "planner", "retrieval", "structured_fact", "analyst"]
    degradation_active: bool
    affected_lanes: List[Literal["kb", "structured_fact"]] = Field(default_factory=list)
    analyst_ran: bool


class DegradationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    intent: PlannerIntent
    route: Literal["kb", "structured_fact", "hybrid"]
    kb: LaneFixture
    structured_fact: LaneFixture
    analyst: AnalystFixture
    expected: DegradationExpected


def load_degradation_cases(path: str | Path) -> List[DegradationCase]:
    cases: List[DegradationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                cases.append(DegradationCase.model_validate(json.loads(raw)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid PR5 degradation row at line {line_no}: {exc}"
                ) from exc
    return cases


def _planner(case: DegradationCase) -> dict:
    kb_requested = case.kb != "not_requested"
    structured_requested = case.structured_fact != "not_requested"
    structured_fact_requests = [
        {
            "subquestion": "What was Apple revenue in FY2024?",
            "metric_hint": "revenue",
            "entity_hint": "Apple",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
        }
    ]
    if case.structured_fact in {
        "structured_mixed_admission",
        "structured_mixed_unusable",
    }:
        structured_fact_requests.append(
            {
                "subquestion": "What was Apple net income in FY2024?",
                "metric_hint": "net_income",
                "entity_hint": "Apple",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
            }
        )
    return {
        "status": "completed",
        "retrieval_needed": kb_requested,
        "intent": case.intent.value,
        "route": case.route,
        "structured_fact_requests": structured_fact_requests if structured_requested else [],
        "metadata": {
            "ticker": "AAPL" if case.intent != PlannerIntent.DEFINITION else None,
            "company_name": "Apple" if case.intent != PlannerIntent.DEFINITION else None,
            "fiscal_year": 2024 if case.intent != PlannerIntent.DEFINITION else None,
            "form_type": "10-K" if case.intent != PlannerIntent.DEFINITION else None,
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
        "task_class": "single_target_fact" if case.intent != PlannerIntent.DEFINITION else "other",
        "targets": (
            [
                {
                    "target_id": 1,
                    "target_key": "AAPL_FY2024",
                    "company_name": "Apple",
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "form_type": "10-K",
                }
            ]
            if case.intent != PlannerIntent.DEFINITION
            else []
        ),
        "retrieval_plan": (
            {
                "fanout_mode": "single_target",
                "jobs": [
                    {
                        "applies_to_target_ids": [1],
                        "goal": "Find filing revenue evidence.",
                        "job_type": "metric_extract",
                    }
                ],
            }
            if kb_requested
            else None
        ),
        "open_issues": [],
        "original_user_query": "What was Apple revenue in FY2024?",
        "effective_user_query": "What was Apple revenue in FY2024?",
        "clarification_history": [],
        "clarification_request": None,
    }


def _kb_context() -> ContextItem:
    return ContextItem(
        context_id="ctx_kb",
        kind=ContextItemKind.TEXT,
        source=SourceRef(ticker="AAPL", fiscal_year=2024),
        payload={"content": "Apple reported revenue in its FY2024 filing."},
    )


def _structured_context() -> ContextItem:
    return ContextItem(
        context_id="ctx_structured",
        kind=ContextItemKind.STRUCTURED_FACT,
        source=SourceRef(ticker="AAPL", fiscal_year=2024),
        structured_fact=StructuredFactEvidence(
            metric_id="revenue",
            metric_label="Revenue",
            value=391_000_000_000.0,
            unit="USD",
            ticker="AAPL",
            fiscal_year=2024,
        ),
    )


def _retrieval_fixture(outcome: LaneFixture) -> Optional[dict]:
    if outcome in {"not_requested", "skipped"}:
        return None
    if outcome == "kb_admission_partial":
        candidates = [
            {
                "doc_id": "pr5-missing-table",
                "doc_type": "table",
                "table": {
                    "payload": {
                        "doc_id": "pr5-missing-table",
                        "doc_type": "table",
                    }
                },
            },
            {
                "doc_id": "pr5-usable-text",
                "doc_type": "text",
                "table": {
                    "payload": {
                        "doc_id": "pr5-usable-text",
                        "doc_type": "text",
                        "content": "Apple reported FY2024 revenue in its Form 10-K.",
                        "ticker": "AAPL",
                        "fiscal_year": 2024,
                        "form_type": "10-K",
                    }
                },
            },
        ]
        target = {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"}
        return {
            "ok": True,
            "top_tables": candidates,
            "partial_failures": [],
            "metadata_used": target,
            "targets": [{"target_id": 1, **target}],
            "trace": {
                "runs": [
                    {
                        "target": target,
                        "attempts": [
                            {
                                "attempt_index": 1,
                                "request": {
                                    "queries": ["Apple FY2024 revenue"],
                                    "top_k": 2,
                                },
                                "retrieval": {"ok": True, "top_tables": candidates},
                            }
                        ],
                        "final_retrieval": {"ok": True, "top_tables": candidates},
                        "review_feedback": {"action": "accept"},
                    }
                ]
            },
            "error": None,
        }
    usable = outcome in {"ok", "partial_usable"}
    return {
        "ok": usable,
        "top_tables": [],
        "partial_failures": (
            [{"error": "fault-injected KB operation failure"}]
            if outcome == "partial_usable"
            else []
        ),
        "metadata_used": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
        "targets": [
            {
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "form_type": "10-K",
                "runs": 1,
                "successful_runs": 1 if usable else 0,
                "failed_runs": 0 if usable else 1,
            }
        ],
        "error": None if usable else "fault-injected KB failure",
    }


def _structured_fixture(outcome: LaneFixture) -> List[dict]:
    if outcome in {"not_requested", "skipped"}:
        return []
    if outcome == "structured_mixed_admission":
        return [
            {
                "subquestion": "What was Apple revenue in FY2024?",
                "metric_hint": "revenue",
                "resolved_ticker": "AAPL",
                "resolved_fiscal_year": 2024,
                "resolved_metric_id": "revenue",
                "resolver_status": "resolved",
                "resolver_reason": "Supported metric.",
                "tool_result": {
                    "ok": True,
                    "status": "ok",
                    "metric_id": "revenue",
                    "value": 391_000_000_000.0,
                    "missing_component_groups": [],
                    "error": None,
                },
            },
            {
                "subquestion": "What was Apple net income in FY2024?",
                "metric_hint": "net_income",
                "resolved_ticker": "AAPL",
                "resolved_fiscal_year": 2024,
                "resolved_metric_id": "net_income",
                "resolver_status": "resolved",
                "resolver_reason": "Supported metric.",
                "tool_result": {
                    "ok": True,
                    "status": "ok",
                    "metric_id": "net_income",
                    "value": "invalid numeric payload",
                    "missing_component_groups": [],
                    "error": None,
                },
            },
        ]
    if outcome == "structured_mixed_unusable":
        return [
            {
                "subquestion": "What was Apple revenue in FY2024?",
                "metric_hint": "revenue",
                "resolved_ticker": "AAPL",
                "resolved_fiscal_year": 2024,
                "resolved_metric_id": "revenue",
                "resolver_status": "resolved",
                "resolver_reason": "Supported metric.",
                "tool_result": {
                    "ok": False,
                    "status": "partial",
                    "metric_id": "revenue",
                    "value": None,
                    "missing_component_groups": [
                        {"group": ["revenue"], "reason": "No matching facts."}
                    ],
                    "error": "Revenue evidence was incomplete.",
                },
            },
            {
                "subquestion": "What was Apple net income in FY2024?",
                "metric_hint": "net_income",
                "resolved_ticker": "AAPL",
                "resolved_fiscal_year": 2024,
                "resolved_metric_id": "net_income",
                "resolver_status": "resolved",
                "resolver_reason": "Supported metric.",
                "tool_result": {
                    "ok": False,
                    "status": "error",
                    "metric_id": "net_income",
                    "value": None,
                    "missing_component_groups": [],
                    "error": "Structured fact execution failed.",
                },
            },
        ]
    status = "ok" if outcome == "ok" else "partial" if outcome == "partial" else "error"
    return [
        {
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
                "value": 391_000_000_000.0 if status == "ok" else None,
                "missing_component_groups": [],
                "error": None if status == "ok" else "fault-injected structured failure",
            },
        }
    ]


def run_fault_injected_case(case: DegradationCase) -> Dict[str, Any]:
    plan = _planner(case)
    retrieval_output = _retrieval_fixture(case.kb)
    structured_results = _structured_fixture(case.structured_fact)
    items: List[ContextItem] = []
    if case.kb in {"ok", "partial_usable"}:
        items.append(_kb_context())
    if case.structured_fact == "ok":
        items.insert(0, _structured_context())
    packet_issues: List[OpenIssue] = []
    if case.kb == "partial_usable":
        packet_issues.append(
            OpenIssue(
                code="RETRIEVAL_PARTIAL_FAILURE",
                message="A fault-injected retrieval operation failed.",
            )
        )
    if case.kb == "kb_admission_partial":
        packet = build_packet_from_retrieval_output(
            plan_id=case.id,
            user_query=plan["original_user_query"],
            intent=case.intent,
            analysis_task=plan["analysis_task"],
            retrieval_output=retrieval_output,
        )
    else:
        packet = AnalystPacket(
            plan_id=case.id,
            user_query=plan["original_user_query"],
            intent=case.intent,
            metadata=FilingMetadata.model_validate(plan["metadata"]),
            analysis_task=AnalysisTask.model_validate(plan["analysis_task"]),
            context_items=items,
            open_issues=packet_issues,
        )
    if case.structured_fact in {
        "structured_mixed_admission",
        "structured_mixed_unusable",
    }:
        packet = _append_structured_fact_context_items(
            packet=packet,
            structured_fact_results=structured_results,
        )
    state: Dict[str, Any] = {
        "plan_obj": plan,
        "planner_dump": plan,
        "retrieval_state": {"targets": plan["targets"]} if retrieval_output is not None else {},
        "retrieval_output": retrieval_output,
        "retrieval_skipped_reason": "MISSING_RETRIEVAL_PLAN" if case.kb == "skipped" else "",
        "structured_fact_results": structured_results,
        "packet": packet,
        "include_evidence_trace": True,
    }
    packet = _packet_with_evidence_status(state=state, packet=packet)
    state["packet"] = packet
    if case.analyst != "skipped":
        state["analyst_result"] = AnalystRunResult(
            ok=case.analyst == "ok",
            status="ok" if case.analyst == "ok" else "error",
            answer="Apple reported revenue." if case.analyst == "ok" else "Analysis failed.",
            intent=case.intent,
            metric="revenue",
            error=None if case.analyst == "ok" else "fault-injected analyst failure",
        )
    return _format_run_output(
        run_id=case.id,
        state_snapshot=SimpleNamespace(values=state, interrupts=()),
    )


def evaluate_degradation_case(case: DegradationCase) -> Dict[str, Any]:
    output = run_fault_injected_case(case)
    shadow_lanes, _metric_ids, _statuses = derive_lane_status(output)
    planner = PlannerRuntimeOutput.model_validate(output["planner"])
    analyst = (
        AnalystRunResult.model_validate(output["analyst"])
        if output.get("analyst") is not None
        else None
    )
    shadow_status = _derive_effective_status(
        output.get("status"),
        shadow_lanes,
        planner,
        analyst,
    )
    shadow_failure_stage = _derive_failure_stage_shadow(
        reported_status=output.get("status"),
        planner=planner,
        analyst=analyst,
        lanes=shadow_lanes,
    )
    runtime_lanes = output["lanes"]
    expected = case.expected
    classification_correct = all(
        [
            runtime_lanes["kb"]["status"] == expected.kb_status,
            shadow_lanes.kb.usable == expected.kb_usable,
            runtime_lanes["structured_fact"]["status"] == expected.structured_fact_status,
            shadow_lanes.structured_fact.usable == expected.structured_fact_usable,
            output["status"] == expected.status,
            output["failure_stage"] == expected.failure_stage,
            output["degradation"]["active"] == expected.degradation_active,
            output["degradation"]["affected_lanes"] == expected.affected_lanes,
            (output.get("analyst") is not None) == expected.analyst_ran,
        ]
    )
    runtime_evaluator_consistent = all(
        [
            runtime_lanes["kb"]["status"] == shadow_lanes.kb.status,
            runtime_lanes["structured_fact"]["status"] == shadow_lanes.structured_fact.status,
            output["status"] == shadow_status,
            output["failure_stage"] == shadow_failure_stage,
        ]
    )
    disclosure_correct = (
        not expected.degradation_active
        or (
            bool(output["degradation"]["notice"])
            and output["degradation"]["notice"]
            == output["evaluation_trace"]["analyst_packet"]["degradation"]["notice"]
            and "fault-injected" not in output["degradation"]["notice"]
        )
    )
    all_unusable_filing = (
        case.intent in {PlannerIntent.FILING_FACT, PlannerIntent.FILING_CALC}
        and not shadow_lanes.kb.usable
        and not shadow_lanes.structured_fact.usable
    )
    fail_closed = (
        not all_unusable_filing
        or (output["status"] == "failed" and output.get("analyst") is None)
    )
    containment = (
        expected.analyst_ran
        or output.get("analyst") is None
    )
    return {
        "id": case.id,
        "output": output,
        "shadow": {
            "lanes": shadow_lanes.model_dump(mode="json"),
            "status": shadow_status,
            "failure_stage": shadow_failure_stage,
        },
        "checks": {
            "classification_correct": classification_correct,
            "failure_contained": containment,
            "all_lanes_unusable_fail_closed": fail_closed,
            "degradation_disclosed": disclosure_correct,
            "runtime_evaluator_consistent": runtime_evaluator_consistent,
            "overall_correct": all(
                [classification_correct, containment, fail_closed, disclosure_correct, runtime_evaluator_consistent]
            ),
        },
    }


def evaluate_degradation_matrix(cases: Sequence[DegradationCase]) -> Dict[str, Any]:
    rows = [evaluate_degradation_case(case) for case in cases]

    def rate(name: str) -> float:
        return (
            sum(bool(row["checks"][name]) for row in rows) / float(len(rows))
            if rows
            else 0.0
        )

    return {
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "degradation_classification_accuracy": rate("classification_correct"),
            "failure_containment_rate": rate("failure_contained"),
            "all_lanes_unusable_fail_closed_rate": rate("all_lanes_unusable_fail_closed"),
            "degradation_disclosure_rate": rate("degradation_disclosed"),
            "runtime_evaluator_consistency_rate": rate("runtime_evaluator_consistent"),
            "overall_graceful_degradation_behavior_accuracy": rate("overall_correct"),
        },
    }
