from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.analyst import (
    ANALYST_CONTEXT_ITEM_LIMIT,
    AnalystRunResult,
    render_structured_fact_evidence,
)
from agents.contracts import (
    AnalystPacket,
    ContextItemKind,
    DegradationSummary,
    FormType,
    PlannerRuntimeOutput,
    normalize_missing_component_groups,
)
from evals.agent_eval_v1_contracts import (
    AgentDeterministicChecksV1,
    AgentEvalExampleV1,
    AgentEvalRowV1,
    AgentExpectedV1,
    AgentLaneObservation,
    AgentLaneStatusSet,
    EffectiveStatus,
    load_agent_eval_examples_v1,
)


MODE = "route_aware_v1"
INCLUDE_EVIDENCE_TRACE = True
load_agent_eval_examples = load_agent_eval_examples_v1

_EFFECTIVE_STATUSES = {"completed", "degraded", "failed", "interrupted"}
_FAILURE_STAGES = {
    "none",
    "planner",
    "retrieval",
    "structured_fact",
    "analyst",
    "interrupted",
}
_KB_PACKET_EVIDENCE_LIMIT = 3
_KB_ADMISSION_LOSS_CODES = {
    "RETRIEVAL_CANDIDATE_UNSUPPORTED",
    "TABLE_HYDRATION_FAILED",
    "TABLE_MARKDOWN_EMPTY",
    "EMPTY_TEXT_CONTEXT",
}
_STRUCTURED_ADMISSION_LOSS_CODE = "STRUCTURED_FACT_INVALID_EVIDENCE"
_STRUCTURED_CAPABILITY_REJECTION_CODE = "STRUCTURED_FACT_CAPABILITY_REJECTED"
_DEFENSIVE_REJECTION_OUTCOME = "defensive_rejection"
_UNSUPPORTED_STRUCTURED_QUESTION_CLASSES = {
    "unsupported_derived_metric",
    "unsupported_ratio",
    "unsupported_per_share",
    "unsupported_comparison",
}


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_lower(value: Any) -> Optional[str]:
    normalized = _normalize_text(value)
    return normalized.lower() if normalized else None


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_form_type(value: Any) -> Optional[str]:
    normalized = _normalize_text(getattr(value, "value", value))
    return normalized.upper() if normalized else None


def _structured_form_matches(
    expected: Optional[FormType],
    returned_value: Any,
) -> bool:
    returned_text = _normalize_form_type(returned_value)
    if returned_text is None:
        return True
    try:
        returned = FormType(returned_text)
    except ValueError:
        return False
    if expected is None:
        return True
    if expected == FormType.TEN_K:
        return returned in {FormType.TEN_K, FormType.TEN_K_A}
    return returned == expected


def _metadata_match(
    expected: AgentExpectedV1,
    planner: Optional[PlannerRuntimeOutput],
) -> Optional[bool]:
    if planner is None:
        return None
    checks: List[bool] = []
    if expected.ticker is not None:
        checks.append((planner.metadata.ticker or "").upper() == expected.ticker)
    if expected.fiscal_year is not None:
        checks.append(planner.metadata.fiscal_year == expected.fiscal_year)
    if expected.form_type is not None:
        checks.append(
            _normalize_form_type(planner.metadata.form_type) == expected.form_type
        )
    return all(checks) if checks else None


def _packet_from_trace(run_output: Dict[str, Any]) -> Optional[AnalystPacket]:
    packet_dump = (run_output.get("evaluation_trace") or {}).get("analyst_packet")
    if packet_dump is None:
        return None
    try:
        return AnalystPacket.model_validate(packet_dump)
    except Exception:
        return None


def _analyst_issue_codes(
    run_output: Dict[str, Any],
    packet: Optional[AnalystPacket],
) -> set[str]:
    codes = {
        normalized.upper()
        for issue in run_output.get("open_issues") or []
        if isinstance(issue, dict)
        and (normalized := _normalize_text(issue.get("code")))
    }
    if packet is not None:
        codes.update(
            normalized.upper()
            for issue in packet.open_issues
            if (normalized := _normalize_text(issue.code))
        )
    return codes


def _retrieved_kb_candidate_count(
    retrieval: Dict[str, Any],
) -> Tuple[Optional[int], bool]:
    compact_count = _normalize_int(retrieval.get("retrieved_candidate_count"))
    if compact_count is not None:
        return max(0, compact_count), True

    top_tables = retrieval.get("top_tables")
    if isinstance(top_tables, list):
        return sum(isinstance(item, dict) for item in top_tables), True

    targets = [
        item for item in retrieval.get("targets") or [] if isinstance(item, dict)
    ]
    target_counts = [
        _normalize_int(item.get("tables_retrieved")) for item in targets
    ]
    known_target_counts = [
        max(0, int(count)) for count in target_counts if count is not None
    ]

    attempt_result_counts: List[int] = []
    observed_doc_ids: set[str] = set()
    for item in retrieval.get("attempts") or []:
        if not isinstance(item, dict) or not isinstance(item.get("results"), list):
            continue
        results = [result for result in item["results"] if isinstance(result, dict)]
        attempt_result_counts.append(len(results))
        observed_doc_ids.update(
            doc_id
            for result in results
            if (doc_id := _normalize_text(result.get("doc_id")))
        )
    observed_counts = known_target_counts + attempt_result_counts
    if observed_doc_ids:
        observed_counts.append(len(observed_doc_ids))
    if observed_counts:
        return max(observed_counts), False
    return None, False


def _derive_kb_lane(
    run_output: Dict[str, Any],
    packet: Optional[AnalystPacket],
) -> AgentLaneObservation:
    planner = run_output.get("planner") or {}
    route = _normalize_lower(run_output.get("route") or planner.get("route")) or "kb"
    requested = route in {"kb", "hybrid"} and bool(
        planner.get("retrieval_needed")
    )
    retrieval = run_output.get("retrieval") or {}
    attempts = [
        item for item in retrieval.get("attempts") or [] if isinstance(item, dict)
    ]
    targets = [
        item for item in retrieval.get("targets") or [] if isinstance(item, dict)
    ]
    total_runs = sum(int(item.get("runs") or 0) for item in targets)
    attempted = requested and bool(
        retrieval.get("attempted") is True
        or attempts
        or total_runs > 0
        or retrieval.get("ok") is True
        or retrieval.get("target")
    )
    visible_items = (
        packet.context_items[:ANALYST_CONTEXT_ITEM_LIMIT]
        if packet is not None
        else []
    )
    usable_evidence_count = sum(
        1
        for item in visible_items
        if item.kind != ContextItemKind.STRUCTURED_FACT
    )
    usable = usable_evidence_count > 0
    issue_codes = _analyst_issue_codes(run_output, packet)
    retrieved_candidate_count, exact_candidate_count = (
        _retrieved_kb_candidate_count(retrieval)
    )
    expected_admission_count = (
        min(retrieved_candidate_count, _KB_PACKET_EVIDENCE_LIMIT)
        if retrieved_candidate_count is not None
        else None
    )
    has_admission_issue = bool(
        issue_codes.intersection(_KB_ADMISSION_LOSS_CODES)
    )
    admission_loss = bool(
        expected_admission_count is not None
        and usable_evidence_count < expected_admission_count
        and (exact_candidate_count or has_admission_issue)
    ) or bool(
        retrieved_candidate_count is not None
        and usable_evidence_count < retrieved_candidate_count
        and has_admission_issue
    )

    if not requested:
        status = "not_requested"
    elif not attempted:
        status = "skipped"
    elif usable:
        failed_runs = sum(int(item.get("failed_runs") or 0) for item in targets)
        if (
            failed_runs
            or retrieval.get("ok") is False
            or "RETRIEVAL_PARTIAL_FAILURE" in issue_codes
            or admission_loss
        ):
            status = "partial"
        else:
            status = "ok"
    else:
        status = "failed"
    return AgentLaneObservation(
        requested=requested,
        attempted=attempted,
        status=status,
        usable=usable,
        usable_evidence_count=usable_evidence_count,
    )


def _structured_result_status(result: Dict[str, Any]) -> str:
    resolver_status = _normalize_lower(result.get("resolver_status")) or "unknown"
    if resolver_status != "resolved":
        return resolver_status
    tool_result = result.get("tool_result")
    if not isinstance(tool_result, dict):
        return "failed"
    tool_status = _normalize_lower(tool_result.get("status"))
    if tool_status:
        return tool_status
    return "ok" if bool(tool_result.get("ok")) else "failed"


def _defensive_structured_rejection_classes(
    run_output: Dict[str, Any],
    packet: Optional[AnalystPacket],
    *,
    result_count: int,
) -> List[str]:
    issues = (
        [issue.model_dump(mode="json") for issue in packet.open_issues]
        if packet is not None
        else [
            issue
            for issue in run_output.get("open_issues") or []
            if isinstance(issue, dict)
        ]
    )
    rejection_metadata = [
        metadata
        for issue in issues
        if _normalize_text(issue.get("code"))
        == _STRUCTURED_CAPABILITY_REJECTION_CODE
        and isinstance((metadata := issue.get("metadata")), dict)
        and _normalize_lower(metadata.get("outcome"))
        == _DEFENSIVE_REJECTION_OUTCOME
    ]
    if not rejection_metadata or result_count <= 0:
        return []

    indexed_metadata = [
        metadata
        for metadata in rejection_metadata
        if "request_index" in metadata
    ]
    if not indexed_metadata:
        if len(rejection_metadata) != result_count:
            return []
        return [
            _normalize_lower(metadata.get("question_class")) or "unknown"
            for metadata in rejection_metadata
        ]
    if len(indexed_metadata) != len(rejection_metadata):
        return []

    classes_by_index: Dict[int, str] = {}
    for metadata in indexed_metadata:
        request_index = _normalize_int(metadata.get("request_index"))
        if request_index is None or request_index in classes_by_index:
            return []
        classes_by_index[request_index] = (
            _normalize_lower(metadata.get("question_class")) or "unknown"
        )
    if set(classes_by_index) != set(range(result_count)):
        return []
    return [classes_by_index[index] for index in range(result_count)]


def _derive_structured_lane(
    run_output: Dict[str, Any],
    packet: Optional[AnalystPacket],
) -> Tuple[AgentLaneObservation, List[str], List[str]]:
    planner = run_output.get("planner") or {}
    route = _normalize_lower(run_output.get("route") or planner.get("route")) or "kb"
    requests = [
        item
        for item in planner.get("structured_fact_requests") or []
        if isinstance(item, dict)
    ]
    requested = route in {"structured_fact", "hybrid"} and bool(requests)
    raw_structured_results = run_output.get("structured_fact_results")
    raw_results_are_sequence = isinstance(
        raw_structured_results, Sequence
    ) and not isinstance(raw_structured_results, (str, bytes, bytearray))
    raw_results = list(raw_structured_results) if raw_results_are_sequence else []
    results = [item for item in raw_results if isinstance(item, dict)]
    results_are_well_formed = (
        raw_results_are_sequence and len(results) == len(raw_results)
    )
    complete_operation_coverage = (
        results_are_well_formed and len(results) == len(requests)
    )
    attempted = raw_structured_results is not None and (
        not raw_results_are_sequence or bool(raw_results)
    )
    visible_items = (
        packet.context_items[:ANALYST_CONTEXT_ITEM_LIMIT]
        if packet is not None
        else []
    )
    usable_evidence_count = sum(
        1
        for item in visible_items
        if item.kind == ContextItemKind.STRUCTURED_FACT
    )
    usable = usable_evidence_count > 0

    statuses = [_structured_result_status(result) for result in results]
    successful_result_count = sum(status == "ok" for status in statuses)
    issue_codes = _analyst_issue_codes(run_output, packet)
    defensive_rejection_classes = _defensive_structured_rejection_classes(
        run_output,
        packet,
        result_count=len(results),
    )
    admission_loss = (
        successful_result_count != usable_evidence_count
        or _STRUCTURED_ADMISSION_LOSS_CODE in issue_codes
    )
    metric_ids = [
        metric_id
        for result in results
        if (metric_id := _normalize_lower(result.get("resolved_metric_id")))
    ]

    if not requested:
        aggregate_status = "not_requested"
    elif not attempted:
        aggregate_status = "skipped"
    elif (
        usable
        and statuses
        and complete_operation_coverage
        and all(status == "ok" for status in statuses)
        and not admission_loss
    ):
        aggregate_status = "ok"
    elif successful_result_count and not usable:
        aggregate_status = "failed"
    elif usable:
        aggregate_status = "partial"
    elif complete_operation_coverage and statuses and all(
        status == "partial" for status in statuses
    ):
        aggregate_status = "partial"
    elif "partial" in statuses:
        aggregate_status = "failed"
    elif complete_operation_coverage and defensive_rejection_classes:
        question_classes = set(defensive_rejection_classes)
        if question_classes == {"ambiguous"}:
            aggregate_status = "ambiguous"
        elif question_classes <= _UNSUPPORTED_STRUCTURED_QUESTION_CLASSES:
            aggregate_status = "unsupported"
        else:
            aggregate_status = "failed"
    elif complete_operation_coverage and statuses and all(
        status == "ambiguous" for status in statuses
    ):
        aggregate_status = "ambiguous"
    elif complete_operation_coverage and statuses and all(
        status in {"unsupported", "unsupported_metric"} for status in statuses
    ):
        aggregate_status = "unsupported"
    else:
        aggregate_status = "failed"

    return (
        AgentLaneObservation(
            requested=requested,
            attempted=attempted,
            status=aggregate_status,
            usable=usable,
            usable_evidence_count=usable_evidence_count,
        ),
        metric_ids,
        statuses,
    )


def derive_lane_status(
    run_output: Dict[str, Any],
) -> Tuple[AgentLaneStatusSet, List[str], List[str]]:
    packet = _packet_from_trace(run_output)
    structured_lane, metric_ids, statuses = _derive_structured_lane(
        run_output, packet
    )
    return (
        AgentLaneStatusSet(
            kb=_derive_kb_lane(run_output, packet),
            structured_fact=structured_lane,
        ),
        metric_ids,
        statuses,
    )


def _lane_status_consistent(
    runtime: AgentLaneStatusSet,
    derived: AgentLaneStatusSet,
) -> bool:
    for lane_name in ("kb", "structured_fact"):
        runtime_lane = getattr(runtime, lane_name)
        derived_lane = getattr(derived, lane_name)
        if (
            runtime_lane.requested != derived_lane.requested
            or runtime_lane.attempted != derived_lane.attempted
            or runtime_lane.status != derived_lane.status
        ):
            return False
    return True


def _derive_effective_status(
    reported_status: Optional[str],
    lanes: AgentLaneStatusSet,
    planner: Optional[PlannerRuntimeOutput],
    analyst: Optional[AnalystRunResult],
) -> Optional[EffectiveStatus]:
    if reported_status == "interrupted":
        return reported_status
    if analyst is not None and not analyst.ok:
        return "failed"
    if planner is not None and planner.status != "completed":
        return "failed"
    if reported_status not in {"completed", "degraded", "failed"}:
        return None
    requested_lanes = [
        lane
        for lane in (lanes.kb, lanes.structured_fact)
        if lane.requested
    ]
    if (
        planner is not None
        and planner.intent.value in {"filing_fact", "filing_calc"}
        and not any(lane.usable for lane in requested_lanes)
    ):
        return "failed"
    if reported_status == "failed":
        return "failed"
    if any(lane.status != "ok" for lane in requested_lanes):
        return "degraded"
    return "completed"


def _derive_failure_stage_shadow(
    *,
    reported_status: Optional[str],
    planner: Optional[PlannerRuntimeOutput],
    analyst: Optional[AnalystRunResult],
    lanes: AgentLaneStatusSet,
) -> Optional[str]:
    if reported_status == "interrupted":
        return "interrupted"
    if planner is None:
        return None
    if planner.status != "completed":
        return "planner"
    if analyst is not None and not analyst.ok:
        return "analyst"
    if planner.intent.value in {"filing_fact", "filing_calc"}:
        requested = [lanes.kb, lanes.structured_fact]
        if not any(lane.usable for lane in requested if lane.requested):
            if lanes.structured_fact.attempted:
                return "structured_fact"
            if lanes.kb.attempted:
                return "retrieval"
            return "planner"
    return "none"


def _expected_degradation(lanes: AgentLaneStatusSet) -> DegradationSummary:
    affected = [
        lane_name
        for lane_name in ("kb", "structured_fact")
        if (lane := getattr(lanes, lane_name)).requested and lane.status != "ok"
    ]
    if not affected:
        return DegradationSummary()
    statuses = "; ".join(
        f"{lane_name}={getattr(lanes, lane_name).status}"
        for lane_name in affected
    )
    return DegradationSummary(
        active=True,
        affected_lanes=affected,
        notice=(
            f"Evidence coverage is degraded ({statuses}). Do not claim coverage "
            "from unavailable or incomplete evidence lanes."
        ),
    )


def _extract_semantic_contexts(
    run_output: Dict[str, Any],
) -> Tuple[List[str], List[str], Optional[AnalystPacket], Optional[str]]:
    evaluation_trace = run_output.get("evaluation_trace") or {}
    packet_dump = evaluation_trace.get("analyst_packet")
    if packet_dump is None:
        return [], [], None, "evaluation_trace.analyst_packet is missing"
    try:
        packet = AnalystPacket.model_validate(packet_dump)
    except Exception as exc:
        return [], [], None, str(exc)

    contexts: List[str] = []
    context_kinds: List[str] = []
    for item in packet.context_items[:ANALYST_CONTEXT_ITEM_LIMIT]:
        if item.kind == ContextItemKind.STRUCTURED_FACT:
            contexts.append(render_structured_fact_evidence(item.structured_fact))
            context_kinds.append(item.kind.value)
            continue
        payload = item.payload or {}
        content = (
            payload.get("table_markdown")
            or payload.get("content")
            or payload.get("text")
            or ""
        )
        normalized = str(content).strip()
        if normalized:
            contexts.append(normalized)
            context_kinds.append(item.kind.value)
    return contexts, context_kinds, packet, None


def _structured_evidence_diagnostics(
    run_output: Dict[str, Any],
    packet: Optional[AnalystPacket],
) -> Dict[str, Any]:
    successful_results = 0
    for result in run_output.get("structured_fact_results") or []:
        if not isinstance(result, dict):
            continue
        tool_result = result.get("tool_result") or {}
        value = tool_result.get("value") if isinstance(tool_result, dict) else None
        resolved_metric_id = _normalize_text(result.get("resolved_metric_id"))
        tool_metric_id = (
            _normalize_text(tool_result.get("metric_id"))
            if isinstance(tool_result, dict)
            else None
        )
        resolved_ticker = _normalize_text(result.get("resolved_ticker"))
        tool_ticker = (
            _normalize_text(tool_result.get("ticker"))
            if isinstance(tool_result, dict)
            else None
        )
        resolved_year = _normalize_int(result.get("resolved_fiscal_year"))
        tool_year = (
            _normalize_int(tool_result.get("fiscal_year"))
            if isinstance(tool_result, dict)
            else None
        )
        raw_components = (
            tool_result.get("components")
            if isinstance(tool_result, dict)
            else None
        )
        try:
            normalize_missing_component_groups(
                tool_result.get("missing_component_groups")
                if isinstance(tool_result, dict)
                else None
            )
            missing_component_groups_valid = True
        except ValueError:
            missing_component_groups_valid = False
        if (
            _normalize_lower(result.get("resolver_status")) == "resolved"
            and resolved_metric_id is not None
            and isinstance(tool_result, dict)
            and (tool_metric_id is None or tool_metric_id == resolved_metric_id)
            and (
                resolved_ticker is None
                or tool_ticker is None
                or resolved_ticker.upper() == tool_ticker.upper()
            )
            and (
                resolved_year is None
                or tool_year is None
                or resolved_year == tool_year
            )
            and _structured_form_matches(
                packet.metadata.form_type if packet is not None else None,
                tool_result.get("form_type"),
            )
            and (
                raw_components is None
                or (
                    isinstance(raw_components, list)
                    and all(
                        isinstance(component, dict)
                        for component in raw_components
                    )
                )
            )
            and missing_component_groups_valid
            and tool_result.get("ok") is True
            and _normalize_lower(tool_result.get("status")) == "ok"
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            successful_results += 1

    items = list(packet.context_items) if packet is not None else []
    typed_items = [
        item for item in items if item.kind == ContextItemKind.STRUCTURED_FACT
    ]
    visible_typed_items = [
        item
        for item in items[:ANALYST_CONTEXT_ITEM_LIMIT]
        if item.kind == ContextItemKind.STRUCTURED_FACT
    ]
    synthetic_text_items = 0
    for item in items:
        if item.kind == ContextItemKind.STRUCTURED_FACT:
            continue
        payload = item.payload or {}
        content = payload.get("content") or payload.get("text") or ""
        if str(content).lstrip().lower().startswith("structured fact:"):
            synthetic_text_items += 1

    provenance_fields = {
        "metric_id": lambda evidence: bool(evidence.metric_id),
        "numeric_value": lambda evidence: (
            not isinstance(evidence.value, bool)
            and isinstance(evidence.value, (int, float))
            and math.isfinite(float(evidence.value))
        ),
        "accession": lambda evidence: bool(evidence.accession_number),
        "report_date": lambda evidence: bool(evidence.report_date),
        "filed_date": lambda evidence: bool(evidence.filed_date),
        "source_url": lambda evidence: bool(evidence.source_url),
    }
    coverage_counts = {
        field_name: sum(
            1
            for item in typed_items
            if item.structured_fact is not None and predicate(item.structured_fact)
        )
        for field_name, predicate in provenance_fields.items()
    }
    return {
        "successful_execution_results": successful_results,
        "typed_structured_fact_contexts": len(typed_items),
        "analyst_visible_typed_structured_fact_contexts": len(visible_typed_items),
        "synthetic_structured_text_contexts": synthetic_text_items,
        "provenance_coverage_counts": coverage_counts,
    }


def _extract_timings(
    run_output: Dict[str, Any],
    analyst: Optional[AnalystRunResult],
) -> Dict[str, int]:
    trace = run_output.get("orchestrator_trace") or {}
    planner = trace.get("planner_timing_ms") or {}
    retrieval = trace.get("retrieval_timing_ms") or {}
    structured = trace.get("structured_fact_timing_ms") or {}
    analyst_timing = analyst.trace.timing_ms if analyst is not None else {}
    return {
        "planner": sum(int(value) for value in planner.values() if isinstance(value, (int, float))),
        "retrieval": int(retrieval.get("retrieve_ms") or 0),
        "structured_fact": int(structured.get("structured_facts_ms") or 0),
        "analyst": int(analyst_timing.get("total_ms") or 0),
        "total": int(trace.get("total_ms") or 0),
    }


def _score_checks(
    checks: AgentDeterministicChecksV1,
) -> AgentDeterministicChecksV1:
    excluded = {"score", "weighted_components", "critical_failures"}
    components: Dict[str, float] = {}
    for field_name in type(checks).model_fields:
        if field_name in excluded:
            continue
        value = getattr(checks, field_name)
        if value is not None:
            components[field_name] = 1.0 if bool(value) else 0.0
    checks.weighted_components = components
    checks.score = (
        sum(components.values()) / float(len(components)) if components else 0.0
    )
    return checks


def _append_critical(
    checks: AgentDeterministicChecksV1,
    condition: bool,
    code: str,
) -> None:
    if condition and code not in checks.critical_failures:
        checks.critical_failures.append(code)


def _build_checks(
    *,
    ex: AgentEvalExampleV1,
    row: AgentEvalRowV1,
    planner: Optional[PlannerRuntimeOutput],
    analyst: Optional[AnalystRunResult],
    contract_valid: bool,
) -> AgentDeterministicChecksV1:
    expected = ex.expected
    checks = AgentDeterministicChecksV1(contract_valid=contract_valid)

    if expected.intent is not None:
        checks.intent_match = bool(planner) and planner.intent.value == expected.intent
    if expected.route is not None:
        checks.route_match = row.route == expected.route
    checks.metadata_match = _metadata_match(expected, planner)
    if expected.expect_kb_lane is not None:
        checks.kb_lane_match = (
            row.lane_status.kb.requested == expected.expect_kb_lane
        )
    if expected.expect_structured_fact_lane is not None:
        checks.structured_lane_match = (
            row.lane_status.structured_fact.requested
            == expected.expect_structured_fact_lane
        )
    if expected.expected_structured_metric is not None:
        checks.structured_metric_match = bool(row.structured_metric_ids) and all(
            metric_id == expected.expected_structured_metric
            for metric_id in row.structured_metric_ids
        )
    if expected.expected_structured_status is not None:
        checks.structured_status_match = bool(row.structured_statuses) and all(
            status == expected.expected_structured_status
            for status in row.structured_statuses
        )
    if expected.expected_reported_status is not None:
        checks.reported_status_match = (
            row.reported_status == expected.expected_reported_status
        )
    if expected.expected_effective_status is not None:
        checks.effective_status_match = (
            row.effective_status == expected.expected_effective_status
        )
    if expected.expected_failure_stage is not None:
        checks.failure_stage_match = (
            row.failure_stage == expected.expected_failure_stage
        )
    checks.degradation_match = (
        row.effective_status != "degraded" or expected.allow_degraded
    )
    checks.lane_status_consistent = row.lane_status_consistent
    checks.effective_status_consistent = row.effective_status_consistent
    checks.failure_stage_consistent = row.failure_stage_consistent
    checks.degradation_consistent = row.degradation_consistent

    if expected.expected_analyst_status is not None:
        checks.analyst_status_match = (
            row.analyst_status == expected.expected_analyst_status
        )
    if expected.must_use_financial_evaluator is not None:
        checks.tool_use_match = (
            row.analyst_used_financial_evaluator
            == expected.must_use_financial_evaluator
        )
    if expected.expected_metric is not None:
        checks.compute_match = row.analyst_metric == expected.expected_metric
        if expected.must_use_financial_evaluator is True:
            checks.compute_match = bool(
                checks.compute_match
                and analyst is not None
                and analyst.computation is not None
            )
    if expected.expected_min_citations is not None:
        checks.citation_match = (
            row.analyst_citation_count >= expected.expected_min_citations
        )

    _append_critical(checks, not checks.contract_valid, "CONTRACT_INVALID")
    for field_name, code in (
        ("route_match", "ROUTE_MISMATCH"),
        ("kb_lane_match", "KB_LANE_MISMATCH"),
        ("structured_lane_match", "STRUCTURED_LANE_MISMATCH"),
        ("structured_metric_match", "STRUCTURED_METRIC_MISMATCH"),
        ("structured_status_match", "STRUCTURED_STATUS_MISMATCH"),
        ("reported_status_match", "REPORTED_STATUS_MISMATCH"),
        ("effective_status_match", "EFFECTIVE_STATUS_MISMATCH"),
        ("failure_stage_match", "FAILURE_STAGE_MISMATCH"),
        ("degradation_match", "DEGRADATION_NOT_ALLOWED"),
        ("lane_status_consistent", "LANE_STATUS_INCONSISTENT"),
        ("effective_status_consistent", "EFFECTIVE_STATUS_INCONSISTENT"),
        ("failure_stage_consistent", "FAILURE_STAGE_INCONSISTENT"),
        ("degradation_consistent", "DEGRADATION_INCONSISTENT"),
    ):
        value = getattr(checks, field_name)
        _append_critical(checks, value is False, code)

    _append_critical(
        checks,
        expected.expected_analyst_status is not None
        and checks.analyst_status_match is False,
        "ANALYST_STATUS_MISMATCH",
    )
    _append_critical(
        checks,
        expected.must_use_financial_evaluator is True
        and checks.tool_use_match is False,
        "COMPUTE_TOOL_NOT_USED",
    )
    _append_critical(
        checks,
        bool(expected.expected_min_citations)
        and checks.citation_match is False,
        "CITATION_MINIMUM_NOT_MET",
    )
    return _score_checks(checks)


def build_orchestrator_error(
    ex: AgentEvalExampleV1,
    exc: Exception,
) -> Tuple[AgentEvalRowV1, List[Dict[str, Any]], None]:
    empty_lanes = AgentLaneStatusSet(
        kb=AgentLaneObservation(),
        structured_fact=AgentLaneObservation(),
    )
    row = AgentEvalRowV1(
        id=ex.id,
        query=ex.user_query,
        expected=ex.expected.model_dump(mode="json"),
        orchestrator_ok=False,
        orchestrator_error=str(exc),
        derived_lane_status=empty_lanes,
        lane_status=empty_lanes,
    )
    error = {"id": ex.id, "stage": "orchestrator", "error": str(exc)}
    row.errors.append(error)
    row.deterministic = _score_checks(
        AgentDeterministicChecksV1(
            contract_valid=False,
            critical_failures=["CONTRACT_INVALID"],
        )
    )
    return row, [error], None


def evaluate_run_output(
    ex: AgentEvalExampleV1,
    run_output: Dict[str, Any],
) -> Tuple[AgentEvalRowV1, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    contract_valid = True
    planner: Optional[PlannerRuntimeOutput] = None
    analyst: Optional[AnalystRunResult] = None

    def contract_error(stage: str, exc: Any) -> None:
        nonlocal contract_valid
        contract_valid = False
        errors.append({"id": ex.id, "stage": stage, "error": str(exc)})

    planner_dump = run_output.get("planner")
    try:
        planner = PlannerRuntimeOutput.model_validate(planner_dump)
    except Exception as exc:
        contract_error("planner_contract", exc)

    derived_lanes, metric_ids, structured_statuses = derive_lane_status(run_output)
    runtime_lanes: Optional[AgentLaneStatusSet] = None
    raw_runtime_lanes = run_output.get("lanes")
    if raw_runtime_lanes is not None:
        try:
            runtime_lanes = AgentLaneStatusSet.model_validate(raw_runtime_lanes)
        except Exception as exc:
            contract_error("runtime_lane_contract", exc)

    if runtime_lanes is not None:
        lane_status = runtime_lanes
        lane_source = "runtime"
        lane_consistent: Optional[bool] = _lane_status_consistent(
            runtime_lanes,
            derived_lanes,
        )
    else:
        lane_status = derived_lanes
        lane_source = "evaluator_derived"
        lane_consistent = None

    reported_status = _normalize_lower(run_output.get("status"))
    if reported_status not in _EFFECTIVE_STATUSES:
        contract_error("runtime_status_contract", f"Unknown status: {reported_status}")
        reported_status = None

    analyst_dump = run_output.get("analyst")
    if analyst_dump is not None:
        try:
            analyst = AnalystRunResult.model_validate(analyst_dump)
        except Exception as exc:
            contract_error("analyst_contract", exc)

    derived_effective = _derive_effective_status(
        reported_status,
        derived_lanes,
        planner,
        analyst,
    )
    if runtime_lanes is not None:
        effective_status = reported_status
        effective_source = "runtime"
        effective_consistent = effective_status == derived_effective
    else:
        effective_status = derived_effective
        effective_source = "evaluator_derived"
        effective_consistent = None

    failure_stage = _normalize_lower(run_output.get("failure_stage"))
    if failure_stage not in _FAILURE_STAGES:
        contract_error(
            "failure_stage_contract",
            f"Unknown failure stage: {failure_stage}",
        )
    derived_failure_stage = _derive_failure_stage_shadow(
        reported_status=reported_status,
        planner=planner,
        analyst=analyst,
        lanes=derived_lanes,
    )
    failure_stage_consistent = (
        failure_stage == derived_failure_stage
        if runtime_lanes is not None and derived_failure_stage is not None
        else None
    )

    degradation: Dict[str, Any] = {}
    degradation_consistent: Optional[bool] = None
    raw_degradation = run_output.get("degradation")
    if raw_degradation is not None:
        try:
            degradation_obj = DegradationSummary.model_validate(raw_degradation)
            degradation = degradation_obj.model_dump(mode="json")
            expected_degradation = _expected_degradation(derived_lanes)
            packet = _packet_from_trace(run_output)
            packet_dump = (run_output.get("evaluation_trace") or {}).get(
                "analyst_packet"
            )
            packet_required = (
                effective_status in {"completed", "degraded"}
                or packet_dump is not None
            )
            degradation_consistent = (
                degradation_obj == expected_degradation
                and (
                    not packet_required
                    or (
                        packet is not None
                        and packet.degradation == expected_degradation
                    )
                )
            )
        except Exception as exc:
            contract_error("runtime_degradation_contract", exc)
            degradation_consistent = False
    elif runtime_lanes is not None:
        contract_error(
            "runtime_degradation_contract",
            "Runtime lane summaries require a degradation summary",
        )
        degradation_consistent = False

    if analyst_dump is None and effective_status in {"completed", "degraded"}:
        contract_error(
            "analyst_contract",
            "Completed or degraded run is missing an analyst result",
        )

    semantic_contexts, semantic_context_kinds, packet, packet_error = (
        _extract_semantic_contexts(run_output)
    )
    if packet_error and effective_status in {"completed", "degraded"}:
        contract_error("analyst_packet_contract", packet_error)

    route = _normalize_lower(run_output.get("route"))
    if planner is not None and route != planner.route:
        contract_error(
            "route_trace_contract",
            f"Runtime route {route!r} does not match planner route {planner.route!r}",
        )

    row = AgentEvalRowV1(
        id=ex.id,
        query=ex.user_query,
        expected=ex.expected.model_dump(mode="json"),
        orchestrator_ok=True,
        reported_status=reported_status,
        effective_status=effective_status,
        effective_status_source=effective_source,
        derived_effective_status=derived_effective,
        effective_status_consistent=effective_consistent,
        failure_stage=failure_stage,
        derived_failure_stage=derived_failure_stage,
        failure_stage_consistent=failure_stage_consistent,
        degradation=degradation,
        degradation_consistent=degradation_consistent,
        route=route,
        lane_status_source=lane_source,
        runtime_lane_status=runtime_lanes,
        derived_lane_status=derived_lanes,
        lane_status=lane_status,
        lane_status_consistent=lane_consistent,
        planner_intent=planner.intent.value if planner is not None else None,
        planner_ticker=planner.metadata.ticker if planner is not None else None,
        planner_fiscal_year=(
            planner.metadata.fiscal_year if planner is not None else None
        ),
        planner_form_type=(
            _normalize_form_type(planner.metadata.form_type)
            if planner is not None
            else None
        ),
        structured_metric_ids=metric_ids,
        structured_statuses=structured_statuses,
        analyst_status=_normalize_lower(analyst.status) if analyst is not None else None,
        analyst_metric=_normalize_lower(analyst.metric) if analyst is not None else None,
        analyst_answer=analyst.answer if analyst is not None else "",
        analyst_used_financial_evaluator=(
            bool(analyst.trace.used_financial_evaluator)
            if analyst is not None
            else None
        ),
        analyst_citation_count=len(analyst.citations or []) if analyst is not None else 0,
        timings_ms=_extract_timings(run_output, analyst),
        semantic_contexts=semantic_contexts,
        semantic_context_kinds=semantic_context_kinds,
        structured_evidence=_structured_evidence_diagnostics(run_output, packet),
        trace={
            "planner": run_output.get("planner"),
            "lanes": run_output.get("lanes"),
            "degradation": run_output.get("degradation"),
            "retrieval": run_output.get("retrieval"),
            "structured_fact_results": run_output.get("structured_fact_results") or [],
            "analyst": run_output.get("analyst"),
            "evaluation_trace": run_output.get("evaluation_trace"),
            "orchestrator_trace": run_output.get("orchestrator_trace") or {},
        },
        errors=errors,
    )
    row.deterministic = _build_checks(
        ex=ex,
        row=row,
        planner=planner,
        analyst=analyst,
        contract_valid=contract_valid,
    )

    ragas_sample = None
    if analyst is not None:
        ragas_sample = {
            "id": ex.id,
            "question": ex.user_query,
            "answer": row.analyst_answer,
            "reference": ex.gold_answer,
            "contexts": semantic_contexts,
        }
    return row, errors, ragas_sample


def deterministic_summary(rows: Sequence[AgentEvalRowV1]) -> Dict[str, float]:
    def mean(values: Sequence[float]) -> float:
        return float(sum(values)) / float(len(values)) if values else 0.0

    def rate(field_name: str) -> float:
        values: List[float] = []
        for row in rows:
            value = getattr(row.deterministic, field_name)
            if value is not None:
                values.append(1.0 if bool(value) else 0.0)
        return mean(values)

    check_fields = [
        name
        for name in AgentDeterministicChecksV1.model_fields
        if name not in {"score", "weighted_components", "critical_failures"}
    ]
    summary = {
        "score_mean": mean([float(row.deterministic.score) for row in rows]),
        **{f"{field_name}_rate": rate(field_name) for field_name in check_fields},
    }
    critical_count = sum(
        1 for row in rows if row.deterministic.critical_failures
    )
    summary["critical_failure_rate"] = (
        float(critical_count) / float(len(rows)) if rows else 0.0
    )
    for timing_name in ("planner", "retrieval", "structured_fact", "analyst", "total"):
        summary[f"{timing_name}_ms_mean"] = mean(
            [float(row.timings_ms.get(timing_name, 0)) for row in rows]
        )
    successful_results = sum(
        int(row.structured_evidence.get("successful_execution_results") or 0)
        for row in rows
    )
    typed_contexts = sum(
        int(row.structured_evidence.get("typed_structured_fact_contexts") or 0)
        for row in rows
    )
    visible_typed_contexts = sum(
        int(
            row.structured_evidence.get(
                "analyst_visible_typed_structured_fact_contexts"
            )
            or 0
        )
        for row in rows
    )
    synthetic_text_contexts = sum(
        int(row.structured_evidence.get("synthetic_structured_text_contexts") or 0)
        for row in rows
    )
    summary.update(
        {
            "structured_successful_execution_results": float(successful_results),
            "structured_typed_contexts": float(typed_contexts),
            "structured_analyst_visible_typed_contexts": float(
                visible_typed_contexts
            ),
            "structured_synthetic_text_contexts": float(synthetic_text_contexts),
            "structured_typed_representation_rate": (
                float(typed_contexts) / float(successful_results)
                if successful_results
                else 0.0
            ),
            "structured_analyst_visibility_rate": (
                float(visible_typed_contexts) / float(typed_contexts)
                if typed_contexts
                else 0.0
            ),
        }
    )
    for field_name in (
        "metric_id",
        "numeric_value",
        "accession",
        "report_date",
        "filed_date",
        "source_url",
    ):
        covered = sum(
            int(
                (
                    row.structured_evidence.get("provenance_coverage_counts")
                    or {}
                ).get(field_name)
                or 0
            )
            for row in rows
        )
        summary[f"structured_{field_name}_coverage_rate"] = (
            float(covered) / float(typed_contexts) if typed_contexts else 0.0
        )
    return summary


def build_summary_config(
    *,
    eval_path: str,
    analyst_model: str,
    planner_model: Optional[str],
    enable_ragas: bool,
    thresholds: Any,
    total_ms: int,
) -> Dict[str, Any]:
    return {
        "mode": MODE,
        "eval_path": str(eval_path),
        "analyst_model": analyst_model,
        "planner_model": planner_model,
        "enable_ragas": bool(enable_ragas),
        "thresholds": {
            "deterministic_score_min": thresholds.deterministic_score_min,
            "max_critical_failure_rate": thresholds.max_critical_failure_rate,
            "ragas_faithfulness_min": thresholds.ragas_faithfulness_min,
            "ragas_answer_relevancy_min": thresholds.ragas_answer_relevancy_min,
            "max_ragas_error_rate": thresholds.max_ragas_error_rate,
        },
        "timing_ms": {"total": total_ms},
    }
