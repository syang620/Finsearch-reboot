from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.analyst.agent import AnalystRunResult
from agents.contracts import PlannerOutput, RetrieveTablesResponse
from evals.agent_eval_contracts import (
    AgentDeterministicChecks,
    AgentEvalExample,
    AgentEvalRow,
    AgentExpected,
    load_agent_eval_examples,
)


MODE = "legacy_v0"
INCLUDE_EVIDENCE_TRACE = False


def _extract_table_payload(top_table: Dict[str, Any]) -> Dict[str, Any]:
    table_obj = top_table.get("table")
    if isinstance(table_obj, dict):
        payload = table_obj.get("payload")
        if isinstance(payload, dict):
            return payload
    payload_obj = getattr(table_obj, "payload", None)
    if isinstance(payload_obj, dict):
        return payload_obj
    if hasattr(table_obj, "model_dump"):
        try:
            dumped = table_obj.model_dump()
            payload = dumped.get("payload")
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _extract_context_from_payload(payload: Dict[str, Any]) -> str:
    for key in ("rerank_table_summary", "content", "rerank_original_content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_contexts_from_retrieval_output(
    retrieval_output: Any,
    *,
    max_items: int = 5,
) -> List[str]:
    if not isinstance(retrieval_output, dict):
        return []
    top_tables = retrieval_output.get("top_tables") or []
    contexts: List[str] = []
    for item in top_tables[:max_items]:
        payload = _extract_table_payload(item if isinstance(item, dict) else {})
        context = _extract_context_from_payload(payload)
        if context:
            contexts.append(context)
    return contexts


def _normalize_form_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().upper()
        return normalized or None
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value.strip().upper() or None
    return str(value).strip().upper() or None


def _metadata_match(
    expected: AgentExpected,
    planner_metadata: Dict[str, Any],
) -> Optional[bool]:
    checks: List[bool] = []
    if expected.ticker is not None:
        checks.append(
            str(planner_metadata.get("ticker") or "").upper() == expected.ticker
        )
    if expected.fiscal_year is not None:
        try:
            got_year = int(planner_metadata.get("fiscal_year"))
        except Exception:
            got_year = None
        checks.append(got_year == expected.fiscal_year)
    if expected.form_type is not None:
        checks.append(
            _normalize_form_type(planner_metadata.get("form_type"))
            == expected.form_type
        )
    return all(checks) if checks else None


def _score_deterministic_checks(
    checks: AgentDeterministicChecks,
) -> AgentDeterministicChecks:
    weights: Dict[str, float] = {
        "intent_match": 0.125,
        "retrieval_needed_match": 0.125,
        "metadata_match": 0.20,
        "tool_use_match": 0.15,
        "compute_match": 0.15,
        "citation_match": 0.15,
        "contract_valid": 0.10,
    }
    total_weight = 0.0
    weighted_sum = 0.0
    components: Dict[str, float] = {}
    for key, weight in weights.items():
        value = getattr(checks, key)
        if value is None:
            continue
        numeric = 1.0 if bool(value) else 0.0
        total_weight += weight
        weighted_sum += weight * numeric
        components[key] = numeric
    checks.weighted_components = components
    checks.score = weighted_sum / total_weight if total_weight > 0 else 0.0
    return checks


def build_deterministic_checks(
    *,
    ex: AgentEvalExample,
    planner_obj: Optional[PlannerOutput],
    analyst_obj: Optional[AnalystRunResult],
    contract_valid: bool,
) -> AgentDeterministicChecks:
    checks = AgentDeterministicChecks(contract_valid=bool(contract_valid))
    expected = ex.expected
    planner_intent = planner_obj.intent.value if planner_obj else None
    retrieval_needed = planner_obj.retrieval_needed if planner_obj else None

    if expected.intent is not None and planner_intent is not None:
        checks.intent_match = planner_intent == expected.intent
    if expected.retrieval_needed is not None and retrieval_needed is not None:
        checks.retrieval_needed_match = bool(retrieval_needed) == bool(
            expected.retrieval_needed
        )
    if planner_obj is not None:
        checks.metadata_match = _metadata_match(
            expected,
            planner_obj.metadata.model_dump(mode="json"),
        )

    used_tool = None
    citation_count = 0
    analyst_metric = None
    has_computation = False
    if analyst_obj is not None:
        used_tool = bool(
            getattr(analyst_obj.trace, "used_financial_evaluator", False)
        )
        citation_count = len(analyst_obj.citations or [])
        analyst_metric = str(analyst_obj.metric or "").strip().lower()
        has_computation = analyst_obj.computation is not None

    if expected.must_use_financial_evaluator is not None and used_tool is not None:
        checks.tool_use_match = bool(used_tool) == bool(
            expected.must_use_financial_evaluator
        )
    if expected.expected_metric is not None and analyst_metric:
        checks.compute_match = expected.expected_metric == analyst_metric
        if expected.must_use_financial_evaluator is True and not has_computation:
            checks.compute_match = False
    if expected.expected_min_citations is not None:
        checks.citation_match = citation_count >= int(
            expected.expected_min_citations
        )

    if not checks.contract_valid:
        checks.critical_failures.append("CONTRACT_INVALID")
    if planner_intent in {"filing_fact", "filing_calc"} and retrieval_needed is False:
        checks.critical_failures.append("RETRIEVAL_ROUTING_INVALID")
    if expected.must_use_financial_evaluator is True and used_tool is not True:
        checks.critical_failures.append("COMPUTE_TOOL_NOT_USED")
    return _score_deterministic_checks(checks)


def build_orchestrator_error(
    ex: AgentEvalExample,
    exc: Exception,
) -> Tuple[AgentEvalRow, List[Dict[str, Any]], None]:
    row = AgentEvalRow(
        id=ex.id,
        query=ex.user_query,
        orchestrator_ok=False,
        orchestrator_error=str(exc),
    )
    error = {"id": ex.id, "stage": "orchestrator", "error": str(exc)}
    row.errors.append(error)
    return row, [error], None


def evaluate_run_output(
    ex: AgentEvalExample,
    run_output: Dict[str, Any],
) -> Tuple[AgentEvalRow, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if str(run_output.get("status") or "").strip().lower() == "interrupted":
        return build_orchestrator_error(
            ex,
            RuntimeError(
                f"ORCHESTRATOR_INTERRUPTED: {run_output.get('interrupt')}"
            ),
        )

    row = AgentEvalRow(id=ex.id, query=ex.user_query, orchestrator_ok=True)
    errors: List[Dict[str, Any]] = []
    planner_obj: Optional[PlannerOutput] = None
    analyst_obj: Optional[AnalystRunResult] = None
    contract_valid = True

    planner_dump = run_output.get("planner") or {}
    analyst_dump = run_output.get("analyst") or {}
    retrieval_dump = run_output.get("retrieval")

    try:
        planner_obj = PlannerOutput.model_validate(planner_dump)
    except Exception as exc:
        contract_valid = False
        error = {"id": ex.id, "stage": "planner_contract", "error": str(exc)}
        row.errors.append(error)
        errors.append(error)

    if retrieval_dump is not None:
        try:
            RetrieveTablesResponse.model_validate(retrieval_dump)
        except Exception as exc:
            contract_valid = False
            error = {
                "id": ex.id,
                "stage": "retrieval_contract",
                "error": str(exc),
            }
            row.errors.append(error)
            errors.append(error)

    try:
        analyst_obj = AnalystRunResult.model_validate(analyst_dump)
    except Exception as exc:
        contract_valid = False
        error = {"id": ex.id, "stage": "analyst_contract", "error": str(exc)}
        row.errors.append(error)
        errors.append(error)

    if planner_obj is not None:
        row.planner_intent = planner_obj.intent.value
        row.planner_retrieval_needed = planner_obj.retrieval_needed
        row.planner_ticker = planner_obj.metadata.ticker
        row.planner_fiscal_year = planner_obj.metadata.fiscal_year
        row.planner_form_type = (
            planner_obj.metadata.form_type.value
            if planner_obj.metadata.form_type is not None
            else None
        )

    if analyst_obj is not None:
        row.analyst_metric = analyst_obj.metric
        row.analyst_answer = analyst_obj.answer
        row.analyst_used_financial_evaluator = bool(
            analyst_obj.trace.used_financial_evaluator
        )
        row.analyst_citation_count = len(analyst_obj.citations or [])

    row.deterministic = build_deterministic_checks(
        ex=ex,
        planner_obj=planner_obj,
        analyst_obj=analyst_obj,
        contract_valid=contract_valid,
    )
    orchestrator_trace = run_output.get("orchestrator_trace") or {}
    row.trace = {
        "orchestrator_trace": orchestrator_trace,
        "planner_timing_ms": orchestrator_trace.get("planner_timing_ms") or {},
        "retrieval_timing_ms": orchestrator_trace.get("retrieval_timing_ms") or {},
    }

    ragas_sample = {
        "id": ex.id,
        "question": ex.user_query,
        "answer": row.analyst_answer,
        "reference": ex.gold_answer,
        "contexts": _extract_contexts_from_retrieval_output(
            retrieval_dump,
            max_items=5,
        ),
    }
    return row, errors, ragas_sample


def deterministic_summary(rows: Sequence[AgentEvalRow]) -> Dict[str, float]:
    def mean(values: Sequence[float]) -> float:
        return float(sum(values)) / float(len(values)) if values else 0.0

    def rate(field: str) -> float:
        values = []
        for row in rows:
            value = getattr(row.deterministic, field)
            if value is not None:
                values.append(1.0 if bool(value) else 0.0)
        return mean(values)

    summary = {
        "score_mean": mean([float(row.deterministic.score) for row in rows]),
        "contract_valid_rate": mean(
            [1.0 if row.deterministic.contract_valid else 0.0 for row in rows]
        ),
        "intent_match_rate": rate("intent_match"),
        "retrieval_needed_match_rate": rate("retrieval_needed_match"),
        "metadata_match_rate": rate("metadata_match"),
        "tool_use_match_rate": rate("tool_use_match"),
        "compute_match_rate": rate("compute_match"),
        "citation_match_rate": rate("citation_match"),
    }
    critical_count = sum(
        1 for row in rows if row.deterministic.critical_failures
    )
    summary["critical_failure_rate"] = (
        float(critical_count) / float(len(rows)) if rows else 0.0
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
