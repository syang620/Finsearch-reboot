from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.analyst.agent import AnalystRunResult
from agents.contracts import PlannerOutput, RetrieveTablesResponse
from agents.orchestrator.agent_orchestrator import run_multi_agent_orchestration
from agents.planner import InteractivePlannerAgent
from evals.agent_eval_contracts import (
    AgentDeterministicChecks,
    AgentEvalExample,
    AgentEvalRow,
    AgentEvalSummary,
    AgentExpected,
    AgentGateStatus,
    load_agent_eval_examples,
)
from evals.ragas_agent_metrics import RagasAgentConfig, evaluate_ragas_agents


@dataclass(frozen=True)
class AgentEvalThresholds:
    deterministic_score_min: float = 0.90
    max_critical_failure_rate: float = 0.05
    ragas_faithfulness_min: float = 0.75
    ragas_answer_relevancy_min: float = 0.75
    max_ragas_error_rate: float = 0.10


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        fv = float(value)
    except Exception:
        return False
    return not math.isnan(fv)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


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


def _extract_contexts_from_retrieval_output(retrieval_output: Any, *, max_items: int = 5) -> List[str]:
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
        s = value.strip().upper()
        return s or None
    v = getattr(value, "value", None)
    if isinstance(v, str):
        return v.strip().upper() or None
    return str(value).strip().upper() or None


def _metadata_match(expected: AgentExpected, planner_metadata: Dict[str, Any]) -> Optional[bool]:
    checks: List[bool] = []

    if expected.ticker is not None:
        checks.append(str(planner_metadata.get("ticker") or "").upper() == expected.ticker)

    if expected.fiscal_year is not None:
        try:
            got_year = int(planner_metadata.get("fiscal_year"))
        except Exception:
            got_year = None
        checks.append(got_year == expected.fiscal_year)

    if expected.form_type is not None:
        checks.append(_normalize_form_type(planner_metadata.get("form_type")) == expected.form_type)

    if not checks:
        return None
    return all(checks)


def _score_deterministic_checks(checks: AgentDeterministicChecks) -> AgentDeterministicChecks:
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
    checks.score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    return checks


def _build_deterministic_checks(
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
        checks.retrieval_needed_match = bool(retrieval_needed) == bool(expected.retrieval_needed)

    if planner_obj is not None:
        checks.metadata_match = _metadata_match(expected, planner_obj.metadata.model_dump(mode="json"))

    used_tool = None
    citation_count = 0
    analyst_metric = None
    has_computation = False
    if analyst_obj is not None:
        used_tool = bool(getattr(analyst_obj.trace, "used_financial_evaluator", False))
        citation_count = len(analyst_obj.citations or [])
        analyst_metric = str(analyst_obj.metric or "").strip().lower()
        has_computation = analyst_obj.computation is not None

    if expected.must_use_financial_evaluator is not None and used_tool is not None:
        checks.tool_use_match = bool(used_tool) == bool(expected.must_use_financial_evaluator)

    if expected.expected_metric is not None and analyst_metric:
        checks.compute_match = expected.expected_metric == analyst_metric
        if expected.must_use_financial_evaluator is True and not has_computation:
            checks.compute_match = False

    if expected.expected_min_citations is not None:
        checks.citation_match = citation_count >= int(expected.expected_min_citations)

    if not checks.contract_valid:
        checks.critical_failures.append("CONTRACT_INVALID")

    if planner_intent in {"filing_fact", "filing_calc"} and retrieval_needed is False:
        checks.critical_failures.append("RETRIEVAL_ROUTING_INVALID")

    if expected.must_use_financial_evaluator is True and used_tool is not True:
        checks.critical_failures.append("COMPUTE_TOOL_NOT_USED")

    return _score_deterministic_checks(checks)


async def _run_examples_async(
    *,
    examples: Sequence[AgentEvalExample],
    analyst_model: str,
    planner_model: Optional[str],
    fail_fast: bool,
) -> Tuple[List[AgentEvalRow], List[Dict[str, Any]], List[Dict[str, Any]]]:
    planner = (
        InteractivePlannerAgent(model=planner_model, log_timing=False)
        if planner_model
        else InteractivePlannerAgent(log_timing=False)
    )

    rows: List[AgentEvalRow] = []
    errors: List[Dict[str, Any]] = []
    ragas_samples: List[Dict[str, Any]] = []

    for ex in examples:
        row = AgentEvalRow(id=ex.id, query=ex.user_query)
        run_output: Dict[str, Any] = {}

        try:
            run_output = await run_multi_agent_orchestration(
                ex.user_query,
                planner=planner,
                analyst_model=analyst_model,
                debug=False,
            )
            if str(run_output.get("status") or "").strip().lower() == "interrupted":
                raise RuntimeError(f"ORCHESTRATOR_INTERRUPTED: {run_output.get('interrupt')}")
            row.orchestrator_ok = True
        except Exception as exc:
            row.orchestrator_ok = False
            row.orchestrator_error = str(exc)
            err = {
                "id": ex.id,
                "stage": "orchestrator",
                "error": str(exc),
            }
            row.errors.append(err)
            errors.append(err)
            rows.append(row)
            if fail_fast:
                break
            continue

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
            err = {
                "id": ex.id,
                "stage": "planner_contract",
                "error": str(exc),
            }
            row.errors.append(err)
            errors.append(err)

        if retrieval_dump is not None:
            try:
                RetrieveTablesResponse.model_validate(retrieval_dump)
            except Exception as exc:
                contract_valid = False
                err = {
                    "id": ex.id,
                    "stage": "retrieval_contract",
                    "error": str(exc),
                }
                row.errors.append(err)
                errors.append(err)

        try:
            analyst_obj = AnalystRunResult.model_validate(analyst_dump)
        except Exception as exc:
            contract_valid = False
            err = {
                "id": ex.id,
                "stage": "analyst_contract",
                "error": str(exc),
            }
            row.errors.append(err)
            errors.append(err)

        if planner_obj is not None:
            row.planner_intent = planner_obj.intent.value
            row.planner_retrieval_needed = planner_obj.retrieval_needed
            row.planner_ticker = planner_obj.metadata.ticker
            row.planner_fiscal_year = planner_obj.metadata.fiscal_year
            row.planner_form_type = planner_obj.metadata.form_type.value

        if analyst_obj is not None:
            row.analyst_metric = analyst_obj.metric
            row.analyst_answer = analyst_obj.answer
            row.analyst_used_financial_evaluator = bool(analyst_obj.trace.used_financial_evaluator)
            row.analyst_citation_count = len(analyst_obj.citations or [])

        row.deterministic = _build_deterministic_checks(
            ex=ex,
            planner_obj=planner_obj,
            analyst_obj=analyst_obj,
            contract_valid=contract_valid,
        )

        row.trace = {
            "orchestrator_trace": run_output.get("orchestrator_trace") or {},
            "planner_timing_ms": ((run_output.get("orchestrator_trace") or {}).get("planner_timing_ms") or {}),
            "retrieval_timing_ms": ((run_output.get("orchestrator_trace") or {}).get("retrieval_timing_ms") or {}),
        }

        rows.append(row)

        contexts = _extract_contexts_from_retrieval_output(retrieval_dump, max_items=5)
        ragas_samples.append(
            {
                "id": ex.id,
                "question": ex.user_query,
                "answer": row.analyst_answer,
                "reference": ex.gold_answer,
                "contexts": contexts,
            }
        )

        if fail_fast and row.errors:
            break

    return rows, errors, ragas_samples


def run_agent_eval(
    *,
    eval_path: str,
    out_dir: str,
    analyst_model: str = "qwen3:14b",
    planner_model: Optional[str] = None,
    enable_ragas: bool = True,
    ragas_config: Optional[RagasAgentConfig] = None,
    thresholds: Optional[AgentEvalThresholds] = None,
    fail_fast: bool = False,
) -> Tuple[AgentEvalSummary, List[AgentEvalRow], List[Dict[str, Any]]]:
    examples = load_agent_eval_examples(eval_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    rows, errors, ragas_samples = asyncio.run(
        _run_examples_async(
            examples=examples,
            analyst_model=analyst_model,
            planner_model=planner_model,
            fail_fast=fail_fast,
        )
    )

    valid_rows = [r for r in rows if r.orchestrator_ok]

    det_scores = [float(r.deterministic.score) for r in rows]
    contract_valid_rate = _mean([1.0 if r.deterministic.contract_valid else 0.0 for r in rows])

    def _rate(field: str) -> float:
        vals: List[float] = []
        for r in rows:
            v = getattr(r.deterministic, field)
            if v is None:
                continue
            vals.append(1.0 if bool(v) else 0.0)
        return _mean(vals)

    deterministic_summary: Dict[str, float] = {
        "score_mean": _mean(det_scores),
        "contract_valid_rate": contract_valid_rate,
        "intent_match_rate": _rate("intent_match"),
        "retrieval_needed_match_rate": _rate("retrieval_needed_match"),
        "metadata_match_rate": _rate("metadata_match"),
        "tool_use_match_rate": _rate("tool_use_match"),
        "compute_match_rate": _rate("compute_match"),
        "citation_match_rate": _rate("citation_match"),
    }

    critical_count = sum(1 for r in rows if (r.deterministic.critical_failures or []))
    critical_failure_rate = (float(critical_count) / float(len(rows))) if rows else 0.0
    deterministic_summary["critical_failure_rate"] = critical_failure_rate

    ragas_summary: Dict[str, float] = {}
    ragas_error_count = 0

    if enable_ragas and ragas_samples:
        cfg = ragas_config or RagasAgentConfig()
        per_sample_ragas, ragas_summary, ragas_errors = evaluate_ragas_agents(
            ragas_samples,
            config=cfg,
        )
        ragas_error_count = len(ragas_errors)
        errors.extend(ragas_errors)
        for row in rows:
            if row.id in per_sample_ragas:
                row.ragas = per_sample_ragas[row.id]

    cfg_thresholds = thresholds or AgentEvalThresholds()

    deterministic_gate_pass = (
        deterministic_summary["score_mean"] >= float(cfg_thresholds.deterministic_score_min)
        and critical_failure_rate <= float(cfg_thresholds.max_critical_failure_rate)
    )

    if not enable_ragas:
        ragas_gate_pass = True
    else:
        faithfulness = ragas_summary.get("faithfulness")
        answer_rel = ragas_summary.get("answer_relevancy")
        ragas_error_rate = (float(ragas_error_count) / float(len(ragas_samples))) if ragas_samples else 1.0

        ragas_gate_pass = (
            _is_number(faithfulness)
            and _is_number(answer_rel)
            and float(faithfulness) >= float(cfg_thresholds.ragas_faithfulness_min)
            and float(answer_rel) >= float(cfg_thresholds.ragas_answer_relevancy_min)
            and ragas_error_rate <= float(cfg_thresholds.max_ragas_error_rate)
        )

    gate = AgentGateStatus(
        deterministic_gate_pass=deterministic_gate_pass,
        ragas_gate_pass=ragas_gate_pass,
        overall_pass=bool(deterministic_gate_pass and ragas_gate_pass),
    )

    total_ms = int((time.perf_counter() - start) * 1000)
    summary = AgentEvalSummary(
        num_queries=len(rows),
        num_valid_queries=len(valid_rows),
        num_failures=len(errors),
        deterministic=deterministic_summary,
        ragas=ragas_summary,
        gate=gate,
        config={
            "eval_path": str(eval_path),
            "analyst_model": analyst_model,
            "planner_model": planner_model,
            "enable_ragas": bool(enable_ragas),
            "thresholds": {
                "deterministic_score_min": cfg_thresholds.deterministic_score_min,
                "max_critical_failure_rate": cfg_thresholds.max_critical_failure_rate,
                "ragas_faithfulness_min": cfg_thresholds.ragas_faithfulness_min,
                "ragas_answer_relevancy_min": cfg_thresholds.ragas_answer_relevancy_min,
                "max_ragas_error_rate": cfg_thresholds.max_ragas_error_rate,
            },
            "timing_ms": {
                "total": total_ms,
            },
        },
    )

    _write_jsonl(out_path / "per_query.jsonl", [r.model_dump(mode="json") for r in rows])
    (out_path / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(out_path / "errors.jsonl", errors)

    return summary, rows, errors
