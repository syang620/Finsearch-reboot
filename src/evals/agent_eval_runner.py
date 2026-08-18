from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from agents.orchestrator.agent_orchestrator import run_multi_agent_orchestration
from agents.planner import InteractivePlannerAgent
from evals.agent_eval_contracts import AgentEvalSummary, AgentGateStatus
from evals.ragas_agent_metrics import RagasAgentConfig, evaluate_ragas_agents


AgentEvalMode = Literal["legacy_v0", "route_aware_v1"]


@dataclass(frozen=True)
class AgentEvalThresholds:
    deterministic_score_min: float = 0.90
    max_critical_failure_rate: float = 0.05
    ragas_faithfulness_min: float = 0.75
    ragas_answer_relevancy_min: float = 0.75
    max_ragas_error_rate: float = 0.10


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except Exception:
        return False
    return not math.isnan(numeric)


def _get_mode_handler(mode: AgentEvalMode) -> Any:
    if mode == "legacy_v0":
        from evals import agent_eval_legacy_v0

        return agent_eval_legacy_v0
    if mode == "route_aware_v1":
        from evals import agent_eval_route_aware_v1

        return agent_eval_route_aware_v1
    raise ValueError(f"Unsupported agent evaluation mode: {mode}")


async def _run_examples_async(
    *,
    examples: Sequence[Any],
    analyst_model: str,
    planner_model: Optional[str],
    fail_fast: bool,
    mode_handler: Any,
) -> Tuple[List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    planner = (
        InteractivePlannerAgent(model=planner_model, log_timing=False)
        if planner_model
        else InteractivePlannerAgent(log_timing=False)
    )
    rows: List[Any] = []
    errors: List[Dict[str, Any]] = []
    ragas_samples: List[Dict[str, Any]] = []

    for example in examples:
        try:
            run_output = await run_multi_agent_orchestration(
                example.user_query,
                planner=planner,
                analyst_model=analyst_model,
                debug=False,
                include_evidence_trace=bool(
                    mode_handler.INCLUDE_EVIDENCE_TRACE
                ),
            )
            row, row_errors, ragas_sample = mode_handler.evaluate_run_output(
                example,
                run_output,
            )
        except Exception as exc:
            row, row_errors, ragas_sample = mode_handler.build_orchestrator_error(
                example,
                exc,
            )

        rows.append(row)
        errors.extend(row_errors)
        if ragas_sample is not None:
            ragas_samples.append(ragas_sample)
        if fail_fast and row_errors:
            break

    return rows, errors, ragas_samples


def run_agent_eval(
    *,
    eval_path: str,
    out_dir: str,
    analyst_model: str = "qwen3:14b",
    planner_model: Optional[str] = None,
    enable_ragas: Optional[bool] = None,
    ragas_config: Optional[RagasAgentConfig] = None,
    thresholds: Optional[AgentEvalThresholds] = None,
    fail_fast: bool = False,
    mode: AgentEvalMode = "legacy_v0",
) -> Tuple[AgentEvalSummary, List[Any], List[Dict[str, Any]]]:
    mode_handler = _get_mode_handler(mode)
    ragas_enabled = (
        mode == "legacy_v0" if enable_ragas is None else bool(enable_ragas)
    )
    examples = mode_handler.load_agent_eval_examples(eval_path)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows, errors, ragas_samples = asyncio.run(
        _run_examples_async(
            examples=examples,
            analyst_model=analyst_model,
            planner_model=planner_model,
            fail_fast=fail_fast,
            mode_handler=mode_handler,
        )
    )
    deterministic_summary = mode_handler.deterministic_summary(rows)

    ragas_summary: Dict[str, float] = {}
    ragas_error_count = 0
    if ragas_enabled and ragas_samples:
        ragas_settings = ragas_config or RagasAgentConfig()
        per_sample_ragas, ragas_summary, ragas_errors = evaluate_ragas_agents(
            ragas_samples,
            config=ragas_settings,
        )
        ragas_error_count = len(ragas_errors)
        errors.extend(ragas_errors)
        for row in rows:
            if row.id in per_sample_ragas:
                row.ragas = per_sample_ragas[row.id]

    configured_thresholds = thresholds
    if configured_thresholds is None:
        configured_thresholds = (
            AgentEvalThresholds(max_critical_failure_rate=0.0)
            if mode == "route_aware_v1"
            else AgentEvalThresholds()
        )

    critical_failure_rate = float(
        deterministic_summary.get("critical_failure_rate", 0.0)
    )
    deterministic_gate_pass = (
        float(deterministic_summary.get("score_mean", 0.0))
        >= configured_thresholds.deterministic_score_min
        and critical_failure_rate
        <= configured_thresholds.max_critical_failure_rate
    )

    if not ragas_enabled:
        ragas_gate_pass = True
    else:
        faithfulness = ragas_summary.get("faithfulness")
        answer_relevancy = ragas_summary.get("answer_relevancy")
        ragas_error_rate = (
            float(ragas_error_count) / float(len(ragas_samples))
            if ragas_samples
            else 1.0
        )
        ragas_gate_pass = (
            _is_number(faithfulness)
            and _is_number(answer_relevancy)
            and float(faithfulness)
            >= configured_thresholds.ragas_faithfulness_min
            and float(answer_relevancy)
            >= configured_thresholds.ragas_answer_relevancy_min
            and ragas_error_rate <= configured_thresholds.max_ragas_error_rate
        )

    total_ms = int((time.perf_counter() - started) * 1000)
    gate = AgentGateStatus(
        deterministic_gate_pass=deterministic_gate_pass,
        ragas_gate_pass=ragas_gate_pass,
        overall_pass=bool(deterministic_gate_pass and ragas_gate_pass),
    )
    summary = AgentEvalSummary(
        num_queries=len(rows),
        num_valid_queries=sum(1 for row in rows if row.orchestrator_ok),
        num_failures=len(errors),
        deterministic=deterministic_summary,
        ragas=ragas_summary,
        gate=gate,
        config=mode_handler.build_summary_config(
            eval_path=eval_path,
            analyst_model=analyst_model,
            planner_model=planner_model,
            enable_ragas=ragas_enabled,
            thresholds=configured_thresholds,
            total_ms=total_ms,
        ),
    )

    _write_jsonl(
        output_path / "per_query.jsonl",
        [row.model_dump(mode="json") for row in rows],
    )
    (output_path / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(output_path / "errors.jsonl", errors)
    return summary, rows, errors
