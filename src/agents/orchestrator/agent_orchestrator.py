"""Multi-agent orchestration entrypoint for planner -> retrieval -> analyst."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from agents.analyst import AnalystAgent, build_packet_from_retrieval_output
from agents.contracts import AnalystPacket, ContextQuality, OpenIssue, Severity
from agents.planner.agent import PlannerAgent
from agents.retrieval.agent import retrieval_agent
from agents.retrieval.mcp_client import SecRetrievalMCPClient


def _build_packet_without_retrieval(*, user_query: str, plan_obj: Any, plan_id: str) -> AnalystPacket:
    return AnalystPacket(
        plan_id=plan_id,
        user_query=user_query,
        intent=plan_obj.intent,
        metadata=plan_obj.metadata,
        analysis_task=plan_obj.analysis_task,
        context_items=[],
        context_quality=ContextQuality.LOW,
        open_issues=list(plan_obj.open_issues),
    )


async def run_multi_agent_orchestration(
    user_query: str,
    *,
    planner: Optional[PlannerAgent] = None,
    analyst_model: str = "qwen3:14b",
    tables_dir: str = "../data/chunked",
    debug: bool = True,
) -> Dict[str, Any]:
    t_total = time.perf_counter()
    plan_id = f"run-{uuid.uuid4().hex[:8]}"

    planner = planner or PlannerAgent()
    plan_obj = planner.plan(user_query, include_trace=False)

    retrieval_output: Optional[Dict[str, Any]] = None
    retrieval_timing_ms: Dict[str, int] = {}

    if plan_obj.retrieval_needed:
        ticker = plan_obj.metadata.ticker
        fiscal_year = plan_obj.metadata.fiscal_year

        if not ticker or fiscal_year is None:
            packet = _build_packet_without_retrieval(
                user_query=user_query,
                plan_obj=plan_obj,
                plan_id=plan_id,
            )
            packet.open_issues.append(
                OpenIssue(
                    code="RETRIEVAL_SKIPPED_MISSING_METADATA",
                    message="Retrieval was required but ticker/fiscal_year is missing.",
                    severity=Severity.WARNING,
                )
            )
        else:
            state = {
                "queries": list(plan_obj.query_bundle.queries),
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "form_type": plan_obj.metadata.form_type.value,
                "doc_types": plan_obj.metadata.doc_types,
            }

            t_ret = time.perf_counter()
            async with SecRetrievalMCPClient() as client:
                ret_state = await retrieval_agent(state, client)
            retrieval_timing_ms["retrieve_ms"] = int((time.perf_counter() - t_ret) * 1000)
            retrieval_output = ret_state.get("retrieval")

            packet = build_packet_from_retrieval_output(
                user_query=user_query,
                retrieval_output=retrieval_output or {},
                tables_dir=tables_dir,
                plan_id=plan_id,
                intent=plan_obj.intent,
                metric=plan_obj.analysis_task.metric,
                max_tables=min(plan_obj.quality_requirements.max_context_items, 5),
            )

            # Carry planner issues into analyst packet.
            packet.open_issues = list(plan_obj.open_issues) + list(packet.open_issues)

            # Minimal quality gate hook (single-pass): add issue if retrieval is below planner thresholds.
            top_tables = (retrieval_output or {}).get("top_tables", []) if isinstance(retrieval_output, dict) else []
            got_results = len(top_tables)
            max_score = (retrieval_output or {}).get("max_total_score") if isinstance(retrieval_output, dict) else None
            min_results = plan_obj.quality_requirements.min_results
            min_score = plan_obj.quality_requirements.min_total_score
            quality_ok = got_results >= min_results and (max_score is None or max_score >= min_score)
            if not quality_ok:
                packet.open_issues.append(
                    OpenIssue(
                        code="RETRIEVAL_QUALITY_LOW",
                        message=(
                            f"Retrieval below threshold: results={got_results} (min={min_results}), "
                            f"max_total_score={max_score} (min={min_score})."
                        ),
                        severity=Severity.WARNING,
                    )
                )
    else:
        packet = _build_packet_without_retrieval(
            user_query=user_query,
            plan_obj=plan_obj,
            plan_id=plan_id,
        )
        packet.open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_BY_PLANNER",
                message="Planner set retrieval_needed=False; analyst ran without retrieved filing context.",
                severity=Severity.INFO,
            )
        )

    analyst = AnalystAgent(
        model=analyst_model,
        max_context_items=plan_obj.quality_requirements.max_context_items,
    )
    await analyst.abuild()
    try:
        analyst_result = await analyst.arun(packet, debug=debug)
    finally:
        await analyst.aclose()

    total_ms = int((time.perf_counter() - t_total) * 1000)

    return {
        "run_id": plan_id,
        "planner": plan_obj.model_dump(mode="json"),
        "retrieval": retrieval_output,
        "analyst": analyst_result.model_dump(mode="json"),
        "orchestrator_trace": {
            "total_ms": total_ms,
            "planner_timing_ms": dict(planner.last_timing_ms),
            "retrieval_timing_ms": retrieval_timing_ms,
        },
    }

