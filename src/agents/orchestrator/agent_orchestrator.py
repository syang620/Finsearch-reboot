"""Multi-agent orchestration entrypoint for planner -> retrieval -> analyst."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time
import uuid
from typing import Any, Dict, Optional, TypedDict

from agents.analyst import AnalystAgent, build_packet_from_retrieval_output
from agents.contracts import (
    AnalystPacket,
    ContextQuality,
    OpenIssue,
    PlannerOutput,
    Severity,
)
from agents.planner.agent import PlannerAgent
from agents.retrieval.agent import retrieval_agent
from agents.retrieval.mcp_client import SecRetrievalMCPClient
from langgraph.graph import END, StateGraph


class OrchestratorState(TypedDict, total=False):
    user_query: str
    plan_id: str
    planner: PlannerAgent
    analyst_model: str
    tables_dir: str
    debug: bool
    start_time: float

    plan_obj: PlannerOutput
    planner_dump: Dict[str, Any]
    planner_timing_ms: Dict[str, int]

    retrieval_state: Dict[str, Any]
    retrieval_output: Dict[str, Any]
    retrieval_timing_ms: Dict[str, int]
    retrieval_skipped_reason: str

    packet: AnalystPacket
    analyst_result: Any

    total_ms: int


def _build_packet_without_retrieval(*, user_query: str, plan_obj: PlannerOutput, plan_id: str) -> AnalystPacket:
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


def _resolve_tables_dir(tables_dir: str) -> str:
    raw = str(tables_dir or "").strip()
    repo_root = Path(__file__).resolve().parents[3]

    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(repo_root / p)

    candidates.append(repo_root / "data" / "chunked")
    candidates.append(Path("../data/chunked"))
    candidates.append(Path("data/chunked"))

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_dir():
            return str(c.resolve())

    fallback = candidates[0] if candidates else (repo_root / "data" / "chunked")
    return str(fallback.resolve())


def _init_node(state: OrchestratorState) -> Dict[str, Any]:
    return {
        "start_time": time.perf_counter(),
        "retrieval_timing_ms": {},
        "retrieval_skipped_reason": "",
    }


def _planner_node(state: OrchestratorState) -> Dict[str, Any]:
    planner = state["planner"]
    plan_obj = planner.plan(state["user_query"], include_trace=False)
    return {
        "plan_obj": plan_obj,
        "planner_dump": plan_obj.model_dump(mode="json"),
        "planner_timing_ms": dict(planner.last_timing_ms),
    }


def _route_after_planner(state: OrchestratorState) -> str:
    plan_obj = state["plan_obj"]
    return "check_retrieval_metadata" if plan_obj.retrieval_needed else "build_packet_without_retrieval"


def _check_retrieval_metadata_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    ticker = plan_obj.metadata.ticker
    fiscal_year = plan_obj.metadata.fiscal_year

    if not ticker or fiscal_year is None:
        return {"retrieval_state": {}, "retrieval_skipped_reason": "MISSING_METADATA"}

    retrieval_state = {
        "queries": list(plan_obj.query_bundle.queries),
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "form_type": plan_obj.metadata.form_type.value,
        "doc_types": plan_obj.metadata.doc_types,
    }
    return {"retrieval_state": retrieval_state, "retrieval_skipped_reason": ""}


def _route_after_retrieval_metadata(state: OrchestratorState) -> str:
    retrieval_state = state.get("retrieval_state") or {}
    return "retrieval" if retrieval_state else "build_packet_without_retrieval"


async def _retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    t_ret = time.perf_counter()
    async with SecRetrievalMCPClient() as client:
        ret_state = await retrieval_agent(state["retrieval_state"], client)
    retrieval_timing_ms = dict(state.get("retrieval_timing_ms") or {})
    retrieval_timing_ms["retrieve_ms"] = int((time.perf_counter() - t_ret) * 1000)
    return {
        "retrieval_output": ret_state.get("retrieval"),
        "retrieval_timing_ms": retrieval_timing_ms,
    }


def _build_packet_from_retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    retrieval_output = state.get("retrieval_output")
    packet = build_packet_from_retrieval_output(
        user_query=state["user_query"],
        retrieval_output=retrieval_output or {},
        tables_dir=state["tables_dir"],
        plan_id=state["plan_id"],
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
    min_score = plan_obj.quality_requirements.min_total_score  # min score from planner is not useful
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

    return {"packet": packet}


def _build_packet_without_retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    packet = _build_packet_without_retrieval(
        user_query=state["user_query"],
        plan_obj=plan_obj,
        plan_id=state["plan_id"],
    )
    if plan_obj.retrieval_needed and state.get("retrieval_skipped_reason") == "MISSING_METADATA":
        packet.open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_MISSING_METADATA",
                message="Retrieval was required but ticker/fiscal_year is missing.",
                severity=Severity.WARNING,
            )
        )
    elif not plan_obj.retrieval_needed:
        packet.open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_BY_PLANNER",
                message="Planner set retrieval_needed=False; analyst ran without retrieved filing context.",
                severity=Severity.INFO,
            )
        )
    return {"packet": packet}


async def _analyst_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    analyst = AnalystAgent(
        model=state["analyst_model"],
        max_context_items=plan_obj.quality_requirements.max_context_items,
    )
    await analyst.abuild()
    try:
        analyst_result = await analyst.arun(state["packet"], debug=state["debug"])
    finally:
        await analyst.aclose()
    return {"analyst_result": analyst_result}


def _finalize_node(state: OrchestratorState) -> Dict[str, Any]:
    total_ms = int((time.perf_counter() - state["start_time"]) * 1000)
    return {"total_ms": total_ms}


@lru_cache(maxsize=1)
def _get_orchestrator_graph():
    builder = StateGraph(OrchestratorState)
    builder.add_node("init", _init_node)
    builder.add_node("planner", _planner_node)
    builder.add_node("check_retrieval_metadata", _check_retrieval_metadata_node)
    builder.add_node("retrieval", _retrieval_node)
    builder.add_node("build_packet_from_retrieval", _build_packet_from_retrieval_node)
    builder.add_node("build_packet_without_retrieval", _build_packet_without_retrieval_node)
    builder.add_node("analyst", _analyst_node)
    builder.add_node("finalize", _finalize_node)

    builder.set_entry_point("init")
    builder.add_edge("init", "planner")
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "check_retrieval_metadata": "check_retrieval_metadata",
            "build_packet_without_retrieval": "build_packet_without_retrieval",
        },
    )
    builder.add_conditional_edges(
        "check_retrieval_metadata",
        _route_after_retrieval_metadata,
        {
            "retrieval": "retrieval",
            "build_packet_without_retrieval": "build_packet_without_retrieval",
        },
    )
    builder.add_edge("retrieval", "build_packet_from_retrieval")
    builder.add_edge("build_packet_from_retrieval", "analyst")
    builder.add_edge("build_packet_without_retrieval", "analyst")
    builder.add_edge("analyst", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


async def run_multi_agent_orchestration(
    user_query: str,
    *,
    planner: Optional[PlannerAgent] = None,
    analyst_model: str = "qwen3:14b",
    tables_dir: str = "data/chunked",
    debug: bool = True,
) -> Dict[str, Any]:
    plan_id = f"run-{uuid.uuid4().hex[:8]}"
    resolved_tables_dir = _resolve_tables_dir(tables_dir)
    graph = _get_orchestrator_graph()
    final_state = await graph.ainvoke(
        {
            "user_query": user_query,
            "plan_id": plan_id,
            "planner": planner or PlannerAgent(),
            "analyst_model": analyst_model,
            "tables_dir": resolved_tables_dir,
            "debug": debug,
        }
    )

    return {
        "run_id": plan_id,
        "planner": final_state["planner_dump"],
        "retrieval": final_state.get("retrieval_output"),
        "analyst": final_state["analyst_result"].model_dump(mode="json"),
        "orchestrator_trace": {
            "total_ms": final_state["total_ms"],
            "planner_timing_ms": dict(final_state.get("planner_timing_ms") or {}),
            "retrieval_timing_ms": dict(final_state.get("retrieval_timing_ms") or {}),
        },
    }
