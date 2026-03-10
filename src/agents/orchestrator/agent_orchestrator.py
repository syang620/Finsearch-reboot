"""Multi-agent orchestration entrypoint for planner -> retrieval -> analyst."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from agents.analyst import AnalystAgent, build_packet_from_retrieval_output
from agents.contracts import (
    AnalysisTask,
    AnalystPacket,
    ContextQuality,
    FilingMetadata,
    OpenIssue,
    PlannerIntent,
    Severity,
)
from agents.planner import InteractivePlannerAgent
from agents.retrieval.query_planner import retrieval_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt


class OrchestratorState(TypedDict, total=False):
    user_query: str
    plan_id: str
    analyst_model: str
    tables_dir: str
    debug: bool
    start_time: float

    planner_turn: Dict[str, Any]
    planner_resume_answers: Any
    plan_obj: Dict[str, Any]
    planner_dump: Dict[str, Any]
    planner_timing_ms: Dict[str, int]

    retrieval_state: Dict[str, Any]
    retrieval_output: Dict[str, Any]
    retrieval_timing_ms: Dict[str, int]
    retrieval_skipped_reason: str

    packet: AnalystPacket
    analyst_result: Any

    total_ms: int


_ORCHESTRATOR_CHECKPOINTER = InMemorySaver()
_ORCHESTRATOR_RUN_CONTEXTS: Dict[str, Dict[str, Any]] = {}


def _get_runtime_planner() -> Any:
    planner = (get_config().get("configurable") or {}).get("planner")
    if planner is None:
        raise RuntimeError("Planner instance missing from orchestrator runtime config.")
    return planner


def _graph_config(*, run_id: str, planner: Any) -> Dict[str, Any]:
    return {
        "configurable": {
            "thread_id": run_id,
            "planner": planner,
        }
    }


def _remember_run_context(*, run_id: str, planner: Any) -> None:
    _ORCHESTRATOR_RUN_CONTEXTS[run_id] = {"planner": planner}


def _get_run_context(run_id: str) -> Dict[str, Any]:
    context = _ORCHESTRATOR_RUN_CONTEXTS.get(str(run_id).strip())
    if context is None:
        raise ValueError(f"No resumable orchestrator run found for run_id={run_id!r}.")
    return context


def _forget_run_context(run_id: str) -> None:
    _ORCHESTRATOR_RUN_CONTEXTS.pop(str(run_id).strip(), None)


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _serialize_interrupts(interrupts: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for item in interrupts or ():
        out.append(
            {
                "id": getattr(item, "id", None),
                "value": getattr(item, "value", None),
            }
        )
    return out


def _serialize_analyst_result(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result


def _coerce_intent(plan_obj: Dict[str, Any]) -> PlannerIntent:
    raw = str(plan_obj.get("intent") or PlannerIntent.FILING_FACT.value).strip()
    try:
        return PlannerIntent(raw)
    except Exception:
        return PlannerIntent.FILING_FACT


def _coerce_metadata(plan_obj: Dict[str, Any]) -> FilingMetadata:
    return FilingMetadata.model_validate(plan_obj.get("metadata") or {})


def _coerce_analysis_task(plan_obj: Dict[str, Any]) -> AnalysisTask:
    raw = plan_obj.get("analysis_task") or {}
    if not raw:
        raw = {"task_type": "extract", "metric": "filing evidence"}
    return AnalysisTask.model_validate(raw)


def _coerce_open_issues(plan_obj: Dict[str, Any]) -> list[OpenIssue]:
    issues: list[OpenIssue] = []
    for issue in plan_obj.get("open_issues") or []:
        try:
            issues.append(OpenIssue.model_validate(issue))
        except Exception:
            continue
    return issues


def _build_packet_without_retrieval(*, user_query: str, plan_obj: Dict[str, Any], plan_id: str) -> AnalystPacket:
    return AnalystPacket(
        plan_id=plan_id,
        user_query=user_query,
        intent=_coerce_intent(plan_obj),
        metadata=_coerce_metadata(plan_obj),
        analysis_task=_coerce_analysis_task(plan_obj),
        context_items=[],
        context_quality=ContextQuality.LOW,
        open_issues=_coerce_open_issues(plan_obj),
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


def _build_retrieval_failure_output(*, retrieval_state: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    targets = [
        dict(target)
        for target in (retrieval_state.get("targets") or [])
        if isinstance(target, dict)
    ]
    primary_target = next(
        (
            target
            for target in targets
            if _normalize_text(target.get("ticker")) and _normalize_int(target.get("fiscal_year")) is not None
        ),
        targets[0] if targets else {},
    )
    return {
        "ok": False,
        "queries_used": [],
        "rerank_query": _normalize_text(retrieval_state.get("original_user_query")) or "",
        "top_tables": [],
        "max_total_score": None,
        "metadata_used": {
            "ticker": _normalize_text(primary_target.get("ticker")),
            "fiscal_year": _normalize_int(primary_target.get("fiscal_year")),
            "form_type": _normalize_text(primary_target.get("form_type")),
            "original_user_query": _normalize_text(retrieval_state.get("original_user_query")),
            "targets": targets,
            "retrieval_plan": dict(retrieval_state.get("retrieval_plan") or {}),
            "error_trace": str(exc),
        },
        "error": f"RETRIEVAL_EXECUTION_FAILED: {exc}",
        "trace": {
            "error": str(exc),
            "stage": "retrieval_node",
        },
    }


def _coerce_ms(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _compact_retrieval_attempt(
    *,
    attempt: Dict[str, Any],
    review_feedback: Optional[Dict[str, Any]],
    is_final: bool = False,
) -> Dict[str, Any]:
    request = dict(attempt.get("request") or {})
    retrieval = dict(attempt.get("retrieval") or {})
    retrieval_compact = dict(attempt.get("retrieval_compact") or {})

    reason = _normalize_text(request.get("reason")) or "tool"
    request_reason = str(reason or "").strip().lower()
    query_source = "fallback" if "fallback" in request_reason else "tool"

    request_used = {
        "queries": list(request.get("queries") or []),
        "doc_types": request.get("doc_types"),
        "top_k": request.get("top_k") or 3,
        "min_total_score": request.get("min_total_score", 0.0),
    }

    candidate_rows = list(retrieval_compact.get("top_items") or [])
    if not candidate_rows:
        rows = list(retrieval.get("results") or retrieval.get("top_tables") or [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate_rows.append(
                {
                    "doc_id": row.get("doc_id"),
                    "section_path": row.get("section_path") or row.get("section") or row.get("path"),
                    "doc_type": row.get("doc_type"),
                    "total_score": row.get("total_score") or row.get("score"),
                    "summary": row.get("summary"),
                }
            )

    results = []
    for row in candidate_rows[:3]:
        if not isinstance(row, dict):
            continue
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        score = row.get("score")
        if score is None:
            score = row.get("total_score")
        results.append(
            {
                "doc_id": doc_id,
                "score": score,
            }
        )

    timing_payload = dict((retrieval.get("trace") or {}).get("timing_ms") or {})
    hybrid_retrieval_ms = _coerce_ms(
        timing_payload.get("hybrid_retrieval")
        or timing_payload.get("hybrid_retrieval_ms")
    )
    rerank_ms = _coerce_ms(
        timing_payload.get("rerank")
        or timing_payload.get("rerank_ms")
    )
    total_ms = _coerce_ms(
        timing_payload.get("total")
        or timing_payload.get("total_ms")
        or timing_payload.get("retrieval_total")
        or timing_payload.get("hybrid_plus_rerank")
    )
    if total_ms is None and hybrid_retrieval_ms is not None and rerank_ms is not None:
        total_ms = hybrid_retrieval_ms + rerank_ms

    fallback_review = {
        "action": "accept",
        "reason": _normalize_text(retrieval.get("error")) or "retrieval completed",
    }
    review = dict(fallback_review)
    if review_feedback:
        action = _normalize_text(review_feedback.get("action")) or "accept"
        reason_review = _normalize_text(review_feedback.get("reason")) or fallback_review["reason"]
        review = {"action": action, "reason": reason_review}
    elif is_final:
        action = "accept"
        if _normalize_text(retrieval.get("error")):
            action = "retry"
        review = {
            "action": action,
            "reason": (
                _normalize_text(retrieval.get("error"))
                or "retrieval completed"
            ),
        }

    return {
        "attempt_index": int(attempt.get("attempt_index") or 1),
        "query_source": query_source,
        "request_used": request_used,
        "results": results,
        "review": review,
        "timing_ms": {
            "hybrid_retrieval": hybrid_retrieval_ms,
            "rerank": rerank_ms,
            "total": total_ms,
        },
    }


def _compact_retrieval_result_for_user(*, retrieval_output: Any) -> Dict[str, Any]:
    if not isinstance(retrieval_output, dict):
        return {
            "type": "retrieval",
            "ok": False,
            "original_user_query": None,
            "target": {},
            "attempts": [],
        }

    attempts: List[Dict[str, Any]] = []
    runs = []
    trace = dict(retrieval_output.get("trace") or {})
    trace_runs = [dict(run) for run in trace.get("runs") if isinstance(run, dict)] if isinstance(trace.get("runs"), list) else []
    if trace_runs:
        runs = trace_runs
    elif isinstance(retrieval_output.get("job_runs"), list):
        runs.extend([dict(run) for run in retrieval_output.get("job_runs") if isinstance(run, dict)])

    for run in runs:
        run_attempts = list(run.get("attempts") or [])
        run_review = dict(run.get("review_feedback") or run.get("review") or {})
        review_action = _normalize_text(run_review.get("action"))
        for idx, attempt in enumerate(run_attempts):
            if not isinstance(attempt, dict):
                continue
            is_final = idx == len(run_attempts) - 1
            if is_final and review_action:
                attempt_review = run_review
            elif review_action == "retry" and not is_final:
                attempt_review = run_review
            else:
                attempt_review = None
            attempts.append(
                _compact_retrieval_attempt(
                    attempt=attempt,
                    review_feedback=attempt_review,
                    is_final=is_final,
                )
            )

    targets = [dict(target) for target in (retrieval_output.get("targets") or []) if isinstance(target, dict)]
    target = {}
    primary_target = None
    for candidate in targets:
        if _normalize_text(candidate.get("ticker")) and _normalize_int(candidate.get("fiscal_year")) is not None:
            primary_target = candidate
            break
    if primary_target is None:
        for run in runs:
            run_target = dict(run.get("target") or {})
            if _normalize_text(run_target.get("ticker")) and _normalize_int(run_target.get("fiscal_year")) is not None:
                primary_target = run_target
                break
    if primary_target is None:
        metadata_used = dict(retrieval_output.get("metadata_used") or {})
        if _normalize_text(metadata_used.get("ticker")) and _normalize_int(metadata_used.get("fiscal_year")) is not None:
            primary_target = metadata_used
    if primary_target:
        target = {
            "ticker": _normalize_text(primary_target.get("ticker")),
            "fiscal_year": _normalize_int(primary_target.get("fiscal_year")),
            "form_type": _normalize_text(primary_target.get("form_type")) or None,
        }

    if not target:
        for run in runs:
            if not isinstance(run, dict):
                continue
            metadata_used = dict(run.get("target") or {})
            if metadata_used.get("ticker") and metadata_used.get("fiscal_year") is not None:
                target = {
                    "ticker": _normalize_text(metadata_used.get("ticker")),
                    "fiscal_year": _normalize_int(metadata_used.get("fiscal_year")),
                    "form_type": _normalize_text(metadata_used.get("form_type")) or None,
                }
                break

    if not runs:
        metadata_used = dict(retrieval_output.get("metadata_used") or {})
        if metadata_used.get("ticker") or metadata_used.get("fiscal_year") is not None or metadata_used.get("form_type"):
            target = {
                "ticker": _normalize_text(metadata_used.get("ticker")),
                "fiscal_year": _normalize_int(metadata_used.get("fiscal_year")),
                "form_type": _normalize_text(metadata_used.get("form_type")) or None,
            }

    return {
        "type": "retrieval",
        "ok": bool(retrieval_output.get("ok", False) or bool(retrieval_output.get("top_tables"))),
        "original_user_query": _normalize_text(retrieval_output.get("original_user_query")),
        "target": target,
        "attempts": attempts,
    }


def _init_node(state: OrchestratorState) -> Dict[str, Any]:
    return {
        "start_time": time.perf_counter(),
        "retrieval_timing_ms": {},
        "retrieval_skipped_reason": "",
    }


def _planner_start_node(state: OrchestratorState) -> Dict[str, Any]:
    planner = _get_runtime_planner()
    t0 = time.perf_counter()
    planner_turn = planner.start(state["user_query"])
    planner_dump = dict(planner_turn.get("planner_output") or {})
    planner_timing_ms = dict(state.get("planner_timing_ms") or {})
    planner_timing_ms["planner_start_ms"] = int((time.perf_counter() - t0) * 1000)
    return {
        "planner_turn": planner_turn,
        "plan_obj": planner_dump,
        "planner_dump": planner_dump,
        "planner_timing_ms": planner_timing_ms,
    }


def _planner_interrupt_node(state: OrchestratorState) -> Dict[str, Any]:
    planner_turn = dict(state.get("planner_turn") or {})
    planner_output = dict(planner_turn.get("planner_output") or {})
    answer = interrupt(
        {
            "kind": "planner_clarification",
            "run_id": state["plan_id"],
            "user_query": state["user_query"],
            "clarification_request": planner_turn.get("clarification_request"),
            "planner_state": planner_turn.get("planner_state"),
            "planner_output": planner_output,
            "clarification_history": list(planner_output.get("clarification_history") or []),
        }
    )
    return {"planner_resume_answers": answer}


def _planner_resume_node(state: OrchestratorState) -> Dict[str, Any]:
    planner = _get_runtime_planner()
    t0 = time.perf_counter()
    planner_turn = planner.resume(
        state["planner_turn"],
        state.get("planner_resume_answers"),
    )
    planner_dump = dict(planner_turn.get("planner_output") or {})
    planner_timing_ms = dict(state.get("planner_timing_ms") or {})
    planner_timing_ms["planner_resume_ms"] = int(planner_timing_ms.get("planner_resume_ms", 0)) + int(
        (time.perf_counter() - t0) * 1000
    )
    planner_timing_ms["planner_resume_count"] = int(planner_timing_ms.get("planner_resume_count", 0)) + 1
    return {
        "planner_turn": planner_turn,
        "planner_resume_answers": None,
        "plan_obj": planner_dump,
        "planner_dump": planner_dump,
        "planner_timing_ms": planner_timing_ms,
    }


def _route_after_planner_turn(state: OrchestratorState) -> str:
    plan_obj = state["plan_obj"]
    status = str(plan_obj.get("status") or "").strip().lower()
    if status == "needs_clarification":
        return "planner_interrupt"
    return "check_retrieval_metadata" if status == "completed" and bool(plan_obj.get("retrieval_needed")) else "build_packet_without_retrieval"


def _check_retrieval_metadata_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    targets = [
        dict(target)
        for target in (plan_obj.get("targets") or [])
        if isinstance(target, dict)
    ]
    valid_targets = [
        target
        for target in targets
        if target.get("ticker") and target.get("fiscal_year") is not None
    ]
    retrieval_plan = dict(plan_obj.get("retrieval_plan") or {})

    if not retrieval_plan:
        return {"retrieval_state": {}, "retrieval_skipped_reason": "MISSING_RETRIEVAL_PLAN"}
    if not valid_targets:
        return {"retrieval_state": {}, "retrieval_skipped_reason": "MISSING_TARGET_METADATA"}

    retrieval_state = {
        "targets": targets,
        "retrieval_plan": retrieval_plan,
        "original_user_query": plan_obj.get("original_user_query") or state["user_query"],
        "clarification_history": list(plan_obj.get("clarification_history") or []),
    }
    return {"retrieval_state": retrieval_state, "retrieval_skipped_reason": ""}


def _route_after_retrieval_metadata(state: OrchestratorState) -> str:
    retrieval_state = state.get("retrieval_state") or {}
    return "retrieval" if retrieval_state else "build_packet_without_retrieval"


async def _retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    t_ret = time.perf_counter()
    try:
        ret_state = await retrieval_agent(state["retrieval_state"])
        retrieval_output = ret_state.get("retrieval")
    except Exception as exc:
        retrieval_output = _build_retrieval_failure_output(
            retrieval_state=state["retrieval_state"],
            exc=exc,
        )
    retrieval_timing_ms = dict(state.get("retrieval_timing_ms") or {})
    retrieval_timing_ms["retrieve_ms"] = int((time.perf_counter() - t_ret) * 1000)
    return {
        "retrieval_output": retrieval_output,
        "retrieval_timing_ms": retrieval_timing_ms,
    }


def _build_packet_from_retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    retrieval_output = state.get("retrieval_output")
    analysis_task = _coerce_analysis_task(plan_obj)
    packet = build_packet_from_retrieval_output(
        user_query=state["user_query"],
        retrieval_output=retrieval_output or {},
        tables_dir=state["tables_dir"],
        plan_id=state["plan_id"],
        intent=_coerce_intent(plan_obj),
        analysis_task=analysis_task,
        max_tables=3,
    )

    # Carry planner issues into analyst packet.
    packet.open_issues = _coerce_open_issues(plan_obj) + list(packet.open_issues)

    return {"packet": packet}


def _build_packet_without_retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    packet = _build_packet_without_retrieval(
        user_query=state["user_query"],
        plan_obj=plan_obj,
        plan_id=state["plan_id"],
    )
    if str(plan_obj.get("status") or "").strip().lower() == "needs_clarification":
        packet.open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_CLARIFICATION_REQUIRED",
                message="Planner requested clarification before retrieval could proceed.",
                severity=Severity.WARNING,
            )
        )
    elif plan_obj.get("retrieval_needed") and state.get("retrieval_skipped_reason") in {"MISSING_METADATA", "MISSING_TARGET_METADATA", "MISSING_RETRIEVAL_PLAN"}:
        packet.open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_MISSING_METADATA",
                message="Retrieval was required but retrieval metadata or target resolution was incomplete.",
                severity=Severity.WARNING,
            )
        )
    elif not plan_obj.get("retrieval_needed"):
        packet.open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_BY_PLANNER",
                message="Planner set retrieval_needed=False; analyst ran without retrieved filing context.",
                severity=Severity.INFO,
            )
        )
    return {"packet": packet}


async def _analyst_node(state: OrchestratorState) -> Dict[str, Any]:
    analyst = AnalystAgent(
        model=state["analyst_model"],
        max_context_items=5,
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
    builder.add_node("planner_start", _planner_start_node)
    builder.add_node("planner_interrupt", _planner_interrupt_node)
    builder.add_node("planner_resume", _planner_resume_node)
    builder.add_node("check_retrieval_metadata", _check_retrieval_metadata_node)
    builder.add_node("retrieval", _retrieval_node)
    builder.add_node("build_packet_from_retrieval", _build_packet_from_retrieval_node)
    builder.add_node("build_packet_without_retrieval", _build_packet_without_retrieval_node)
    builder.add_node("analyst", _analyst_node)
    builder.add_node("finalize", _finalize_node)

    builder.set_entry_point("init")
    builder.add_edge("init", "planner_start")
    builder.add_conditional_edges(
        "planner_start",
        _route_after_planner_turn,
        {
            "planner_interrupt": "planner_interrupt",
            "check_retrieval_metadata": "check_retrieval_metadata",
            "build_packet_without_retrieval": "build_packet_without_retrieval",
        },
    )
    builder.add_edge("planner_interrupt", "planner_resume")
    builder.add_conditional_edges(
        "planner_resume",
        _route_after_planner_turn,
        {
            "planner_interrupt": "planner_interrupt",
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
    return builder.compile(checkpointer=_ORCHESTRATOR_CHECKPOINTER)


def _format_run_output(
    *,
    run_id: str,
    state_snapshot: Any,
) -> Dict[str, Any]:
    state_values = dict(getattr(state_snapshot, "values", {}) or {})
    interrupted = bool(getattr(state_snapshot, "interrupts", ()) or ())
    start_time = state_values.get("start_time")
    total_ms = state_values.get("total_ms")
    if total_ms is None and isinstance(start_time, (int, float)):
        total_ms = int((time.perf_counter() - start_time) * 1000)

    out = {
        "run_id": run_id,
        "status": "interrupted" if interrupted else "completed",
        "planner": state_values.get("planner_dump"),
        "planner_turn": state_values.get("planner_turn"),
        "retrieval": _compact_retrieval_result_for_user(
            retrieval_output=state_values.get("retrieval_output"),
        ),
        "analyst": _serialize_analyst_result(state_values.get("analyst_result")),
        "interrupt": _serialize_interrupts(getattr(state_snapshot, "interrupts", ()) or ()),
        "orchestrator_trace": {
            "total_ms": total_ms,
            "planner_timing_ms": dict(state_values.get("planner_timing_ms") or {}),
            "retrieval_timing_ms": dict(state_values.get("retrieval_timing_ms") or {}),
        },
    }
    return out


async def _invoke_orchestrator(
    payload: Any,
    *,
    run_id: str,
    planner: Any,
) -> Dict[str, Any]:
    graph = _get_orchestrator_graph()
    config = _graph_config(run_id=run_id, planner=planner)
    try:
        await graph.ainvoke(payload, config=config)
    except Exception:
        _forget_run_context(run_id)
        raise
    state_snapshot = graph.get_state(config)
    output = _format_run_output(run_id=run_id, state_snapshot=state_snapshot)
    if output["status"] == "interrupted":
        return output
    _forget_run_context(run_id)
    return output


async def run_multi_agent_orchestration(
    user_query: str,
    *,
    planner: Optional[Any] = None,
    analyst_model: str = "qwen2.5-14b-instruct-1m",
    tables_dir: str = "data/chunked",
    debug: bool = True,
) -> Dict[str, Any]:
    plan_id = f"run-{uuid.uuid4().hex[:8]}"
    resolved_tables_dir = _resolve_tables_dir(tables_dir)
    runtime_planner = planner or InteractivePlannerAgent(log_timing=False)
    _remember_run_context(run_id=plan_id, planner=runtime_planner)
    return await _invoke_orchestrator(
        {
            "user_query": user_query,
            "plan_id": plan_id,
            "analyst_model": analyst_model,
            "tables_dir": resolved_tables_dir,
            "debug": debug,
        },
        run_id=plan_id,
        planner=runtime_planner,
    )


async def resume_multi_agent_orchestration(
    run_id: str,
    answers: Any,
) -> Dict[str, Any]:
    context = _get_run_context(run_id)
    return await _invoke_orchestrator(
        Command(resume=answers),
        run_id=str(run_id).strip(),
        planner=context["planner"],
    )
