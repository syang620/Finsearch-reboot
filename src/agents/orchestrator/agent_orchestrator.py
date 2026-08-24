"""Multi-agent orchestration entrypoint for planner -> retrieval -> analyst."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
import json
import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
import operator
import time
import uuid
from collections import OrderedDict
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from agents.analyst import AnalystAgent, AnalystRunResult, build_packet_from_retrieval_output
from agents.contracts import (
    AnalysisTask,
    AnalystPacket,
    ContextItem,
    ContextItemKind,
    ContextQuality,
    FilingMetadata,
    FormType,
    OpenIssue,
    PlannerIntent,
    PlannerRuntimeOutput,
    Severity,
    SourceRef,
)
from agents.planner import InteractivePlannerAgent
from agents.planner.interactive_target_resolution import (
    _dedupe_ints,
    _normalize_clarification_turns,
)
from agents.retrieval.query_planner_v2 import retrieval_agent
from agents.text_utils import normalize_text
from mcp_server.tools.sec_metric_registry import METRIC_REGISTRY
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_config
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError


logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict, total=False):
    user_query: str
    plan_id: str
    analyst_model: str
    tables_dir: str
    debug: bool
    include_evidence_trace: bool
    start_time: float
    clarification_turns: Annotated[list[Dict[str, str]], operator.add]
    open_issues: Annotated[list[Dict[str, Any]], _dedupe_open_issue_payloads]

    planner_turn: Dict[str, Any]
    planner_resume_answers: Any
    plan_obj: Dict[str, Any]
    planner_dump: Dict[str, Any]
    planner_timing_ms: Dict[str, int]

    retrieval_state: Dict[str, Any]
    retrieval_output: Dict[str, Any]
    retrieval_timing_ms: Dict[str, int]
    retrieval_skipped_reason: str
    structured_fact_results: List[Dict[str, Any]]
    structured_fact_timing_ms: Dict[str, int]

    packet: AnalystPacket
    analyst_result: Any

    total_ms: int
def _orchestrator_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return Path(
        os.getenv(
            "FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH",
            str(repo_root / "artifacts" / "orchestrator_checkpointer.sqlite"),
        )
    )


def _orchestrator_checkpoint_ttl_seconds() -> int:
    raw = os.getenv("FINSEARCH_ORCHESTRATOR_CHECKPOINTER_TTL_SECONDS", "").strip()
    if not raw:
        return 7 * 24 * 60 * 60
    try:
        return max(0, int(raw))
    except Exception:
        return 7 * 24 * 60 * 60


_ORCHESTRATOR_CHECKPOINTER: Optional[AsyncSqliteSaver] = None
_ORCHESTRATOR_CHECKPOINTER_OWNER: Optional[
    AbstractAsyncContextManager[AsyncSqliteSaver]
] = None
_ORCHESTRATOR_CHECKPOINTER_LOCK: Optional[asyncio.Lock] = None
_ORCHESTRATOR_CHECKPOINTER_CLOSING = False
_ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK: Optional[asyncio.Task[None]] = None
_ANALYST_CACHE: "OrderedDict[str, AnalystAgent]" = OrderedDict()
_ANALYST_BUILD_LOCKS: Dict[str, asyncio.Lock] = {}
_ANALYST_DEFAULT_MODEL = "qwen2.5-14b-instruct-1m"
_ANALYST_MAX_CONTEXT_ITEMS = 5
_ANALYST_CACHE_MAX_SIZE = 4
_ORCHESTRATOR_LAST_PRUNE_TS = 0.0
_ORCHESTRATOR_MCP_CLIENT: Optional[Any] = None
_ORCHESTRATOR_MCP_CLIENT_LOCK: Optional[asyncio.Lock] = None
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_TICKER_LIKE_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
# Keep aliases conservative. Broad single-word finance terms are intentionally
# avoided here to reduce false positives in metric auto-resolution.
_STRUCTURED_FACT_ALIAS_MAP: Dict[str, tuple[str, ...]] = {
    "total_debt": (
        "total debt",
        "interest bearing debt",
        "borrowings",
    ),
    "revenue": ("revenue", "sales", "total revenue"),
    "gross_profit": ("gross profit", "gross earnings"),
    "operating_income": ("operating income", "operating profit", "ebit"),
    "net_income": ("net income", "net earnings"),
    "cash_and_cash_equivalents": (
        "cash and cash equivalents",
        "cash equivalents",
        "cash",
    ),
    "total_assets": ("total assets",),
    "total_liabilities": ("total liabilities",),
    "stockholders_equity": (
        "stockholders equity",
        "shareholders equity",
    ),
    "operating_cash_flow": (
        "operating cash flow",
        "cash flow from operations",
        "cash from operations",
        "cfo",
        "cash",
    ),
    "capex": ("capex", "capital expenditures", "capital expenditure"),
}


def _observe_background_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Checkpoint-pruning background task failed")


async def _get_orchestrator_checkpointer() -> AsyncSqliteSaver:
    global _ORCHESTRATOR_CHECKPOINTER
    global _ORCHESTRATOR_CHECKPOINTER_OWNER
    global _ORCHESTRATOR_CHECKPOINTER_LOCK
    if _ORCHESTRATOR_CHECKPOINTER_LOCK is None:
        _ORCHESTRATOR_CHECKPOINTER_LOCK = asyncio.Lock()

    async with _ORCHESTRATOR_CHECKPOINTER_LOCK:
        if _ORCHESTRATOR_CHECKPOINTER_CLOSING:
            raise RuntimeError("Orchestrator checkpointer lifecycle is closing.")
        if _ORCHESTRATOR_CHECKPOINTER is not None:
            return _ORCHESTRATOR_CHECKPOINTER

        path = _orchestrator_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        owner = AsyncSqliteSaver.from_conn_string(str(path))
        entered = False
        try:
            saver = await owner.__aenter__()
            entered = True
            await saver.conn.execute("PRAGMA journal_mode=WAL;")
            await saver.conn.execute("PRAGMA busy_timeout = 5000;")
            await saver.conn.commit()
            await saver.setup()
        except BaseException:
            exc_info = sys.exc_info()
            if entered:
                try:
                    await owner.__aexit__(*exc_info)
                except BaseException:
                    logger.exception(
                        "Failed to close orchestrator checkpointer after initialization error"
                    )
            raise

        _ORCHESTRATOR_CHECKPOINTER_OWNER = owner
        _ORCHESTRATOR_CHECKPOINTER = saver
        return saver


async def _prune_stale_orchestrator_runs(*, max_age_seconds: int) -> int:
    if not max_age_seconds:
        return 0

    saver = await _get_orchestrator_checkpointer()
    cutoff = time.time() - max_age_seconds
    stale_threads: set[str] = set()

    async with saver.conn.execute(
        """
        SELECT thread_id
        FROM checkpoints
        WHERE CAST(json_extract(CAST(metadata AS TEXT), '$.orchestrator_started_at') AS REAL) < ?
        """,
        (cutoff,),
    ) as cursor:
        rows = await cursor.fetchall()

    for thread_id, in rows:
        if thread_id is not None:
            stale_threads.add(str(thread_id))

    for thread_id in sorted(stale_threads):
        await _delete_thread_checkpoints(saver=saver, thread_id=thread_id)
    return len(stale_threads)


async def _delete_thread_checkpoints(*, saver: AsyncSqliteSaver, thread_id: str) -> None:
    if not thread_id:
        return
    for table in ("checkpoint_blobs", "checkpoint_writes", "checkpoints", "writes"):
        try:
            await saver.conn.execute(
                f"DELETE FROM {table} WHERE thread_id=?",
                (thread_id,),
            )
        except Exception:
            continue
    try:
        await saver.conn.commit()
    except Exception:
        pass


async def _get_orchestrator_mcp_client() -> Any:
    global _ORCHESTRATOR_MCP_CLIENT, _ORCHESTRATOR_MCP_CLIENT_LOCK
    if _ORCHESTRATOR_MCP_CLIENT is not None:
        if await _orchestrator_mcp_client_is_usable(_ORCHESTRATOR_MCP_CLIENT):
            return _ORCHESTRATOR_MCP_CLIENT
        await _reset_orchestrator_mcp_client(_ORCHESTRATOR_MCP_CLIENT)

    if _ORCHESTRATOR_MCP_CLIENT_LOCK is None:
        _ORCHESTRATOR_MCP_CLIENT_LOCK = asyncio.Lock()

    async with _ORCHESTRATOR_MCP_CLIENT_LOCK:
        if _ORCHESTRATOR_MCP_CLIENT is not None:
            if await _orchestrator_mcp_client_is_usable(_ORCHESTRATOR_MCP_CLIENT):
                return _ORCHESTRATOR_MCP_CLIENT
            await _reset_orchestrator_mcp_client(
                _ORCHESTRATOR_MCP_CLIENT,
                _skip_lock=True,
            )

        from agents.retrieval.mcp_client import SecRetrievalMCPClient

        client = SecRetrievalMCPClient()
        await client.__aenter__()
        _ORCHESTRATOR_MCP_CLIENT = client
        return _ORCHESTRATOR_MCP_CLIENT


async def _orchestrator_mcp_client_is_usable(client: Optional[Any]) -> bool:
    if client is None:
        return False

    session = getattr(client, "_session", None)
    if (
        session is None
        or getattr(client, "_stdio_cm", None) is None
        or getattr(client, "_read", None) is None
        or getattr(client, "_write", None) is None
        or not callable(getattr(session, "call_tool", None))
    ):
        return False
    return True


def _is_mcp_transport_error(message: Any) -> bool:
    text = _normalize_text(message)
    if not text:
        return False
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "eof",
            "broken pipe",
            "pipe",
            "closed",
            "timed out",
            "i/o",
            "transport",
            "connection",
            "session",
        ]
    )


async def _reset_orchestrator_mcp_client(
    failed_client: Optional[Any] = None,
    *,
    _skip_lock: bool = False,
) -> None:
    global _ORCHESTRATOR_MCP_CLIENT
    global _ORCHESTRATOR_MCP_CLIENT_LOCK
    if failed_client is None:
        failed_client = _ORCHESTRATOR_MCP_CLIENT
    if failed_client is None:
        return

    def _swap_if_still_failed() -> bool:
        global _ORCHESTRATOR_MCP_CLIENT
        if _ORCHESTRATOR_MCP_CLIENT is failed_client:
            _ORCHESTRATOR_MCP_CLIENT = None
            return True
        return False

    if _skip_lock:
        swapped = _swap_if_still_failed()
    else:
        if _ORCHESTRATOR_MCP_CLIENT_LOCK is None:
            _ORCHESTRATOR_MCP_CLIENT_LOCK = asyncio.Lock()
        async with _ORCHESTRATOR_MCP_CLIENT_LOCK:
            swapped = _swap_if_still_failed()

    if not swapped:
        return

    try:
        await failed_client.__aexit__(None, None, None)
    except Exception:
        pass


async def _close_orchestrator_checkpointer_lifecycle() -> None:
    global _ORCHESTRATOR_CHECKPOINTER
    global _ORCHESTRATOR_CHECKPOINTER_CLOSING
    global _ORCHESTRATOR_CHECKPOINTER_OWNER
    global _ORCHESTRATOR_CHECKPOINTER_LOCK

    if _ORCHESTRATOR_CHECKPOINTER_LOCK is None:
        _ORCHESTRATOR_CHECKPOINTER_LOCK = asyncio.Lock()

    background_tasks = tuple(_BACKGROUND_TASKS)
    try:
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
    finally:
        _BACKGROUND_TASKS.difference_update(background_tasks)
        async with _ORCHESTRATOR_CHECKPOINTER_LOCK:
            owner = _ORCHESTRATOR_CHECKPOINTER_OWNER
            _ORCHESTRATOR_CHECKPOINTER = None
            _ORCHESTRATOR_CHECKPOINTER_OWNER = None
            try:
                if owner is not None:
                    try:
                        await owner.__aexit__(None, None, None)
                    except Exception:
                        pass
            finally:
                _ORCHESTRATOR_CHECKPOINTER_CLOSING = False


async def aclose_orchestrator_runtime() -> None:
    global _ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK
    global _ORCHESTRATOR_CHECKPOINTER_CLOSING
    global _ORCHESTRATOR_CHECKPOINTER_LOCK
    global _ORCHESTRATOR_MCP_CLIENT

    if _ORCHESTRATOR_CHECKPOINTER_LOCK is None:
        _ORCHESTRATOR_CHECKPOINTER_LOCK = asyncio.Lock()

    async with _ORCHESTRATOR_CHECKPOINTER_LOCK:
        close_task = _ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK
        if close_task is None or close_task.done():
            _ORCHESTRATOR_CHECKPOINTER_CLOSING = True
            close_task = asyncio.create_task(
                _close_orchestrator_checkpointer_lifecycle()
            )
            _ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK = close_task

    await asyncio.shield(close_task)

    cached_analysts = list(_ANALYST_CACHE.values())
    _ANALYST_CACHE.clear()
    _ANALYST_BUILD_LOCKS.clear()
    for analyst in cached_analysts:
        try:
            await analyst.aclose()
        except Exception:
            pass

    failed_client = _ORCHESTRATOR_MCP_CLIENT
    _ORCHESTRATOR_MCP_CLIENT = None
    if failed_client is not None:
        try:
            await failed_client.__aexit__(None, None, None)
        except Exception:
            pass

    _get_orchestrator_graph.cache_clear()


def _orchestrator_prune_interval_seconds() -> int:
    raw = os.getenv("FINSEARCH_ORCHESTRATOR_CHECKPOINT_PRUNE_INTERVAL_SECONDS", "").strip()
    if not raw:
        return 60
    try:
        return max(0, int(raw))
    except Exception:
        return 60


async def _restore_planner_from_config(
    run_id: str,
) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    saver = await _get_orchestrator_checkpointer()
    async with saver.conn.execute(
        "SELECT metadata FROM checkpoints WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
        (str(run_id),),
    ) as cursor:
        row = await cursor.fetchone()
    if not row:
        return None
    metadata = row[0]
    if not metadata:
        return None
    try:
        payload = json.loads(metadata)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("planner_config")


def _normalize_model_name(value: Any) -> str:
    raw = str(value or "").strip()
    if raw:
        return raw
    return _ANALYST_DEFAULT_MODEL


async def _get_pooled_analyst(model: str) -> AnalystAgent:
    cache_key = _normalize_model_name(model)
    lock = _ANALYST_BUILD_LOCKS.get(cache_key)
    if lock is None:
        lock = asyncio.Lock()
        _ANALYST_BUILD_LOCKS[cache_key] = lock
    async with lock:
        analyst = _ANALYST_CACHE.get(cache_key)
        if analyst is None:
            analyst = AnalystAgent(
                model=cache_key,
                max_context_items=_ANALYST_MAX_CONTEXT_ITEMS,
            )
            await analyst.abuild()
            if len(_ANALYST_CACHE) >= _ANALYST_CACHE_MAX_SIZE:
                _, evicted = _ANALYST_CACHE.popitem(last=False)
                await evicted.aclose()
            _ANALYST_CACHE[cache_key] = analyst
        elif not bool(getattr(analyst, "is_ready", False)):
            await analyst.abuild()
        _ANALYST_CACHE.move_to_end(cache_key)

    return analyst


def _get_runtime_planner() -> Any:
    configurable = get_config().get("configurable") or {}
    planner = configurable.get("planner")
    if planner is None:
        planner_config = configurable.get("planner_config") or {}
        model = _normalize_text(planner_config.get("model")) or None
        enable_query_expansion = bool(planner_config.get("enable_query_expansion", True))
        auto_run_full_planner = bool(planner_config.get("auto_run_full_planner", False))
        default_doc_types = planner_config.get("default_doc_types")
        if model is not None:
            return InteractivePlannerAgent(
                model=model,
                enable_query_expansion=enable_query_expansion,
                auto_run_full_planner=auto_run_full_planner,
                default_doc_types=default_doc_types,
                company_ticker_map=planner_config.get("company_ticker_map"),
                log_timing=False,
            )
        return InteractivePlannerAgent(log_timing=False)
    return planner


def _graph_config(*, run_id: str, planner: Any) -> Dict[str, Any]:
    planner_config = {}
    if planner is not None:
        planner_config = {
            "model": _normalize_text(getattr(planner, "model", None))
            or _normalize_text(getattr(planner, "planner_model", None))
            or "qwen2.5-14b-instruct-1m",
            "enable_query_expansion": bool(getattr(planner, "enable_query_expansion", True)),
            "auto_run_full_planner": bool(getattr(planner, "auto_run_full_planner", False)),
            "default_doc_types": list(getattr(planner, "default_doc_types") or []),
            "company_ticker_map": getattr(planner, "company_ticker_map", None),
            "full_planner_include_trace": bool(getattr(planner, "full_planner_include_trace", False)),
        }
    return {
        "configurable": {
            "thread_id": run_id,
            "planner": planner,
            "planner_config": planner_config,
        },
        "metadata": {
            "orchestrator_started_at": time.time(),
            "planner_config": planner_config,
        },
    }


def _normalize_text(value: Any) -> Optional[str]:
    return normalize_text(value)


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


def _coerce_plan_route(plan_obj: Dict[str, Any]) -> str:
    route = _normalize_text((plan_obj or {}).get("route")) or "kb"
    if route in {"kb", "structured_fact", "hybrid"}:
        return route
    return "kb"


def _route_uses_structured_facts(route: str) -> bool:
    return route in {"structured_fact", "hybrid"}


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
        if isinstance(issue, OpenIssue):
            issues.append(issue)
            continue
        try:
            issues.append(OpenIssue.model_validate(issue))
        except Exception:
            continue
    return issues


def _coerce_open_issue_payloads(issues: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for issue in issues or []:
        if isinstance(issue, OpenIssue):
            out.append(issue.model_dump(mode="json"))
            continue
        try:
            out.append(OpenIssue.model_validate(issue).model_dump(mode="json"))
        except Exception:
            continue
    return out


def _dedupe_open_issues(issues: Sequence[OpenIssue]) -> list[OpenIssue]:
    out: list[OpenIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        code = issue.code
        message = issue.message
        severity = str(issue.severity)
        key = (code, message, severity)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def _dedupe_open_issue_payloads(left: Any, right: Any) -> list[Dict[str, Any]]:
    merged = _coerce_open_issue_payloads(left) + _coerce_open_issue_payloads(right)
    merged_issues = _dedupe_open_issues(_coerce_open_issues({"open_issues": merged}))
    return _coerce_open_issue_payloads(merged_issues)


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
    }

    candidate_rows = list(retrieval_compact.get("top_items") or [])
    if not candidate_rows:
        rows = list(retrieval.get("top_tables") or [])
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
            "targets": [],
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

    retrieval_targets: list[Dict[str, Any]] = []
    target_summaries: dict[tuple[str | None, int | None, str | None], Dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_target = dict(run.get("target") or {})
        ticker = _normalize_text(run_target.get("ticker")) or None
        fiscal_year = _normalize_int(run_target.get("fiscal_year"))
        form_type = _normalize_text(run_target.get("form_type")) or None
        key = (ticker, fiscal_year, form_type)

        summary = target_summaries.get(key)
        if summary is None:
            summary = {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "form_type": form_type,
                "runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "tables_retrieved": 0,
                "error": None,
            }
            target_summaries[key] = summary
            retrieval_targets.append(summary)

        final_retrieval = dict(run.get("final_retrieval") or run.get("retrieval") or {})
        run_ok = bool(final_retrieval.get("ok"))
        run_tables = list(final_retrieval.get("top_tables") or final_retrieval.get("results") or [])

        summary["runs"] += 1
        if run_ok:
            summary["successful_runs"] += 1
        else:
            summary["failed_runs"] += 1
            summary_error = _normalize_text(final_retrieval.get("error"))
            if summary_error and summary["error"] is None:
                summary["error"] = summary_error
        summary["tables_retrieved"] += len([row for row in run_tables if isinstance(row, dict)])

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
            if (
                _normalize_text(metadata_used.get("ticker"))
                and _normalize_int(metadata_used.get("fiscal_year")) is not None
            ):
                target = {
                    "ticker": _normalize_text(metadata_used.get("ticker")),
                    "fiscal_year": _normalize_int(metadata_used.get("fiscal_year")),
                    "form_type": _normalize_text(metadata_used.get("form_type")) or None,
                }
                break

    if not runs:
        metadata_used = dict(retrieval_output.get("metadata_used") or {})
        if (
            _normalize_text(metadata_used.get("ticker"))
            or _normalize_int(metadata_used.get("fiscal_year")) is not None
            or _normalize_text(metadata_used.get("form_type"))
        ):
            target = {
                "ticker": _normalize_text(metadata_used.get("ticker")),
                "fiscal_year": _normalize_int(metadata_used.get("fiscal_year")),
                "form_type": _normalize_text(metadata_used.get("form_type")) or None,
            }

    return {
        "type": "retrieval",
        "ok": bool(retrieval_output.get("ok", False)),
        "original_user_query": _normalize_text(retrieval_output.get("original_user_query")),
        "target": target,
        "targets": retrieval_targets,
        "attempts": attempts,
    }


def _init_node(state: OrchestratorState) -> Dict[str, Any]:
    return {
        "start_time": time.time(),
        "retrieval_timing_ms": {},
        "retrieval_skipped_reason": "",
        "structured_fact_timing_ms": {},
        "structured_fact_results": [],
        "clarification_turns": [],
        "planner_timing_ms": {},
        "open_issues": [],
    }


def _coerce_planner_answers(answers: Any, questions: Sequence[str]) -> List[str]:
    if isinstance(answers, str):
        if len(questions) != 1:
            raise ValueError("Expected one answer per clarification question.")
        return [answers.strip()]

    if isinstance(answers, (list, tuple)):
        answer_list = [str(answer).strip() for answer in answers]
        if len(answer_list) != len(questions):
            raise ValueError("Number of answers must match number of clarification questions.")
        return answer_list

    raise TypeError("answers must be a string or a list/tuple of strings.")


def _format_runtime_contract_validation_error(
    exc: ValidationError,
    *,
    max_errors: int = 5,
) -> str:
    errors = exc.errors(include_input=False)
    details = []
    for error in errors[:max_errors]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        field_path = f"planner_output.{location}" if location else "planner_output"
        details.append(f"{field_path}: {error.get('msg', 'Invalid value')}")

    omitted_count = len(errors) - len(details)
    if omitted_count:
        noun = "error" if omitted_count == 1 else "errors"
        details.append(f"... {omitted_count} additional validation {noun} omitted")
    return "\n".join(details) or "Planner runtime contract validation failed."


def _build_runtime_contract_error_plan(
    *,
    user_query: str,
    clarification_history: Sequence[Dict[str, Any]],
    validation_error: str,
) -> Dict[str, Any]:
    clean_query = str(user_query or "").strip() or "Unavailable user query"
    error_message = str(validation_error or "Planner runtime contract validation failed.").strip()
    plan = PlannerRuntimeOutput(
        status="error",
        retrieval_needed=False,
        intent=PlannerIntent.OTHER,
        route="kb",
        structured_fact_requests=[],
        metadata=FilingMetadata(),
        analysis_task=AnalysisTask(
            task_type="extract",
            metric="filing evidence",
            requires_calculation=False,
            expected_artifacts=[],
            output_format="short_answer",
        ),
        task_class="other",
        targets=[],
        retrieval_plan=None,
        open_issues=[
            OpenIssue(
                code="PLANNER_RUNTIME_CONTRACT_INVALID",
                message=error_message,
                severity=Severity.ERROR,
            )
        ],
        original_user_query=clean_query,
        effective_user_query=clean_query,
        clarification_history=_normalize_clarification_turns(
            list(clarification_history or [])
        ),
        clarification_request=None,
    )
    return plan.model_dump(mode="json")


def _serialize_invalid_planner_output(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return {"repr": repr(value)}
    return value


async def _planner_graph_node(state: OrchestratorState) -> Dict[str, Any]:
    planner = _get_runtime_planner()
    t0 = time.perf_counter()
    planner_kwargs = {
        "user_query": state["user_query"],
        "clarification_turns": list(state.get("clarification_turns") or []),
    }
    planner_turn: Dict[str, Any] = {}
    invalid_planner_output: Any = None
    try:
        if hasattr(planner, "aplan_turn"):
            raw_planner_turn = await planner.aplan_turn(**planner_kwargs)
        else:
            raw_planner_turn = await asyncio.to_thread(
                planner.start, state["user_query"]
            )
        if isinstance(raw_planner_turn, dict):
            planner_turn = dict(raw_planner_turn)
        invalid_planner_output = planner_turn.get("planner_output")
        planner_dump = PlannerRuntimeOutput.model_validate(
            invalid_planner_output
        ).model_dump(mode="json")
        planner_turn["planner_output"] = planner_dump
    except ValidationError as exc:
        validation_error = _format_runtime_contract_validation_error(exc)
        planner_dump = _build_runtime_contract_error_plan(
            user_query=state["user_query"],
            clarification_history=state.get("clarification_turns") or [],
            validation_error=validation_error,
        )
        planner_turn.update(
            {
                "planner_output": planner_dump,
                "invalid_planner_output": _serialize_invalid_planner_output(
                    invalid_planner_output
                ),
                "validation_error": validation_error,
                "runtime_contract_error": True,
            }
        )
    planner_timing_ms = dict(state.get("planner_timing_ms") or {})
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if state.get("clarification_turns"):
        planner_timing_ms["planner_resume_ms"] = (
            int(planner_timing_ms.get("planner_resume_ms", 0)) + elapsed_ms
        )
    else:
        planner_timing_ms["planner_start_ms"] = elapsed_ms
    return {
        "planner_turn": planner_turn,
        "plan_obj": planner_dump,
        "planner_dump": planner_dump,
        "planner_timing_ms": planner_timing_ms,
        "open_issues": _coerce_open_issue_payloads(planner_dump.get("open_issues")),
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
    planner_turn = dict(state.get("planner_turn") or {})
    questions = list(
        dict(planner_turn.get("clarification_request") or {}).get("questions") or []
    )
    answer_list = _coerce_planner_answers(state.get("planner_resume_answers"), questions)

    clarification_turns = [
        {"question": question, "answer": answer}
        for question, answer in zip(questions, answer_list)
    ]
    return {
        "planner_resume_answers": None,
        "clarification_turns": clarification_turns,
    }


def _route_after_planner_turn(state: OrchestratorState) -> str:
    plan_obj = state["plan_obj"]
    status = str(plan_obj.get("status") or "").strip().lower()
    if status == "needs_clarification":
        return "planner_interrupt"
    if status == "completed":
        route = _coerce_plan_route(plan_obj)
        if route == "structured_fact":
            return "structured_facts"
        if route == "hybrid":
            return "check_retrieval_metadata"
        return "check_retrieval_metadata" if bool(plan_obj.get("retrieval_needed")) else "build_packet_without_retrieval"
    if status == "error":
        return "planner_error"
    return "planner_error"


def _planner_error_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    planner_turn = dict(state.get("planner_turn") or {})
    packet = _build_packet_without_retrieval(
        user_query=state["user_query"],
        plan_obj=plan_obj,
        plan_id=state["plan_id"],
    )

    error_parts = [
        f"LLM error: {_normalize_text(planner_turn.get('llm_error'))}"
        if planner_turn.get("llm_error")
        else None,
        f"validation error: {_normalize_text(planner_turn.get('validation_error'))}"
        if planner_turn.get("validation_error")
        else None,
    ]
    error_message = " ".join(part for part in error_parts if part) or "Planner execution failed before retrieval."
    error_code = "PLANNER_EXECUTION_ERROR"
    if planner_turn.get("runtime_contract_error"):
        error_code = "PLANNER_RUNTIME_CONTRACT_INVALID"
    elif any(part for part in error_parts):
        error_code = "PLANNER_RUNTIME_ERROR"

    if not any(issue.code == error_code for issue in packet.open_issues):
        packet.open_issues.append(
            OpenIssue(
                code=error_code,
                message=error_message,
                severity=Severity.ERROR,
            )
        )

    analyst_result = AnalystRunResult(
        ok=False,
        status="error",
        answer="Planner failed before analyst could run.",
        intent=packet.intent,
        metric=packet.analysis_task.metric,
        citations=[],
        open_issues=packet.open_issues,
        error=error_message,
    )
    return {"packet": packet, "analyst_result": analyst_result}


def _attach_open_issues_node(state: OrchestratorState) -> Dict[str, Any]:
    packet = state["packet"]
    packet_issues = list(packet.open_issues)
    state_issues = _coerce_open_issues({"open_issues": state.get("open_issues")})
    merged_issues = _dedupe_open_issues(packet_issues + state_issues)
    if list(packet.open_issues) == merged_issues:
        return {}
    return {"packet": packet.model_copy(update={"open_issues": merged_issues})}


def _validate_retrieval_plan_targets(
    retrieval_plan: Any,
    valid_target_ids: Sequence[int],
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(retrieval_plan, dict):
        return None, [
            {
                "code": "INVALID_RETRIEVAL_PLAN",
                "message": "No valid retrieval plan object was provided.",
                "severity": "error",
            }
        ]

    fanout_mode = str(retrieval_plan.get("fanout_mode") or "").strip()
    if fanout_mode not in {"single_target", "per_target"}:
        fanout_mode = "per_target" if len(valid_target_ids) > 1 else "single_target"

    jobs = []
    issues: List[Dict[str, Any]] = []
    valid_target_ids = list(_dedupe_ints(valid_target_ids))
    for index, raw_job in enumerate(retrieval_plan.get("jobs") or []):
        if not isinstance(raw_job, dict):
            issues.append(
                {
                    "code": "INVALID_RETRIEVAL_PLAN_JOB",
                    "message": f"Retrieval plan job #{index + 1} is invalid.",
                    "severity": "warning",
                }
            )
            continue

        goal = _normalize_text(raw_job.get("goal"))
        if not goal:
            issues.append(
                {
                    "code": "INVALID_RETRIEVAL_PLAN_JOB",
                    "message": f"Retrieval plan job #{index + 1} is missing a goal.",
                    "severity": "warning",
                }
            )
            continue

        applies_to_target_ids = _dedupe_ints(raw_job.get("applies_to_target_ids") or [])
        if not applies_to_target_ids:
            applies_to_target_ids = list(valid_target_ids)
            invalid_ids: List[int] = []
        else:
            invalid_ids = [target_id for target_id in applies_to_target_ids if target_id not in valid_target_ids]
            applies_to_target_ids = [target_id for target_id in applies_to_target_ids if target_id in valid_target_ids]

        if invalid_ids:
            issues.append(
                {
                    "code": "INVALID_RETRIEVAL_PLAN_TARGET_IDS",
                    "message": (
                        f"Retrieval plan references unknown target IDs {invalid_ids}; "
                        "they were filtered to resolved targets."
                    ),
                    "severity": "warning",
                }
            )

        if not applies_to_target_ids:
            issues.append(
                {
                    "code": "EMPTY_RETRIEVAL_JOB_TARGETS",
                    "message": (
                        f"Retrieval plan job #{index + 1} had no valid target IDs after filtering."
                    ),
                    "severity": "error",
                }
            )
            continue

        jobs.append(
            {
                "job_type": _normalize_text(raw_job.get("job_type")) or "metric_extract",
                "goal": goal,
                "applies_to_target_ids": applies_to_target_ids,
            }
        )

    if not jobs:
        return None, issues or [
            {
                "code": "EMPTY_RETRIEVAL_PLAN",
                "message": "No valid retrieval jobs remained after target revalidation.",
                "severity": "error",
            }
        ]

    return {"fanout_mode": fanout_mode, "jobs": jobs}, issues


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
        if _normalize_text(target.get("ticker"))
        and _normalize_int(target.get("fiscal_year")) is not None
    ]
    retrieval_plan_raw = dict(plan_obj.get("retrieval_plan") or {})
    valid_target_ids = [
        int(target["target_id"])
        for target in valid_targets
        if _normalize_int(target.get("target_id")) is not None
    ]
    retrieval_plan, plan_issues = _validate_retrieval_plan_targets(
        retrieval_plan_raw,
        valid_target_ids,
    )

    if not retrieval_plan:
        return {
            "retrieval_state": {},
            "retrieval_skipped_reason": (
                "INVALID_RETRIEVAL_PLAN" if retrieval_plan_raw else "MISSING_RETRIEVAL_PLAN"
            ),
            "open_issues": plan_issues,
        }
    if not valid_targets:
        return {"retrieval_state": {}, "retrieval_skipped_reason": "MISSING_TARGET_METADATA"}

    retrieval_state = {
        "targets": valid_targets,
        "retrieval_plan": retrieval_plan,
        "original_user_query": plan_obj.get("original_user_query") or state["user_query"],
        "clarification_history": list(plan_obj.get("clarification_history") or []),
    }
    result: Dict[str, Any] = {"retrieval_state": retrieval_state, "retrieval_skipped_reason": ""}
    if plan_issues:
        result["open_issues"] = plan_issues
    return result


def _route_after_retrieval_metadata(state: OrchestratorState) -> str:
    retrieval_state = state.get("retrieval_state") or {}
    if retrieval_state:
        return "retrieval"
    return "structured_facts" if _route_uses_structured_facts(_coerce_plan_route(state.get("plan_obj") or {})) else "build_packet_without_retrieval"


async def _retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    t_ret = time.perf_counter()
    retrieval_client = None
    try:
        retrieval_client = await _get_orchestrator_mcp_client()
        ret_state = await retrieval_agent(
            state["retrieval_state"],
            client=retrieval_client,
        )
        retrieval_output = ret_state.get("retrieval")
        if _is_mcp_transport_error(retrieval_output.get("error") if isinstance(retrieval_output, dict) else None):
            await _reset_orchestrator_mcp_client(retrieval_client)
    except Exception as exc:
        if _is_mcp_transport_error(str(exc)):
            await _reset_orchestrator_mcp_client(retrieval_client)
        retrieval_output = _build_retrieval_failure_output(
            retrieval_state=state["retrieval_state"],
            exc=exc,
        )

    retrieval_output = dict(retrieval_output or {})
    new_open_issues: list[Dict[str, Any]] = []
    retrieval_failures = retrieval_output.get("partial_failures")
    if isinstance(retrieval_failures, list) and retrieval_failures:
        for failure in retrieval_failures:
            if not isinstance(failure, dict):
                continue
            target = failure.get("target") or {}
            target_ticker = _normalize_text(target.get("ticker"))
            target_fy = _normalize_int(target.get("fiscal_year"))
            job_goal = _normalize_text(failure.get("goal") or (failure.get("job") or {}).get("goal"))
            job_type = _normalize_text(failure.get("job_type") or (failure.get("job") or {}).get("job_type"))
            fail_msg = _normalize_text(failure.get("error")) or "retrieval run returned no usable evidence"
            new_open_issues.append(
                OpenIssue(
                    code="RETRIEVAL_PARTIAL_FAILURE",
                    message=(
                        f"Retrieval run failed for {job_type or 'job'}"
                        f" on target {target_ticker or 'unknown'}"
                        f" fy={target_fy if target_fy is not None else 'unknown'}: {fail_msg}"
                    ),
                    severity=Severity.ERROR,
                    metadata={
                        "job_goal": job_goal,
                        "target": target,
                    },
                ).model_dump(mode="json")
            )
    retrieval_timing_ms = dict(state.get("retrieval_timing_ms") or {})
    retrieval_timing_ms["retrieve_ms"] = int((time.perf_counter() - t_ret) * 1000)
    return {
        "retrieval_output": retrieval_output,
        "open_issues": new_open_issues,
        "retrieval_timing_ms": retrieval_timing_ms,
    }


def _route_after_retrieval_node(state: OrchestratorState) -> str:
    return "structured_facts" if _route_uses_structured_facts(_coerce_plan_route(state.get("plan_obj") or {})) else "build_packet_from_retrieval"


def _normalize_metric_lookup_text(value: Any) -> str:
    text = _normalize_text(value) or ""
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9.\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _metric_registry_lookup_terms(metric_id: str, label: str) -> tuple[str, ...]:
    terms = [_normalize_metric_lookup_text(metric_id), _normalize_metric_lookup_text(label)]
    for alias in _STRUCTURED_FACT_ALIAS_MAP.get(metric_id, ()):
        normalized = _normalize_metric_lookup_text(alias)
        if normalized:
            terms.append(normalized)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term or term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return tuple(deduped)


def _resolve_metric_id_for_structured_fact_request(
    *,
    metric_hint: Any,
    subquestion: Any,
) -> tuple[Optional[str], str, Optional[str]]:
    query = _normalize_metric_lookup_text(metric_hint) or _normalize_metric_lookup_text(subquestion)
    if not query:
        return None, "unresolved", "No metric hint or subquestion text was available for resolution."

    exact_matches: list[str] = []
    for metric_id, definition in METRIC_REGISTRY.items():
        terms = _metric_registry_lookup_terms(metric_id, definition.label)
        if query in terms:
            exact_matches.append(metric_id)

    if len(exact_matches) == 1:
        resolved = exact_matches[0]
        return resolved, "resolved", f"Resolved metric from exact registry match: {resolved}."
    if len(exact_matches) > 1:
        matches = sorted(set(exact_matches))
        return None, "ambiguous", f"Metric text matched multiple registered metrics exactly: {', '.join(matches)}."

    contained_matches: list[str] = []
    query_padded = f" {query} "
    for metric_id, definition in METRIC_REGISTRY.items():
        terms = _metric_registry_lookup_terms(metric_id, definition.label)
        if any(f" {term} " in query_padded for term in terms if term):
            contained_matches.append(metric_id)

    unique_matches = sorted(set(contained_matches))
    if len(unique_matches) == 1:
        resolved = unique_matches[0]
        return resolved, "resolved", f"Resolved metric from registry phrase match: {resolved}."
    if len(unique_matches) > 1:
        return None, "ambiguous", (
            "Metric text matched multiple registered metrics: "
            f"{', '.join(unique_matches)}."
        )
    return None, "unresolved", f"Metric text did not match any registered SEC metric: {query}."


def _entity_hint_looks_like_ticker(entity_hint: Any) -> Optional[str]:
    text = _normalize_text(entity_hint)
    if not text:
        return None
    stripped = str(text).strip()
    return stripped if _TICKER_LIKE_RE.fullmatch(stripped) else None


def _select_matching_target_for_structured_fact(
    *,
    plan_obj: Dict[str, Any],
    entity_hint: Any,
    fiscal_year: Optional[int],
) -> Optional[Dict[str, Any]]:
    targets = [
        dict(target)
        for target in (plan_obj.get("targets") or [])
        if isinstance(target, dict)
    ]
    if not targets:
        return None

    hinted_ticker = _entity_hint_looks_like_ticker(entity_hint)
    normalized_entity = _normalize_metric_lookup_text(entity_hint)
    requested_year = _normalize_int(fiscal_year)

    def _matches(target: Dict[str, Any]) -> bool:
        target_ticker = _normalize_text(target.get("ticker"))
        target_company = _normalize_metric_lookup_text(target.get("company_name"))
        target_year = _normalize_int(target.get("fiscal_year"))
        if requested_year is not None and target_year is not None and requested_year != target_year:
            return False
        if hinted_ticker and target_ticker == hinted_ticker:
            return True
        if normalized_entity and normalized_entity == target_company:
            return True
        return False

    matched = [target for target in targets if _matches(target)]
    if matched:
        return matched[0]
    if len(targets) == 1:
        return targets[0]
    return None


def _resolve_structured_fact_inputs(
    *,
    plan_obj: Dict[str, Any],
    request: Dict[str, Any],
) -> tuple[Optional[str], Optional[int], Optional[Dict[str, Any]]]:
    metadata = _coerce_metadata(plan_obj)
    matched_target = _select_matching_target_for_structured_fact(
        plan_obj=plan_obj,
        entity_hint=request.get("entity_hint"),
        fiscal_year=_normalize_int(request.get("fiscal_year")),
    )

    resolved_ticker = (
        _entity_hint_looks_like_ticker(request.get("entity_hint"))
        or metadata.ticker
        or _normalize_text((matched_target or {}).get("ticker"))
    )
    resolved_year = (
        _normalize_int(request.get("fiscal_year"))
        or metadata.fiscal_year
        or _normalize_int((matched_target or {}).get("fiscal_year"))
    )
    return resolved_ticker, resolved_year, matched_target


def _build_structured_fact_result(
    *,
    request: Dict[str, Any],
    resolved_ticker: Optional[str],
    resolved_year: Optional[int],
    resolved_metric_id: Optional[str],
    resolver_status: str,
    resolver_reason: Optional[str],
    tool_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "subquestion": _normalize_text(request.get("subquestion")),
        "metric_hint": _normalize_text(request.get("metric_hint")),
        "entity_hint": _normalize_text(request.get("entity_hint")),
        "requested_fiscal_year": _normalize_int(request.get("fiscal_year")),
        "requested_fiscal_period": _normalize_text(request.get("fiscal_period")),
        "resolved_ticker": resolved_ticker,
        "resolved_fiscal_year": resolved_year,
        "resolved_metric_id": resolved_metric_id,
        "resolver_status": resolver_status,
        "resolver_reason": resolver_reason,
        "tool_result": tool_result,
    }


async def _execute_structured_fact_requests(
    *,
    plan_obj: Dict[str, Any],
    client: Any,
) -> list[Dict[str, Any]]:
    requests = [
        dict(item)
        for item in (plan_obj.get("structured_fact_requests") or [])
        if isinstance(item, dict)
    ]
    results: list[Dict[str, Any]] = []
    for request in requests:
        resolved_ticker, resolved_year, _matched_target = _resolve_structured_fact_inputs(
            plan_obj=plan_obj,
            request=request,
        )
        resolved_metric_id, resolver_status, resolver_reason = _resolve_metric_id_for_structured_fact_request(
            metric_hint=request.get("metric_hint"),
            subquestion=request.get("subquestion"),
        )

        if not resolved_ticker or resolved_year is None:
            missing_bits = []
            if not resolved_ticker:
                missing_bits.append("ticker")
            if resolved_year is None:
                missing_bits.append("fiscal_year")
            results.append(
                _build_structured_fact_result(
                    request=request,
                    resolved_ticker=resolved_ticker,
                    resolved_year=resolved_year,
                    resolved_metric_id=resolved_metric_id if resolver_status == "resolved" else None,
                    resolver_status="missing_inputs",
                    resolver_reason=(
                        "Missing structured-fact execution inputs: " + ", ".join(missing_bits) + "."
                    ),
                    tool_result=None,
                )
            )
            continue

        if resolver_status != "resolved" or not resolved_metric_id:
            results.append(
                _build_structured_fact_result(
                    request=request,
                    resolved_ticker=resolved_ticker,
                    resolved_year=resolved_year,
                    resolved_metric_id=None,
                    resolver_status=resolver_status,
                    resolver_reason=resolver_reason,
                    tool_result=None,
                )
            )
            continue

        tool_result: Optional[Dict[str, Any]] = None
        try:
            response = await client.get_metric(
                ticker=resolved_ticker,
                fiscal_year=resolved_year,
                metric_id=resolved_metric_id,
            )
            tool_result = dict(response or {})
        except Exception as exc:
            if _is_mcp_transport_error(str(exc)):
                await _reset_orchestrator_mcp_client(client)
            tool_result = {
                "ok": False,
                "status": "error",
                "metric_id": resolved_metric_id,
                "error": str(exc),
            }

        results.append(
            _build_structured_fact_result(
                request=request,
                resolved_ticker=resolved_ticker,
                resolved_year=resolved_year,
                resolved_metric_id=resolved_metric_id,
                resolver_status="resolved",
                resolver_reason=resolver_reason,
                tool_result=tool_result,
            )
        )
    return results


async def _structured_facts_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = dict(state.get("plan_obj") or {})
    requests = list(plan_obj.get("structured_fact_requests") or [])
    timing = dict(state.get("structured_fact_timing_ms") or {})
    if not requests:
        timing["structured_facts_ms"] = 0
        return {"structured_fact_results": [], "structured_fact_timing_ms": timing}

    t0 = time.perf_counter()
    client = None
    try:
        client = await _get_orchestrator_mcp_client()
        results = await _execute_structured_fact_requests(plan_obj=plan_obj, client=client)
    except Exception as exc:
        if _is_mcp_transport_error(str(exc)):
            await _reset_orchestrator_mcp_client(client)
        results = [
            {
                "subquestion": _normalize_text(request.get("subquestion")),
                "metric_hint": _normalize_text(request.get("metric_hint")),
                "entity_hint": _normalize_text(request.get("entity_hint")),
                "requested_fiscal_year": _normalize_int(request.get("fiscal_year")),
                "requested_fiscal_period": _normalize_text(request.get("fiscal_period")),
                "resolved_ticker": None,
                "resolved_fiscal_year": None,
                "resolved_metric_id": None,
                "resolver_status": "unresolved",
                "resolver_reason": f"Structured fact execution failed before tool invocation: {exc}",
                "tool_result": None,
            }
            for request in requests
            if isinstance(request, dict)
        ]

    timing["structured_facts_ms"] = int((time.perf_counter() - t0) * 1000)
    return {"structured_fact_results": results, "structured_fact_timing_ms": timing}


def _route_after_structured_facts(state: OrchestratorState) -> str:
    retrieval_output = state.get("retrieval_output")
    return "build_packet_from_retrieval" if isinstance(retrieval_output, dict) else "build_packet_without_retrieval"


def _coerce_form_type_value(value: Any) -> Optional[FormType]:
    text = _normalize_text(value)
    if not text:
        return None
    try:
        return FormType(text)
    except Exception:
        return None


def _structured_fact_context_text(result: Dict[str, Any]) -> str:
    metric_label = (
        _normalize_text((result.get("tool_result") or {}).get("metric_id"))
        or _normalize_text(result.get("resolved_metric_id"))
        or _normalize_text(result.get("metric_hint"))
        or "structured metric"
    )
    tool_result = result.get("tool_result") or {}
    if isinstance(tool_result, dict) and tool_result.get("ok"):
        value = tool_result.get("value")
        unit = _normalize_text(tool_result.get("unit")) or ""
        form_type = _normalize_text(tool_result.get("form_type")) or ""
        report_date = _normalize_text(tool_result.get("report_date")) or ""
        accession_number = _normalize_text(tool_result.get("accession_number")) or ""
        anchor_bits = [bit for bit in [form_type, report_date, accession_number] if bit]
        anchor_text = f" ({', '.join(anchor_bits)})" if anchor_bits else ""
        return f"Structured fact: {metric_label} = {value} {unit}".strip() + anchor_text + "."

    if isinstance(tool_result, dict) and tool_result:
        status = _normalize_text(tool_result.get("status")) or "error"
        reason = _normalize_text(tool_result.get("error")) or _normalize_text(result.get("resolver_reason")) or "No additional detail."
        return f"Structured fact: {metric_label} returned status {status}. {reason}"

    status = _normalize_text(result.get("resolver_status")) or "unresolved"
    reason = _normalize_text(result.get("resolver_reason")) or "Could not resolve the structured fact request."
    return f"Structured fact request {status}. {reason}"


def _build_structured_fact_context_items(
    *,
    packet: AnalystPacket,
    structured_fact_results: Sequence[Dict[str, Any]],
) -> list[ContextItem]:
    items: list[ContextItem] = []
    next_index = len(packet.context_items) + 1
    for result in structured_fact_results or []:
        if not isinstance(result, dict):
            continue
        tool_result = result.get("tool_result") or {}
        form_type = _coerce_form_type_value((tool_result or {}).get("form_type")) or packet.metadata.form_type
        source = SourceRef(
            ticker=_normalize_text(result.get("resolved_ticker")) or packet.metadata.ticker,
            fiscal_year=_normalize_int(result.get("resolved_fiscal_year")) or packet.metadata.fiscal_year,
            form_type=form_type,
            filing_date=_normalize_text((tool_result or {}).get("filed_date")),
            accession_no=_normalize_text((tool_result or {}).get("accession_number")),
        )
        items.append(
            ContextItem(
                context_id=f"ctx_{next_index}",
                kind=ContextItemKind.TEXT,
                source=source,
                payload={"content": _structured_fact_context_text(result)},
            )
        )
        next_index += 1
    return items


def _append_structured_fact_context_items(
    *,
    packet: AnalystPacket,
    structured_fact_results: Sequence[Dict[str, Any]],
) -> AnalystPacket:
    new_items = _build_structured_fact_context_items(
        packet=packet,
        structured_fact_results=structured_fact_results,
    )
    if not new_items:
        return packet

    updated_context_items = list(packet.context_items) + new_items
    context_quality = packet.context_quality
    if not packet.context_items and new_items and packet.context_quality == ContextQuality.LOW:
        context_quality = ContextQuality.MEDIUM
    return packet.model_copy(
        update={
            "context_items": updated_context_items,
            "context_quality": context_quality,
        }
    )


def _route_after_retrieval_attach_open_issues(state: OrchestratorState) -> str:
    plan_obj = dict(state.get("plan_obj") or {})
    route = _coerce_plan_route(plan_obj)
    retrieval_state = state.get("retrieval_state") or {}
    retrieval_output = state.get("retrieval_output")
    skipped_reason = _normalize_text(state.get("retrieval_skipped_reason"))
    if (
        route == "kb"
        and plan_obj.get("retrieval_needed")
        and skipped_reason in {
            "MISSING_METADATA",
            "MISSING_TARGET_METADATA",
            "MISSING_RETRIEVAL_PLAN",
            "INVALID_RETRIEVAL_PLAN",
        }
    ):
        return "finalize"
    if route == "kb" and retrieval_state and isinstance(retrieval_output, dict) and retrieval_output.get("ok") is False:
        return "finalize"
    return "analyst"


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
    packet = _append_structured_fact_context_items(
        packet=packet,
        structured_fact_results=state.get("structured_fact_results") or [],
    )
    return {"packet": packet}


def _build_packet_without_retrieval_node(state: OrchestratorState) -> Dict[str, Any]:
    plan_obj = state["plan_obj"]
    packet = _build_packet_without_retrieval(
        user_query=state["user_query"],
        plan_obj=plan_obj,
        plan_id=state["plan_id"],
    )
    packet = _append_structured_fact_context_items(
        packet=packet,
        structured_fact_results=state.get("structured_fact_results") or [],
    )
    open_issues: list[Dict[str, Any]] = []
    if str(plan_obj.get("status") or "").strip().lower() == "needs_clarification":
        open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_CLARIFICATION_REQUIRED",
                message="Planner requested clarification before retrieval could proceed.",
                severity=Severity.WARNING,
            ).model_dump(mode="json")
        )
    elif plan_obj.get("retrieval_needed") and state.get("retrieval_skipped_reason") in {
        "MISSING_METADATA",
        "MISSING_TARGET_METADATA",
        "MISSING_RETRIEVAL_PLAN",
        "INVALID_RETRIEVAL_PLAN",
    }:
        open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_MISSING_METADATA",
                message="Retrieval was required but retrieval metadata or target resolution was incomplete.",
                severity=Severity.WARNING,
            ).model_dump(mode="json")
        )
    elif not plan_obj.get("retrieval_needed"):
        open_issues.append(
            OpenIssue(
                code="RETRIEVAL_SKIPPED_BY_PLANNER",
                message="Planner set retrieval_needed=False; analyst ran without retrieved filing context.",
                severity=Severity.INFO,
            ).model_dump(mode="json")
        )
    return {"packet": packet, "open_issues": open_issues}


async def _resolve_runtime_planner(*, run_id: str, planner: Optional[Any]) -> Any:
    if planner is not None:
        return planner

    planner_config = await _restore_planner_from_config(run_id)
    if planner_config:
        model = _normalize_text(planner_config.get("model"))
        return InteractivePlannerAgent(
            model=model or "qwen2.5-14b-instruct-1m",
            enable_query_expansion=bool(planner_config.get("enable_query_expansion", True)),
            auto_run_full_planner=bool(planner_config.get("auto_run_full_planner", False)),
            default_doc_types=planner_config.get("default_doc_types"),
            company_ticker_map=planner_config.get("company_ticker_map"),
            full_planner_include_trace=bool(planner_config.get("full_planner_include_trace", False)),
            log_timing=False,
        )

    return InteractivePlannerAgent(log_timing=False)


async def _analyst_node(state: OrchestratorState) -> Dict[str, Any]:
    analyst = await _get_pooled_analyst(_normalize_model_name(state["analyst_model"]))
    analyst_result = await analyst.arun(state["packet"], debug=state["debug"])
    return {"analyst_result": analyst_result}


def _finalize_node(state: OrchestratorState) -> Dict[str, Any]:
    state_values = dict(state or {})
    return {"total_ms": _compute_orchestrator_total_ms(state_values) or 0}


def _compute_orchestrator_total_ms(state_values: Dict[str, Any]) -> Optional[int]:
    start_time = state_values.get("start_time")
    if isinstance(start_time, (int, float)):
        return int((time.time() - start_time) * 1000)
    return None


def _coerce_analyst_ok(result: Any) -> bool:
    if result is None:
        return True
    if hasattr(result, "ok"):
        try:
            return bool(getattr(result, "ok"))
        except Exception:
            return True
    if isinstance(result, dict):
        return bool(result.get("ok", True))
    return True


def _derive_failure_stage(
    *,
    interrupted: bool,
    state_values: Dict[str, Any],
) -> str:
    if interrupted:
        return "interrupted"

    plan_obj = dict(state_values.get("plan_obj") or {})
    retrieval_output = dict(state_values.get("retrieval_output") or {})
    retrieval_skipped_reason = _normalize_text(state_values.get("retrieval_skipped_reason"))
    retrieval_state = dict(state_values.get("retrieval_state") or {})
    route = _coerce_plan_route(plan_obj)
    planner_status = str(plan_obj.get("status") or "").strip().lower()
    if planner_status and planner_status != "completed":
        return "planner"

    if (
        route == "kb"
        and plan_obj.get("retrieval_needed")
        and not bool(retrieval_state)
        and retrieval_skipped_reason in {
            "MISSING_METADATA",
            "MISSING_TARGET_METADATA",
            "MISSING_RETRIEVAL_PLAN",
            "INVALID_RETRIEVAL_PLAN",
        }
    ):
        return "retrieval"

    if route == "kb" and plan_obj.get("retrieval_needed") and (
        isinstance(retrieval_output, dict) and retrieval_output.get("ok") is False
    ):
        return "retrieval"

    if not _coerce_analyst_ok(state_values.get("analyst_result")):
        return "analyst"

    return "none"


@lru_cache(maxsize=1)
def _get_orchestrator_graph(checkpointer_id: int):
    if _ORCHESTRATOR_CHECKPOINTER is None:
        raise RuntimeError("Orchestrator checkpointer is not initialized.")
    builder = StateGraph(OrchestratorState)
    builder.add_node("init", _init_node)
    builder.add_node("planner", _planner_graph_node)
    builder.add_node("planner_interrupt", _planner_interrupt_node)
    builder.add_node("planner_resume", _planner_resume_node)
    builder.add_node("planner_error", _planner_error_node)
    builder.add_node("check_retrieval_metadata", _check_retrieval_metadata_node)
    builder.add_node("retrieval", _retrieval_node)
    builder.add_node("structured_facts", _structured_facts_node)
    builder.add_node("build_packet_from_retrieval", _build_packet_from_retrieval_node)
    builder.add_node("build_packet_without_retrieval", _build_packet_without_retrieval_node)
    builder.add_node("attach_open_issues", _attach_open_issues_node)
    builder.add_node("analyst", _analyst_node)
    builder.add_node("finalize", _finalize_node)

    builder.set_entry_point("init")
    builder.add_edge("init", "planner")
    builder.add_conditional_edges(
        "planner",
        _route_after_planner_turn,
        {
            "planner_interrupt": "planner_interrupt",
            "check_retrieval_metadata": "check_retrieval_metadata",
            "structured_facts": "structured_facts",
            "build_packet_without_retrieval": "build_packet_without_retrieval",
            "planner_error": "planner_error",
        },
    )
    builder.add_edge("planner_interrupt", "planner_resume")
    builder.add_edge("planner_resume", "planner")
    builder.add_conditional_edges(
        "check_retrieval_metadata",
        _route_after_retrieval_metadata,
        {
            "retrieval": "retrieval",
            "structured_facts": "structured_facts",
            "build_packet_without_retrieval": "build_packet_without_retrieval",
        },
    )
    builder.add_conditional_edges(
        "retrieval",
        _route_after_retrieval_node,
        {
            "structured_facts": "structured_facts",
            "build_packet_from_retrieval": "build_packet_from_retrieval",
        },
    )
    builder.add_conditional_edges(
        "structured_facts",
        _route_after_structured_facts,
        {
            "build_packet_from_retrieval": "build_packet_from_retrieval",
            "build_packet_without_retrieval": "build_packet_without_retrieval",
        },
    )
    builder.add_edge("build_packet_from_retrieval", "attach_open_issues")
    builder.add_edge("build_packet_without_retrieval", "attach_open_issues")
    builder.add_conditional_edges(
        "attach_open_issues",
        _route_after_retrieval_attach_open_issues,
        {
            "analyst": "analyst",
            "finalize": "finalize",
        },
    )
    builder.add_edge("planner_error", "finalize")
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
    total_ms = state_values.get("total_ms")
    if total_ms is None:
        total_ms = _compute_orchestrator_total_ms(state_values)

    failure_stage = _derive_failure_stage(
        interrupted=interrupted,
        state_values=state_values,
    )
    ok = not interrupted and failure_stage == "none"
    packet = state_values.get("packet")
    open_issues = _dedupe_open_issue_payloads(
        state_values.get("open_issues"),
        getattr(packet, "open_issues", []),
    )
    out = {
        "run_id": run_id,
        "route": _coerce_plan_route(state_values.get("plan_obj") or {}),
        "status": (
            "interrupted"
            if interrupted
            else ("failed" if failure_stage != "none" else "completed")
        ),
        "ok": ok,
        "failure_stage": failure_stage,
        "open_issues": open_issues,
        "planner": state_values.get("planner_dump"),
        "planner_turn": state_values.get("planner_turn"),
        "retrieval": _compact_retrieval_result_for_user(
            retrieval_output=state_values.get("retrieval_output"),
        ),
        "structured_fact_results": list(state_values.get("structured_fact_results") or []),
        "analyst": _serialize_analyst_result(state_values.get("analyst_result")),
        "interrupt": _serialize_interrupts(getattr(state_snapshot, "interrupts", ()) or ()),
        "orchestrator_trace": {
            "total_ms": total_ms,
            "planner_timing_ms": dict(state_values.get("planner_timing_ms") or {}),
            "retrieval_timing_ms": dict(state_values.get("retrieval_timing_ms") or {}),
            "structured_fact_timing_ms": dict(state_values.get("structured_fact_timing_ms") or {}),
        },
    }
    if bool(state_values.get("include_evidence_trace")):
        out["evaluation_trace"] = {
            "analyst_packet": (
                packet.model_dump(mode="json")
                if isinstance(packet, AnalystPacket)
                else None
            )
        }
    return out


async def _invoke_orchestrator(
    payload: Any,
    *,
    run_id: str,
    planner: Optional[Any],
) -> Dict[str, Any]:
    global _ORCHESTRATOR_CHECKPOINTER, _ORCHESTRATOR_LAST_PRUNE_TS
    _ORCHESTRATOR_CHECKPOINTER = await _get_orchestrator_checkpointer()

    resolved_planner = await _resolve_runtime_planner(run_id=str(run_id).strip(), planner=planner)
    ttl_seconds = _orchestrator_checkpoint_ttl_seconds()
    prune_interval = _orchestrator_prune_interval_seconds()
    should_prune = (
        ttl_seconds > 0
        and prune_interval > 0
        and (time.time() - _ORCHESTRATOR_LAST_PRUNE_TS) >= prune_interval
    )
    if should_prune:
        _ORCHESTRATOR_LAST_PRUNE_TS = time.time()
        prune_task = asyncio.create_task(_prune_stale_orchestrator_runs(max_age_seconds=ttl_seconds))
        _BACKGROUND_TASKS.add(prune_task)
        prune_task.add_done_callback(_observe_background_task)

    graph = _get_orchestrator_graph(id(_ORCHESTRATOR_CHECKPOINTER))
    config = _graph_config(run_id=run_id, planner=resolved_planner)
    await graph.ainvoke(payload, config=config)
    state_snapshot = await graph.aget_state(config)
    output = _format_run_output(run_id=run_id, state_snapshot=state_snapshot)
    if output["status"] == "interrupted":
        return output
    try:
        clean_run_id = str(run_id).strip()
        await _delete_thread_checkpoints(
            saver=_ORCHESTRATOR_CHECKPOINTER,
            thread_id=clean_run_id,
        )
    except Exception:
        pass
    return output


async def run_multi_agent_orchestration(
    user_query: str,
    *,
    planner: Optional[Any] = None,
    analyst_model: str = "qwen2.5-14b-instruct-1m",
    tables_dir: str = "data/chunked",
    debug: bool = True,
    include_evidence_trace: bool = False,
) -> Dict[str, Any]:
    plan_id = f"run-{uuid.uuid4().hex[:8]}"
    resolved_tables_dir = _resolve_tables_dir(tables_dir)
    return await _invoke_orchestrator(
        {
            "user_query": user_query,
            "plan_id": plan_id,
            "analyst_model": analyst_model,
            "tables_dir": resolved_tables_dir,
            "debug": debug,
            "include_evidence_trace": include_evidence_trace,
        },
        run_id=plan_id,
        planner=planner,
    )


async def resume_multi_agent_orchestration(
    run_id: str,
    answers: Any,
    *,
    planner: Optional[Any] = None,
) -> Dict[str, Any]:
    return await _invoke_orchestrator(
        Command(resume=answers),
        run_id=str(run_id).strip(),
        planner=planner,
    )
