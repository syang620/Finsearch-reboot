"""
Analyst agent for filing-grounded computation and answer generation.

This module provides:
- packet builders from retrieval output -> AnalystPacket
- an MCP-backed analyst agent that can call `financial_evaluator`
- typed result/trace schemas for debugging and integration
"""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from rag10kq.run_analysis import load_table_data

from agents.contracts import (
    AnalystPacket,
    AnalysisTask,
    ContextItem,
    ContextQuality,
    FilingMetadata,
    FormType,
    OpenIssue,
    PlannerIntent,
    RetrieveTablesResponse,
    SourceRef,
)


SYSTEM_PROMPT = """You are a senior financial analyst in an SEC filings RAG system.
Use ONLY the provided context items. Do not use outside knowledge.

Rules:
1. If arithmetic/computation is needed, call the tool `financial_evaluator`.
2. Always pass explicit variable names and values to the tool.
3. If needed values are missing, do NOT guess; state what is missing.
4. End with a concise answer sentence.
"""


class AnalystComputation(BaseModel):
    expression: Optional[str] = None
    variables: Dict[str, str] = Field(default_factory=dict)
    result: Optional[float] = None


class AnalystTrace(BaseModel):
    timing_ms: Dict[str, int] = Field(default_factory=dict)
    used_financial_evaluator: bool = False
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    raw_message_count: int = 0


class AnalystRunResult(BaseModel):
    ok: bool = True
    answer: str
    intent: PlannerIntent
    metric: str
    computation: Optional[AnalystComputation] = None
    citations: List[SourceRef] = Field(default_factory=list)
    open_issues: List[OpenIssue] = Field(default_factory=list)
    trace: AnalystTrace = Field(default_factory=AnalystTrace)
    error: Optional[str] = None


def _default_financial_tool_script() -> str:
    # Prefer src-relative path from this module, fallback to cwd patterns.
    p_from_module = Path(__file__).resolve().parents[2] / "tools" / "financial_evaluator_mcp.py"
    if p_from_module.exists():
        return str(p_from_module)

    p1 = Path("src/tools/financial_evaluator_mcp.py")
    if p1.exists():
        return str(p1)

    p2 = Path("../src/tools/financial_evaluator_mcp.py")
    return str(p2)


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return str(content)


def _first_float(text: str) -> Optional[float]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _context_item_to_text(item: ContextItem, idx: int) -> str:
    payload = item.payload or {}
    table_name = (
        payload.get("table_name")
        or payload.get("section_title")
        or payload.get("title")
        or f"context_{idx}"
    )

    row_headers = payload.get("row_headers") or []
    if not isinstance(row_headers, list):
        row_headers = []
    row_headers_preview = ", ".join(str(x) for x in row_headers[:20])

    content = payload.get("table_markdown") or payload.get("content") or payload.get("text") or ""
    content = str(content)
    if len(content) > 12000:
        head = content[:6000]
        tail = content[-5500:]
        content = head + "\n... [truncated middle] ...\n" + tail

    src = item.source.model_dump(exclude_none=True)
    return (
        f"[Context {idx}]\n"
        f"table_name: {table_name}\n"
        f"source: {src}\n"
        f"row_headers_preview: {row_headers_preview}\n"
        f"content:\n{content}\n"
    )


def build_analyst_prompt(packet: AnalystPacket, *, max_context_items: int = 5) -> str:
    context_blocks = [
        _context_item_to_text(item, i + 1)
        for i, item in enumerate(packet.context_items[:max_context_items])
    ]
    context_text = "\n\n".join(context_blocks) if context_blocks else "[No context items provided]"

    meta = packet.metadata.model_dump(mode="json")
    analysis_task = packet.analysis_task.model_dump(mode="json")

    return (
        f"User query: {packet.user_query}\n"
        f"Intent: {packet.intent.value}\n"
        f"Metadata: {meta}\n"
        f"Analysis task: {analysis_task}\n\n"
        f"Context quality: {packet.context_quality.value}\n"
        f"Open issues: {[x.model_dump(mode='json') for x in packet.open_issues]}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Task:\n"
        "- Answer the user query grounded in the context above.\n"
        "- If calculation is required, call financial_evaluator before finalizing.\n"
        "- If missing data prevents a reliable calculation, explicitly say so.\n"
    )


def build_demo_packet(
    user_query: str,
    table_markdown: str,
    *,
    ticker: str = "AAPL",
    fiscal_year: int = 2024,
    form_type: FormType = FormType.TEN_K,
    metric: str = "total debt",
) -> AnalystPacket:
    """Convenience helper for quick testing without full orchestrator wiring."""
    return AnalystPacket(
        plan_id="demo-plan",
        user_query=user_query,
        intent=PlannerIntent.FILING_CALC,
        metadata=FilingMetadata(
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            doc_types=["table"],
        ),
        analysis_task=AnalysisTask(task_type="compute", metric=metric),
        context_quality=ContextQuality.MEDIUM,
        context_items=[
            ContextItem(
                source=SourceRef(ticker=ticker, fiscal_year=fiscal_year, form_type=form_type),
                payload={"table_name": "retrieved_table", "table_markdown": table_markdown},
            )
        ],
    )


def _extract_payload_from_retrieval_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    table_obj = entry.get("table", {})
    if isinstance(table_obj, dict):
        payload = table_obj.get("payload")
        if isinstance(payload, dict):
            return payload
        return table_obj
    payload = getattr(table_obj, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _table_dict_to_markdown(table_dict: Optional[Dict[str, Any]], max_rows: int = 40) -> str:
    if not isinstance(table_dict, dict):
        return ""
    try:
        df = pd.DataFrame(**table_dict)
    except Exception:
        return ""
    if len(df) > max_rows:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=True)
    except Exception:
        return str(df)


def _parse_agent_messages(messages: List[Any]) -> Dict[str, Any]:
    final_answer = _message_text(messages[-1]) if messages else ""

    tool_calls: List[Dict[str, Any]] = []
    used_financial_evaluator = False
    expression = None
    variables: Dict[str, str] = {}
    numeric_result: Optional[float] = None

    for msg in messages:
        tc = getattr(msg, "tool_calls", None) or []
        for call in tc:
            name = call.get("name")
            args = call.get("args") or {}
            tool_calls.append({"name": name, "args": args, "id": call.get("id")})
            if name == "financial_evaluator":
                used_financial_evaluator = True
                if isinstance(args, dict):
                    expression = args.get("expression") or expression
                    raw_vars = args.get("variables")
                    if isinstance(raw_vars, dict):
                        variables = {str(k): str(v) for k, v in raw_vars.items()}

        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "financial_evaluator":
            used_financial_evaluator = True
            maybe_float = _first_float(_message_text(msg))
            if maybe_float is not None:
                numeric_result = maybe_float

    return {
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "used_financial_evaluator": used_financial_evaluator,
        "expression": expression,
        "variables": variables,
        "numeric_result": numeric_result,
    }


def build_packet_from_retrieval_output(
    *,
    user_query: str,
    retrieval_output: Any,
    tables_dir: str = "../data/chunked",
    plan_id: str = "demo-plan",
    intent: PlannerIntent = PlannerIntent.FILING_CALC,
    metric: str = "financial metric",
    max_tables: int = 3,
) -> AnalystPacket:
    """
    Build AnalystPacket from retrieval output after contract validation + table hydration.
    """
    retrieval = (
        retrieval_output
        if isinstance(retrieval_output, RetrieveTablesResponse)
        else RetrieveTablesResponse.model_validate(retrieval_output)
    )

    metadata_used = retrieval.metadata_used or {}
    ticker = metadata_used.get("ticker")
    fiscal_year = metadata_used.get("fiscal_year")
    form_type_raw = metadata_used.get("form_type", "10-K")
    try:
        form_type = FormType(form_type_raw)
    except Exception:
        form_type = FormType.TEN_K

    context_items: List[ContextItem] = []
    open_issues: List[OpenIssue] = []

    if retrieval.error:
        open_issues.append(
            OpenIssue(
                code="RETRIEVAL_ERROR",
                message=str(retrieval.error),
                severity="error",
            )
        )

    top_tables = retrieval.top_tables or []
    for cand in top_tables[:max_tables]:
        entry = cand.model_dump(mode="python")
        payload = _extract_payload_from_retrieval_entry(entry)
        table_dict = load_table_data(entry, data_dir=tables_dir, verbose=False)
        table_markdown = _table_dict_to_markdown(table_dict)

        if table_dict is None:
            open_issues.append(
                OpenIssue(
                    code="TABLE_HYDRATION_FAILED",
                    message=f"Could not load table_dict for doc_id={payload.get('doc_id')}.",
                    severity="warning",
                )
            )

        merged_payload: Dict[str, Any] = {
            "table_name": entry.get("table_name"),
            "row_headers": entry.get("row_headers"),
            "total_score": entry.get("total_score"),
            "table_markdown": table_markdown,
            "table_dict": table_dict,
            **payload,
        }

        src = SourceRef(
            ticker=payload.get("ticker") or ticker,
            fiscal_year=payload.get("fiscal_year") or fiscal_year,
            form_type=form_type,
            section_path=payload.get("section_path"),
            doc_id=payload.get("doc_id"),
            table_id=str(payload.get("table_index")) if payload.get("table_index") is not None else None,
        )

        context_items.append(
            ContextItem(
                source=src,
                payload=merged_payload,
                total_score=entry.get("total_score"),
            )
        )

    if not context_items:
        open_issues.append(
            OpenIssue(
                code="NO_CONTEXT_ITEMS",
                message="No retrieval context could be converted into AnalystPacket context_items.",
                severity="error",
            )
        )

    context_quality = ContextQuality.MEDIUM
    max_score = retrieval.max_total_score
    if isinstance(max_score, (int, float)):
        if max_score >= 25:
            context_quality = ContextQuality.HIGH
        elif max_score < 10:
            context_quality = ContextQuality.LOW

    return AnalystPacket(
        plan_id=plan_id,
        user_query=user_query,
        intent=intent,
        metadata=FilingMetadata(
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            doc_types=["table"],
        ),
        analysis_task=AnalysisTask(task_type="compute", metric=metric),
        context_items=context_items,
        context_quality=context_quality,
        open_issues=open_issues,
    )


@dataclass
class AnalystAgent:
    model: str = "qwen3:14b"
    temperature: float = 0.0
    num_predict: int = 1024
    financial_tool_script: Optional[str] = None
    max_context_items: int = 5

    _client: Any = None
    _agent: Any = None

    async def abuild(self) -> "AnalystAgent":
        tool_script = self.financial_tool_script or _default_financial_tool_script()

        self._client = MultiServerMCPClient(
            {
                "fin_math": {
                    "transport": "stdio",
                    "command": "python",
                    "args": [tool_script],
                }
            }
        )
        tools = await self._client.get_tools()

        llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )

        self._agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
        )
        return self

    async def aclose(self) -> None:
        if self._client is None:
            return
        close_fn = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close_fn is None:
            return
        maybe = close_fn()
        if inspect.isawaitable(maybe):
            await maybe

    async def arun(self, packet: AnalystPacket, *, debug: bool = False) -> AnalystRunResult:
        if self._agent is None:
            await self.abuild()

        t0 = time.perf_counter()
        prompt = build_analyst_prompt(packet, max_context_items=self.max_context_items)
        t_prompt_ms = int((time.perf_counter() - t0) * 1000)

        t_llm0 = time.perf_counter()
        try:
            result = await self._agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        except Exception as e:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return AnalystRunResult(
                ok=False,
                answer="Analyst agent failed to produce an answer.",
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                citations=[c.source for c in packet.context_items],
                open_issues=packet.open_issues,
                trace=AnalystTrace(
                    timing_ms={
                        "build_prompt_ms": t_prompt_ms,
                        "agent_invoke_ms": int((time.perf_counter() - t_llm0) * 1000),
                        "total_ms": elapsed,
                    },
                    used_financial_evaluator=False,
                    tool_calls=[],
                    raw_message_count=0,
                ),
                error=str(e),
            )

        t_invoke_ms = int((time.perf_counter() - t_llm0) * 1000)
        retry_ms = 0

        messages = result.get("messages", []) or []
        parsed = _parse_agent_messages(messages)
        final_answer = parsed["final_answer"]
        tool_calls = parsed["tool_calls"]
        used_financial_evaluator = parsed["used_financial_evaluator"]
        expression = parsed["expression"]
        variables = parsed["variables"]
        numeric_result = parsed["numeric_result"]

        if packet.analysis_task.task_type == "compute" and not used_financial_evaluator:
            retry_prompt = (
                prompt
                + "\n\nIMPORTANT: This is a compute task. You MUST call financial_evaluator before final answer."
                + " Use numbers from context and show the computed result."
            )
            t_retry0 = time.perf_counter()
            try:
                retry_result = await self._agent.ainvoke({"messages": [HumanMessage(content=retry_prompt)]})
                retry_ms = int((time.perf_counter() - t_retry0) * 1000)
                retry_messages = retry_result.get("messages", []) or []
                if retry_messages:
                    messages = retry_messages
                    parsed = _parse_agent_messages(messages)
                    final_answer = parsed["final_answer"]
                    tool_calls = parsed["tool_calls"]
                    used_financial_evaluator = parsed["used_financial_evaluator"]
                    expression = parsed["expression"]
                    variables = parsed["variables"]
                    numeric_result = parsed["numeric_result"]
            except Exception:
                retry_ms = int((time.perf_counter() - t_retry0) * 1000)

        elapsed = int((time.perf_counter() - t0) * 1000)

        result_open_issues = list(packet.open_issues)
        if packet.analysis_task.task_type == "compute" and not used_financial_evaluator:
            result_open_issues.append(
                OpenIssue(
                    code="COMPUTE_TOOL_NOT_USED",
                    message="Compute task completed without financial_evaluator call.",
                    severity="warning",
                )
            )

        if debug:
            print(
                f"[analyst_timing_ms] build_prompt_ms={t_prompt_ms} agent_invoke_ms={t_invoke_ms} "
                f"agent_retry_ms={retry_ms} total_ms={elapsed}"
            )
            print(f"[analyst_debug] tool_calls={len(tool_calls)} used_financial_evaluator={used_financial_evaluator}")

        computation = None
        if used_financial_evaluator or expression or variables:
            computation = AnalystComputation(
                expression=expression,
                variables=variables,
                result=numeric_result,
            )

        return AnalystRunResult(
            ok=True,
            answer=final_answer,
            intent=packet.intent,
            metric=packet.analysis_task.metric,
            computation=computation,
            citations=[c.source for c in packet.context_items],
            open_issues=result_open_issues,
            trace=AnalystTrace(
                timing_ms={
                    "build_prompt_ms": t_prompt_ms,
                    "agent_invoke_ms": t_invoke_ms,
                    "agent_retry_ms": retry_ms,
                    "total_ms": elapsed,
                },
                used_financial_evaluator=used_financial_evaluator,
                tool_calls=tool_calls,
                raw_message_count=len(messages),
            ),
            error=None,
        )

