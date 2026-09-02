"""
Analyst agent for filing-grounded computation and answer generation.

This module provides:
- packet builders from retrieval output -> AnalystPacket
- an MCP-backed analyst agent that can call `financial_evaluator`
- typed result/trace schemas for debugging and integration
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import math
import os
import re
import socket
import sys
import time
import warnings
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence

from typing_extensions import TypedDict

from pydantic import BaseModel, Field, ValidationError, field_validator

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except Exception:  # pragma: no cover - optional dependency
    ClientSession = None
    sse_client = None

from ingestion.chunk_paths import resolve_chunk_file
from llm_client import build_chat_model
from .table_loader import load_table_data

from agents.contracts import (
    AnalystPacket,
    AnalysisTask,
    ContextItem,
    ContextItemKind,
    ContextQuality,
    FilingMetadata,
    FormType,
    OpenIssue,
    PlannerIntent,
    RetrieveTablesResponse,
    Severity,
    SourceRef,
    StructuredFactEvidence,
)


ANALYST_CONTEXT_ITEM_LIMIT = 5


SYSTEM_PROMPT = """You are a senior financial analyst in an SEC filings RAG system.
Use ONLY the provided context items. Do not use outside knowledge.

Rules:
1. If arithmetic/computation is needed, call the tool `financial_evaluator`.
2. `financial_evaluator` accepts exactly one scalar arithmetic expression per call.
3. Never send assignments, multiple lines, multiple statements, explanatory text, or undefined variable names to `financial_evaluator`.
4. Always pass explicit variable names and values to the tool. Variable names must be valid Python identifiers without spaces.
5. If you need multiple derived values, call `financial_evaluator` once per derived value.
6. After you have the needed calculator result(s), stop calling tools and call `FinalAnswer` immediately.
7. Do NOT call `FinalAnswer` in the same step as `financial_evaluator`.
8. Cite only context IDs you actually used.
9. If needed values are missing, do NOT guess; use status="insufficient_data".
10. If the tool errors or returns no reliable result, use status="tool_error".
"""

SYSTEM_PROMPT_NO_TOOLS = """You are a senior financial analyst in an SEC filings RAG system.
Use ONLY the provided context items. Do not use outside knowledge.

Rules:
1. The `financial_evaluator` tool is currently unavailable. Do not invent tool calls.
2. If arithmetic/computation is needed, use status="tool_error" and explain the tool is unavailable.
3. You must finish by calling the tool `FinalAnswer`.
4. Cite only context IDs you actually used.
5. If needed values are missing, do NOT guess; use status="insufficient_data".
"""


class AnalystComputation(BaseModel):
    expression: Optional[str] = None
    variables: Dict[str, str] = Field(default_factory=dict)
    result: Optional[float] = None


class AnalystCompareRow(BaseModel):
    target_id: Optional[str] = None
    label: str
    value: Optional[str] = None
    context_ids: List[str]

    @field_validator("label")
    @classmethod
    def _label_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("label must be non-empty")
        return value


class AnalystCitation(BaseModel):
    context_id: str
    source: SourceRef


class AnalystStructuredAnswer(BaseModel):
    status: Literal["ok", "insufficient_data", "tool_error"]
    answer: str
    used_context_ids: List[str]
    missing_values: List[str]
    confidence: Optional[float] = None
    calculation: Optional[AnalystComputation] = None
    compare_rows: List[AnalystCompareRow]

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("answer")
    @classmethod
    def _answer_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("answer must be non-empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


class AnalystTrace(BaseModel):
    timing_ms: Dict[str, int] = Field(default_factory=dict)
    used_financial_evaluator: bool = False
    tool_calls: List["_SerializedToolCall"] = Field(default_factory=list)
    raw_message_count: int = 0
    final_output_valid: bool = False
    tool_error_code: Optional[str] = None


class AnalystRunResult(BaseModel):
    ok: bool = True
    status: str = "ok"
    answer: str
    intent: PlannerIntent
    metric: str
    used_context_ids: List[str] = Field(default_factory=list)
    missing_values: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    computation: Optional[AnalystComputation] = None
    compare_rows: List[AnalystCompareRow] = Field(default_factory=list)
    citations: List[AnalystCitation] = Field(default_factory=list)
    open_issues: List[OpenIssue] = Field(default_factory=list)
    trace: AnalystTrace = Field(default_factory=AnalystTrace)
    error: Optional[str] = None


class _SerializedToolCall(TypedDict, total=False):
    name: str
    args: Dict[str, Any]
    id: Optional[str]


class _SerializedAnalystComputation(TypedDict):
    expression: Optional[str]
    variables: Dict[str, str]
    result: float


class _SerializedInvalidToolCall(TypedDict, total=False):
    name: str
    id: Optional[str]
    error: Optional[str]


class _DeferredToolMessage(TypedDict, total=False):
    content: str
    name: str
    tool_call_id: Optional[str]
    artifact: Any
    status: str


class FinancialEvaluatorArgs(BaseModel):
    variables: Dict[str, str] = Field(default_factory=dict)
    expression: str

    @field_validator("expression")
    @classmethod
    def _expression_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("expression must be non-empty")
        return value


async def _financial_evaluator_placeholder(**_: Any) -> str:
    return "financial_evaluator is handled by the analyst workflow."


async def _final_answer_placeholder(**_: Any) -> str:
    return "FinalAnswer is handled by the analyst workflow."


FINANCIAL_EVALUATOR_TOOL = StructuredTool.from_function(
    coroutine=_financial_evaluator_placeholder,
    name="financial_evaluator",
    description="Evaluate a numeric financial expression using explicit named variables from filing context.",
    args_schema=FinancialEvaluatorArgs,
)

FINAL_ANSWER_TOOL = StructuredTool.from_function(
    coroutine=_final_answer_placeholder,
    name="FinalAnswer",
    description=(
        "Finish the analyst run with the final validated answer. "
        "Always include status, answer, used_context_ids, missing_values, confidence, calculation, and compare_rows."
    ),
    args_schema=AnalystStructuredAnswer,
)


@dataclass
class _FinancialToolRuntime:
    tool_script: str
    timeout_s: float = 120.0
    url: Optional[str] = None
    _process: Any = None
    _sse_cm: Any = None
    _session_cm: Any = None
    _session: Any = None

    @classmethod
    async def create(
        cls,
        *,
        tool_script: str,
        timeout_s: float,
        url: Optional[str] = None,
    ) -> "_FinancialToolRuntime":
        runtime = cls(tool_script=tool_script, timeout_s=timeout_s, url=str(url or "").strip() or None)
        if runtime.url:
            await runtime._connect(runtime.url)
            return runtime

        if ClientSession is None or sse_client is None:
            raise RuntimeError("MCP SSE client dependencies are unavailable.")

        host = "127.0.0.1"
        last_error: Optional[Exception] = None
        for _attempt in range(2):
            port = _pick_free_local_port()
            runtime.url = f"http://{host}:{port}/sse"
            runtime._process = await asyncio.create_subprocess_exec(
                sys.executable,
                tool_script,
                "--transport",
                "sse",
                "--host",
                host,
                "--port",
                str(port),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=None,
            )

            deadline = time.monotonic() + min(max(timeout_s, 5.0), 30.0)
            while time.monotonic() < deadline:
                if runtime._process.returncode is not None:
                    break
                try:
                    await runtime._connect(runtime.url)
                    return runtime
                except Exception as exc:  # pragma: no cover - startup race
                    last_error = exc
                    await asyncio.sleep(0.25)
            await runtime.aclose()

        if last_error is not None:
            raise RuntimeError(f"Unable to start financial evaluator SSE runtime: {last_error}") from last_error
        raise RuntimeError("Unable to start financial evaluator SSE runtime.")

    async def _connect(self, url: str) -> None:
        if ClientSession is None or sse_client is None:
            raise RuntimeError("MCP SSE client dependencies are unavailable.")

        if self._session is not None:
            return

        self._sse_cm = sse_client(
            url,
            timeout=max(5.0, min(15.0, self.timeout_s)),
            sse_read_timeout=max(300.0, self.timeout_s),
        )
        read, write = await self._sse_cm.__aenter__()
        try:
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            await asyncio.wait_for(self._session.initialize(), timeout=self.timeout_s)
        except Exception:
            await self.aclose()
            raise

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("Financial evaluator runtime is not connected.")
        result = await asyncio.wait_for(
            self._session.call_tool(str(name), arguments=dict(args or {})),
            timeout=self.timeout_s,
        )

        structured = getattr(result, "structured_content", None)
        if structured is None:
            structured = getattr(result, "structuredContent", None)

        blocks = getattr(result, "content", []) or []
        content_parts: List[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if text is not None:
                content_parts.append(str(text))
            elif isinstance(block, dict) and "text" in block:
                content_parts.append(str(block["text"]))
            else:
                content_parts.append(str(block))
        content = "\n".join(x for x in content_parts if str(x).strip()).strip()

        artifact = structured
        if artifact is None and content:
            artifact = _extract_json_payload(content)
        if isinstance(artifact, dict) and set(artifact.keys()) == {"result"} and isinstance(artifact.get("result"), dict):
            artifact = artifact.get("result")

        is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
        if is_error and not isinstance(artifact, dict):
            artifact = {"error": content or "financial_evaluator returned an error"}
        return {
            "content": content,
            "artifact": artifact,
            "status": "error" if is_error else "success",
        }

    async def aclose(self) -> None:
        try:
            if self._session_cm is not None:
                await self._session_cm.__aexit__(None, None, None)
        finally:
            self._session_cm = None
            self._session = None
            try:
                if self._sse_cm is not None:
                    await self._sse_cm.__aexit__(None, None, None)
            finally:
                self._sse_cm = None
                if self._process is not None:
                    process = self._process
                    self._process = None
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except Exception:
                            process.kill()
                            await process.wait()


def _pick_free_local_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _default_financial_tool_script() -> str:
    p_from_module = (
        Path(__file__).resolve().parents[2] / "mcp_server" / "tools" / "financial_evaluator.py"
    )
    if p_from_module.exists():
        return str(p_from_module)

    p0 = Path("src/mcp_server/tools/financial_evaluator.py")
    if p0.exists():
        return str(p0)

    p1 = Path("../src/mcp_server/tools/financial_evaluator.py")
    if not p1.exists():
        warnings.warn(
            f"Unable to resolve financial evaluator MCP script; using fallback path {p1}",
            RuntimeWarning,
            stacklevel=2,
        )
    return str(p1)


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


_FLOAT_TOKEN_RE = re.compile(r"\(?\$?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?\)?")
_MISSING_CALC_RE = re.compile(
    r"\b(?:missing data|insufficient data|insufficient information|cannot compute|couldn't compute|"
    r"could not compute|unable to compute|not enough data|cannot reliably compute|data is incomplete)\b",
    re.IGNORECASE,
)
_RETRYABLE_TOOL_ERROR_RE = re.compile(
    r"\b(?:syntaxerror|invalid syntax|nameerror|not defined|malformed json|malformed arguments|"
    r"invalid arguments|unexpected eof|unexpected end|unmatched|parse error)\b",
    re.IGNORECASE,
)
_ROW_DOC_ID_RE = re.compile(r"::(?:table_row|row)::(\d+)$")


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "").replace("$", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    if not text:
        return None
    try:
        value_f = float(text)
        return value_f / 100.0 if is_percent else value_f
    except Exception:
        return None


def _first_float_from_object(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        for key in ("result", "output", "value"):
            value = payload.get(key)
            parsed = _to_float(value)
            if parsed is not None:
                return parsed
        for value in payload.values():
            parsed = _first_float_from_object(value)
            if parsed is not None:
                return parsed
        return None
    if isinstance(payload, list):
        for value in payload:
            parsed = _first_float_from_object(value)
            if parsed is not None:
                return parsed
        return None
    return _to_float(payload)


def _extract_json_payload(text: str) -> Any:
    txt = str(text or "").strip()
    if not txt:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        txt = fenced.group(1).strip()

    start = txt.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(txt)):
        ch = txt[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = txt[start : idx + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _is_missing_data_for_computation(text: str) -> bool:
    if not text:
        return False
    return bool(_MISSING_CALC_RE.search(text))


def _first_float(text: str) -> Optional[float]:
    s = str(text or "").strip()
    if not s:
        return None

    parsed_json = _extract_json_payload(s)
    if isinstance(parsed_json, (dict, list)):
        value = _first_float_from_object(parsed_json)
        if value is not None:
            return value

    parsed = _to_float(s)
    if parsed is not None:
        return parsed

    for key in ("result", "output", "value"):
        match = re.search(rf"(?i){key}[^0-9A-Za-z]{{0,20}}?({_FLOAT_TOKEN_RE.pattern})", s)
        if match:
            parsed = _to_float(match.group(1))
            if parsed is not None:
                return parsed
    return None


def _truncate_lines(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) <= 1:
        return text[:max_chars].rstrip() + "\n... [truncated] ..."
    out_lines: List[str] = []
    used = 0
    for line in lines:
        extra = len(line) + (1 if out_lines else 0)
        if used + extra > max_chars:
            if not out_lines:
                return line[:max_chars].rstrip() + "\n... [truncated] ..."
            break
        out_lines.append(line)
        used += extra
    return "\n".join(out_lines + ["... [truncated] ..."])


def _target_id_from_values(
    ticker: Any,
    fiscal_year: Any,
    form_type: Any = None,
    fallback: Optional[str] = None,
) -> Optional[str]:
    t = str(ticker or "").strip().upper()
    fy = None
    try:
        fy = int(fiscal_year) if fiscal_year is not None else None
    except Exception:
        fy = None
    form = str(form_type or "").strip().upper()
    if t and fy is not None:
        if form:
            return f"{t}:{fy}:{form}"
        return f"{t}:{fy}"
    return fallback


def render_structured_fact_evidence(evidence: StructuredFactEvidence) -> str:
    """Render typed structured evidence without flattening it into KB payload text."""

    fields = {
        "metric_id": evidence.metric_id,
        "metric_label": evidence.metric_label,
        "status": evidence.status,
        "value": evidence.value,
        "unit": evidence.unit,
        "ticker": evidence.ticker,
        "fiscal_year": evidence.fiscal_year,
        "form_type": evidence.form_type.value if evidence.form_type is not None else None,
        "accession_number": evidence.accession_number,
        "report_date": evidence.report_date,
        "filed_date": evidence.filed_date,
        "source_url": evidence.source_url,
        "components": evidence.components,
        "missing_component_groups": evidence.missing_component_groups,
    }
    return "structured_fact:\n" + json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _context_item_to_text(item: ContextItem, idx: int) -> str:
    src = item.source.model_dump(exclude_none=True, mode="json")
    target_id = item.target_id or _target_id_from_values(
        item.source.ticker,
        item.source.fiscal_year,
        item.source.form_type.value if item.source.form_type is not None else None,
    )
    if item.kind == ContextItemKind.STRUCTURED_FACT:
        return (
            f"[Context {idx} | id={item.context_id} | target_id={target_id or 'unknown'}]\n"
            f"source: {src}\n"
            f"{render_structured_fact_evidence(item.structured_fact)}\n"
        )

    payload = item.payload or {}
    table_name = (
        payload.get("table_name")
        or payload.get("section_title")
        or payload.get("title")
        or item.context_id
        or f"context_{idx}"
    )
    row_headers = payload.get("row_headers") or []
    if not isinstance(row_headers, list):
        row_headers = []
    row_headers_preview = ", ".join(str(x) for x in row_headers[:20])
    content = payload.get("table_markdown") or payload.get("content") or payload.get("text") or ""
    content = _truncate_lines(str(content), max_chars=12000)

    return (
        f"[Context {idx} | id={item.context_id} | target_id={target_id or 'unknown'}]\n"
        f"table_name: {table_name}\n"
        f"source: {src}\n"
        f"row_headers_preview: {row_headers_preview}\n"
        f"content:\n{content}\n"
    )


def build_analyst_prompt(
    packet: AnalystPacket,
    *,
    max_context_items: int = ANALYST_CONTEXT_ITEM_LIMIT,
    tools_available: bool = True,
) -> str:
    grouped: Dict[str, List[str]] = {}
    for i, item in enumerate(packet.context_items[:max_context_items], start=1):
        target_id = item.target_id or "default"
        grouped.setdefault(target_id, []).append(_context_item_to_text(item, i))

    context_blocks: List[str] = []
    for target_id, blocks in grouped.items():
        context_blocks.append(f"Target: {target_id}\n" + "\n\n".join(blocks))
    context_text = "\n\n".join(context_blocks) if context_blocks else "[No context items provided]"

    meta = packet.metadata.model_dump(mode="json")
    analysis_task = packet.analysis_task.model_dump(mode="json")
    targets = list(packet.targets or [])
    definition_notes = [str(note).strip() for note in list(packet.analysis_task.definition_notes or []) if str(note).strip()]
    definition_notes_block = ""
    if definition_notes:
        definition_notes_block = "Metric definition notes:\n" + "\n".join(f"- {note}" for note in definition_notes) + "\n\n"
    calculation_instruction = (
        "- If calculation is required, call financial_evaluator before finishing.\n"
        "- financial_evaluator accepts exactly one scalar arithmetic expression per call.\n"
        "- Do not use assignments, multiple lines, or invented variable names in the expression.\n"
        "- If several derived values are needed, compute them one scalar at a time, then call FinalAnswer.\n"
        if tools_available
        else '- If arithmetic is required and financial_evaluator is unavailable, return status="tool_error" instead of inventing a calculation.\n'
    )
    degradation_notice = packet.degradation.model_dump(mode="json")
    return (
        f"User query: {packet.user_query}\n"
        f"Intent: {packet.intent.value}\n"
        f"Metadata: {meta}\n"
        f"Targets: {targets}\n"
        f"Analysis task: {analysis_task}\n\n"
        f"{definition_notes_block}"
        f"Context quality: {packet.context_quality.value}\n"
        f"Evidence lanes: {packet.lanes.model_dump(mode='json')}\n"
        f"Degradation: {degradation_notice}\n"
        f"Open issues: {[x.model_dump(mode='json') for x in packet.open_issues]}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Task:\n"
        "- Answer the user query grounded in the context above.\n"
        "- Typed structured facts are direct evidence; use their numeric value and SEC provenance as shown.\n"
        "- Follow the typed degradation notice exactly; do not claim coverage from unavailable lanes.\n"
        "- For compare or table-style requests, fill compare_rows with one row per target/value.\n"
        f"{calculation_instruction}"
        "- Finish by calling FinalAnswer with the validated final result.\n"
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
    target_id = _target_id_from_values(ticker, fiscal_year, form_type.value)
    return AnalystPacket(
        plan_id="demo-plan",
        user_query=user_query,
        intent=PlannerIntent.FILING_CALC,
        metadata=FilingMetadata(
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
        ),
        analysis_task=AnalysisTask(task_type="compute", metric=metric),
        targets=[
            {
                "target_id": target_id,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "form_type": form_type.value,
            }
        ],
        context_quality=ContextQuality.MEDIUM,
        context_items=[
            ContextItem(
                context_id="ctx_1",
                target_id=target_id,
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


def _classify_retrieval_entry_type(*, entry: Dict[str, Any], payload: Dict[str, Any]) -> str:
    doc_type = payload.get("doc_type")
    if isinstance(doc_type, str):
        normalized = doc_type.strip().lower()
        if normalized in {"table", "table_row"}:
            return "table"
        if normalized in {"text", "text_chunk"}:
            return "text"

    payload_doc_id = payload.get("doc_id")
    if isinstance(payload_doc_id, str):
        if "::table::" in payload_doc_id or "::table_row::" in payload_doc_id or "::row::" in payload_doc_id:
            return "table"

    entry_doc_type = entry.get("doc_type")
    if isinstance(entry_doc_type, str):
        normalized = entry_doc_type.strip().lower()
        if normalized in {"table", "table_row"}:
            return "table"
        if normalized in {"text", "text_chunk"}:
            return "text"

    entry_doc_id = entry.get("doc_id")
    if isinstance(entry_doc_id, str):
        if "::table::" in entry_doc_id or "::table_row::" in entry_doc_id or "::row::" in entry_doc_id:
            return "table"

    return "unsupported"


def _normalize_text_content(payload: Dict[str, Any]) -> str:
    for candidate in (
        payload.get("content"),
        payload.get("text"),
        payload.get("summary"),
        payload.get("match_text"),
        payload.get("value"),
    ):
        if isinstance(candidate, str):
            text = candidate.strip()
            if text:
                return text
    return ""


def _table_dict_to_markdown(table_dict: Optional[Dict[str, Any]], max_rows: int = 40) -> str:
    if not isinstance(table_dict, dict):
        return ""
    if pd is None:
        try:
            return json.dumps(table_dict, ensure_ascii=False, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to stringify table_dict without pandas: {exc}", RuntimeWarning, stacklevel=2)
            return ""
    try:
        try:
            df = pd.DataFrame(table_dict)
        except Exception:
            df = pd.DataFrame.from_dict(table_dict, orient="index")
    except Exception as exc:
        warnings.warn(f"Failed to convert table_dict to DataFrame: {exc}", RuntimeWarning, stacklevel=2)
        try:
            return json.dumps(table_dict, ensure_ascii=False, indent=2)
        except Exception:
            return ""
    if len(df) > max_rows:
        df = df.head(max_rows)
    try:
        return df.to_markdown(index=True)
    except Exception as exc:
        warnings.warn(f"Failed to render table markdown: {exc}", RuntimeWarning, stacklevel=2)
        try:
            return df.to_csv(index=True)
        except Exception:
            return str(df)


def _normalize_tool_artifact(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        parsed = _extract_json_payload(raw)
        return parsed if parsed is not None else raw
    structured = getattr(raw, "structured_content", None)
    if structured is None:
        structured = getattr(raw, "structuredContent", None)
    if structured is not None:
        return _normalize_tool_artifact(structured)
    artifact = getattr(raw, "artifact", None)
    if artifact is not None:
        return _normalize_tool_artifact(artifact)
    content = getattr(raw, "content", None)
    if content is not None:
        text = _message_text(raw)
        parsed = _extract_json_payload(text)
        return parsed if parsed is not None else text
    return raw


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return text
    try:
        rendered = repr(exc).strip()
    except Exception:
        rendered = ""
    if rendered:
        return rendered
    return exc.__class__.__name__


def _structured_tool_payload(msg: ToolMessage) -> Any:
    artifact = getattr(msg, "artifact", None)
    normalized = _normalize_tool_artifact(artifact)
    if normalized is not None:
        return normalized
    return _extract_json_payload(_message_text(msg))


def _parse_structured_final_answer(text: str) -> Optional[AnalystStructuredAnswer]:
    payload = _extract_json_payload(text)
    if not isinstance(payload, dict):
        return None
    try:
        return AnalystStructuredAnswer.model_validate(payload)
    except ValidationError:
        return None


def _serialize_final_output(final_output: Optional[Any]) -> Optional[Dict[str, Any]]:
    if final_output is None:
        return None
    if isinstance(final_output, dict):
        return dict(final_output)
    return final_output.model_dump(mode="json")


def _validated_final_output_from_parsed(parsed: Dict[str, Any] | _SerializedAnalystParsedState) -> Optional[AnalystStructuredAnswer]:
    payload = parsed.get("final_output")
    if isinstance(payload, AnalystStructuredAnswer):
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        return AnalystStructuredAnswer.model_validate(payload)
    except ValidationError:
        return None


def _serialize_parsed_state(parsed: Dict[str, Any]) -> _SerializedAnalystParsedState:
    serialized: _SerializedAnalystParsedState = dict(parsed)
    serialized["final_output"] = _serialize_final_output(parsed.get("final_output"))
    return serialized


def _normalize_tool_call(raw_call: Any) -> _SerializedToolCall:
    if isinstance(raw_call, dict):
        raw_args = raw_call.get("args") or {}
        args = raw_args if isinstance(raw_args, dict) else {}
        return {
            "name": str(raw_call.get("name") or "").strip(),
            "args": args,
            "id": str(raw_call.get("id")) if raw_call.get("id") is not None else None,
        }

    raw_args = getattr(raw_call, "args", None) or {}
    args = raw_args if isinstance(raw_args, dict) else {}
    raw_id = getattr(raw_call, "id", None)
    return {
        "name": str(getattr(raw_call, "name", "") or "").strip(),
        "args": args,
        "id": str(raw_id) if raw_id is not None else None,
    }


def _normalize_invalid_tool_call(raw_call: Any) -> _SerializedInvalidToolCall:
    if isinstance(raw_call, dict):
        return {
            "name": str(raw_call.get("name") or "").strip(),
            "id": str(raw_call.get("id")) if raw_call.get("id") is not None else None,
            "error": str(raw_call.get("error")) if raw_call.get("error") is not None else None,
        }

    raw_id = getattr(raw_call, "id", None)
    raw_error = getattr(raw_call, "error", None)
    return {
        "name": str(getattr(raw_call, "name", "") or "").strip(),
        "id": str(raw_id) if raw_id is not None else None,
        "error": str(raw_error) if raw_error is not None else None,
    }


def _parse_agent_messages(messages: List[Any]) -> Dict[str, Any]:
    final_output: Optional[AnalystStructuredAnswer] = None
    final_output_error: Optional[str] = None
    final_tool_called = False
    final_answer_text = ""
    tool_calls: List[_SerializedToolCall] = []
    used_financial_evaluator = False
    expression = None
    variables: Dict[str, str] = {}
    numeric_result: Optional[float] = None
    tool_error: Optional[str] = None
    tool_error_code: Optional[str] = None
    successful_computations: List[_SerializedAnalystComputation] = []
    evaluator_call_inputs: Dict[str, Dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, HumanMessage):
            final_output = None
            final_output_error = None
            final_tool_called = False
            final_answer_text = ""

        tc = getattr(msg, "tool_calls", None) or []
        for call in tc:
            name = str(call.get("name") or "").strip()
            args = call.get("args") or {}
            tool_calls.append({"name": name, "args": args, "id": call.get("id")})
            if name == "financial_evaluator":
                used_financial_evaluator = True
                if isinstance(args, dict):
                    expression = args.get("expression") or expression
                    raw_vars = args.get("variables")
                    if isinstance(raw_vars, dict):
                        variables = {str(k): str(v) for k, v in raw_vars.items()}
                    call_id = call.get("id")
                    if call_id is not None:
                        evaluator_call_inputs[str(call_id)] = {
                            "expression": args.get("expression"),
                            "variables": dict(raw_vars) if isinstance(raw_vars, dict) else {},
                        }
            elif name == "FinalAnswer":
                final_tool_called = True
                try:
                    final_output = AnalystStructuredAnswer.model_validate(args)
                    final_output_error = None
                    final_answer_text = final_output.answer
                except ValidationError as exc:
                    final_output = None
                    final_output_error = str(exc)

        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "financial_evaluator":
            used_financial_evaluator = True
            structured = _structured_tool_payload(msg)
            tool_call_id = getattr(msg, "tool_call_id", None)
            call_inputs = evaluator_call_inputs.get(str(tool_call_id), {}) if tool_call_id is not None else {}
            if isinstance(structured, dict):
                expression = structured.get("expression") or call_inputs.get("expression") or expression
                raw_vars = structured.get("variables")
                if not isinstance(raw_vars, dict):
                    raw_vars = call_inputs.get("variables")
                if isinstance(raw_vars, dict):
                    variables = {str(k): str(v) for k, v in raw_vars.items()}
                raw_error = structured.get("error")
                raw_error_code = structured.get("error_code")
                message_failed = bool(raw_error) or str(getattr(msg, "status", "") or "").strip().lower() == "error"
                if message_failed:
                    tool_error = str(raw_error or _message_text(msg) or "financial_evaluator returned an error")
                    tool_error_code = str(raw_error_code).strip() if raw_error_code is not None else None
                    numeric_result = None
                    maybe_float = None
                else:
                    tool_error = None
                    tool_error_code = None
                    maybe_float = _first_float_from_object(structured)
                    explicit_result = _to_float(structured.get("result"))
                    if explicit_result is not None:
                        successful_computations.append(
                            {
                                "expression": str(expression) if expression is not None else None,
                                "variables": dict(variables),
                                "result": explicit_result,
                            }
                        )
            else:
                maybe_float = _first_float(_message_text(msg))
                tool_error = None
                tool_error_code = None
            if maybe_float is not None:
                numeric_result = maybe_float

    if not final_answer_text:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_answer_text = _message_text(msg)
                if final_answer_text:
                    break
        if not final_answer_text and messages:
            final_answer_text = _message_text(messages[-1])

    if final_output is None:
        fallback = _parse_structured_final_answer(final_answer_text)
        if fallback is not None:
            final_output = fallback
            final_output_error = None
            final_answer_text = fallback.answer

    return {
        "final_answer": final_answer_text,
        "final_output": final_output,
        "final_output_valid": final_output is not None,
        "final_output_error": final_output_error,
        "final_tool_called": final_tool_called,
        "calculation_blocked": _is_missing_data_for_computation(final_answer_text)
        or (final_output is not None and final_output.status == "insufficient_data"),
        "tool_calls": tool_calls,
        "used_financial_evaluator": used_financial_evaluator,
        "expression": expression,
        "variables": variables,
        "numeric_result": numeric_result,
        "successful_computations": successful_computations,
        "tool_error": tool_error,
        "tool_error_code": tool_error_code,
    }


class _AnalystWorkflowState(TypedDict, total=False):
    packet: AnalystPacket
    messages: Annotated[List[AnyMessage], add_messages]
    attempt: int
    max_attempts: int
    tool_rounds: int
    max_tool_rounds: int
    parsed: "_SerializedAnalystParsedState"
    error: Optional[str]
    timing_ms: Dict[str, int]
    should_retry: bool
    tool_setup_error: Optional[str]
    pending_tool_calls: List[_SerializedToolCall]
    tools_available: bool
    tool_round_limit_exceeded: bool
    ordered_tool_calls: List[_SerializedToolCall]
    deferred_tool_messages: List[_DeferredToolMessage]


class _SerializedAnalystParsedState(TypedDict, total=False):
    final_answer: str
    final_output: Optional[Dict[str, Any]]
    final_output_valid: bool
    final_output_error: Optional[str]
    final_tool_called: bool
    calculation_blocked: bool
    tool_calls: List[_SerializedToolCall]
    used_financial_evaluator: bool
    expression: Optional[str]
    variables: Dict[str, str]
    numeric_result: Optional[float]
    successful_computations: List[_SerializedAnalystComputation]
    tool_error: Optional[str]
    tool_error_code: Optional[str]
    tool_round_limit_exceeded: bool


def _retry_correction_message() -> str:
    return (
        "Your previous response did not satisfy the required JSON contract for this task. "
        "If a calculator is needed, call financial_evaluator, then finish by calling FinalAnswer with valid structured arguments only."
    )


_EXPRESSION_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_UNSUPPORTED_FUNCTION_NAMES = {"max", "min", "sum", "abs", "round"}


def _validate_financial_evaluator_args(args: Any) -> Optional[Dict[str, str]]:
    if not isinstance(args, dict):
        return {
            "error": "financial_evaluator arguments must be a JSON object with `expression` and optional `variables`.",
            "error_code": "invalid_input",
        }

    expression = str(args.get("expression") or "").strip()
    if not expression:
        return {
            "error": "financial_evaluator requires a non-empty `expression`.",
            "error_code": "invalid_input",
        }
    if "\n" in expression or "\r" in expression or ";" in expression or "=" in expression:
        return {
            "error": (
                "financial_evaluator accepts exactly one scalar arithmetic expression. "
                "Do not use assignments, multiple lines, or multiple statements."
            ),
            "error_code": "invalid_expression",
        }

    raw_variables = args.get("variables") or {}
    if raw_variables is None:
        raw_variables = {}
    if not isinstance(raw_variables, dict):
        return {
            "error": "financial_evaluator `variables` must be an object mapping variable names to numeric strings.",
            "error_code": "invalid_input",
        }

    variables = {str(key).strip() for key in raw_variables.keys() if str(key).strip()}
    referenced = set(_EXPRESSION_IDENTIFIER_RE.findall(expression))
    unknown = sorted(name for name in referenced if name not in variables)
    if unknown:
        unsupported_functions = [name for name in unknown if name.lower() in _UNSUPPORTED_FUNCTION_NAMES]
        if unsupported_functions:
            return {
                "error": (
                    "financial_evaluator does not support functions like "
                    + ", ".join(unsupported_functions)
                    + ". Use only arithmetic operators on provided variables. "
                    "If the needed comparison can be read directly from the provided values, call FinalAnswer without another tool call."
                ),
                "error_code": "unsupported_function",
            }
        return {
            "error": (
                "financial_evaluator expression referenced variables that were not provided: "
                + ", ".join(unknown)
                + ". Use only variables passed in `variables`, or add the missing values explicitly."
            ),
            "error_code": "unknown_variable",
        }
    return None


def _retry_reason_message(packet: Optional[AnalystPacket], parsed: Dict[str, Any]) -> str:
    tool_error = str(parsed.get("tool_error") or "").strip()
    tool_error_code = str(parsed.get("tool_error_code") or "").strip().lower()
    if tool_error:
        if tool_error_code == "unsupported_function":
            return (
                "financial_evaluator does not support functions like max, min, sum, abs, or round. "
                "Use only arithmetic operators on the provided variables. If the comparison is already obvious from the values in context, stop calling tools and call FinalAnswer now."
            )
        if tool_error_code in {"invalid_expression", "invalid_syntax", "unsupported_characters"}:
            return (
                "The financial_evaluator only accepts one scalar arithmetic expression per call. "
                "Do not use assignments, newlines, or multiple statements. Rewrite the calculation as a single expression. "
                "If several values are needed, compute one scalar at a time, then call FinalAnswer."
            )
        if tool_error_code == "unknown_variable":
            return (
                "The financial_evaluator expression referenced variables that were not passed in `variables`. "
                "Rewrite the expression to use only provided variable names, or add the missing variables explicitly. "
                "Compute one scalar at a time, then call FinalAnswer."
            )
        return (
            "The financial_evaluator returned an error: "
            f"{tool_error}. Correct the expression or variables, then call FinalAnswer with valid structured arguments."
        )
    final_output_error = str(parsed.get("final_output_error") or "").strip()
    if final_output_error:
        return (
            "Your FinalAnswer payload failed validation: "
            f"{final_output_error}. Call FinalAnswer again with valid structured arguments only."
        )
    if _requires_calculation(packet) and not parsed.get("used_financial_evaluator"):
        return (
            "This task requires grounded calculation. Call financial_evaluator first, "
            "then call FinalAnswer after the tool result is available."
        )
    if _requires_calculation(packet) and parsed.get("numeric_result") is None:
        return (
            "The calculation completed without a reliable numeric result. "
            "Call financial_evaluator again with corrected inputs, then finish with FinalAnswer."
        )
    if _requires_calculation(packet) and parsed.get("numeric_result") is not None:
        return (
            "You already have a valid financial_evaluator result available. "
            "Do not call financial_evaluator again unless exactly one additional scalar is still missing. "
            "Use the available calculator result(s) and call FinalAnswer now."
        )
    return _retry_correction_message()


def _requires_calculation(packet: Optional[AnalystPacket]) -> bool:
    if packet is None:
        return False
    task = getattr(packet, "analysis_task", None)
    if task is None:
        return False
    if bool(getattr(task, "requires_calculation", False)):
        return True
    return str(getattr(task, "task_type", "") or "").strip().lower() == "compute"


def _computation_from_parsed_state(parsed: Dict[str, Any] | _SerializedAnalystParsedState) -> Optional[AnalystComputation]:
    if not (
        parsed.get("used_financial_evaluator")
        or parsed.get("expression")
        or parsed.get("variables")
        or parsed.get("numeric_result") is not None
    ):
        return None
    return AnalystComputation(
        expression=parsed.get("expression"),
        variables=parsed.get("variables") or {},
        result=parsed.get("numeric_result"),
    )


def _successful_computations_from_parsed_state(
    parsed: Dict[str, Any] | _SerializedAnalystParsedState,
) -> List[AnalystComputation]:
    computations: List[AnalystComputation] = []
    raw_computations = parsed.get("successful_computations")
    if isinstance(raw_computations, list):
        for raw_computation in raw_computations:
            if not isinstance(raw_computation, dict):
                continue
            try:
                computation = AnalystComputation.model_validate(raw_computation)
            except ValidationError:
                continue
            if computation.result is not None:
                computations.append(computation)
        return computations

    legacy_computation = _computation_from_parsed_state(parsed)
    if legacy_computation is not None and legacy_computation.result is not None and not parsed.get("tool_error"):
        computations.append(legacy_computation)
    return computations


def _computations_match(
    structured: Optional[AnalystComputation],
    tool_computation: Optional[AnalystComputation],
) -> bool:
    if structured is None or tool_computation is None:
        return True
    structured_result = structured.result
    tool_result = tool_computation.result
    if structured_result is None or tool_result is None:
        return structured_result is None and tool_result is None
    return math.isclose(float(structured_result), float(tool_result), rel_tol=1e-3, abs_tol=1e-9)


def _normalized_computation_expression(expression: Optional[str]) -> Optional[str]:
    try:
        parsed = ast.parse(str(expression or "").strip(), mode="eval")
    except (SyntaxError, ValueError, TypeError):
        return None
    return ast.dump(parsed, annotate_fields=True, include_attributes=False)


def _normalized_computation_variable(value: str) -> Optional[Decimal]:
    text = str(value).strip().replace(",", "").replace("$", "")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    if not text:
        return None
    try:
        normalized = Decimal(text)
    except InvalidOperation:
        return None
    if not normalized.is_finite():
        return None
    return normalized / Decimal(100) if is_percent else normalized


def _computation_variables_match(
    expression: Optional[str],
    structured: Dict[str, str],
    tool_variables: Dict[str, str],
) -> bool:
    referenced_variables = set(_EXPRESSION_IDENTIFIER_RE.findall(str(expression or "")))
    if not referenced_variables.issubset(structured) or not referenced_variables.issubset(
        tool_variables
    ):
        return False
    for name in referenced_variables:
        structured_value = structured[name]
        tool_value = tool_variables[name]
        structured_number = _normalized_computation_variable(structured_value)
        tool_number = _normalized_computation_variable(tool_value)
        if structured_number is not None and tool_number is not None:
            if structured_number != tool_number:
                return False
        elif str(structured_value).strip() != str(tool_value).strip():
            return False
    return True


def _computation_provenance_matches(
    structured: AnalystComputation,
    tool_computation: AnalystComputation,
) -> bool:
    structured_expression = _normalized_computation_expression(structured.expression)
    return (
        structured_expression is not None
        and structured_expression
        == _normalized_computation_expression(tool_computation.expression)
        and _computation_variables_match(
            structured.expression,
            structured.variables,
            tool_computation.variables,
        )
    )


def _resolve_matching_successful_computation(
    structured: Optional[AnalystComputation],
    successful_computations: List[AnalystComputation],
) -> tuple[Optional[AnalystComputation], bool]:
    if structured is None:
        return None, False
    numeric_matches = [
        computation
        for computation in successful_computations
        if _computations_match(structured, computation)
    ]
    if len(numeric_matches) == 1:
        return numeric_matches[0], False
    if not numeric_matches:
        return None, False

    provenance_matches = [
        computation
        for computation in numeric_matches
        if _computation_provenance_matches(structured, computation)
    ]
    if len(provenance_matches) == 1:
        return provenance_matches[0], False
    return None, True


def _tool_error_is_retryable(
    parsed: Dict[str, Any] | _SerializedAnalystParsedState,
    final_output: Optional[AnalystStructuredAnswer] = None,
) -> bool:
    if final_output is not None and final_output.status != "tool_error":
        return False
    tool_error_code = str(parsed.get("tool_error_code") or "").strip().lower()
    if tool_error_code:
        return tool_error_code in {
            "invalid_syntax",
            "invalid_expression",
            "unknown_variable",
            "invalid_input",
            "unsupported_characters",
        }
    tool_error = str(parsed.get("tool_error") or "").strip()
    if not tool_error:
        return False
    return bool(_RETRYABLE_TOOL_ERROR_RE.search(tool_error))


def _parsed_tool_error_code(parsed: Dict[str, Any] | _SerializedAnalystParsedState) -> Optional[str]:
    raw = parsed.get("tool_error_code")
    return str(raw) if raw is not None else None


def _should_retry_response(
    packet: Optional[AnalystPacket],
    parsed: Dict[str, Any] | _SerializedAnalystParsedState,
    attempt: int,
    max_attempts: int,
    *,
    tools_available: bool,
) -> bool:
    if packet is None or attempt >= max_attempts:
        return False
    if parsed.get("tool_round_limit_exceeded"):
        return False
    if not parsed.get("final_output_valid"):
        return True

    if not _requires_calculation(packet):
        return False
    if not tools_available:
        return False
    if parsed.get("calculation_blocked"):
        return False
    final_output = _validated_final_output_from_parsed(parsed)
    if final_output is not None and final_output.status == "tool_error":
        return _tool_error_is_retryable(parsed, final_output)
    if not parsed.get("used_financial_evaluator"):
        return True
    if parsed.get("tool_error"):
        return True
    if parsed.get("numeric_result") is None:
        return True
    return False


def _should_retry_compute(
    packet: Optional[AnalystPacket],
    parsed: Dict[str, Any] | _SerializedAnalystParsedState,
    attempt: int,
    max_attempts: int,
    *,
    tools_available: bool,
) -> bool:
    if packet is None or not _requires_calculation(packet):
        return False
    if not tools_available:
        return False
    if parsed.get("calculation_blocked"):
        return False
    return _should_retry_response(
        packet,
        parsed,
        attempt,
        max_attempts,
        tools_available=tools_available,
    )


def _collect_packet_targets(retrieval_data: Dict[str, Any], metadata_used: Dict[str, Any]) -> List[Dict[str, Any]]:
    targets = [dict(target) for target in (retrieval_data.get("targets") or []) if isinstance(target, dict)]
    if targets:
        for target in targets:
            if not target.get("target_id"):
                target["target_id"] = _target_id_from_values(
                    target.get("ticker"),
                    target.get("fiscal_year"),
                    target.get("form_type"),
                )
        return targets

    target_id = _target_id_from_values(
        metadata_used.get("ticker"),
        metadata_used.get("fiscal_year"),
        metadata_used.get("form_type"),
    )
    if target_id or metadata_used:
        return [
            {
                "target_id": target_id,
                "ticker": metadata_used.get("ticker"),
                "fiscal_year": metadata_used.get("fiscal_year"),
                "form_type": metadata_used.get("form_type"),
            }
        ]
    return []


def _resolve_target_id_from_packet_targets(
    packet_targets: Sequence[Dict[str, Any]],
    *,
    ticker: Any,
    fiscal_year: Any,
    form_type: Any,
) -> Optional[str]:
    normalized_ticker = str(ticker or "").strip().upper()
    normalized_form = str(form_type or "").strip().upper()
    try:
        normalized_year = int(fiscal_year) if fiscal_year is not None else None
    except Exception:
        normalized_year = None

    for target in packet_targets:
        if not isinstance(target, dict):
            continue
        target_ticker = str(target.get("ticker") or "").strip().upper()
        target_form = str(target.get("form_type") or "").strip().upper()
        try:
            target_year = int(target.get("fiscal_year")) if target.get("fiscal_year") is not None else None
        except Exception:
            target_year = None
        if normalized_ticker and target_ticker and normalized_ticker != target_ticker:
            continue
        if normalized_year is not None and target_year is not None and normalized_year != target_year:
            continue
        if normalized_form and target_form and normalized_form != target_form:
            continue
        target_id = target.get("target_id")
        if target_id is not None and str(target_id).strip():
            return str(target_id).strip()

    return _target_id_from_values(ticker, fiscal_year, form_type)


def _row_evidence_text(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    matched = payload.get("match_text")
    if isinstance(matched, str) and matched.strip():
        parts.append(f"matched_row: {matched.strip()}")
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"summary: {summary.strip()}")
    value = payload.get("value")
    if isinstance(value, str) and value.strip():
        parts.append(f"value: {value.strip()}")
    return "\n".join(parts)


def build_packet_from_retrieval_output(
    *,
    user_query: str,
    retrieval_output: Any,
    tables_dir: str = "data/chunked",
    plan_id: str = "demo-plan",
    intent: PlannerIntent = PlannerIntent.FILING_CALC,
    analysis_task: Any = None,
    metric: str = "financial metric",
    max_tables: int = 3,
) -> AnalystPacket:
    retrieval_data = retrieval_output if isinstance(retrieval_output, dict) else {}
    retrieval = (
        retrieval_output
        if isinstance(retrieval_output, RetrieveTablesResponse)
        else RetrieveTablesResponse.model_validate(retrieval_output)
    )

    metadata_used = retrieval.metadata_used or {}
    ticker = metadata_used.get("ticker")
    fiscal_year = metadata_used.get("fiscal_year")
    form_type_raw = metadata_used.get("form_type")
    form_type: Optional[FormType] = None
    if form_type_raw is not None:
        try:
            form_type = FormType(form_type_raw)
        except Exception:
            form_type = None

    packet_targets = _collect_packet_targets(retrieval_data, metadata_used)
    context_items: List[ContextItem] = []
    open_issues: List[OpenIssue] = []

    if retrieval.error:
        open_issues.append(
            OpenIssue(
                code="RETRIEVAL_ERROR",
                message=str(retrieval.error),
                severity=Severity.ERROR,
            )
        )

    top_tables = retrieval.top_tables or []
    for cand in top_tables:
        if len(context_items) >= max_tables:
            break
        entry = cand.model_dump(mode="python")
        payload = _extract_payload_from_retrieval_entry(entry)
        entry_type = _classify_retrieval_entry_type(entry=entry, payload=payload)
        if entry_type == "unsupported":
            continue

        entry_ticker = payload.get("ticker") or ticker
        entry_fiscal_year = payload.get("fiscal_year") or fiscal_year
        entry_form_type = payload.get("form_type") or (form_type.value if form_type is not None else None)
        target_id = _resolve_target_id_from_packet_targets(
            packet_targets,
            ticker=entry_ticker,
            fiscal_year=entry_fiscal_year,
            form_type=entry_form_type,
        )
        context_id = f"ctx_{len(context_items) + 1}"

        if entry_type == "table":
            table_dict = load_table_data(entry, data_dir=tables_dir, verbose=False)
            table_markdown = _table_dict_to_markdown(table_dict)
            if table_dict is None:
                doc_id = payload.get("doc_id")
                prefix = payload.get("prefix")
                table_index = payload.get("table_index")
                if prefix is None and isinstance(doc_id, str) and "::" in doc_id:
                    prefix = doc_id.split("::", 1)[0]
                if table_index is None and isinstance(doc_id, str):
                    table_match = re.search(r"::table::(\d+)$", doc_id)
                    row_match = _ROW_DOC_ID_RE.search(doc_id)
                    if table_match:
                        table_index = int(table_match.group(1))
                    elif row_match:
                        table_index = int(row_match.group(1))
                attempted_file = None
                if prefix is not None:
                    resolved = resolve_chunk_file(tables_dir, prefix, f"{prefix}.tables.jsonl")
                    attempted_file_path = resolved if resolved is not None else Path(tables_dir) / f"{prefix}.tables.jsonl"
                    attempted_file = str(attempted_file_path.resolve())
                open_issues.append(
                    OpenIssue(
                        code="TABLE_HYDRATION_FAILED",
                        message=(
                            "Could not load table_dict for "
                            f"doc_id={doc_id}, table_index={table_index}, attempted_file={attempted_file}."
                        ),
                        severity=Severity.WARNING,
                    )
                )
                continue
            if not table_markdown:
                open_issues.append(
                    OpenIssue(
                        code="TABLE_MARKDOWN_EMPTY",
                        message="Loaded table_dict but failed to render markdown for analyst context.",
                        severity=Severity.WARNING,
                    )
                )
                continue

            row_evidence = _row_evidence_text(payload)
            if row_evidence:
                table_markdown = row_evidence + "\n\n" + table_markdown

            merged_payload: Dict[str, Any] = {
                "table_name": entry.get("table_name"),
                "row_headers": entry.get("row_headers"),
                "total_score": entry.get("total_score"),
                "table_markdown": table_markdown,
                "matched_row_text": payload.get("match_text"),
                **payload,
            }
            merged_payload.pop("table_dict", None)
            source_form_type = None
            if form_type is not None:
                source_form_type = form_type
            elif entry_form_type in {x.value for x in FormType}:
                source_form_type = FormType(entry_form_type)
            src = SourceRef(
                ticker=entry_ticker,
                fiscal_year=entry_fiscal_year,
                form_type=source_form_type,
                section_path=payload.get("section_path"),
                doc_id=payload.get("doc_id"),
                table_id=str(payload.get("table_index")) if payload.get("table_index") is not None else None,
            )
            context_items.append(
                ContextItem(
                    context_id=context_id,
                    target_id=target_id,
                    kind=ContextItemKind.TABLE,
                    source=src,
                    payload=merged_payload,
                    total_score=entry.get("total_score"),
                )
            )
            continue

        text_content = _normalize_text_content(payload)
        if not text_content:
            open_issues.append(
                OpenIssue(
                    code="EMPTY_TEXT_CONTEXT",
                    message="Skipping retrieval result with missing text content.",
                    severity=Severity.WARNING,
                )
            )
            continue

        merged_payload = {
            "table_name": entry.get("table_name"),
            "row_headers": entry.get("row_headers"),
            "total_score": entry.get("total_score"),
            "content": text_content,
            "table_markdown": text_content,
            **payload,
        }
        source_form_type = None
        if form_type is not None:
            source_form_type = form_type
        elif entry_form_type in {x.value for x in FormType}:
            source_form_type = FormType(entry_form_type)
        src = SourceRef(
            ticker=entry_ticker,
            fiscal_year=entry_fiscal_year,
            form_type=source_form_type,
            section_path=payload.get("section_path"),
            doc_id=payload.get("doc_id"),
            table_id=str(payload.get("table_index")) if payload.get("table_index") is not None else None,
        )
        context_items.append(
            ContextItem(
                context_id=context_id,
                target_id=target_id,
                kind=ContextItemKind.TEXT,
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
                severity=Severity.ERROR,
            )
        )

    context_quality = ContextQuality.MEDIUM
    max_score = retrieval.max_total_score
    if isinstance(max_score, (int, float)):
        if max_score >= 25:
            context_quality = ContextQuality.HIGH
        elif max_score < 10:
            context_quality = ContextQuality.LOW

    resolved_analysis_task = (
        analysis_task
        if isinstance(analysis_task, AnalysisTask)
        else AnalysisTask.model_validate(analysis_task or {"task_type": "compute", "metric": metric})
    )

    return AnalystPacket(
        plan_id=plan_id,
        user_query=user_query,
        intent=intent,
        metadata=FilingMetadata(
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
        ),
        analysis_task=resolved_analysis_task,
        targets=packet_targets,
        context_items=context_items,
        context_quality=context_quality,
        open_issues=open_issues,
    )


@dataclass
class AnalystAgent:
    model: str = "qwen2.5-14b-instruct-1m"
    temperature: float = 0.0
    num_predict: int = 2048
    timeout_s: float = 120.0
    financial_tool_script: Optional[str] = None
    financial_mcp_url: Optional[str] = None
    max_context_items: int = ANALYST_CONTEXT_ITEM_LIMIT
    max_attempts: int = 2
    max_tool_rounds: int = 6

    _graph: Any = None
    _build_lock: Any = None
    _bound_model_override: Any = None
    _tool_map: Optional[Dict[str, Any]] = None
    _tools_available: bool = False
    _tool_setup_error: Optional[str] = None
    _tool_runtime: Optional[_FinancialToolRuntime] = None
    _tool_runtime_lock: Any = None

    @property
    def is_ready(self) -> bool:
        return self._graph is not None

    def _num_predict_for_task(self, task_type: str) -> int:
        normalized = str(task_type or "").strip().lower()
        if normalized in {"compare", "trend"}:
            return max(int(self.num_predict), 4096)
        return max(int(self.num_predict), 2048)

    def _build_bound_model(self, packet: AnalystPacket, *, tools_available: bool) -> Any:
        if self._bound_model_override is not None and hasattr(self._bound_model_override, "ainvoke"):
            return self._bound_model_override
        llm = build_chat_model(
            model=self.model,
            temperature=self.temperature,
            num_predict=self._num_predict_for_task(packet.analysis_task.task_type),
            timeout=self.timeout_s,
        )
        tools = [FINAL_ANSWER_TOOL]
        if tools_available:
            tools = [FINANCIAL_EVALUATOR_TOOL, FINAL_ANSWER_TOOL]
        return llm.bind_tools(tools)

    async def _invoke_tool(self, tool: Any, args: Dict[str, Any]) -> Any:
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(args)
        if hasattr(tool, "invoke"):
            result = tool.invoke(args)
            if inspect.isawaitable(result):
                return await result
            return result
        if hasattr(tool, "arun"):
            return await tool.arun(**args)
        if hasattr(tool, "run"):
            return tool.run(**args)
        raise RuntimeError(f"Tool {getattr(tool, 'name', '<unknown>')} is not invokable.")

    async def _ensure_tool_runtime(self) -> Optional[_FinancialToolRuntime]:
        if self._bound_model_override is not None or self._tool_map:
            return None
        if self._tool_runtime_lock is None:
            self._tool_runtime_lock = asyncio.Lock()
        async with self._tool_runtime_lock:
            if self._tool_runtime is not None:
                self._tools_available = True
                self._tool_setup_error = None
                return self._tool_runtime
            tool_script = self.financial_tool_script or _default_financial_tool_script()
            runtime = await _FinancialToolRuntime.create(
                tool_script=tool_script,
                timeout_s=self.timeout_s,
                url=self.financial_mcp_url or os.getenv("FINSEARCH_ANALYST_FINANCIAL_MCP_SSE_URL"),
            )
            self._tool_runtime = runtime
            self._tools_available = True
            self._tool_setup_error = None
            return runtime

    async def _invalidate_tool_runtime(self, runtime: Optional[_FinancialToolRuntime] = None) -> None:
        if self._tool_runtime_lock is None:
            self._tool_runtime_lock = asyncio.Lock()
        async with self._tool_runtime_lock:
            active = self._tool_runtime
            if runtime is not None and active is not runtime:
                return
            self._tool_runtime = None
            self._tools_available = False
        if active is not None:
            await active.aclose()

    async def _call_managed_tool_runtime(self, runtime: Optional[_FinancialToolRuntime], name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._tool_runtime_lock is None:
            self._tool_runtime_lock = asyncio.Lock()
        async with self._tool_runtime_lock:
            active_runtime = self._tool_runtime if self._tool_runtime is not None else runtime
            if active_runtime is None:
                raise RuntimeError("financial_evaluator runtime is unavailable")
        try:
            result = await active_runtime.call_tool(name, args)
            self._tools_available = True
            self._tool_setup_error = None
            return result
        except Exception:
            async with self._tool_runtime_lock:
                if self._tool_runtime is active_runtime:
                    self._tool_runtime = None
                    self._tools_available = False
                else:
                    active_runtime = None
            if active_runtime is not None:
                await active_runtime.aclose()
            raise

    def _build_workflow(self, *, checkpointer: Any = None) -> Any:
        def _route_after_model(state: _AnalystWorkflowState) -> str:
            if state.get("error"):
                return "finalize"
            if state.get("pending_tool_calls") and not state.get("tool_round_limit_exceeded"):
                return "execute_tools"
            return "assess"

        def _route_after_assess(state: _AnalystWorkflowState) -> str:
            if state.get("error"):
                return "finalize"
            return "call_model" if state.get("should_retry") else "finalize"

        def _timing_update(state: Dict[str, Any], key: str, value: int) -> Dict[str, int]:
            timing = dict(state.get("timing_ms") or {})
            timing[key] = int(timing.get(key, 0)) + int(value)
            return timing

        async def build_prompt_node(state: _AnalystWorkflowState) -> Dict[str, Any]:
            packet = state["packet"]
            t0 = time.perf_counter()
            prompt = build_analyst_prompt(
                packet,
                max_context_items=self.max_context_items,
                tools_available=bool(state.get("tools_available")),
            )
            return {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT if state.get("tools_available") else SYSTEM_PROMPT_NO_TOOLS),
                    HumanMessage(content=prompt),
                ],
                "attempt": 0,
                "tool_rounds": 0,
                "max_attempts": max(1, int(state.get("max_attempts") or self.max_attempts)),
                "max_tool_rounds": max(1, int(state.get("max_tool_rounds") or self.max_tool_rounds)),
                "timing_ms": _timing_update({"timing_ms": {}}, "build_prompt_ms", int((time.perf_counter() - t0) * 1000)),
            }

        async def call_model_node(state: _AnalystWorkflowState) -> Dict[str, Any]:
            attempt = int(state.get("attempt", 0))
            packet = state["packet"]
            messages = list(state.get("messages") or [])
            t0 = time.perf_counter()
            try:
                bound_model = self._build_bound_model(packet, tools_available=bool(state.get("tools_available")))
                ai_message = await asyncio.wait_for(
                    bound_model.ainvoke(messages),
                    timeout=self.timeout_s,
                )
                synthetic_tool_messages: List[ToolMessage] = []
                raw_tool_calls = [_normalize_tool_call(call) for call in list(getattr(ai_message, "tool_calls", None) or [])]
                invalid_tool_calls = [
                    _normalize_invalid_tool_call(call) for call in list(getattr(ai_message, "invalid_tool_calls", None) or [])
                ]
                pending_tool_calls = [
                    call for call in raw_tool_calls if str(call.get("name") or "").strip() != "FinalAnswer"
                ]
                had_requested_nonfinal_tools = bool(pending_tool_calls)
                has_pending_nonfinal_tools = had_requested_nonfinal_tools
                mixed_with_nonfinal_tools = has_pending_nonfinal_tools
                tool_round_limit_exceeded = bool(pending_tool_calls) and int(state.get("tool_rounds", 0)) >= int(
                    state.get("max_tool_rounds", self.max_tool_rounds)
                )
                deferred_tool_messages: List[_DeferredToolMessage] = []
                for call in invalid_tool_calls:
                    artifact = {
                        "error": f"Invalid arguments: {call.get('error') or 'Malformed JSON arguments.'} Please fix your JSON syntax."
                    }
                    synthetic_tool_messages.append(
                        ToolMessage(
                            content=json.dumps(artifact, ensure_ascii=False),
                            name=str(call.get("name") or "unknown_tool"),
                            tool_call_id=str(call.get("id") or "invalid_tool_call"),
                            artifact=artifact,
                            status="error",
                        )
                    )
                if tool_round_limit_exceeded:
                    for call in pending_tool_calls:
                        artifact = {"error": "financial_evaluator tool-call round limit exceeded for this run."}
                        synthetic_tool_messages.append(
                            ToolMessage(
                                content=json.dumps(artifact, ensure_ascii=False),
                                name=str(call.get("name") or ""),
                                tool_call_id=call.get("id"),
                                artifact=artifact,
                                status="error",
                            )
                        )
                    pending_tool_calls = []
                    has_pending_nonfinal_tools = False
                for call in raw_tool_calls:
                    if str(call.get("name") or "").strip() == "FinalAnswer":
                        if mixed_with_nonfinal_tools:
                            artifact = {"error": "Do not call FinalAnswer in the same step as financial_evaluator. Wait for tool results first."}
                            content = json.dumps(artifact, ensure_ascii=False)
                            status = "error"
                        else:
                            try:
                                AnalystStructuredAnswer.model_validate(call.get("args") or {})
                                artifact = {"status": "recorded"}
                                content = "FinalAnswer recorded."
                                status = "success"
                            except ValidationError as exc:
                                artifact = {"error": f"FinalAnswer validation failed: {exc}"}
                                content = json.dumps(artifact, ensure_ascii=False)
                                status = "error"
                        payload: _DeferredToolMessage = {
                            "content": content,
                            "name": "FinalAnswer",
                            "tool_call_id": call.get("id"),
                            "artifact": artifact,
                            "status": status,
                        }
                        if mixed_with_nonfinal_tools and pending_tool_calls:
                            deferred_tool_messages.append(payload)
                        else:
                            synthetic_tool_messages.append(ToolMessage(**payload))
                updated_messages = [*messages, ai_message, *synthetic_tool_messages]
                parsed = _parse_agent_messages(updated_messages)
                if pending_tool_calls or (tool_round_limit_exceeded and had_requested_nonfinal_tools):
                    parsed["final_output"] = None
                    parsed["final_output_valid"] = False
                    parsed["final_tool_called"] = False
                parsed["tool_round_limit_exceeded"] = tool_round_limit_exceeded
                err = None
            except asyncio.TimeoutError:
                ai_message = None
                updated_messages = messages
                parsed = {}
                pending_tool_calls = []
                tool_round_limit_exceeded = False
                err = f"ANALYST_MODEL_TIMEOUT after {self.timeout_s:.1f}s"
            except Exception as exc:
                ai_message = None
                updated_messages = messages
                parsed = {}
                pending_tool_calls = []
                tool_round_limit_exceeded = False
                err = _error_text(exc)

            elapsed = int((time.perf_counter() - t0) * 1000)
            update: Dict[str, Any] = {
                "attempt": attempt,
                "parsed": _serialize_parsed_state(parsed),
                "error": err,
                "pending_tool_calls": pending_tool_calls,
                "tool_round_limit_exceeded": tool_round_limit_exceeded,
            }
            if ai_message is not None:
                update["messages"] = [ai_message, *synthetic_tool_messages]
            update["ordered_tool_calls"] = raw_tool_calls if ai_message is not None else []
            update["deferred_tool_messages"] = deferred_tool_messages if ai_message is not None else []
            update["timing_ms"] = _timing_update(
                state,
                "agent_invoke_ms" if attempt <= 0 else "agent_retry_ms",
                elapsed,
            )
            return update

        async def execute_tools_node(state: _AnalystWorkflowState) -> Dict[str, Any]:
            pending_tool_calls: List[_SerializedToolCall] = list(state.get("pending_tool_calls") or [])
            deferred_tool_messages: List[_DeferredToolMessage] = list(state.get("deferred_tool_messages") or [])
            tool_message_map: Dict[str, ToolMessage] = {}
            tool_map = dict(self._tool_map or {})
            t0 = time.perf_counter()
            for call in pending_tool_calls:
                tool_name = str(call.get("name") or "").strip()
                args = call.get("args") or {}
                tool_call_id = call.get("id")
                if tool_name != "financial_evaluator":
                    artifact = {"error": f"Unknown tool: {tool_name}"}
                    tool_message_map[str(tool_call_id)] = ToolMessage(
                        content=json.dumps(artifact, ensure_ascii=False),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        artifact=artifact,
                        status="error",
                    )
                    continue
                validation_error = _validate_financial_evaluator_args(args)
                if validation_error is not None:
                    tool_message_map[str(tool_call_id)] = ToolMessage(
                        content=json.dumps(validation_error, ensure_ascii=False),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        artifact=validation_error,
                        status="error",
                    )
                    continue
                injected_tool = tool_map.get(tool_name)
                if injected_tool is not None:
                    try:
                        raw_result = await self._invoke_tool(injected_tool, args if isinstance(args, dict) else {})
                        artifact = _normalize_tool_artifact(raw_result)
                        content = json.dumps(artifact, ensure_ascii=False) if isinstance(artifact, (dict, list)) else str(raw_result)
                        tool_message_map[str(tool_call_id)] = ToolMessage(
                            content=content,
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            artifact=artifact,
                            status="error" if isinstance(artifact, dict) and artifact.get("error") else "success",
                        )
                    except Exception as exc:
                        artifact = {"error": _error_text(exc)}
                        tool_message_map[str(tool_call_id)] = ToolMessage(
                            content=json.dumps(artifact, ensure_ascii=False),
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            artifact=artifact,
                            status="error",
                        )
                    continue
                try:
                    raw_result = await self._call_managed_tool_runtime(None, tool_name, args if isinstance(args, dict) else {})
                    artifact = raw_result.get("artifact")
                    content = str(raw_result.get("content") or "")
                    status = str(raw_result.get("status") or "success")
                    tool_message_map[str(tool_call_id)] = ToolMessage(
                        content=content or (json.dumps(artifact, ensure_ascii=False) if artifact is not None else ""),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        artifact=artifact,
                        status=status,
                    )
                except Exception as exc:
                    artifact = {"error": _error_text(exc)}
                    tool_message_map[str(tool_call_id)] = ToolMessage(
                        content=json.dumps(artifact, ensure_ascii=False),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        artifact=artifact,
                        status="error",
                    )

            tool_messages: List[ToolMessage] = []
            deferred_tool_message_map = {
                str(item.get("tool_call_id") or ""): ToolMessage(
                    content=str(item.get("content") or ""),
                    name=str(item.get("name") or ""),
                    tool_call_id=item.get("tool_call_id"),
                    artifact=item.get("artifact"),
                    status=str(item.get("status") or "success"),
                )
                for item in deferred_tool_messages
                if isinstance(item, dict)
            }
            ordered_tool_calls = list(state.get("ordered_tool_calls") or [])
            if ordered_tool_calls:
                for call in ordered_tool_calls:
                    call_id = str(call.get("id") or "")
                    if call_id in tool_message_map:
                        tool_messages.append(tool_message_map.pop(call_id))
                    elif call_id in deferred_tool_message_map:
                        tool_messages.append(deferred_tool_message_map.pop(call_id))
            tool_messages.extend(tool_message_map.values())
            tool_messages.extend(deferred_tool_message_map.values())

            elapsed = int((time.perf_counter() - t0) * 1000)
            updated_messages = [*(state.get("messages") or []), *tool_messages]
            parsed = _parse_agent_messages(updated_messages)
            parsed["tool_round_limit_exceeded"] = False
            return {
                "messages": tool_messages,
                "parsed": _serialize_parsed_state(parsed),
                "pending_tool_calls": [],
                "ordered_tool_calls": [],
                "deferred_tool_messages": [],
                "tool_rounds": int(state.get("tool_rounds", 0)) + (1 if pending_tool_calls else 0),
                "timing_ms": _timing_update(state, "tool_exec_ms", elapsed),
            }

        async def assess_node(state: _AnalystWorkflowState) -> Dict[str, Any]:
            if state.get("error"):
                return {"should_retry": False}
            parsed = dict(state.get("parsed") or {})
            packet = state.get("packet")
            attempt = int(state.get("attempt", 0))
            max_attempts = int(state.get("max_attempts", 1))
            should_retry = _should_retry_response(
                packet,
                parsed,
                attempt,
                max_attempts,
                tools_available=bool(state.get("tools_available")),
            )
            update: Dict[str, Any] = {"should_retry": should_retry}
            if should_retry:
                update["messages"] = [HumanMessage(content=_retry_reason_message(packet, parsed))]
                update["attempt"] = attempt + 1
            return update

        builder = StateGraph(_AnalystWorkflowState)
        builder.add_node("build_prompt", build_prompt_node)
        builder.add_node("call_model", call_model_node)
        builder.add_node("execute_tools", execute_tools_node)
        builder.add_node("assess", assess_node)
        builder.add_node(
            "finalize",
            lambda state: {
                "parsed": _serialize_parsed_state(dict(state.get("parsed") or {})),
                "error": state.get("error"),
                "timing_ms": dict(state.get("timing_ms") or {}),
                "attempt": int(state.get("attempt", 0)),
                "tool_rounds": int(state.get("tool_rounds", 0)),
            },
        )
        builder.add_edge(START, "build_prompt")
        builder.add_edge("build_prompt", "call_model")
        builder.add_conditional_edges(
            "call_model",
            _route_after_model,
            {
                "execute_tools": "execute_tools",
                "assess": "assess",
                "finalize": "finalize",
            },
        )
        builder.add_edge("execute_tools", "call_model")
        builder.add_conditional_edges(
            "assess",
            _route_after_assess,
            {
                "call_model": "call_model",
                "finalize": "finalize",
            },
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=checkpointer)

    async def abuild(self) -> "AnalystAgent":
        if self._build_lock is None:
            self._build_lock = asyncio.Lock()
        if self._tool_runtime_lock is None:
            self._tool_runtime_lock = asyncio.Lock()
        async with self._build_lock:
            if self._graph is None:
                self._graph = self._build_workflow()
            if self._bound_model_override is None and not self._tool_map and self._tool_runtime is None:
                try:
                    await self._ensure_tool_runtime()
                except Exception as exc:
                    self._tools_available = False
                    self._tool_setup_error = _error_text(exc)
        return self

    async def aclose(self) -> None:
        if self._tool_runtime_lock is None:
            self._tool_runtime_lock = asyncio.Lock()
        async with self._tool_runtime_lock:
            runtime = self._tool_runtime
            self._tool_runtime = None
            self._tools_available = False
        if runtime is not None:
            await runtime.aclose()

    async def arun(self, packet: AnalystPacket, *, debug: bool = False) -> AnalystRunResult:
        await self.abuild()

        tool_runtime: Optional[_FinancialToolRuntime] = self._tool_runtime
        tools_available = bool(self._tool_map) or bool(self._tools_available)
        tool_setup_error: Optional[str] = self._tool_setup_error
        if self._bound_model_override is None and not self._tool_map and tool_runtime is None:
            try:
                tool_runtime = await self._ensure_tool_runtime()
                tools_available = True
                tool_setup_error = None
            except Exception as exc:
                tool_setup_error = _error_text(exc)

        if _requires_calculation(packet) and not tools_available:
            open_issues = list(packet.open_issues) + [
                OpenIssue(
                    code="TOOL_UNAVAILABLE_FOR_COMPUTE",
                    message="financial_evaluator was unavailable for a calculation-required task; analyst execution failed closed.",
                    severity=Severity.ERROR,
                    metadata={"tool_setup_error": tool_setup_error},
                )
            ]
            return AnalystRunResult(
                ok=False,
                status="tool_error",
                answer="Computation could not be completed because financial_evaluator is unavailable.",
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                open_issues=open_issues,
                error=tool_setup_error or "financial_evaluator unavailable",
            )

        t0 = time.perf_counter()
        try:
            final_state = await self._graph.ainvoke(
                {
                    "packet": packet,
                    "messages": [],
                    "tools_available": tools_available,
                    "tool_setup_error": tool_setup_error,
                    "max_attempts": max(1, int(self.max_attempts)),
                    "max_tool_rounds": max(1, int(self.max_tool_rounds)),
                }
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - t0) * 1000)
            err_text = _error_text(exc)
            result_open_issues = list(packet.open_issues) + [
                OpenIssue(
                    code="ANALYST_RUNTIME_ERROR",
                    message=err_text,
                    severity=Severity.ERROR,
                )
            ]
            return AnalystRunResult(
                ok=False,
                status="tool_error" if _requires_calculation(packet) else "error",
                answer="Analyst agent failed to produce an answer.",
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                open_issues=result_open_issues,
                trace=AnalystTrace(
                    timing_ms={"total_ms": elapsed},
                    used_financial_evaluator=False,
                    tool_calls=[],
                    raw_message_count=0,
                    final_output_valid=False,
                    tool_error_code=None,
                ),
                error=err_text,
            )

        timing_ms = final_state.get("timing_ms") or {}
        parsed = final_state.get("parsed") or {}
        messages = final_state.get("messages") or []
        error = final_state.get("error")
        elapsed = int((time.perf_counter() - t0) * 1000)

        result_open_issues = list(packet.open_issues)
        if parsed.get("tool_round_limit_exceeded"):
            result_open_issues.append(
                OpenIssue(
                    code="ANALYST_TOOL_LOOP_LIMIT",
                    message="Analyst exceeded the financial_evaluator tool-call round limit.",
                    severity=Severity.ERROR,
                )
            )

        if error:
            error_text = str(error or "").strip() or repr(error)
            code = "ANALYST_MODEL_TIMEOUT" if "TIMEOUT" in error_text else "ANALYST_RUNTIME_ERROR"
            result_open_issues.append(
                OpenIssue(
                    code=code,
                    message=error_text,
                    severity=Severity.ERROR,
                )
            )
            return AnalystRunResult(
                ok=False,
                status="tool_error" if _requires_calculation(packet) else "error",
                answer="Analyst agent failed to produce an answer.",
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                open_issues=result_open_issues,
                trace=AnalystTrace(
                    timing_ms={**timing_ms, "total_ms": elapsed},
                    used_financial_evaluator=bool(parsed.get("used_financial_evaluator")),
                    tool_calls=list(parsed.get("tool_calls") or []),
                    raw_message_count=len(messages),
                    final_output_valid=False,
                    tool_error_code=_parsed_tool_error_code(parsed),
                ),
                error=error_text,
            )

        final_output = _validated_final_output_from_parsed(parsed)
        if final_output is None:
            message = parsed.get("final_output_error") or "Analyst did not return a valid FinalAnswer payload."
            result_open_issues.append(
                OpenIssue(
                    code="ANALYST_OUTPUT_INVALID",
                    message=str(message),
                    severity=Severity.ERROR,
                )
            )
            return AnalystRunResult(
                ok=False,
                status="error",
                answer=parsed.get("final_answer") or "Analyst output was invalid.",
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                open_issues=result_open_issues,
                trace=AnalystTrace(
                    timing_ms={**timing_ms, "total_ms": elapsed},
                    used_financial_evaluator=bool(parsed.get("used_financial_evaluator")),
                    tool_calls=list(parsed.get("tool_calls") or []),
                    raw_message_count=len(messages),
                    final_output_valid=False,
                    tool_error_code=_parsed_tool_error_code(parsed),
                ),
                error="ANALYST_OUTPUT_INVALID",
            )

        if parsed.get("tool_error"):
            result_open_issues.append(
                OpenIssue(
                    code="FINANCIAL_EVALUATOR_ERROR",
                    message=str(parsed.get("tool_error")),
                    severity=Severity.ERROR,
                    metadata=(
                        {"tool_error_code": str(parsed.get("tool_error_code"))}
                        if parsed.get("tool_error_code") is not None
                        else None
                    ),
                )
            )

        requires_calculation = _requires_calculation(packet)
        insufficient_data_terminal = (
            final_output.status == "insufficient_data"
            and not parsed.get("tool_error")
        )
        tool_computation = _computation_from_parsed_state(parsed)
        successful_computations = _successful_computations_from_parsed_state(parsed)
        final_calculation = final_output.calculation
        computation = tool_computation if tool_computation is not None else final_calculation
        calculation_mismatch = False
        calculation_ambiguous = False

        if requires_calculation and not insufficient_data_terminal:
            if parsed.get("tool_error"):
                calculation_mismatch = tool_computation is not None and not _computations_match(
                    final_calculation,
                    tool_computation,
                )
            elif final_calculation is not None:
                matched_computation, calculation_ambiguous = (
                    _resolve_matching_successful_computation(
                        final_calculation,
                        successful_computations,
                    )
                )
                if matched_computation is None and not calculation_ambiguous:
                    calculation_mismatch = True
                elif matched_computation is not None:
                    computation = matched_computation
                else:
                    computation = None
            elif len(successful_computations) == 1:
                computation = successful_computations[0]
            elif len(successful_computations) > 1:
                calculation_ambiguous = True
                computation = None
            else:
                computation = None

        if calculation_ambiguous:
            result_open_issues.append(
                OpenIssue(
                    code="CALCULATION_RESULT_AMBIGUOUS",
                    message=(
                        "FinalAnswer calculation could not be uniquely matched to one "
                        "successful financial_evaluator computation."
                    ),
                    severity=Severity.ERROR,
                )
            )
            return AnalystRunResult(
                ok=False,
                status="tool_error",
                answer=final_output.answer,
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                used_context_ids=list(final_output.used_context_ids),
                missing_values=list(final_output.missing_values),
                confidence=final_output.confidence,
                computation=None,
                compare_rows=list(final_output.compare_rows),
                open_issues=result_open_issues,
                trace=AnalystTrace(
                    timing_ms={**timing_ms, "total_ms": elapsed},
                    used_financial_evaluator=bool(parsed.get("used_financial_evaluator")),
                    tool_calls=list(parsed.get("tool_calls") or []),
                    raw_message_count=len(messages),
                    final_output_valid=True,
                    tool_error_code=_parsed_tool_error_code(parsed),
                ),
                error="CALCULATION_RESULT_AMBIGUOUS",
            )

        if calculation_mismatch:
            result_open_issues.append(
                OpenIssue(
                    code="CALCULATION_RESULT_MISMATCH",
                    message="FinalAnswer calculation did not match the financial_evaluator result.",
                    severity=Severity.ERROR,
                )
            )
            return AnalystRunResult(
                ok=False,
                status="tool_error",
                answer=final_output.answer,
                intent=packet.intent,
                metric=packet.analysis_task.metric,
                used_context_ids=list(final_output.used_context_ids),
                missing_values=list(final_output.missing_values),
                confidence=final_output.confidence,
                computation=tool_computation,
                compare_rows=list(final_output.compare_rows),
                open_issues=result_open_issues,
                trace=AnalystTrace(
                    timing_ms={**timing_ms, "total_ms": elapsed},
                    used_financial_evaluator=bool(parsed.get("used_financial_evaluator")),
                    tool_calls=list(parsed.get("tool_calls") or []),
                    raw_message_count=len(messages),
                    final_output_valid=True,
                    tool_error_code=_parsed_tool_error_code(parsed),
                ),
                error="CALCULATION_RESULT_MISMATCH",
            )

        if requires_calculation and not insufficient_data_terminal:
            if computation is None or computation.result is None:
                result_open_issues.append(
                    OpenIssue(
                        code="COMPUTE_RESULT_MISSING",
                        message="Calculation-required task finished without a reliable numeric result.",
                        severity=Severity.ERROR,
                    )
                )
                return AnalystRunResult(
                    ok=False,
                    status=final_output.status if final_output.status in {"tool_error", "insufficient_data"} else "error",
                    answer=final_output.answer,
                    intent=packet.intent,
                    metric=packet.analysis_task.metric,
                    used_context_ids=list(final_output.used_context_ids),
                    missing_values=list(final_output.missing_values),
                    confidence=final_output.confidence,
                    computation=computation,
                    compare_rows=list(final_output.compare_rows),
                    open_issues=result_open_issues,
                    trace=AnalystTrace(
                        timing_ms={**timing_ms, "total_ms": elapsed},
                        used_financial_evaluator=bool(parsed.get("used_financial_evaluator")),
                        tool_calls=list(parsed.get("tool_calls") or []),
                        raw_message_count=len(messages),
                        final_output_valid=True,
                        tool_error_code=_parsed_tool_error_code(parsed),
                    ),
                    error="COMPUTE_RESULT_MISSING",
                )

        context_map = {item.context_id: item for item in packet.context_items}
        cited_context_ids: List[str] = []
        seen_context_ids: set[str] = set()
        for context_id in list(final_output.used_context_ids):
            normalized = str(context_id or "").strip()
            if normalized and normalized not in seen_context_ids:
                seen_context_ids.add(normalized)
                cited_context_ids.append(normalized)
        for row in list(final_output.compare_rows):
            for context_id in list(row.context_ids or []):
                normalized = str(context_id or "").strip()
                if normalized and normalized not in seen_context_ids:
                    seen_context_ids.add(normalized)
                    cited_context_ids.append(normalized)

        citations: List[AnalystCitation] = []
        for context_id in cited_context_ids:
            item = context_map.get(context_id)
            if item is None:
                result_open_issues.append(
                    OpenIssue(
                        code="UNKNOWN_CONTEXT_ID",
                        message=f"Analyst referenced unknown context_id={context_id}.",
                        severity=Severity.WARNING,
                    )
                )
                continue
            citations.append(AnalystCitation(context_id=context_id, source=item.source))

        if debug:
            print(
                f"[analyst_timing_ms] build_prompt_ms={timing_ms.get('build_prompt_ms', 0)} "
                f"agent_invoke_ms={timing_ms.get('agent_invoke_ms', 0)} "
                f"agent_retry_ms={timing_ms.get('agent_retry_ms', 0)} "
                f"tool_exec_ms={timing_ms.get('tool_exec_ms', 0)} total_ms={elapsed}"
            )

        ok = final_output.status == "ok" or insufficient_data_terminal
        if requires_calculation and final_output.status == "ok":
            ok = ok and computation is not None and computation.result is not None

        return AnalystRunResult(
            ok=ok,
            status=final_output.status,
            answer=final_output.answer,
            intent=packet.intent,
            metric=packet.analysis_task.metric,
            used_context_ids=cited_context_ids,
            missing_values=list(final_output.missing_values),
            confidence=final_output.confidence,
            computation=computation,
            compare_rows=list(final_output.compare_rows),
            citations=citations,
            open_issues=result_open_issues,
            trace=AnalystTrace(
                timing_ms={**timing_ms, "total_ms": elapsed},
                used_financial_evaluator=bool(parsed.get("used_financial_evaluator")),
                tool_calls=list(parsed.get("tool_calls") or []),
                raw_message_count=len(messages),
                final_output_valid=True,
                tool_error_code=_parsed_tool_error_code(parsed),
            ),
            error=None if ok else final_output.status,
        )
