from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_score(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    match_text = _normalize_text(payload.get("match_text"))
    if match_text:
        payload.setdefault("content", match_text)
        payload.setdefault("text", match_text)
    return payload


def _normalize_result_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    if isinstance(result.get("table"), dict):
        normalized = dict(result)
        table_payload = result["table"].get("payload")
        if isinstance(table_payload, dict):
            normalized["table"] = {
                **result["table"],
                "payload": _normalize_result_payload(table_payload),
            }
        total_score = _normalize_score(normalized.get("total_score"))
        if total_score is None:
            total_score = _normalize_score(normalized.get("score"))
        if total_score is not None:
            normalized["total_score"] = total_score
        return normalized

    payload = _normalize_result_payload(result)
    total_score = _normalize_score(payload.get("total_score"))
    if total_score is None:
        total_score = _normalize_score(payload.get("score"))

    row_label = _normalize_text(payload.get("row_label"))
    row_headers = [row_label] if row_label else []
    table_name = (
        _normalize_text(payload.get("table_name"))
        or _normalize_text(payload.get("section_title"))
        or _normalize_text(payload.get("item_title"))
        or _normalize_text(payload.get("doc_id"))
    )

    normalized = {
        "table": {"payload": payload},
        "table_name": table_name,
        "row_headers": row_headers,
        "total_score": total_score,
        "doc_id": payload.get("doc_id"),
        "table_id": payload.get("table_doc_id") or payload.get("doc_id"),
        "section_path": payload.get("section_path"),
        "doc_type": payload.get("doc_type"),
        "ticker": payload.get("ticker"),
        "fiscal_year": payload.get("fiscal_year"),
        "form_type": payload.get("form_type"),
    }
    return {k: v for k, v in normalized.items() if v is not None}


def _normalize_retrieval_payload(payload: Any, *, args: Dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    raw_results = normalized.get("top_tables")
    if raw_results is None:
        raw_results = normalized.get("results")

    if isinstance(raw_results, list):
        normalized_results = [
            _normalize_result_entry(item)
            for item in raw_results
            if isinstance(item, dict)
        ]
        normalized["top_tables"] = normalized_results
        normalized.setdefault("results", normalized_results)

        if normalized.get("max_total_score") is None:
            scores = [
                score
                for score in (
                    _normalize_score(item.get("total_score"))
                    for item in normalized_results
                )
                if score is not None
            ]
            if scores:
                normalized["max_total_score"] = max(scores)

    metadata_used = dict(normalized.get("metadata_used") or {})
    requested_doc_types = args.get("doc_types")
    if isinstance(requested_doc_types, list) and requested_doc_types:
        metadata_used.setdefault("doc_types", list(requested_doc_types))
    normalized["metadata_used"] = metadata_used
    return normalized


@dataclass
class SecRetrievalMCPClient:
    server_command: str = sys.executable
    server_args: Optional[List[str]] = None

    _session: Optional[ClientSession] = None
    _read = None
    _write = None
    _stdio_cm = None
    _session_cm = None

    async def __aenter__(self):
        if self.server_args is None:
            # Prefer new MCP backend path; fallback to compatibility shim.
            candidates = [
                Path("src/mcp_server/server.py"),
                Path("../src/mcp_server/server.py"),
                Path("src/tools/server.py"),
                Path("../src/tools/server.py"),
            ]
            server_path = next((p for p in candidates if p.exists()), candidates[0])
            self.server_args = [str(server_path)]

        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
        )

        self._stdio_cm = stdio_client(server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(self._read, self._write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)

    async def retrieve_tables(
        self,
        *,
        queries: List[str],
        ticker: str,
        fiscal_year: int,
        form_type: str = "10-K",
        doc_types: Optional[List[str]] = None,
        top_k: int = 3,
        min_total_score: int = 0,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        assert self._session is not None, "Client not initialized. Use 'async with'."
        if (
            str(__import__("os").environ.get("RETRIEVAL_DEBUG_TOOL_CALLS", ""))
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        ):
            print(
                json.dumps(
                    {
                        "event": "mcp_client_retrieve_tables_args",
                        "args": {
                            "queries": list(queries or []),
                            "ticker": ticker,
                            "fiscal_year": fiscal_year,
                            "form_type": form_type,
                            "doc_types": list(doc_types or []),
                            "top_k": top_k,
                            "min_total_score": min_total_score,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        args = {
            "queries": queries,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "form_type": form_type,
            "doc_types": doc_types,
            "top_k": top_k,
            "min_total_score": min_total_score,
        }

        async def _call():
            result = await self._session.call_tool("sec_retrieve_tables", arguments=args)

            # MCP SDK compatibility: some versions expose camelCase fields.
            structured = getattr(result, "structured_content", None)
            if structured is None:
                structured = getattr(result, "structuredContent", None)
            if structured is not None:
                return _normalize_retrieval_payload(structured, args=args)

            is_error = bool(
                getattr(result, "is_error", False) or getattr(result, "isError", False)
            )

            out_text = []
            for block in getattr(result, "content", []) or []:
                if isinstance(block, types.TextContent):
                    out_text.append(block.text)
                    # Some servers return JSON as text; parse first valid dict/list.
                    try:
                        parsed = json.loads(block.text)
                        if isinstance(parsed, (dict, list)):
                            return _normalize_retrieval_payload(parsed, args=args)
                    except Exception:
                        pass

            return {
                "ok": not is_error,
                "unstructured": out_text,
                "args": args,
            }

        try:
            return await asyncio.wait_for(_call(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": (
                    f"MCP tool call timed out after {timeout_s:.0f}s. "
                    "First run may need to load embedding/reranker models."
                ),
                "args": args,
            }
