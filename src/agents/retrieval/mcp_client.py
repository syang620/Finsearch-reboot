from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


@dataclass
class SecRetrievalMCPClient:
    server_command: str = "python"
    server_args: Optional[List[str]] = None

    _session: Optional[ClientSession] = None
    _read = None
    _write = None
    _stdio_cm = None
    _session_cm = None

    async def __aenter__(self):
        if self.server_args is None:
            # If cwd is repo root, use src/tools/server.py.
            # If cwd is notebooks/, use ../src/tools/server.py.
            p1 = Path("src/tools/server.py")
            p2 = Path("../src/tools/server.py")
            server_path = p1 if p1.exists() else p2
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
                return structured

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
                            return parsed
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

