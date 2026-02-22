from __future__ import annotations

from typing import Any, Dict


async def retrieval_agent(state: Dict[str, Any], client) -> Dict[str, Any]:
    """
    Call retrieval MCP tool once using orchestrator-produced queries + metadata.
    """
    resp = await client.retrieve_tables(
        queries=state["queries"],
        ticker=state["ticker"],
        fiscal_year=state["fiscal_year"],
        form_type=state.get("form_type", "10-K"),
        doc_types=state.get("doc_types"),
        top_k=3,
        min_total_score=0,
    )
    return {**state, "retrieval": resp}

