from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys
import time

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient

# Allow running this file directly without installing the package.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag10kq.pipeline import (  # noqa: E402
    FinanceRAGPipeline,
    LexicalScoringConfig,
    PipelineConfig,
    RetrievalConfig,
)
from rag10kq.retrieval_evaluator import score_and_select_tables  # noqa: E402

TABLES_DIR = os.getenv("TABLES_DIR", str((SRC_ROOT.parent / "data" / "chunked").resolve()))
DEFAULT_DOC_TYPES = ["table"]
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "sec_docs_hybrid")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


@lru_cache(maxsize=1)
def _get_pipeline() -> FinanceRAGPipeline:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    config = PipelineConfig(
        retrieval=RetrievalConfig(collection_name=COLLECTION_NAME),
        lexical_scoring=LexicalScoringConfig(tables_dir=TABLES_DIR),
    )
    return FinanceRAGPipeline(client, config)


class RetrieveTablesResponse(BaseModel):
    ok: bool = True
    queries_used: List[str]
    rerank_query: str
    top_tables: List[Dict[str, Any]] = Field(default_factory=list)
    max_total_score: Optional[float] = None
    metadata_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


class RetrievalQueries(BaseModel):
    queries: List[str] = Field(..., description="1-4 short retrieval queries")

    @field_validator("queries")
    @classmethod
    def _validate_queries(cls, v: List[str]) -> List[str]:
        v = [str(x).strip() for x in v if str(x).strip()]
        if not v:
            raise ValueError("queries must be non-empty")
        return v[:4]


def sec_retrieve_tables(
    *,
    queries: List[str],
    ticker: str,
    fiscal_year: int,
    form_type: str = "10-K",
    doc_types: Optional[List[str]] = None,
    top_k: int = 3,
    min_total_score: int = 0,
) -> RetrieveTablesResponse:
    """
    Deterministic SEC table retrieval:
    hybrid retrieval + rerank + lexical scoring (score_and_select_tables).
    """
    try:
        pipeline = _get_pipeline()
        doc_types = doc_types or DEFAULT_DOC_TYPES
        validated_queries = RetrievalQueries(queries=queries).queries

        t0 = time.time()
        rerank_query, fused, reranked = pipeline.run_hybrid_search_pipeline(
            queries=validated_queries,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            doc_types=doc_types,
        )
        t1 = time.time()
        scored = score_and_select_tables(
            reranked,
            validated_queries,
            str(fiscal_year),
            tables_dir=TABLES_DIR,
        )

        scored = [t for t in scored if (t.get("total_score") or 0) >= min_total_score]
        top_tables = scored[:top_k]
        max_score = top_tables[0].get("total_score") if top_tables else None
        t2 = time.time()

        return RetrieveTablesResponse(
            ok=True,
            queries_used=validated_queries,
            rerank_query=rerank_query,
            top_tables=top_tables,
            max_total_score=max_score,
            metadata_used={"ticker": ticker, "fiscal_year": fiscal_year, "form_type": form_type},
            trace={
                "timing_ms": {
                    "hybrid_plus_rerank": int((t1 - t0) * 1000),
                    "lexical_scoring": int((t2 - t1) * 1000),
                    "total": int((t2 - t0) * 1000),
                },
                "counts": {
                    "fused_candidates": len(fused) if fused is not None else None,
                    "reranked": len(reranked) if reranked is not None else None,
                    "scored": len(scored),
                },
            },
        )
    except Exception as e:
        safe_queries = [str(x).strip() for x in queries][:4] if isinstance(queries, list) else []
        return RetrieveTablesResponse(
            ok=False,
            queries_used=safe_queries,
            rerank_query="",
            top_tables=[],
            max_total_score=None,
            metadata_used={"ticker": ticker, "fiscal_year": fiscal_year, "form_type": form_type},
            error=str(e),
        )


def register_tools(mcp: FastMCP) -> None:
    mcp.tool()(sec_retrieve_tables)


def build_mcp_server() -> FastMCP:
    mcp = FastMCP("sec-retrieval")
    register_tools(mcp)
    return mcp


def main() -> None:
    build_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
