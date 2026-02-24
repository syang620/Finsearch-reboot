from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

from pydantic import BaseModel, Field, field_validator

# Import your existing functions/modules
from rag10kq.pipeline import FinanceRAGPipeline  # expects pipeline.run_hybrid_search_pipeline(...)
from rag10kq.retrieval_evaluator import score_and_select_tables

TABLES_DIR = "../data/chunked"
# os.getenv("TABLES_DIR", "PATH/TO/YOUR/TABLES_DIR")
DEFAULT_DOC_TYPES = ["table"]  # replace with your real defaults
COLLECTION_NAME = "sec_docs_hybrid"

client = QdrantClient(host="localhost", port=6333)
config = PipelineConfig(
retrieval=RetrievalConfig(collection_name=COLLECTION_NAME),
)
pipeline = FinanceRAGPipeline(client, config)

class RetrieveTablesResponse(BaseModel):
    ok: bool = True
    queries_used: List[str]
    rerank_query: str
    top_tables: List[Dict[str, Any]] = Field(default_factory=list)
    max_total_score: Optional[int] = None
    metadata_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    trace: Dict[str, Any] = None

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
        doc_types = doc_types or DEFAULT_DOC_TYPES
        queries = RetrievalQueries(queries=queries).queries

        t0 = time.time()
        rerank_query, _fused, reranked = pipeline.run_hybrid_search_pipeline(
            queries=queries,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            doc_types=doc_types,
        )
        t1 = time.time()
        scored = score_and_select_tables(
            reranked,
            queries,
            str(fiscal_year),
            tables_dir=TABLES_DIR,
        )

        # apply min score + top_k
        scored = [t for t in scored if (t.get("total_score") or 0) >= min_total_score]
        top_tables = scored[:top_k]
        max_score = (top_tables[0].get("total_score") if top_tables else None)
        t2 = time.time()
        
        return RetrieveTablesResponse(
            ok=True,
            queries_used=queries,
            rerank_query=rerank_query,
            top_tables=top_tables,
            max_total_score=max_score,
            metadata_used={"ticker": ticker, "fiscal_year": fiscal_year, "form_type": form_type},
            trace = {
                "timing_ms": {
                    "hybrid_plus_rerank": int((t1 - t0) * 1000),
                    "lexical_scoring": int((t2 - t1) * 1000),
                    "total": int((t2 - t0) * 1000),
                },
                "counts": {
                    "fused_candidates": len(_fused) if _fused is not None else None,
                    "reranked": len(reranked) if reranked is not None else None,
                    "scored": len(scored),
                },
            }
        )
    except Exception as e:
        return RetrieveTablesResponse(
            ok=False,
            queries_used=queries[:4] if isinstance(queries, list) else [],
            rerank_query="",
            top_tables=[],
            max_total_score=None,
            metadata_used={"ticker": ticker, "fiscal_year": fiscal_year, "form_type": form_type},
            error=str(e),
        )