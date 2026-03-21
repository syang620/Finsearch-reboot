from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys
import time

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient, models

# Allow running this file directly without installing the package.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieval.evaluator import (  # noqa: E402
    dedupe_scored_points,
    embed_query_qwen3,
    hybrid_search_sec_docs_rrf,
    normalize_doc_id_to_table,
    rerank_with_qwen3_reranker_api,
    rrf_fuse,
)
from retrieval.rerank_enricher import enrich_candidates_with_table_summaries  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DOC_TYPES = ["text_chunk", "table", "table_row"]
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "sec_docs_dense_bm25")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QWEN3_EMBED_API_URL = os.getenv("QWEN3_EMBED_API_URL", "http://localhost:11434/api/embed")
QWEN3_EMBED_MODEL = os.getenv("QWEN3_EMBED_MODEL", "qwen3-embedding:8b")
QWEN3_DASHSCOPE_RERANK_API_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
RERANK_MODEL = os.getenv("SEC_RERANK_MODEL") or os.getenv(
    "QWEN3_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"
)
QWEN3_RERANK_API_KEY = (
    os.getenv("QWEN3_RERANK_API_KEY", "").strip()
    or os.getenv("DASHSCOPE_API_KEY", "").strip()
)
QWEN3_RERANK_API_URL = os.getenv("QWEN3_RERANK_API_URL", "").strip()
if not QWEN3_RERANK_API_URL and QWEN3_RERANK_API_KEY:
    QWEN3_RERANK_API_URL = QWEN3_DASHSCOPE_RERANK_API_URL
RETRIEVAL_TOP_K = int(os.getenv("SEC_RETRIEVAL_TOP_K", "50"))
RERANK_CANDIDATE_LIMIT = int(os.getenv("SEC_RERANK_CANDIDATE_LIMIT", "10"))
RERANK_TOP_K = int(os.getenv("SEC_RERANK_TOP_K", "10"))


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _point_doc_id(point: models.ScoredPoint) -> str:
    payload = point.payload or {}
    return str(payload.get("doc_id") or payload.get("id") or "")


def _scored_point_with_score(point: models.ScoredPoint, new_score: float) -> models.ScoredPoint:
    data = point.model_dump() if hasattr(point, "model_dump") else point.dict()
    data["score"] = float(new_score)
    return models.ScoredPoint(**data)


def _build_rerank_query(queries: List[str]) -> str:
    if not queries:
        return ""
    if len(queries) == 1:
        return str(queries[0])
    return f"{queries[0]} ({', '.join(queries[1:])})"


def _serialize_point(point: models.ScoredPoint) -> Dict[str, Any]:
    payload = dict(point.payload or {})
    match_text = payload.get("rerank_original_content") or payload.get("content")
    context_summary = payload.get("rerank_table_summary")
    return {
        "score": float(point.score),
        "doc_id": payload.get("doc_id"),
        "doc_type": payload.get("doc_type"),
        "ticker": payload.get("ticker"),
        "fiscal_year": payload.get("fiscal_year"),
        "form_type": payload.get("form_type"),
        "table_index": payload.get("table_index"),
        "row_index": payload.get("row_index"),
        "row_label": payload.get("row_label"),
        "section_title": payload.get("section_title"),
        "section_path": payload.get("section_path"),
        "item_id": payload.get("item_id"),
        "item_title": payload.get("item_title"),
        "source_html": payload.get("source_html"),
        "table_doc_id": payload.get("rerank_table_doc_id"),
        "match_text": match_text,
        "context_summary": context_summary,
    }


def _current_rerank_model() -> str:
    return (
        os.getenv("SEC_RERANK_MODEL", "").strip()
        or os.getenv("QWEN3_RERANK_MODEL", "").strip()
        or RERANK_MODEL
    )

def _current_qwen3_rerank_api_key() -> str:
    return (
        os.getenv("QWEN3_RERANK_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or QWEN3_RERANK_API_KEY
    )


def _current_qwen3_rerank_api_url() -> str:
    env_url = os.getenv("QWEN3_RERANK_API_URL", "").strip()
    if env_url:
        return env_url
    if _current_qwen3_rerank_api_key():
        return QWEN3_DASHSCOPE_RERANK_API_URL
    return QWEN3_RERANK_API_URL


def _rerank_candidates(
    query: str,
    candidates: List[models.ScoredPoint],
) -> tuple[List[models.ScoredPoint], Dict[str, Any]]:
    if not candidates:
        return [], {
            "selected_backend": None,
            "applied_backend": None,
            "fallback_used": False,
            "fallback_reason": None,
        }

    model_name = _current_rerank_model().strip()
    qwen3_rerank_api_key = _current_qwen3_rerank_api_key()
    qwen3_rerank_api_url = _current_qwen3_rerank_api_url()
    rerank_top_k = max(top_k for top_k in (RERANK_TOP_K, len(candidates)) if top_k is not None)
    lower_model_name = model_name.lower()
    wants_qwen3_rerank = ("qwen3-reranker" in lower_model_name) or ("qwen3-rerank" in lower_model_name)
    selected_backend = "qwen3_api"

    if not wants_qwen3_rerank:
        raise RuntimeError(
            f"Unsupported reranker configuration: {model_name}. Only Qwen3 API reranking is allowed."
        )
    if not qwen3_rerank_api_key:
        raise RuntimeError(
            "Qwen3 API reranking requires QWEN3_RERANK_API_KEY or DASHSCOPE_API_KEY."
        )
    if not qwen3_rerank_api_url.strip():
        raise RuntimeError(
            "Qwen3 API reranking requires QWEN3_RERANK_API_URL."
        )

    try:
        return rerank_with_qwen3_reranker_api(
            query,
            candidates,
            api_key=qwen3_rerank_api_key,
            api_url=qwen3_rerank_api_url,
            model_name=model_name,
            top_k=rerank_top_k,
        ), {
            "selected_backend": selected_backend,
            "applied_backend": selected_backend,
            "fallback_used": False,
            "fallback_reason": None,
        }
    except Exception as exc:
        logger.exception("Qwen3 API reranker failed.")
        raise RuntimeError(f"Qwen3 API reranker failed: {type(exc).__name__}: {exc}") from exc


def _run_dense_bm25_retrieval(
    *,
    client: QdrantClient,
    queries: List[str],
    ticker: str,
    fiscal_year: int,
    form_type: Optional[str],
    doc_types: List[str],
) -> tuple[str, List[models.ScoredPoint], List[models.ScoredPoint], Dict[str, Any]]:
    hits_by_query: Dict[str, List[models.ScoredPoint]] = {}
    per_query_timings: List[Dict[str, Any]] = []
    total_embed_ms = 0.0
    total_retrieval_ms = 0.0
    total_query_embed_cache_hits = 0
    total_query_embed_cache_misses = 0
    for query in queries:
        query_embed_ms = 0.0
        query_embed_cache_hits = 0
        query_embed_cache_misses = 0

        def _timed_embed(value: str) -> List[float]:
            nonlocal query_embed_ms
            nonlocal query_embed_cache_hits
            nonlocal query_embed_cache_misses
            cache_stats: Dict[str, Any] = {}
            embed_t0 = time.perf_counter()
            out = embed_query_qwen3(
                value,
                api_url=QWEN3_EMBED_API_URL,
                model=QWEN3_EMBED_MODEL,
                cache_stats=cache_stats,
            )
            query_embed_ms += (time.perf_counter() - embed_t0) * 1000.0
            if cache_stats.get("cache_hit") is True:
                query_embed_cache_hits += 1
            elif cache_stats.get("cache_hit") is False:
                query_embed_cache_misses += 1
            return out

        query_t0 = time.perf_counter()
        hits_by_query[query] = hybrid_search_sec_docs_rrf(
            query,
            client=client,
            embed_fn=_timed_embed,
            collection_name=COLLECTION_NAME,
            using_dense="dense",
            using_bm25="bm25",
            top_k=RETRIEVAL_TOP_K,
            doc_types=doc_types,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
        )
        query_total_ms = (time.perf_counter() - query_t0) * 1000.0
        total_embed_ms += query_embed_ms
        total_retrieval_ms += query_total_ms
        total_query_embed_cache_hits += query_embed_cache_hits
        total_query_embed_cache_misses += query_embed_cache_misses
        per_query_timings.append(
            {
                "query": query,
                "retrieval_total_ms": int(query_total_ms),
                "query_embedding_ms": int(query_embed_ms),
                "query_embedding_cache_hit": bool(query_embed_cache_hits) and not bool(query_embed_cache_misses),
                "query_embedding_cache_hits": query_embed_cache_hits,
                "query_embedding_cache_misses": query_embed_cache_misses,
                "search_and_fusion_ms": int(max(query_total_ms - query_embed_ms, 0.0)),
                "hits": len(hits_by_query[query]),
            }
        )

    fusion_t0 = time.perf_counter()
    ranked_lists = [hits_by_query.get(query, []) for query in queries]
    fused_scores, ranks = rrf_fuse(ranked_lists)

    by_id: Dict[str, models.ScoredPoint] = {}
    for ranked in ranked_lists:
        for point in ranked:
            by_id.setdefault(str(point.id), point)

    def sort_key(point_id: str) -> tuple[float, int, str]:
        best_rank = min(ranks.get(point_id, {}).values() or [10**9])
        return (-fused_scores[point_id], best_rank, point_id)

    ordered_ids = sorted(fused_scores.keys(), key=sort_key)[:RETRIEVAL_TOP_K]
    fused_candidates = [_scored_point_with_score(by_id[point_id], fused_scores[point_id]) for point_id in ordered_ids]
    fusion_ms = (time.perf_counter() - fusion_t0) * 1000.0

    dedupe_t0 = time.perf_counter()
    deduped = dedupe_scored_points(
        fused_candidates,
        key_fn=lambda point: normalize_doc_id_to_table(_point_doc_id(point)),
    )[:RERANK_CANDIDATE_LIMIT]
    dedupe_ms = (time.perf_counter() - dedupe_t0) * 1000.0

    enrich_t0 = time.perf_counter()
    enriched = enrich_candidates_with_table_summaries(
        deduped,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    enrichment_ms = (time.perf_counter() - enrich_t0) * 1000.0

    rerank_query = _build_rerank_query(queries)
    rerank_t0 = time.perf_counter()
    reranked, rerank_info = _rerank_candidates(rerank_query, enriched)
    rerank_ms = (time.perf_counter() - rerank_t0) * 1000.0

    stage_timings = {
        "query_embedding_ms": int(total_embed_ms),
        "query_embedding_cache_hits": int(total_query_embed_cache_hits),
        "query_embedding_cache_misses": int(total_query_embed_cache_misses),
        "hybrid_retrieval_ms": int(total_retrieval_ms),
        "search_and_fusion_inside_query_ms": int(max(total_retrieval_ms - total_embed_ms, 0.0)),
        "cross_query_fusion_ms": int(fusion_ms),
        "dedupe_ms": int(dedupe_ms),
        "enrichment_ms": int(enrichment_ms),
        "rerank_ms": int(rerank_ms),
        "per_query": per_query_timings,
        "rerank": {
            "model_name": _current_rerank_model().strip(),
            **rerank_info,
        },
    }

    return rerank_query, fused_candidates, reranked, stage_timings


class RetrieveTablesResponse(BaseModel):
    ok: bool = True
    queries_used: List[str]
    rerank_query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
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


def retrieve_scored_points(
    *,
    queries: List[str],
    ticker: str,
    fiscal_year: int,
    form_type: Optional[str] = None,
    doc_types: Optional[List[str]] = None,
) -> tuple[str, List[models.ScoredPoint], List[models.ScoredPoint], Dict[str, Any]]:
    client = _get_client()
    resolved_doc_types = doc_types or DEFAULT_DOC_TYPES
    resolved_form_type = (form_type or "").strip() or None
    validated_queries = RetrievalQueries(queries=queries).queries
    return _run_dense_bm25_retrieval(
        client=client,
        queries=validated_queries,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=resolved_form_type,
        doc_types=resolved_doc_types,
    )


def sec_retrieve_tables(
    *,
    queries: List[str],
    ticker: str,
    fiscal_year: int,
    form_type: Optional[str] = None,
    doc_types: Optional[List[str]] = None,
    top_k: int = 3,
    min_total_score: float = 0.0,
) -> RetrieveTablesResponse:
    """
    Deterministic SEC retrieval:
    dense+BM25 retrieval + rerank, returning one ranked list of hits.
    """
    try:
        resolved_doc_types = doc_types or DEFAULT_DOC_TYPES
        resolved_form_type = (form_type or "").strip() or None
        validated_queries = RetrievalQueries(queries=queries).queries

        t0 = time.time()
        rerank_query, fused, reranked, retrieval_timing = retrieve_scored_points(
            queries=validated_queries,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=resolved_form_type,
            doc_types=resolved_doc_types,
        )
        t1 = time.time()
        raw_results = [_serialize_point(point) for point in reranked[:top_k]]
        t2 = time.time()
        min_score = float(min_total_score or 0.0)
        if min_score > 0.0:
            results = [
                item
                for item in raw_results
                if item.get("score") is not None and float(item.get("score")) >= min_score
            ]
        else:
            results = raw_results

        return RetrieveTablesResponse(
            ok=True,
            queries_used=validated_queries,
            rerank_query=rerank_query,
            results=results,
            error=(
                (
                    f"No retrieval results met min_total_score={min_score}"
                    if min_score > 0.0 and not results
                    else None
                )
            ),
            metadata_used={"ticker": ticker, "fiscal_year": fiscal_year, "form_type": resolved_form_type},
            trace={
                "timing_ms": {
                    "hybrid_plus_rerank": int((t1 - t0) * 1000),
                    "postprocess": int((t2 - t1) * 1000),
                    "total": int((t2 - t0) * 1000),
                    "query_embedding": retrieval_timing["query_embedding_ms"],
                    "hybrid_retrieval": retrieval_timing["hybrid_retrieval_ms"],
                    "search_and_fusion_inside_query": retrieval_timing["search_and_fusion_inside_query_ms"],
                    "cross_query_fusion": retrieval_timing["cross_query_fusion_ms"],
                    "dedupe": retrieval_timing["dedupe_ms"],
                    "enrichment": retrieval_timing["enrichment_ms"],
                    "rerank": retrieval_timing["rerank_ms"],
                },
                "counts": {
                    "fused_candidates": len(fused) if fused is not None else None,
                    "reranked": len(reranked) if reranked is not None else None,
                    "results_before_min_score_filter": len(raw_results),
                    "results_after_min_score_filter": len(results),
                    "results": len(results),
                    "query_embedding_cache_hits": retrieval_timing["query_embedding_cache_hits"],
                    "query_embedding_cache_misses": retrieval_timing["query_embedding_cache_misses"],
                },
                "per_query_timing_ms": retrieval_timing["per_query"],
                "rerank": dict(retrieval_timing.get("rerank") or {}),
            },
        )
    except Exception as e:
        safe_queries = [str(x).strip() for x in queries][:4] if isinstance(queries, list) else []
        return RetrieveTablesResponse(
            ok=False,
            queries_used=safe_queries,
            rerank_query="",
            results=[],
            metadata_used={"ticker": ticker, "fiscal_year": fiscal_year, "form_type": resolved_form_type},
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
