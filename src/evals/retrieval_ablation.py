from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient, models

from mcp_server.tools.sec_retrieval import (
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QWEN3_EMBED_API_URL,
    QWEN3_EMBED_MODEL,
    RERANK_CANDIDATE_LIMIT,
    RETRIEVAL_TOP_K,
    retrieve_scored_points,
)
from retrieval.evaluator import (
    bm25_search_points,
    build_sec_filter,
    dedupe_scored_points,
    dense_search_points,
    embed_query_qwen3,
    normalize_doc_id_to_table,
    rrf_fuse,
)

RETRIEVAL_MODES: Tuple[str, ...] = (
    "bm25_only",
    "dense_only",
    "hybrid",
    "hybrid_reranker",
)

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "bm25_only": {
        "dense_enabled": False,
        "bm25_enabled": True,
        "reranker_enabled": False,
    },
    "dense_only": {
        "dense_enabled": True,
        "bm25_enabled": False,
        "reranker_enabled": False,
    },
    "hybrid": {
        "dense_enabled": True,
        "bm25_enabled": True,
        "reranker_enabled": False,
    },
    "hybrid_reranker": {
        "dense_enabled": True,
        "bm25_enabled": True,
        "reranker_enabled": True,
    },
}


def _point_key(point: models.ScoredPoint) -> str:
    return str(point.id)


def _payload_doc_id(point: models.ScoredPoint) -> str:
    payload = point.payload or {}
    return str(payload.get("doc_id") or payload.get("id") or "")


def _with_score(point: models.ScoredPoint, score: float) -> models.ScoredPoint:
    data = point.model_dump() if hasattr(point, "model_dump") else point.dict()
    data["score"] = float(score)
    return models.ScoredPoint(**data)


def _fuse_ranked_lists(
    ranked_lists: Sequence[Sequence[models.ScoredPoint]],
    *,
    top_k: int,
) -> List[models.ScoredPoint]:
    fused_scores, ranks = rrf_fuse(ranked_lists, k=60, weights=[1.0] * len(ranked_lists))
    by_id: Dict[str, models.ScoredPoint] = {}
    for ranked in ranked_lists:
        for point in ranked:
            by_id.setdefault(_point_key(point), point)

    def sort_key(point_id: str) -> tuple[float, int, str]:
        best_rank = min(ranks.get(point_id, {}).values() or [10**9])
        return (-fused_scores[point_id], best_rank, point_id)

    ordered_ids = sorted(fused_scores, key=sort_key)[:top_k]
    return [_with_score(by_id[point_id], fused_scores[point_id]) for point_id in ordered_ids]


def _client(host: str, port: int) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def retrieve_ablation_points(
    *,
    retrieval_mode: str,
    query: str,
    ticker: str,
    fiscal_year: int,
    form_type: Optional[str],
    doc_types: List[str],
    client: Optional[QdrantClient] = None,
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    embed_api_url: str = QWEN3_EMBED_API_URL,
    embed_model: str = QWEN3_EMBED_MODEL,
) -> tuple[str, List[models.ScoredPoint], List[models.ScoredPoint], Dict[str, Any]]:
    """Run one frozen retrieval ablation without changing production defaults."""
    mode = str(retrieval_mode or "").strip().lower()
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval_mode={retrieval_mode!r}")

    resolved_host = qdrant_host or os.getenv("QDRANT_HOST", QDRANT_HOST)
    resolved_port = int(qdrant_port or os.getenv("QDRANT_PORT", str(QDRANT_PORT)))
    resolved_collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", COLLECTION_NAME)
    qdrant = client or _client(resolved_host, resolved_port)
    provenance = {
        "qdrant": {
            "host": resolved_host,
            "port": resolved_port,
            "collection": resolved_collection,
        },
        "embedding": (
            {
                "api_url": embed_api_url,
                "model": embed_model,
            }
            if ABLATION_CONFIGS[mode]["dense_enabled"]
            else None
        ),
    }
    if mode == "hybrid_reranker":
        rerank_query, fused, reranked, timing = retrieve_scored_points(
            queries=[query],
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            doc_types=doc_types,
            client=qdrant,
            collection_name=resolved_collection,
            embed_api_url=embed_api_url,
            embed_model=embed_model,
        )
        candidate_ms = sum(
            int(timing.get(key) or 0)
            for key in (
                "hybrid_retrieval_ms",
                "cross_query_fusion_ms",
                "dedupe_ms",
                "enrichment_ms",
            )
        )
        rerank_ms = int(timing.get("rerank_ms") or 0)
        return rerank_query, fused, reranked, {
            **timing,
            "candidate_retrieval_ms": candidate_ms,
            "total_retrieval_ms": candidate_ms + rerank_ms,
            "components": dict(ABLATION_CONFIGS[mode]),
            "provenance": provenance,
        }

    qfilter = build_sec_filter(
        doc_types=doc_types,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
    )
    branch_limit = max(50, RETRIEVAL_TOP_K * 10)
    timings: Dict[str, Any] = {
        "query_embedding_ms": 0,
        "dense_search_ms": 0,
        "bm25_search_ms": 0,
        "fusion_ms": 0,
        "dedupe_ms": 0,
        "enrichment_ms": 0,
        "rerank_ms": 0,
    }
    cache_stats: Dict[str, Any] = {}
    dense_hits: List[models.ScoredPoint] = []
    bm25_hits: List[models.ScoredPoint] = []
    total_t0 = time.perf_counter()

    if ABLATION_CONFIGS[mode]["dense_enabled"]:
        embed_t0 = time.perf_counter()
        query_vector = embed_query_qwen3(
            query,
            api_url=embed_api_url,
            model=embed_model,
            cache_stats=cache_stats,
        )
        timings["query_embedding_ms"] = int((time.perf_counter() - embed_t0) * 1000)
        dense_t0 = time.perf_counter()
        dense_hits = dense_search_points(
            query_vector,
            client=qdrant,
            collection_name=resolved_collection,
            using_dense="dense",
            qfilter=qfilter,
            limit=branch_limit,
        )
        timings["dense_search_ms"] = int((time.perf_counter() - dense_t0) * 1000)

    if ABLATION_CONFIGS[mode]["bm25_enabled"]:
        bm25_t0 = time.perf_counter()
        bm25_hits = bm25_search_points(
            query,
            client=qdrant,
            collection_name=resolved_collection,
            using_bm25="bm25",
            qfilter=qfilter,
            limit=branch_limit,
        )
        timings["bm25_search_ms"] = int((time.perf_counter() - bm25_t0) * 1000)

    fusion_t0 = time.perf_counter()
    if mode == "hybrid":
        candidates = _fuse_ranked_lists([dense_hits, bm25_hits], top_k=RETRIEVAL_TOP_K)
    elif mode == "dense_only":
        candidates = dense_hits[:RETRIEVAL_TOP_K]
    else:
        candidates = bm25_hits[:RETRIEVAL_TOP_K]
    timings["fusion_ms"] = int((time.perf_counter() - fusion_t0) * 1000)

    dedupe_t0 = time.perf_counter()
    ranked = dedupe_scored_points(
        candidates,
        key_fn=lambda point: normalize_doc_id_to_table(_payload_doc_id(point)),
    )[:RERANK_CANDIDATE_LIMIT]
    timings["dedupe_ms"] = int((time.perf_counter() - dedupe_t0) * 1000)
    total_ms = int((time.perf_counter() - total_t0) * 1000)
    timings.update(
        {
            "candidate_retrieval_ms": total_ms,
            "total_retrieval_ms": total_ms,
            "query_embedding_cache_hits": int(cache_stats.get("cache_hit") is True),
            "query_embedding_cache_misses": int(cache_stats.get("cache_hit") is False),
            "components": dict(ABLATION_CONFIGS[mode]),
            "provenance": provenance,
            "counts": {
                "dense_branch": len(dense_hits),
                "bm25_branch": len(bm25_hits),
                "fused_candidates": len(candidates),
                "ranked": len(ranked),
            },
        }
    )
    return query, candidates, ranked, timings
