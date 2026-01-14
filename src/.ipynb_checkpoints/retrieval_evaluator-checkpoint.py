from __future__ import annotations

"""
Utilities for evaluating dense retrieval over SEC table questions.

Typical usage from a notebook (after setting PYTHONPATH=src):

    from qdrant_client import QdrantClient
    from retrieval_evaluator import (
        embed_query_qwen3,
        dense_search_sec_docs,
        load_table_eval,
        evaluate_table_queries,
        pretty_print_table_example,
    )

    client = QdrantClient(host="localhost", port=6333)

    summary, results = evaluate_table_queries(
        client=client,
        eval_path="notebooks/table_eval.jsonl",
        ticker="AAPL",
        fiscal_year=2024,
        form_type="10-K",
        top_k=5,
        embed_fn=embed_query_qwen3,
    )
    print(summary)
    pretty_print_table_example(
        load_table_eval("notebooks/table_eval.jsonl")[0],
        results[0],
    )
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from sentence_transformers import CrossEncoder

import requests
from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------------
# Basic hit wrapper (mirrors the notebook's SecDocHit)
# ---------------------------------------------------------------------------


@dataclass
class SecDocHit:
    score: float
    doc_id: str
    content: str
    doc_type: str
    metadata: Dict[str, Any]

    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[str] = None
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    row_label: Optional[str] = None
    section_path: Optional[str] = None


def to_sec_doc_hit(scored_point: models.ScoredPoint) -> SecDocHit:
    payload = scored_point.payload or {}
    doc_id = payload.get("doc_id") or payload.get("id") or ""
    content = payload.get("content") or ""
    doc_type = payload.get("doc_type") or payload.get("metadata", {}).get(
        "doc_type",
        "unknown",
    )

    md = payload

    return SecDocHit(
        score=float(scored_point.score),
        doc_id=doc_id,
        content=content,
        doc_type=doc_type,
        metadata=md,
        ticker=md.get("ticker"),
        fiscal_year=md.get("fiscal_year"),
        form_type=md.get("form_type"),
        table_index=md.get("table_index"),
        row_index=md.get("row_index"),
        row_label=md.get("row_label"),
        section_path=md.get("section_path")
        if isinstance(md.get("section_path"), str)
        else None,
    )


def dedupe_hits(hits: Sequence[SecDocHit]) -> List[SecDocHit]:
    seen: set[str] = set()
    out: List[SecDocHit] = []
    for h in hits:
        if h.doc_id and h.doc_id not in seen:
            seen.add(h.doc_id)
            out.append(h)
    return out

def dedupe_scored_points(
    points: List[models.ScoredPoint],
    key_fn: Callable[[models.ScoredPoint], str],
) -> List[models.ScoredPoint]:
    best: Dict[str, models.ScoredPoint] = {}
    for p in points:
        k = key_fn(p)
        if k not in best or float(p.score) > float(best[k].score):
            best[k] = p
    # keep stable order by score desc
    return sorted(best.values(), key=lambda x: float(x.score), reverse=True)

def normalize_doc_id_to_table(doc_id: str) -> str:
    """
    Collapse table rows to their parent table id.
    Leaves table-level and text ids as-is.
    """
    if not doc_id:
        return ""
    # If it's a table row, strip the row suffix
    if "::table::" in doc_id and "::row::" in doc_id:
        return doc_id.split("::row::", 1)[0]
    return doc_id

def table_group_key(p: models.ScoredPoint) -> str:
    md = p.payload or {}
    doc_id = md.get("doc_id") or md.get("id") or ""
    return normalize_doc_id_to_table(doc_id)

def cap_per_group(
    points: List[models.ScoredPoint],
    key_fn: Callable[[models.ScoredPoint], str],
    *,
    cap: int = 2,
    max_total: int = 120,
) -> List[models.ScoredPoint]:
    buckets = defaultdict(list)
    for p in sorted(points, key=lambda x: float(x.score), reverse=True):
        k = key_fn(p)
        if len(buckets[k]) < cap:
            buckets[k].append(p)
    # flatten in score order and apply max_total
    out = []
    for p in sorted([p for lst in buckets.values() for p in lst], key=lambda x: float(x.score), reverse=True):
        out.append(p)
        if len(out) >= max_total:
            break
    return out
    
def doc_id_table_key(p: models.ScoredPoint) -> str:
    md = p.payload or {}
    doc_id = md.get("doc_id") or md.get("id") or ""
    return normalize_doc_id_to_table(doc_id)
    
# ---------------------------------------------------------------------------
# Embedding + filter + dense search (mirrors notebook helpers)
# ---------------------------------------------------------------------------


def embed_query_qwen3(
    query: str,
    api_url: str = "http://localhost:11434/api/embed",
    model: str = "qwen3-embedding:8b",
    timeout: int = 60,
) -> List[float]:
    """
    Embed a query string using qwen3-embedding:8b via an Ollama-style endpoint.
    """
    payload = {
        "model": model,
        "input": [query],
    }
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


# ---------------------------------------------------------------------------
# BGE-M3 embedding (dense + sparse) for hybrid retrieval
# ---------------------------------------------------------------------------


_BGE_M3_MODEL = None


def get_bge_m3_model(model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
    """Lazy-load and cache BGE-M3 model."""
    global _BGE_M3_MODEL
    if _BGE_M3_MODEL is None:
        from FlagEmbedding import BGEM3FlagModel

        _BGE_M3_MODEL = BGEM3FlagModel(model_name, use_fp16=use_fp16)
    return _BGE_M3_MODEL


def lexical_weights_to_sparse_vector(
    lexical_weights: dict,
    *,
    top_k: int = 256,
) -> models.SparseVector:
    """
    Convert BGE-M3 lexical_weights (token_id -> weight) into Qdrant SparseVector.

    top_k: keep only the top_k highest-weight tokens to control index size/latency.
    """
    items = list(lexical_weights.items())

    if top_k is not None and len(items) > top_k:
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:top_k]

    # Deterministic ordering
    items.sort(key=lambda kv: int(kv[0]))

    indices = [int(i) for i, _ in items]
    values = [float(w) for _, w in items]
    return models.SparseVector(indices=indices, values=values)


def embed_query_bge_m3(
    query: str,
    *,
    bge_model=None,
    sparse_top_k: int = 256,
) -> Tuple[list[float], models.SparseVector]:
    """
    Returns:
      (bge_m3_dense_vec, bge_m3_sparse_vec)
    """
    if bge_model is None:
        bge_model = get_bge_m3_model()

    out = bge_model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
    )
    dense = out["dense_vecs"][0]
    dense = dense.tolist() if hasattr(dense, "tolist") else list(dense)

    lexical = out["lexical_weights"][0]  # dict[token_id -> weight]
    sparse = lexical_weights_to_sparse_vector(lexical, top_k=sparse_top_k)

    return dense, sparse


def hybrid_search_sec_docs_bge_m3(
    query: str,
    *,
    client: QdrantClient,
    collection_name: str = "sec_docs",
    top_k: int = 10,
    using_bge_dense: str = "bge_m3_dense",
    using_bge_sparse: str = "bge_m3_sparse",
    prefetch_k: Optional[int] = None,
    bge_sparse_top_k: int = 256,
    doc_types: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    form_type: Optional[str] = None,
    bge_model=None,
) -> List[models.ScoredPoint]:
    """
    Hybrid retrieval using:
      - BGE-M3 sparse (lexical weights) on `using_bge_sparse`
      - BGE-M3 dense on `using_bge_dense`
    fused with Reciprocal Rank Fusion (RRF).
    """
    if prefetch_k is None:
        prefetch_k = max(50, top_k * 10)

    qfilter = build_sec_filter(
        doc_types=doc_types,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
    )

    bge_dense_vec, bge_sparse_vec = embed_query_bge_m3(
        query,
        bge_model=bge_model,
        sparse_top_k=bge_sparse_top_k,
    )

    resp = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=bge_sparse_vec,
                using=using_bge_sparse,
                limit=prefetch_k,
                filter=qfilter,
            ),
            models.Prefetch(
                query=bge_dense_vec,
                using=using_bge_dense,
                limit=prefetch_k,
                filter=qfilter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=qfilter,
    )

    return list(resp.points)


def bm25_search_points(
    query_text: str,
    *,
    client: QdrantClient,
    collection_name: str,
    using_bm25: str = "bm25_sparse_vector",
    qfilter: Optional[models.Filter] = None,
    limit: int = 100,
    avg_len: Optional[float] = None,
) -> List[models.ScoredPoint]:
    # Qdrant BM25 query object (requires Qdrant inference support / configured BM25 field)
    bm25_query = models.Document(
        text=query_text,
        model="Qdrant/bm25",
        options={"avg_len": avg_len} if avg_len is not None else None,
    )

    resp = client.query_points(
        collection_name=collection_name,
        query=bm25_query,
        using=using_bm25,
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(resp.points)


def dense_search_points(
    query_vec: List[float],
    *,
    client: QdrantClient,
    collection_name: str,
    using_dense: str = "dense",
    qfilter: Optional[models.Filter] = None,
    limit: int = 100,
) -> List[models.ScoredPoint]:
    resp = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        using=using_dense,
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(resp.points)


def format_passage_for_rerank(p: models.ScoredPoint, *, max_chars: int = 2000) -> str:
    md = p.payload or {}
    parts = [
        f"doc_type: {md.get('doc_type')}",
        f"ticker: {md.get('ticker')} | fiscal_year: {md.get('fiscal_year')} | form_type: {md.get('form_type')}",
        f"table_index: {md.get('table_index')} | row_index: {md.get('row_index')} | row_label: {md.get('row_label')}",
        f"section_title: {md.get('section_title')}",
        f"section_path: {md.get('section_path')}",
        "content:",
        (md.get("content") or ""),
    ]
    text = "\n".join([x for x in parts if x and x != "None"])
    return text[:max_chars]

def rerank_with_minilm_l6(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    batch_size: int = 32,
) -> List[models.ScoredPoint]:
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    pairs = [(query, format_passage_for_rerank(p, max_chars=max_passage_chars)) for p in candidates]
    scores = ce.predict(pairs, batch_size=batch_size)

    rescored = []
    for p, s in zip(candidates, scores):
        data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
        data["score"] = float(s)
        rescored.append(models.ScoredPoint(**data))

    rescored.sort(key=lambda x: float(x.score), reverse=True)
    return rescored[:top_k]
    
def _point_key(pid: Any) -> str:
    # ScoredPoint.id can be int/str/uuid-ish; normalize to string key.
    return str(pid)


def rrf_fuse(
    ranked_lists: Sequence[Sequence[models.ScoredPoint]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[int, int]]]:
    """
    Compute Reciprocal Rank Fusion scores.

    Returns:
      fused_scores:  point_id_str -> fused_score
      ranks:         point_id_str -> {list_index: rank_1_based}
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match number of ranked_lists")

    fused: Dict[str, float] = {}
    ranks: Dict[str, Dict[int, int]] = {}

    for li, (lst, w) in enumerate(zip(ranked_lists, weights)):
        for rank0, p in enumerate(lst):
            pid = _point_key(p.id)
            rank = rank0 + 1  # 1-based
            fused[pid] = fused.get(pid, 0.0) + (float(w) / (k + rank))
            ranks.setdefault(pid, {})[li] = rank

    return fused, ranks


def _scored_point_with_score(p: models.ScoredPoint, new_score: float) -> models.ScoredPoint:
    # Works across pydantic versions.
    data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
    data["score"] = float(new_score)
    return models.ScoredPoint(**data)


def hybrid_search_sec_docs_rrf(
    query: str,
    *,
    client: QdrantClient,
    embed_fn: Callable[[str], List[float]],
    collection_name: str = "sec_docs",
    using_dense: str = "dense",
    using_bm25: str = "bm25_sparse_vector",
    top_k: int = 10,
    dense_limit: Optional[int] = None,
    bm25_limit: Optional[int] = None,
    rrf_k: int = 60,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    avg_len: Optional[float] = None,
    doc_types: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    form_type: Optional[str] = None,
) -> List[models.ScoredPoint]:
    """
    Runs dense and BM25 separately, then fuses rankings with client-side RRF.
    Returns top_k ScoredPoints with score replaced by the RRF score.
    """
    if dense_limit is None:
        dense_limit = max(50, top_k * 10)
    if bm25_limit is None:
        bm25_limit = max(50, top_k * 10)

    qfilter = build_sec_filter(
        doc_types=doc_types,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
    )

    # 1) dense retrieval
    qvec = embed_fn(query)
    dense_hits = dense_search_points(
        qvec,
        client=client,
        collection_name=collection_name,
        using_dense=using_dense,
        qfilter=qfilter,
        limit=dense_limit,
    )

    # 2) bm25 retrieval
    bm25_hits = bm25_search_points(
        query,
        client=client,
        collection_name=collection_name,
        using_bm25=using_bm25,
        qfilter=qfilter,
        limit=bm25_limit,
        avg_len=avg_len,
    )

    # 3) RRF fuse (rank-only)
    fused_scores, ranks = rrf_fuse(
        [dense_hits, bm25_hits],
        k=rrf_k,
        weights=[w_dense, w_bm25],
    )

    # 4) pick a representative ScoredPoint object per id
    by_id: Dict[str, models.ScoredPoint] = {}
    for p in dense_hits:
        by_id.setdefault(_point_key(p.id), p)
    for p in bm25_hits:
        by_id.setdefault(_point_key(p.id), p)

    # 5) sort by fused score desc; tie-break by best (lowest) rank across lists
    def sort_key(pid: str):
        best_rank = min(ranks.get(pid, {}).values() or [10**9])
        return (-fused_scores[pid], best_rank, pid)

    ordered_ids = sorted(fused_scores.keys(), key=sort_key)[:top_k]

    # 6) return ScoredPoints with RRF score
    out: List[models.ScoredPoint] = []
    for pid in ordered_ids:
        out.append(_scored_point_with_score(by_id[pid], fused_scores[pid]))
    return out


def build_sec_filter(
    doc_types: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    form_type: Optional[str] = None,
) -> Optional[models.Filter]:
    """
    Build a Qdrant filter for the sec_docs collection.
    """
    must: List[models.Condition] = []

    if doc_types:
        must.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchAny(any=doc_types),
            )
        )

    if ticker is not None:
        must.append(
            models.FieldCondition(
                key="ticker",
                match=models.MatchValue(value=ticker),
            )
        )

    if fiscal_year is not None:
        must.append(
            models.FieldCondition(
                key="fiscal_year",
                match=models.MatchValue(value=fiscal_year),
            )
        )

    if form_type is not None:
        must.append(
            models.FieldCondition(
                key="form_type",
                match=models.MatchValue(value=form_type),
            )
        )

    if not must:
        return None

    return models.Filter(must=must)


def dense_search_sec_docs(
    query: str,
    *,
    client: QdrantClient,
    embed_fn: Callable[[str], List[float]],
    collection_name: str = "sec_docs",
    top_k: int = 10,
    doc_types: Optional[List[str]] = None,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    form_type: Optional[str] = None,
) -> List[models.ScoredPoint]:
    """
    Dense search over the sec_docs collection.

    Returns the raw Qdrant ScoredPoint objects, whose `.payload` contains
    the stored document fields.
    """
    query_vec = embed_fn(query)
    qfilter = build_sec_filter(
        doc_types=doc_types,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
    )

    hits = client.query_points(
        collection_name=collection_name,
        query=query_vec,
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return list(hits.points)


# ---------------------------------------------------------------------------
# Table eval helpers
# ---------------------------------------------------------------------------


def load_table_eval(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file of table retrieval eval examples.

    Expected schema (per line), simplified:
        {
          "query_id": 1,
          "query": "...",
          "gold_answer": "...",
          "gold_value": {...},
          "relevant_tables": [
            {
              "table_index": 10,
              "section_title": "...",
              "section_path": "...",
              "item_id": "...",
              "item_title": "...",
              "target_rows": [...]
            },
            ...
          ]
        }
    """
    p = Path(path)
    examples: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def is_relevant_hit(hit: SecDocHit, example: Dict[str, Any]) -> bool:
    """
    Treat a hit as correct if it points to any table listed in the
    example's `relevant_tables` (matching table_index).
    """
    md = hit.metadata or {}
    table_idx = md.get("table_index")

    relevant_tables = example.get("relevant_tables") or []
    relevant_indices = {t.get("table_index") for t in relevant_tables}

    # Fallback: older schema that had a single top-level table_index
    if not relevant_indices and example.get("table_index") is not None:
        relevant_indices.add(example.get("table_index"))

    return table_idx in relevant_indices


def eval_one_table_query(
    example: Dict[str, Any],
    *,
    client: QdrantClient,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    search_fn: Optional[Callable[[str], List[models.ScoredPoint]]] = None,
    ticker: str,
    fiscal_year: int,
    form_type: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run retrieval for a single eval example and compute metrics.

    Provide either:
      - `search_fn(query) -> list[ScoredPoint]` for custom pipelines (hybrid, rerank, etc.), or
      - `embed_fn(query) -> list[float]` to use `dense_search_sec_docs`.
    """
    if search_fn is not None:
        raw_hits = search_fn(example["query"])
    else:
        if embed_fn is None:
            embed_fn = embed_query_qwen3

        raw_hits = dense_search_sec_docs(
            example["query"],
            client=client,
            embed_fn=embed_fn,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            top_k=top_k,
            doc_types=["table", "table_row"],
        )

    hits = [to_sec_doc_hit(h) for h in raw_hits]
    hits = dedupe_hits(hits)[:top_k]

    rel_flags = [is_relevant_hit(h, example) for h in hits]
    num_rel = sum(rel_flags)
    hit_at_k = 1.0 if num_rel > 0 else 0.0
    precision_at_k = num_rel / len(hits) if hits else 0.0

    # Total number of relevant tables for this query (for recall@k)
    relevant_tables = example.get("relevant_tables") or []
    relevant_indices = {t.get("table_index") for t in relevant_tables}
    if not relevant_indices and example.get("table_index") is not None:
        relevant_indices.add(example.get("table_index"))
    total_relevant = len(relevant_indices) or 0
    recall_at_k = num_rel / total_relevant if total_relevant > 0 else 0.0

    first_rel_rank: Optional[int] = None
    for idx, is_rel in enumerate(rel_flags):
        if is_rel:
            first_rel_rank = idx
            break
    mrr_at_k = 1.0 / (first_rel_rank + 1) if first_rel_rank is not None else 0.0

    return {
        "query_id": example.get("query_id"),
        "query": example.get("query"),
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "num_rel_in_top_k": num_rel,
        "total_relevant": total_relevant,
        "top_k": top_k,
        "hits": hits,
        "rel_flags": rel_flags,
    }


def evaluate_table_queries(
    *,
    client: QdrantClient,
    eval_path: str,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    search_fn: Optional[Callable[[str], List[models.ScoredPoint]]] = None,
    ticker: str,
    fiscal_year: int,
    form_type: str,
    top_k: int = 5,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate retrieval over all queries in a table eval JSONL file.

    Returns (summary_metrics, per_query_results).
    """
    if search_fn is None and embed_fn is None:
        embed_fn = embed_query_qwen3

    examples = load_table_eval(eval_path)
    results: List[Dict[str, Any]] = []

    for ex in examples:
        res = eval_one_table_query(
            ex,
            client=client,
            embed_fn=embed_fn,
            search_fn=search_fn,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            top_k=top_k,
        )
        results.append(res)

    n = len(results) or 1
    hit_rate = sum(r["hit_at_k"] for r in results) / n
    mean_precision = sum(r["precision_at_k"] for r in results) / n
    mean_recall = sum(r["recall_at_k"] for r in results) / n
    mean_mrr = sum(r["mrr_at_k"] for r in results) / n

    summary = {
        "num_queries": len(results),
        f"hit_rate@{top_k}": hit_rate,
        f"precision@{top_k}": mean_precision,
        f"recall@{top_k}": mean_recall,
        f"mrr@{top_k}": mean_mrr,
    }
    return summary, results


def pretty_print_table_example(
    example: Dict[str, Any],
    result: Dict[str, Any],
    *,
    max_hits: int = 5,
) -> None:
    """
    Print a single eval example together with top retrieval hits.
    """
    k = result.get("top_k", len(result.get("hits", [])) or 0)

    print("=== Query ===")
    print(f"[{example.get('query_id')}] {example.get('query')}\n")

    print("=== Per-query metrics ===")
    print(
        f"Hit@{k}: {result.get('hit_at_k'):.3f}  "
        f"P@{k}: {result.get('precision_at_k'):.3f}  "
        f"R@{k}: {result.get('recall_at_k'):.3f}  "
        f"MRR@{k}: {result.get('mrr_at_k'):.3f}"
    )
    print()

    print("=== Gold target ===")
    relevant_tables = example.get("relevant_tables") or []
    print(f"num_relevant_tables: {len(relevant_tables)}")
    for t in relevant_tables:
        print(
            "  table_index   :",
            t.get("table_index"),
            "| section_title:",
            t.get("section_title"),
        )
        print("    section_path :", t.get("section_path"))
        print("    item_id/title:", t.get("item_id"), "/", t.get("item_title"))
        print("    target_rows  :", t.get("target_rows"))
    print()

    print("=== Top hits ===")
    hits: Sequence[SecDocHit] = result.get("hits", [])
    rel_flags: Sequence[bool] = result.get("rel_flags", [])

    for rank, (hit, is_rel) in enumerate(
        zip(hits[:max_hits], rel_flags[:max_hits]),
        start=1,
    ):
        md = hit.metadata or {}
        print(f"Rank {rank} | score={hit.score:.4f} | relevant={is_rel}")
        print(" doc_type     :", md.get("doc_type"))
        print(" table_index  :", md.get("table_index"))
        print(" row_index    :", md.get("row_index"), "row_label:", md.get("row_label"))
        print(" item_id/title:", md.get("item_id"), "/", md.get("item_title"))
        print(" section_title:", md.get("section_title"))
        print(" section_path :", md.get("section_path"))
        content_preview = (hit.content or "").replace("\n", " ")
        print(" content      :", content_preview[:220], "...\n")


def run_table_eval_and_print(
    *,
    client: QdrantClient,
    eval_path: str,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    search_fn: Optional[Callable[[str], List[models.ScoredPoint]]] = None,
    ticker: str,
    fiscal_year: int,
    form_type: str,
    top_k: int = 5,
    max_examples_to_show: int = 3,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Convenience wrapper: run evaluation over a JSONL file and print
    summary + per-query metrics.

    Returns (summary_metrics, per_query_results).
    """
    summary, results = evaluate_table_queries(
        client=client,
        eval_path=eval_path,
        embed_fn=embed_fn,
        search_fn=search_fn,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
        top_k=top_k,
    )

    print("=== Summary metrics ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print()

    examples = load_table_eval(eval_path)

    # Per-query metric table (no retrieval details here)
    print("=== Per-query metrics ===")
    k = top_k
    header = (
        f"{'qid':>4}  {'Hit':>4}  {'P':>5}  {'R':>5}  {'MRR':>5}  "
        f"{'#rel@k':>7}  {'#rel_total':>10}  {'query'}"
    )
    print(header)
    for ex, res in zip(examples, results):
        qid = ex.get("query_id")
        hit = float(res.get("hit_at_k", 0.0))
        prec = float(res.get("precision_at_k", 0.0))
        rec = float(res.get("recall_at_k", 0.0))
        mrr = float(res.get("mrr_at_k", 0.0))
        num_rel = int(res.get("num_rel_in_top_k", 0))
        total_rel = int(res.get("total_relevant", 0))
        qtext = (ex.get("query") or "").replace("\n", " ")
        if len(qtext) > 80:
            qtext = qtext[:77] + "..."
        print(
            f"{qid:>4}  {hit:4.2f}  {prec:5.2f}  {rec:5.2f}  "
            f"{mrr:5.2f}  {num_rel:7d}  {total_rel:10d}  {qtext}"
        )
    print()

    return summary, results


def print_query_retrieval_details(
    query_id: int,
    *,
    eval_path: str,
    results: Sequence[Dict[str, Any]],
    max_hits: int = 5,
) -> None:
    """
    Given a query_id and a previously computed `results` list from
    `evaluate_table_queries` / `run_table_eval_and_print`, print the
    detailed retrieval output for that query.
    """
    examples = load_table_eval(eval_path)

    ex_by_id: Dict[int, Dict[str, Any]] = {}
    for ex in examples:
        qid = ex.get("query_id")
        if isinstance(qid, int):
            ex_by_id[qid] = ex

    res_by_id: Dict[int, Dict[str, Any]] = {}
    for r in results:
        qid = r.get("query_id")
        if isinstance(qid, int):
            res_by_id[qid] = r

    ex = ex_by_id.get(query_id)
    res = res_by_id.get(query_id)
    if ex is None or res is None:
        print(f"[WARN] No example / result found for query_id={query_id}")
        return

    pretty_print_table_example(ex, res, max_hits=max_hits)


__all__ = [
    "SecDocHit",
    "to_sec_doc_hit",
    "dedupe_hits",
    "embed_query_qwen3",
    "get_bge_m3_model",
    "lexical_weights_to_sparse_vector",
    "embed_query_bge_m3",
    "hybrid_search_sec_docs_bge_m3",
    "bm25_search_points",
    "dense_search_points",
    "_point_key",
    "rrf_fuse",
    "_scored_point_with_score",
    "hybrid_search_sec_docs_rrf",
    "build_sec_filter",
    "dense_search_sec_docs",
    "load_table_eval",
    "is_relevant_hit",
    "eval_one_table_query",
    "evaluate_table_queries",
    "pretty_print_table_example",
    "run_table_eval_and_print",
    "print_query_retrieval_details",
]
