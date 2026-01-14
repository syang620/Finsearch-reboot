from __future__ import annotations

"""
Utilities for evaluating dense retrieval over SEC table questions.

Typical usage from a notebook (after setting PYTHONPATH=src):

    from qdrant_client import QdrantClient
    from rag10kq.retrieval_evaluator import (
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def multi_query_hybrid_search_bge_m3(
    queries: Sequence[str],
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
    max_workers: Optional[int] = None,
    fuse: bool = False,
    rrf_k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, List[models.ScoredPoint]] | Tuple[List[models.ScoredPoint], Dict[str, List[models.ScoredPoint]]]:
    """
    Run hybrid (BGE-M3 dense+sparse) search for multiple queries.

    Implementation notes:
      - Embedding runs sequentially in one batch encode() call (avoids threading issues with BGE-M3).
      - Only the Qdrant calls are parallelized.

    Returns:
      - If fuse=False: {query: [ScoredPoint, ...]} for each query.
      - If fuse=True: (fused_points, hits_by_query) where fused_points are top_k results
        after client-side RRF across the per-query ranked lists.
    """
    qs = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    if not qs:
        return {} if not fuse else ([], {})

    if prefetch_k is None:
        prefetch_k = max(50, top_k * 10)

    qfilter = build_sec_filter(
        doc_types=doc_types,
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
    )

    if bge_model is None:
        bge_model = get_bge_m3_model()

    # Embed once (batch) to keep it deterministic and avoid BGE thread-safety issues.
    out = bge_model.encode(qs, return_dense=True, return_sparse=True)
    dense_vecs = out["dense_vecs"]
    lexical_weights = out["lexical_weights"]

    dense_sparse_by_q: Dict[str, Tuple[List[float], models.SparseVector]] = {}
    for i, q in enumerate(qs):
        d = dense_vecs[i]
        d_list = d.tolist() if hasattr(d, "tolist") else list(d)
        sparse = lexical_weights_to_sparse_vector(
            lexical_weights[i],
            top_k=bge_sparse_top_k,
        )
        dense_sparse_by_q[q] = (d_list, sparse)

    def one_qdrant_search(q: str) -> List[models.ScoredPoint]:
        dense_vec, sparse_vec = dense_sparse_by_q[q]
        resp = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vec,
                    using=using_bge_sparse,
                    limit=prefetch_k,
                    filter=qfilter,
                ),
                models.Prefetch(
                    query=dense_vec,
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

    if max_workers is None:
        max_workers = min(8, len(qs))

    hits_by_query: Dict[str, List[models.ScoredPoint]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(one_qdrant_search, q): q for q in qs}
        for fut in as_completed(futs):
            q = futs[fut]
            hits_by_query[q] = fut.result()

    if not fuse:
        return hits_by_query

    ranked_lists = [hits_by_query.get(q, []) for q in qs]
    fused_scores, ranks = rrf_fuse(ranked_lists, k=rrf_k, weights=weights)

    by_id: Dict[str, models.ScoredPoint] = {}
    for lst in ranked_lists:
        for p in lst:
            by_id.setdefault(_point_key(p.id), p)

    def sort_key(pid: str):
        best_rank = min(ranks.get(pid, {}).values() or [10**9])
        return (-fused_scores[pid], best_rank, pid)

    ordered_ids = sorted(fused_scores.keys(), key=sort_key)[:top_k]
    fused_points = [_scored_point_with_score(by_id[pid], fused_scores[pid]) for pid in ordered_ids]

    return fused_points, hits_by_query


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


def score_and_select_tables(
    candidate_tables: List[Any],
    expanded_query_terms: List[str],
    target_year: str,
    tables_dir: str = "../data/chunked",
) -> List[Dict[str, Any]]:
    def _get(table: Any, key: str, default: Any = "") -> Any:
        if isinstance(table, dict):
            return table.get(key, default)
        return getattr(table, key, default)

    def _get_payload(table: Any) -> Optional[Dict[str, Any]]:
        payload = _get(table, "payload", None)
        return payload if isinstance(payload, dict) else None

    def _get_doc_id(table: Any) -> Optional[str]:
        doc_id = _get(table, "doc_id", None)
        if doc_id:
            return doc_id
        payload = _get_payload(table)
        if payload:
            return payload.get("doc_id")
        return None

    def _iter_cells(row: Any) -> Iterable[Any]:
        if isinstance(row, (list, tuple)):
            return row
        if isinstance(row, dict):
            return row.values()
        return [row]

    def _parse_doc_id(doc_id: str) -> tuple[Optional[str], Optional[int]]:
        if not doc_id:
            return None, None
        parts = doc_id.split("::")
        try:
            table_idx = parts.index("table")
            doc_prefix = "::".join(parts[:table_idx])
            table_id = int(parts[table_idx + 1])
            return doc_prefix, table_id
        except (ValueError, IndexError):
            return None, None

    def _load_table_from_doc(doc_prefix: str, table_id: int) -> Optional[Dict[str, Any]]:
        if not doc_prefix or table_id is None:
            return None
        path = Path(tables_dir) / f"{doc_prefix}.tables.jsonl"
        if not path.exists():
            return None
        for idx, line in enumerate(path.open()):
            if idx == table_id:
                return json.loads(line)
        return None

    target_year_str = str(target_year or "")
    terms = [t for t in (expanded_query_terms or []) if t]
    terms_lower = [t.lower() for t in terms]

    scored: List[Dict[str, Any]] = []

    for table in candidate_tables or []:
        payload = _get_payload(table) or {}

        doc_prefix = payload.get("prefix")
        table_id = payload.get("table_index")
        if table_id is not None:
            try:
                table_id = int(table_id)
            except (TypeError, ValueError):
                table_id = None

        if doc_prefix is None or table_id is None:
            rerank_doc_id = payload.get("rerank_table_doc_id")
            if rerank_doc_id:
                doc_prefix, table_id = _parse_doc_id(rerank_doc_id)
            else:
                doc_id = _get_doc_id(table)
                doc_prefix, table_id = _parse_doc_id(doc_id or "")

        table_obj = _load_table_from_doc(doc_prefix, table_id) if doc_prefix else None

        table_dict = (table_obj or {}).get("table_dict", {}) if table_obj else {}
        table_data = table_dict.get("data", []) or []
        column_headers = table_data[0] if table_data else []
        data_rows = table_data[1:] if len(table_data) > 1 else table_data

        table_name = str(_get(table, "table_name", "") or "")
        if not table_name and table_obj:
            table_name = str(
                table_obj.get("section_title", "")
                or table_obj.get("item_title", "")
                or ""
            )
        if not table_name:
            table_name = str(
                payload.get("section_title", "") or payload.get("item_title", "") or ""
            )

        fiscal_year = str(_get(table, "fiscal_year", "") or "")
        if not fiscal_year and table_obj:
            fiscal_year = str((table_obj.get("meta") or {}).get("fiscal_year", "") or "")
        if not fiscal_year:
            fiscal_year = str(payload.get("fiscal_year", "") or "")

        row_headers = []
        for row in data_rows:
            header_val = ""
            if isinstance(row, (list, tuple)):
                if row:
                    header_val = row[0]
            elif isinstance(row, dict):
                if column_headers:
                    header_val = row.get(column_headers[0], "")
                else:
                    for v in row.values():
                        header_val = v
                        break
            else:
                header_val = row
            row_headers.append(str(header_val))

        meta = " ".join([table_name, " ".join(map(str, column_headers)), fiscal_year])
        if target_year_str not in meta:
            scored.append(
                {
                    "table": table,
                    "table_name": table_name,
                    "row_headers": row_headers,
                    "rule_a_score": -100,
                    "rule_b_score": 0,
                    "rule_c_score": 0,
                    "rule_d_score": 0,
                    "total_score": -100,
                }
            )
            continue

        matched_terms = set()
        for term in terms_lower:
            for header in row_headers:
                if term in header.lower():
                    matched_terms.add(term)
                    break
        rule_b_score = 15 * len(matched_terms)

        table_name_lower = table_name.lower()
        rule_c_score = 20 if any(term in table_name_lower for term in terms_lower) else 0

        total_cells = 0
        digit_cells = 0
        for row in data_rows:
            for cell in _iter_cells(row):
                total_cells += 1
                if any(ch.isdigit() for ch in str(cell)):
                    digit_cells += 1
        density = (digit_cells / total_cells) if total_cells else 0
        rule_d_score = -20 if density < 0.15 else 0

        total_score = rule_b_score + rule_c_score + rule_d_score

        scored.append(
            {
                "table": table,
                "table_name": table_name,
                "row_headers": row_headers,
                "rule_a_score": 0,
                "rule_b_score": rule_b_score,
                "rule_c_score": rule_c_score,
                "rule_d_score": rule_d_score,
                "total_score": total_score,
            }
        )

    scored_sorted = sorted(scored, key=lambda x: x["total_score"], reverse=True)
    if not scored_sorted:
        return []

    max_score = scored_sorted[0]["total_score"]
    selected = [scored_sorted[0]]

    for entry in scored_sorted[1:]:
        if entry["total_score"] >= (max_score * 0.80):
            selected.append(entry)
        if len(selected) >= 3:
            break

    return selected


def rerank_with_minilm_l6(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    batch_size: int = 32,
) -> List[models.ScoredPoint]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "rerank_with_minilm_l6 requires `sentence-transformers` to be installed."
        ) from exc

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
    

_JINA_RERANKER_V3 = None


def get_jina_reranker_v3_model(model_name: str = "jinaai/jina-reranker-v3"):
    global _JINA_RERANKER_V3
    if _JINA_RERANKER_V3 is None:
        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "get_jina_reranker_v3_model requires `transformers` (and a backend like torch)."
            ) from exc

        _JINA_RERANKER_V3 = AutoModel.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
        )
        _JINA_RERANKER_V3.eval()
    return _JINA_RERANKER_V3


def rerank_with_jina_v3(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    model=None,
) -> List[models.ScoredPoint]:
    """
    Re-rank candidates with `jinaai/jina-reranker-v3`.

    Uses the model's `model.rerank(query, documents)` API and replaces each
    ScoredPoint.score with the reranker score.
    """
    if not candidates:
        return []

    if model is None:
        model = get_jina_reranker_v3_model()

    documents = [
        format_passage_for_rerank(p, max_chars=max_passage_chars) for p in candidates
    ]
    results = model.rerank(query, documents)

    # Expected (typical) formats include:
    # - list[dict(index=int, relevance_score=float, ...)]
    # - dict(results=[...])
    if isinstance(results, dict) and "results" in results:
        results_list = results["results"]
    else:
        results_list = results

    if not isinstance(results_list, list):
        raise TypeError(f"Unexpected jina rerank output: {type(results_list)}")

    rescored: List[models.ScoredPoint] = []
    for item in results_list:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            idx = item.get("doc_id")
        score = (
            item.get("relevance_score")
            if item.get("relevance_score") is not None
            else item.get("score")
        )
        if idx is None or score is None:
            continue
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= len(candidates):
            continue
        p = candidates[idx_int]
        data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
        data["score"] = float(score)
        rescored.append(models.ScoredPoint(**data))

    rescored.sort(key=lambda x: float(x.score), reverse=True)
    return rescored[:top_k]


_BGE_RERANKER_LARGE = None


def get_bge_reranker_large_model(
    model_name: str = "BAAI/bge-reranker-large",
    *,
    use_fp16: bool = True,
):
    global _BGE_RERANKER_LARGE
    if _BGE_RERANKER_LARGE is None:
        try:
            from FlagEmbedding import FlagReranker
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "get_bge_reranker_large_model requires `FlagEmbedding` to be installed."
            ) from exc

        _BGE_RERANKER_LARGE = FlagReranker(model_name, use_fp16=use_fp16)
    return _BGE_RERANKER_LARGE


def rerank_with_bge_reranker_large(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    model=None,
) -> List[models.ScoredPoint]:
    """
    Re-rank candidates with `BAAI/bge-reranker-large` via FlagEmbedding FlagReranker.

    Replaces each ScoredPoint.score with the reranker score.
    """
    if not candidates:
        return []

    if model is None:
        model = get_bge_reranker_large_model()

    pairs = [
        [query, format_passage_for_rerank(p, max_chars=max_passage_chars)]
        for p in candidates
    ]
    scores = model.compute_score(pairs)

    rescored: List[models.ScoredPoint] = []
    for p, s in zip(candidates, scores):
        data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
        data["score"] = float(s)
        rescored.append(models.ScoredPoint(**data))

    rescored.sort(key=lambda x: float(x.score), reverse=True)
    return rescored[:top_k]


_GTE_MULTILINGUAL_RERANKER = None
_GTE_MULTILINGUAL_TOKENIZER = None


def get_gte_multilingual_reranker_base(
    model_name_or_path: str = "Alibaba-NLP/gte-multilingual-reranker-base",
    *,
    device: str | None = None,
):
    """
    Lazy-load and cache Alibaba GTE multilingual reranker + tokenizer.

    Returns (model, tokenizer).
    """
    global _GTE_MULTILINGUAL_RERANKER, _GTE_MULTILINGUAL_TOKENIZER

    if _GTE_MULTILINGUAL_RERANKER is not None and _GTE_MULTILINGUAL_TOKENIZER is not None:
        return _GTE_MULTILINGUAL_RERANKER, _GTE_MULTILINGUAL_TOKENIZER

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "get_gte_multilingual_reranker_base requires `torch` and `transformers`."
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model.to(device)

    _GTE_MULTILINGUAL_RERANKER = model
    _GTE_MULTILINGUAL_TOKENIZER = tokenizer
    return model, tokenizer


def rerank_with_gte_multilingual_reranker_base(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    batch_size: int = 32,
    max_length: int = 512,
    model=None,
    tokenizer=None,
    device: str | None = None,
) -> List[models.ScoredPoint]:
    """
    Re-rank candidates with `Alibaba-NLP/gte-multilingual-reranker-base`.

    Replaces each ScoredPoint.score with the reranker score.
    """
    if not candidates:
        return []

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("rerank_with_gte_multilingual_reranker_base requires `torch`.") from exc

    if model is None or tokenizer is None:
        model, tokenizer = get_gte_multilingual_reranker_base(device=device)

    if device is None:
        device = str(next(model.parameters()).device)

    passages = [
        format_passage_for_rerank(p, max_chars=max_passage_chars) for p in candidates
    ]
    pairs = [[query, doc] for doc in passages]

    scores_all: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores_all.extend([float(x) for x in logits.detach().cpu().tolist()])

    rescored: List[models.ScoredPoint] = []
    for p, s in zip(candidates, scores_all):
        data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
        data["score"] = float(s)
        rescored.append(models.ScoredPoint(**data))

    rescored.sort(key=lambda x: float(x.score), reverse=True)
    return rescored[:top_k]


_GRANITE_RERANKER_R2 = None


def get_granite_reranker_english_r2_model(
    model_name: str = "ibm-granite/granite-embedding-reranker-english-r2",
):
    """Lazy-load and cache IBM Granite reranker (SentenceTransformers CrossEncoder)."""
    global _GRANITE_RERANKER_R2
    if _GRANITE_RERANKER_R2 is None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "get_granite_reranker_english_r2_model requires `sentence-transformers`."
            ) from exc

        _GRANITE_RERANKER_R2 = CrossEncoder(model_name)
    return _GRANITE_RERANKER_R2


def rerank_with_granite_english_r2(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    model=None,
) -> List[models.ScoredPoint]:
    """
    Re-rank candidates with `ibm-granite/granite-embedding-reranker-english-r2`.

    Uses CrossEncoder.rank(query, passages) and replaces each ScoredPoint.score
    with the Granite reranker score.
    """
    if not candidates:
        return []

    if model is None:
        model = get_granite_reranker_english_r2_model()

    passages = [
        format_passage_for_rerank(p, max_chars=max_passage_chars) for p in candidates
    ]
    ranks = model.rank(query, passages, return_documents=False)

    # Expected items: {"corpus_id": int, "score": float, ...}
    rescored: List[models.ScoredPoint] = []
    for item in ranks:
        corpus_id = item.get("corpus_id")
        score = item.get("score")
        if corpus_id is None or score is None:
            continue
        idx = int(corpus_id)
        if idx < 0 or idx >= len(candidates):
            continue
        p = candidates[idx]
        data = p.model_dump() if hasattr(p, "model_dump") else p.dict()
        data["score"] = float(score)
        rescored.append(models.ScoredPoint(**data))

    rescored.sort(key=lambda x: float(x.score), reverse=True)
    return rescored[:top_k]


_QWEN3_RERANKER_CACHE: Dict[str, Dict[str, Any]] = {}


def get_qwen3_reranker_model(
    model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    *,
    device: str | None = None,
    torch_dtype: Any | None = None,
    attn_implementation: str | None = None,
    max_length: int = 8192,
):
    """
    Lazy-load and cache Qwen3 reranker (0.6B/4B/8B) + tokenizer and prompt tokens.

    Notes:
      - Requires `transformers>=4.51.0` and `torch`.
      - This function intentionally does not manage model downloads; ensure the model
        is available in your local HF cache or your environment allows downloads.

    Returns a dict with keys:
      - model, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens, max_length
    """
    if model_name in _QWEN3_RERANKER_CACHE:
        return _QWEN3_RERANKER_CACHE[model_name]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise ImportError("get_qwen3_reranker_model requires `torch` and `transformers`.") from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
    model.to(device)

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    if token_false_id is None or token_true_id is None:
        raise ValueError("Tokenizer did not return ids for tokens 'yes'/'no'.")

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    obj = {
        "model": model,
        "tokenizer": tokenizer,
        "token_false_id": int(token_false_id),
        "token_true_id": int(token_true_id),
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "max_length": int(max_length),
    }
    _QWEN3_RERANKER_CACHE[model_name] = obj
    return obj


def rerank_with_qwen3_reranker(
    query: str,
    candidates: List[models.ScoredPoint],
    *,
    top_k: int = 20,
    max_passage_chars: int = 2000,
    batch_size: int = 8,
    model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    instruction: str | None = None,
    device: str | None = None,
    torch_dtype: Any | None = None,
    attn_implementation: str | None = None,
    max_length: int = 8192,
) -> List[models.ScoredPoint]:
    """
    Re-rank candidates with Qwen3 reranker models (0.6B/4B/8B).

    Uses a yes/no likelihood scoring scheme and replaces ScoredPoint.score with
    the probability of "yes".
    """
    if not candidates:
        return []

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("rerank_with_qwen3_reranker requires `torch`.") from exc

    cfg = get_qwen3_reranker_model(
        model_name,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        max_length=max_length,
    )
    model = cfg["model"]
    tokenizer = cfg["tokenizer"]
    token_false_id = cfg["token_false_id"]
    token_true_id = cfg["token_true_id"]
    prefix_tokens = cfg["prefix_tokens"]
    suffix_tokens = cfg["suffix_tokens"]
    max_length = cfg["max_length"]

    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"

    def format_instruction(q: str, doc: str) -> str:
        return (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction,
                query=q,
                doc=doc,
            )
        )

    def process_inputs(texts: List[str]):
        inputs = tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens
        padded = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in padded:
            padded[key] = padded[key].to(model.device)
        return padded

    def compute_scores(padded_inputs) -> List[float]:
        logits = model(**padded_inputs).logits[:, -1, :]
        true_vector = logits[:, token_true_id]
        false_vector = logits[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        return batch_scores[:, 1].exp().tolist()

    passages = [
        format_passage_for_rerank(p, max_chars=max_passage_chars) for p in candidates
    ]
    prompts = [format_instruction(query, doc) for doc in passages]

    scores_all: List[float] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            padded = process_inputs(batch_prompts)
            scores_all.extend([float(x) for x in compute_scores(padded)])

    rescored: List[models.ScoredPoint] = []
    for p, s in zip(candidates, scores_all):
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
    using_dense: str = "dense",
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
        using=using_dense,
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
    "get_jina_reranker_v3_model",
    "rerank_with_jina_v3",
    "get_bge_reranker_large_model",
    "rerank_with_bge_reranker_large",
    "get_gte_multilingual_reranker_base",
    "rerank_with_gte_multilingual_reranker_base",
    "get_granite_reranker_english_r2_model",
    "rerank_with_granite_english_r2",
    "get_qwen3_reranker_model",
    "rerank_with_qwen3_reranker",
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
