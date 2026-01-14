"""
sec_retrieval.py

Unified retrieval over SEC filings stored in Qdrant.

Collections assumed:
- sec_10k_text_chunks
- sec_10k_tables
- sec_10k_table_rows

Each point payload should include:
- doc_id (original logical ID, e.g. "AAPL_10-K_2024::table::0::row::0")
- ticker
- form_type
- fiscal_year
- doc_type ("text_chunk", "table", "table_row")
- content (text used for embedding)
- section_title, section_path, item_id, item_title, etc. (optional but useful)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def build_sec_filter(
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    form_type: Optional[str] = None,
) -> Optional[models.Filter]:
    """
    Build a Qdrant filter for SEC filings.

    If all arguments are None, returns None (no filter).
    """
    must: List[models.FieldCondition] = []

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


# ---------------------------------------------------------------------------
# Core search helpers
# ---------------------------------------------------------------------------

def _search_collection(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    k: int = 10,
    flt: Optional[models.Filter] = None,
) -> List[models.ScoredPoint]:
    """
    Low-level helper: search a single collection with a vector + optional filter.
    """
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=flt,
        limit=k,
        with_payload=True,
        with_vectors=False,
    )
    return hits


def search_sec(
    query: str,
    *,
    client: QdrantClient,
    embed_fn,  # function: str -> List[float]
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    form_type: Optional[str] = None,
    k_text: int = 8,
    k_tables: int = 5,
    k_rows: int = 10,
) -> Dict[str, Any]:
    """
    Embed a query and search across SEC collections in Qdrant.

    Parameters
    ----------
    query : str
        User query text.
    client : QdrantClient
        Active Qdrant client.
    embed_fn : callable
        Function that takes a string and returns a single embedding vector (list[float]),
        e.g. wrapping qwen3-embedding:8b via Ollama.
    ticker, fiscal_year, form_type : optional
        Filter down to a specific filing universe (e.g., ticker="AAPL", fiscal_year=2024, form_type="10-K").
    k_text, k_tables, k_rows : int
        Top-k results per collection.

    Returns
    -------
    dict with keys:
        - "query"
        - "query_vector"
        - "text_hits"
        - "table_hits"
        - "row_hits"
    """
    # 1) Embed query
    query_vec = embed_fn(query)

    # 2) Build filter
    flt = build_sec_filter(ticker=ticker, fiscal_year=fiscal_year, form_type=form_type)

    # 3) Search each collection
    text_hits = _search_collection(
        client,
        "sec_10k_text_chunks",
        query_vec,
        k=k_text,
        flt=flt,
    )

    table_hits = _search_collection(
        client,
        "sec_10k_tables",
        query_vec,
        k=k_tables,
        flt=flt,
    )

    row_hits = _search_collection(
        client,
        "sec_10k_table_rows",
        query_vec,
        k=k_rows,
        flt=flt,
    )

    return {
        "query": query,
        "query_vector": query_vec,
        "text_hits": text_hits,
        "table_hits": table_hits,
        "row_hits": row_hits,
    }
