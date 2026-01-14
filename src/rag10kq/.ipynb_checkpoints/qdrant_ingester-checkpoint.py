"""
qdrant_ingester.py

Utilities for uploading pre-computed embeddings into Qdrant collections.

Assumes each document has the following structure:

    {
        "id": "AAPL_10-K_2024::table::0::row::0",  # your logical ID
        "content": "text used for embedding / retrieval",
        "metadata": { ... },                       # dict with ticker, form_type, etc.
        "embedding": [float, float, ...]           # list[float], all same length
    }

Qdrant requirements:
- Point IDs must be unsigned ints or valid UUID strings.
- We map your string "id" -> deterministic UUID and store the original as "doc_id".
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, List, Sequence, Tuple, cast

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client / collection helpers
# ---------------------------------------------------------------------------

def get_qdrant_client(
    host: str = "localhost",
    port: int = 6333,
    https: bool = False,
    api_key: str | None = None,
) -> QdrantClient:
    """
    Create a QdrantClient for local or remote Qdrant.

    For your local Docker container, the defaults (localhost:6333) are fine.
    """
    client = QdrantClient(
        host=host,
        port=port,
        https=https,
        api_key=api_key,
    )
    logger.info("Connected to Qdrant at %s:%s (https=%s)", host, port, https)
    return client


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Create the collection if it doesn't exist.
    If it already exists, leave it as-is.
    """
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        logger.info(
            "Collection '%s' already exists; leaving as-is.",
            collection_name,
        )
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s' (dim=%d).", collection_name, vector_size)


def count_points(client: QdrantClient, collection_name: str) -> int:
    """
    Return the number of points stored in a collection.
    """
    res = client.count(collection_name=collection_name, exact=True)
    logger.info("Collection '%s' has %d points.", collection_name, res.count)
    return res.count


# ---------------------------------------------------------------------------
# ID mapping & conversion to Qdrant points
# ---------------------------------------------------------------------------

def make_qdrant_id(doc_id: str) -> str:
    """
    Deterministically map a logical string doc_id to a UUID string.

    Qdrant expects IDs to be unsigned integers or UUIDs. We preserve your
    original string ID in the payload as 'doc_id'.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def docs_to_points(
    docs: Sequence[Dict[str, Any]],
    *,
    dense_key: str = "embedding",
    dense_vector_name: str = "dense",
    include_bm25: bool = False,
    bm25_vector_name: str = "bm25",
    bm25_model: str = "Qdrant/bm25",
    bm25_avg_len: float | None = None,
    bge_m3_dense_key: str | None = None,
    bge_m3_dense_vector_name: str = "bge_m3_dense",
    bge_m3_sparse_key: str | None = None,
    bge_m3_sparse_vector_name: str = "bge_m3_sparse",
) -> List[models.PointStruct]:
    """
    Convert a sequence of embedded document dicts into Qdrant PointStructs.

    Vector handling:
      - If a doc already provides `vector` (preferred) or `embedding` as a dict,
        this function passes that through directly to Qdrant.
      - Otherwise, it can *build* a multi-vector dict from common field names
        (e.g. dense + bm25 + bge-m3 dense/sparse), so the vectors config isn't
        repeated per-document.

    Expected doc keys:
      - id: str
      - content: str
      - metadata: dict (optional)
      - plus vector fields (one of):
          - vector: any Qdrant-compatible vector payload
          - embedding: dense list[float] (or dict for multi-vector)
          - other keys referenced by `dense_key` / `bge_m3_*_key`
    """
    points: List[models.PointStruct] = []

    computed_bm25_avg_len: float | None = None
    if include_bm25 and bm25_avg_len is None and docs:
        lengths: List[int] = []
        for d in docs:
            content = d.get("content")
            if isinstance(content, str) and content.strip():
                lengths.append(len(content.split()))
        computed_bm25_avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    for d in docs:
        original_id = d["id"]

        # Pass-through: caller already prepared Qdrant-ready vectors per doc.
        if "vector" in d:
            vector: Any = d["vector"]
        else:
            vectors: Dict[str, Any] = {}

            if dense_key in d:
                vectors[dense_vector_name] = d[dense_key]
            elif dense_key == "embedding" and "embedding" in d:
                vectors[dense_vector_name] = d["embedding"]

            if include_bm25:
                avg_len = bm25_avg_len if bm25_avg_len is not None else computed_bm25_avg_len
                if avg_len is None:
                    raise ValueError("include_bm25=True requires bm25_avg_len or non-empty docs")
                vectors[bm25_vector_name] = models.Document(
                    text=cast(str, d["content"]),
                    model=bm25_model,
                    options={"avg_len": avg_len},
                )

            if bge_m3_dense_key is not None and bge_m3_dense_key in d:
                vectors[bge_m3_dense_vector_name] = d[bge_m3_dense_key]

            if bge_m3_sparse_key is not None and bge_m3_sparse_key in d:
                sparse_obj = d[bge_m3_sparse_key]
                if isinstance(sparse_obj, models.SparseVector):
                    vectors[bge_m3_sparse_vector_name] = sparse_obj
                elif isinstance(sparse_obj, dict) and "indices" in sparse_obj and "values" in sparse_obj:
                    vectors[bge_m3_sparse_vector_name] = models.SparseVector(
                        indices=sparse_obj["indices"],
                        values=sparse_obj["values"],
                    )
                else:
                    raise TypeError(
                        f"Unsupported sparse vector type for {bge_m3_sparse_key}: {type(sparse_obj)}"
                    )

            if not vectors:
                raise KeyError(
                    "No vectors found for doc; provide `vector`, `embedding` (dict), "
                    f"or fields referenced by dense_key={dense_key!r}/bge_m3_*_key."
                )

            vector = vectors

        qdrant_id = make_qdrant_id(original_id)

        # Payload is content + metadata + your original logical ID
        payload: Dict[str, Any] = {
            "doc_id": original_id,
            "content": d["content"],
        }
        # Metadata should be a flat dict (JSON-serializable types).
        if "metadata" in d and isinstance(d["metadata"], dict):
            payload.update(d["metadata"])

        point = models.PointStruct(
            id=qdrant_id,
            vector=vector,
            payload=payload,
        )
        points.append(point)

    return points


# ---------------------------------------------------------------------------
# Ingestion / upsert
# ---------------------------------------------------------------------------

def upsert_docs(
    client: QdrantClient,
    collection_name: str,
    docs: Sequence[Dict[str, Any]],
    batch_size: int = 128,
) -> None:
    """
    Upsert a list of embedded docs into a Qdrant collection in batches.

    Parameters
    ----------
    client : QdrantClient
        Active client.
    collection_name : str
        Name of the collection you want to write to.
    docs : sequence of dict
        Each dict must have keys 'id', 'content', and optionally 'metadata'.
        For vectors, either supply:
          - 'vector' (Qdrant-ready), or
          - 'embedding' (dense list[float]) or dict (multi-vector).
    batch_size : int
        Number of docs to send per upsert call.
    """
    total = len(docs)
    if total == 0:
        logger.warning("No docs to upsert into '%s'.", collection_name)
        return

    logger.info(
        "Upserting %d docs into collection '%s' (batch_size=%d)...",
        total, collection_name, batch_size,
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = docs[start:end]
        points = docs_to_points(batch)

        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info("  - upserted docs %d–%d", start, end - 1)

    logger.info("Finished upserting %d docs into '%s'.", total, collection_name)


# ---------------------------------------------------------------------------
# Debug / inspection helpers
# ---------------------------------------------------------------------------

def scroll_points(
    client: QdrantClient,
    collection_name: str,
    limit: int = 10,
    with_vectors: bool = False,
) -> Tuple[List[models.Record], Any]:
    """
    Scroll through points in a collection (for debugging / inspection).

    Returns (points, next_page_offset). Prints a small preview to the logger.
    """
    points, next_page = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
    )

    logger.info("Scrolled %d points from '%s'.", len(points), collection_name)
    for p in points:
        payload = p.payload or {}
        logger.info(
            "ID=%s doc_id=%s ticker=%s content_preview=%r",
            p.id,
            payload.get("doc_id"),
            payload.get("ticker"),
            (payload.get("content") or "")[:120],
        )

    return points, next_page
