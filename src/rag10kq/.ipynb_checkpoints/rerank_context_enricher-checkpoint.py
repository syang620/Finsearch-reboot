from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from qdrant_client import QdrantClient, models


def make_qdrant_id(doc_id: str) -> str:
    """
    Deterministically map a logical string doc_id to a UUID string.

    Must match the ID mapping used during ingestion.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _payload_doc_id(p: models.ScoredPoint) -> str:
    md = p.payload or {}
    return str(md.get("doc_id") or md.get("id") or "")


def _normalize_doc_id_to_table(doc_id: str) -> str:
    if not doc_id:
        return ""
    if "::table::" in doc_id and "::row::" in doc_id:
        return doc_id.split("::row::", 1)[0]
    return doc_id


def _default_table_key_fn(p: models.ScoredPoint) -> str:
    return _normalize_doc_id_to_table(_payload_doc_id(p))


def fetch_table_summaries_for_candidates(
    candidates: List[models.ScoredPoint],
    *,
    client: QdrantClient,
    collection_name: str,
    batch_size: int = 256,
    table_key_fn: Callable[[models.ScoredPoint], str] | None = None,
    table_content_field: str = "content",
) -> Dict[str, str]:
    """
    Batch-fetch parent table "summary/content" for a list of candidate points.

    Returns:
      table_doc_id -> table_summary (string)
    """
    if not candidates:
        return {}

    if table_key_fn is None:
        table_key_fn = _default_table_key_fn

    table_doc_ids = sorted({table_key_fn(p) for p in candidates if table_key_fn(p)})
    if not table_doc_ids:
        return {}

    table_point_ids = [make_qdrant_id(did) for did in table_doc_ids]
    id_to_doc_id = dict(zip(table_point_ids, table_doc_ids))

    out: Dict[str, str] = {}
    for i in range(0, len(table_point_ids), batch_size):
        batch_ids = table_point_ids[i : i + batch_size]
        points = client.retrieve(
            collection_name=collection_name,
            ids=batch_ids,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            doc_id = id_to_doc_id.get(str(pt.id))
            if not doc_id:
                continue
            payload = pt.payload or {}
            val = payload.get(table_content_field) or ""
            if isinstance(val, str) and val.strip():
                out[doc_id] = val
    return out


def build_rerank_text(
    p: models.ScoredPoint,
    table_summary_by_doc_id: Dict[str, str],
    *,
    table_key_fn: Callable[[models.ScoredPoint], str] | None = None,
    content_field: str = "content",
    joiner: str = "\n\n",
) -> str:
    """
    Build the rerank passage text for a candidate by prepending its parent table summary (if available).
    """
    if table_key_fn is None:
        table_key_fn = _default_table_key_fn

    md = p.payload or {}
    row_or_chunk = md.get(content_field) or ""
    table_doc_id = table_key_fn(p)
    table_summary = table_summary_by_doc_id.get(table_doc_id, "")

    if isinstance(table_summary, str) and table_summary.strip():
        # Avoid double-including identical text.
        if table_summary.strip() == str(row_or_chunk).strip():
            return str(row_or_chunk)
        return f"{table_summary}{joiner}{row_or_chunk}".strip()

    return str(row_or_chunk)


def enrich_point_for_rerank(
    p: models.ScoredPoint,
    table_summary_by_doc_id: Dict[str, str],
    *,
    table_key_fn: Callable[[models.ScoredPoint], str] | None = None,
    content_field: str = "content",
    keep_original_fields: bool = True,
    joiner: str = "\n\n",
) -> models.ScoredPoint:
    """
    Return a copy of `p` with payload[content_field] replaced by the enriched rerank text.
    """
    if table_key_fn is None:
        table_key_fn = _default_table_key_fn

    payload = dict(p.payload or {})
    original = payload.get(content_field) or ""
    table_doc_id = table_key_fn(p)
    table_summary = table_summary_by_doc_id.get(table_doc_id, "")

    enriched = original
    if isinstance(table_summary, str) and table_summary.strip():
        if str(table_summary).strip() != str(original).strip():
            enriched = f"{table_summary}{joiner}{original}".strip()

    payload[content_field] = enriched

    if keep_original_fields:
        payload.setdefault("rerank_original_content", original)
        payload.setdefault("rerank_table_summary", table_summary)
        payload.setdefault("rerank_table_doc_id", table_doc_id)

    data: Dict[str, Any] = p.model_dump() if hasattr(p, "model_dump") else p.dict()
    data["payload"] = payload
    return models.ScoredPoint(**data)

