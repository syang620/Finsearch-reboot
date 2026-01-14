import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ---------- Generic helpers ----------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def embed_batch_with_qwen3(
    texts: List[str],
    api_url: str = "http://192.168.1.237:11434/api/embed",
    model: str = "qwen3-embedding:8b",
    timeout: float = 120.0,
) -> List[List[float]]:
    """
    Call Ollama's embedding endpoint on a batch of texts.
    Expects Ollama-style /api/embed with:
      { "model": model, "input": [text1, text2, ...] }
      -> { "embeddings": [[...], [...], ...] }
    """
    if not texts:
        return []

    payload = {
        "model": model,
        "input": texts,
    }
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError(f"Unexpected embedding response: {data}")

    return embeddings


# ---------- Builders for content strings ----------

def build_text_content(rec: Dict[str, Any]) -> Optional[str]:
    """
    Turn one text-chunk record into a string for embedding.
    Adjust field names here to match your text JSONL schema.
    """
    # Adjust these keys if your schema differs
    text = rec.get("text") or rec.get("chunk_text") or rec.get("content")
    if not isinstance(text, str) or not text.strip():
        return None

    section_path = rec.get("section_path")
    section_title = rec.get("section_title")

    header_bits = []
    if isinstance(section_path, str) and section_path.strip():
        header_bits.append(section_path.strip())
    if isinstance(section_title, str) and section_title.strip():
        # avoid duplicating if already in path
        if not header_bits or section_title not in header_bits[-1]:
            header_bits.append(section_title.strip())

    if header_bits:
        header = " – ".join(header_bits)
        return header + "\n\n" + text.strip()
    else:
        return text.strip()


def build_table_content_from_annotation(
    ann: Dict[str, Any],
    max_row_descriptions: int = 4,
) -> Optional[str]:
    """
    Build a table-level content string from the annotation dict:
      { "table_summary": "...", "row_summaries": [...] }
    """
    table_summary = ann.get("table_summary")
    if not isinstance(table_summary, str) or not table_summary.strip():
        return None

    parts = [f"Table summary: {table_summary.strip()}"]

    row_summaries = ann.get("row_summaries") or []
    desc_snippets: List[str] = []
    for row in row_summaries[:max_row_descriptions]:
        label = (row.get("row_label") or "").strip()
        desc = (row.get("description") or "").strip()
        if desc:
            if label:
                desc_snippets.append(f"{label}: {desc}")
            else:
                desc_snippets.append(desc)

    if desc_snippets:
        parts.append("Rows: " + " ".join(desc_snippets))

    return "\n".join(parts)


def build_row_content(row: Dict[str, Any]) -> Optional[str]:
    """
    Build a row-level content string from a single row_summaries entry:
      { "row_index": ..., "row_label": "...", "description": "..." }
    """
    label = (row.get("row_label") or "").strip()
    desc = (row.get("description") or "").strip()

    if not desc:
        return None

    if label:
        return f"{label}: {desc}"
    else:
        return desc


# ---------- Building docs (no embeddings yet) ----------

def build_text_docs(
    text_path: Path,
    common_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build 'text_chunk' docs (no embedding yet) from text JSONL file.
    Each doc:
      {
        "id": "...",
        "content": "...",
        "metadata": { ... }
      }
    """
    records = load_jsonl(text_path)
    docs: List[Dict[str, Any]] = []

    for rec in records:
        content = build_text_content(rec)
        if not content:
            continue

        prefix = rec.get("prefix", common_meta.get("prefix"))
        chunk_index = rec.get("chunk_index", len(docs))

        metadata = {
            **common_meta,
            "doc_type": "text_chunk",
            "prefix": prefix,
            "chunk_index": chunk_index,
            "section_title": rec.get("section_title"),
            "section_path": rec.get("section_path"),
            "source": "text",
        }

        doc = {
            "id": f"{prefix}::text::{chunk_index}",
            "content": content,
            "metadata": metadata,
        }
        docs.append(doc)

    return docs


def build_table_and_row_docs(
    table_summaries_path: Path,
    common_meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build 'table' and 'table_row' docs (no embedding yet) from table summaries JSONL.
    Returns:
      {
        "tables": [ ... ],
        "rows": [ ... ]
      }
    """
    records = load_jsonl(table_summaries_path)
    table_docs: List[Dict[str, Any]] = []
    row_docs: List[Dict[str, Any]] = []

    for rec in records:
        prefix = rec.get("prefix", common_meta.get("prefix"))
        table_index = rec.get("table_index")
        ann = rec.get("annotation") or {}
        error = rec.get("error")

        # ---- table-level doc ----
        if not error:
            table_content = build_table_content_from_annotation(ann)
        else:
            table_content = None

        if table_content:
            t_meta = {
                **common_meta,
                "doc_type": "table",
                "prefix": prefix,
                "table_index": table_index,
                "section_title": rec.get("section_title"),
                "section_path": rec.get("section_path"),
                "source": "table",
            }
            t_doc = {
                "id": f"{prefix}::table::{table_index}",
                "content": table_content,
                "metadata": t_meta,
            }
            table_docs.append(t_doc)

        # ---- row-level docs ----
        row_summaries = ann.get("row_summaries") or []
        for row in row_summaries:
            row_index = row.get("row_index")
            row_content = build_row_content(row)
            if not row_content:
                continue

            r_meta = {
                **common_meta,
                "doc_type": "table_row",
                "prefix": prefix,
                "table_index": table_index,
                "row_index": row_index,
                "row_label": row.get("row_label"),
                "section_title": rec.get("section_title"),
                "section_path": rec.get("heading_path"),
                "source": "table_row",
            }
            r_doc = {
                "id": f"{prefix}::table::{table_index}::row::{row_index}",
                "content": row_content,
                "metadata": r_meta,
            }
            row_docs.append(r_doc)

    return {"tables": table_docs, "rows": row_docs}


# ---------- Embedding driver ----------

def embed_docs(
    docs: List[Dict[str, Any]],
    api_url: str = "http://localhost:11434/api/embed",
    model: str = "qwen3-embedding:8b",
    batch_size: int = 16,
    timeout: float = 180.0,
) -> List[Dict[str, Any]]:
    """
    Take a list of docs {id, content, metadata} and return the same docs
    with an 'embedding' field added.
    """
    embedded_docs: List[Dict[str, Any]] = []

    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        texts = [d["content"] for d in batch]

        embeddings = embed_batch_with_qwen3(
            texts=texts,
            api_url=api_url,
            model=model,
            timeout=timeout,
        )

        if len(embeddings) != len(batch):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} vs {len(batch)}"
            )

        for d, emb in zip(batch, embeddings):
            d_with_emb = {**d, "embedding": emb}
            embedded_docs.append(d_with_emb)

        print(
            f"[EMBED] Embedded docs {start}–{start + len(batch) - 1} "
            f"with model={model}"
        )

    return embedded_docs
