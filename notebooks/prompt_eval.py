# prompt_eval.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Sequence

import numpy as np
import pandas as pd
import requests


# -------------------------------------------------------------------
# Ollama embedding helper
# -------------------------------------------------------------------

def ollama_embed(
    texts: Sequence[str],
    model: str = "nomic-embed-text:latest",
    base_url: str = "http://localhost:11434",
) -> np.ndarray:
    """
    Embed a list of texts using Ollama's /api/embeddings endpoint.

    Returns:
        np.ndarray of shape (len(texts), dim)
    """
    vectors: List[np.ndarray] = []
    url = f"{base_url}/api/embeddings"

    for t in texts:
        payload = {"model": model, "prompt": t}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        obj = resp.json()
        emb = obj.get("embedding")
        if emb is None:
            raise ValueError(f"No 'embedding' field in response: {obj}")
        vectors.append(np.array(emb, dtype=np.float32))

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors)


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass
class PromptAnnotation:
    """
    Holds the annotation for one prompt variant for a single table.

    `annotation` is the raw dict from annotate_table_with_ollama, e.g.:

        {
          "table_summary": "...",
          "row_summaries": [
            {"row_index": 0, "row_label": "...", "description": "..."},
            ...
          ]
        }
    """
    prompt_id: str
    annotation: Dict[str, Any]


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between 1D vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def annotation_to_docs(pa: PromptAnnotation) -> List[Dict[str, Any]]:
    """
    Turn a PromptAnnotation into a list of doc dicts:

    Each doc has:
      - prompt_id
      - scope: "table" or "row"
      - row_index: int or None
      - row_label: str or None
      - text: summary text
    """
    ann = pa.annotation
    docs: List[Dict[str, Any]] = []

    # Table-level doc
    table_sum = ann.get("table_summary", "") or ""
    if table_sum:
        docs.append({
            "prompt_id": pa.prompt_id,
            "scope": "table",
            "row_index": None,
            "row_label": None,
            "text": table_sum.strip(),
        })

    # Row-level docs
    for row in ann.get("row_summaries", []) or []:
        idx = row.get("row_index")
        label = row.get("row_label")
        desc = (row.get("description") or "").strip()
        if not desc:
            continue
        docs.append({
            "prompt_id": pa.prompt_id,
            "scope": "row",
            "row_index": idx,
            "row_label": label,
            "text": desc,
        })

    return docs


# -------------------------------------------------------------------
# Main evaluation function
# -------------------------------------------------------------------

def score_prompt_annotations_for_query_with_rows(
    query: str,
    prompt_annotations: Sequence[PromptAnnotation],
    embed_model: str = "nomic-embed-text:latest",
    base_url: str = "http://localhost:11434",
) -> pd.DataFrame:
    """
    For a single query + single table, compare multiple prompt variants
    at both table summary and row summary level.

    Args
    ----
    query:
        Natural language question you care about, e.g.
        "What were net sales in Greater China in 2025?"
    prompt_annotations:
        List of PromptAnnotation, one per prompt variant.
    embed_model:
        Name of the Ollama embedding model, default "nomic-embed-text:latest".
    base_url:
        Ollama base URL, default "http://localhost:11434".

    Returns
    -------
    pandas.DataFrame with columns:
        - prompt_id
        - scope        ("table" or "row")
        - row_index    (None for table, int for row)
        - row_label    (None or str)
        - similarity
        - text         (summary text used for embedding)

    The DataFrame is sorted by similarity DESC.
    """
    # Build docs: one doc per table summary / row summary per prompt
    docs: List[Dict[str, Any]] = []
    for pa in prompt_annotations:
        docs.extend(annotation_to_docs(pa))

    if not docs:
        return pd.DataFrame(columns=[
            "prompt_id", "scope", "row_index", "row_label", "similarity", "text"
        ])

    # Embed query + all docs
    texts = [query] + [d["text"] for d in docs]
    vecs = ollama_embed(texts, model=embed_model, base_url=base_url)

    q_vec = vecs[0]
    doc_vecs = vecs[1:]

    sims = [cosine_sim(q_vec, v) for v in doc_vecs]

    # Attach similarities
    for d, sim in zip(docs, sims):
        d["similarity"] = sim

    # Build DataFrame
    df = pd.DataFrame(docs)
    df = df[["prompt_id", "scope", "row_index", "row_label", "similarity", "text"]]
    df = df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
    return df
