from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from .utils import Chunk, simple_sentence_split

def best_narrative(chosen: List[Tuple[str,float]], chunks_by_id: Dict[str,Chunk]) -> Dict[str,Any]:
    # take top chunk and provide brief extract (first ~3 sentences)
    top_id, score = chosen[0]
    c = chunks_by_id[top_id]
    sents = simple_sentence_split(c.text)
    summary = " ".join(sents[:3]) if sents else c.text[:800]
    return {
        "type": "narrative",
        "answer": summary,
        "citations": [{
            "chunk_id": c.chunk_id,
            "form": c.filing.form_type,
            "section_path": c.section_path,
            "source_path": c.extra.get("source_path")
        }]
    }

def numeric_from_tables(metric_key: str, table_jsons: List[dict],
                        prefer_form: str = None) -> Dict[str,Any] | None:
    # search tables for row label matching alias set
    aliases = {
        "net sales": ["net sales","total net sales","net revenue","revenue","revenues"],
        "gross margin": ["gross margin","total gross margin"],
        "operating income": ["operating income","income from operations"],
        "eps": ["diluted earnings per share","earnings per share - diluted","earnings per share (diluted)","diluted"]
    }.get(metric_key, [metric_key])
    cand = []
    for t in table_jsons:
        if prefer_form and t["filing"]["form_type"] != prefer_form:
            continue
        for row in t["rows"]:
            label = str(row.get("label","")).strip().lower()
            if any(a in label for a in aliases):
                # pick the first non-null value column
                for col in t["columns"]:
                    v = row["values"].get(col["id"])
                    if v not in (None, "", "-"):
                        cand.append((t, row, col, v))
                        break
    if not cand:
        return None
    # choose the candidate from the latest filing period_end if present; else first
    best = cand[0]
    t, row, col, v = best
    return {
        "type": "numeric",
        "metric": metric_key,
        "value": v,
        "column_label": col["label"],
        "table_title": t["title"],
        "citations": [{
            "table_id": t["table_id"],
            "form": t["filing"]["form_type"],
            "section_path": t["location"]["section_path"],
            "source_path": t["filing"]["source_path"]
        }]
    }
