from __future__ import annotations
import re, json, os, math, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterable

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def file_stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def to_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8")]

def simple_sentence_split(text: str) -> List[str]:
    # conservative splitter
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[\.!\?])\s+(?=[A-Z(])", text)

def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

def minmax_scale(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

@dataclass
class FilingMeta:
    ticker: str
    form_type: str  # "10-K" | "10-Q"
    fiscal_year: Optional[int] = None
    period_type: Optional[str] = None  # "FY" | "Q1" | ...
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    source_path: Optional[str] = None

@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: List[str]
    is_table: bool
    statement_type: Optional[str]
    filing: FilingMeta
    extra: Dict[str, Any]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filing"] = asdict(self.filing)
        return d


def load_table_rowlabels_config(path: str) -> List[Dict[str, Any]]:
    """
    Load a table-rowlabel config JSON file.

    Supports either:
      - list[table] (current format), or
      - {"tables": list[table]} (legacy format).
    """
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("tables"), list):
        return obj["tables"]
    raise ValueError(f"Unrecognized config format in {path}")


def table_rowlabels_to_llm_digest(tables: List[Dict[str, Any]]) -> str:
    """
    Format table row-label schema into a compact text digest for an LLM.
    """
    blocks: List[str] = []
    for t in tables:
        table_id = t.get("table_id", "")
        purpose = t.get("table_purpose")
        row_labels = t.get("row_labels") or []

        header = [f"Table ID: {table_id}"]
        if isinstance(purpose, str) and purpose.strip():
            header.append(f"Purpose: {purpose.strip()}")

        block = "\n".join(header) + "\nRows:\n- " + "\n- ".join(
            [str(r) for r in row_labels]
        )

        aggregate_hints = t.get("aggregate_hints")
        if isinstance(aggregate_hints, dict) and aggregate_hints:
            block += "\nAggregate definitions:"
            for k, v in aggregate_hints.items():
                if isinstance(v, list):
                    block += f"\n- {k}: {', '.join(str(x) for x in v)}"
                else:
                    block += f"\n- {k}: {v}"

        blocks.append(block.strip())

    return "\n\n".join(blocks)


def table_rowlabels_config_file_to_llm_digest(path: str) -> str:
    """
    Convenience: read a row-label config JSON file and return an LLM digest string.
    """
    return table_rowlabels_to_llm_digest(load_table_rowlabels_config(path))
