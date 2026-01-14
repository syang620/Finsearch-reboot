from __future__ import annotations
import ast
from pathlib import Path
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

def load_table_jsonl(
    tables_dir: str,
    doc_prefix: str,
    table_id: int | None = None,
) -> List[dict] | dict | None:
    """
    Load table content(s) from a <prefix>.tables.jsonl file.

    If table_id is None, returns all tables as a list.
    If table_id is set, returns the matching table dict or None.
    """
    path = Path(tables_dir) / f"{doc_prefix}.tables.jsonl"
    tables: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            if table_id is None:
                tables.append(obj)
            elif idx == table_id:
                return obj
    return tables if table_id is None else None

def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def simple_sentence_split(text: str) -> List[str]:
    # conservative splitter
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[\.!\?])\s+(?=[A-Z(])", text)

def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()

def parse_llm_list_output(text: str) -> list[str]:
    """
    Best-effort parse of an LLM "list" response into list[str].

    Supports:
      - JSON list (preferred)
      - Python literal list
      - Bullet/numbered lines
      - Comma-separated fallback
    """
    if not isinstance(text, str):
        return []
    t = _strip_code_fences(text)
    if not t or t.startswith("Error:") or t.startswith("Unexpected error:"):
        return []

    # 1) JSON list
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    # 2) Python literal list
    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    # 3) Bullets/numbered lines
    lines = []
    for line in t.splitlines():
        s = line.strip()
        s = re.sub(r"^[-*•]\s+", "", s)
        s = re.sub(r"^\d+[\.)]\s+", "", s)
        s = s.strip().strip('"').strip("'")
        if s:
            lines.append(s)
    if len(lines) >= 2:
        return lines

    # 4) Comma-separated fallback
    t2 = t.strip().strip("[]")
    parts = [p.strip().strip('"').strip("'") for p in t2.split(",")]
    return [p for p in parts if p]

def minmax_scale(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

def chat_with_ollama(
    prompt: str,
    model: str = "llama2",
    *,
    as_list: bool = True,
    options: dict[str, Any] | None = None,
):
    """
    Call a local Ollama server via the `ollama` Python package.

    Returns either the raw string response or `list[str]` if `as_list=True`.
    """
    try:
        import ollama
    except Exception as exc:
        raise ImportError(
            "chat_with_ollama requires the `ollama` Python package and a running Ollama server."
        ) from exc

    if options is None:
        options = {
            "temperature": 0.1,
            "num_predict": 512,
        }

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=options,
        )
        out = response["response"]
        return parse_llm_list_output(out) if as_list else out
    except Exception as e:
        return [] if as_list else f"Error: {e}"

def accounting_terms_to_llm_digest(
    terms: List[str],
    *,
    title: str = "Accounting Terms",
    max_terms: int | None = None,
) -> str:
    """
    Format a list of accounting terms into a compact text block for an LLM prompt.
    """
    cleaned: List[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s:
            cleaned.append(s)

    if max_terms is not None:
        cleaned = cleaned[:max_terms]

    lines = [f"{title} ({len(cleaned)}):"]
    lines.extend([f"- {t}" for t in cleaned])
    return "\n".join(lines)


def accounting_terms_file_to_llm_digest(
    path: str,
    *,
    title: str = "Accounting Terms",
    max_terms: int | None = None,
) -> str:
    """
    Read an accounting terms JSON file and return an LLM-friendly digest string.

    Accepts either:
      - list[str] (current format), or
      - {"terms": list[str]} (legacy format).
    """
    obj = load_json(path)
    if isinstance(obj, list):
        terms = obj
    elif isinstance(obj, dict) and isinstance(obj.get("terms"), list):
        terms = obj["terms"]
    else:
        raise ValueError(f"Unrecognized accounting terms format in {path}")

    return accounting_terms_to_llm_digest(
        terms,
        title=title,
        max_terms=max_terms,
    )

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
