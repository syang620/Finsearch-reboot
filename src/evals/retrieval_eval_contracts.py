from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

_FILING_KEY_RE = re.compile(
    r"^(?P<ticker>[A-Za-z0-9._-]+)_(?P<form>10-[A-Za-z0-9]+)_(?P<year>\d{4})(?P<rest>[A-Za-z0-9]*)$",
    flags=re.IGNORECASE,
)


def _strip_split_suffix(doc_id: str) -> str:
    if not doc_id:
        return doc_id
    return doc_id.split("::split::", 1)[0]


def _canonicalize_doc_id_for_text(prefix: str, value: Union[int, str, None]) -> Optional[str]:
    if value is None:
        return None

    base_prefix = prefix.split("::text::", 1)[0]

    if isinstance(value, int):
        return f"{base_prefix}::text::{int(value)}"

    text = str(value).strip()
    if not text:
        return None

    if "::text::" in text:
        return _strip_split_suffix(text)

    if text.startswith(base_prefix):
        rest = text[len(base_prefix) :].strip()
        if rest.startswith("::"):
            rest = rest[2:]
            if rest:
                return f"{base_prefix}::text::{_strip_split_suffix(rest)}"
        if text.isdigit():
            return f"{base_prefix}::text::{int(text)}"

    if text.isdigit():
        return f"{prefix}::text::{int(text)}"

    return _strip_split_suffix(text)


def _parse_filing_keys(filing_keys: List[str]) -> List[tuple[str, str, int]]:
    out: List[tuple[str, str, int]] = []
    for key in filing_keys:
        if not isinstance(key, str):
            continue
        m = _FILING_KEY_RE.match(key.strip())
        if not m:
            continue
        ticker = m.group("ticker").upper()
        form = m.group("form").upper()
        year = int(m.group("year"))
        out.append((ticker, form, year))
    return out


class RelevantTable(BaseModel):
    table_index: int = Field(..., ge=0)
    section_title: Optional[str] = None
    section_path: Optional[str] = None
    item_id: Optional[str] = None
    item_title: Optional[str] = None


class RetrievalEvalExample(BaseModel):
    query_id: Union[int, str]
    query: str
    gold_answer: Optional[str] = None
    relevant_tables: List[RelevantTable] = Field(default_factory=list)
    relevant_doc_ids: List[Union[int, str]] = Field(default_factory=list)
    relevant_chunk_ids: List[Union[int, str]] = Field(default_factory=list)
    relevant_chunk_uids: List[str] = Field(default_factory=list)
    relevant_headings: List[str] = Field(default_factory=list)

    # Optional per-row metadata overrides.
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[str] = None
    doc_types: Optional[List[str]] = None
    filing_keys: List[str] = Field(default_factory=list)
    tickers: List[str] = Field(default_factory=list)
    years: List[int] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def _query_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("query must be non-empty")
        return value

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().upper()
        return value or None

    @field_validator("fiscal_year")
    @classmethod
    def _validate_fiscal_year(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        iv = int(value)
        if not (1900 <= iv <= 2100):
            raise ValueError("fiscal_year out of range (1900-2100)")
        return iv

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().upper()
        return value or None

    @field_validator("doc_types")
    @classmethod
    def _normalize_doc_types(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = [str(v).strip().lower() for v in value if str(v).strip()]
        return cleaned or None

    @property
    def example_id(self) -> str:
        return str(self.query_id)

    def relevant_table_indices(self) -> List[int]:
        return sorted({int(t.table_index) for t in self.relevant_tables})

    def infer_ticker(self, default_ticker: str) -> str:
        if self.ticker:
            return self.ticker
        if self.tickers:
            return str(self.tickers[0]).strip().upper()

        parsed = _parse_filing_keys(self.filing_keys)
        if parsed:
            return parsed[0][0]

        return default_ticker

    def infer_form_type(self, default_form_type: str) -> str:
        if self.form_type:
            return self.form_type
        parsed = _parse_filing_keys(self.filing_keys)
        if parsed:
            return parsed[0][1]
        return default_form_type

    def infer_fiscal_year(self, default_fiscal_year: int) -> int:
        if self.fiscal_year is not None:
            return int(self.fiscal_year)

        if self.years:
            try:
                return int(self.years[0])
            except Exception:
                pass

        parsed = _parse_filing_keys(self.filing_keys)
        if parsed:
            return int(parsed[0][2])

        return int(default_fiscal_year)

    def relevant_text_doc_ids(self, *, ticker: str, fiscal_year: int, form_type: str) -> List[str]:
        """
        Normalize text gold labels into canonical doc_id strings used in Qdrant payloads.
        Supports either:
          - integer chunk ids (e.g., 41) -> AAPL_10-K_2024::text::41
          - fully-qualified doc_id strings
        """
        out: List[str] = []
        prefix = f"{ticker}_{form_type}_{int(fiscal_year)}::text::"
        raw_labels: List[Union[int, str]] = []
        raw_labels.extend(self.relevant_doc_ids)
        raw_labels.extend(self.relevant_chunk_ids)

        for value in raw_labels:
            doc_id = _canonicalize_doc_id_for_text(prefix, value)
            if doc_id:
                out.append(doc_id)

        return sorted(set(out))

    def has_table_labels(self) -> bool:
        return bool(self.relevant_tables)

    def has_text_labels(self) -> bool:
        return bool(self.relevant_doc_ids or self.relevant_chunk_ids)


class RetrievalEvalRow(BaseModel):
    id: str
    mode: str = "table"
    query: str
    ticker: Optional[str]
    fiscal_year: Optional[int]
    form_type: Optional[str]

    relevant_table_indices: List[int] = Field(default_factory=list)
    relevant_text_doc_ids: List[str] = Field(default_factory=list)
    retrieved_table_indices: List[Optional[int]] = Field(default_factory=list)
    retrieved_doc_ids: List[str] = Field(default_factory=list)

    metrics: Dict[str, float] = Field(default_factory=dict)
    retrieval_ok: bool = False
    retrieval_error: Optional[str] = None

    ragas: Optional[Dict[str, float]] = None
    trace: Dict[str, Any] = Field(default_factory=dict)


class RetrievalEvalSummary(BaseModel):
    num_queries: int
    num_valid_queries: int
    num_failures: int

    deterministic: Dict[str, float] = Field(default_factory=dict)
    ragas: Dict[str, float] = Field(default_factory=dict)

    config: Dict[str, Any] = Field(default_factory=dict)


def load_retrieval_eval_examples(path: str | Path) -> List[RetrievalEvalExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")

    examples: List[RetrievalEvalExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(RetrievalEvalExample.model_validate(obj))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid eval row at line {line_no}: {exc}") from exc
    return examples
