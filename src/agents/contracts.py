# pydantic schema for Planner agent

"""
Pydantic schemas for the SEC RAG planner/orchestrator (crawl mode).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------
# Common enums / literals
# -----------------------------

class FormType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"


class PlannerIntent(str, Enum):
    FILING_FACT = "filing_fact"   # answerable from filings, mostly extraction
    FILING_CALC = "filing_calc"   # requires computation using retrieved filing facts
    DEFINITION = "definition"     # conceptual, can be answered without filings
    OTHER = "other"


class ContextQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContextItemKind(str, Enum):
    TABLE = "table"
    TEXT = "text"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


PlannerRoute = Literal["kb", "structured_fact", "hybrid"]


# -----------------------------
# Planner output (LLM contract)
# -----------------------------

class OpenIssue(BaseModel):
    """Planner-discovered uncertainty / ambiguity to surface downstream."""
    code: str = Field(..., description="Short machine-readable code, e.g. TICKER_MISSING")
    message: str = Field(..., description="Human-readable description")
    severity: Severity = Field(default=Severity.WARNING)
    metadata: Optional[Dict[str, Any]] = None


class FilingMetadata(BaseModel):
    """
    Filing constraints the planner extracts from user input.

    Crawl mode assumes a single primary ticker + year, matching the retrieval tool's signature.
    (You can extend to multi-ticker later in walk/run mode.)
    """
    ticker: Optional[str] = Field(
        default=None,
        description="Primary ticker symbol (e.g., AAPL). None if unknown/ambiguous.",
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Company name mentioned by the user (e.g., Apple). None if unknown/ambiguous.",
    )
    fiscal_year: Optional[int] = Field(
        default=None,
        description="Fiscal year (e.g., 2024). None if unknown/ambiguous.",
    )
    form_type: Optional[FormType] = Field(default=None)
    doc_types: Optional[List[str]] = Field(
        default=None,
        description="Retriever doc types (e.g., ['table']). If None, server defaults apply.",
    )
    fiscal_quarter: Optional[Literal["Q1", "Q2", "Q3", "Q4"]] = Field(
        default=None,
        description="Optional quarter constraint (primarily for 10-Q).",
    )

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        # Allow dot tickers like BRK.B; keep crawl-mode permissive
        if not v:
            return None
        return v

    @field_validator("company_name")
    @classmethod
    def _normalize_company_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        return v

    @field_validator("fiscal_year")
    @classmethod
    def _validate_year(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if not (1900 <= int(v) <= 2100):
            raise ValueError("fiscal_year out of reasonable range (1900-2100)")
        return int(v)

    @field_validator("doc_types")
    @classmethod
    def _normalize_doc_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned = [str(x).strip().lower() for x in v if str(x).strip()]
        return cleaned or None

class AnalysisTask(BaseModel):
    """
    High-level analysis instruction for the analyst agent.
    In crawl mode, keep it minimal and descriptive.
    """
    task_type: Literal["extract", "compute", "compare", "trend"] = Field(default="extract")
    metric: str = Field(..., description="Metric name, e.g. 'total debt', 'net debt', 'FCF'.")
    definition_notes: List[str] = Field(
        default_factory=list,
        description="Notes to disambiguate metric definition / formula expectations.",
    )
    requires_calculation: bool = Field(
        default=False,
        description="Whether the analyst must use grounded calculation tooling before returning a final answer.",
    )
    expected_artifacts: List[Literal["table", "row", "text"]] = Field(
        default_factory=lambda: ["table"],
        description="Which artifact types are expected to support the answer.",
    )
    output_format: Literal["short_answer", "step_by_step", "table"] = Field(default="step_by_step")

    @field_validator("metric")
    @classmethod
    def _metric_non_empty(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("metric must be non-empty")
        return v

    @field_validator("definition_notes")
    @classmethod
    def _normalize_notes(cls, v: List[str]) -> List[str]:
        return [str(x).strip() for x in v if str(x).strip()]


class StructuredFactRequest(BaseModel):
    subquestion: str = Field(..., description="User-facing subquestion for a future structured fact lane.")
    metric_hint: Optional[str] = Field(default=None)
    entity_hint: Optional[str] = Field(default=None)
    fiscal_year: Optional[int] = Field(default=None)
    fiscal_period: Optional[str] = Field(default=None)

    @field_validator("subquestion")
    @classmethod
    def _subquestion_non_empty(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("subquestion must be non-empty")
        return text

    @field_validator("metric_hint", "entity_hint", "fiscal_period")
    @classmethod
    def _normalize_optional_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        text = str(v).strip()
        return text or None

    @field_validator("fiscal_year")
    @classmethod
    def _validate_year(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        year = int(v)
        if not (1900 <= year <= 2100):
            raise ValueError("fiscal_year out of reasonable range (1900-2100)")
        return year


class PlannerOutput(BaseModel):
    """
    What the planner LLM must output (crawl mode).
    This is the core plan object that drives the orchestrator state machine.
    """
    retrieval_needed: bool = Field(default=True)
    intent: PlannerIntent = Field(default=PlannerIntent.FILING_FACT)
    route: PlannerRoute = Field(default="kb")
    structured_fact_requests: List[StructuredFactRequest] = Field(default_factory=list)
    metadata: FilingMetadata
    analysis_task: AnalysisTask
    open_issues: List[OpenIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sanity_checks(self) -> "PlannerOutput":
        # If planner says retrieval not needed, keep the issue list as a traceable signal.
        # but flag missing context expectation.
        if not self.retrieval_needed and self.intent in (PlannerIntent.FILING_FACT, PlannerIntent.FILING_CALC):
            self.open_issues.append(
                OpenIssue(
                    code="RETRIEVAL_DISABLED",
                    message="Planner set retrieval_needed=False for a filing-based intent.",
                    severity=Severity.WARNING,
                )
            )
        return self


# -----------------------------
# Retrieval request/response
# -----------------------------

class RetrievalRequest(BaseModel):
    """
    Request format aligned with SecRetrievalMCPClient.retrieve_tables(...)
    and the underlying MCP tool `sec_retrieve_tables`.
    """
    queries: List[str] = Field(..., description="1-4 short retrieval queries.")
    ticker: str = Field(..., description="Ticker symbol, required by retrieval tool.")
    fiscal_year: int = Field(..., description="Fiscal year, required by retrieval tool.")
    form_type: Optional[FormType] = Field(default=None)
    doc_types: Optional[List[str]] = None
    top_k: int = Field(default=3, ge=1, le=50)

    @field_validator("queries")
    @classmethod
    def _validate_queries(cls, v: List[str]) -> List[str]:
        cleaned = [str(x).strip() for x in (v or []) if str(x).strip()]
        if not cleaned:
            raise ValueError("queries must be non-empty")
        return cleaned[:4]

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        v = str(v).strip().upper()
        if not v:
            raise ValueError("ticker must be non-empty")
        return v

    @field_validator("fiscal_year")
    @classmethod
    def _validate_year(cls, v: int) -> int:
        v = int(v)
        if not (1900 <= v <= 2100):
            raise ValueError("fiscal_year out of reasonable range (1900-2100)")
        return v

    @field_validator("doc_types")
    @classmethod
    def _normalize_doc_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        cleaned = [str(x).strip().lower() for x in v if str(x).strip()]
        return cleaned or None


class TableCandidate(BaseModel):
    """
    A single retrieved table candidate.

    The retrieval pipeline returns dicts from `score_and_select_tables(...)`.
    Keys can vary by implementation; we keep this permissive and optionally
    expose common fields like `total_score`.
    """
    model_config = ConfigDict(extra="allow")

    total_score: Optional[float] = Field(default=None)
    table_id: Optional[str] = Field(default=None)
    doc_id: Optional[str] = Field(default=None)
    section_path: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _lift_common_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Try to map common variants
        if "total_score" in data and data.get("total_score") is not None:
            data["total_score"] = float(data["total_score"])
        if "table_id" not in data:
            for k in ("tableId", "table_key", "tableKey"):
                if k in data:
                    data["table_id"] = data[k]
                    break
        if "doc_id" not in data:
            for k in ("docId", "doc_key", "docKey"):
                if k in data:
                    data["doc_id"] = data[k]
                    break
        return data


class RetrieveTablesResponse(BaseModel):
    """
    Mirrors `RetrieveTablesResponse` in sec_retrieval.py (server side),
    but uses typed TableCandidate entries.
    """
    ok: bool = True
    queries_used: List[str] = Field(default_factory=list)
    rerank_query: str = ""
    top_tables: List[TableCandidate] = Field(default_factory=list)
    max_total_score: Optional[float] = None
    metadata_used: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None

    @field_validator("queries_used")
    @classmethod
    def _normalize_queries_used(cls, v: List[str]) -> List[str]:
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    @model_validator(mode="after")
    def _derive_max_score(self) -> "RetrieveTablesResponse":
        if self.max_total_score is None and self.top_tables:
            # Prefer the first candidate's total_score if available
            s0 = self.top_tables[0].total_score
            if s0 is not None:
                self.max_total_score = float(s0)
        return self


# -----------------------------
# Analyst context packet
# -----------------------------

class SourceRef(BaseModel):
    """Normalized provenance attached to context items for citation."""
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[FormType] = None

    # Optional rich provenance (extend later)
    filing_date: Optional[str] = None
    accession_no: Optional[str] = None
    section_path: Optional[str] = None
    doc_id: Optional[str] = None
    table_id: Optional[str] = None


class ContextItem(BaseModel):
    """
    Single context unit for the analyst agent.
    We store the raw candidate (dict) plus normalized provenance fields when available.
    """
    context_id: str = Field(..., description="Stable analyst-visible context identifier, e.g. ctx_1.")
    target_id: Optional[str] = Field(default=None, description="Stable target identifier for compare/multi-target analysis.")
    kind: ContextItemKind = ContextItemKind.TABLE
    source: SourceRef = Field(default_factory=SourceRef)

    # Raw content / payload
    payload: Dict[str, Any] = Field(default_factory=dict)

    # Scores are optional but helpful for debugging
    total_score: Optional[float] = None

    @classmethod
    def from_table_candidate(
        cls,
        cand: Union[TableCandidate, Dict[str, Any]],
        *,
        context_id: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        form_type: Optional[FormType] = None,
    ) -> "ContextItem":
        normalized_context_id = str(context_id or "").strip()
        if not normalized_context_id:
            raise ValueError("ContextItem.from_table_candidate requires a non-empty context_id.")
        c = cand if isinstance(cand, TableCandidate) else TableCandidate.model_validate(cand)
        payload = dict(c.model_dump(exclude_none=True))
        # Keep extras too
        if isinstance(cand, dict):
            payload = {**cand, **payload}

        source = SourceRef(
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            section_path=getattr(c, "section_path", None) or payload.get("section_path"),
            doc_id=getattr(c, "doc_id", None) or payload.get("doc_id"),
            table_id=getattr(c, "table_id", None) or payload.get("table_id"),
        )
        return cls(
            context_id=normalized_context_id,
            kind=ContextItemKind.TABLE,
            source=source,
            payload=payload,
            total_score=c.total_score,
        )


class AnalystPacket(BaseModel):
    """
    What the planner hands to the analyst agent.
    """
    plan_id: str = Field(..., description="Traceable ID for the end-to-end run.")
    user_query: str
    intent: PlannerIntent
    metadata: FilingMetadata
    analysis_task: AnalysisTask
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    context_items: List[ContextItem] = Field(default_factory=list)
    context_quality: ContextQuality = ContextQuality.MEDIUM
    open_issues: List[OpenIssue] = Field(default_factory=list)

    @field_validator("user_query")
    @classmethod
    def _user_query_non_empty(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("user_query must be non-empty")
        return v


# -----------------------------
# Orchestrator internal state (optional)
# -----------------------------

class OrchestratorState(BaseModel):
    """
    Optional: typed state object for the orchestrator state machine.
    Helps avoid ad-hoc dict passing.
    """
    plan_id: str
    user_query: str
    plan: Optional[PlannerOutput] = None
    retrieval_request: Optional[RetrievalRequest] = None
    retrieval_response: Optional[RetrieveTablesResponse] = None
    attempt: int = 0
    max_attempts: int = 2  # crawl mode: 1 try + 1 retry

    model_config = ConfigDict(extra="forbid")
