# pydantic schema for Planner agent

"""
Pydantic schemas for the SEC RAG planner/orchestrator (crawl mode).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -----------------------------
# Common enums / literals
# -----------------------------

class FormType(str, Enum):
    TEN_K = "10-K"
    TEN_K_A = "10-K/A"
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
    STRUCTURED_FACT = "structured_fact"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EvidenceLaneStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"


PlannerRoute = Literal["kb", "structured_fact", "hybrid"]
PlannerTaskClass = Literal[
    "single_target_fact",
    "multi_target_compare",
    "multi_target_screen",
    "other",
]


# -----------------------------
# Planner output (LLM contract)
# -----------------------------

class OpenIssue(BaseModel):
    """Planner-discovered uncertainty / ambiguity to surface downstream."""
    code: str = Field(..., description="Short machine-readable code, e.g. TICKER_MISSING")
    message: str = Field(..., description="Human-readable description")
    severity: Severity = Field(default=Severity.WARNING)
    metadata: Optional[Dict[str, Any]] = None


class EvidenceLaneSummary(BaseModel):
    """Authoritative public outcome for one runtime evidence lane."""

    model_config = ConfigDict(extra="forbid")

    requested: bool = False
    attempted: bool = False
    status: EvidenceLaneStatus = EvidenceLaneStatus.NOT_REQUESTED
    issues: List[OpenIssue] = Field(default_factory=list)


class EvidenceLaneStatusSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb: EvidenceLaneSummary = Field(default_factory=EvidenceLaneSummary)
    structured_fact: EvidenceLaneSummary = Field(default_factory=EvidenceLaneSummary)


class DegradationSummary(BaseModel):
    """Sanitized evidence-coverage disclosure shared with callers and analyst."""

    model_config = ConfigDict(extra="forbid")

    active: bool = False
    affected_lanes: List[Literal["kb", "structured_fact"]] = Field(
        default_factory=list
    )
    notice: str = ""

    @model_validator(mode="after")
    def _validate_active_shape(self) -> "DegradationSummary":
        canonical = [
            lane
            for lane in ("kb", "structured_fact")
            if lane in self.affected_lanes
        ]
        if self.affected_lanes != canonical:
            raise ValueError("affected_lanes must be unique and canonically ordered")
        if self.active != bool(self.affected_lanes):
            raise ValueError("active must match whether affected_lanes is non-empty")
        if self.active != bool(self.notice):
            raise ValueError("notice must be present exactly when degradation is active")
        return self


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
    model_config = ConfigDict(extra="forbid")

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


class PlannerTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: int = Field(..., ge=1)
    target_key: str
    company_name: Optional[str]
    ticker: Optional[str]
    fiscal_year: Optional[int]
    form_type: Optional[FormType]

    @field_validator("target_key")
    @classmethod
    def _target_key_non_empty(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("target_key must be non-empty")
        return text

    @field_validator("company_name", "ticker")
    @classmethod
    def _normalize_optional_target_text(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        text = str(v).strip()
        return text or None

    @field_validator("ticker")
    @classmethod
    def _normalize_target_ticker(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v is not None else None

    @field_validator("fiscal_year")
    @classmethod
    def _validate_target_year(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        year = int(v)
        if not (1900 <= year <= 2100):
            raise ValueError("fiscal_year out of reasonable range (1900-2100)")
        return year


class RetrievalPlanJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applies_to_target_ids: List[int]
    goal: str
    job_type: Literal["metric_extract", "narrative_extract"]

    @field_validator("applies_to_target_ids")
    @classmethod
    def _validate_target_ids(cls, v: List[int]) -> List[int]:
        target_ids = [int(target_id) for target_id in v]
        if not target_ids:
            raise ValueError("applies_to_target_ids must be non-empty")
        if any(target_id < 1 for target_id in target_ids):
            raise ValueError("applies_to_target_ids must contain positive integers")
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("applies_to_target_ids must be unique")
        return target_ids

    @field_validator("goal")
    @classmethod
    def _goal_non_empty(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("goal must be non-empty")
        return text


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fanout_mode: Literal["single_target", "per_target"]
    jobs: List[RetrievalPlanJob]

    @field_validator("jobs")
    @classmethod
    def _jobs_non_empty(cls, v: List[RetrievalPlanJob]) -> List[RetrievalPlanJob]:
        if not v:
            raise ValueError("jobs must be non-empty")
        return v


class PlannerClarificationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str

    @field_validator("question")
    @classmethod
    def _question_non_empty(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("question must be non-empty")
        return text

    @field_validator("answer")
    @classmethod
    def _normalize_answer(cls, v: str) -> str:
        return str(v).strip()


class PlannerClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Optional[str]
    questions: List[str]

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        text = str(v).strip()
        return text or None

    @field_validator("questions")
    @classmethod
    def _questions_non_empty(cls, v: List[str]) -> List[str]:
        questions = [str(question).strip() for question in v]
        if not questions or any(not question for question in questions):
            raise ValueError("questions must contain at least one non-empty question")
        return questions


class PlannerRuntimeOutput(BaseModel):
    """Normalized planner output consumed by the orchestrator runtime."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "needs_clarification", "error"]
    retrieval_needed: bool
    intent: PlannerIntent
    route: PlannerRoute
    structured_fact_requests: List[StructuredFactRequest]
    metadata: FilingMetadata
    analysis_task: AnalysisTask
    task_class: PlannerTaskClass
    targets: List[PlannerTarget]
    retrieval_plan: Optional[RetrievalPlan]
    open_issues: List[OpenIssue]
    original_user_query: str
    effective_user_query: str
    clarification_history: List[PlannerClarificationTurn]
    clarification_request: Optional[PlannerClarification]

    @field_validator("original_user_query", "effective_user_query")
    @classmethod
    def _query_non_empty(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("query fields must be non-empty")
        return text

    @model_validator(mode="after")
    def _validate_runtime_semantics(self) -> "PlannerRuntimeOutput":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target_id values must be unique")

        target_keys = [target.target_key for target in self.targets]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("target_key values must be unique")

        if self.retrieval_plan is not None:
            known_target_ids = set(target_ids)
            for job in self.retrieval_plan.jobs:
                unknown_target_ids = sorted(
                    set(job.applies_to_target_ids) - known_target_ids
                )
                if unknown_target_ids:
                    raise ValueError(
                        "retrieval plan references unknown target IDs: "
                        f"{unknown_target_ids}"
                    )

        if self.status != "completed":
            if self.retrieval_needed or self.retrieval_plan is not None:
                raise ValueError(
                    "non-completed planner output cannot contain executable retrieval"
                )
            if self.structured_fact_requests:
                raise ValueError(
                    "non-completed planner output cannot contain structured fact requests"
                )
        elif self.route == "kb":
            if self.structured_fact_requests:
                raise ValueError("kb route cannot contain structured fact requests")
            if (
                self.intent in {PlannerIntent.FILING_FACT, PlannerIntent.FILING_CALC}
                and not self.retrieval_needed
            ):
                raise ValueError(
                    "kb route requires retrieval for filing-based intents"
                )
            if self.retrieval_needed != (self.retrieval_plan is not None):
                raise ValueError(
                    "kb retrieval plan must be present iff retrieval is needed"
                )
        elif self.route == "structured_fact":
            if self.retrieval_needed or self.retrieval_plan is not None:
                raise ValueError("structured_fact route cannot execute KB retrieval")
            if not self.structured_fact_requests:
                raise ValueError(
                    "structured_fact route requires structured fact requests"
                )
        elif self.route == "hybrid":
            if not self.retrieval_needed or self.retrieval_plan is None:
                raise ValueError("hybrid route requires KB retrieval")
            if not self.structured_fact_requests:
                raise ValueError("hybrid route requires structured fact requests")

        if self.status == "needs_clarification":
            if self.clarification_request is None:
                raise ValueError(
                    "needs_clarification status requires a clarification request"
                )
        elif self.clarification_request is not None:
            raise ValueError(
                "clarification_request must be null unless clarification is needed"
            )

        return self


# Backward-compatible alias. Prefer PlannerRuntimeOutput in new code.
PlannerOutput = PlannerRuntimeOutput


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
    report_date: Optional[str] = None
    source_url: Optional[str] = None
    section_path: Optional[str] = None
    doc_id: Optional[str] = None
    table_id: Optional[str] = None


def normalize_missing_component_groups(value: Any) -> List[str]:
    """Validate and normalize missing structured-metric component group IDs."""

    if not isinstance(value, list):
        raise ValueError("missing_component_groups must be a list of strings")

    normalized_groups: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("missing_component_groups must contain only strings")
        normalized = item.strip()
        if not normalized:
            raise ValueError("missing_component_groups must not contain blank strings")
        normalized_groups.append(normalized)
    return normalized_groups


class StructuredFactEvidence(BaseModel):
    """Typed analyst-boundary representation of a successful SEC metric result."""

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    metric_label: str
    status: Literal["ok"] = "ok"
    value: float
    unit: Optional[str] = None

    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[FormType] = None

    accession_number: Optional[str] = None
    report_date: Optional[str] = None
    filed_date: Optional[str] = None
    source_url: Optional[str] = None
    start_date: Optional[str] = None

    components: List[Dict[str, Any]] = Field(default_factory=list)
    missing_component_groups: List[str] = Field(default_factory=list)

    @field_validator(
        "unit",
        "accession_number",
        "report_date",
        "filed_date",
        "source_url",
        "start_date",
    )
    @classmethod
    def _normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("metric_id", "metric_label")
    @classmethod
    def _normalize_required_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("value must be non-empty")
        return normalized

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().upper()
        return normalized or None

    @field_validator("fiscal_year")
    @classmethod
    def _validate_fiscal_year(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        year = int(value)
        if not 1900 <= year <= 2100:
            raise ValueError("fiscal_year out of reasonable range (1900-2100)")
        return year

    @field_validator("value", mode="before")
    @classmethod
    def _validate_numeric_value(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be a finite numeric value and not bool")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("value must be finite")
        return numeric

    @field_validator("missing_component_groups", mode="before")
    @classmethod
    def _normalize_missing_groups(cls, value: Any) -> List[str]:
        return normalize_missing_component_groups(value)


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
    structured_fact: Optional[StructuredFactEvidence] = None

    # Scores are optional but helpful for debugging
    total_score: Optional[float] = None

    @model_validator(mode="after")
    def _validate_kind_payload(self) -> "ContextItem":
        if self.kind == ContextItemKind.STRUCTURED_FACT:
            if self.structured_fact is None:
                raise ValueError("structured_fact context requires typed structured_fact evidence")
            if self.payload:
                raise ValueError("structured_fact context must not use payload as evidence")
        elif self.structured_fact is not None:
            raise ValueError("non-structured context cannot contain structured_fact evidence")
        return self

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
    lanes: EvidenceLaneStatusSet = Field(default_factory=EvidenceLaneStatusSet)
    degradation: DegradationSummary = Field(default_factory=DegradationSummary)

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
