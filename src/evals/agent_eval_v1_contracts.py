from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evals.agent_eval_contracts import PlannerIntentStr


AgentRoute = Literal["kb", "structured_fact", "hybrid"]
ReportedStatusExpectation = Literal["completed", "degraded", "failed", "interrupted"]
EffectiveStatus = Literal["completed", "degraded", "failed", "interrupted"]
StatusSource = Literal["runtime", "evaluator_derived"]
LaneStatusSource = Literal["runtime", "evaluator_derived"]


class AgentExpectedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Optional[PlannerIntentStr] = None
    route: Optional[AgentRoute] = None

    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[str] = None

    expect_kb_lane: Optional[bool] = None
    expect_structured_fact_lane: Optional[bool] = None
    expected_structured_metric: Optional[str] = None
    expected_structured_status: Optional[str] = None

    expected_reported_status: Optional[ReportedStatusExpectation] = None
    expected_effective_status: Optional[EffectiveStatus] = None
    expected_failure_stage: Optional[str] = None
    expected_analyst_status: Optional[str] = None

    must_use_financial_evaluator: Optional[bool] = None
    expected_metric: Optional[str] = None
    expected_min_citations: Optional[int] = Field(default=None, ge=0)
    allow_degraded: bool = False

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
            raise ValueError("fiscal_year out of range (1900-2100)")
        return year

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().upper()
        return normalized or None

    @field_validator(
        "expected_structured_metric",
        "expected_structured_status",
        "expected_failure_stage",
        "expected_analyst_status",
        "expected_metric",
    )
    @classmethod
    def _normalize_expected_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    @model_validator(mode="after")
    def _degraded_expectation_requires_permission(self) -> "AgentExpectedV1":
        if self.expected_effective_status == "degraded" and not self.allow_degraded:
            raise ValueError(
                "expected_effective_status='degraded' requires allow_degraded=True"
            )
        return self


class AgentEvalExampleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    user_query: str
    gold_answer: str
    expected: AgentExpectedV1 = Field(default_factory=AgentExpectedV1)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("id", "user_query", "gold_answer")
    @classmethod
    def _required_text_non_empty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("value must be non-empty")
        return normalized

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: List[str]) -> List[str]:
        return [
            normalized
            for item in value or []
            if (normalized := str(item).strip().lower())
        ]


class AgentLaneObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requested: bool = False
    attempted: bool = False
    status: str = "not_requested"
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    usable: bool = False
    usable_evidence_count: int = Field(default=0, ge=0)

    @field_validator("status")
    @classmethod
    def _normalize_status(cls, value: str) -> str:
        return str(value or "").strip().lower() or "unknown"


class AgentLaneStatusSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kb: AgentLaneObservation
    structured_fact: AgentLaneObservation


class AgentDeterministicChecksV1(BaseModel):
    contract_valid: bool = False
    intent_match: Optional[bool] = None
    route_match: Optional[bool] = None
    metadata_match: Optional[bool] = None
    kb_lane_match: Optional[bool] = None
    structured_lane_match: Optional[bool] = None
    structured_metric_match: Optional[bool] = None
    structured_status_match: Optional[bool] = None
    reported_status_match: Optional[bool] = None
    effective_status_match: Optional[bool] = None
    failure_stage_match: Optional[bool] = None
    degradation_match: Optional[bool] = None
    lane_status_consistent: Optional[bool] = None
    effective_status_consistent: Optional[bool] = None
    failure_stage_consistent: Optional[bool] = None
    degradation_consistent: Optional[bool] = None
    analyst_status_match: Optional[bool] = None
    tool_use_match: Optional[bool] = None
    compute_match: Optional[bool] = None
    citation_match: Optional[bool] = None
    grounding_consistent: Optional[bool] = None

    score: float = 0.0
    weighted_components: Dict[str, float] = Field(default_factory=dict)
    critical_failures: List[str] = Field(default_factory=list)


class AgentEvalRowV1(BaseModel):
    id: str
    query: str
    expected: Dict[str, Any] = Field(default_factory=dict)

    orchestrator_ok: bool = False
    orchestrator_error: Optional[str] = None

    reported_status: Optional[str] = None
    effective_status: Optional[EffectiveStatus] = None
    effective_status_source: Optional[StatusSource] = None
    derived_effective_status: Optional[EffectiveStatus] = None
    effective_status_consistent: Optional[bool] = None
    failure_stage: Optional[str] = None
    derived_failure_stage: Optional[str] = None
    failure_stage_consistent: Optional[bool] = None
    degradation: Dict[str, Any] = Field(default_factory=dict)
    degradation_consistent: Optional[bool] = None
    route: Optional[str] = None

    lane_status_source: LaneStatusSource = "evaluator_derived"
    runtime_lane_status: Optional[AgentLaneStatusSet] = None
    derived_lane_status: AgentLaneStatusSet
    lane_status: AgentLaneStatusSet
    lane_status_consistent: Optional[bool] = None

    planner_intent: Optional[str] = None
    planner_ticker: Optional[str] = None
    planner_fiscal_year: Optional[int] = None
    planner_form_type: Optional[str] = None

    structured_metric_ids: List[str] = Field(default_factory=list)
    structured_statuses: List[str] = Field(default_factory=list)

    analyst_status: Optional[str] = None
    analyst_metric: Optional[str] = None
    analyst_answer: str = ""
    analyst_used_financial_evaluator: Optional[bool] = None
    analyst_citation_count: int = 0

    timings_ms: Dict[str, int] = Field(default_factory=dict)
    semantic_contexts: List[str] = Field(default_factory=list)
    semantic_context_kinds: List[str] = Field(default_factory=list)
    structured_evidence: Dict[str, Any] = Field(default_factory=dict)
    grounding: Dict[str, Any] = Field(default_factory=dict)
    deterministic: AgentDeterministicChecksV1 = Field(
        default_factory=AgentDeterministicChecksV1
    )
    ragas: Dict[str, float] = Field(default_factory=dict)
    trace: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


def load_agent_eval_examples_v1(path: str | Path) -> List[AgentEvalExampleV1]:
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    examples: List[AgentEvalExampleV1] = []
    with eval_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                examples.append(AgentEvalExampleV1.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(
                    f"Invalid route-aware eval row at line {line_no}: {exc}"
                ) from exc
    return examples
