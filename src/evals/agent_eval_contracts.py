from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


PlannerIntentStr = Literal["filing_fact", "filing_calc", "definition", "other"]


class AgentExpected(BaseModel):
    intent: Optional[PlannerIntentStr] = None
    retrieval_needed: Optional[bool] = None

    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[str] = None

    must_use_financial_evaluator: Optional[bool] = None
    expected_metric: Optional[str] = None
    expected_min_citations: int = Field(default=1, ge=0)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip().upper()
        return value or None

    @field_validator("fiscal_year")
    @classmethod
    def _validate_fiscal_year(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        year = int(value)
        if not (1900 <= year <= 2100):
            raise ValueError("fiscal_year out of range (1900-2100)")
        return year

    @field_validator("form_type")
    @classmethod
    def _normalize_form_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip().upper()
        return value or None

    @field_validator("expected_metric")
    @classmethod
    def _normalize_metric(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip().lower()
        return value or None


class AgentEvalExample(BaseModel):
    id: str
    user_query: str
    gold_answer: str
    expected: AgentExpected = Field(default_factory=AgentExpected)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("id must be non-empty")
        return value

    @field_validator("user_query")
    @classmethod
    def _query_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("user_query must be non-empty")
        return value

    @field_validator("gold_answer")
    @classmethod
    def _gold_answer_non_empty(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("gold_answer must be non-empty for agent RAGAS evaluation")
        return value

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: List[str]) -> List[str]:
        out: List[str] = []
        for tag in value or []:
            t = str(tag).strip().lower()
            if t:
                out.append(t)
        return out


class AgentDeterministicChecks(BaseModel):
    contract_valid: bool = False
    intent_match: Optional[bool] = None
    retrieval_needed_match: Optional[bool] = None
    metadata_match: Optional[bool] = None
    tool_use_match: Optional[bool] = None
    citation_match: Optional[bool] = None
    compute_match: Optional[bool] = None

    score: float = 0.0
    weighted_components: Dict[str, float] = Field(default_factory=dict)

    critical_failures: List[str] = Field(default_factory=list)


class AgentEvalRow(BaseModel):
    id: str
    query: str

    orchestrator_ok: bool = False
    orchestrator_error: Optional[str] = None

    planner_intent: Optional[str] = None
    planner_retrieval_needed: Optional[bool] = None
    planner_ticker: Optional[str] = None
    planner_fiscal_year: Optional[int] = None
    planner_form_type: Optional[str] = None

    analyst_metric: Optional[str] = None
    analyst_answer: str = ""
    analyst_used_financial_evaluator: Optional[bool] = None
    analyst_citation_count: int = 0

    deterministic: AgentDeterministicChecks = Field(default_factory=AgentDeterministicChecks)
    ragas: Dict[str, float] = Field(default_factory=dict)

    trace: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class AgentGateStatus(BaseModel):
    deterministic_gate_pass: bool
    ragas_gate_pass: bool
    overall_pass: bool


class AgentEvalSummary(BaseModel):
    num_queries: int
    num_valid_queries: int
    num_failures: int

    deterministic: Dict[str, float] = Field(default_factory=dict)
    ragas: Dict[str, float] = Field(default_factory=dict)

    gate: AgentGateStatus
    config: Dict[str, Any] = Field(default_factory=dict)


def load_agent_eval_examples(path: str | Path) -> List[AgentEvalExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")

    examples: List[AgentEvalExample] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(AgentEvalExample.model_validate(obj))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid eval row at line {line_no}: {exc}") from exc
    return examples
