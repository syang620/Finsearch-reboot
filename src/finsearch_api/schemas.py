"""Request and response contracts for the private FinSearch API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.contracts import DegradationSummary, EvidenceLaneStatusSet


class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRequest(StrictApiModel):
    query: str = Field(min_length=1, max_length=10_000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must be non-empty")
        return query


class ResumeRequest(StrictApiModel):
    answers: List[str] = Field(min_length=1, max_length=20)

    @field_validator("answers")
    @classmethod
    def normalize_answers(cls, values: List[str]) -> List[str]:
        answers = [str(value).strip() for value in values]
        if any(not answer for answer in answers):
            raise ValueError("answers must contain only non-empty strings")
        if any(len(answer) > 10_000 for answer in answers):
            raise ValueError("each answer must contain at most 10000 characters")
        return answers


class RunResponse(StrictApiModel):
    request_id: str
    run_id: str
    route: Optional[Literal["kb", "structured_fact", "hybrid"]]
    status: Literal["completed", "degraded", "interrupted", "failed"]
    ok: bool
    failure_stage: str
    lanes: EvidenceLaneStatusSet
    degradation: DegradationSummary
    open_issues: List[Dict[str, Any]]
    planner: Optional[Dict[str, Any]]
    planner_turn: Optional[Dict[str, Any]]
    retrieval: Dict[str, Any]
    structured_fact_results: List[Any]
    analyst: Optional[Dict[str, Any]]
    interrupt: List[Dict[str, Any]]
    orchestrator_trace: Dict[str, Any]


class ErrorDetail(StrictApiModel):
    code: str
    message: str


class ErrorResponse(StrictApiModel):
    request_id: str
    error: ErrorDetail


class HealthResponse(StrictApiModel):
    status: Literal["ok", "ready"]
