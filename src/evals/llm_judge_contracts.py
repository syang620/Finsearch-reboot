from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


JudgeMode = Literal["answer_only", "evidence_based"]
JudgeVerdict = Literal["correct", "partially_correct", "incorrect"]


class LLMJudgeDimensionScores(BaseModel):
    correctness: int = Field(..., ge=0, le=4)
    completeness: int = Field(..., ge=0, le=3)
    grounding: Optional[int] = Field(default=None, ge=0, le=2)
    inference_handling: int = Field(..., ge=0, le=1)


class LLMJudgeEvidenceChunkIds(BaseModel):
    text: List[int] = Field(default_factory=list)
    tables: List[int] = Field(default_factory=list)

    @field_validator("text", "tables", mode="before")
    @classmethod
    def _normalize_chunk_ids(cls, value: Any) -> List[int]:
        if value is None:
            return []
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out


class LLMJudgeModelOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdict: Optional[JudgeVerdict] = None
    score: Optional[int] = None
    dimension_scores: LLMJudgeDimensionScores
    matched_key_points: List[str] = Field(default_factory=list)
    missed_key_points: List[str] = Field(default_factory=list)
    unsupported_or_wrong_claims: List[str] = Field(default_factory=list)
    used_evidence_chunk_ids: LLMJudgeEvidenceChunkIds = Field(default_factory=LLMJudgeEvidenceChunkIds)
    explanation: str = ""


class LLMJudgeEvalRow(BaseModel):
    id: str
    prompt_index: int
    prompt: str
    judge_mode: JudgeMode
    judge_model: str

    retrieval_ok: Optional[bool] = None
    analyst_ok: Optional[bool] = None

    gold_answer: str
    candidate_answer: str
    evidence_provided: bool = False

    score: int = 0
    score_max: int = 10
    verdict: JudgeVerdict = "incorrect"
    dimension_scores: LLMJudgeDimensionScores

    matched_key_points: List[str] = Field(default_factory=list)
    missed_key_points: List[str] = Field(default_factory=list)
    unsupported_or_wrong_claims: List[str] = Field(default_factory=list)
    used_evidence_chunk_ids: LLMJudgeEvidenceChunkIds = Field(default_factory=LLMJudgeEvidenceChunkIds)

    evidence_chunk_ids: LLMJudgeEvidenceChunkIds = Field(default_factory=LLMJudgeEvidenceChunkIds)
    retrieved_chunk_doc_ids: List[str] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class LLMJudgeEvalSummary(BaseModel):
    num_queries: int
    num_judged: int
    num_failures: int
    judge_mode: JudgeMode
    judge_model: str

    verdict_counts: Dict[str, int] = Field(default_factory=dict)
    mean_score: float = 0.0
    mean_score_ratio: float = 0.0
    dimension_means: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
