"""Deterministic claim binding; deliberately not an entailment checker."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.contracts import AnalystPacket, ContextItemKind, PlannerIntent


INSUFFICIENT_ANSWER = "The available evidence is insufficient to answer this question."
GROUNDING_FAILURE_ANSWER = "The analyst could not produce an answer with valid evidence bindings."


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: Literal["structured_numeric", "kb_numeric", "calculation", "narrative", "attribution"]
    text: str
    context_ids: list[str]
    metric_id: str | None = None

    @field_validator("claim_id", "text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim identity and text must be non-empty")
        return value

    @field_validator("metric_id")
    @classmethod
    def normalize_metric(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class GroundingDecision(BaseModel):
    valid: bool
    required: bool
    claims: list[GroundedClaim] = Field(default_factory=list)
    context_ids: list[str] = Field(default_factory=list)
    invalid_context_ids: list[str] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)
    compare_rows: list[dict] = Field(default_factory=list)
    answer: str = ""


def validate_grounding(packet: AnalystPacket, candidate: dict, *, limit: int) -> GroundingDecision:
    """Sanitize only redundant bad references, never drop an unsupported claim."""
    required = packet.intent in {PlannerIntent.FILING_FACT, PlannerIntent.FILING_CALC}
    visible = packet.context_items[:max(0, limit)]
    contexts = {item.context_id: item for item in visible}
    usable_ids = {
        item.context_id for item in visible
        if item.structured_fact is not None or (
            item.kind in {ContextItemKind.TEXT, ContextItemKind.TABLE}
            and str(item.payload.get("table_markdown") or item.payload.get("content") or item.payload.get("text") or "").strip()
        )
    }
    codes: list[str] = []
    invalid: list[str] = []
    accepted: list[str] = []
    claims: list[GroundedClaim] = []
    identities: set[str] = set()
    raw_claims = candidate.get("claims") or []
    if len(contexts) != len(visible):
        codes.append("DUPLICATE_VISIBLE_CONTEXT_ID")
    if required and candidate.get("status") == "ok" and not raw_claims:
        codes.append("GROUNDING_CLAIMS_MISSING")
    for raw in raw_claims:
        claim = GroundedClaim.model_validate(raw)
        if claim.claim_id in identities:
            codes.append("GROUNDING_CLAIM_ID_INVALID")
        identities.add(claim.claim_id)
        refs: list[str] = []
        for raw_ref in claim.context_ids:
            ref = raw_ref.strip()
            if ref not in contexts:
                invalid.append(ref)
            elif ref not in usable_ids:
                invalid.append(ref)
                codes.append("GROUNDING_CONTEXT_UNUSABLE")
            elif ref not in refs:
                refs.append(ref)
        if not refs:
            codes.append("GROUNDING_CLAIM_UNSUPPORTED")
        evidence = [contexts[ref] for ref in refs]
        if claim.claim_type == "structured_numeric":
            if not claim.metric_id or not any(
                item.kind == ContextItemKind.STRUCTURED_FACT
                and item.structured_fact is not None
                and item.structured_fact.metric_id == claim.metric_id
                for item in evidence
            ):
                codes.append("GROUNDING_METRIC_MISMATCH")
        elif claim.claim_type in {"kb_numeric", "narrative", "attribution"}:
            if not any(item.kind in {ContextItemKind.TEXT, ContextItemKind.TABLE} for item in evidence):
                codes.append("GROUNDING_EVIDENCE_TYPE_MISMATCH")
        claims.append(claim.model_copy(update={"context_ids": refs}))
        accepted.extend(ref for ref in refs if ref not in accepted)

    by_id = {claim.claim_id: claim for claim in claims}
    rows: list[dict] = []
    for row in candidate.get("compare_rows", []):
        claim = by_id.get(row.get("claim_id"))
        if claim is None:
            codes.append("GROUNDING_ROW_UNBOUND")
            continue
        # Row strings must already occur in the bound prose. This is a rendering
        # constraint, not a claim that substring matching proves entailment.
        if any(str(row.get(key) or "").strip() not in claim.text for key in ("label", "value")):
            codes.append("GROUNDING_ROW_TEXT_MISMATCH")
        target = row.get("target_id")
        if target and not any(contexts[ref].target_id == target for ref in claim.context_ids):
            codes.append("GROUNDING_ROW_TARGET_MISMATCH")
        rows.append({**row, "context_ids": claim.context_ids})
    legacy_refs = list(candidate.get("used_context_ids") or [])
    for row in candidate.get("compare_rows", []):
        legacy_refs.extend(row.get("context_ids") or [])
    invalid.extend(ref.strip() for ref in legacy_refs if ref.strip() not in contexts)
    if required:
        answer = "\n\n".join(claim.text for claim in claims)
        if candidate.get("status") == "insufficient_data":
            answer = "\n\n".join(filter(None, [answer, INSUFFICIENT_ANSWER]))
    else:
        answer = candidate.get("answer", "")
    return GroundingDecision(
        valid=not codes, required=required, claims=claims, context_ids=accepted,
        invalid_context_ids=list(dict.fromkeys(invalid)), issue_codes=list(dict.fromkeys(codes)),
        compare_rows=rows, answer=answer,
    )
