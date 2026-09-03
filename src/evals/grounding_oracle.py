"""Independent PR6 oracle. Never import the analyst's grounding policy here.

This checks structural bindings, not natural-language entailment. In particular,
legacy flat citations cannot be upgraded into observed claim bindings.
"""
from __future__ import annotations

from typing import Any


SAFE_INSUFFICIENT = "The available evidence is insufficient to answer this question."
SAFE_FAILURE = "The analyst could not produce an answer with valid evidence bindings."


def inspect_claims(packet: dict, candidate: dict, limit: int = 5) -> dict:
    visible = packet.get("context_items", [])[:limit]
    ids = [item.get("context_id") for item in visible]
    contexts = {item.get("context_id"): item for item in visible}
    errors, rejected, accepted, cleaned = [], [], [], []
    filing = packet.get("intent") in {"filing_fact", "filing_calc"}
    claims = candidate.get("claims") or []
    if len(ids) != len(set(ids)):
        errors.append("DUPLICATE_VISIBLE_CONTEXT_ID")
    if filing and candidate.get("status") == "ok" and not claims:
        errors.append("GROUNDING_CLAIMS_MISSING")
    seen = set()
    for claim in claims:
        cid = claim.get("claim_id")
        kind = claim.get("claim_type")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            errors.append("GROUNDING_CLAIM_ID_INVALID")
        seen.add(cid)
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            errors.append("GROUNDING_CLAIM_TEXT_MISSING")
        refs = []
        for ref in claim.get("context_ids", []):
            ref = str(ref).strip()
            if ref not in contexts:
                rejected.append(ref)
            elif ref not in refs:
                refs.append(ref)
        if not refs:
            errors.append("GROUNDING_CLAIM_UNSUPPORTED")
        evidence = [contexts[ref] for ref in refs]
        if kind == "structured_numeric":
            metric = claim.get("metric_id")
            if not metric or not any(
                item.get("kind") == "structured_fact"
                and (item.get("structured_fact") or {}).get("metric_id") == metric
                for item in evidence
            ):
                errors.append("GROUNDING_METRIC_MISMATCH")
        elif kind in {"kb_numeric", "narrative", "attribution"}:
            if not any(item.get("kind") in {"text", "table"} for item in evidence):
                errors.append("GROUNDING_EVIDENCE_TYPE_MISMATCH")
        elif kind != "calculation":
            errors.append("GROUNDING_CLAIM_TYPE_INVALID")
        for ref in refs:
            if ref not in accepted:
                accepted.append(ref)
        cleaned.append({**claim, "context_ids": refs})
    by_id = {claim.get("claim_id"): claim for claim in cleaned}
    rows = []
    for row in candidate.get("compare_rows", []):
        claim = by_id.get(row.get("claim_id"))
        if claim is None:
            errors.append("GROUNDING_ROW_UNBOUND")
            continue
        if any(str(row.get(key) or "").strip() not in claim.get("text", "") for key in ("label", "value")):
            errors.append("GROUNDING_ROW_TEXT_MISMATCH")
        target = row.get("target_id")
        if target and not any(contexts[ref].get("target_id") == target for ref in claim["context_ids"]):
            errors.append("GROUNDING_ROW_TARGET_MISMATCH")
        rows.append({**row, "context_ids": claim["context_ids"]})
    # Redundant legacy references cannot supply support but must not disappear
    # without a diagnostic when they contain unknown/non-visible IDs.
    for ref in candidate.get("used_context_ids", []):
        if str(ref).strip() not in contexts:
            rejected.append(str(ref).strip())
    for row in candidate.get("compare_rows", []):
        for ref in row.get("context_ids", []):
            if str(ref).strip() not in contexts:
                rejected.append(str(ref).strip())
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "rejected_context_ids": list(dict.fromkeys(rejected)),
        "accepted_context_ids": accepted,
        "claims": cleaned,
        "compare_rows": rows,
    }


def inspect_final(packet: dict, result: dict, limit: int = 5) -> dict:
    """Check actual finalized claims/citations independently of runtime verdicts."""
    report = inspect_claims(packet, result, limit)
    contexts = {item["context_id"]: item for item in packet.get("context_items", [])[:limit]}
    citations = result.get("citations", [])
    cited = [citation.get("context_id") for citation in citations]
    successful = result.get("ok") is True and result.get("status") in {"ok", "insufficient_data"}
    if successful and packet.get("intent") in {"filing_fact", "filing_calc"}:
        expected_answer = "\n\n".join(claim["text"] for claim in report["claims"])
        if result.get("status") == "insufficient_data":
            expected_answer = "\n\n".join(filter(None, [expected_answer, SAFE_INSUFFICIENT]))
        if result.get("answer") != expected_answer:
            report["errors"].append("GROUNDING_UNBOUND_ANSWER")
        if cited != report["accepted_context_ids"] or result.get("used_context_ids", []) != cited:
            report["errors"].append("GROUNDING_FINAL_CITATIONS_MISMATCH")
        if result.get("compare_rows", []) != report["compare_rows"]:
            report["errors"].append("GROUNDING_FINAL_ROWS_MISMATCH")
        for citation in citations:
            context = contexts.get(citation.get("context_id"))
            if context is None or citation.get("source") != context.get("source"):
                report["errors"].append("GROUNDING_PROVENANCE_MISMATCH")
    report["valid"] = not report["errors"]
    report["successful"] = successful
    return report
