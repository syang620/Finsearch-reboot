"""Status and denominator contracts. No production validators are imported.

An accepted answer is a channel, not a semantic verdict. In particular an
insufficient_data status is an abstention *candidate*, not a correct abstention.
"""
from collections import Counter

from evals.semantic_answer_v1 import rate

ACCEPTED = {"ok", "insufficient_data"}
PRECEDENCE = ("clarification_stop", "planner_failure", "analyst_timeout",
              "grounding_fail_closed", "retrieval_failure", "tool_failure",
              "unknown_execution_failure")


def execution(output):
    analyst = output.get("analyst") or {}
    issues = list(output.get("open_issues") or []) + list(analyst.get("open_issues") or [])
    codes = {i.get("code", "") for i in issues if isinstance(i, dict)}
    stage = output.get("failure_stage")
    signals = set()
    if output.get("status") == "interrupted" or stage == "interrupted":
        signals.add("clarification_stop")
    if stage == "planner" or any(c.startswith("PLANNER_") and ("ERROR" in c or "INVALID" in c) for c in codes):
        signals.add("planner_failure")
    # Only explicit runtime diagnostics, never the answer's prose, classify timeouts.
    if "ANALYST_MODEL_TIMEOUT" in codes or "ANALYST_MODEL_TIMEOUT" in str(analyst.get("error") or ""):
        signals.add("analyst_timeout")
    if analyst.get("status") == "grounding_error": signals.add("grounding_fail_closed")
    if stage == "retrieval": signals.add("retrieval_failure")
    if stage == "structured_fact" or analyst.get("status") == "tool_error": signals.add("tool_failure")
    accepted = (output.get("ok") is True and output.get("status") in {"completed", "degraded"}
                and stage in {None, "none"} and analyst.get("ok") is True
                and analyst.get("status") in ACCEPTED and not analyst.get("error") and not output.get("error")
                and isinstance(analyst.get("answer"), str) and bool(analyst["answer"].strip()))
    # Recovered lane/tool issues remain diagnostics; completed degraded answers
    # may be eligible. Terminal timeout/grounding/planner signals cannot be.
    if signals & {"clarification_stop", "planner_failure", "analyst_timeout", "grounding_fail_closed"}:
        accepted = False
    if accepted:
        primary = "substantive_answer" if analyst["status"] == "ok" else "abstention_candidate"
    else:
        signals.add("unknown_execution_failure")
        primary = next(s for s in PRECEDENCE if s in signals)
    diagnostics = [str(output.get("error") or ""), str(analyst.get("error") or "")]
    diagnostics += [str(i.get("message") or "") for i in issues if isinstance(i, dict)]
    service_error = any(any(s in d.lower() for s in ("connection refused", "connectionerror", "connecterror", "service unavailable", "http 503", "http 429")) for d in diagnostics)
    return {"primary": primary, "signals": sorted(signals), "eligible": accepted,
            "analyst_status": analyst.get("status"), "service_error_signal": service_error,
            "issue_codes": sorted(codes)}


def accepted_answer(output):
    """Never expose retained rejected claims/error prose to semantic scorers."""
    if not execution(output)["eligible"]: return None
    analyst = output["analyst"]
    claims = analyst.get("claims") or []
    if not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims):
        raise ValueError("Invalid emitted claims")
    ids = [c.get("claim_id") for c in claims]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate/empty emitted claim IDs")
    for c in claims:
        if not isinstance(c.get("text"), str) or not c["text"].strip(): raise ValueError("Empty emitted claim text")
        refs = c.get("context_ids", [])
        if not isinstance(refs, list) or any(not isinstance(i, str) or not i for i in refs): raise ValueError("Invalid citation IDs")
    return {k: analyst.get(k) for k in ("status", "answer", "claims", "compare_rows", "computation")}


def summarize_outcomes(rows):
    counts = Counter(r["execution"]["primary"] for r in rows)
    return {"cases": len(rows), "outcomes": dict(sorted(counts.items())),
            "eligible_produced_answers": rate(sum(r["execution"]["eligible"] for r in rows), len(rows)),
            "substantive_answer_rate": rate(counts["substantive_answer"], len(rows)),
            "abstention_candidate_rate": rate(counts["abstention_candidate"], len(rows)),
            "service_error_signal_rate": rate(sum(r["execution"]["service_error_signal"] for r in rows), len(rows)),
            "note": "Correct abstention requires separate source-based semantic assessment; status alone earns no correctness credit."}


def summarize_numeric(rows):
    """Fixed gold requirements, not variable emitted-claim counts.

    Ineligible requirements are unassessed execution losses, not wrong claims.
    Eligible missing/unknown requirements stay in the conditional denominator.
    """
    all_checks = [c for r in rows for c in r["numeric_checks"]]
    eligible = [c for r in rows if r["execution"]["eligible"] for c in r["numeric_checks"]]
    counts = Counter(c["truth"] for c in eligible)
    correct = counts["correct"]
    resolved = counts["correct"] + counts["incorrect"]
    return {"gold_numeric_requirements": len(all_checks), "eligible_requirement_outcomes": dict(sorted(counts.items())),
            "unassessed_due_to_execution": len(all_checks) - len(eligible),
            "verified_numeric_credit_over_all_gold": rate(correct, len(all_checks)),
            "verified_numeric_credit_given_eligible_answer": rate(correct, len(eligible)),
            "numeric_correctness_resolved_only": rate(correct, resolved),
            "deterministic_resolution_coverage_given_eligible": rate(resolved, len(eligible)),
            "note": "Verified-credit rates are not general semantic accuracy. Unknown and missing are not asserted wrong; resolved-only correctness requires its coverage."}
