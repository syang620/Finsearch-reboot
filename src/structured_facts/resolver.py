"""Pure extraction of merged PR24 metric and target resolution.

Target metadata is returned unchanged; this module does not interpret filing forms,
authorize requests, retrieve evidence or execute tools.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from agents.contracts import FilingMetadata, PlannerTarget, StructuredFactRequest
from agents.text_utils import normalize_text as _normalize_text
from mcp_server.tools.sec_metric_registry import METRIC_REGISTRY
from structured_facts.models import StructuredFactResolution


def _normalize_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


_TICKER_LIKE_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
# Keep aliases conservative. Broad single-word finance terms are intentionally
# avoided here to reduce false positives in metric auto-resolution.
_STRUCTURED_FACT_ALIAS_MAP: Dict[str, tuple[str, ...]] = {
    "total_debt": (
        "total debt",
        "interest bearing debt",
        "borrowings",
        "debt balance",
    ),
    "revenue": ("revenue", "sales", "total revenue"),
    "gross_profit": ("gross profit", "gross earnings"),
    "operating_income": ("operating income", "operating profit", "ebit"),
    "net_income": ("net income", "net earnings"),
    "cash_and_cash_equivalents": (
        "cash and cash equivalents",
        "cash equivalents",
        "cash",
    ),
    "total_assets": ("total assets",),
    "total_liabilities": ("total liabilities",),
    "stockholders_equity": (
        "stockholders equity",
        "shareholders equity",
    ),
    "operating_cash_flow": (
        "operating cash flow",
        "cash flow from operations",
        "cash from operations",
        "cfo",
        "cash",
    ),
    "capex": ("capex", "capital expenditures", "capital expenditure"),
}


def _normalize_metric_lookup_text(value: Any) -> str:
    text = _normalize_text(value) or ""
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9.\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _metric_registry_lookup_terms(metric_id: str, label: str) -> tuple[str, ...]:
    terms = [_normalize_metric_lookup_text(metric_id), _normalize_metric_lookup_text(label)]
    for alias in _STRUCTURED_FACT_ALIAS_MAP.get(metric_id, ()):
        normalized = _normalize_metric_lookup_text(alias)
        if normalized:
            terms.append(normalized)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term or term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return tuple(deduped)


def resolve_metric_id_for_structured_fact_request(
    *,
    metric_hint: Any,
    subquestion: Any,
) -> tuple[Optional[str], str, Optional[str]]:
    query = _normalize_metric_lookup_text(metric_hint) or _normalize_metric_lookup_text(subquestion)
    if not query:
        return None, "unresolved", "No metric hint or subquestion text was available for resolution."

    exact_matches: list[str] = []
    for metric_id, definition in METRIC_REGISTRY.items():
        terms = _metric_registry_lookup_terms(metric_id, definition.label)
        if query in terms:
            exact_matches.append(metric_id)

    if len(exact_matches) == 1:
        resolved = exact_matches[0]
        return resolved, "resolved", f"Resolved metric from exact registry match: {resolved}."
    if len(exact_matches) > 1:
        matches = sorted(set(exact_matches))
        return None, "ambiguous", f"Metric text matched multiple registered metrics exactly: {', '.join(matches)}."

    contained_matches: list[str] = []
    query_padded = f" {query} "
    for metric_id, definition in METRIC_REGISTRY.items():
        terms = _metric_registry_lookup_terms(metric_id, definition.label)
        if any(f" {term} " in query_padded for term in terms if term):
            contained_matches.append(metric_id)

    unique_matches = sorted(set(contained_matches))
    if len(unique_matches) == 1:
        resolved = unique_matches[0]
        return resolved, "resolved", f"Resolved metric from registry phrase match: {resolved}."
    if len(unique_matches) > 1:
        return None, "ambiguous", (
            "Metric text matched multiple registered metrics: "
            f"{', '.join(unique_matches)}."
        )
    return None, "unresolved", f"Metric text did not match any registered SEC metric: {query}."


def _entity_hint_looks_like_ticker(entity_hint: Any) -> Optional[str]:
    text = _normalize_text(entity_hint)
    if not text:
        return None
    stripped = str(text).strip()
    return stripped if _TICKER_LIKE_RE.fullmatch(stripped) else None


def _select_matching_target_for_structured_fact(
    *,
    targets: Sequence[PlannerTarget | Dict[str, Any]],
    entity_hint: Any,
    fiscal_year: Optional[int],
) -> Optional[Dict[str, Any]]:
    targets = [
        target.model_dump(mode="json") if isinstance(target, PlannerTarget) else dict(target)
        for target in (targets or [])
        if isinstance(target, (dict, PlannerTarget))
    ]
    if not targets:
        return None

    hinted_ticker = _entity_hint_looks_like_ticker(entity_hint)
    normalized_entity = _normalize_metric_lookup_text(entity_hint)
    requested_year = _normalize_int(fiscal_year)

    def _matches(target: Dict[str, Any]) -> bool:
        target_ticker = _normalize_text(target.get("ticker"))
        target_company = _normalize_metric_lookup_text(target.get("company_name"))
        target_year = _normalize_int(target.get("fiscal_year"))
        if requested_year is not None and target_year is not None and requested_year != target_year:
            return False
        if hinted_ticker and target_ticker == hinted_ticker:
            return True
        if normalized_entity and normalized_entity == target_company:
            return True
        return False

    matched = [target for target in targets if _matches(target)]
    if matched:
        return matched[0]
    if len(targets) == 1:
        return targets[0]
    return None


def resolve_structured_fact_inputs(
    *,
    request: StructuredFactRequest | Dict[str, Any],
    targets: Sequence[PlannerTarget | Dict[str, Any]],
    metadata: FilingMetadata,
) -> tuple[Optional[str], Optional[int], Optional[Dict[str, Any]]]:
    # Keep defensive dictionary inputs as-is; validating them would change PR24 behavior.
    if isinstance(request, StructuredFactRequest):
        request = request.model_dump(mode="json")
    matched_target = _select_matching_target_for_structured_fact(
        targets=targets,
        entity_hint=request.get("entity_hint"),
        fiscal_year=_normalize_int(request.get("fiscal_year")),
    )

    resolved_ticker = (
        _entity_hint_looks_like_ticker(request.get("entity_hint"))
        or metadata.ticker
        or _normalize_text((matched_target or {}).get("ticker"))
    )
    resolved_year = (
        _normalize_int(request.get("fiscal_year"))
        or metadata.fiscal_year
        or _normalize_int((matched_target or {}).get("fiscal_year"))
    )
    return resolved_ticker, resolved_year, matched_target


def resolve_structured_fact_request(
    request: StructuredFactRequest | Dict[str, Any],
    targets: Sequence[PlannerTarget | Dict[str, Any]],
    metadata: FilingMetadata,
) -> StructuredFactResolution:
    """Resolve identity only. A resolved request still needs capability permission."""
    if isinstance(request, StructuredFactRequest):
        request = request.model_dump(mode="json")
    ticker, fiscal_year, selected_target = resolve_structured_fact_inputs(
        request=request, targets=targets, metadata=metadata,
    )
    metric_id, status, reason = resolve_metric_id_for_structured_fact_request(
        metric_hint=request.get("metric_hint"), subquestion=request.get("subquestion"),
    )
    if not ticker or fiscal_year is None:
        missing_bits = []
        if not ticker:
            missing_bits.append("ticker")
        if fiscal_year is None:
            missing_bits.append("fiscal_year")
        metric_id = metric_id if status == "resolved" else None
        status = "missing_inputs"
        reason = "Missing structured-fact execution inputs: " + ", ".join(missing_bits) + "."
    return StructuredFactResolution(
        status=status, metric_id=metric_id, ticker=ticker, fiscal_year=fiscal_year,
        selected_target=selected_target, reason=reason,
    )
