from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import math
import re
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError, field_validator

# Allow running this file directly without installing the package.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mcp_server.tools.sec_metric_client import SecCompanyFactsClient  # noqa: E402
from mcp_server.tools.sec_metric_registry import (  # noqa: E402
    MetricComponentGroup,
    MetricDefinition,
    MetricFactCandidate,
    get_metric_definition,
)

MetricStatus = Literal[
    "ok",
    "partial",
    "not_found",
    "unsupported_metric",
    "ambiguous",
]

DerivedMetricStrategy = Callable[..., "SecGetMetricResult"]


class SecGetMetricRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. AAPL.")
    fiscal_year: int = Field(..., description="Fiscal year to anchor the annual filing.")
    metric_id: str = Field(..., description="Registry-backed structured metric identifier.")

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, value: str) -> str:
        ticker = str(value).strip().upper()
        if not ticker:
            raise ValueError("ticker must be non-empty")
        return ticker

    @field_validator("fiscal_year")
    @classmethod
    def _validate_fiscal_year(cls, value: int) -> int:
        fiscal_year = int(value)
        if not (1900 <= fiscal_year <= 2100):
            raise ValueError("fiscal_year out of reasonable range (1900-2100)")
        return fiscal_year

    @field_validator("metric_id")
    @classmethod
    def _normalize_metric_id(cls, value: str) -> str:
        return str(value).strip()


class MetricFactResult(BaseModel):
    taxonomy: str
    concept_name: str
    unit: str
    value: float
    accession_number: Optional[str] = None
    report_date: Optional[str] = None
    filed_date: Optional[str] = None
    form_type: Optional[str] = None
    fp: Optional[str] = None
    source_url: Optional[str] = None
    matched_by_accession: bool = False
    matched_by_report_date: bool = False
    start_date: Optional[str] = None


class MetricComponentResult(MetricFactResult):
    group_id: str
    group_label: str


class SecGetMetricResult(BaseModel):
    ok: bool
    status: MetricStatus
    metric_id: str
    value: Optional[float] = None
    unit: Optional[str] = None
    ticker: str
    cik: Optional[str] = None
    fiscal_year: int
    form_type: Optional[str] = None
    accession_number: Optional[str] = None
    report_date: Optional[str] = None
    filed_date: Optional[str] = None
    source_url: Optional[str] = None
    primary_fact: Optional[MetricFactResult] = None
    components: List[MetricComponentResult] = Field(default_factory=list)
    missing_component_groups: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    trace: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class FilingAnchor:
    cik: str
    ticker: str
    fiscal_year: int
    accession_number: str
    report_date: str
    filed_date: str
    form_type: str
    source_url: Optional[str]


@dataclass(frozen=True)
class FactCandidate:
    taxonomy: str
    concept_name: str
    unit: str
    value: float
    accession_number: str
    report_date: Optional[str]
    filed_date: Optional[str]
    form_type: Optional[str]
    fiscal_year: Optional[int]
    fp: Optional[str]
    frame: Optional[str]
    source_url: Optional[str]
    start_date: Optional[str] = None


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return None
    try:
        return date.fromisoformat(text)
    except Exception:
        return None


def _build_filing_source_url(*, cik: str, accession_number: str, primary_document: Optional[str]) -> Optional[str]:
    document_name = str(primary_document or "").strip()
    if not document_name:
        return None
    cik_no_leading_zeros = str(int(str(cik).zfill(10)))
    accession_no_dashes = accession_number.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_no_dashes}/{document_name}"
    )


def _filing_rows(submissions: Dict[str, Any]) -> List[Dict[str, str]]:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    names = {"accession_number": "accessionNumber", "report_date": "reportDate",
             "filed_date": "filingDate", "form_type": "form", "primary_document": "primaryDocument"}
    columns = {key: value if isinstance(value := recent.get(name), list) else []
               for key, name in names.items()}
    return [{key: str(values[i] or "").strip() if i < len(values) else ""
             for key, values in columns.items()} for i in range(len(columns["form_type"]))]


def _filing_anchor_decision(*, submissions: Dict[str, Any], ticker: str, cik: str,
                            fiscal_year: int) -> tuple[FilingAnchor | None, str]:
    rows = _filing_rows(submissions)
    annual = [row for row in rows if row["form_type"].upper() == "10-K"
              and row["report_date"].startswith(f"{fiscal_year}-")]
    if any(not row["accession_number"] or not _parse_iso_date(row["report_date"])
           or not _parse_iso_date(row["filed_date"]) for row in annual):
        return None, "INVALID_FILING_METADATA"
    unique = {json.dumps(row, sort_keys=True): row for row in annual}
    if len(unique) > 1:
        return None, "FISCAL_YEAR_UNRESOLVED"
    if not unique:
        amended = any(row["form_type"].upper() == "10-K/A"
                      and row["report_date"].startswith(f"{fiscal_year}-")
                      and _parse_iso_date(row["report_date"]) for row in rows)
        return None, "AMENDMENT_ONLY" if amended else "NO_ORIGINAL_FILING"
    selected = next(iter(unique.values()))
    return FilingAnchor(
        cik=str(cik).zfill(10),
        ticker=str(ticker).strip().upper(),
        fiscal_year=int(fiscal_year),
        accession_number=selected["accession_number"],
        report_date=selected["report_date"],
        filed_date=selected["filed_date"],
        form_type="10-K",
        source_url=_build_filing_source_url(
            cik=str(cik).zfill(10),
            accession_number=selected["accession_number"],
            primary_document=selected["primary_document"],
        ),
    ), "ORIGINAL_FILING_SELECTED"


def _extract_filing_anchor(*, submissions: Dict[str, Any], ticker: str, cik: str, fiscal_year: int) -> FilingAnchor | None:
    return _filing_anchor_decision(submissions=submissions, ticker=ticker, cik=cik,
                                   fiscal_year=fiscal_year)[0]


def _iter_companyfacts_candidates(
    *,
    companyfacts: Dict[str, Any],
    candidate: MetricFactCandidate,
    cik: str,
    fiscal_year: int,
    expected_unit: Optional[str],
) -> List[FactCandidate]:
    facts = (((companyfacts or {}).get("facts") or {}).get(candidate.taxonomy) or {}).get(candidate.concept_name) or {}
    units = facts.get("units") or {}
    source_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json"
    results: List[FactCandidate] = []
    expected_unit_normalized = str(expected_unit or "").strip().upper() or None
    for unit_name, items in units.items():
        unit_text = str(unit_name).strip()
        if expected_unit_normalized and unit_text.upper() != expected_unit_normalized:
            continue
        for item in items or []:
            if not isinstance(item, dict):
                continue
            form_type = str(item.get("form") or "").strip().upper()
            try:
                numeric_value = float(item.get("val"))
            except Exception:
                continue
            if isinstance(item.get("val"), bool) or not math.isfinite(numeric_value):
                continue
            accession_number = str(item.get("accn") or "").strip()
            report_date = str(item.get("end") or "").strip() or None
            filed_date = str(item.get("filed") or "").strip() or None
            fp = str(item.get("fp") or "").strip().upper() or None
            candidate_fiscal_year = item.get("fy")
            try:
                candidate_fiscal_year = int(candidate_fiscal_year) if candidate_fiscal_year is not None else None
            except Exception:
                candidate_fiscal_year = -1
            results.append(
                FactCandidate(
                    taxonomy=candidate.taxonomy,
                    concept_name=candidate.concept_name,
                    unit=unit_text,
                    value=numeric_value,
                    accession_number=accession_number,
                    report_date=report_date,
                    filed_date=filed_date,
                    form_type=form_type or None,
                    fiscal_year=candidate_fiscal_year,
                    fp=fp,
                    frame=str(item.get("frame") or "").strip() or None,
                    source_url=source_url,
                    start_date=str(item.get("start")).strip() if item.get("start") is not None else None,
                )
            )
    return results


ANNUAL_DURATION_DAYS = frozenset({364, 365, 366, 371})


def _fact_rejection(candidate: FactCandidate, anchor: FilingAnchor,
                    period_type: Literal["instant", "duration"]) -> str:
    if candidate.accession_number != anchor.accession_number:
        return "ACCESSION_MISMATCH"
    if candidate.form_type != anchor.form_type:
        return "FORM_MISMATCH"
    if candidate.report_date != anchor.report_date:
        return "END_DATE_MISMATCH"
    if candidate.fiscal_year not in {None, anchor.fiscal_year}:
        return "FISCAL_YEAR_UNRESOLVED"
    if period_type == "instant":
        return "PERIOD_TYPE_MISMATCH" if candidate.start_date is not None else "ELIGIBLE"
    start, end = _parse_iso_date(candidate.start_date), _parse_iso_date(candidate.report_date)
    if start is None or end is None:
        return "INVALID_DURATION_START"
    if start >= end or (end - start).days + 1 not in ANNUAL_DURATION_DAYS:
        return "NON_ANNUAL_DURATION"
    return "ELIGIBLE"


def _select_best_anchored_fact(
    *,
    candidates: Sequence[FactCandidate],
    anchor: FilingAnchor,
    period_type: Literal["instant", "duration"],
) -> tuple[FactCandidate | None, bool]:
    anchored = [
        candidate
        for candidate in candidates
        if _fact_rejection(candidate, anchor, period_type) == "ELIGIBLE"
    ]
    if not anchored:
        return None, False

    if len({(candidate.value, candidate.start_date) for candidate in anchored}) > 1:
        return None, True
    # Metadata only breaks ties after factual identity agrees; never input order.
    return min(anchored, key=lambda c: (c.filed_date or "", c.fp or "",
                                       str(c.fiscal_year), c.frame or "")), False


def _to_fact_result(candidate: FactCandidate) -> MetricFactResult:
    return MetricFactResult(
        taxonomy=candidate.taxonomy,
        concept_name=candidate.concept_name,
        unit=candidate.unit,
        value=candidate.value,
        accession_number=candidate.accession_number,
        report_date=candidate.report_date,
        filed_date=candidate.filed_date,
        form_type=candidate.form_type,
        fp=candidate.fp,
        source_url=candidate.source_url,
        start_date=candidate.start_date,
    )


def _to_component_result(*, candidate: FactCandidate, group: MetricComponentGroup, anchor: FilingAnchor) -> MetricComponentResult:
    return MetricComponentResult(
        group_id=group.group_id,
        group_label=group.label,
        taxonomy=candidate.taxonomy,
        concept_name=candidate.concept_name,
        unit=candidate.unit,
        value=candidate.value,
        accession_number=candidate.accession_number,
        report_date=candidate.report_date,
        filed_date=candidate.filed_date,
        form_type=candidate.form_type,
        fp=candidate.fp,
        source_url=candidate.source_url,
        start_date=candidate.start_date,
        matched_by_accession=candidate.accession_number == anchor.accession_number,
        matched_by_report_date=candidate.report_date == anchor.report_date,
    )


def compute_total_debt_carrying_amount(
    *,
    components: Sequence[MetricComponentResult],
    required_group_ids: Sequence[str],
) -> Dict[str, Any]:
    if not components:
        return {
            "status": "not_found",
            "value": None,
            "missing_component_groups": list(required_group_ids),
        }

    present_group_ids = {component.group_id for component in components}
    missing_group_ids = [group_id for group_id in required_group_ids if group_id not in present_group_ids]
    total_value = float(sum(component.value for component in components))

    if not missing_group_ids:
        status: MetricStatus = "ok"
    elif present_group_ids:
        status = "partial"
    else:
        status = "not_found"

    return {
        "status": status,
        "value": total_value if status == "ok" else None,
        "missing_component_groups": missing_group_ids,
    }


def _unsupported_metric_result(*, request: SecGetMetricRequest, error: Optional[str] = None) -> SecGetMetricResult:
    return SecGetMetricResult(
        ok=False,
        status="unsupported_metric",
        metric_id=request.metric_id,
        ticker=request.ticker,
        fiscal_year=request.fiscal_year,
        error=error or f"Metric '{request.metric_id}' is not registered.",
    )


def _base_result_kwargs(*, request: SecGetMetricRequest, anchor: FilingAnchor, cik: str) -> Dict[str, Any]:
    return {
        "metric_id": request.metric_id,
        "ticker": request.ticker,
        "cik": cik,
        "fiscal_year": request.fiscal_year,
        "form_type": anchor.form_type,
        "accession_number": anchor.accession_number,
        "report_date": anchor.report_date,
        "filed_date": anchor.filed_date,
        "source_url": anchor.source_url,
    }


def _execute_atomic_metric(
    *,
    request: SecGetMetricRequest,
    metric_definition: MetricDefinition,
    anchor: FilingAnchor,
    cik: str,
    companyfacts: Dict[str, Any],
) -> SecGetMetricResult:
    selection_trace: List[Dict[str, Any]] = []
    ambiguous_candidates: List[str] = []
    for candidate in metric_definition.atomic_candidates:
        candidates = _iter_companyfacts_candidates(
            companyfacts=companyfacts,
            candidate=candidate,
            cik=cik,
            fiscal_year=request.fiscal_year,
            expected_unit=metric_definition.unit,
        )
        chosen, ambiguous = _select_best_anchored_fact(
            candidates=candidates,
            anchor=anchor,
            period_type=metric_definition.period_type,
        )
        selection_trace.append(
            {
                "concept_name": candidate.concept_name,
                "taxonomy": candidate.taxonomy,
                "candidates_considered": len(candidates),
                "selected": chosen.concept_name if chosen is not None else None,
                "ambiguous": ambiguous,
            }
        )
        if ambiguous:
            ambiguous_candidates.append(candidate.concept_name)
            break
        if chosen is None:
            continue

        primary_fact = _to_fact_result(chosen)
        primary_fact.matched_by_accession = chosen.accession_number == anchor.accession_number
        primary_fact.matched_by_report_date = chosen.report_date == anchor.report_date
        return SecGetMetricResult(
            ok=True,
            status="ok",
            value=chosen.value,
            unit=metric_definition.unit or chosen.unit,
            primary_fact=primary_fact,
            trace={
                "anchor": {
                    "accession_number": anchor.accession_number,
                    "report_date": anchor.report_date,
                    "filed_date": anchor.filed_date,
                    "form_type": anchor.form_type,
                },
                "selection": selection_trace,
            },
            **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
        )

    if ambiguous_candidates:
        return SecGetMetricResult(
            ok=False,
            status="ambiguous",
            error="Ambiguous anchored fact candidates prevented metric resolution.",
            trace={
                "anchor": {
                    "accession_number": anchor.accession_number,
                    "report_date": anchor.report_date,
                    "filed_date": anchor.filed_date,
                    "form_type": anchor.form_type,
                },
                "selection": selection_trace,
                "ambiguous_candidates": ambiguous_candidates,
            },
            **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
        )

    return SecGetMetricResult(
        ok=False,
        status="not_found",
        error=f"No anchored fact found for metric '{request.metric_id}'.",
        trace={
            "anchor": {
                "accession_number": anchor.accession_number,
                "report_date": anchor.report_date,
                "filed_date": anchor.filed_date,
                "form_type": anchor.form_type,
            },
            "selection": selection_trace,
        },
        **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
    )


def _execute_total_debt_metric(
    *,
    request: SecGetMetricRequest,
    metric_definition: MetricDefinition,
    anchor: FilingAnchor,
    cik: str,
    companyfacts: Dict[str, Any],
) -> SecGetMetricResult:
    component_results: List[MetricComponentResult] = []
    ambiguous_groups: List[str] = []
    missing_groups: List[str] = []
    selection_trace: Dict[str, Any] = {}

    for group in metric_definition.component_groups:
        selected_component: MetricComponentResult | None = None
        group_trace: List[Dict[str, Any]] = []
        for candidate in group.candidates:
            candidates = _iter_companyfacts_candidates(
                companyfacts=companyfacts,
                candidate=candidate,
                cik=cik,
                fiscal_year=request.fiscal_year,
                expected_unit=metric_definition.unit,
            )
            chosen, ambiguous = _select_best_anchored_fact(
                candidates=candidates,
                anchor=anchor,
                period_type=metric_definition.period_type,
            )
            group_trace.append(
                {
                    "concept_name": candidate.concept_name,
                    "taxonomy": candidate.taxonomy,
                    "candidates_considered": len(candidates),
                    "selected": chosen.concept_name if chosen is not None else None,
                    "ambiguous": ambiguous,
                }
            )
            if ambiguous:
                ambiguous_groups.append(group.group_id)
                break
            if chosen is None:
                continue
            selected_component = _to_component_result(candidate=chosen, group=group, anchor=anchor)
            break

        selection_trace[group.group_id] = group_trace
        if selected_component is not None:
            component_results.append(selected_component)
        elif group.required:
            missing_groups.append(group.group_id)

    if ambiguous_groups:
        return SecGetMetricResult(
            ok=False,
            status="ambiguous",
            components=component_results,
            missing_component_groups=missing_groups,
            error="Ambiguous anchored fact candidates prevented metric resolution.",
            trace={
                "anchor": {
                    "accession_number": anchor.accession_number,
                    "report_date": anchor.report_date,
                    "filed_date": anchor.filed_date,
                    "form_type": anchor.form_type,
                },
                "selection": selection_trace,
                "ambiguous_groups": ambiguous_groups,
            },
            **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
        )

    computed = compute_total_debt_carrying_amount(
        components=component_results,
        required_group_ids=[group.group_id for group in metric_definition.component_groups if group.required],
    )
    status = computed["status"]
    value = computed["value"]
    missing_component_groups = list(computed["missing_component_groups"])
    error = None
    if status == "partial":
        error = f"Missing carrying-amount components: {', '.join(missing_component_groups)}"
    elif status == "not_found":
        error = "No anchored carrying-amount components were found for the metric."

    return SecGetMetricResult(
        ok=status == "ok",
        status=status,
        value=value,
        unit=metric_definition.unit if value is not None else None,
        components=component_results,
        missing_component_groups=missing_component_groups,
        error=error,
        trace={
            "anchor": {
                "accession_number": anchor.accession_number,
                "report_date": anchor.report_date,
                "filed_date": anchor.filed_date,
                "form_type": anchor.form_type,
            },
            "selection": selection_trace,
            "missing_groups_from_selection": missing_groups,
        },
        **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
    )


def _execute_derived_metric(
    *,
    request: SecGetMetricRequest,
    metric_definition: MetricDefinition,
    anchor: FilingAnchor,
    cik: str,
    companyfacts: Dict[str, Any],
) -> SecGetMetricResult:
    strategy_key = metric_definition.compute_strategy
    if not strategy_key:
        return SecGetMetricResult(
            ok=False,
            status="unsupported_metric",
            error=f"Derived metric '{request.metric_id}' is registered without a compute strategy.",
            **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
        )
    strategy = _DERIVED_COMPUTE_STRATEGIES.get(strategy_key)
    if strategy is not None:
        return strategy(
            request=request,
            metric_definition=metric_definition,
            anchor=anchor,
            cik=cik,
            companyfacts=companyfacts,
        )
    return SecGetMetricResult(
        ok=False,
        status="unsupported_metric",
        error=(
            f"Derived metric '{request.metric_id}' references unknown compute strategy "
            f"'{strategy_key}'."
        ),
        **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
    )


def _amendment_observation(submissions: Dict[str, Any], companyfacts: Dict[str, Any],
                           fiscal_year: int, report_date: Optional[str]) -> Dict[str, Any]:
    def same_period(end: Any) -> bool:
        parsed = _parse_iso_date(end)
        return bool(parsed and (str(end) == report_date if report_date else parsed.year == fiscal_year))

    history = sorted({r["accession_number"] for r in _filing_rows(submissions)
                      if r["form_type"].upper() == "10-K/A" and r["accession_number"]
                      and same_period(r["report_date"])})
    facts = sorted({str(f["accn"]) for taxonomy in (companyfacts.get("facts") or {}).values()
                    for concept in taxonomy.values() for records in (concept.get("units") or {}).values()
                    for f in records if isinstance(f, dict) and f.get("accn")
                    and str(f.get("form", "")).upper() == "10-K/A" and same_period(f.get("end"))})
    return {"state": "observed" if history or facts else "unknown", "coverage": "supplied_inputs_only",
            "filing_metadata_accessions": history, "fact_candidate_accessions": facts}


def _selection_audit(companyfacts: Dict[str, Any], definition: MetricDefinition,
                     anchor: FilingAnchor) -> List[Dict[str, Any]]:
    concepts = set(definition.atomic_candidates)
    for group in definition.component_groups:
        concepts.update(group.candidates)
    audit = []
    for concept in sorted(concepts, key=lambda c: (c.taxonomy, c.concept_name)):
        units = (((companyfacts.get("facts") or {}).get(concept.taxonomy) or {}).get(
            concept.concept_name) or {}).get("units") or {}
        for unit, records in units.items():
            for item in records:
                if not isinstance(item, dict):
                    continue
                single = {"facts": {concept.taxonomy: {concept.concept_name: {"units": {unit: [item]}}}}}
                candidates = _iter_companyfacts_candidates(companyfacts=single, candidate=concept,
                    cik=anchor.cik, fiscal_year=anchor.fiscal_year, expected_unit=None)
                if definition.unit and unit.upper() != definition.unit.upper():
                    reason = "UNIT_MISMATCH"
                elif not candidates:
                    reason = "INVALID_NUMERIC_VALUE"
                else:
                    reason = _fact_rejection(candidates[0], anchor, definition.period_type)
                value = item.get("val")
                if isinstance(value, float) and not math.isfinite(value):
                    value = str(value)
                audit.append({"taxonomy": concept.taxonomy, "concept_name": concept.concept_name,
                              "unit": unit, "value": value, "accession_number": item.get("accn"),
                              "report_date": item.get("end"), "start_date": item.get("start"),
                              "form_type": item.get("form"), "reason": reason})
    return sorted(audit, key=lambda row: json.dumps(row, sort_keys=True))


def _finalize_selection(result: SecGetMetricResult, *, definition: MetricDefinition,
                         submissions: Dict[str, Any], companyfacts: Dict[str, Any],
                         anchor: Optional[FilingAnchor], reason: Optional[str] = None) -> SecGetMetricResult:
    audit = _selection_audit(companyfacts, definition, anchor) if anchor else []
    if any(row["reason"] == "FISCAL_YEAR_UNRESOLVED" for row in audit):
        reason = "FISCAL_YEAR_UNRESOLVED"
        result.ok, result.status, result.value, result.unit = False, "not_found", None, None
        result.primary_fact, result.components, result.missing_component_groups = None, [], []
    # Arithmetic is never sufficient to establish compatible evidence provenance.
    if result.ok and result.components:
        identities = {(c.accession_number, c.form_type, c.report_date,
                       "duration" if c.start_date is not None else "instant", c.start_date)
                      for c in result.components}
        if len(identities) != 1:
            reason = "INCOMPATIBLE_COMPONENT_PERIODS"
            result.ok, result.status, result.value, result.unit = False, "ambiguous", None, None
    if reason is None:
        if result.ok:
            reason = "SELECTED_ORIGINAL_ANNUAL_FACT" if definition.period_type == "duration" else "SELECTED_ORIGINAL_INSTANT_FACT"
        elif result.status == "ambiguous":
            reason = "OVERLAPPING_COMPONENTS" if "overlaps" in (result.error or "") else "CONFLICTING_ELIGIBLE_FACTS"
        elif result.status == "partial":
            reason = "MISSING_COMPONENTS"
        else:
            reason = "NO_ELIGIBLE_ANNUAL_PERIOD" if audit and definition.period_type == "duration" else "NO_ELIGIBLE_FACT"
    trace: Dict[str, Any] = {
        "policy": "original_as_filed_v1", "reason": reason,
        "amendments": _amendment_observation(submissions, companyfacts, result.fiscal_year,
                                              anchor.report_date if anchor else None),
        "selection": audit,
    }
    if anchor:
        trace["anchor"] = {"accession_number": anchor.accession_number, "report_date": anchor.report_date,
                           "filed_date": anchor.filed_date, "form_type": anchor.form_type}
        if result.metric_id == "total_debt":
            trace["missing_groups_from_selection"] = list(result.missing_component_groups)
    result.trace = trace
    if not result.ok:
        result.error = {
            "MISSING_COMPONENTS": "Missing carrying-amount components: " + ", ".join(result.missing_component_groups),
            "OVERLAPPING_COMPONENTS": "Fallback capex total overlaps with additive capex components.",
        }.get(reason, f"No anchored fact found for requested original annual filing: {reason}.")
    return result


async def get_metric(
    *,
    ticker: str,
    fiscal_year: int,
    metric_id: str,
    client: Optional[SecCompanyFactsClient] = None,
) -> SecGetMetricResult:
    request = SecGetMetricRequest(
        ticker=ticker,
        fiscal_year=fiscal_year,
        metric_id=metric_id,
    )

    metric_definition = get_metric_definition(request.metric_id)
    if metric_definition is None:
        return _unsupported_metric_result(request=request)

    active_client = client or SecCompanyFactsClient()
    cik = await active_client.resolve_cik(request.ticker)
    submissions = await active_client.get_submissions(cik)
    anchor, anchor_reason = _filing_anchor_decision(
        submissions=submissions,
        ticker=request.ticker,
        cik=cik,
        fiscal_year=request.fiscal_year,
    )
    if anchor is None:
        return _finalize_selection(SecGetMetricResult(
            ok=False,
            status="not_found",
            metric_id=request.metric_id,
            ticker=request.ticker,
            cik=cik,
            fiscal_year=request.fiscal_year,
            error=f"No annual filing anchor found for {request.ticker} fiscal year {request.fiscal_year}.",
        ), definition=metric_definition, submissions=submissions, companyfacts={},
            anchor=None, reason=anchor_reason)

    companyfacts = await active_client.get_companyfacts(cik)
    if metric_definition.kind == "atomic":
        result = _execute_atomic_metric(
            request=request,
            metric_definition=metric_definition,
            anchor=anchor,
            cik=cik,
            companyfacts=companyfacts,
        )
    elif metric_definition.kind == "derived":
        result = _execute_derived_metric(
            request=request,
            metric_definition=metric_definition,
            anchor=anchor,
            cik=cik,
            companyfacts=companyfacts,
        )
    else:
        return _unsupported_metric_result(
            request=request,
            error=f"Metric '{request.metric_id}' is registered with unsupported kind '{metric_definition.kind}'.",
        )
    return _finalize_selection(result, definition=metric_definition, submissions=submissions,
                                companyfacts=companyfacts, anchor=anchor)


def compute_capex_value(
    *,
    primary_cash_capex: Optional[MetricComponentResult],
    productive_assets_additional: Optional[MetricComponentResult],
    fallback_capex_total: Optional[MetricComponentResult],
) -> Dict[str, Any]:
    components: List[MetricComponentResult] = []
    if primary_cash_capex is not None:
        components.append(primary_cash_capex)
    if productive_assets_additional is not None:
        components.append(productive_assets_additional)

    if fallback_capex_total is not None and components:
        return {
            "status": "ambiguous",
            "value": None,
            "components": components + [fallback_capex_total],
            "missing_component_groups": [],
            "error": "Fallback capex total overlaps with additive capex components.",
        }

    if components:
        return {
            "status": "ok",
            "value": float(sum(component.value for component in components)),
            "components": components,
            "missing_component_groups": [],
            "error": None,
        }

    if fallback_capex_total is not None:
        return {
            "status": "ok",
            "value": float(fallback_capex_total.value),
            "components": [fallback_capex_total],
            "missing_component_groups": [],
            "error": None,
        }

    return {
        "status": "not_found",
        "value": None,
        "components": [],
        "missing_component_groups": [],
        "error": "No anchored capex facts were found for the metric.",
    }


def _select_component_for_group(
    *,
    group: MetricComponentGroup,
    request: SecGetMetricRequest,
    metric_definition: MetricDefinition,
    anchor: FilingAnchor,
    cik: str,
    companyfacts: Dict[str, Any],
) -> tuple[Optional[MetricComponentResult], List[Dict[str, Any]], bool]:
    selected_component: MetricComponentResult | None = None
    group_trace: List[Dict[str, Any]] = []
    ambiguous = False
    for candidate in group.candidates:
        candidates = _iter_companyfacts_candidates(
            companyfacts=companyfacts,
            candidate=candidate,
            cik=cik,
            fiscal_year=request.fiscal_year,
            expected_unit=metric_definition.unit,
        )
        chosen, candidate_ambiguous = _select_best_anchored_fact(
            candidates=candidates,
            anchor=anchor,
            period_type=metric_definition.period_type,
        )
        group_trace.append(
            {
                "concept_name": candidate.concept_name,
                "taxonomy": candidate.taxonomy,
                "candidates_considered": len(candidates),
                "selected": chosen.concept_name if chosen is not None else None,
                "ambiguous": candidate_ambiguous,
            }
        )
        if candidate_ambiguous:
            ambiguous = True
            break
        if chosen is None:
            continue
        selected_component = _to_component_result(candidate=chosen, group=group, anchor=anchor)
        break
    return selected_component, group_trace, ambiguous


def _execute_capex_metric(
    *,
    request: SecGetMetricRequest,
    metric_definition: MetricDefinition,
    anchor: FilingAnchor,
    cik: str,
    companyfacts: Dict[str, Any],
) -> SecGetMetricResult:
    selected_by_group: Dict[str, Optional[MetricComponentResult]] = {}
    selection_trace: Dict[str, Any] = {}
    ambiguous_groups: List[str] = []

    for group in metric_definition.component_groups:
        selected, group_trace, ambiguous = _select_component_for_group(
            group=group,
            request=request,
            metric_definition=metric_definition,
            anchor=anchor,
            cik=cik,
            companyfacts=companyfacts,
        )
        selection_trace[group.group_id] = group_trace
        selected_by_group[group.group_id] = selected
        if ambiguous:
            ambiguous_groups.append(group.group_id)

    if ambiguous_groups:
        return SecGetMetricResult(
            ok=False,
            status="ambiguous",
            components=[
                component
                for component in selected_by_group.values()
                if component is not None
            ],
            error="Ambiguous anchored capex fact candidates prevented metric resolution.",
            trace={
                "anchor": {
                    "accession_number": anchor.accession_number,
                    "report_date": anchor.report_date,
                    "filed_date": anchor.filed_date,
                    "form_type": anchor.form_type,
                },
                "selection": selection_trace,
                "ambiguous_groups": ambiguous_groups,
            },
            **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
        )

    computed = compute_capex_value(
        primary_cash_capex=selected_by_group.get("primary_cash_capex"),
        productive_assets_additional=selected_by_group.get("productive_assets_additional"),
        fallback_capex_total=selected_by_group.get("fallback_capex_total"),
    )
    status = computed["status"]
    return SecGetMetricResult(
        ok=status == "ok",
        status=status,
        value=computed["value"],
        unit=metric_definition.unit if computed["value"] is not None else None,
        components=list(computed["components"]),
        missing_component_groups=list(computed["missing_component_groups"]),
        error=computed["error"],
        trace={
            "anchor": {
                "accession_number": anchor.accession_number,
                "report_date": anchor.report_date,
                "filed_date": anchor.filed_date,
                "form_type": anchor.form_type,
            },
            "selection": selection_trace,
        },
        **_base_result_kwargs(request=request, anchor=anchor, cik=cik),
    )


_DERIVED_COMPUTE_STRATEGIES: dict[str, DerivedMetricStrategy] = {
    "total_debt_carrying_amount": _execute_total_debt_metric,
    "capex_value": _execute_capex_metric,
}


async def fetch_metric_direct(
    *,
    ticker: str,
    fiscal_year: int,
    metric_id: str,
    client: Optional[SecCompanyFactsClient] = None,
) -> SecGetMetricResult:
    return await get_metric(
        ticker=ticker,
        fiscal_year=fiscal_year,
        metric_id=metric_id,
        client=client,
    )


async def sec_get_metric(
    *,
    ticker: str,
    fiscal_year: int,
    metric_id: str,
) -> Dict[str, Any]:
    try:
        result = await get_metric(
            ticker=ticker,
            fiscal_year=fiscal_year,
            metric_id=metric_id,
        )
    except ValidationError as exc:
        result = SecGetMetricResult(
            ok=False,
            status="not_found",
            metric_id=str(metric_id).strip(),
            ticker=str(ticker).strip().upper(),
            fiscal_year=int(fiscal_year),
            error=f"Invalid request: {exc}",
        )
    return result.model_dump(mode="json")


def register_tools(mcp: FastMCP) -> None:
    mcp.tool()(sec_get_metric)


def build_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    mount_path: str = "/",
) -> FastMCP:
    mcp = FastMCP("sec-metric", host=host, port=port, mount_path=mount_path)
    register_tools(mcp)
    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SEC metric MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mount-path", default=None)
    args = parser.parse_args(argv)

    build_mcp_server(
        host=args.host or "127.0.0.1",
        port=int(args.port) if args.port is not None else 8000,
        mount_path=args.mount_path or "/",
    ).run(
        transport=args.transport,
        mount_path=args.mount_path,
    )


if __name__ == "__main__":
    main()
