from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
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


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
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


def _extract_filing_anchor(*, submissions: Dict[str, Any], ticker: str, cik: str, fiscal_year: int) -> FilingAnchor | None:
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms = list(recent.get("form") or [])
    accession_numbers = list(recent.get("accessionNumber") or [])
    report_dates = list(recent.get("reportDate") or [])
    filing_dates = list(recent.get("filingDate") or [])
    primary_documents = list(recent.get("primaryDocument") or [])

    candidates: List[Dict[str, Any]] = []
    for index, form_type in enumerate(forms):
        form_text = str(form_type or "").strip().upper()
        report_date = str(report_dates[index] or "").strip() if index < len(report_dates) else ""
        if form_text not in {"10-K", "10-K/A"}:
            continue
        if not report_date.startswith(f"{fiscal_year}-"):
            continue
        candidates.append(
            {
                "accession_number": str(accession_numbers[index] or "").strip(),
                "report_date": report_date,
                "filed_date": str(filing_dates[index] or "").strip(),
                "form_type": form_text,
                "primary_document": str(primary_documents[index] or "").strip(),
            }
        )

    if not candidates:
        return None

    def _candidate_sort_key(candidate: Dict[str, Any]) -> tuple[int, date, date]:
        return (
            1 if candidate["form_type"] == "10-K" else 0,
            _parse_iso_date(candidate.get("report_date")) or date.min,
            _parse_iso_date(candidate.get("filed_date")) or date.min,
        )

    selected = max(candidates, key=_candidate_sort_key)
    return FilingAnchor(
        cik=str(cik).zfill(10),
        ticker=str(ticker).strip().upper(),
        fiscal_year=int(fiscal_year),
        accession_number=selected["accession_number"],
        report_date=selected["report_date"],
        filed_date=selected["filed_date"],
        form_type=selected["form_type"],
        source_url=_build_filing_source_url(
            cik=str(cik).zfill(10),
            accession_number=selected["accession_number"],
            primary_document=selected["primary_document"],
        ),
    )


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
            if form_type not in {"10-K", "10-K/A"}:
                continue
            if item.get("frame"):
                continue
            try:
                numeric_value = float(item.get("val"))
            except Exception:
                continue
            accession_number = str(item.get("accn") or "").strip()
            report_date = str(item.get("end") or "").strip() or None
            filed_date = str(item.get("filed") or "").strip() or None
            fp = str(item.get("fp") or "").strip().upper() or None
            candidate_fiscal_year = item.get("fy")
            try:
                candidate_fiscal_year = int(candidate_fiscal_year) if candidate_fiscal_year is not None else None
            except Exception:
                candidate_fiscal_year = None
            report_year = _parse_iso_date(report_date).year if _parse_iso_date(report_date) else None
            if candidate_fiscal_year not in {None, fiscal_year} and report_year != fiscal_year:
                continue
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
                )
            )
    return results


def _select_best_anchored_fact(
    *,
    candidates: Sequence[FactCandidate],
    anchor: FilingAnchor,
    period_type: Literal["instant", "duration"],
) -> tuple[FactCandidate | None, bool]:
    anchored = [
        candidate
        for candidate in candidates
        if candidate.accession_number == anchor.accession_number or candidate.report_date == anchor.report_date
    ]
    if not anchored:
        return None, False

    def _sort_key(candidate: FactCandidate) -> tuple[int, int, int, int, int, date]:
        accession_match = 1 if candidate.accession_number == anchor.accession_number else 0
        annual_form_match = 1 if (candidate.form_type or "").upper() in {"10-K", "10-K/A"} else 0
        fy_match = 1 if (candidate.fp or "").upper() == "FY" else 0
        report_date_match = 1 if candidate.report_date == anchor.report_date else 0
        fiscal_year_match = 1 if candidate.fiscal_year == anchor.fiscal_year else 0
        filed_date = _parse_iso_date(candidate.filed_date) or date.min
        if period_type == "duration":
            return (
                accession_match,
                annual_form_match,
                fy_match,
                report_date_match,
                fiscal_year_match,
                filed_date,
            )
        return (
            accession_match,
            report_date_match,
            annual_form_match,
            fiscal_year_match,
            fy_match,
            filed_date,
        )

    ordered = sorted(anchored, key=_sort_key, reverse=True)
    best = ordered[0]
    best_key = _sort_key(best)
    tied = [candidate for candidate in ordered if _sort_key(candidate) == best_key]
    distinct_values = {round(candidate.value, 8) for candidate in tied}
    if len(distinct_values) > 1:
        return None, True
    return best, False


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
                continue
            if chosen is None:
                continue
            selected_component = _to_component_result(candidate=chosen, group=group, anchor=anchor)
            break

        selection_trace[group.group_id] = group_trace
        if selected_component is not None:
            component_results.append(selected_component)
        elif group.required:
            missing_groups.append(group.group_id)

    if ambiguous_groups and not component_results:
        return SecGetMetricResult(
            ok=False,
            status="ambiguous",
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
    anchor = _extract_filing_anchor(
        submissions=submissions,
        ticker=request.ticker,
        cik=cik,
        fiscal_year=request.fiscal_year,
    )
    if anchor is None:
        return SecGetMetricResult(
            ok=False,
            status="not_found",
            metric_id=request.metric_id,
            ticker=request.ticker,
            cik=cik,
            fiscal_year=request.fiscal_year,
            error=f"No annual filing anchor found for {request.ticker} fiscal year {request.fiscal_year}.",
        )

    companyfacts = await active_client.get_companyfacts(cik)
    if metric_definition.kind == "atomic":
        return _execute_atomic_metric(
            request=request,
            metric_definition=metric_definition,
            anchor=anchor,
            cik=cik,
            companyfacts=companyfacts,
        )
    if metric_definition.kind == "derived":
        return _execute_derived_metric(
            request=request,
            metric_definition=metric_definition,
            anchor=anchor,
            cik=cik,
            companyfacts=companyfacts,
        )
    return _unsupported_metric_result(
        request=request,
        error=f"Metric '{request.metric_id}' is registered with unsupported kind '{metric_definition.kind}'.",
    )


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
            continue
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
