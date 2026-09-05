"""One-time PR8 oracle capture. Expected selections are declared, never runtime-derived."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = "f09e8844a2c12c0b74d003e1ab9b601353a62a04"
DEST = Path("data/evals/agents/v1/filing_period_pr8.json")
ACC = "0000320193-25-000073"
END = "2025-09-27"
START = "2024-09-29"
REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
DEBT = ["LongTermDebtCurrent", "LongTermDebtNoncurrent"]
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
LABELS = {DEBT[0]: ("current_debt", "Current debt carrying amount"),
          DEBT[1]: ("noncurrent_debt", "Noncurrent debt carrying amount"),
          CAPEX[0]: ("primary_cash_capex", "Primary cash capital expenditures"),
          CAPEX[1]: ("productive_assets_additional", "Additional productive asset purchases")}


def fact(**updates):
    return {"start": START, "end": END, "val": 100, "accn": ACC,
            "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-10-31", **updates}


def filing(**updates):
    return {"accessionNumber": ACC, "reportDate": END, "filingDate": "2025-10-31",
            "form": "10-K", "primaryDocument": "aapl-20250927.htm", **updates}


def case(name, facts=None, *, metric="revenue", filings=None, chosen=None,
         status="ok", reason=None, missing=None):
    facts = facts if facts is not None else {REVENUE: [fact()]}
    filings = [filing()] if filings is None else filings
    if chosen is None:
        chosen = [(next(iter(facts)), 0)] if status == "ok" else []
    return {"id": name, "request": {"ticker": "AAPL", "fiscal_year": 2025, "metric_id": metric},
            "submissions": {"filings": {"recent": {k: [f.get(k, "") for f in filings]
                                                     for k in filing()}}},
            "companyfacts": {"facts": {"us-gaap": {k: {"units": {"USD": v}} for k, v in facts.items()}}},
            "spec": {"chosen": chosen, "status": status,
                     "reason": reason or ("SELECTED_ORIGINAL_INSTANT_FACT" if metric == "total_debt"
                                           else "SELECTED_ORIGINAL_ANNUAL_FACT"),
                     "missing": missing or []}}


def cases():
    rows = [case("original"), case("no_facts", {}, status="not_found", reason="NO_ELIGIBLE_FACT")]
    amend = filing(accessionNumber="0000320193-25-000074", form="10-K/A", filingDate="2025-11-15")
    rows += [case("amendment_no_metric", filings=[filing(), amend]),
             case("amendment_changed", {REVENUE: [fact(), fact(accn=amend["accessionNumber"], form="10-K/A", val=999)]}, filings=[amend, filing()]),
             case("amendment_candidate_only", {REVENUE: [fact(), fact(accn=amend["accessionNumber"], form="10-K/A", val=999)]}),
             case("amendment_only", filings=[amend], status="not_found", reason="AMENDMENT_ONLY"),
             case("no_history", filings=[], status="not_found", reason="NO_ORIGINAL_FILING"),
             case("two_originals", filings=[filing(), filing(accessionNumber="0000320193-25-000080")], status="not_found", reason="FISCAL_YEAR_UNRESOLVED"),
             case("duplicate_filing", filings=[filing(), filing()]),
             case("missing_accession", filings=[filing(accessionNumber="")], status="not_found", reason="INVALID_FILING_METADATA"),
             case("invalid_report_date", filings=[filing(reportDate="2025-99-27")], status="not_found", reason="INVALID_FILING_METADATA")]
    for days in [1, 90, 180, 270, 330, 363, 364, 365, 366, 367, 370, 371, 372, 400]:
        valid = days in {364, 365, 366, 371}
        rows.append(case(f"duration_{days}", {REVENUE: [fact(start=str(date.fromisoformat(END)-timedelta(days=days-1)))]},
                         status="ok" if valid else "not_found",
                         reason=None if valid else "NO_ELIGIBLE_ANNUAL_PERIOD"))
    for name, changes in [("missing_start", {"start": None}), ("invalid_start", {"start": "bad"}),
                          ("reverse_period", {"start": "2025-09-28"}),
                          ("wrong_end", {"end": "2024-09-27"}),
                          ("wrong_accession", {"accn": "0000320193-26-000001"}),
                          ("wrong_form", {"form": "10-K/A"})]:
        rows.append(case(name, {REVENUE: [fact(**changes)]}, status="not_found", reason="NO_ELIGIBLE_ANNUAL_PERIOD"))
    rows += [case("quarter_plus_annual", {REVENUE: [fact(start="2025-07-01", val=20), fact()]}, chosen=[(REVENUE, 1)]),
             case("annual_without_fp", {REVENUE: [fact(fp=None)]}),
             case("annual_with_frame", {REVENUE: [fact(frame="CY2025")]}),
             case("fiscal_label_mismatch", {REVENUE: [fact(fy=2024)]}, status="not_found", reason="FISCAL_YEAR_UNRESOLVED"),
             case("identical_duplicates", {REVENUE: [fact(), fact()]}),
             case("different_metadata_same_value", {REVENUE: [fact(filed="2025-11-01"), fact()]}, chosen=[(REVENUE, 1)]),
             case("conflicting_duplicates", {REVENUE: [fact(), fact(val=101)]}, status="ambiguous", reason="CONFLICTING_ELIGIBLE_FACTS"),
             case("different_filed_conflict", {REVENUE: [fact(), fact(val=101, filed="2025-11-01")]}, status="ambiguous", reason="CONFLICTING_ELIGIBLE_FACTS"),
             case("different_annual_starts", {REVENUE: [fact(), fact(start="2024-09-28")]}, status="ambiguous", reason="CONFLICTING_ELIGIBLE_FACTS"),
             case("preferred_conflict_no_fallback", {REVENUE: [fact(), fact(val=101)], "Revenues": [fact(val=99)]}, status="ambiguous", reason="CONFLICTING_ELIGIBLE_FACTS"),
             case("valid_registry_fallback", {REVENUE: [fact(start="2025-07-01")], "Revenues": [fact(val=99)]}, chosen=[("Revenues", 0)])]
    debt = {DEBT[0]: [fact(start=None, val=10)], DEBT[1]: [fact(start=None, val=20)]}
    rows.append(case("debt_original", debt, metric="total_debt", chosen=[(t, 0) for t in DEBT]))
    for name, change in [("debt_amendment_component", {"accn": amend["accessionNumber"], "form": "10-K/A"}),
                         ("debt_wrong_end", {"end": "2024-09-27"}),
                         ("debt_duration_component", {"start": START})]:
        d = deepcopy(debt)
        d[DEBT[1]][0].update(change)
        rows.append(case(name, d, metric="total_debt", chosen=[(DEBT[0], 0)], status="partial", reason="MISSING_COMPONENTS", missing=["noncurrent_debt"]))
    d = deepcopy(debt)
    d[DEBT[1]].append(fact(start=None, val=21))
    rows.append(case("debt_conflict_with_survivor", d, metric="total_debt", chosen=[(DEBT[0], 0)], status="ambiguous", reason="CONFLICTING_ELIGIBLE_FACTS", missing=["noncurrent_debt"]))
    capex = {CAPEX[0]: [fact(val=10)], CAPEX[1]: [fact(val=20)]}
    rows.append(case("capex_original", capex, metric="capex", chosen=[(t, 0) for t in CAPEX]))
    d = deepcopy(capex)
    d[CAPEX[1]][0]["start"] = "2024-09-28"
    rows.append(case("capex_mixed_starts", d, metric="capex", chosen=[(t, 0) for t in CAPEX], status="ambiguous", reason="INCOMPATIBLE_COMPONENT_PERIODS"))
    return rows


def independent_audit(c, anchor):
    """Raw-data oracle; no runtime registry, resolver or selection imports."""
    audit = []
    instant = c["request"]["metric_id"] == "total_debt"
    for concept, payload in c["companyfacts"]["facts"]["us-gaap"].items():
        for unit, records in payload["units"].items():
            for f in records:
                row = {"taxonomy": "us-gaap", "concept_name": concept, "unit": unit,
                       "value": f["val"], "accession_number": f.get("accn"),
                       "report_date": f.get("end"), "start_date": f.get("start"), "form_type": f.get("form")}
                reason = "ELIGIBLE"
                if unit != "USD": reason = "UNIT_MISMATCH"
                elif f.get("accn") != anchor["accession_number"]: reason = "ACCESSION_MISMATCH"
                elif f.get("form") != "10-K": reason = "FORM_MISMATCH"
                elif f.get("end") != anchor["report_date"]: reason = "END_DATE_MISMATCH"
                elif f.get("fy") not in (None, 2025): reason = "FISCAL_YEAR_UNRESOLVED"
                elif instant and f.get("start") is not None: reason = "PERIOD_TYPE_MISMATCH"
                elif not instant:
                    try:
                        s, e = date.fromisoformat(f["start"]), date.fromisoformat(f["end"])
                        if s >= e or (e-s).days+1 not in (364, 365, 366, 371): reason = "NON_ANNUAL_DURATION"
                    except (ValueError, TypeError, KeyError): reason = "INVALID_DURATION_START"
                row["reason"] = reason
                audit.append(row)
    return sorted(audit, key=lambda x: json.dumps(x, sort_keys=True))


def expected(c):
    """Build full expected response from declared outcomes and raw fixture provenance."""
    spec = c["spec"]
    request = c["request"]
    result = {"ok": spec["status"] == "ok", "status": spec["status"], **request,
              "value": None, "unit": None, "cik": "0000320193", "form_type": None,
              "accession_number": None, "report_date": None, "filed_date": None,
              "source_url": None, "primary_fact": None, "components": [],
              "missing_component_groups": spec["missing"], "error": None}
    no_anchor = spec["reason"] in {"AMENDMENT_ONLY", "NO_ORIGINAL_FILING", "INVALID_FILING_METADATA", "FISCAL_YEAR_UNRESOLVED"} and c["id"] != "fiscal_label_mismatch"
    history = c["submissions"]["filings"]["recent"]
    known_history = sorted(set(a for a, f, e in zip(history["accessionNumber"], history["form"], history["reportDate"])
                               if f == "10-K/A" and e == END and a))
    known_facts = [] if no_anchor else sorted(set(f["accn"] for p in c["companyfacts"]["facts"]["us-gaap"].values()
                                                  for records in p["units"].values() for f in records
                                                  if f.get("form") == "10-K/A" and f.get("end") == END and f.get("accn")))
    trace = {"policy": "original_as_filed_v1", "reason": spec["reason"],
             "amendments": {"state": "observed" if known_history or known_facts else "unknown",
                            "coverage": "supplied_inputs_only", "filing_metadata_accessions": known_history,
                            "fact_candidate_accessions": known_facts}, "selection": []}
    if not no_anchor:
        result.update(form_type="10-K", accession_number=ACC, report_date=END, filed_date="2025-10-31",
                      source_url=f"https://www.sec.gov/Archives/edgar/data/320193/{ACC.replace('-', '')}/aapl-20250927.htm")
        trace["anchor"] = {k: result[k] for k in ("accession_number", "report_date", "filed_date", "form_type")}
        trace["selection"] = independent_audit(c, trace["anchor"])
        selected = []
        for concept, index in spec["chosen"]:
            f = c["companyfacts"]["facts"]["us-gaap"][concept]["units"]["USD"][index]
            item = {"taxonomy": "us-gaap", "concept_name": concept, "unit": "USD", "value": float(f["val"]),
                    "accession_number": f["accn"], "report_date": f["end"], "filed_date": f["filed"],
                    "form_type": f["form"], "fp": f.get("fp"), "start_date": f.get("start"),
                    "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
                    "matched_by_accession": True, "matched_by_report_date": True}
            if request["metric_id"] in {"total_debt", "capex"}:
                item.update(group_id=LABELS[concept][0], group_label=LABELS[concept][1])
            selected.append(item)
        if request["metric_id"] in {"total_debt", "capex"}:
            result["components"] = selected
            if request["metric_id"] == "total_debt": trace["missing_groups_from_selection"] = spec["missing"]
        elif selected: result["primary_fact"] = selected[0]
        if result["ok"]:
            result["value"] = sum(x["value"] for x in selected)
            result["unit"] = "USD"
    if not result["ok"]:
        result["error"] = {
            "MISSING_COMPONENTS": "Missing carrying-amount components: " + ", ".join(spec["missing"]),
            "OVERLAPPING_COMPONENTS": "Fallback capex total overlaps with additive capex components.",
        }.get(spec["reason"], f"No anchored fact found for requested original annual filing: {spec['reason']}.")
    result["trace"] = trace
    return result


class CapturedClient:
    def __init__(self, c): self.c, self.calls = c, []
    async def resolve_cik(self, ticker): self.calls.append(["resolve_cik", ticker]); return "0000320193"
    async def get_submissions(self, cik): self.calls.append(["get_submissions", cik]); return deepcopy(self.c["submissions"])
    async def get_companyfacts(self, cik): self.calls.append(["get_companyfacts", cik]); return deepcopy(self.c["companyfacts"])


def changed_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(a.keys() | b.keys()) for p in (
            changed_paths(a[k], b[k], f"{prefix}/{k}") if k in a and k in b else [f"{prefix}/{k}"])]
    return [] if a == b else [prefix]


async def main():
    from mcp_server.tools.sec_metric import get_metric
    assert not DEST.exists(), "Frozen oracle must not be overwritten"
    for path in ["src/mcp_server/tools/sec_metric.py", "src/mcp_server/tools/sec_metric_registry.py"]:
        assert Path(path).read_bytes() == subprocess.check_output(["git", "show", f"{BASE}:{path}"])
    rows = cases()
    for c in rows:
        client = CapturedClient(c)
        old = (await get_metric(**c["request"], client=client)).model_dump(mode="json")
        new = expected(c)
        c.update(old_result=old, old_calls=client.calls, expected_pr8_result=new,
                 difference_expected=old != new, allowed_fields=changed_paths(old, new),
                 semantic_change_expected=any(old[k] != new[k] for k in ("status", "value", "accession_number")),
                 reason=c["spec"]["reason"])
    periods = []
    for name in ["data/html_filings/AAPL/10-K/10-K_2025.html", "data/html_filings/MSFT/10-K/10-K_2024.html"]:
        raw = Path(name).read_bytes()
        pairs = sorted(set(re.findall(r"<xbrli:period><xbrli:startDate>([^<]+)</xbrli:startDate><xbrli:endDate>([^<]+)</xbrli:endDate>", raw.decode())))
        periods.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "annual_contexts": [
            {"start": s, "end": e, "inclusive_days": (date.fromisoformat(e)-date.fromisoformat(s)).days+1}
            for s, e in pairs if 360 <= (date.fromisoformat(e)-date.fromisoformat(s)).days+1 <= 375]})
    DEST.write_text(json.dumps({"source_commit": BASE, "annual_days": [364, 365, 366, 371],
                               "period_sources": periods, "cases": rows}, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"frozen_cases": len(rows), "source_commit": BASE}))


if __name__ == "__main__": asyncio.run(main())
