"""Build a new source-first question set before observing any system outputs."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import shutil
import unicodedata

from bs4 import BeautifulSoup

from build_semantic_sources_v1 import CORPUS, sha
from semantic_annotations_v1 import NAMES, PASSAGES, HARD_VALUES, OVERRIDES, QUESTIONS

FILINGS = [("AAPL", 2024), ("AAPL", 2025), ("AMZN", 2023),
           ("AMZN", 2024), ("MSFT", 2024), ("MSFT", 2025)]


def canonical(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def numeric_claim(fact, claim_id):
    return {"claim_id": claim_id, "claim_type": "structured_numeric",
        "requirement": f"{NAMES[fact['ticker']]} FY{fact['fact_fiscal_year']} {fact['metric_id']} is {fact['value']} USD.",
        "evidence_policy": "structured_fact", "sources": [{"kind": "inline_xbrl", **fact}],
        "numeric": {"metric_id": fact["metric_id"], "ticker": fact["ticker"],
                    "fiscal_year": fact["fact_fiscal_year"], "value": fact["value"],
                    "unit": "USD", "requested_display_unit": "USD_millions", "absolute_tolerance": 500000}}


def kb_claim(docs, ticker, year, group, claim_id):
    kind, anchors, requirement = PASSAGES[ticker][group]
    override_anchors, override_requirement = OVERRIDES.get((ticker, year, group), (None, None))
    anchors = [s.format(year=year) for s in (override_anchors or anchors)]
    requirement = (override_requirement or requirement).format(**HARD_VALUES[(ticker, year)])
    selected = [d for d in docs if d["metadata"]["ticker"] == ticker
        and d["metadata"]["fiscal_year"] == year and d["metadata"]["doc_type"] == kind
        and all(anchor in d["content"] for anchor in anchors)]
    if ticker == "MSFT" and group == "hard_numeric":
        selected = [d for d in selected if re.search(r"\|\s*Research and development\s*\|", d["content"])]
    if not selected:
        raise ValueError(f"No independently anchored source: {ticker}/{year}/{group}")
    source_path = Path(selected[0]["metadata"]["source_html"])
    source_text = canonical(BeautifulSoup(source_path.read_text(), "xml").get_text(" ", strip=True))
    for anchor in anchors:
        if canonical(anchor) not in source_text:
            raise ValueError("Anchor not independently found in original filing")
    evidence = []
    for doc in selected:
        spans = []
        for anchor in anchors:
            start = doc["content"].index(anchor)
            if kind == "table":
                start, end = 0, len(doc["content"])
            else:
                end = doc["content"].find(".", start + len(anchor)) + 1
                end = end if end > start else len(doc["content"])
                # A complete sentence anchor does not need the following sentence.
                if anchor.endswith("."): end = start + len(anchor)
            span = {"start": start, "end": end, "quote": doc["content"][start:end],
                    "anchor": anchor, "normalized_source_offset": source_text.index(canonical(anchor))}
            if span not in spans: spans.append(span)
        evidence.append({"kind": "kb", "evidence_id": doc["id"],
                         "content_sha256": doc["content_sha256"], **doc["metadata"], "spans": spans})
    return {"claim_id": claim_id, "claim_type": "attribution" if group == "attribution" else "narrative",
            "requirement": requirement, "evidence_policy": "kb",
            "sources": evidence, "alternatives_policy": "Any listed chunk supports the same required facet; no ranked retrieval was used."}


def build(source_root, out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ("numeric_source_facts.jsonl", "source_manifest.json"):
        shutil.copyfile(source_root / name, out / name)
    shutil.copytree(source_root / "tables", out / "tables")
    facts = [json.loads(x) for x in (out / "numeric_source_facts.jsonl").read_text().splitlines()]
    docs = [json.loads(x) for x in (CORPUS / "corpus.jsonl").read_text().splitlines()]
    cases = []
    for ticker, year in FILINGS:
        def fact(metric, fact_year=year):
            found = [f for f in facts if f["ticker"] == ticker and f["filing_fiscal_year"] == year
                     and f["fact_fiscal_year"] == fact_year and f["metric_id"] == metric]
            if len(found) != 1: raise ValueError("Ambiguous source fact")
            return found[0]

        def add(number, stratum, question, claims, **extras):
            cases.append({"id": f"SEM1_{ticker}_{year}_{number:02}", "stratum": stratum,
                "user_query": question, "ticker": ticker, "fiscal_year": year, "form_type": "10-K",
                "expected_answerability": "answerable", "required_claims": claims,
                "annotation_method": "Source-inspected assistant annotation, not retriever/answer-derived or human-adjudicated.",
                **extras})

        name = NAMES[ticker]
        revenue, cash = fact("revenue"), fact("cash_and_cash_equivalents")
        previous = fact("revenue", year - 1)
        add(1, "structured_numeric", f"What total revenue did {name} report for FY{year} in its 10-K? Give the value in USD millions.", [numeric_claim(revenue, "revenue")])
        add(2, "structured_numeric", f"What cash and cash equivalents did {name} report at the end of FY{year}? Give the carrying value in USD millions, excluding short-term investments.", [numeric_claim(cash, "cash")])
        add(3, "narrative", QUESTIONS[ticker]["policy"].format(year=year), [kb_claim(docs, ticker, year, "policy", "policy")])
        add(4, "hybrid", QUESTIONS[ticker]["growth"].format(year=year), [numeric_claim(revenue, "revenue"), kb_claim(docs, ticker, year, "growth", "growth_drivers")])
        add(5, "comparison", f"Compare {name}'s FY{year} and FY{year-1} total revenues side by side in USD millions, using the amounts reported in its FY{year} 10-K. Identify the fiscal year of each amount.", [numeric_claim(revenue, "current_revenue"), numeric_claim(previous, "previous_revenue")])
        change = (Decimal(revenue["value"]) - Decimal(previous["value"])) / Decimal(previous["value"]) * 100
        calculation = {"claim_id": "growth_percent", "claim_type": "calculation", "evidence_policy": "calculation_inputs",
            "requirement": f"Revenue grew by {change.quantize(Decimal('0.01'))}% from FY{year-1} to FY{year}.",
            "sources": [{"kind": "inline_xbrl", **f} for f in (previous, revenue)],
            "numeric": {"metric_id": "revenue_growth_percent", "ticker": ticker, "fiscal_year": year,
                        "value": float(change), "unit": "percent", "requested_display_unit": "percent", "absolute_tolerance": 0.005001},
            "formula": "(current_revenue - previous_revenue) / previous_revenue * 100"}
        add(6, "calculator", f"Calculate {name}'s revenue growth from FY{year-1} to FY{year} using its FY{year} 10-K. Show the two revenues in USD millions and the percentage change rounded to two decimals.", [numeric_claim(previous, "previous_revenue"), numeric_claim(revenue, "current_revenue"), calculation], calculator_required=True)
        add(7, "multiple_claims", QUESTIONS[ticker]["multi"].format(year=year), [kb_claim(docs, ticker, year, "business", "business_or_governance"), kb_claim(docs, ticker, year, "risk", "risk")])
        add(8, "difficult_attribution", QUESTIONS[ticker]["attribution"].format(year=year), [kb_claim(docs, ticker, year, "attribution", "attribution")])
        hard = kb_claim(docs, ticker, year, "hard_numeric", "correct_measure")
        hard["claim_type"] = "kb_numeric"
        values = HARD_VALUES[(ticker, year)]
        specs = ({"services_gross_margin_percent": (values["value"], "percent")} if ticker == "AAPL" else
                 {"research_and_development": (values["value"] * 1000000, "USD")} if ticker == "MSFT" else
                 {"aws_leased_area": (values["leased"], "thousand_square_feet"), "aws_owned_area": (values["owned"], "thousand_square_feet")})
        hard_claims = []
        for metric, (value, unit) in specs.items():
            requirement = hard["requirement"]
            if ticker == "AMZN":
                requirement = f"AWS {('owned' if metric == 'aws_owned_area' else 'leased')} facility area is {value} thousand square feet in FY{year}."
            hard_claims.append({**hard, "claim_id": metric, "requirement": requirement, "numeric": {"metric_id": metric, "ticker": ticker,
                "fiscal_year": year, "value": value, "unit": unit,
                "requested_display_unit": "USD_millions" if unit == "USD" else unit,
                "absolute_tolerance": 500000 if unit == "USD" else 0.000001,
                "source_cell": {"row_label": "Services" if ticker == "AAPL" else "AWS" if ticker == "AMZN" else "Research and development",
                                "value_column": 2 if metric == "aws_owned_area" else 1,
                                "scale": 6 if unit == "USD" else 0}}})
        add(9, "plausible_wrong_evidence", QUESTIONS[ticker]["hard"].format(year=year), hard_claims)
        cases[-1]["known_wrong_evidence"] = kb_claim(docs, ticker, year, "negative", "wrong_measure")["sources"]
        add(10, "insufficient_data", f"Using only {name}'s FY{year} 10-K, what was its audited actual net income for FY{year+1}? Do not use later filings or forecasts; say if the specified filing cannot establish it.", [],
            expected_answerability="insufficient_data", answerability_reason=f"The specified FY{year} annual filing does not report audited actual results for future FY{year+1}; later filings are outside the question's evidence scope.",
            answerability_sources=[{"kind": "inline_xbrl", **f} for f in facts if f["ticker"] == ticker and f["filing_fiscal_year"] == year and f["metric_id"] == "net_income"])
    cases.sort(key=lambda c: c["id"])
    rng = random.Random(20260907)
    audited = set()
    for stratum in sorted({c["stratum"] for c in cases}):
        audited.update(c["id"] for c in rng.sample([c for c in cases if c["stratum"] == stratum], 2))
    for case in cases: case["audit_selected"] = case["id"] in audited
    (out / "queries.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases))
    (out / "draft_counts.json").write_text(json.dumps({"cases": len(cases), "strata": dict(Counter(c["stratum"] for c in cases)), "audit_ids": sorted(audited)}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    build(a.source_root, a.out_dir)
