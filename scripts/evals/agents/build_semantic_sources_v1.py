"""Source-only benchmark preparation: no retriever, analyst or metric-tool calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup

from ingestion.sec_chunker import parse_html_to_tree, build_table_chunks

CORPUS = Path("data/evals/retrieval/benchmark_v2")
CONCEPTS = {
    "revenue": {"us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet", "us-gaap:Revenues"},
    "net_income": {"us-gaap:NetIncomeLoss"},
    "operating_income": {"us-gaap:OperatingIncomeLoss"},
    "cash_and_cash_equivalents": {"us-gaap:CashAndCashEquivalentsAtCarryingValue"},
    "total_assets": {"us-gaap:Assets"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_facts(source):
    path = Path(source["source_html"])
    soup = BeautifulSoup(path.read_text(), "xml")
    contexts = {t["id"]: t for t in soup.find_all("context")}
    units = {t["id"]: t.get_text(" ", strip=True) for t in soup.find_all("unit")}
    facts = {}
    for tag in soup.find_all("nonFraction"):
        concept = tag.get("name")
        metric = next((k for k, values in CONCEPTS.items() if concept in values), None)
        context = contexts.get(tag.get("contextRef"))
        if metric is None or context is None or context.find("segment") or context.find("scenario"):
            continue
        instant, start, end = (context.find(k) for k in ("instant", "startDate", "endDate"))
        period_end = (instant or end).get_text() if instant or end else None
        period_start = start.get_text() if start else None
        if not period_end or (period_start and not 330 <= (date.fromisoformat(period_end) - date.fromisoformat(period_start)).days <= 380):
            continue
        unit = units[tag["unitRef"]]
        if unit != "iso4217:USD":
            raise ValueError(f"Unexpected source unit: {unit}")
        literal = tag.get_text().strip()
        value = Decimal(literal.replace(",", "")) * Decimal(10) ** int(tag.get("scale", "0"))
        if tag.get("sign") == "-":
            value = -value
        key = (metric, period_start, period_end, str(value))
        record = facts.setdefault(key, {
            "fact_id": f"{source['ticker']}_{source['fiscal_year']}:{metric}:{period_start or 'instant'}:{period_end}",
            "metric_id": metric, "ticker": source["ticker"],
            "filing_fiscal_year": source["fiscal_year"], "fact_fiscal_year": int(period_end[:4]),
            "form_type": source["form_type"], "value": int(value), "unit": "USD",
            "start_date": period_start, "report_date": period_end,
            "source_html": path.as_posix(), "source_sha256": sha(path), "source_elements": [],
        })
        record["source_elements"].append({"element_id": tag.get("id"), "concept": concept,
            "context_ref": tag["contextRef"], "literal": literal,
            "scale": int(tag.get("scale", "0")), "sign": tag.get("sign", "+")})
    result = sorted(facts.values(), key=lambda r: r["fact_id"])
    if len({r["fact_id"] for r in result}) != len(result):
        raise ValueError("Conflicting source values for one consolidated metric/period")
    return result


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    tables = out / "tables"
    tables.mkdir()
    manifest = json.loads((CORPUS / "corpus_manifest.json").read_text())
    docs = {r["id"]: r for r in map(json.loads, (CORPUS / "corpus.jsonl").read_text().splitlines())}
    facts = []
    sources = []
    for source in manifest["sources"]:
        path = Path(source["source_html"])
        if sha(path) != source["source_sha256"]:
            raise ValueError("Source filing changed")
        facts.extend(extract_facts(source))
        raw_tables = build_table_chunks(parse_html_to_tree(path.read_text()))
        prefix = f"{source['ticker']}_10-K_{source['fiscal_year']}"
        for index, table in enumerate(raw_tables):
            if table.text != docs[f"{prefix}::table::{index}"]["content"]:
                raise ValueError("Hydration sidecar/corpus table mismatch")
        target = tables / f"{prefix}.tables.jsonl"
        target.write_text("".join(json.dumps(asdict(t), sort_keys=True) + "\n" for t in raw_tables))
        sources.append({"source_html": path.as_posix(), "source_sha256": sha(path),
                        "table_sidecar": target.relative_to(out).as_posix(),
                        "table_sha256": sha(target), "tables": len(raw_tables)})
        print(prefix, len(raw_tables), "verified source tables", flush=True)
    (out / "numeric_source_facts.jsonl").write_text("".join(json.dumps(f, sort_keys=True) + "\n" for f in facts))
    (out / "source_manifest.json").write_text(json.dumps({
        "method": "Independent consolidated annual/instant inline-XBRL elements from source HTML; not sec_metric or system results.",
        "corpus_path": CORPUS.as_posix(), "corpus_sha256": sha(CORPUS / "corpus.jsonl"),
        "numeric_facts_sha256": sha(out / "numeric_source_facts.jsonl"), "sources": sources,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    build(parser.parse_args().out_dir)
