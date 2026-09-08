"""Source-only provenance preparation for v3; never imports retrieval ranking.

Derived inspection material is written exclusively to a new output directory.
The unchanged v2 corpus is verified, not rebuilt or modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ingestion.sec_chunker import parse_html_to_tree, build_table_chunks

PARENT = Path("data/evals/retrieval/benchmark_v2")
HEADINGS = {
    "1": "business", "1A": "riskfactors", "1B": "unresolvedstaffcomments",
    "1C": "cybersecurity", "2": "properties", "3": "legalproceedings",
    "5": "marketfor", "7": "management", "7A": "quantitative",
    "8": "financialstatements", "9": "changesin",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    # Typography normalization for SOURCE LOCATIONS only, not relevance inference.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def html_text(html):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        return BeautifulSoup(html, "lxml").get_text(" ")


def occurrences(needle, text):
    return [m.start() for m in re.finditer(re.escape(needle), text)] if needle else []


def source_sections(text):
    sections = []
    for item, title in HEADINGS.items():
        matches = list(re.finditer("item" + item + r"\.?" + title, text, re.I))
        # All six source filings were inspected: first is table of contents,
        # second is the actual section heading. Fail if this ceases to be true.
        if len(matches) != 2:
            raise ValueError(f"Source heading needs adjudication: Item {item}")
        match = matches[1]
        sections.append({"item": item, "start": match.start(), "heading_anchor": match.group()})
    sections.sort(key=lambda s: s["start"])
    for index, section in enumerate(sections):
        section["end"] = sections[index + 1]["start"] if index + 1 < len(sections) else len(text)
    return sections


def section_at(sections, position):
    return next((s["item"] for s in sections if s["start"] <= position < s["end"]), "front_matter")


def prepare(out):
    if out.exists():
        raise FileExistsError("Inspection output already exists; do not overwrite.")
    manifest = json.loads((PARENT / "corpus_manifest.json").read_text())
    if digest(PARENT / "corpus.jsonl") != manifest["corpus_sha256"]:
        raise ValueError("Historical corpus changed")
    documents = {d["id"]: d for d in map(json.loads, (PARENT / "corpus.jsonl").read_text().splitlines())}
    records, source_records = [], []
    for source in manifest["sources"]:
        path = Path(source["source_html"])
        if digest(path) != source["source_sha256"]:
            raise ValueError("Historical filing changed")
        html = path.read_text()
        text = normalize(html_text(html))
        sections = source_sections(text)
        tables = build_table_chunks(parse_html_to_tree(html))
        prefix = f'{source["ticker"]}_10-K_{source["fiscal_year"]}'
        if len(tables) != source["tables"]:
            raise ValueError("Table extraction count drift")
        for index, table in enumerate(tables):
            identifier = f"{prefix}::table::{index}"
            document = documents[identifier]
            if document["content"] != table.text:
                raise ValueError("Table representation drift")
            raw = normalize(html_text(table.table_html))
            positions = occurrences(raw, text)
            if len(positions) != 1:
                raise ValueError(f"Ambiguous original HTML table: {identifier}: {len(positions)}")
            start = positions[0]
            records.append({"evidence_id": identifier, "source_html": str(path),
                "source_sha256": source["source_sha256"], "content_sha256": document["content_sha256"],
                "source_normalized_start": start, "source_normalized_end": start + len(raw),
                "canonical_item": section_at(sections, start), "source_normalized_table": raw,
                "table_html_sha256": hashlib.sha256(table.table_html.encode()).hexdigest()})
        source_records.append({"source_html": str(path), "source_sha256": source["source_sha256"],
            "normalized_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "sections": sections})
        print(json.dumps({"source": str(path), "tables_verified": len(tables)}), flush=True)
    out.mkdir(parents=True)
    for name, value in [("tables.json", records), ("sections.json", source_records)]:
        with (out / name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
