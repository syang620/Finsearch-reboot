"""Build a new source-adjudicated version without reading ranked results.

Historical artifacts are hashed for preservation only, never parsed for labels.
Use a new output directory for every draft; the final dataset is written once.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from evals.retrieval_benchmark_v2 import load_dataset as load_parent, read_jsonl, sha256
from evals.retrieval_benchmark_v3 import composition, correlation_metadata, validate
from scripts.evals.retrieval.adjudications_v3 import decisions, REVIEW_NOTES
from scripts.evals.retrieval.inspect_sources_v3 import html_text, normalize, occurrences, section_at

BASE = "25c15afbab31212a42c97e650e2418f6f82a8674"
PARENT = Path("data/evals/retrieval/benchmark_v2")


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def table_span(document):
    content = document["content"]
    # Exact complete table is manually inspectable, including adjacent row labels.
    return [{"start": 0, "end": len(content), "quote": content, "anchor": content.splitlines()[0]}]


def new_judgment(decision, document):
    metadata = document["metadata"]
    if "quote" in decision:
        quote = decision["quote"]
        if document["content"].count(quote) != 1:
            raise ValueError("Explicit source quotation needs adjudication")
        start = document["content"].index(quote)
        spans = [{"start": start, "end": start + len(quote), "quote": quote, "anchor": quote}]
        context = decision.get("context_anchor")
        if context:
            if document["content"].count(context) != 1:
                raise ValueError("Explicit policy-subject context needs adjudication")
            start = document["content"].index(context)
            spans.append({"start": start, "end": start + len(context), "quote": context, "anchor": context})
    else:
        spans = table_span(document)
    cells = []
    for value in decision.get("source_cell_values", []):
        rows = [(i, line) for i, line in enumerate(document["content"].splitlines()) if value in line]
        if len(rows) != 1:
            raise ValueError("Explicit numeric row needs adjudication")
        i, line = rows[0]
        cells.append({"line_index": i, "row_text": line, "literal": value,
                      "fiscal_year": decision["fiscal_year"], "unit": decision["unit"]})
    return {"evidence_id": document["id"], "grade": decision["grade"], "groups": decision["groups"],
        "spans": spans, "content_sha256": document["content_sha256"],
        **{k: metadata[k] for k in ("source_html", "source_sha256", "section_path")},
        "adjudicated_cells": cells, "adjudication_reason": decision["reason"],
        "reason_code": decision["reason_code"]}


def provenance(judgment, source_text, sections, tables):
    table = tables.get(judgment["evidence_id"])
    if table:
        return [{"start": table["source_normalized_start"], "end": table["source_normalized_end"],
                 "quote_normalized": table["source_normalized_table"], "item": table["canonical_item"],
                 "binding": "original_html_table", "table_html_sha256": table["table_html_sha256"]}]
    # Source tables retain exact HTML provenance. Text paragraphs occasionally
    # omit source page furniture; in those cases bind the literal authored anchor
    # and report the narrower binding rather than claim whole-paragraph identity.
    result = []
    for span in judgment["spans"]:
        quote = normalize(span["quote"])
        positions = occurrences(quote, source_text)
        binding = "whole_normalized_quote"
        if not positions:
            quote = normalize(span["anchor"])
            positions = occurrences(quote, source_text)
            binding = "literal_source_anchor_page_furniture_differs"
        if not positions:
            raise ValueError("Judgment absent from original filing")
        for position in positions:
            # A short incidental year anchor alone does not locate a section;
            # an 80-character exact context shared by source and chunk can.
            located_quote = quote
            located_binding = binding
            if len(quote) < 20:
                located_quote = source_text[position:position + 80]
                if len(located_quote) != 80 or located_quote not in normalize(span["quote"]):
                    continue
                located_binding = "exact_shared_80_character_source_context"
            result.append({"start": position, "end": position + len(located_quote),
                           "quote_normalized": located_quote, "item": section_at(sections, position), "binding": located_binding})
    if not result:
        raise ValueError("No substantive source binding")
    return sorted({json.dumps(r, sort_keys=True): r for r in result}.values(), key=lambda r: (r["start"], r["end"]))


def build(out, inspection):
    if out.exists():
        raise FileExistsError("Do not overwrite a draft or frozen dataset")
    old_cases, documents, _ = load_parent(PARENT)
    docmap = {d["id"]: d for d in documents}
    tables = {t["evidence_id"]: t for t in json.loads((inspection / "tables.json").read_text())}
    sections = {s["source_html"]: s for s in json.loads((inspection / "sections.json").read_text())}
    source_text = {p: normalize(html_text(Path(p).read_text())) for p in sections}
    import hashlib
    for path, entry in sections.items():
        if sha256(Path(path)) != entry["source_sha256"] or hashlib.sha256(source_text[path].encode()).hexdigest() != entry["normalized_text_sha256"]:
            raise ValueError("Inspection/source mismatch")
    additions = decisions()
    cases, adjudications, changes = [], [], []
    for original in old_cases:
        case = deepcopy(original)
        case["parent_case_id"] = original["id"]
        case["id"] = original["id"].replace("KBV2_", "KBV3_", 1)
        case["filing_id"] = f'{case["ticker"]}_{case["fiscal_year"]}'
        family = original["id"].split("_")[-1]
        if case["ticker"] == "MSFT" and case["fiscal_year"] == 2025 and family in {"04", "17"}:
            family += "_goodwill"
        case["family_id"] = f'{case["ticker"]}:q{family}'
        case["annotation"] = "v3 source-inspected assistant adjudication; not human-blinded or exhaustive relevance labels; no ranking-based selection"
        for judgment in case["judgments"]:
            judgment["reason_code"] = "RETAIN_SOURCE_SUPPORTED_PARENT_JUDGMENT"
            judgment["adjudication_reason"] = REVIEW_NOTES[case["stratum"]]
        for decision in additions:
            if decision["parent_case_id"] != original["id"]:
                continue
            previous = next((j for j in original["judgments"] if j["evidence_id"] == decision["evidence_id"]), None)
            if previous is not None:
                raise ValueError("Expected formerly unjudged evidence; do not overwrite")
            case["judgments"].append(new_judgment(decision, docmap[decision["evidence_id"]]))
            adjudications.append({**decision, "case_id": case["id"], "previous_judgment": None,
                "previous_status": "unjudged_scored_zero_not_proven_irrelevant",
                "previous_gold_ids": [j["evidence_id"] for j in original["judgments"] if j["grade"] == 2],
                "reviewer": "source-inspecting coding assistant; benchmark-quality review required before comparison"})
        for judgment in case["judgments"]:
            path = judgment["source_html"]
            locations = provenance(judgment, source_text[path], sections[path]["sections"], tables)
            if any(source_text[path][loc["start"]:loc["end"]] != loc["quote_normalized"] for loc in locations):
                raise ValueError("Incorrect original source offset")
            judgment["source_locations"] = locations
            judgment["canonical_items"] = sorted({loc["item"] for loc in locations})
        case["judgments"].sort(key=lambda j: j["evidence_id"])
        cases.append(case)
    components = correlation_metadata(cases)
    for case in cases:
        case["correlation_component"] = components.get(case["id"])
    for old, new in zip(old_cases, cases):
        old_labels = {j["evidence_id"]: j for j in old["judgments"]}
        for j in new["judgments"]:
            changes.append({"parent_case_id": old["id"], "case_id": new["id"],
                "evidence_id": j["evidence_id"], "reason_code": j["reason_code"],
                "change_kind": "label_added" if j["evidence_id"] not in old_labels else "provenance_enriched_grade_and_groups_unchanged",
                "before": old_labels.get(j["evidence_id"]), "after": j})
    validate(cases, documents)
    out.mkdir(parents=True)
    with (out / "queries.jsonl").open("x") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "label_changes.jsonl").open("x") as stream:
        for change in changes:
            stream.write(json.dumps(change, ensure_ascii=False, sort_keys=True) + "\n")
    for adjudication in adjudications:
        new = next(c for c in cases if c["id"] == adjudication["case_id"])
        adjudication["corrected_gold_ids"] = [j["evidence_id"] for j in new["judgments"] if j["grade"] == 2]
    save(out / "adjudications.json", adjudications)
    save(out / "source_sections.json", list(sections.values()))
    source_manifest = json.loads((PARENT / "corpus_manifest.json").read_text())
    save(out / "corpus_ref.json", {"path": str(PARENT / "corpus.jsonl"),
        "sha256": sha256(PARENT / "corpus.jsonl"), "reuse": "byte-identical historical corpus; no extraction/index change",
        "sources": source_manifest["sources"]})
    with (out / "comparison_config.json").open("xb") as stream:
        stream.write((PARENT / "comparison_config.json").read_bytes())
    parent_files = {p.name: sha256(p) for p in sorted(PARENT.iterdir()) if p.is_file()}
    save(out / "lineage.json", {
        "parent_benchmark": "sec_retrieval_benchmark_v2", "parent_files_sha256": parent_files,
        "new_benchmark": "sec_retrieval_benchmark_v3", "new_queries_sha256": sha256(out / "queries.jsonl"),
        "membership_additions": [], "membership_removals": [],
        "identity_mapping": [{"old": c["parent_case_id"], "new": c["id"], "reason_code": "EXPLICIT_VERSION_ID_MIGRATION"} for c in cases],
        "membership_reason": "Preserve all correlated parent cases and exclusions; no outcome-based selection",
        "grade_changes_on_previously_judged_ids": 0, "new_judgments": len(additions),
        "audit_head": "f190180c2e0e48d10a95f13d9d81082ca214d029",
        "audit_findings": ["B1", "B5"], "label_diff": "label_changes.jsonl",
        "metadata_reason_codes": ["SOURCE_VERIFIED_SECTION_PROVENANCE", "CORRELATED_FAMILY_AND_SHARED_EVIDENCE_GROUPING"],
        "scope_decision": "retain 3 issuers / 6 filings; no sector-diversity or unseen-holdout claim",
    })
    historical = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE, "--", "data/evals", "artifacts/evals"], text=True).splitlines()
    save(out / "historical_sha256.json", {p: sha256(Path(p)) for p in historical})
    manifest = {"benchmark_id": "sec_retrieval_benchmark_v3", "version": 3, "base_commit": BASE,
        "status": "annotation_candidate_requires_benchmark_quality_review_before_comparison",
        "composition": composition(cases, documents),
        "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
        "construction_sha256": {p: sha256(Path(p)) for p in [
            "scripts/evals/retrieval/build_benchmark_v3.py", "scripts/evals/retrieval/adjudications_v3.py",
            "scripts/evals/retrieval/inspect_sources_v3.py", "src/evals/retrieval_benchmark_v3.py"]},
        "freeze_rule": "Commit immutable candidate hashes before benchmark-quality review. Separate approval binds exact dataset hash. No comparisons until approval.",
    }
    save(out / "dataset_manifest.json", manifest)
    print(json.dumps({"dataset_sha256": sha256(out / "queries.jsonl"), "manifest_sha256": sha256(out / "dataset_manifest.json"),
                      "composition": manifest["composition"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, default=Path(".cache/retrieval_v3_source_inspection"))
    args = parser.parse_args()
    build(args.out, args.inspection)
