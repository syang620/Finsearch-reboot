"""Versioned known-label contracts and correlation-aware reporting; no ranking."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from evals.retrieval_benchmark_v2 import (
    METRICS, aggregate, metrics, read_jsonl, sha256, validate as validate_parent,
)

PARENT = Path("data/evals/retrieval/benchmark_v2")


def normalized_question(text):
    import re
    return re.sub(r"20\d{2}", "YEAR", " ".join(text.split()))


def correlation_metadata(cases):
    """Conservative components join year-paired families and shared positives."""
    parent = {c["id"]: c["id"] for c in cases if c["status"] == "answerable"}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for case in sorted(cases, key=lambda c: c["id"]):
        if case["status"] != "answerable":
            continue
        keys = ["family:" + case["family_id"]]
        keys += ["evidence:" + j["evidence_id"] for j in case["judgments"] if j["grade"] == 2]
        for key in keys:
            if key in owners:
                left, right = sorted([find(case["id"]), find(owners[key])])
                parent[right] = left
            else:
                owners[key] = case["id"]
    return {identifier: "component:" + find(identifier) for identifier in sorted(parent)}


def composition(cases, docs):
    scored = [c for c in cases if c["status"] == "answerable"]
    positives = [j for c in scored for j in c["judgments"] if j["grade"] == 2]
    docmap = {d["id"]: d for d in docs}
    return {
        "queries": len(cases), "answerable": len(scored), "excluded": len(cases) - len(scored),
        "documents": len(docs), "issuers": dict(sorted(Counter(c["ticker"] for c in scored).items())),
        "filings": dict(sorted(Counter(c["filing_id"] for c in scored).items())),
        "strata": dict(sorted(Counter(c["stratum"] for c in scored).items())),
        "normalized_question_families": len({normalized_question(c["query"]) for c in scored}),
        "declared_topic_families": len({c["family_id"] for c in scored}),
        "correlation_components": len(set(correlation_metadata(cases).values())),
        "unique_filing_evidence_groups": len({(c["filing_id"], g) for c in scored for g in c["required_groups"]}),
        "unique_positive_documents": len({j["evidence_id"] for j in positives}),
        "grade_counts": dict(sorted(Counter(str(j["grade"]) for c in scored for j in c["judgments"]).items())),
        "positive_doc_types": dict(sorted(Counter(docmap[j["evidence_id"]]["metadata"]["doc_type"] for j in positives).items())),
        "canonical_positive_items": dict(sorted(Counter(item for j in positives for item in set(j["canonical_items"])).items())),
        "section_count_unit": "positive query/document/item memberships; multi-location text may count in multiple items",
    }


def validate(cases, docs):
    result = validate_parent(cases, docs)
    parent_cases = {c["id"]: c for c in read_jsonl(PARENT / "queries.jsonl")}
    if len(cases) != len(parent_cases) or {c["parent_case_id"] for c in cases} != set(parent_cases):
        raise ValueError("v3 membership must preserve every parent case exactly once")
    docmap = {d["id"]: d for d in docs}
    components = correlation_metadata(cases)
    for case in cases:
        old = parent_cases[case["parent_case_id"]]
        if case["id"] != old["id"].replace("KBV2_", "KBV3_", 1):
            raise ValueError("Invalid version identity mapping")
        if any(case[k] != old[k] for k in ("query", "ticker", "fiscal_year", "form_type", "status", "stratum", "required_groups")):
            raise ValueError("Unexpected membership/semantic scope change")
        if case["filing_id"] != f'{case["ticker"]}_{case["fiscal_year"]}' or not case["family_id"]:
            raise ValueError("Missing correlation metadata")
        if case["status"] == "answerable" and case["correlation_component"] != components[case["id"]]:
            raise ValueError("Incorrect evidence/family component")
        for judgment in case["judgments"]:
            if not judgment["canonical_items"] or not judgment["source_locations"]:
                raise ValueError("Missing independently located source provenance")
            if not judgment.get("adjudication_reason"):
                raise ValueError("Missing adjudication rationale")
            if judgment["grade"] == 2 and not set(judgment["groups"]) <= set(case["required_groups"]):
                raise ValueError("Positive assigned to unknown required group")
            content = docmap[judgment["evidence_id"]]["content"]
            for cell in judgment.get("adjudicated_cells", []):
                if content.splitlines()[cell["line_index"]] != cell["row_text"] or cell["literal"] not in cell["row_text"]:
                    raise ValueError("Adjudicated cell does not exist")
                if cell["fiscal_year"] != case["fiscal_year"] or cell["unit"] != "USD_millions":
                    raise ValueError("Adjudicated cell scope mismatch")
    return result


def safe_relative(path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe manifest path")
    return path


def load_dataset(root=Path("data/evals/retrieval/benchmark_v3")):
    root = Path(root)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    if manifest["benchmark_id"] != "sec_retrieval_benchmark_v3" or manifest["version"] != 3:
        raise ValueError("Wrong benchmark version")
    for name, expected in manifest["files"].items():
        if sha256(root / safe_relative(name)) != expected:
            raise ValueError(f"Frozen v3 hash mismatch: {name}")
    lineage = json.loads((root / "lineage.json").read_text())
    if lineage["parent_benchmark"] != "sec_retrieval_benchmark_v2":
        raise ValueError("Wrong lineage")
    for name, expected in lineage["parent_files_sha256"].items():
        if sha256(PARENT / safe_relative(name)) != expected:
            raise ValueError("Historical v2 changed")
    if sha256(root / "queries.jsonl") != lineage["new_queries_sha256"]:
        raise ValueError("Lineage child hash mismatch")
    corpus_ref = json.loads((root / "corpus_ref.json").read_text())
    if sha256(safe_relative(corpus_ref["path"])) != corpus_ref["sha256"]:
        raise ValueError("Historical corpus changed")
    for source in corpus_ref["sources"]:
        if sha256(safe_relative(source["source_html"])) != source["source_sha256"]:
            raise ValueError("Source filing changed")
    cases, docs = read_jsonl(root / "queries.jsonl"), read_jsonl(Path(corpus_ref["path"]))
    validate(cases, docs)
    verify_source_locations(cases, json.loads((root / "source_sections.json").read_text()))
    if composition(cases, docs) != manifest["composition"]:
        raise ValueError("Composition metadata mismatch")
    return cases, docs, manifest


def verify_source_locations(cases, source_sections):
    import hashlib
    import re
    import unicodedata
    import warnings
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    texts, sections = {}, {}
    for source in source_sections:
        path = str(safe_relative(source["source_html"]))
        if sha256(Path(path)) != source["source_sha256"]:
            raise ValueError("Source provenance hash mismatch")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            text = BeautifulSoup(Path(path).read_text(), "lxml").get_text(" ")
        text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
        if hashlib.sha256(text.encode()).hexdigest() != source["normalized_text_sha256"]:
            raise ValueError("Source normalization drift")
        texts[path], sections[path] = text, source["sections"]
        for index, section in enumerate(source["sections"]):
            start, end = section["start"], section["end"]
            if not 0 <= start < end <= len(text) or not text[start:].startswith(section["heading_anchor"]):
                raise ValueError("Invalid canonical source heading")
            if index and source["sections"][index - 1]["end"] != start:
                raise ValueError("Non-contiguous source sections")
    for case in cases:
        for judgment in case["judgments"]:
            path = judgment["source_html"]
            text = texts[path]
            for loc in judgment["source_locations"]:
                start, end = loc["start"], loc["end"]
                if not 0 <= start < end <= len(text) or text[start:end] != loc["quote_normalized"]:
                    raise ValueError("Invalid source evidence span")
                item = next((s["item"] for s in sections[path] if s["start"] <= start < s["end"]), "front_matter")
                if item != loc["item"]:
                    raise ValueError("Incorrect canonical source item")
            if judgment["canonical_items"] != sorted({loc["item"] for loc in judgment["source_locations"]}):
                raise ValueError("Canonical item membership mismatch")


def verify_history(root):
    paths = json.loads((Path(root) / "historical_sha256.json").read_text())
    for path, expected in paths.items():
        if sha256(safe_relative(path)) != expected:
            raise ValueError(f"Historical dataset/artifact changed: {path}")
    return {"historical_files_verified": len(paths)}


def classify_results(predicted, case, docs):
    corpus = {d["id"]: d for d in docs}
    labels = {j["evidence_id"]: j["grade"] for j in case["judgments"]}
    return {
        "duplicate_returned_ids": len(predicted) - len(set(predicted)),
        "missing_corpus_ids": sum(i not in corpus for i in predicted),
        "unjudged_returned_ids": sum(i in corpus and i not in labels for i in predicted),
        "explicit_irrelevant_returned_ids": sum(labels.get(i) == 0 for i in predicted),
        "incompatible_filter_ids": sum(i in corpus and any(corpus[i]["metadata"][k] != case[k]
                                          for k in ("ticker", "fiscal_year", "form_type")) for i in predicted),
    }


def grouped_summary(rows, cases, key):
    """Equal cluster weight, not a claim that clusters are independent samples."""
    case_map = {c["id"]: c for c in cases}
    groups = defaultdict(list)
    for row in rows:
        groups[case_map[row["id"]][key]].append(row)
    group_metrics = {g: {m: statistics.mean(r["metrics"][m] for r in rs) for m in METRICS}
                     for g, rs in sorted(groups.items())}
    return {
        "grouping": key, "groups": len(groups), "queries": len(rows),
        "group_sizes": {g: len(rs) for g, rs in sorted(groups.items())},
        "macro_metrics": {m: statistics.mean(v[m] for v in group_metrics.values()) if groups else None for m in METRICS},
        "by_group": group_metrics,
        "interpretation": "equal-group descriptive macro-average; no iid confidence interval or effective independent sample-size claim",
    }


def validate_pairs(rows, cases, modes, complete=True):
    expected = {(c["id"], m) for c in cases if c["status"] == "answerable" for m in modes}
    pairs = [(r["id"], r["mode"]) for r in rows]
    if len(pairs) != len(set(pairs)) or not set(pairs) <= expected:
        raise ValueError("Duplicate or unknown case/mode pair")
    if complete and set(pairs) != expected:
        raise ValueError("Incomplete frozen comparison")
    return {"expected_pairs": len(expected), "completed_pairs": len(pairs)}


def summarize(rows, cases, modes, complete=True):
    coverage = validate_pairs(rows, cases, modes, complete)
    case_map = {c["id"]: c for c in cases}
    result = {**coverage, "complete": complete, "modes": {}}
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        result["modes"][mode] = {
            "overall": aggregate(subset),
            "by_stratum": {s: aggregate([r for r in subset if case_map[r["id"]]["stratum"] == s])
                           for s in sorted({c["stratum"] for c in cases if c["status"] == "answerable"})},
            "by_issuer": {t: aggregate([r for r in subset if case_map[r["id"]]["ticker"] == t])
                          for t in sorted({c["ticker"] for c in cases})},
            "grouped": {key: grouped_summary(subset, cases, key)
                        for key in ("family_id", "correlation_component", "filing_id", "ticker")},
        }
    return result
