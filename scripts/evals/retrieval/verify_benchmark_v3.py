"""Offline verification only; never contacts ranking/model services."""
import argparse
import json
from pathlib import Path

from evals.retrieval_benchmark_v3 import (
    METRICS, classify_results, load_dataset, metrics, read_jsonl, safe_relative,
    sha256, summarize, verify_history, validate_pairs,
)
from scripts.evals.retrieval.verify_benchmark_v2 import values_match


def verify(dataset, baseline):
    cases, docs, _ = load_dataset(dataset)
    manifest = json.loads((baseline / "manifest.json").read_text())
    recorded = json.loads((baseline / "summary.json").read_text())
    if baseline.name != manifest["implementation_sha"]:
        raise ValueError("Baseline is not implementation-SHA keyed")
    if manifest["dataset_manifest_sha256"] != sha256(dataset / "dataset_manifest.json"):
        raise ValueError("Dataset/baseline mismatch")
    for name, digest in manifest["raw_sha256"].items():
        if sha256(baseline / safe_relative(name)) != digest:
            raise ValueError("Raw artifact hash mismatch")
    for name, digest in manifest["source_sha256"].items():
        if sha256(safe_relative(name)) != digest:
            raise ValueError("Evaluated source changed")
    approval = manifest["annotation_approval"]
    if approval["status"] != "approved_for_narrow_known_label_baseline" or approval["dataset_manifest_sha256"] != manifest["dataset_manifest_sha256"] or approval["reviewed_commit"] != manifest["dataset_freeze_sha"]:
        raise ValueError("Invalid annotation approval binding")
    case_map = {c["id"]: c for c in cases}
    rows = read_jsonl(baseline / "per_query.jsonl")
    modes = manifest["config"]["modes"]
    if manifest["config"] != json.loads((dataset / "comparison_config.json").read_text()):
        raise ValueError("Frozen comparison configuration changed")
    validate_pairs(rows, cases, modes, complete=recorded["complete"])
    for row in rows:
        case = case_map[row["id"]]
        if any(row[k] != case[k] for k in ("query", "stratum", "ticker", "fiscal_year")):
            raise ValueError("Case metadata mismatch")
        if any(row[k] != v for k, v in classify_results(row["ranked_ids"], case, docs).items()):
            raise ValueError("Unknown/negative/missing-ID counts mismatch")
        expected = metrics(row["ranked_ids"], case) if row["error"] is None else {m: 0.0 for m in METRICS}
        if not values_match(expected, row["metrics"]):
            raise ValueError("Per-query metric mismatch")
    actual = summarize(rows, cases, modes, complete=recorded["complete"])
    for key, value in actual.items():
        if not values_match(value, recorded[key]):
            raise ValueError("Aggregate/group metrics mismatch")
    if recorded["complete"] and (manifest["fatal_error"] is not None or not manifest["indexes_unchanged"] or not manifest["embedding_model_unchanged"]):
        raise ValueError("Invalid completion claim")
    if manifest["indexes_unchanged"] != (manifest["index_before"] == manifest["index_after"] and manifest["historical_index_before"] == manifest["historical_index_after"]):
        raise ValueError("Invalid index preservation claim")
    return {"verified": True, "complete": recorded["complete"], "pairs": len(rows),
            "dataset_sha256": sha256(dataset / "queries.jsonl"), **verify_history(dataset)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/evals/retrieval/benchmark_v3"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.dataset, args.baseline)
    if args.out:
        with args.out.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
    print(json.dumps(result, sort_keys=True))
