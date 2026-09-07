"""Recompute scores and source-audit agreement without model calls or rewriting evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

from evals.semantic_answer_v1 import (deterministic_case, load_dataset, rate, read_jsonl,
    sha, summarize_deterministic, summarize_semantic, validate_judgment)
from scripts.evals.agents.run_semantic_v1 import DATA, save

def verify(out):
    cases, counts = load_dataset(DATA); case_map = {c["id"]: c for c in cases}
    manifest = json.loads((out / "answer_manifest.json").read_text())
    if out.name != manifest["implementation_sha"]: raise ValueError("Not implementation-SHA keyed")
    if manifest["dataset_manifest_sha256"] != sha(DATA / "manifest.json"): raise ValueError("Dataset identity changed")
    if manifest["status"] != "complete" or not manifest["index_unchanged"]: raise ValueError("Incomplete/changed-index baseline")
    for name, digest in manifest["files_sha256"].items():
        if sha(out / name) != digest: raise ValueError("Answer artifact hash mismatch")
    for name, digest in manifest["source_sha256"].items():
        if sha(name) != digest: raise ValueError("Source/evaluator changed")
    records = read_jsonl(out / "raw_answers.jsonl")
    ids = [r["case_id"] for r in records]
    if len(ids) != len(cases) or set(ids) != set(case_map) or ids != manifest["schedule"]: raise ValueError("Case/order coverage mismatch")
    outputs = {r["case_id"]: r["output"] for r in records}
    rows = [deterministic_case(case_map[r["case_id"]], r["output"]) for r in records]
    if rows != read_jsonl(out / "deterministic.jsonl"): raise ValueError("Deterministic scores not reproducible")
    expected = {"overall": summarize_deterministic(rows), "by_stratum": {s: summarize_deterministic([r for r in rows if r["stratum"]==s]) for s in sorted({c["stratum"] for c in cases})}}
    if expected != json.loads((out / "deterministic_summary.json").read_text()): raise ValueError("Deterministic aggregates not reproducible")
    audit_path = out / "source_audit.jsonl"
    audit = read_jsonl(audit_path)
    audit_ids = [r["case_id"] for r in audit]
    if len(audit_ids) != len(set(audit_ids)) or set(audit_ids) != {c["id"] for c in cases if c["audit_selected"]}: raise ValueError("Frozen audit coverage mismatch")
    audit_judgments = {r["case_id"]: validate_judgment(case_map[r["case_id"]], outputs[r["case_id"]], r["judgment"]) for r in audit}
    judge_dir = out / "judge"
    judge_manifest = json.loads((judge_dir / "manifest.json").read_text())
    for name, digest in judge_manifest["files_sha256"].items():
        if sha(judge_dir / name) != digest: raise ValueError("Judge artifact hash mismatch")
    if json.loads((judge_dir / "started.json").read_text())["audit_sha256"] != sha(audit_path): raise ValueError("Audit changed after judge began")
    judgments = read_jsonl(judge_dir / "judgments.jsonl")
    if [j["case_id"] for j in judgments] != ids: raise ValueError("Judge case/order coverage mismatch")
    valid = {r["case_id"]: validate_judgment(case_map[r["case_id"]], outputs[r["case_id"]], r["judgment"]) for r in judgments if r["status"]=="valid"}
    semantic = {"overall": summarize_semantic(cases, outputs, valid), "by_stratum": {s: summarize_semantic([c for c in cases if c["stratum"]==s], outputs, valid) for s in sorted({c["stratum"] for c in cases})}}
    if semantic != json.loads((judge_dir / "summary.json").read_text()): raise ValueError("Semantic aggregates not reproducible")
    flags = {r["case_id"]: {c["claim_id"]: bool(c["flags"]) for c in r["claim_checks"]} for r in rows}
    detection, confusion = Counter(), Counter()
    disagreements = []
    for cid, reviewed in audit_judgments.items():
        predicted = {r["claim_id"]: r for r in valid.get(cid, {}).get("claims", [])}
        for claim in reviewed["claims"]:
            truth = claim["support"] == "unsupported"; detected = flags[cid][claim["claim_id"]]
            detection["tp" if truth and detected else "fn" if truth else "fp" if detected else "tn"] += 1
            other = predicted.get(claim["claim_id"])
            if other:
                confusion[claim["support"] + " -> " + other["support"]] += 1
                if claim["support"] != other["support"]:
                    disagreements.append({"case_id": cid, "claim_id": claim["claim_id"], "audit": claim, "judge": other})
        if cid in valid:
            for key in ("answer_relevant", "answerability_correct", "unbound_factual_prose"):
                if reviewed[key] != valid[cid][key]: disagreements.append({"case_id": cid, "field": key, "audit": reviewed[key], "judge": valid[cid][key]})
            predicted_required = {r["claim_id"]: r for r in valid[cid]["requirements"]}
            for required in reviewed["requirements"]:
                other = predicted_required[required["claim_id"]]
                if required["fulfillment"] != other["fulfillment"]: disagreements.append({"case_id": cid, "gold_claim_id": required["claim_id"], "audit": required, "judge": other})
    total = sum(confusion.values()); matched = sum(v for k, v in confusion.items() if k.split(" -> ")[0]==k.split(" -> ")[1])
    return {"verified": True, "validation": counts, "implementation_sha": manifest["implementation_sha"],
        "dataset_sha256": sha(DATA/"queries.jsonl"), "deterministic": expected,
        "secondary_semantic": semantic, "judge_status_counts": dict(Counter(r["status"] for r in judgments)),
        "source_audit": {"reviewer": "source-inspecting assistant; not independent human adjudication", "cases": len(audit),
            "metrics": summarize_semantic([c for c in cases if c["audit_selected"]], outputs, audit_judgments),
            "judge_claim_agreement": rate(matched, total), "confusion_audit_to_judge": dict(confusion),
            "structural_unsupported_detection": {**{k: detection[k] for k in ("tp", "tn", "fp", "fn")},
                "precision": rate(detection["tp"], detection["tp"]+detection["fp"]),
                "recall": rate(detection["tp"], detection["tp"]+detection["fn"]),
                "note": "Positive truth is audited unsupported; partially supported is separate, not relabeled unsupported. Null means no applicable denominator."},
            "disagreements": disagreements}}

def main():
    p = argparse.ArgumentParser(); p.add_argument("baseline", type=Path); p.add_argument("--out", type=Path)
    a = p.parse_args(); result = verify(a.baseline)
    if a.out: save(a.out, result)
    print(json.dumps({"verified": result["verified"], "cases": result["validation"]["cases"], "judge_status_counts": result["judge_status_counts"]}, indent=2))

if __name__ == "__main__": main()
