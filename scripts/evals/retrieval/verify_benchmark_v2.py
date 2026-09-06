#!/usr/bin/env python3
"""Read-only verification of baseline hashes, pair coverage and metric arithmetic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.retrieval_benchmark_v2 import METRICS,aggregate,load_dataset,metrics,read_jsonl,sha256


def verify(dataset: Path,baseline: Path):
    cases,_,validation=load_dataset(dataset)
    by_id={c["id"]:c for c in cases if c["status"]=="answerable"}
    manifest=json.loads((baseline/"manifest.json").read_text())
    summary=json.loads((baseline/"summary.json").read_text())
    if manifest["dataset_manifest_sha256"]!=sha256(dataset/"dataset_manifest.json"):
        raise ValueError("Wrong dataset manifest")
    for name,digest in manifest["raw_sha256"].items():
        if Path(name).name!=name or sha256(baseline/name)!=digest:
            raise ValueError("Artifact hash mismatch")
    rows=read_jsonl(baseline/"per_query.jsonl")
    pairs=[(r["id"],r["mode"]) for r in rows]
    expected={(i,m) for i in by_id for m in manifest["config"]["modes"]}
    if len(pairs)!=len(set(pairs)) or not set(pairs)<=expected:
        raise ValueError("Duplicate or unexpected query-mode pair")
    if summary["complete"] and set(pairs)!=expected:
        raise ValueError("Incomplete pair coverage")
    if summary["completed_pairs"]!=len(rows) or summary["expected_pairs"]!=len(expected):
        raise ValueError("Pair count mismatch")
    for r in rows:
        c=by_id[r["id"]]
        if any(r[k]!=c[k] for k in ("query","stratum","ticker","fiscal_year")):
            raise ValueError("Case metadata mismatch")
        actual=metrics(r["ranked_ids"],c) if r["error"] is None else {k:0.0 for k in METRICS}
        if actual!=r["metrics"]:raise ValueError("Per-query metric mismatch")
    for mode,reported in summary["modes"].items():
        subset=[r for r in rows if r["mode"]==mode]
        if aggregate(subset)!=reported["overall"]:raise ValueError("Overall metric mismatch")
        for stratum,value in reported["by_stratum"].items():
            if aggregate([r for r in subset if r["stratum"]==stratum])!=value:
                raise ValueError("Stratum metric mismatch")
        for company,value in reported["by_company"].items():
            if aggregate([r for r in subset if r["ticker"]==company])!=value:
                raise ValueError("Company metric mismatch")
    return {"pairs":len(rows),"complete":summary["complete"],"missing_labels":validation["missing_labels"],"hashes_and_metrics_verified":True}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=Path("data/evals/retrieval/benchmark_v2"))
    p.add_argument("--baseline",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.dataset,a.baseline)))
