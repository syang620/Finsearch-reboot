"""Evaluation-only contracts and metrics; no production retrieval changes."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

METRICS=("recall@5","recall@10","mrr@10","ndcg@5","ndcg@10","evidence_group_recall@10")
STRATA={"direct_fact","narrative","risk_factors","business_growth","mda",
        "paraphrase","section_specific","hard_negative","multi_evidence","unanswerable"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result=[]
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at line {number}")
        value=json.loads(line)
        if not isinstance(value,dict):
            raise ValueError(f"Non-object JSONL record at line {number}")
        result.append(value)
    return result


def validate(cases: list[dict], documents: list[dict]) -> dict:
    for label,rows,key in [("query",cases,"id"),("corpus",documents,"id")]:
        ids=[r[key] for r in rows]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
            raise ValueError(f"Duplicate or empty {label} IDs")
    docs={d["id"]:d for d in documents}
    for d in documents:
        if hashlib.sha256(d["content"].encode()).hexdigest()!=d["content_sha256"]:
            raise ValueError("Corpus content hash mismatch")
    for c in cases:
        if c["stratum"] not in STRATA or not c.get("query","").strip():
            raise ValueError("Invalid stratum/query")
        if c["status"] not in {"answerable","excluded"}:
            raise ValueError("Invalid case status")
        ids=[j["evidence_id"] for j in c["judgments"]]
        if len(ids)!=len(set(ids)):
            raise ValueError("Duplicate judgment IDs")
        if c["status"]=="excluded":
            if not c.get("exclusion_reason") or c["judgments"] or c["required_groups"]:
                raise ValueError("Invalid exclusion")
            continue
        if not c["required_groups"] or len(c["required_groups"])!=len(set(c["required_groups"])):
            raise ValueError("Missing or duplicate evidence groups")
        covered=set()
        for j in c["judgments"]:
            if j["evidence_id"] not in docs:
                raise ValueError(f"Missing label in corpus: {j['evidence_id']}")
            d=docs[j["evidence_id"]]
            if type(j["grade"]) is not int or j["grade"] not in (0,1,2):
                raise ValueError("Invalid relevance grade")
            if any(d["metadata"][k]!=c[k] for k in ("ticker","fiscal_year","form_type")):
                raise ValueError("Corpus filter incompatibility")
            for k in ("source_html","source_sha256","section_path"):
                if j[k]!=d["metadata"][k]:
                    raise ValueError(f"Incompatible label provenance: {k}")
            if j["content_sha256"]!=d["content_sha256"] or not j["spans"]:
                raise ValueError("Missing span or mismatched label content")
            for s in j["spans"]:
                if not (0<=s["start"]<s["end"]<=len(d["content"])) or d["content"][s["start"]:s["end"]]!=s["quote"]:
                    raise ValueError("Invalid evidence span")
                if not s.get("anchor") or s["anchor"] not in s["quote"]:
                    raise ValueError("Invalid evidence anchor")
            if j["grade"]==2:
                covered.update(j["groups"])
        if not set(c["required_groups"])<=covered:
            raise ValueError("Missing relevant labels for required evidence group")
        if c["stratum"]=="hard_negative" and not any(j["grade"]==0 for j in c["judgments"]):
            raise ValueError("Hard-negative case missing explicit negative")
        if c["stratum"]=="multi_evidence" and len(c["required_groups"])<2:
            raise ValueError("Multi-evidence case missing multiple groups")
    return {"queries":len(cases),"answerable":sum(c["status"]=="answerable" for c in cases),
            "excluded":sum(c["status"]=="excluded" for c in cases),"documents":len(documents),
            "missing_labels":0,"strata":dict(sorted(Counter(c["stratum"] for c in cases).items()))}


def load_dataset(root: Path, repo: Path=Path(".")) -> tuple[list[dict],list[dict],dict]:
    manifest=json.loads((root/"dataset_manifest.json").read_text())
    for name,digest in manifest["files"].items():
        if Path(name).name!=name or sha256(root/name)!=digest:
            raise ValueError(f"Frozen dataset hash mismatch: {name}")
    source_manifest=json.loads((root/"corpus_manifest.json").read_text())
    for s in source_manifest["sources"]:
        p=Path(s["source_html"])
        if p.is_absolute() or ".." in p.parts or sha256(repo/p)!=s["source_sha256"]:
            raise ValueError("Source filing hash/path mismatch")
    cases=read_jsonl(root/"queries.jsonl")
    docs=read_jsonl(root/"corpus.jsonl")
    report=validate(cases,docs)
    if report["queries"]!=manifest["query_count"] or report["answerable"]!=manifest["answerable_count"] or report["strata"]!=manifest["strata"]:
        raise ValueError("Dataset manifest counts mismatch")
    return cases,docs,report


def metrics(predicted: list[str], case: dict) -> dict[str,float]:
    if case["status"]!="answerable":
        raise ValueError("Excluded cases do not have ranking metrics")
    labels={j["evidence_id"]:j for j in case["judgments"]}
    relevant={i for i,j in labels.items() if j["grade"]==2}
    if not relevant:
        raise ValueError("Missing relevant labels")
    seen=set()
    ranked=[]
    for i in predicted[:10]:
        ranked.append(i if i not in seen else None)
        seen.add(i)
    out={}
    for k in (5,10):
        out[f"recall@{k}"]=len(set(ranked[:k]) & relevant)/len(relevant)
        gains=[2**labels.get(i,{"grade":0})["grade"]-1 for i in ranked[:k]]
        dcg=sum(g/math.log2(n+2) for n,g in enumerate(gains))
        ideal=sorted((2**j["grade"]-1 for j in labels.values()),reverse=True)[:k]
        idcg=sum(g/math.log2(n+2) for n,g in enumerate(ideal))
        out[f"ndcg@{k}"]=dcg/idcg if idcg else 0.0
    out["mrr@10"]=next((1/(n+1) for n,i in enumerate(ranked) if i in relevant),0.0)
    groups=set()
    for i in ranked:
        if i in relevant:
            groups.update(labels[i]["groups"])
    required=set(case["required_groups"])
    out["evidence_group_recall@10"]=len(groups & required)/len(required)
    return out


def percentile(values: list[float], p: float) -> float | None:
    if not 0<p<=1:
        raise ValueError("Percentile must be in (0,1]")
    if not values:
        return None
    if any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError("Invalid latency")
    return sorted(values)[math.ceil(p*len(values))-1]


def aggregate(rows: list[dict]) -> dict:
    quality={k:statistics.mean(r["metrics"][k] for r in rows) if rows else None for k in METRICS}
    latency={}
    for name in ("retrieval_ms","reranker_ms"):
        values=[r[name] for r in rows if r.get(name) is not None]
        latency[name]={"n":len(values),"p50":percentile(values,.5),"p95":percentile(values,.95)}
    return {"queries":len(rows),"errors":sum(r["error"] is not None for r in rows),
            "duplicate_returned_ids":sum(r.get("duplicate_returned_ids",0) for r in rows),
            "unjudged_returned_ids":sum(r.get("unjudged_returned_ids",0) for r in rows),
            "missing_labels":0,"metrics":quality,"latency":latency}
