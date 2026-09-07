#!/usr/bin/env python3
"""One frozen four-mode evaluation. Outputs are append-only and never overwritten."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import time

import requests
from qdrant_client import QdrantClient

from evals.retrieval_benchmark_v2 import METRICS, aggregate,load_dataset,metrics,sha256

FREEZE="0cc20ce37c42d18412cfb518f8b06f775d07257b"


def git(*args):
    return subprocess.check_output(["git",*args],text=True).strip()


def snapshot(client,collection):
    records=[]
    offset=None
    while True:
        batch,offset=client.scroll(collection_name=collection,offset=offset,limit=128,with_payload=True,with_vectors=True)
        records.extend(p.model_dump(mode="json") for p in batch)
        if offset is None:break
    records.sort(key=lambda p:str(p["id"]))
    digest=hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode())
    return {"points":len(records),"payload_vectors_sha256":digest.hexdigest(),
            "config":client.get_collection(collection).config.model_dump(mode="json")},records


def verify_index(records,docs):
    indexed={}
    for p in records:
        payload=p["payload"]
        doc_id=payload.get("doc_id")
        if not doc_id or doc_id in indexed:raise ValueError("Duplicate/missing indexed doc ID")
        indexed[doc_id]=payload
    if set(indexed)!={d["id"] for d in docs}:raise ValueError("Index/corpus ID-set mismatch")
    for d in docs:
        p=indexed[d["id"]]
        if p.get("content")!=d["content"] or any(p.get(k)!=v for k,v in d["metadata"].items()):
            raise ValueError("Index/corpus content or metadata mismatch")


def controls():
    if os.uname().sysname!="Darwin":return {"platform":os.uname().sysname,"power_state":"unavailable"}
    batt=subprocess.check_output(["pmset","-g","batt"],text=True)
    custom=subprocess.check_output(["pmset","-g","custom"],text=True)
    ac="Now drawing from 'AC Power'" in batt
    section=custom.split("AC Power:" if ac else "Battery Power:")[-1].split("AC Power:")[0]
    lpm=re.search(r"lowpowermode\s+(\d+)",section)
    heavy=[]
    browsers=0
    for line in subprocess.check_output(["ps","-axo","pcpu=,comm="],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        cpu,name=fields
        browsers+=int("Google Chrome" in name or "/Safari.app/" in name)
        if float(cpu)>=50 and not any(s in name.lower() for s in ("ollama","qdrant","com.docker","virtualization")):
            heavy.append({"process":Path(name).name,"cpu":float(cpu)})
    return {"at":datetime.now(timezone.utc).isoformat(),"ac_power":ac,
            "low_power_mode":int(lpm.group(1)) if lpm else None,
            "browser_process_count":browsers,"heavy_non_model_processes":heavy}


def hardware():
    result={"platform":os.uname().sysname,"machine":os.uname().machine,"os_release":os.uname().release}
    if os.uname().sysname=="Darwin":
        for key in ("hw.model","hw.memsize","hw.ncpu","machdep.cpu.brand_string"):
            result[key]=subprocess.check_output(["sysctl","-n",key],text=True).strip()
    return result


def schedule(cases,config):
    ordered=sorted((c for c in cases if c["status"]=="answerable"),key=lambda c:c["id"])
    random.Random(config["order_seed"]).shuffle(ordered)
    result=[]
    modes=config["modes"]
    for n,c in enumerate(ordered):
        rotation=n%len(modes)
        result.extend((c,m) for m in modes[rotation:]+modes[:rotation])
    return result


def check_runtime(config):
    from mcp_server.tools import sec_retrieval as runtime
    actual=(runtime.RETRIEVAL_TOP_K,runtime.RERANK_CANDIDATE_LIMIT,runtime.RERANK_TOP_K,
            runtime.QWEN3_EMBED_MODEL,runtime._current_rerank_model())
    expected=(config["retrieval_top_k"],config["rerank_candidate_limit"],config["rerank_top_k"],
              config["embedding_model"],config["reranker_model"])
    if actual!=expected:raise ValueError("Runtime settings differ from frozen comparison config")
    if os.getenv("SEC_QUERY_EMBED_CACHE","1").lower() in {"0","false","no","off"}:
        raise ValueError("Frozen cache policy requires default caching enabled")
    if not runtime._current_qwen3_rerank_api_key():raise ValueError("Missing existing reranker credential")


def clean_checkout():
    if git("status","--porcelain","--untracked-files=no"):
        raise ValueError("Tracked worktree is dirty; freeze implementation first")
    untracked=git("ls-files","--others","--exclude-standard").splitlines()
    if any(Path(p).parts[0]!="artifacts" for p in untracked):
        raise ValueError("Untracked files outside artifacts can affect runtime")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=Path("data/evals/retrieval/benchmark_v2"))
    p.add_argument("--out-root",type=Path,default=Path("artifacts/evals/retrieval/benchmark_v2/baselines"))
    p.add_argument("--index-manifest",type=Path,required=True)
    p.add_argument("--qdrant-host",default="127.0.0.1")
    p.add_argument("--qdrant-port",type=int,default=6333)
    p.add_argument("--embedding-url",default="http://127.0.0.1:11434/api/embed")
    p.add_argument("--env-file",type=Path)
    a=p.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file,override=False)
    # Imports occur AFTER optional credential loading, exactly as production
    # configuration does; explicit client/collection/embed arguments still win.
    from evals.retrieval_ablation import retrieve_ablation_points,RETRIEVAL_MODES
    from mcp_server.tools import sec_retrieval as runtime
    head=git("rev-parse","HEAD")
    clean_checkout()
    if git("diff",FREEZE,"--",a.dataset.as_posix()):
        raise ValueError("Dataset changed after pre-retrieval freeze")
    cases,docs,validation=load_dataset(a.dataset)
    config=json.loads((a.dataset/"comparison_config.json").read_text())
    if config["modes"]!=list(RETRIEVAL_MODES):raise ValueError("Mode contract mismatch")
    check_runtime(config)
    index=json.loads(a.index_manifest.read_text())
    if not index["completed"] or index["corpus_sha256"]!=sha256(a.dataset/"corpus.jsonl"):
        raise ValueError("Incomplete/incompatible benchmark index")
    collection="finsearch_benchmark_v2_"+index["corpus_sha256"][:16]
    expected_url=f"http://{a.qdrant_host}:{a.qdrant_port}"
    if index["collection"]!=collection or index["qdrant_url"]!=expected_url or index["embedding_url"]!=a.embedding_url:
        raise ValueError("Index/runtime endpoint provenance mismatch")
    tags=requests.get(a.embedding_url.rsplit("/api/",1)[0]+"/api/tags",timeout=10)
    tags.raise_for_status()
    if not any(m["name"]==config["embedding_model"] and m["digest"]==config["embedding_digest"] for m in tags.json()["models"]):
        raise ValueError("Embedding digest mismatch")
    client=QdrantClient(host=a.qdrant_host,port=a.qdrant_port,timeout=120)
    before,records=snapshot(client,collection)
    verify_index(records,docs)
    # Keep a read-only guard on the historical collection as well.
    historical="sec_docs_dense_bm25_pr2_63dcec0"
    historical_before,_=snapshot(client,historical)
    preflight=controls()
    if preflight.get("ac_power") is not True or preflight.get("low_power_mode")!=0 or preflight.get("browser_process_count")!=0:
        raise ValueError("Baseline latency preflight requires AC, LPM off and closed browsers")
    out=a.out_root/head
    out.mkdir(parents=True,exist_ok=False)
    cache=Path(".cache/retrieval_benchmark_v2_runs")/head
    cache.mkdir(parents=True,exist_ok=False)
    source_paths=git("ls-files","src","scripts/evals/retrieval","data/evals/retrieval/benchmark_v2").splitlines()
    manifest={"implementation_sha":head,"dataset_freeze_sha":FREEZE,
              "dataset_manifest_sha256":sha256(a.dataset/"dataset_manifest.json"),
              "config":config,"validation":validation,"index":index,
              "index_before":before,"historical_index_before":historical_before,
              "tracked_worktree_clean":True,"started_at":datetime.now(timezone.utc).isoformat(),
              "packages":{name:importlib.metadata.version(name) for name in ["qdrant-client","pytest","requests","langchain-ollama"]},
              "python":os.sys.version.split()[0],"hardware":hardware(),
              "preflight_controls":preflight,"source_sha256":{s:sha256(Path(s)) for s in source_paths},
              "reranker":{"model":runtime._current_rerank_model(),"url":runtime._current_qwen3_rerank_api_url(),
                          "immutable_service_digest":None,"credential":"existing environment credential; not recorded"},
              "status":"running","latency_policy":"single pass; per-mode empty query caches; rotated mode order; no harness retries; shared model warm state"}
    (out/"started.json").write_text(json.dumps(manifest,indent=2)+"\n")
    doc_ids={d["id"] for d in docs}
    rows=[]
    guard=subprocess.Popen(["caffeinate","-dimsu"])
    started=time.perf_counter()
    fatal=None
    try:
        with (out/"per_query.jsonl").open("x") as stream, (out/"errors.jsonl").open("x") as errors:
            for c,mode in schedule(cases,config):
                state=controls()
                if not state["ac_power"] or state["low_power_mode"]!=0 or state["browser_process_count"] or guard.poll() is not None:
                    raise RuntimeError("Power/browser/awake controls changed; preserve partial run")
                os.environ["SEC_QUERY_EMBED_CACHE_DIR"]=str((cache/mode).resolve())
                row={"id":c["id"],"mode":mode,"stratum":c["stratum"],"ticker":c["ticker"],"fiscal_year":c["fiscal_year"],
                     "query":c["query"],"error":None,"reranker_ms":None,"controls_before":state,"ranked_ids":[],"timing":{}}
                t0=time.perf_counter()
                try:
                    _,_,ranked,timing=retrieve_ablation_points(retrieval_mode=mode,query=c["query"],ticker=c["ticker"],
                        fiscal_year=c["fiscal_year"],form_type=c["form_type"],doc_types=config["doc_types"],client=client,
                        qdrant_host=a.qdrant_host,qdrant_port=a.qdrant_port,collection_name=collection,
                        embed_api_url=a.embedding_url,embed_model=config["embedding_model"])
                    row["retrieval_ms"]=(time.perf_counter()-t0)*1000
                    row["ranked_ids"]=[str((p.payload or {}).get("doc_id","")) for p in ranked[:10]]
                    row["scores"]=[float(p.score) for p in ranked[:10]]
                    if any(i not in doc_ids for i in row["ranked_ids"]):raise ValueError("Returned ID absent from frozen corpus")
                    row["timing"]=timing
                    if mode=="hybrid_reranker":
                        row["reranker_ms"]=float(timing["rerank_ms"])
                        if timing["rerank"]["applied_backend"]!="qwen3_api" or timing["rerank"]["fallback_used"]:
                            raise ValueError("Reranker did not apply frozen backend")
                    row["metrics"]=metrics(row["ranked_ids"],c)
                except Exception as exc:
                    logging.exception("Benchmark retrieval failed for %s/%s",c["id"],mode)
                    row["retrieval_ms"]=(time.perf_counter()-t0)*1000
                    row["error"]={"type":type(exc).__name__,"message":"Retrieval failed; local execution log contains diagnostic detail. No retry or silent case removal."}
                    row["metrics"]={k:0.0 for k in METRICS}
                    errors.write(json.dumps({"id":c["id"],"mode":mode,"error":row["error"]})+"\n");errors.flush()
                row["duplicate_returned_ids"]=len(row["ranked_ids"])-len(set(row["ranked_ids"]))
                judged={j["evidence_id"] for j in c["judgments"]}
                row["unjudged_returned_ids"]=sum(i not in judged for i in row["ranked_ids"])
                rows.append(row)
                row["controls_after"]=controls()
                stream.write(json.dumps(row,sort_keys=True)+"\n");stream.flush()
                print(json.dumps({"completed":len(rows),"total":validation["answerable"]*4,"id":c["id"],"mode":mode,"error":row["error"] is not None}),flush=True)
                state=row["controls_after"]
                if not state["ac_power"] or state["low_power_mode"]!=0 or state["browser_process_count"]:
                    raise RuntimeError("Power/browser controls changed during retrieval; preserve partial run")
    except Exception as exc:
        fatal={"type":type(exc).__name__,"message":str(exc)}
    finally:
        guard.terminate();guard.wait(timeout=10)
        manifest["wall_seconds"]=time.perf_counter()-started
        manifest["finished_at"]=datetime.now(timezone.utc).isoformat()
    after,_=snapshot(client,collection)
    historical_after,_=snapshot(client,historical)
    tags_after=requests.get(a.embedding_url.rsplit("/api/",1)[0]+"/api/tags",timeout=10)
    tags_after.raise_for_status()
    model_unchanged=any(m["name"]==config["embedding_model"] and m["digest"]==config["embedding_digest"] for m in tags_after.json()["models"])
    manifest.update(index_after=after,historical_index_after=historical_after,
                    indexes_unchanged=before==after and historical_before==historical_after,
                    embedding_model_unchanged=model_unchanged,
                    fatal_error=fatal,completed_pairs=len(rows),status="completed" if fatal is None else "partial",
                    controls_after=controls())
    summary={"expected_pairs":validation["answerable"]*4,"completed_pairs":len(rows),"excluded_queries":validation["excluded"],
             "missing_labels":0,"fatal_error":fatal,"modes":{},"indexes_unchanged":manifest["indexes_unchanged"]}
    for mode in config["modes"]:
        subset=[r for r in rows if r["mode"]==mode]
        summary["modes"][mode]={"overall":aggregate(subset),
            "by_stratum":{s:aggregate([r for r in subset if r["stratum"]==s]) for s in sorted({c["stratum"] for c in cases if c["status"]=="answerable"})},
            "by_company":{s:aggregate([r for r in subset if r["ticker"]==s]) for s in sorted({c["ticker"] for c in cases})}}
    summary["complete"]=fatal is None and len(rows)==summary["expected_pairs"] and manifest["indexes_unchanged"] and model_unchanged
    clean_checkout()
    if git("rev-parse","HEAD")!=head:raise ValueError("Implementation changed during evaluation")
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    manifest["raw_sha256"]={name:sha256(out/name) for name in ("per_query.jsonl","errors.jsonl","summary.json")}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if not summary["complete"]:raise SystemExit(2)


if __name__=="__main__":main()
