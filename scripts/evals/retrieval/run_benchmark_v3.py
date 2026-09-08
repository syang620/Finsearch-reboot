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

from evals.retrieval_benchmark_v3 import (METRICS, load_dataset, metrics, sha256,
    classify_results, summarize, verify_history)
from scripts.evals.retrieval.index_provenance_v3 import verify_frozen_index

BASE="25c15afbab31212a42c97e650e2418f6f82a8674"
REVIEW_REPOSITORY="syang620/Finsearch-reboot"
REVIEW_AUTHOR="chatgpt-codex-connector[bot]"


def committed_approval(path):
    path=Path(path)
    contents=path.read_bytes()
    root=Path(git("rev-parse","--show-toplevel")).resolve()
    try:
        relative=path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Approval must be committed inside this repository") from exc
    try:
        committed=subprocess.check_output(["git","show",f"HEAD:{relative}"],stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise ValueError("Approval is not committed at HEAD") from exc
    if contents!=committed:
        raise ValueError("Approval has uncommitted changes")
    return json.loads(contents)


def github_json(endpoint,paginate=False):
    command=["gh","api",endpoint]
    if paginate: command.extend(["--paginate","--slurp"])
    try:
        value=json.loads(subprocess.check_output(command,text=True,stderr=subprocess.DEVNULL))
    except (OSError,subprocess.CalledProcessError,json.JSONDecodeError) as exc:
        raise ValueError("Cannot verify benchmark review with GitHub; comparison remains blocked") from exc
    return [item for page in value for item in page] if paginate else value


def verify_remote_review(approval):
    pr=approval.get("pull_request")
    comment_id=approval.get("review_comment_id")
    if type(pr) is not int or pr<=0 or type(comment_id) is not int or comment_id<=0:
        raise ValueError("Approval requires numeric PR and review-comment identities")
    prefix=f"repos/{REVIEW_REPOSITORY}"
    comment=github_json(f"{prefix}/issues/comments/{comment_id}")
    expected_url=f"https://github.com/{REVIEW_REPOSITORY}/pull/{pr}#issuecomment-{comment_id}"
    if (comment.get("id")!=comment_id or comment.get("html_url")!=expected_url
        or approval.get("review_url")!=expected_url
        or comment.get("issue_url")!=f"https://api.github.com/{prefix}/issues/{pr}"):
        raise ValueError("Review URL/PR/comment provenance mismatch")
    author=comment.get("user") or {}
    if author.get("login")!=REVIEW_AUTHOR or author.get("type")!="Bot":
        raise ValueError("Approval is not from the expected Codex reviewer")
    body=comment.get("body","")
    if hashlib.sha256(body.encode()).hexdigest()!=approval.get("review_body_sha256"):
        raise ValueError("Review body changed or was not captured accurately")
    reviewed=re.search(r"\*\*Reviewed commit:\*\*\s*`([0-9a-f]{10,40})`",body)
    if (not reviewed or not approval["reviewed_commit"].startswith(reviewed.group(1))
        or not body.strip().startswith("Codex Review: Didn't find any major issues.")):
        raise ValueError("No clean Codex review of the exact candidate")
    pull=github_json(f"{prefix}/pulls/{pr}")
    if (pull.get("base",{}).get("repo",{}).get("full_name")!=REVIEW_REPOSITORY
        or pull.get("state")!="open"):
        raise ValueError("Review does not belong to the open benchmark PR")
    reviews=github_json(f"{prefix}/pulls/{pr}/reviews",paginate=True)
    if any(r.get("commit_id")==approval["reviewed_commit"] and r.get("state")=="CHANGES_REQUESTED" for r in reviews):
        raise ValueError("Reviewed commit has requested changes")
    findings=github_json(f"{prefix}/pulls/{pr}/comments",paginate=True)
    if any((r.get("user") or {}).get("login")==REVIEW_AUTHOR
           and r.get("original_commit_id",r.get("commit_id"))==approval["reviewed_commit"] for r in findings):
        raise ValueError("Reviewed commit has inline Codex findings; renewed clean review required")


def verify_approval(path, dataset):
    approval=committed_approval(path)
    if approval.get("status")!="approved_for_narrow_known_label_baseline":
        raise ValueError("Benchmark-quality approval is required before comparison")
    freeze=approval.get("reviewed_commit","")
    if not re.fullmatch(r"[0-9a-f]{40}",freeze):
        raise ValueError("Approval must identify an exact reviewed commit")
    if approval.get("dataset_manifest_sha256")!=sha256(dataset/"dataset_manifest.json"):
        raise ValueError("Approval does not bind this dataset hash")
    if not approval.get("review_url","").startswith("https://github.com/syang620/Finsearch-reboot/"):
        raise ValueError("Missing source review provenance")
    git("merge-base","--is-ancestor",freeze,"HEAD")
    if git("diff",freeze,"--","src","scripts/evals/retrieval",dataset.as_posix()):
        raise ValueError("Reviewed annotation/scoring/runtime changed; renewed review required")
    if git("diff",BASE,"--","src/agents","src/mcp_server","src/ingestion","src/evals/retrieval_ablation.py"):
        raise ValueError("This baseline must measure unchanged production and ranking behavior")
    verify_remote_review(approval)
    return freeze,approval


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
    p.add_argument("--dataset",type=Path,default=Path("data/evals/retrieval/benchmark_v3"))
    p.add_argument("--out-root",type=Path,default=Path("artifacts/evals/retrieval/benchmark_v3/baselines"))
    p.add_argument("--index-manifest",type=Path,required=True)
    p.add_argument("--approval",type=Path,required=True)
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
    freeze,approval=verify_approval(a.approval,a.dataset)
    cases,docs,dataset_manifest=load_dataset(a.dataset)
    validation=dataset_manifest["composition"]
    history_before=verify_history(a.dataset)
    corpus_ref=json.loads((a.dataset/"corpus_ref.json").read_text())
    config=json.loads((a.dataset/"comparison_config.json").read_text())
    if config["modes"]!=list(RETRIEVAL_MODES):raise ValueError("Mode contract mismatch")
    check_runtime(config)
    index=json.loads(a.index_manifest.read_text())
    if not index["completed"] or index["corpus_sha256"]!=corpus_ref["sha256"]:
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
    embedded_digest=sha256(a.index_manifest.parent/"embedded.jsonl")
    index_provenance=verify_frozen_index(index,before,embedded_digest)
    # Keep a read-only guard on the historical collection as well.
    historical="sec_docs_dense_bm25_pr2_63dcec0"
    historical_before,_=snapshot(client,historical)
    preflight=controls()
    if preflight.get("ac_power") is not True or preflight.get("low_power_mode")!=0 or preflight.get("browser_process_count")!=0:
        raise ValueError("Baseline latency preflight requires AC, LPM off and closed browsers")
    out=a.out_root/head
    out.mkdir(parents=True,exist_ok=False)
    cache=Path(".cache/retrieval_benchmark_v3_runs")/head
    cache.mkdir(parents=True,exist_ok=False)
    source_paths=git("ls-files","src","scripts/evals/retrieval","data/evals/retrieval/benchmark_v3").splitlines()
    manifest={"implementation_sha":head,"dataset_freeze_sha":freeze,"annotation_approval":approval,
              "annotation_approval_sha256":sha256(a.approval),"runtime_base":BASE,
              "annotation_approval_path":a.approval.resolve().relative_to(Path(git("rev-parse","--show-toplevel")).resolve()).as_posix(),
              "review_verified_live_at":datetime.now(timezone.utc).isoformat(),
              "historical_integrity_before":history_before,
              "dataset_manifest_sha256":sha256(a.dataset/"dataset_manifest.json"),
              "config":config,"validation":validation,"index":index,
              "index_provenance":index_provenance,"build_embedding_cache_sha256":embedded_digest,
              "index_before":before,"historical_index_before":historical_before,
              "tracked_worktree_clean":True,"started_at":datetime.now(timezone.utc).isoformat(),
              "packages":{name:importlib.metadata.version(name) for name in ["qdrant-client","pytest","requests","langchain-ollama"]},
              "python":os.sys.version.split()[0],"hardware":hardware(),
              "preflight_controls":preflight,"source_sha256":{s:sha256(Path(s)) for s in source_paths},
              "reranker":{"model":runtime._current_rerank_model(),"url":runtime._current_qwen3_rerank_api_url(),
                          "immutable_service_digest":None,"credential":"existing environment credential; not recorded"},
              "status":"running","latency_policy":"single pass; per-mode empty query caches; rotated mode order; no harness retries; shared model warm state"}
    (out/"started.json").write_text(json.dumps(manifest,indent=2)+"\n")
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
                    classified=classify_results(row["ranked_ids"],c,docs)
                    if classified["missing_corpus_ids"] or classified["incompatible_filter_ids"]:
                        raise ValueError("Returned evidence violates frozen corpus/filter contract")
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
                row.update(classify_results(row["ranked_ids"],c,docs))
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
    is_complete=fatal is None and len(rows)==validation["answerable"]*4 and manifest["indexes_unchanged"] and model_unchanged
    summary=summarize(rows,cases,config["modes"],complete=is_complete)
    summary.update(excluded_queries=validation["excluded"],fatal_error=fatal,
                   indexes_unchanged=manifest["indexes_unchanged"],
                   missing_labels=0,missing_corpus_ids=sum(r["missing_corpus_ids"] for r in rows),
                   incompatible_filter_ids=sum(r["incompatible_filter_ids"] for r in rows),
                   unjudged_policy="unknown, not proven irrelevant; zero gain under known-label scoring only")
    manifest["historical_integrity_after"]=verify_history(a.dataset)
    clean_checkout()
    if git("rev-parse","HEAD")!=head:raise ValueError("Implementation changed during evaluation")
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    manifest["raw_sha256"]={name:sha256(out/name) for name in ("per_query.jsonl","errors.jsonl","summary.json")}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if not summary["complete"]:raise SystemExit(2)


if __name__=="__main__":main()
