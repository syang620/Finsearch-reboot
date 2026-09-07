"""Append-only current-runtime answers, followed by a separate frozen judge pass."""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import time

import requests
from qdrant_client import QdrantClient

from evals.semantic_answer_v1 import (deterministic_case, judge_packet, load_dataset,
    read_jsonl, sha, sha_text, summarize_deterministic, summarize_semantic, validate_judgment)
from scripts.evals.retrieval.run_benchmark_v2 import clean_checkout, controls, git, hardware, snapshot, verify_index

DATA = Path("data/evals/semantic_answer/v1")
OUT = Path("artifacts/evals/semantic_answer/v1/baselines")
REFERENCE = Path("artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f/manifest.json")

def now(): return datetime.now(timezone.utc).isoformat()

def sanitize(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    text = text.replace(str(Path.cwd()), "<WORKTREE>").replace(str(Path.home()), "<LOCAL_HOME>")
    for key in ("SEC_USER_AGENT", "DASHSCOPE_API_KEY", "QWEN3_RERANK_API_KEY"):
        secret = os.environ.get(key, "")
        if len(secret) >= 8: text = text.replace(secret, "<REDACTED>")
    return json.loads(text)

def save(path, value):
    with path.open("x") as f: json.dump(sanitize(value), f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")

def append(path, value):
    with path.open("a") as f:
        f.write(json.dumps(sanitize(value), ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())

def models(config):
    result = requests.get("http://127.0.0.1:11434/api/tags", timeout=10); result.raise_for_status()
    names = {m["name"]: m for m in result.json()["models"]}
    for name, digest in ((config["analyst_model"].removeprefix("ollama/"), config["model_digest"]),
                         (config["embedding_model"], config["embedding_digest"]),
                         (config["judge"]["model"], config["judge"]["digest"])):
        if names[name]["digest"] != digest: raise ValueError("Frozen model digest mismatch")
    version = requests.get("http://127.0.0.1:11434/api/version", timeout=10); version.raise_for_status()
    return {"ollama": version.json(), "models": {k: names[k] for k in (config["analyst_model"].removeprefix("ollama/"), config["embedding_model"], config["judge"]["model"])}}

def dataset_freeze():
    return git("log", "--diff-filter=A", "--format=%H", "--", str(DATA / "manifest.json")).splitlines()[0]

def provenance(config):
    clean_checkout()
    base = config["base_runtime_sha"]
    production = [p for p in git("ls-files", "src").splitlines() if not p.startswith("src/evals/")]
    if git("diff", base, "--", *production): raise ValueError("Production source changed from post-PR8 base")
    frozen = dataset_freeze()
    if git("diff", frozen, "--", str(DATA)): raise ValueError("Dataset modified after freeze")
    paths = git("ls-files", "src", "scripts/evals/agents", "scripts/evals/retrieval/run_benchmark_v2.py").splitlines()
    return {"implementation_sha": git("rev-parse", "HEAD"), "dataset_freeze_sha": frozen,
        "dataset_manifest_sha256": sha(DATA / "manifest.json"), "dataset_sha256": sha(DATA / "queries.jsonl"),
        "source_sha256": {p: sha(p) for p in paths}, "production_identical_to": base,
        "tracked_worktree_clean": True, "config": config, "model_provenance": models(config),
        "hardware": hardware(), "python": os.sys.version.split()[0],
        "packages": {n: importlib.metadata.version(n) for n in ("pytest", "requests", "qdrant-client", "langchain-ollama")}}

def runtime_environment(config, cache):
    # Evaluation resource selection only; no production defaults are edited.
    if os.getenv("SEC_METRIC_FIXTURE_ROOT"): raise ValueError("This baseline forbids metric fixtures")
    if not os.getenv("SEC_USER_AGENT"): raise ValueError("Provide existing SEC contact in local environment only")
    os.environ.update(QDRANT_COLLECTION_NAME=config["collection"], QDRANT_HOST="127.0.0.1", QDRANT_PORT="6333",
        QWEN3_EMBED_API_URL="http://127.0.0.1:11434/api/embed", OLLAMA_BASE_URL="http://127.0.0.1:11434",
        FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH=str(cache / "checkpoints.sqlite"),
        SEC_QUERY_EMBED_CACHE_DIR=str(cache / "query_embeddings"))
    from mcp_server.tools import sec_retrieval as runtime
    if (runtime.RETRIEVAL_TOP_K, runtime.RERANK_CANDIDATE_LIMIT, runtime.RERANK_TOP_K, runtime.QWEN3_EMBED_MODEL, runtime._current_rerank_model()) != (50, 10, 10, config["embedding_model"], config["reranker_model"]):
        raise ValueError("Retrieval setting differs from unchanged defaults")
    if not runtime._current_qwen3_rerank_api_key(): raise ValueError("Existing reranker credential missing")
    if os.getenv("SEC_QUERY_EMBED_CACHE", "1").lower() in {"0", "false", "no", "off"}: raise ValueError("Cache policy mismatch")
    return {"reranker_model": runtime._current_rerank_model(), "reranker_url": runtime._current_qwen3_rerank_api_url(),
        "reranker_service_digest": None, "sec_mode": "unchanged live SEC client; contact supplied locally, redacted",
        "resources": {k: os.environ[k] for k in ("QDRANT_COLLECTION_NAME", "QDRANT_HOST", "QDRANT_PORT", "QWEN3_EMBED_API_URL", "OLLAMA_BASE_URL")}}

async def answers():
    cases, counts = load_dataset(DATA)
    config = json.loads((DATA / "evaluation_config.json").read_text())
    manifest = provenance(config)
    head = manifest["implementation_sha"]
    out, cache = OUT / head, Path(".cache/semantic_answer_v1") / head
    # Refuse any previous attempt at this implementation identity.
    if out.exists() or cache.exists(): raise ValueError("SHA-keyed attempt already exists; never overwrite/rerun in place")
    environment = runtime_environment(config, cache)
    client = QdrantClient(host="127.0.0.1", port=6333, timeout=120)
    before, records = snapshot(client, config["collection"])
    historical, _ = snapshot(client, config["historical_collection"])
    verify_index(records, read_jsonl("data/evals/retrieval/benchmark_v2/corpus.jsonl"))
    reference = json.loads(REFERENCE.read_text())
    if before != reference["index_after"] or historical != reference["historical_index_after"]: raise ValueError("Frozen index snapshot mismatch")
    state = controls()
    if state.get("ac_power") is not True or state.get("low_power_mode") != 0: raise ValueError("Begin on AC with Low Power Mode off; no system setting changes made")
    out.mkdir(parents=True, exist_ok=False); cache.mkdir(parents=True, exist_ok=False)
    ordered = sorted(cases, key=lambda c: c["id"]); random.Random(config["order_seed"]).shuffle(ordered)
    manifest.update(started_at=now(), validation=counts, environment=environment, controls_before=state,
        index_before=before, historical_index_before=historical, schedule=[c["id"] for c in ordered],
        status="running", corpus_manifest_sha256=sha("data/evals/retrieval/benchmark_v2/corpus_manifest.json"),
        artifact_redaction="Local worktree/home paths and credential strings only; financial evidence unchanged")
    save(out / "started.json", manifest)
    from agents.planner.interactive_target_resolution import InteractivePlannerAgent
    from agents.orchestrator.agent_orchestrator import run_multi_agent_orchestration, aclose_orchestrator_runtime
    planner = InteractivePlannerAgent(model=config["planner_model"], log_timing=False)
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    rows = []
    try:
        for index, case in enumerate(ordered):
            start = time.perf_counter(); started = now(); error = None
            print(f"START {index+1}/{len(ordered)} {case['id']}", flush=True)
            state_before = controls()
            try:
                output = await run_multi_agent_orchestration(case["user_query"], planner=planner,
                    analyst_model=config["analyst_model"], tables_dir=str(DATA / "tables"),
                    debug=False, include_evidence_trace=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                output = {"status": "harness_captured_runtime_error", "error": error}
            append(out / "raw_answers.jsonl", {"case_id": case["id"], "started_at": started,
                "wall_ms": (time.perf_counter()-start)*1000, "controls_before": state_before,
                "controls_after": controls(), "error": error, "output": output})
            row = deterministic_case(case, output); rows.append(row)
            append(out / "deterministic.jsonl", row)
            print(f"END {case['id']} runtime={output.get('status')} analyst={row['status']} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    finally:
        await aclose_orchestrator_runtime()
        awake.terminate(); awake.wait(timeout=10)
        after, _ = snapshot(client, config["collection"])
        historical_after, _ = snapshot(client, config["historical_collection"])
        manifest.update(finished_at=now(), status="complete" if len(rows)==len(cases) else "incomplete",
            completed_cases=len(rows), index_after=after, historical_index_after=historical_after,
            index_unchanged=before==after and historical==historical_after, model_provenance_after=models(config),
            controls_after=controls(), files_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
        save(out / "answer_manifest.json", manifest)
        save(out / "deterministic_summary.json", {"overall": summarize_deterministic(rows),
            "by_stratum": {s: summarize_deterministic([r for r in rows if r["stratum"]==s]) for s in sorted({c["stratum"] for c in cases})}})
        client.close()

def judge(out, audit):
    cases, _ = load_dataset(DATA)
    config = json.loads((DATA / "evaluation_config.json").read_text())
    baseline = json.loads((out / "answer_manifest.json").read_text())
    if baseline["status"] != "complete": raise ValueError("Do not judge a selectively incomplete answer run")
    clean_checkout()
    for path, digest in baseline["source_sha256"].items():
        if sha(path) != digest: raise ValueError("Evaluator/source changed; re-freeze required")
    for name, digest in baseline["files_sha256"].items():
        if sha(out / name) != digest: raise ValueError("Answer evidence changed")
    # The selected source audit must exist and be committed before judge output.
    audit_rows = read_jsonl(audit)
    if {r["case_id"] for r in audit_rows} != {c["id"] for c in cases if c["audit_selected"]}: raise ValueError("Audit coverage mismatch")
    if not git("ls-files", "--", str(audit)) or git("diff", "HEAD", "--", str(audit)): raise ValueError("Freeze audit before exposing judge predictions")
    config_j = config["judge"]
    judge_dir = out / "judge"; judge_dir.mkdir(exist_ok=False)
    model_state = models(config)
    raw = {r["case_id"]: r["output"] for r in read_jsonl(out / "raw_answers.jsonl")}
    rubric = (DATA / "judge_rubric.txt").read_text()
    valid = {}
    save(judge_dir / "started.json", {"started_at": now(), "implementation_sha": baseline["implementation_sha"],
        "judge_execution_head": git("rev-parse", "HEAD"), "audit_sha256": sha(audit), "models": model_state,
        "rubric_sha256": sha(DATA / "judge_rubric.txt"), "config": config_j})
    for i, cid in enumerate(baseline["schedule"]):
        case = next(c for c in cases if c["id"] == cid); output = raw[cid]
        analyst = output.get("analyst") or {}
        record = {"case_id": cid, "started_at": now()}
        if not analyst.get("ok") or analyst.get("status") not in {"ok", "insufficient_data"}:
            record.update(status="not_assessable", error="No emitted successful final answer; not a semantic pass")
        else:
            packet = judge_packet(case, output)
            messages = [{"role": "system", "content": rubric}, {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]
            record["input_sha256"] = sha_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            started = time.perf_counter()
            try:
                response = requests.post("http://127.0.0.1:11434/api/chat", timeout=config_j["timeout_s"], json={
                    "model": config_j["model"], "messages": messages, "stream": False,
                    "think": config_j["think"], "format": config_j["format"],
                    "options": {k: config_j[k] for k in ("temperature", "num_ctx", "num_predict")}})
                response.raise_for_status(); payload = response.json(); record["response"] = payload
                if payload.get("done_reason") == "length" or payload.get("prompt_eval_count", 0) >= config_j["num_ctx"]-config_j["num_predict"]:
                    raise ValueError("Generation/context capacity reached; assessment unknown")
                verdict = validate_judgment(case, output, json.loads(payload["message"]["content"]))
                valid[cid] = verdict; record.update(status="valid", judgment=verdict)
            except Exception as exc: record.update(status="judge_error", error=f"{type(exc).__name__}: {exc}")
            record["wall_ms"] = (time.perf_counter()-started)*1000
        append(judge_dir / "judgments.jsonl", record)
        print(f"JUDGE {i+1}/{len(cases)} {cid} {record['status']}", flush=True)
    save(judge_dir / "summary.json", {"overall": summarize_semantic(cases, raw, valid),
        "by_stratum": {s: summarize_semantic([c for c in cases if c["stratum"]==s], raw, valid) for s in sorted({c["stratum"] for c in cases})}})
    save(judge_dir / "manifest.json", {"finished_at": now(), "completed_cases": len(cases),
        "model_provenance_after": models(config), "files_sha256": {p.name: sha(p) for p in sorted(judge_dir.iterdir()) if p.is_file()}})

def main():
    p = argparse.ArgumentParser(); p.add_argument("phase", choices=["answers", "judge"])
    p.add_argument("--baseline", type=Path); p.add_argument("--audit", type=Path)
    a = p.parse_args(); logging.basicConfig(level=logging.WARNING)
    if a.phase == "answers": asyncio.run(answers())
    else:
        if not a.baseline or not a.audit: p.error("judge requires --baseline and --audit")
        judge(a.baseline, a.audit)

if __name__ == "__main__": main()
