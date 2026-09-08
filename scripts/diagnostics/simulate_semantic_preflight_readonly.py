"""Read-only simulation of semantic-v2 preflight; never enters case execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import requests
from qdrant_client import QdrantClient

from evals.semantic_dataset_v2 import read, sha
from scripts.evals.agents.run_semantic_baseline_v2 import controls, snapshot, verify_frozen_index, verify_index
from scripts.evals.agents.run_semantic_baseline_v2_1 import verify_launcher

QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
CONFIG = Path("data/evals/semantic_answer/v1/evaluation_config.json")
CORPUS = Path("data/evals/retrieval/benchmark_v2/corpus.jsonl")
OLD_INDEX = Path("artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f/manifest.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def timed(name, function):
    started_at = now()
    timer = time.perf_counter()
    try:
        result = function()
        return {"name": name, "started_at": started_at,
                "wall_ms": (time.perf_counter() - timer) * 1000,
                "status": "ok", "result": result}
    except Exception as exc:
        return {"name": name, "started_at": started_at,
                "wall_ms": (time.perf_counter() - timer) * 1000,
                "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def get_json(url, headers=None):
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def run(index_manifest, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(CONFIG.read_text())
    records = {
        "status": "diagnostic_only",
        "started_at": now(),
        "implementation_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "policy": "Read-only launch-sequence simulation. No benchmark case, planner call, analyst call, retrieval query, embedding, reranker call, or model inference.",
        "steps": [],
    }

    records["steps"].append(timed("repository_and_freeze_checks", lambda: {
        "tracked_status": subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True),
        "launcher_contract": verify_launcher(QUALITY)[2]["status"],
    }))
    records["steps"].append(timed("local_service_identity", lambda: {
        "ollama_version": get_json("http://127.0.0.1:11434/api/version"),
        "model_digests": {m["name"]: m.get("digest") for m in get_json("http://127.0.0.1:11434/api/tags")["models"]
                          if m["name"] in {config["analyst_model"].removeprefix("ollama/"), config["embedding_model"]}},
        "qdrant": get_json("http://127.0.0.1:6333/").get("title"),
    }))
    records["steps"].append(timed("sec_service_health", lambda: {
        "status_code": requests.get(
            "https://data.sec.gov/submissions/CIK0000320193.json",
            headers={"User-Agent": os.environ["SEC_USER_AGENT"]}, timeout=30
        ).status_code
    }))

    client = QdrantClient(host="127.0.0.1", port=6333, timeout=120)
    def index_identity():
        current, points = snapshot(client, config["collection"])
        historical, _ = snapshot(client, config["historical_collection"])
        verify_index(points, read(CORPUS))
        manifest = json.loads(Path(index_manifest).read_text())
        verify_frozen_index(manifest, current, sha(Path(index_manifest).parent / "embedded.jsonl"))
        old = json.loads(OLD_INDEX.read_text())
        if historical != old["historical_index_after"]:
            raise ValueError("Historical collection changed")
        return {"current": current, "historical": historical}
    records["steps"].append(timed("index_identity", index_identity))
    client.close()

    def planner_setup():
        from agents.planner.interactive_target_resolution import InteractivePlannerAgent
        from agents.orchestrator.agent_orchestrator import aclose_orchestrator_runtime
        planner = InteractivePlannerAgent(model=config["planner_model"], log_timing=False)
        return {"planner_class": type(planner).__name__, "runtime_close_imported": callable(aclose_orchestrator_runtime)}
    records["steps"].append(timed("planner_import_and_construction", planner_setup))
    records["controls_after_setup"] = controls()
    records["steps"].append(timed("unchanged_30_second_settle", lambda: time.sleep(30)))
    records["controls_after_settle"] = controls()
    records["finished_at"] = now()
    records["errors"] = [step for step in records["steps"] if step["status"] != "ok"]
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"output": str(output), "errors": len(records["errors"])}))
    return bool(records["errors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.index_manifest, args.output))
