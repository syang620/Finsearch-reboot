"""One diagnostic batch of the six captured PR7 timeout packets, analyst only.

Uses frozen runtime unchanged, including 120-second timeout and existing retries.
No planner, retrieval, resolver, SEC execution, gate evaluator, or release scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import llm_client
from agents.analyst.agent import AnalystAgent
from agents.contracts import AnalystPacket
from investigate_pr7_timeouts import CASES, FROZEN, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def safe(value):
    if isinstance(value, str):
        return re.sub(r"/Users/[^\s\"']+", "<local-path>", value)
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    return value


async def main(args):
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == FROZEN
    assert not subprocess.check_output(["git", "diff", FROZEN, "--", "src"], text=True)
    assert not args.output_dir.exists(), "Never overwrite diagnostic evidence"
    rows = {r["id"]: r for r in map(json.loads, args.per_query.read_text().splitlines())}
    assert [k for k, r in rows.items() if "ANALYST_MODEL_TIMEOUT" in str(r["trace"].get("analyst"))] == CASES
    url = llm_client.ollama_base_url().rstrip("/")
    assert url in {"http://127.0.0.1:11434", "http://localhost:11434"}, "Use the captured local provider only"
    tags = requests.get(url + "/api/tags", timeout=10).json()
    model = next(m for m in tags["models"] if m["name"] == "qwen2.5:14b-instruct")
    assert model["digest"] == "7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6"
    args.output_dir.mkdir(parents=True)
    state = {"case": None, "calls": [], "active": 0}
    lock = threading.Lock()
    original = llm_client._ollama_chat_completion

    def observed(messages, **kwargs):
        call = {"case_id": state["case"], "started_at_utc": now(),
                "message_count": len(messages), "messages_sha256": digest(messages),
                "message_chars": sum(len(str(m.get("content", ""))) for m in messages),
                "tool_schema_sha256": digest(kwargs.get("tools")),
                "options": kwargs.get("options"), "model": kwargs.get("model"),
                "timeout_s": kwargs.get("timeout")}
        with lock:
            state["active"] += 1
            state["calls"].append(call)
        start = time.perf_counter()
        try:
            result = original(messages, **kwargs)
            call["provider_response"] = {k: result.get(k) for k in (
                "model", "created_at", "done", "done_reason", "total_duration", "load_duration",
                "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")}
            call["response_message"] = result.get("message")
            return result
        except Exception as exc:
            call["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            call["elapsed_ms"] = round((time.perf_counter() - start) * 1000)
            call["finished_at_utc"] = now()
            with lock:
                state["active"] -= 1

    provenance = {
        "diagnostic_only": True, "not_release_evidence": True, "implementation": FROZEN,
        "started_at_utc": now(), "source_per_query_sha256": digest(args.per_query.read_bytes()),
        "case_ids": CASES, "python": platform.python_version(), "pytest": importlib.metadata.version("pytest"),
        "model_digest": model["digest"], "provider_version": requests.get(url + "/api/version", timeout=10).json(),
        "loaded_models_before": requests.get(url + "/api/ps", timeout=10).json(),
        "power_before": subprocess.check_output(["pmset", "-g", "batt"], text=True),
        "power_settings": subprocess.check_output(["pmset", "-g", "custom"], text=True),
        "provider_log_start_bytes": args.provider_log.stat().st_size,
        "runtime_changes": False, "full_gate_reruns": 0, "timeout_s": 120,
        "max_attempts": 2, "max_tool_rounds": 6, "max_context_items": 5,
        "temperature": 0.0, "base_num_predict": 2048,
        "conditions": "Sequential analyst-only replay; current machine state, cold provider initially; no power, timeout, cache, or provider tuning.",
        "instrumentation": "Pass-through observer of the unchanged provider request function; records metadata without changing requests or returned responses.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(safe(provenance), indent=2) + "\n")
    llm_client._ollama_chat_completion = observed
    agent = AnalystAgent(model="ollama/qwen2.5:14b-instruct")
    try:
        for case in CASES:
            state["case"] = case
            packet_raw = rows[case]["trace"]["evaluation_trace"]["analyst_packet"]
            packet = AnalystPacket.model_validate(packet_raw)
            print(json.dumps({"starting": case, "utc": now()}), flush=True)
            start = time.perf_counter()
            result = await agent.arun(packet, debug=True)
            elapsed = round((time.perf_counter() - start) * 1000)
            # Observe completion of the existing synchronous HTTP worker after cancellation.
            # This does not extend the analyst timeout or change its result.
            for _ in range(100):
                if state["active"] == 0:
                    break
                await asyncio.sleep(0.1)
            assert state["active"] == 0, "Stop rather than overlap a still-running provider request"
            record = {"id": case, "packet_sha256": digest(packet_raw), "elapsed_ms": elapsed,
                      "result": result.model_dump(mode="json"),
                      "provider_calls": [c for c in state["calls"] if c["case_id"] == case]}
            with (args.output_dir / "per_case.jsonl").open("a") as handle:
                handle.write(json.dumps(safe(record), sort_keys=True) + "\n")
            print(json.dumps({"finished": case, "elapsed_ms": elapsed, "status": result.status,
                              "error": result.error, "model_calls": len(record["provider_calls"])}), flush=True)
    finally:
        await agent.aclose()
        llm_client._ollama_chat_completion = original
        ending = {"finished_at_utc": now(), "provider_log_end_bytes": args.provider_log.stat().st_size,
                  "power_after": subprocess.check_output(["pmset", "-g", "batt"], text=True),
                  "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "runtime_diff": subprocess.check_output(["git", "diff", FROZEN, "--", "src"], text=True)}
        (args.output_dir / "completion.json").write_text(json.dumps(safe(ending), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", required=True, type=Path)
    parser.add_argument("--provider-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
