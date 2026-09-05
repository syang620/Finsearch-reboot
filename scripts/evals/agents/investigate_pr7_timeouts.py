"""Read-only comparison of frozen PR24/PR7 evidence and local provider logs.

This is diagnostic tooling, not a release evaluator. No runtime files are changed.
Local input paths are supplied by the caller; outputs contain no home-directory paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from agents.analyst.agent import SYSTEM_PROMPT, build_analyst_prompt
from agents.contracts import AnalystPacket

FROZEN = "c149f73d073a306cf34d938955ae6cc739191528"
CASES = ["AGENT_V1_" + s for s in (
    "KB_002", "KB_003", "HYBRID_002", "HYBRID_003", "ANALYST_001", "ANALYST_003"
)]


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def sizes(value):
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return {"chars": len(raw), "utf8_bytes": len(raw.encode()), "sha256": digest(raw.encode())}


def packet_stats(row):
    raw = row["trace"]["evaluation_trace"]["analyst_packet"]
    packet = AnalystPacket.model_validate(raw)
    prompt = build_analyst_prompt(packet, max_context_items=5, tools_available=True)
    trace = row["trace"]["analyst"]["trace"]
    return {
        "packet": sizes(raw), "user_prompt": sizes(prompt), "system_prompt": sizes(SYSTEM_PROMPT),
        "contexts": [{"id": c.context_id, "kind": c.kind.value, "size": sizes(c.model_dump(mode="json"))}
                     for c in packet.context_items],
        "visible_context_count": min(5, len(packet.context_items)),
        "task": packet.analysis_task.model_dump(mode="json"),
        "timings_ms": row["timings_ms"], "analyst_timings_ms": trace["timing_ms"],
        "tool_call_counts": dict(Counter(c["name"] for c in trace.get("tool_calls", []))),
        "raw_message_count": trace.get("raw_message_count"),
        "grounding_attempt_count": len(trace.get("grounding_attempts", [])),
        "status": row["analyst_status"], "error": row["trace"]["analyst"].get("error"),
    }


def provider_requests(lines):
    requests = []
    start = 0
    for i, line in enumerate(lines):
        m = re.search(r'\[GIN\] (\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2}) \| (\d+) \|\s*([^|]+)\|.*POST\s+"/api/chat"', line)
        if not m:
            continue
        block = "\n".join(lines[start:i + 1])
        prompts = re.findall(r'task\s+(\d+) \| new prompt.*task.n_tokens = (\d+)', block)
        generations = re.findall(r'n_gen =\s*(\d+), tg =\s*([\d.]+) t/s', block)
        evaluations = re.findall(r'(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second', block)
        prefills = re.findall(r'prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second', block)
        requests.append({
            "end_local": datetime.strptime(m[1], "%Y/%m/%d - %H:%M:%S").replace(tzinfo=ZoneInfo("America/New_York")).isoformat(),
            "http_status": int(m[2]), "reported_duration": m[3].strip(),
            "line_start": start + 1, "line_end": i + 1,
            "task_prompts": [{"task_id": int(t), "tokens": int(n)} for t, n in prompts],
            "generation_samples": [{"tokens": int(n), "tokens_per_second": float(t)} for n, t in generations],
            "completed_prefills": [{"ms": float(ms), "tokens": int(n), "tokens_per_second": float(t)} for ms, n, t in prefills],
            "completed_decodes": [{"ms": float(ms), "tokens": int(n), "tokens_per_second": float(t)} for ms, n, t in evaluations],
        })
        start = i + 1
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--provider-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == FROZEN
    assert not args.output.exists(), "Never overwrite diagnostic evidence"
    log_raw = args.provider_log.read_bytes()
    requests = provider_requests(log_raw.decode(errors="replace").splitlines())
    report = {"implementation": FROZEN, "diagnostic_only": True,
              "log_sha256_at_read": digest(log_raw), "cases": {}, "provenance": {}}
    for sha in ("7062f48", "c149f73"):
        path = args.evidence_root / sha / "per_query.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        manifest = json.loads((path.parent / "manifest.json").read_text())
        report["provenance"][sha] = {"per_query_sha256": digest(path.read_bytes()),
                                      "model": manifest["model_provenance"],
                                      "pytest": manifest["pytest_version"]}
        end = datetime.fromisoformat(manifest["measurement_finished_at_utc"])
        for row in reversed(rows):
            if row["id"] in CASES:
                entry = packet_stats(row)
                lower = end - timedelta(milliseconds=row["timings_ms"]["analyst"])
                entry["estimated_stage_end_utc"] = end.isoformat()
                entry["provider_requests"] = [r for r in requests if lower - timedelta(seconds=2) < datetime.fromisoformat(r["end_local"]) <= end + timedelta(seconds=2)]
                # The request ending at the stage's lower boundary belongs to retrieval.
                entry["provider_requests"] = [r for r in entry["provider_requests"] if abs((datetime.fromisoformat(r["end_local"]) - lower).total_seconds()) > 2]
                report["cases"].setdefault(row["id"], {})[sha] = entry
            end -= timedelta(milliseconds=row["timings_ms"]["total"])
    for case, entries in report["cases"].items():
        old, new = entries["7062f48"], entries["c149f73"]
        entries["comparison"] = {
            "exact_user_prompt_match": old["user_prompt"]["sha256"] == new["user_prompt"]["sha256"],
            "user_prompt_char_delta": new["user_prompt"]["chars"] - old["user_prompt"]["chars"],
            "context_ids_match": [c["id"] for c in old["contexts"]] == [c["id"] for c in new["contexts"]],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for case, e in sorted(report["cases"].items()):
        print(case, json.dumps(e["comparison"]))
        for sha in ("7062f48", "c149f73"):
            x = e[sha]
            print(sha, "prompt", x["user_prompt"]["chars"], "tools", x["tool_call_counts"],
                  "calls", [(r["end_local"], r["reported_duration"], r["task_prompts"], r["completed_decodes"] or r["generation_samples"][-1:]) for r in x["provider_requests"]])


if __name__ == "__main__":
    main()
