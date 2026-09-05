"""Capture path-redacted source, prompt-order, and machine-log support evidence."""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from investigate_pr7_timeouts import CASES, FROZEN, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--provider-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    sources = ["src/agents/analyst/agent.py", "src/agents/analyst/grounding.py",
               "src/agents/contracts.py", "src/llm_client.py", "src/structured_facts/capabilities.py"]
    refs = ["7062f48", "2860c43", FROZEN]
    source_hashes = {path: {ref: digest(subprocess.check_output(["git", "show", f"{ref}:{path}"])) for ref in refs} for path in sources}
    rows = {}
    for sha in ("7062f48", "c149f73"):
        rows[sha] = {r["id"]: r for r in map(json.loads, (a.evidence_root / sha / "per_query.jsonl").read_text().splitlines())}
    comparisons = {}
    for case in CASES:
        packets = {s: rows[s][case]["trace"]["evaluation_trace"]["analyst_packet"] for s in rows}
        old, new = packets.values()
        contexts = {}
        for s, packet in packets.items():
            contexts[s] = [{"context_id": c["context_id"], "source": c.get("source"),
                            "payload_sha256": digest(c.get("payload"))} for c in packet["context_items"]]
        comparisons[case] = {"packet_sha256": {s: digest(p) for s, p in packets.items()},
                             "changed_packet_fields": [k for k in set(old) | set(new) if old.get(k) != new.get(k)],
                             "contexts": contexts, "route": rows["c149f73"][case]["route"]}
    power = subprocess.check_output(["pmset", "-g", "log"], text=True)
    events = [line for line in power.splitlines() if re.match(r"2026-09-04 (18|19):", line)
              and any(term in line for term in ("Entering Sleep state", "Wake from Deep Idle"))]
    log_lines = a.provider_log.read_text().splitlines()
    service = [{"line": i + 1, "text": line} for i, line in enumerate(log_lines)
               if (42363 <= i + 1 <= 42600 or 46211 <= i + 1 <= 46450 or 47235 <= i + 1 <= 47738)
               and any(term in line for term in ("system memory", "gpu memory", "evicting", "offloaded", "n_ctx =", "n_ctx_per_seq =", "n_batch =", "n_ubatch ="))]
    # Preserve only numeric prompt/decoding/cancellation evidence, never server startup env.
    comparison = json.loads((a.output.parent / "comparison.json").read_text())
    excerpts = {}
    for case, entry in comparison["cases"].items():
        excerpts[case] = {}
        for sha in rows:
            excerpt = []
            for request in entry[sha]["provider_requests"]:
                for i in range(request["line_start"] - 1, min(request["line_end"] + 3, len(log_lines))):
                    line = log_lines[i]
                    if any(t in line for t in ("new prompt,", "prompt processing,", "prompt eval time", "        eval time", "n_gen =", "cancel task,", "stop processing:", "[GIN]")):
                        excerpt.append({"line": i + 1, "text": line})
            excerpts[case][sha] = excerpt
    result = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "source_hashes": source_hashes,
              "all_selected_sources_identical": all(len(set(v.values())) == 1 for v in source_hashes.values()),
              "packet_comparisons": comparisons, "power_log_sha256_at_read": digest(power.encode()),
              "sleep_wake_events": events, "service_load_records": service, "provider_excerpts": excerpts,
              "current_thermal_status": subprocess.check_output(["pmset", "-g", "therm"], text=True),
              "limits": ["Current power/thermal readings are not historical measurements.",
                         "Historical per-call thermal, CPU/GPU frequency and process-load samples were not captured.",
                         "Provider request timestamps are matched from final measurement time minus sequential case durations (two-second tolerance).",
                         "Only control-case root checkpoints survive in both SQLite checkpoint stores; analyst message checkpoints are unavailable."]}
    text = json.dumps(result, indent=2) + "\n"
    assert "/Users/" not in text
    a.output.write_text(text)
    print(json.dumps({"all_selected_sources_identical": result["all_selected_sources_identical"],
                      "sleep_wake_events": events, "changed_fields": {k: v["changed_packet_fields"] for k, v in comparisons.items()}}))


if __name__ == "__main__":
    main()
