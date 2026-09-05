"""Verify and summarize the single controlled diagnostic batch, not a gate."""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import convert_to_openai_messages
from agents.analyst.agent import AnalystAgent, SYSTEM_PROMPT, build_analyst_prompt
from agents.contracts import AnalystPacket
from investigate_pr7_timeouts import CASES, FROZEN, digest, provider_requests


def main(a):
    assert not a.output.exists(), "Never overwrite comparison evidence"
    rp = a.diagnostic_root / "analyst-replay-ac-01"
    control = a.diagnostic_root / "ac-control-01"
    provenance = json.loads((rp / "provenance.json").read_text())
    ending = json.loads((rp / "completion.json").read_text())
    power_end = json.loads((control / "completion.json").read_text())
    samples = [json.loads(x) for x in (control / "system_samples.jsonl").read_text().splitlines()]
    replays = [json.loads(x) for x in (rp / "per_case.jsonl").read_text().splitlines()]
    frozen = {x["id"]: x for x in map(json.loads, a.per_query.read_text().splitlines())}
    old = json.loads((a.diagnostic_root / "comparison.json").read_text())
    assert [r["id"] for r in replays] == CASES
    assert provenance["source_per_query_sha256"] == digest(a.per_query.read_bytes())
    assert ending["head"] == FROZEN and ending["runtime_diff"] == ""
    assert power_end["power_conditions_held_in_samples"] and power_end["awake_guard_released"]
    assert not subprocess.check_output(["git", "diff", FROZEN, "--", "src", "requirements.txt", "pyproject.toml"])
    raw = a.provider_log.read_bytes()
    start, end = provenance["provider_log_start_bytes"], ending["provider_log_end_bytes"]
    offset = raw[:start].count(b"\n")
    requests = provider_requests(raw[start:end].decode().splitlines())
    for r in requests:
        r["line_start"] += offset
        r["line_end"] += offset
    rows = []
    for replay in replays:
        case = replay["id"]
        packet_raw = frozen[case]["trace"]["evaluation_trace"]["analyst_packet"]
        packet = AnalystPacket.model_validate(packet_raw)
        agent = AnalystAgent(model="ollama/qwen2.5:14b-instruct")
        bound = agent._build_bound_model(packet, tools_available=True)
        messages = convert_to_openai_messages([SystemMessage(SYSTEM_PROMPT), HumanMessage(build_analyst_prompt(packet))])
        first = replay["provider_calls"][0]
        assert digest(packet_raw) == replay["packet_sha256"]
        assert digest(messages) == first["messages_sha256"]
        assert digest(bound.bound_tools) == first["tool_schema_sha256"]
        calls = []
        for call in replay["provider_calls"]:
            assert call["timeout_s"] == 120
            assert call["options"] == {"temperature": 0.0, "num_predict": agent._num_predict_for_task(packet.analysis_task.task_type)}
            match = [r for r in requests if abs((datetime.fromisoformat(r["end_local"]) - datetime.fromisoformat(call["finished_at_utc"])).total_seconds()) < 2]
            assert len(match) == 1, (case, call["finished_at_utc"])
            response = call.get("provider_response") or {}
            generation_ns = response.get("eval_duration")
            tokens = response.get("eval_count")
            calls.append({"elapsed_s": call["elapsed_ms"] / 1000, "error": call.get("error"),
                          "prompt_tokens": response.get("prompt_eval_count"),
                          "generated_tokens": tokens,
                          "generation_s": generation_ns / 1e9 if generation_ns is not None else None,
                          "tokens_per_second": tokens / (generation_ns / 1e9) if generation_ns else None,
                          "provider_total_s": response.get("total_duration", 0) / 1e9 if response else None,
                          "prefill_s": response.get("prompt_eval_duration", 0) / 1e9 if response else None,
                          "load_s": response.get("load_duration", 0) / 1e9 if response else None,
                          "provider_log": match[0]})
        assert calls[0]["provider_log"]["task_prompts"][0]["tokens"] == old["cases"][case]["c149f73"]["provider_requests"][0]["task_prompts"][0]["tokens"]
        timeout = "TIMEOUT" in str(replay["result"].get("error", "")).upper() or any("TIMEOUT" in str(c.get("error", "")).upper() for c in calls)
        rows.append({"id": case, "timeout": timeout, "status": replay["result"]["status"],
                     "ok": replay["result"]["ok"], "error": replay["result"].get("error"),
                     "case_elapsed_s": replay["elapsed_ms"] / 1000,
                     "analyst_timing_ms": replay["result"]["trace"]["timing_ms"],
                     "model_calls": calls, "all_completed_generation_s": sum(c["generation_s"] or 0 for c in calls),
                     "generation_total_is_censored": any(c["generation_s"] is None for c in calls),
                     "input_and_settings_exact": True})
    summary = {"diagnostic_only": True, "implementation": FROZEN, "case_count": len(rows),
               "timeout_free_cases": sum(not r["timeout"] for r in rows),
               "ok_cases": sum(r["ok"] for r in rows), "cases": rows,
               "power_conditions_held_in_samples": True,
               "power_sample_count": len(samples), "awake_guard_released": True,
               "provider_log_slice_sha256": digest(raw[start:end]),
               "provider_request_count": len(requests),
               "source_per_query_sha256": digest(a.per_query.read_bytes()),
               "all_initial_inputs_and_settings_exact": True,
               "runtime_diff": "", "full_gate_runs": 0, "repeated_trials": 0,
               "limits": "Timeout calls have censored generation durations; log samples are partial rates, not completed output rates. Power/load sampled, not continuously profiled."}
    assert len(requests) == sum(len(r["model_calls"]) for r in rows)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}))
    for row in rows:
        print(row["id"], row["status"], row["case_elapsed_s"],
              [(c["elapsed_s"], c["generation_s"], c["tokens_per_second"]) for c in row["model_calls"]])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic-root", type=Path, required=True)
    p.add_argument("--per-query", type=Path, required=True)
    p.add_argument("--provider-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
