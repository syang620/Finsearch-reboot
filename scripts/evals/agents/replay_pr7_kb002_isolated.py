"""One KB_002-only diagnostic with browser-absence and power/load controls.

Reuses the original observer/analyst runner without changing product code, input,
timeout, model options or retries. Invalidated controls do not become test passes.
"""
import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import replay_pr7_timeout_packets as replay
from replay_pr7_ac_packets import command, power, workload

CASE = "AGENT_V1_KB_002"


def browsers():
    found = []
    for line in command("ps", "-Ao", "pid=,comm=").splitlines():
        if "/Safari.app/" in line or "/Google Chrome.app/" in line:
            pid, path = line.strip().split(None, 1)
            found.append({"pid": int(pid), "name": Path(path).name})
    return found


async def main(a):
    assert command("git", "rev-parse", "HEAD").strip() == replay.FROZEN
    assert not command("git", "diff", replay.FROZEN, "--", "src", "requirements.txt", "pyproject.toml")
    assert not a.output_dir.exists(), "Never overwrite diagnostic evidence"
    lines = a.per_query.read_text().splitlines()
    line = next(x for x in lines if json.loads(x)["id"] == CASE)
    a.output_dir.mkdir(parents=True)
    singleton = a.output_dir / "captured_case.jsonl"
    singleton.write_text(line + "\n")
    assert json.loads(line) == json.loads(singleton.read_text())
    samples, violations = [], []
    stop = asyncio.Event()
    guard = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
    original_cases = replay.CASES
    runner = monitor_task = None
    state = "preflight"

    async def sample(phase):
        p = await asyncio.to_thread(power)
        b = await asyncio.to_thread(browsers)
        w = await asyncio.to_thread(workload)
        data = {"phase": phase, "power": p, "browsers": b, "workload": w}
        samples.append(data)
        with (a.output_dir / "system_samples.jsonl").open("a") as f:
            f.write(json.dumps(replay.safe(data)) + "\n")
        return data

    def invalid(sample):
        return not sample["power"]["ac"] or not sample["power"]["low_power_off"] or bool(sample["browsers"])

    def other_busy(sample):
        return [p for p in sample["workload"]["processes"]
                if p["pid"] != os.getpid() and p["name"] not in {"llama-server", "ollama", "top", "caffeinate"}
                and p["cpu_percent_one_core"] >= 50]

    async def monitor():
        busy_streak = 0
        while not stop.is_set():
            s = await sample("during")
            busy_streak = busy_streak + 1 if other_busy(s) else 0
            if invalid(s) or busy_streak >= 2:
                violations.append({"utc": s["power"]["utc"], "power_or_browser_invalid": invalid(s),
                                   "sustained_other_busy": other_busy(s) if busy_streak >= 2 else []})
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass

    try:
        quiet = 0
        for _ in range(24):
            s = await sample("preflight")
            assert not invalid(s), "AC/LPM/browser precondition not met; no case started"
            quiet = quiet + 1 if s["workload"]["cpu_idle_percent"] >= 90 and not other_busy(s) else 0
            print(json.dumps({"preflight_idle_percent": s["workload"]["cpu_idle_percent"], "quiet_samples": quiet}), flush=True)
            if quiet == 2:
                break
            await asyncio.sleep(10)
        assert quiet == 2, "Background work did not settle; no case started"
        provenance = {"started_at_utc": replay.now(), "implementation": replay.FROZEN,
                      "case_id": CASE, "runs_authorized": 1, "full_gate_runs": 0,
                      "original_per_query_sha256": replay.digest(a.per_query.read_bytes()),
                      "singleton_sha256": replay.digest(singleton.read_bytes()),
                      "packet_sha256": replay.digest(json.loads(line)["trace"]["evaluation_trace"]["analyst_packet"]),
                      "runner_sha256": replay.digest(Path(replay.__file__).read_bytes()),
                      "wrapper_sha256": replay.digest(Path(__file__).read_bytes()),
                      "caffeinate_pid": guard.pid, "experiment_pid": os.getpid(),
                      "awake_assertions": command("pmset", "-g", "assertions"),
                      "preflight": "AC, active Low Power Mode off, Safari/Chrome absent; two >=90% CPU-idle samples, 10 seconds apart, no other process >=50% of one core.",
                      "monitor": "Approximately 12-second power/browser/load samples; invalidate on power/browser loss or two consecutive other-process >=50% CPU samples. No runtime settings changed.",
                      "app_action": "Safari and Chrome were quit normally before preflight; no force quit, no profile deletion, no system services terminated.",
                      "input_selection": "One original JSONL row copied verbatim; prior diagnostic runner CASES bound to singleton only for this invocation."}
        assert f"pid {guard.pid}(caffeinate)" in provenance["awake_assertions"]
        (a.output_dir / "control_provenance.json").write_text(json.dumps(replay.safe(provenance), indent=2) + "\n")
        replay.CASES = [CASE]
        state = "running"
        monitor_task = asyncio.create_task(monitor())
        runner = asyncio.create_task(replay.main(SimpleNamespace(per_query=singleton, provider_log=a.provider_log,
                                                                output_dir=a.output_dir / "run")))
        await asyncio.wait({runner, monitor_task}, return_when=asyncio.FIRST_COMPLETED)
        if monitor_task.done():
            await monitor_task
            if violations:
                state = "invalidated"
                runner.cancel()
                try:
                    await runner
                except asyncio.CancelledError:
                    pass
                raise RuntimeError("Isolation control lost; trial invalidated, no automatic retry")
        await runner
        state = "completed"
    finally:
        stop.set()
        if monitor_task is not None and not monitor_task.done():
            await monitor_task
        replay.CASES = original_cases
        await sample("postflight")
        guard.terminate()
        guard.wait(timeout=5)
        finish = {"finished_at_utc": replay.now(), "trial_state": state, "violations": violations,
                  "all_sampled_power_browser_conditions_valid": all(not invalid(s) for s in samples),
                  "awake_guard_released": True, "runtime_head": command("git", "rev-parse", "HEAD").strip(),
                  "runtime_diff": command("git", "diff", replay.FROZEN, "--", "src")}
        (a.output_dir / "control_completion.json").write_text(json.dumps(finish, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-query", type=Path, required=True)
    p.add_argument("--provider-log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
