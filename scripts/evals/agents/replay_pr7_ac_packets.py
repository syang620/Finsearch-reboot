"""Guarded, one-batch AC/awake comparison around the original diagnostic runner.

No product runtime or previous diagnostic runner is modified. Power checks occur
outside AnalystAgent.arun; the original model observer and timeout stay intact.
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

import replay_pr7_timeout_packets as replay


def command(*args):
    return subprocess.check_output(args, text=True)


def power():
    battery = command("pmset", "-g", "batt")
    settings = command("pmset", "-g")
    return {"utc": replay.now(), "battery": battery, "active_settings": settings,
            "ac": "Now drawing from 'AC Power'" in battery,
            "low_power_off": bool(re.search(r"^\s*lowpowermode\s+0\s*$", settings, re.M))}


def workload():
    raw = command("top", "-l", "2", "-s", "1", "-n", "12", "-o", "cpu",
                  "-stats", "pid,command,cpu,mem")
    last = "Processes:" + raw.split("Processes:")[-1]
    idle = float(re.search(r"([\d.]+)% idle", last)[1])
    processes = []
    for line in last.splitlines():
        m = re.match(r"\s*(\d+)\s+(.+?)\s+([\d.]+)\s+\S+\s*$", line)
        if m:
            processes.append({"pid": int(m[1]), "name": m[2], "cpu_percent_one_core": float(m[3])})
    return {"utc": replay.now(), "cpu_idle_percent": idle, "processes": processes,
            "sample": last, "memory_pressure_raw": command("sysctl", "-n", "kern.memorystatus_vm_pressure_level").strip()}


async def main(args):
    assert not args.output_dir.exists(), "Never overwrite a diagnostic batch"
    assert not args.control_dir.exists(), "Never overwrite control evidence"
    args.control_dir.mkdir(parents=True)
    samples = []
    original_agent = replay.AnalystAgent
    stop = asyncio.Event()
    keep_awake = subprocess.Popen(["caffeinate", "-dims", "-w", str(os.getpid())])
    monitor_task = None

    def save_sample(sample):
        samples.append(sample)
        with (args.control_dir / "system_samples.jsonl").open("a") as f:
            f.write(json.dumps(replay.safe(sample)) + "\n")

    async def monitor():
        while not stop.is_set():
            sample = {"phase": "during", "power": await asyncio.to_thread(power),
                      "workload": await asyncio.to_thread(workload)}
            save_sample(sample)
            try:
                await asyncio.wait_for(stop.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass

    class GuardedAgent(original_agent):
        async def arun(self, packet, **kwargs):
            check = await asyncio.to_thread(power)
            save_sample({"phase": "case_boundary", "power": check})
            assert check["ac"] and check["low_power_off"], "Power condition changed; stop before another case"
            assert all(s["power"]["ac"] and s["power"]["low_power_off"] for s in samples), "Power control was lost"
            return await super().arun(packet, **kwargs)

    try:
        # Operational preflight, not a product or release threshold: two quiet
        # samples 20 seconds apart; never stop or modify other user processes.
        quiet = 0
        for _ in range(12):
            p = await asyncio.to_thread(power)
            assert p["ac"] and p["low_power_off"], "AC power / Low Power Mode precondition not met"
            w = await asyncio.to_thread(workload)
            save_sample({"phase": "preflight", "power": p, "workload": w})
            busiest = max((v["cpu_percent_one_core"] for v in w["processes"]), default=0)
            quiet = quiet + 1 if w["cpu_idle_percent"] >= 85 and busiest < 50 else 0
            print(json.dumps({"preflight_idle_percent": w["cpu_idle_percent"], "busiest_process_cpu": busiest,
                              "consecutive_quiet_samples": quiet}), flush=True)
            if quiet == 2:
                break
            await asyncio.sleep(20)
        assert quiet == 2, "Background workloads did not settle; no analyst cases started"
        control = {"started_at_utc": replay.now(), "caffeinate_pid": keep_awake.pid,
                   "awake_assertions": command("pmset", "-g", "assertions"),
                   "preflight_rule": "Two samples 20 seconds apart: >=85% CPU idle, no sampled process >=50% of one core.",
                   "conditions": "AC power, active Low Power Mode off, caffeinate -dims; one sequential batch of the same six packets.",
                   "monitor": "Read-only power and one-second top samples approximately every 30 seconds; no workloads stopped.",
                   "runtime_changes": False, "repeated_trials": 0, "full_gate_runs": 0}
        assert f"pid {keep_awake.pid}(caffeinate)" in control["awake_assertions"]
        (args.control_dir / "provenance.json").write_text(json.dumps(replay.safe(control), indent=2) + "\n")
        replay.AnalystAgent = GuardedAgent
        monitor_task = asyncio.create_task(monitor())
        await replay.main(args)
    finally:
        stop.set()
        if monitor_task is not None:
            await monitor_task
        replay.AnalystAgent = original_agent
        save_sample({"phase": "postflight", "power": await asyncio.to_thread(power),
                     "workload": await asyncio.to_thread(workload)})
        keep_awake.terminate()
        keep_awake.wait(timeout=5)
        completion = {"finished_at_utc": replay.now(), "awake_guard_released": True,
                      "power_conditions_held_in_samples": all(s["power"]["ac"] and s["power"]["low_power_off"] for s in samples),
                      "runtime_head": command("git", "rev-parse", "HEAD").strip(),
                      "runtime_diff": command("git", "diff", replay.FROZEN, "--", "src")}
        (args.control_dir / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-query", type=Path, required=True)
    p.add_argument("--provider-log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
