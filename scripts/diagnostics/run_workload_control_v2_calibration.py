"""Run one preregistered workload-control-v2 scenario without semantic cases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from scripts.diagnostics.observe_semantic_workload import (
    ancestor_chain,
    command_line,
    cwd,
    process_table,
    sanitize,
)
from scripts.diagnostics.workload_control_v2 import (
    classify_sample,
    evaluate_samples,
    load_preregistration,
    sha256,
    summarize_evaluation,
)
from scripts.evals.retrieval.run_benchmark_v3 import controls


DEFAULT_PREREGISTRATION = Path("docs/evals/workload_control_v2_preregistration.json")
BROWSER_MARKERS = ("/Google Chrome.app/", "/Safari.app/")
DETAIL_MARKERS = (
    "chatgpt",
    "codex",
    "docker",
    "ollama",
    "qdrant",
    "control_v2",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def capture_processes(observer_pid, controlled_pids):
    rows = process_table()
    result = []
    for row in rows.values():
        executable = row["executable"]
        lower = executable.lower()
        browser = any(marker.lower() in lower for marker in BROWSER_MARKERS)
        if row["cpu"] <= 0 and not browser:
            continue
        detailed = row["cpu"] >= 5 or browser or any(marker in lower for marker in DETAIL_MARKERS)
        item = dict(row)
        item["process_group_id"] = process_group_id(row["pid"])
        item["command_line"] = command_line(row["pid"]) if detailed else None
        item["working_directory"] = (
            cwd(row["pid"]) if row["cpu"] >= 50 or browser or row["pid"] in controlled_pids else None
        )
        item["ancestors"] = ancestor_chain(row["pid"], rows)
        item["controlled_external"] = row["pid"] in controlled_pids
        item["observer_owned"] = (
            row["pid"] == observer_pid
            or row["ppid"] == observer_pid and row["pid"] not in controlled_pids
        )
        result.append(item)
    return sanitize(sorted(result, key=lambda item: item["pid"]))


def process_group_id(pid):
    try:
        value = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "pgid="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(value) if value else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def write_record(stream, record):
    stream.write(json.dumps(sanitize(record), sort_keys=True) + "\n")
    stream.flush()


def start_busy(label, duration_seconds):
    code = (
        "import time; end=time.monotonic()+float(__import__('sys').argv[1]); "
        "x=0; "
        "exec(\"while time.monotonic()<end:\\n x=(x+1)%1000003\")"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code, str(duration_seconds), f"CONTROL_V2_BUSY_{label}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class ScenarioController:
    def __init__(self, scenario, stream, index_manifest, preflight_output):
        self.scenario = scenario
        self.stream = stream
        self.index_manifest = index_manifest
        self.preflight_output = preflight_output
        self.started = set()
        self.children = {}
        self.child_keys = {}
        self.events = []
        self.browser_profile = None

    @property
    def controlled_pids(self):
        return {pid for pid, child in self.children.items() if child.poll() is None}

    def event(self, kind, elapsed, **details):
        record = {"type": "event", "event": kind, "at": now(), "elapsed_seconds": elapsed, **details}
        self.events.append(record)
        write_record(self.stream, record)

    def _start_child(self, key, child, elapsed, category):
        self.started.add(key)
        self.children[child.pid] = child
        self.child_keys[child.pid] = key
        self.event("workload_start", elapsed, key=key, pid=child.pid, category=category)

    def poll(self, elapsed):
        if self.scenario == "S3_REQUIRED_SERVICE_ACTIVITY" and elapsed >= 15 and "preflight" not in self.started:
            if not self.index_manifest or not self.preflight_output:
                raise ValueError("Scenario 3 requires --index-manifest and --preflight-output")
            if self.preflight_output.exists():
                raise FileExistsError(self.preflight_output)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "scripts/diagnostics/simulate_semantic_preflight_readonly.py",
                    "--index-manifest",
                    str(self.index_manifest),
                    "--output",
                    str(self.preflight_output),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._start_child("preflight", child, elapsed, "calibration_harness")

        if self.scenario == "S4_SUSTAINED_CPU_INTERFERENCE":
            for repetition, start in enumerate((15, 75, 135), 1):
                key = f"sustained_{repetition}"
                if elapsed >= start and key not in self.started:
                    self._start_child(
                        key,
                        start_busy(f"SUSTAINED_{repetition}", 45),
                        elapsed,
                        "unrelated_external_workload",
                    )

        if self.scenario == "S5_SHORT_CPU_BURSTS":
            for repetition, start in enumerate((15, 30, 45, 60, 75, 90), 1):
                key = f"short_{repetition}"
                if elapsed >= start and key not in self.started:
                    self._start_child(
                        key,
                        start_busy(f"SHORT_{repetition}", 1.5),
                        elapsed,
                        "unrelated_external_workload",
                    )

        if self.scenario == "S6_BROWSER_WORKLOAD" and elapsed >= 10 and "browser" not in self.started:
            chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
            if not chrome.exists():
                raise FileNotFoundError(chrome)
            self.browser_profile = Path(tempfile.mkdtemp(prefix="finsearch-control-v2-chrome-"))
            page = "data:text/html,<script>setInterval(()=>{let x=0;for(let i=0;i<50000000;i++){x+=i}},0)</script>"
            child = subprocess.Popen(
                [
                    str(chrome),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    f"--user-data-dir={self.browser_profile}",
                    page,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._start_child("browser", child, elapsed, "user_browser_workload")

        if self.scenario == "S6_BROWSER_WORKLOAD" and elapsed >= 40 and "browser_stop" not in self.started:
            self.started.add("browser_stop")
            for pid, child in list(self.children.items()):
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)
                    self.event("workload_stop", elapsed, key="browser", pid=pid, returncode=child.returncode)

        for pid, child in list(self.children.items()):
            if child.poll() is not None and f"reaped_{pid}" not in self.started:
                self.started.add(f"reaped_{pid}")
                self.event(
                    "workload_exit",
                    elapsed,
                    pid=pid,
                    returncode=child.returncode,
                    key=self.child_keys[pid],
                )

    def cleanup(self, elapsed):
        for pid, child in self.children.items():
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                self.event("owned_process_cleanup", elapsed, pid=pid, returncode=child.returncode)
        if self.browser_profile:
            profile = self.browser_profile
            if profile.parent == Path(tempfile.gettempdir()) and profile.name.startswith("finsearch-control-v2-chrome-"):
                shutil.rmtree(profile)


def scenario_duration(preregistration, scenario):
    return preregistration["scenarios"][scenario]["duration_seconds"]


def run(args):
    preregistration = load_preregistration(args.preregistration)
    if args.scenario not in preregistration["calibration_order"]:
        raise ValueError(f"Unregistered scenario: {args.scenario}")
    duration = scenario_duration(preregistration, args.scenario)
    if args.raw_output.exists() or args.summary_output.exists():
        raise FileExistsError("Calibration outputs are append-only")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    samples = []
    with args.raw_output.open("x") as stream:
        write_record(
            stream,
            {
                "type": "header",
                "scenario": args.scenario,
                "created_at": now(),
                "implementation_sha": git("rev-parse", "HEAD"),
                "preregistration_sha256": sha256(args.preregistration),
                "duration_seconds": duration,
                "sample_interval_seconds": preregistration["scope"]["sample_interval_seconds"],
                "semantic_cases": 0,
                "policy": "Control-only calibration; no semantic retrieval or inference.",
                "observer_pid": os.getpid(),
                "awake_pid": awake.pid,
            },
        )
        delay = preregistration["scenarios"][args.scenario].get("pre_start_delay_seconds", 0)
        if delay:
            write_record(stream, {"type": "event", "event": "pre_start_delay", "at": now(), "seconds": delay})
            time.sleep(delay)
        started = time.monotonic()
        controller = ScenarioController(args.scenario, stream, args.index_manifest, args.preflight_output)
        try:
            index = 0
            interval = preregistration["scope"]["sample_interval_seconds"]
            while time.monotonic() - started < duration:
                elapsed = time.monotonic() - started
                controller.poll(elapsed)
                frozen = controls()
                sample = {
                    "type": "sample",
                    "scenario": args.scenario,
                    "index": index,
                    "observed_at": now(),
                    "elapsed_seconds": elapsed,
                    "ac_power": frozen.get("ac_power"),
                    "low_power_mode": frozen.get("low_power_mode"),
                    "browser_process_count": frozen.get("browser_process_count", 0),
                    "frozen_v1_cpu_violation": bool(frozen.get("heavy_non_model_processes")),
                    "frozen_v1_heavy_processes": frozen.get("heavy_non_model_processes", []),
                    "processes": capture_processes(os.getpid(), controller.controlled_pids),
                }
                sample = classify_sample(sample)
                samples.append(sample)
                write_record(stream, sample)
                index += 1
                time.sleep(max(0, started + index * interval - time.monotonic()))
        finally:
            controller.cleanup(time.monotonic() - started)
            awake.terminate()
            awake.wait(timeout=10)
            write_record(
                stream,
                {
                    "type": "footer",
                    "finished_at": now(),
                    "sample_count": len(samples),
                    "owned_process_cleanup": "Only scenario-owned workloads and the monitor-owned caffeinate child are terminated.",
                },
            )

    evaluated = evaluate_samples(samples, preregistration)
    categories = {}
    unknown_samples = 0
    retained = 0
    for row in evaluated:
        seen = set()
        for process in row["processes"]:
            retained += 1
            category = process["category"]
            categories[category] = categories.get(category, 0) + 1
            seen.add(category)
        unknown_samples += int("unknown" in seen)
    summary = {
        "status": "complete_control_only_calibration",
        "scenario": args.scenario,
        "implementation_sha": git("rev-parse", "HEAD"),
        "preregistration_sha256": sha256(args.preregistration),
        "raw_sha256": hashlib.sha256(args.raw_output.read_bytes()).hexdigest(),
        "sample_count": len(samples),
        "candidate_metrics": summarize_evaluation(evaluated),
        "process_category_occurrences": categories,
        "unknown_process_samples": unknown_samples,
        "retained_process_records": retained,
        "events": controller.events,
        "semantic_cases": 0,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scenario": args.scenario, "samples": len(samples), "summary": str(args.summary_output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
