"""Observe the frozen workload control without running benchmark cases.

The imported ``controls`` function is the authoritative pass/fail observation.
The richer process snapshot is diagnostic metadata captured immediately after it;
it does not replace or alter the frozen logic or threshold.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.evals.retrieval.run_benchmark_v3 import controls

THRESHOLD = 50.0
BROWSER_MARKERS = ("Google Chrome", "/Safari.app/")
EXEMPT_MARKERS = ("ollama", "qdrant", "com.docker", "virtualization")
PRIVATE_PREFIX = str(Path.home())
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")


def now():
    return datetime.now(timezone.utc).isoformat()


def sanitize(value):
    if isinstance(value, str):
        value = value.replace(PRIVATE_PREFIX, "$USER_HOME")
        return EMAIL_PATTERN.sub("$EMAIL", value)
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value


def process_table():
    rows = {}
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,pcpu=,pmem=,lstart=,comm="], text=True
    )
    for line in output.splitlines():
        parts = line.strip().split(None, 9)
        if len(parts) != 10:
            continue
        try:
            row = {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "cpu": float(parts[2]),
                "memory": float(parts[3]),
                "start_time": " ".join(parts[4:9]),
                "executable": parts[9],
            }
        except ValueError:
            continue
        rows[row["pid"]] = row
    return rows


def cwd(pid):
    try:
        output = subprocess.check_output(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return next((line[1:] for line in output.splitlines() if line.startswith("n")), None)


def ancestor_chain(pid, rows):
    chain = []
    seen = set()
    current = rows.get(pid)
    while current and current["pid"] not in seen and len(chain) < 16:
        seen.add(current["pid"])
        chain.append(
            {key: current[key] for key in ("pid", "ppid", "executable", "start_time")}
        )
        current = rows.get(current["ppid"])
    return chain


def details(rows):
    found = []
    for row in rows.values():
        name = row["executable"]
        browser = any(marker in name for marker in BROWSER_MARKERS)
        heavy = row["cpu"] >= THRESHOLD and not any(
            marker in name.lower() for marker in EXEMPT_MARKERS
        )
        if not browser and not heavy:
            continue
        item = dict(row)
        item["categories"] = [
            category
            for category, present in (("browser", browser), ("heavy_non_model", heavy))
            if present
        ]
        item["command_line"] = command_line(row["pid"])
        item["working_directory"] = cwd(row["pid"])
        item["ancestors"] = ancestor_chain(row["pid"], rows)
        found.append(item)
    return sorted(found, key=lambda item: item["pid"])


def command_line(pid):
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def sample(index):
    observed_at = now()
    frozen = controls()
    rows = process_table()
    enriched = sanitize(details(rows))
    return {
        "type": "sample",
        "index": index,
        "observed_at": observed_at,
        "frozen_control": frozen,
        "detailed_violating_processes": enriched,
        "detail_temporal_limitation": (
            "Frozen control and rich process table are sequential snapshots; a very short process may exit between them."
        ),
    }


def summarize(samples):
    groups = {}
    for record in samples:
        for item in record["detailed_violating_processes"]:
            key = (item["pid"], item["start_time"], tuple(item["categories"]))
            group = groups.setdefault(
                key,
                {
                    "pid": item["pid"],
                    "start_time": item["start_time"],
                    "executable": item["executable"],
                    "categories": item["categories"],
                    "first_observed": record["observed_at"],
                    "last_observed": record["observed_at"],
                    "samples": 0,
                    "peak_cpu": 0.0,
                    "peak_memory": 0.0,
                    "burst_starts": [],
                    "max_consecutive_samples": 0,
                    "_last_index": None,
                    "_consecutive": 0,
                    "last_detail": item,
                },
            )
            if group["_last_index"] != record["index"] - 1:
                group["_consecutive"] = 0
                group["burst_starts"].append(record["observed_at"])
            group["_consecutive"] += 1
            group["_last_index"] = record["index"]
            group["max_consecutive_samples"] = max(
                group["max_consecutive_samples"], group["_consecutive"]
            )
            group["samples"] += 1
            group["last_observed"] = record["observed_at"]
            group["peak_cpu"] = max(group["peak_cpu"], item["cpu"])
            group["peak_memory"] = max(group["peak_memory"], item["memory"])
            group["last_detail"] = item
    result = []
    for group in groups.values():
        bursts = len(group["burst_starts"])
        if group["samples"] == 1:
            shape = "isolated_single_sample"
        elif bursts >= 3:
            shape = "recurring_periodic_process"
        elif group["max_consecutive_samples"] >= 10:
            shape = "sustained_competing_workload"
        else:
            shape = "short_burst"
        for private in ("_last_index", "_consecutive"):
            del group[private]
        group["shape"] = shape
        result.append(group)
    return sorted(result, key=lambda item: (item["first_observed"], item["pid"]))


def run(output, duration_seconds, interval_seconds):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    started = time.monotonic()
    samples = []
    with output.open("x") as stream:
        header = {
            "type": "header",
            "started_at": now(),
            "duration_seconds": duration_seconds,
            "interval_seconds": interval_seconds,
            "threshold_percent": THRESHOLD,
            "browser_markers": BROWSER_MARKERS,
            "exempt_markers": EXEMPT_MARKERS,
            "observer_pid": os.getpid(),
            "awake_pid": awake.pid,
            "privacy": "User-home prefixes are replaced with $USER_HOME; other command-line content is retained.",
            "policy": "No benchmark cases, model inference, benchmark or semantic-search retrieval, external-process termination, or control modification.",
        }
        stream.write(json.dumps(header) + "\n")
        try:
            index = 0
            while time.monotonic() - started < duration_seconds:
                record = sample(index)
                samples.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                index += 1
                target = started + index * interval_seconds
                time.sleep(max(0, target - time.monotonic()))
        finally:
            awake.terminate()
            awake.wait(timeout=10)
            footer = {
                "type": "footer",
                "finished_at": now(),
                "sample_count": len(samples),
                "frozen_control_violation_samples": sum(
                    bool(row["frozen_control"].get("browser_process_count") or row["frozen_control"].get("heavy_non_model_processes"))
                    for row in samples
                ),
                "process_summaries": summarize(samples),
                "owned_process_cleanup": "Observer terminates only its own caffeinate child after the final sample; no pre-existing or competing workload process is terminated.",
            }
            stream.write(json.dumps(footer) + "\n")
    print(json.dumps({"output": str(output), "sample_count": len(samples)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=1200)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    run(args.output, args.duration_seconds, args.interval_seconds)
