import json
from pathlib import Path

from scripts.diagnostics import observe_semantic_workload as observer


def row(pid, cpu, executable, ppid=1):
    return {
        "pid": pid,
        "ppid": ppid,
        "cpu": cpu,
        "memory": 1.0,
        "start_time": "Mon Sep 8 10:00:00 2026",
        "executable": executable,
    }


def test_details_matches_frozen_browser_and_heavy_rules(monkeypatch):
    monkeypatch.setattr(observer, "cwd", lambda pid: "/repo")
    monkeypatch.setattr(observer, "command_line", lambda pid: "command")
    rows = {
        1: row(1, 1, "/sbin/launchd", 0),
        2: row(2, 2, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        3: row(3, 50, "/usr/bin/git", 2),
        4: row(4, 99, "/usr/local/bin/ollama"),
        5: row(5, 49.9, "/usr/bin/python"),
    }
    found = observer.details(rows)
    assert [(r["pid"], r["categories"]) for r in found] == [
        (2, ["browser"]),
        (3, ["heavy_non_model"]),
    ]
    assert found[1]["ancestors"][1]["pid"] == 2


def test_summary_classifies_single_short_sustained_and_recurring():
    samples = []
    for index in range(12):
        items = []
        if index == 0:
            items.append(row(10, 60, "/usr/bin/one"))
        if index in (1, 2):
            items.append(row(11, 70, "/usr/bin/short"))
        if index < 10:
            items.append(row(12, 80, "/usr/bin/sustained"))
        if index in (1, 4, 8):
            items.append(row(13, 90, "/usr/bin/periodic"))
        for item in items:
            item.update(categories=["heavy_non_model"], command_line=None,
                        working_directory=None, ancestors=[])
        samples.append({"index": index, "observed_at": f"t{index}",
                        "detailed_violating_processes": items})
    shapes = {item["pid"]: item["shape"] for item in observer.summarize(samples)}
    assert shapes == {10: "isolated_single_sample", 11: "short_burst",
                      12: "sustained_competing_workload", 13: "recurring_periodic_process"}


def test_sanitize_redacts_user_home_recursively():
    value = {
        "command": [
            f"{observer.PRIVATE_PREFIX}/bin/git",
            "person@example.com",
            "safe",
        ]
    }
    assert observer.sanitize(value) == {
        "command": ["$USER_HOME/bin/git", "$EMAIL", "safe"]
    }


def test_published_summary_reproduces_from_raw_observation():
    root = Path("artifacts/diagnostics/semantic-v2-workload-controls/a252feeb204ae417e2e024788cfbbff941b319ce")
    rows = [json.loads(line) for line in (root / "formal_observation.jsonl").read_text().splitlines()]
    samples = [row for row in rows if row["type"] == "sample"]
    summary = json.loads((root / "summary.json").read_text())
    observed = summary["observation"]
    assert len(samples) == observed["sample_count"] == 1200
    assert sum(bool(row["frozen_control"].get("browser_process_count") or
                    row["frozen_control"].get("heavy_non_model_processes"))
               for row in samples) == observed["frozen_control_violation_samples"] == 235
    assert sum(row["frozen_control"].get("browser_process_count", 0) > 0
               for row in samples) == observed["browser_process_samples"] == 0
    assert sum(row["frozen_control"].get("ac_power") is not True
               for row in samples) == observed["ac_power_failure_samples"] == 0
    assert sum(row["frozen_control"].get("low_power_mode") != 0
               for row in samples) == observed["low_power_mode_failure_samples"] == 0
    counts = {}
    for row in samples:
        for process in row["frozen_control"].get("heavy_non_model_processes", []):
            counts[process["process"]] = counts.get(process["process"], 0) + 1
    assert counts == {row["process"]: row["samples"]
                      for row in summary["authoritative_frozen_control_process_occurrences"]}
