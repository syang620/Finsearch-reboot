"""Pure candidate-policy and process-classification logic for control-v2 calibration."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path


CPU_SCORED_CATEGORIES = {
    "user_browser_workload",
    "run_supervision_ui_tooling",
    "unrelated_external_workload",
    "os_background_service",
    "unknown",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(process):
    ancestors = " ".join(
        str(item.get("executable", "")) for item in process.get("ancestors", [])
    )
    return " ".join(
        str(process.get(key) or "")
        for key in ("executable", "command_line", "working_directory")
    ).lower() + " " + ancestors.lower()


def _canonical_executable(process):
    return str(process.get("executable") or "unavailable").lower()


def _nearest_identified_ancestor(process):
    process_pid = process.get("pid")
    for ancestor in process.get("ancestors", []):
        if ancestor.get("pid") == process_pid:
            continue
        executable = str(ancestor.get("executable") or "").lower()
        basename = Path(executable).name
        if ".app/" in executable or basename in {
            "ollama",
            "qdrant",
            "com.docker.backend",
        }:
            return executable
    return "none"


def _stable_group(category, process):
    return ":".join(
        (category, _canonical_executable(process), _nearest_identified_ancestor(process))
    )


def classify_process(process):
    """Return a stable category/group without granting application-wide exemptions."""
    text = _text(process)
    executable = str(process.get("executable") or "")
    lower_executable = executable.lower()
    command = str(process.get("command_line") or "").lower()

    if "control_v2_busy_" in command:
        category = "unrelated_external_workload"
        return category, _stable_group(category, process)
    if process.get("observer_owned"):
        category = "calibration_harness"
        return category, _stable_group(category, process)
    if (
        "run_workload_control_v2_calibration.py" in command
        or "simulate_semantic_preflight_readonly.py" in command
    ):
        category = "calibration_harness"
        return category, _stable_group(category, process)
    if any(
        marker in text
        for marker in (
            "run_semantic_baseline_v2.py",
            "run_semantic_baseline_v2_1.py",
            "semantic-baseline-v2",
        )
    ):
        category = "benchmark_model_process"
        return category, _stable_group(category, process)
    real_chrome = lower_executable.endswith(
        "/google chrome.app/contents/macos/google chrome"
    )
    real_safari = lower_executable.endswith("/safari.app/contents/macos/safari")
    if real_chrome or real_safari:
        category = "user_browser_workload"
        return category, _stable_group(category, process)
    if "chatgpt.app/" in text or "codex framework.framework/" in text:
        category = "run_supervision_ui_tooling"
        return category, _stable_group(category, process)
    if "docker desktop helper (renderer).app/" in text or (
        "docker desktop.app/" in text and "com.docker.backend" not in lower_executable
    ):
        category = "unrelated_external_workload"
        return category, _stable_group(category, process)
    basename = Path(executable).name.lower()
    if basename in {"ollama", "qdrant", "com.docker.backend"}:
        category = "required_service_host"
        return category, _stable_group(category, process)
    if lower_executable.startswith(("/system/", "/usr/libexec/", "/usr/sbin/")):
        category = "os_background_service"
        return category, _stable_group(category, process)
    if process.get("controlled_external"):
        category = "unrelated_external_workload"
        return category, _stable_group(category, process)
    category = "unknown"
    return category, _stable_group(category, process)


def classify_sample(sample):
    result = deepcopy(sample)
    classified = []
    for process in sample.get("processes", []):
        item = dict(process)
        item["category"], item["group_key"] = classify_process(item)
        classified.append(item)
    result["processes"] = classified
    result["real_browser_process_count"] = sum(
        item["category"] == "user_browser_workload" for item in classified
    )
    return result


def _hard_reasons(sample, frozen_v1=False):
    reasons = []
    if sample.get("ac_power") is not True:
        reasons.append("ac_power")
    if sample.get("low_power_mode") != 0:
        reasons.append("low_power_mode")
    browser_count = (
        sample.get("browser_process_count", 0)
        if frozen_v1
        else sample.get("real_browser_process_count", 0)
    )
    if browser_count:
        reasons.append("browser")
    return reasons


def _group_cpu(sample):
    grouped = defaultdict(float)
    categories = {}
    for process in sample.get("processes", []):
        category = process.get("category")
        if category not in CPU_SCORED_CATEGORIES:
            continue
        key = process["group_key"]
        grouped[key] += max(0.0, float(process.get("cpu", 0.0)))
        categories[key] = category
    return dict(grouped), categories


class CandidateEvaluator:
    """Stateful deterministic evaluator over chronologically ordered samples."""

    def __init__(self, preregistration):
        self.preregistration = preregistration
        self.candidates = preregistration["candidate_policies"]
        self.streaks = defaultdict(lambda: defaultdict(int))
        self.occupancy = defaultdict(deque)
        self.burdens = defaultdict(deque)

    def evaluate(self, sample):
        sample = classify_sample(sample)
        grouped, categories = _group_cpu(sample)
        results = {}
        for candidate in self.candidates:
            candidate_id = candidate["id"]
            family = candidate["family"]
            hard = _hard_reasons(
                sample, frozen_v1=family == "current_instantaneous_reference"
            )
            cpu_violation = False
            trigger_groups = []
            diagnostic_value = None

            if family == "current_instantaneous_reference":
                cpu_violation = bool(sample.get("frozen_v1_cpu_violation"))
                trigger_groups = sorted(
                    key for key, cpu in grouped.items() if cpu >= 50.0
                )
            elif family == "same_process_group_consecutive_samples":
                threshold = candidate["threshold_percent"]
                active = {key for key, cpu in grouped.items() if cpu >= threshold}
                for key in set(self.streaks[candidate_id]) | set(grouped):
                    self.streaks[candidate_id][key] = (
                        self.streaks[candidate_id][key] + 1 if key in active else 0
                    )
                trigger_groups = sorted(
                    key
                    for key, count in self.streaks[candidate_id].items()
                    if count >= candidate["consecutive_samples"]
                )
                cpu_violation = bool(trigger_groups)
                diagnostic_value = max(self.streaks[candidate_id].values(), default=0)
            elif family == "rolling_window_occupancy":
                occupied = any(
                    cpu >= candidate["threshold_percent"] for cpu in grouped.values()
                )
                window = self.occupancy[candidate_id]
                window.append(occupied)
                while len(window) > candidate["window_samples"]:
                    window.popleft()
                occupied_count = sum(window)
                cpu_violation = occupied_count >= candidate["required_occupied_samples"]
                trigger_groups = sorted(
                    key
                    for key, cpu in grouped.items()
                    if cpu >= candidate["threshold_percent"]
                )
                diagnostic_value = occupied_count
            elif family == "rolling_cpu_burden":
                cap = candidate["per_process_sample_cap_percent"]
                sample_burden = sum(
                    min(cap, max(0.0, float(process.get("cpu", 0.0))))
                    for process in sample.get("processes", [])
                    if process.get("category") in CPU_SCORED_CATEGORIES
                )
                window = self.burdens[candidate_id]
                window.append(sample_burden)
                while len(window) > candidate["window_samples"]:
                    window.popleft()
                mean_burden = sum(window) / candidate["window_samples"]
                cpu_violation = mean_burden >= candidate["mean_single_core_burden_percent"]
                trigger_groups = sorted(key for key, cpu in grouped.items() if cpu > 0)
                diagnostic_value = mean_burden
            else:
                raise ValueError(f"Unknown policy family: {family}")

            results[candidate_id] = {
                "hard_violation": bool(hard),
                "hard_reasons": hard,
                "cpu_violation": cpu_violation,
                "invalid": bool(hard) or cpu_violation,
                "trigger_groups": trigger_groups,
                "trigger_categories": sorted(
                    {categories[key] for key in trigger_groups if key in categories}
                ),
                "diagnostic_value": diagnostic_value,
            }
        return sample, results


def evaluate_samples(samples, preregistration):
    evaluator = CandidateEvaluator(preregistration)
    output = []
    for sample in sorted(samples, key=lambda item: item["index"]):
        classified, candidates = evaluator.evaluate(sample)
        output.append(
            {
                "index": classified["index"],
                "observed_at": classified.get("observed_at"),
                "processes": classified["processes"],
                "candidates": candidates,
            }
        )
    return output


def episodes(values):
    starts = []
    active = False
    for index, value in enumerate(values):
        if value and not active:
            starts.append(index)
        active = value
    return starts


def summarize_evaluation(evaluated):
    if not evaluated:
        return {}
    candidate_ids = list(evaluated[0]["candidates"])
    summary = {}
    for candidate_id in candidate_ids:
        rows = [item["candidates"][candidate_id] for item in evaluated]
        cpu_values = [row["cpu_violation"] for row in rows]
        hard_values = [row["hard_violation"] for row in rows]
        invalid_values = [row["invalid"] for row in rows]
        summary[candidate_id] = {
            "samples": len(rows),
            "cpu_violation_samples": sum(cpu_values),
            "cpu_violation_episodes": len(episodes(cpu_values)),
            "hard_violation_samples": sum(hard_values),
            "hard_violation_episodes": len(episodes(hard_values)),
            "invalid_samples": sum(invalid_values),
            "invalid_episodes": len(episodes(invalid_values)),
        }
    return summary


def load_preregistration(path):
    data = json.loads(Path(path).read_text())
    if data.get("status") != "pre_registered_before_calibration":
        raise ValueError("Calibration requires the frozen preregistration")
    return data
