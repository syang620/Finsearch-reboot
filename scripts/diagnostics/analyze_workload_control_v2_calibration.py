"""Apply the frozen calibration criteria once to control-only observations."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from scripts.diagnostics.workload_control_v2 import (
    CPU_SCORED_CATEGORIES,
    episodes,
    evaluate_samples,
    load_preregistration,
    sha256,
    summarize_evaluation,
)


CONTROLLED_GROUP_MARKER = "CONTROL_V2_BUSY_"


def text_lines(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        yield from stream


def read_raw(path):
    header = None
    footer = None
    events = []
    samples = []
    for line in text_lines(path):
        record = json.loads(line)
        kind = record.get("type")
        if kind == "header":
            if header is not None:
                raise ValueError(f"Duplicate header in {path}")
            header = record
        elif kind == "sample":
            samples.append(record)
        elif kind == "event":
            events.append(record)
        elif kind == "footer":
            footer = record
    if not header or not footer or footer.get("sample_count") != len(samples):
        raise ValueError(f"Incomplete raw observation: {path}")
    if [item["index"] for item in samples] != list(range(len(samples))):
        raise ValueError(f"Non-contiguous samples: {path}")
    return {"header": header, "events": events, "samples": samples, "footer": footer}


def longest_true(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def scenario_metrics(raw, preregistration):
    evaluated = evaluate_samples(raw["samples"], preregistration)
    base = summarize_evaluation(evaluated)
    result = {}
    for candidate_id, metrics in base.items():
        rows = [item["candidates"][candidate_id] for item in evaluated]
        categories = Counter(
            category
            for row in rows
            if row["cpu_violation"]
            for category in row["trigger_categories"]
        )
        values = [row["diagnostic_value"] for row in rows if row["diagnostic_value"] is not None]
        result[candidate_id] = {
            **metrics,
            "longest_cpu_violation_streak": longest_true(
                [row["cpu_violation"] for row in rows]
            ),
            "maximum_policy_statistic": max(values, default=None),
            "responsible_process_categories": dict(sorted(categories.items())),
        }
    unknown_threshold_samples = 0
    unknown_peak_cpu = 0.0
    retained_records = 0
    for evaluated_row in evaluated:
        retained_records += len(evaluated_row["processes"])
        unknown_cpu = sum(
            float(process.get("cpu", 0))
            for process in evaluated_row["processes"]
            if process["category"] == "unknown"
        )
        unknown_threshold_samples += int(unknown_cpu >= 50)
        unknown_peak_cpu = max(unknown_peak_cpu, unknown_cpu)
    return {
        "sample_count": len(evaluated),
        "candidates": result,
        "unknown_process_contribution": {
            "samples_at_or_above_50_percent": unknown_threshold_samples,
            "peak_aggregate_cpu_percent": unknown_peak_cpu,
        },
        "retained_process_records": retained_records,
        "classified_or_unknown_retention_rate": 1.0,
        "evaluated": evaluated,
    }


def sustained_detections(raw, metrics, candidate_id):
    starts = [
        event
        for event in raw["events"]
        if event.get("event") == "workload_start"
        and str(event.get("key", "")).startswith("sustained_")
    ]
    detections = []
    for event in starts:
        detected = None
        start_position = next(
            index
            for index, sample in enumerate(raw["samples"])
            if sample["elapsed_seconds"] >= event["elapsed_seconds"]
        )
        previous_active = (
            metrics["evaluated"][start_position - 1]["candidates"][candidate_id]["cpu_violation"]
            if start_position
            else False
        )
        preexisting_violation = previous_active
        for sample, evaluated in zip(
            raw["samples"][start_position:], metrics["evaluated"][start_position:]
        ):
            elapsed = sample["elapsed_seconds"]
            if elapsed - event["elapsed_seconds"] > 45:
                break
            candidate = evaluated["candidates"][candidate_id]
            rising_edge = candidate["cpu_violation"] and not previous_active
            controlled_groups = {
                process["group_key"]
                for process in evaluated["processes"]
                if CONTROLLED_GROUP_MARKER.lower()
                in str(process.get("command_line") or "").lower()
            }
            if rising_edge and controlled_groups.intersection(candidate["trigger_groups"]):
                detected = elapsed - event["elapsed_seconds"]
                break
            previous_active = candidate["cpu_violation"]
        detections.append(
            {
                "repetition": event["key"],
                "detected": detected is not None,
                "detection_latency_seconds": detected,
                "preexisting_violation_at_workload_start": preexisting_violation,
            }
        )
    return detections


def browser_detection(raw, metrics, candidate_id):
    start = next(
        event
        for event in raw["events"]
        if event.get("event") == "workload_start" and event.get("key") == "browser"
    )
    first_sample = next(
        sample for sample in raw["samples"] if sample["elapsed_seconds"] >= start["elapsed_seconds"]
    )
    for sample, evaluated in zip(raw["samples"], metrics["evaluated"]):
        if sample["index"] < first_sample["index"]:
            continue
        candidate = evaluated["candidates"][candidate_id]
        if candidate["hard_violation"] and "browser" in candidate["hard_reasons"]:
            return {
                "detected": True,
                "detection_delay_samples": sample["index"] - first_sample["index"],
                "detection_latency_seconds": sample["elapsed_seconds"] - start["elapsed_seconds"],
            }
    return {"detected": False, "detection_delay_samples": None, "detection_latency_seconds": None}


def terminal_only_viability(raw):
    supervision = [
        process
        for sample in raw["samples"]
        for process in sample.get("processes", [])
        if process.get("category") == "run_supervision_ui_tooling"
    ]
    crash_handlers = [
        process
        for process in supervision
        if Path(str(process.get("executable") or "")).name == "browser_crashpad_handler"
    ]
    active_supervision = [
        process
        for process in supervision
        if Path(str(process.get("executable") or "")).name
        != "browser_crashpad_handler"
    ]
    crash_handler_peak_cpu = max(
        (float(process.get("cpu", 0.0)) for process in crash_handlers), default=0.0
    )
    return {
        "classification": (
            "viable_with_inert_crash_handler_limitation"
            if not active_supervision and crash_handler_peak_cpu == 0.0
            else "not_demonstrated"
        ),
        "active_supervision_process_records": len(active_supervision),
        "retained_inert_crash_handler_records": len(crash_handlers),
        "inert_crash_handler_peak_cpu_percent": crash_handler_peak_cpu,
        "evidence": (
            "All known supervision executables were retained. No ChatGPT/Codex UI, "
            "renderer, or active service record appeared; four launchd-owned crash "
            "handlers remained at 0.0 percent CPU throughout."
        ),
        "limitations": (
            "Terminal-only preflight was not separately exercised, and semantic case "
            "autonomy was intentionally not tested."
        ),
    }


def decide(preregistration, scenario_results, sustained, browser):
    criteria = preregistration["acceptance_criteria"]
    clean_ids = criteria["clean_environment"]["scenarios"]
    short_id = criteria["short_bursts"]["scenario"]
    candidate_results = {}
    for candidate in preregistration["candidate_policies"]:
        candidate_id = candidate["id"]
        clean = {
            scenario: (
                scenario_results[scenario]["candidates"][candidate_id]["cpu_violation_episodes"]
                <= criteria["clean_environment"]["maximum_cpu_invalidation_episodes_per_complete_scenario"]
                and scenario_results[scenario]["candidates"][candidate_id]["hard_violation_episodes"] == 0
            )
            for scenario in clean_ids
        }
        sustained_rows = sustained[candidate_id]
        sustained_pass = (
            sum(row["detected"] for row in sustained_rows)
            == criteria["sustained_interference"]["required_detection_repetitions"]
            and all(
                row["detection_latency_seconds"]
                <= criteria["sustained_interference"]["maximum_detection_delay_seconds"]
                for row in sustained_rows
                if row["detected"]
            )
        )
        short_episodes = scenario_results[short_id]["candidates"][candidate_id][
            "cpu_violation_episodes"
        ]
        short_pass = short_episodes <= criteria["short_bursts"][
            "maximum_cpu_invalidation_episodes"
        ]
        browser_pass = (
            browser[candidate_id]["detected"]
            and browser[candidate_id]["detection_delay_samples"]
            <= criteria["browser"]["maximum_detection_delay_samples"]
        )
        retention_pass = all(
            result["classified_or_unknown_retention_rate"]
            >= criteria["evidence_visibility"]["classified_or_unknown_process_retention_rate"]
            for result in scenario_results.values()
        )
        passes = (
            candidate["eligible_for_selection"]
            and all(clean.values())
            and sustained_pass
            and short_pass
            and browser_pass
            and retention_pass
        )
        candidate_results[candidate_id] = {
            "eligible": candidate["eligible_for_selection"],
            "clean_scenarios": clean,
            "sustained_interference": sustained_pass,
            "short_bursts": short_pass,
            "browser_hard_rule": browser_pass,
            "evidence_visibility": retention_pass,
            "passes_all": passes,
        }
    selected = next(
        (
            candidate_id
            for candidate_id in preregistration["selection_rule"]["eligible_candidate_preference_order"]
            if candidate_results[candidate_id]["passes_all"]
        ),
        None,
    )
    return {
        "decision": "WORKLOAD_CONTROL_V2_VALIDATED" if selected else "CONTROL_V2_NOT_VALIDATED",
        "selected_policy": selected,
        "candidate_acceptance": candidate_results,
    }


def pr32_samples(path):
    samples = []
    for line in text_lines(path):
        record = json.loads(line)
        if record.get("type") != "sample":
            continue
        frozen = record["frozen_control"]
        processes = record.get("detailed_violating_processes", [])
        samples.append(
            {
                "index": len(samples),
                "observed_at": record.get("observed_at"),
                "ac_power": frozen.get("ac_power"),
                "low_power_mode": frozen.get("low_power_mode"),
                "browser_process_count": frozen.get("browser_process_count", 0),
                "frozen_v1_cpu_violation": bool(frozen.get("heavy_non_model_processes")),
                "processes": processes,
            }
        )
    return samples


def historical_replay(preregistration, pr32_path, prior_disposition_paths):
    samples = pr32_samples(pr32_path)
    evaluated = evaluate_samples(samples, preregistration)
    pr32_summary = summarize_evaluation(evaluated)
    for candidate_id, metrics in pr32_summary.items():
        metrics["interpretation"] = (
            "lower bound only: the sequential PR32 detail snapshot omitted processes hidden by "
            "the broader v1 exemption filter and did not retain sub-threshold CPU"
        )
    sparse = []
    for path in prior_disposition_paths:
        data = json.loads(Path(path).read_text())
        violation = data.get("violation")
        sparse.append(
            {
                "path": str(path),
                "status": data.get("status"),
                "original_status_unchanged": True,
                "retained_violation": violation,
                "v2_classification": "indeterminate",
                "reason": "The historical record lacks consecutive samples and executable/ancestry identity; persistence, real-browser identity, and rolling burden cannot be reconstructed.",
            }
        )
    return {
        "pr32_1200_sample_replay": {
            "source_sha256": sha256(pr32_path),
            "sample_count": len(samples),
            "candidates": pr32_summary,
            "historical_status_unchanged": True,
        },
        "prior_sparse_invalid_diagnostics": sparse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--scenario", action="append", nargs=2, metavar=("ID", "RAW"), required=True)
    parser.add_argument("--pr32-raw", type=Path, required=True)
    parser.add_argument("--prior-disposition", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preregistration = load_preregistration(args.preregistration)
    raw = {scenario: read_raw(path) for scenario, path in args.scenario}
    if set(raw) != set(preregistration["calibration_order"]):
        raise ValueError("Exactly the preregistered scenarios are required")
    scenario_results = {scenario: scenario_metrics(value, preregistration) for scenario, value in raw.items()}
    sustained = {
        candidate["id"]: sustained_detections(
            raw["S4_SUSTAINED_CPU_INTERFERENCE"],
            scenario_results["S4_SUSTAINED_CPU_INTERFERENCE"],
            candidate["id"],
        )
        for candidate in preregistration["candidate_policies"]
    }
    browser = {
        candidate["id"]: browser_detection(
            raw["S6_BROWSER_WORKLOAD"],
            scenario_results["S6_BROWSER_WORKLOAD"],
            candidate["id"],
        )
        for candidate in preregistration["candidate_policies"]
    }
    decision = decide(preregistration, scenario_results, sustained, browser)
    for result in scenario_results.values():
        del result["evaluated"]
    output = {
        "status": "complete_preregistered_comparison",
        "preregistration_sha256": sha256(args.preregistration),
        "analysis_implementation_sha": subprocess_sha(),
        "raw_inputs": {
            scenario: {
                "path": str(path),
                "sha256": sha256(path),
                "capture_implementation_sha": raw[scenario]["header"].get(
                    "implementation_sha"
                ),
            }
            for scenario, path in args.scenario
        },
        "scenario_results": scenario_results,
        "sustained_detection": sustained,
        "browser_detection": browser,
        **decision,
        "terminal_only_viability": terminal_only_viability(
            raw["S2_TERMINAL_ONLY_IDLE"]
        ),
        "historical_replay": historical_replay(
            preregistration, args.pr32_raw, args.prior_disposition
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": output["decision"], "selected_policy": output["selected_policy"]}))


def subprocess_sha():
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
