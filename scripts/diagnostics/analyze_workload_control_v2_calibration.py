"""Apply the frozen calibration criteria once to control-only observations."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from datetime import datetime
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


def scenario_capture_validation(raw_by_scenario, preregistration, preregistration_path):
    expected_preregistration_sha = sha256(preregistration_path)
    interval = preregistration["scope"]["sample_interval_seconds"]
    scenarios = {}
    for scenario, raw in raw_by_scenario.items():
        duration = preregistration["scenarios"][scenario]["duration_seconds"]
        expected_samples = round(duration / interval)
        samples = raw["samples"]
        cadence_matches = len(samples) == expected_samples and all(
            abs(float(sample["elapsed_seconds"]) - index * interval)
            <= interval * 0.25
            for index, sample in enumerate(samples)
        )
        checks = {
            "header_scenario_matches": raw["header"].get("scenario") == scenario,
            "sample_scenarios_match": all(
                sample.get("scenario") == scenario for sample in samples
            ),
            "preregistration_hash_matches": raw["header"].get(
                "preregistration_sha256"
            )
            == expected_preregistration_sha,
            "duration_matches": raw["header"].get("duration_seconds") == duration,
            "interval_matches": raw["header"].get("sample_interval_seconds")
            == interval,
            "sample_count_matches": len(samples) == expected_samples,
            "cadence_matches": cadence_matches,
        }
        scenarios[scenario] = {
            "viable": all(checks.values()),
            "expected_samples": expected_samples,
            "observed_samples": len(samples),
            "checks": checks,
        }
    return {
        "viable": bool(scenarios) and all(row["viable"] for row in scenarios.values()),
        "scenarios": scenarios,
    }


def global_hard_control_validation(raw_by_scenario):
    scenarios = {}
    for scenario, raw in raw_by_scenario.items():
        samples = raw["samples"]
        ac_lpm_valid = all(
            sample.get("ac_power") is True and sample.get("low_power_mode") == 0
            for sample in samples
        )
        browser_rows = [
            (sample, process)
            for sample in samples
            for process in sample.get("processes", [])
            if process.get("category") == "user_browser_workload"
        ]
        if scenario == "S6_BROWSER_WORKLOAD":
            starts = [
                event
                for event in raw["events"]
                if event.get("event") == "workload_start" and event.get("key") == "browser"
            ]
            stops = [
                event
                for event in raw["events"]
                if event.get("event") == "workload_stop" and event.get("key") == "browser"
            ]
            browser_valid = (
                len(starts) == 1
                and len(stops) == 1
                and starts[0].get("pid") == stops[0].get("pid")
                and stops[0].get("returncode") == 0
                and bool(browser_rows)
                and all(
                    process.get("pid") == starts[0]["pid"]
                    and process.get("controlled_external") is True
                    and starts[0]["elapsed_seconds"]
                    <= sample["elapsed_seconds"]
                    <= stops[0]["elapsed_seconds"]
                    for sample, process in browser_rows
                )
            )
        else:
            browser_valid = not browser_rows
        scenarios[scenario] = {
            "viable": ac_lpm_valid and browser_valid,
            "ac_and_low_power_mode_valid": ac_lpm_valid,
            "browser_identity_valid": browser_valid,
            "actual_browser_process_records": len(browser_rows),
        }
    return {
        "viable": bool(scenarios) and all(row["viable"] for row in scenarios.values()),
        "scenarios": scenarios,
    }


def short_burst_validation(raw, preregistration):
    scenario = preregistration["scenarios"]["S5_SHORT_CPU_BURSTS"]
    count = scenario["burst_count"]
    expected_keys = {f"short_{index}" for index in range(1, count + 1)}
    start_events = [
        event
        for event in raw["events"]
        if event.get("event") == "workload_start"
        and event.get("key") in expected_keys
    ]
    exit_events = [
        event
        for event in raw["events"]
        if event.get("event") == "workload_exit"
        and event.get("key") in expected_keys
    ]
    starts = {
        event.get("key"): event
        for event in start_events
    }
    exits = {
        event.get("key"): event
        for event in exit_events
    }
    rows = []
    for index in range(1, count + 1):
        key = f"short_{index}"
        start = starts.get(key, {})
        exit_event = exits.get(key, {})
        expected_start = scenario["warmup_seconds"] + (
            index - 1
        ) * scenario["burst_start_interval_seconds"]
        observed = [
            process
            for sample in raw["samples"]
            for process in sample.get("processes", [])
            if process.get("pid") == start.get("pid")
            and process.get("controlled_external") is True
            and process.get("category") == "unrelated_external_workload"
            and float(process.get("cpu", 0)) >= 50
        ]
        viable = (
            bool(start)
            and bool(exit_event)
            and start.get("pid") == exit_event.get("pid")
            and exit_event.get("returncode") == 0
            and abs(start.get("elapsed_seconds", -1000) - expected_start) <= 0.25
            and scenario["burst_seconds"]
            <= exit_event.get("elapsed_seconds", -1000)
            - start.get("elapsed_seconds", 1000)
            <= scenario["burst_seconds"] + 1.0
            and bool(observed)
        )
        rows.append(
            {
                "key": key,
                "viable": viable,
                "pid": start.get("pid"),
                "start_elapsed_seconds": start.get("elapsed_seconds"),
                "exit_elapsed_seconds": exit_event.get("elapsed_seconds"),
                "returncode": exit_event.get("returncode"),
                "samples_at_or_above_50_percent": len(observed),
            }
        )
    return {
        "viable": len(start_events) == count
        and len(exit_events) == count
        and set(starts) == expected_keys
        and set(exits) == expected_keys
        and all(row["viable"] for row in rows),
        "expected_bursts": count,
        "bursts": rows,
    }


def canonicalize(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float in comparison output")
        rounded = round(value, 9)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(item) for key, item in value.items()}
    return value


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
    unique_crash_handlers = {
        (
            process.get("pid"),
            process.get("start_time"),
            process.get("executable"),
        )
        for process in crash_handlers
    }
    samples_with_crash_handlers = sum(
        any(
            Path(str(process.get("executable") or "")).name
            == "browser_crashpad_handler"
            for process in sample.get("processes", [])
        )
        for sample in raw["samples"]
    )
    crash_handlers_launchd_owned = all(
        any(
            str(ancestor.get("executable") or "").lower() == "/sbin/launchd"
            for ancestor in process.get("ancestors", [])
            if ancestor.get("pid") != process.get("pid")
        )
        for process in crash_handlers
    )
    viable = (
        not active_supervision
        and crash_handler_peak_cpu == 0.0
        and crash_handlers_launchd_owned
    )
    return {
        "classification": (
            "viable_with_inert_crash_handler_limitation"
            if viable and crash_handlers
            else "viable_no_supervision_processes"
            if viable
            else "not_demonstrated"
        ),
        "viable": viable,
        "active_supervision_process_records": len(active_supervision),
        "retained_inert_crash_handler_records": len(crash_handlers),
        "unique_inert_crash_handlers": len(unique_crash_handlers),
        "samples_with_inert_crash_handlers": samples_with_crash_handlers,
        "total_samples": len(raw["samples"]),
        "inert_crash_handler_peak_cpu_percent": crash_handler_peak_cpu,
        "all_retained_crash_handlers_launchd_owned": crash_handlers_launchd_owned,
        "evidence": (
            f"Observed {len(active_supervision)} active supervision records and "
            f"{len(crash_handlers)} records from {len(unique_crash_handlers)} unique "
            f"crash handlers across {samples_with_crash_handlers}/{len(raw['samples'])} "
            f"samples; crash-handler peak CPU was {crash_handler_peak_cpu:.1f} percent "
            f"and launchd_owned_all={str(crash_handlers_launchd_owned).lower()}."
        ),
        "limitations": (
            "Terminal-only preflight was not separately exercised, and semantic case "
            "autonomy was intentionally not tested."
        ),
    }


def awake_protection_validation(raw_by_scenario):
    scenarios = {}
    for scenario, raw in raw_by_scenario.items():
        header_pid = raw["header"].get("awake_pid")
        samples = raw["samples"]
        matching_samples = sum(
            sample.get("awake_protection")
            == {"pid": header_pid, "active": True}
            for sample in samples
        )
        footer_active = raw["footer"].get(
            "awake_protection_active_through_final_sample"
        ) is True
        returncode_before_cleanup = raw["footer"].get(
            "awake_protection_returncode_before_cleanup", "missing"
        )
        viable = (
            isinstance(header_pid, int)
            and header_pid > 0
            and bool(samples)
            and matching_samples == len(samples)
            and footer_active
            and returncode_before_cleanup is None
        )
        scenarios[scenario] = {
            "viable": viable,
            "awake_pid": header_pid,
            "active_matching_samples": matching_samples,
            "total_samples": len(samples),
            "footer_active_through_final_sample": footer_active,
            "returncode_before_cleanup": returncode_before_cleanup,
        }
    return {
        "viable": bool(scenarios) and all(row["viable"] for row in scenarios.values()),
        "scenarios": scenarios,
    }


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def s3_preflight_validation(preflight, frozen_provenance, s3_raw):
    steps = {step.get("name"): step for step in preflight.get("steps", [])}
    required_steps = {
        "repository_and_freeze_checks",
        "local_service_identity",
        "sec_service_health",
        "index_identity",
        "planner_import_and_construction",
        "unchanged_30_second_settle",
    }
    expected_runtime = frozen_provenance["runtime_config"]
    expected_service = frozen_provenance["service_preflight"]
    service = steps.get("local_service_identity", {}).get("result", {})
    index = steps.get("index_identity", {}).get("result", {})
    expected_models = {
        expected_runtime["analyst_model"].removeprefix("ollama/"): expected_runtime[
            "model_digest"
        ],
        expected_runtime["embedding_model"]: expected_runtime["embedding_digest"],
    }
    observed_qdrant = service.get("qdrant")
    expected_qdrant = expected_service["qdrant_service"]
    qdrant_matches = isinstance(observed_qdrant, dict) and all(
        observed_qdrant.get(key) == expected_qdrant.get(key)
        for key in ("title", "version", "commit")
    )
    starts = [
        event
        for event in s3_raw.get("events", [])
        if event.get("event") == "workload_start" and event.get("key") == "preflight"
    ]
    exits = [
        event
        for event in s3_raw.get("events", [])
        if event.get("event") == "workload_exit" and event.get("key") == "preflight"
    ]
    event_binding = (
        len(starts) == 1
        and len(exits) == 1
        and starts[0].get("pid") == exits[0].get("pid")
        and exits[0].get("returncode") == 0
        and preflight.get("implementation_sha")
        == s3_raw.get("header", {}).get("implementation_sha")
        and parse_time(starts[0]["at"])
        <= parse_time(preflight["started_at"])
        <= parse_time(preflight["finished_at"])
        <= parse_time(exits[0]["at"])
    )
    checks = {
        "complete_required_steps": len(preflight.get("steps", [])) == len(required_steps)
        and set(steps) == required_steps
        and all(steps[name].get("status") == "ok" for name in required_steps),
        "no_reported_errors": preflight.get("errors") == [],
        "repository_clean": steps.get("repository_and_freeze_checks", {})
        .get("result", {})
        .get("tracked_status")
        == "",
        "model_digests_match": service.get("model_digests") == expected_models,
        "ollama_identity_matches": service.get("ollama_version")
        == expected_service["ollama_version"],
        "qdrant_identity_matches": qdrant_matches,
        "sec_service_healthy": steps.get("sec_service_health", {})
        .get("result", {})
        .get("status_code")
        == expected_service["sec_health"]["status_code"],
        "current_index_matches": index.get("current")
        == frozen_provenance["index_before"],
        "historical_index_matches": index.get("historical")
        == frozen_provenance["historical_index_before"],
        "bound_to_successful_s3_event": event_binding,
    }
    return {"viable": all(checks.values()), "checks": checks}


def decide(
    preregistration,
    scenario_results,
    sustained,
    browser,
    terminal_viable,
    awake_viable=True,
    s3_preflight_viable=True,
    global_hard_controls_viable=True,
    scenario_captures_viable=True,
    short_workloads_viable=True,
):
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
            and terminal_viable
            and awake_viable
            and s3_preflight_viable
            and global_hard_controls_viable
            and scenario_captures_viable
            and short_workloads_viable
            and sustained_pass
            and short_pass
            and browser_pass
            and retention_pass
        )
        candidate_results[candidate_id] = {
            "eligible": candidate["eligible_for_selection"],
            "clean_scenarios": clean,
            "terminal_only_scenario": terminal_viable,
            "awake_protection": awake_viable,
            "required_service_identity": s3_preflight_viable,
            "global_hard_controls": global_hard_controls_viable,
            "complete_scenario_captures": scenario_captures_viable,
            "short_workloads_exercised": short_workloads_viable,
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
    parser.add_argument("--s3-preflight", type=Path, required=True)
    parser.add_argument("--frozen-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    preregistration = load_preregistration(args.preregistration)
    raw = {scenario: read_raw(path) for scenario, path in args.scenario}
    if set(raw) != set(preregistration["calibration_order"]):
        raise ValueError("Exactly the preregistered scenarios are required")
    scenario_results = {scenario: scenario_metrics(value, preregistration) for scenario, value in raw.items()}
    capture_validation = scenario_capture_validation(raw, preregistration, args.preregistration)
    global_hard_controls = global_hard_control_validation(raw)
    short_workloads = short_burst_validation(
        raw["S5_SHORT_CPU_BURSTS"], preregistration
    )
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
    terminal = terminal_only_viability(raw["S2_TERMINAL_ONLY_IDLE"])
    awake = awake_protection_validation(raw)
    s3_preflight = s3_preflight_validation(
        json.loads(args.s3_preflight.read_text()),
        json.loads(args.frozen_provenance.read_text()),
        raw["S3_REQUIRED_SERVICE_ACTIVITY"],
    )
    decision = decide(
        preregistration,
        scenario_results,
        sustained,
        browser,
        terminal["viable"],
        awake["viable"],
        s3_preflight["viable"],
        global_hard_controls["viable"],
        capture_validation["viable"],
        short_workloads["viable"],
    )
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
        "terminal_only_viability": terminal,
        "awake_protection_validation": awake,
        "scenario_capture_validation": capture_validation,
        "global_hard_control_validation": global_hard_controls,
        "short_burst_validation": short_workloads,
        "s3_preflight_validation": {
            **s3_preflight,
            "path": str(args.s3_preflight),
            "sha256": sha256(args.s3_preflight),
            "frozen_provenance_path": str(args.frozen_provenance),
            "frozen_provenance_sha256": sha256(args.frozen_provenance),
        },
        "historical_replay": historical_replay(
            preregistration, args.pr32_raw, args.prior_disposition
        ),
    }
    output["canonical_float_decimals"] = 9
    output = canonicalize(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": output["decision"], "selected_policy": output["selected_policy"]}))


def subprocess_sha():
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
