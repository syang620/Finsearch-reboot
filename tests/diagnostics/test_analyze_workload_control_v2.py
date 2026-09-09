from copy import deepcopy
from pathlib import Path

from scripts.diagnostics.analyze_workload_control_v2_calibration import (
    awake_protection_validation,
    canonicalize,
    decide,
    global_hard_control_validation,
    read_raw,
    scenario_capture_validation,
    short_burst_validation,
    s3_preflight_validation,
    terminal_only_viability,
)


CANDIDATES = [
    {"id": "A", "eligible_for_selection": False},
    {"id": "B3", "eligible_for_selection": True},
    {"id": "B5", "eligible_for_selection": True},
]


def preregistration():
    return {
        "candidate_policies": CANDIDATES,
        "acceptance_criteria": {
            "clean_environment": {
                "scenarios": ["clean1", "clean2", "clean3"],
                "maximum_cpu_invalidation_episodes_per_complete_scenario": 0,
            },
            "sustained_interference": {
                "required_detection_repetitions": 3,
                "maximum_detection_delay_seconds": 15,
            },
            "short_bursts": {"scenario": "short", "maximum_cpu_invalidation_episodes": 0},
            "browser": {"maximum_detection_delay_samples": 2},
            "evidence_visibility": {"classified_or_unknown_process_retention_rate": 1.0},
        },
        "selection_rule": {"eligible_candidate_preference_order": ["B3", "B5"]},
    }


def scenario_results():
    result = {}
    for scenario in ("clean1", "clean2", "clean3", "short"):
        result[scenario] = {
            "classified_or_unknown_retention_rate": 1.0,
            "candidates": {
                item["id"]: {"cpu_violation_episodes": 0, "hard_violation_episodes": 0}
                for item in CANDIDATES
            },
        }
    return result


def sustained():
    return {
        item["id"]: [
            {"detected": True, "detection_latency_seconds": delay}
            for delay in (3, 4, 5)
        ]
        for item in CANDIDATES
    }


def browser():
    return {
        item["id"]: {"detected": True, "detection_delay_samples": 0}
        for item in CANDIDATES
    }


def test_selection_uses_preregistered_preference_order():
    result = decide(preregistration(), scenario_results(), sustained(), browser(), True)
    assert result["decision"] == "WORKLOAD_CONTROL_V2_VALIDATED"
    assert result["selected_policy"] == "B3"
    assert not result["candidate_acceptance"]["A"]["passes_all"]


def test_any_failed_clean_scenario_rejects_candidate():
    scenarios = scenario_results()
    scenarios["clean2"]["candidates"]["B3"]["cpu_violation_episodes"] = 1
    result = decide(preregistration(), scenarios, sustained(), browser(), True)
    assert not result["candidate_acceptance"]["B3"]["passes_all"]
    assert result["selected_policy"] == "B5"


def test_no_post_result_compromise_when_all_candidates_fail():
    detections = sustained()
    for candidate in ("B3", "B5"):
        detections[candidate][0] = {"detected": False, "detection_latency_seconds": None}
    result = decide(preregistration(), scenario_results(), detections, browser(), True)
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert result["selected_policy"] is None


def test_detection_over_fifteen_seconds_fails_fixed_bound():
    detections = sustained()
    detections["B3"][1]["detection_latency_seconds"] = 15.1
    result = decide(preregistration(), scenario_results(), detections, browser(), True)
    assert not result["candidate_acceptance"]["B3"]["sustained_interference"]


def test_browser_delay_beyond_two_samples_fails():
    browser_rows = deepcopy(browser())
    browser_rows["B3"]["detection_delay_samples"] = 3
    result = decide(preregistration(), scenario_results(), sustained(), browser_rows, True)
    assert not result["candidate_acceptance"]["B3"]["browser_hard_rule"]


def test_preexisting_violation_cannot_count_as_new_detection():
    detections = sustained()
    detections["B3"][0] = {
        "detected": False,
        "detection_latency_seconds": None,
        "preexisting_violation_at_workload_start": True,
    }
    result = decide(preregistration(), scenario_results(), detections, browser(), True)
    assert not result["candidate_acceptance"]["B3"]["sustained_interference"]


def test_terminal_viability_retains_but_distinguishes_inert_crash_handlers():
    raw = {
        "samples": [
            {
                "processes": [
                    {
                        "category": "run_supervision_ui_tooling",
                        "executable": "/Applications/ChatGPT.app/Helpers/browser_crashpad_handler",
                        "cpu": 0.0,
                        "pid": 10,
                        "start_time": "now",
                        "ancestors": [
                            {"pid": 10, "executable": "/Applications/ChatGPT.app/Helpers/browser_crashpad_handler"},
                            {"pid": 1, "executable": "/sbin/launchd"},
                        ],
                    }
                ]
            }
        ]
    }
    result = terminal_only_viability(raw)
    assert result["viable"]
    assert result["classification"] == "viable_with_inert_crash_handler_limitation"
    assert result["active_supervision_process_records"] == 0
    assert result["retained_inert_crash_handler_records"] == 1
    assert result["unique_inert_crash_handlers"] == 1
    assert result["samples_with_inert_crash_handlers"] == 1
    assert "1 unique crash handlers across 1/1 samples" in result["evidence"]


def test_terminal_viability_rejects_active_supervision_process():
    raw = {
        "samples": [
            {
                "processes": [
                    {
                        "category": "run_supervision_ui_tooling",
                        "executable": "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT",
                        "cpu": 0.0,
                    }
                ]
            }
        ]
    }
    result = terminal_only_viability(raw)
    assert not result["viable"]
    assert result["classification"] == "not_demonstrated"


def test_terminal_failure_blocks_otherwise_passing_candidate():
    result = decide(
        preregistration(), scenario_results(), sustained(), browser(), False
    )
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert result["selected_policy"] is None
    assert not result["candidate_acceptance"]["B3"]["terminal_only_scenario"]


def test_awake_failure_blocks_otherwise_passing_candidate():
    result = decide(
        preregistration(), scenario_results(), sustained(), browser(), True, False, True
    )
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert not result["candidate_acceptance"]["B3"]["awake_protection"]


def test_service_identity_failure_blocks_otherwise_passing_candidate():
    result = decide(
        preregistration(), scenario_results(), sustained(), browser(), True, True, False
    )
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert not result["candidate_acceptance"]["B3"]["required_service_identity"]


def test_global_hard_control_failure_blocks_otherwise_passing_candidate():
    result = decide(
        preregistration(), scenario_results(), sustained(), browser(), True,
        True, True, False, True
    )
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert not result["candidate_acceptance"]["B3"]["global_hard_controls"]


def test_incomplete_capture_blocks_otherwise_passing_candidate():
    result = decide(
        preregistration(), scenario_results(), sustained(), browser(), True,
        True, True, True, False
    )
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert not result["candidate_acceptance"]["B3"]["complete_scenario_captures"]


def test_unexercised_short_workloads_block_otherwise_passing_candidate():
    result = decide(
        preregistration(), scenario_results(), sustained(), browser(), True,
        True, True, True, True, False
    )
    assert result["decision"] == "CONTROL_V2_NOT_VALIDATED"
    assert not result["candidate_acceptance"]["B3"]["short_workloads_exercised"]


def test_awake_validation_requires_active_evidence_in_every_sample():
    raw = {
        "scenario": {
            "header": {"awake_pid": 42},
            "samples": [
                {"awake_protection": {"pid": 42, "active": True}},
                {"awake_protection": {"pid": 42, "active": False}},
            ],
            "footer": {
                "awake_protection_active_through_final_sample": False,
                "awake_protection_returncode_before_cleanup": 1,
            },
        }
    }
    result = awake_protection_validation(raw)
    assert not result["viable"]
    assert result["scenarios"]["scenario"]["active_matching_samples"] == 1


def test_s3_preflight_requires_exact_frozen_identity():
    frozen = {
        "runtime_config": {
            "analyst_model": "ollama/model",
            "model_digest": "model-sha",
            "embedding_model": "embedding",
            "embedding_digest": "embedding-sha",
        },
        "service_preflight": {
            "ollama_version": {"version": "1"},
            "qdrant_service": {"title": "qdrant", "version": "1", "commit": "abc"},
            "sec_health": {"status_code": 200},
        },
        "index_before": {"points": 1},
        "historical_index_before": {"points": 2},
    }
    results = {
        "repository_and_freeze_checks": {"tracked_status": ""},
        "local_service_identity": {
            "ollama_version": {"version": "1"},
            "model_digests": {"model": "model-sha", "embedding": "embedding-sha"},
            "qdrant": {"title": "qdrant", "version": "1", "commit": "abc"},
        },
        "sec_service_health": {"status_code": 200},
        "index_identity": {"current": {"points": 1}, "historical": {"points": 2}},
        "planner_import_and_construction": {},
        "unchanged_30_second_settle": None,
    }
    preflight = {
        "implementation_sha": "capture-sha",
        "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": "2026-01-01T00:00:02+00:00",
        "steps": [
            {"name": name, "status": "ok", "result": result}
            for name, result in results.items()
        ],
        "errors": [],
    }
    raw = {
        "header": {"implementation_sha": "capture-sha"},
        "events": [
            {"event": "workload_start", "key": "preflight", "pid": 42, "at": "2026-01-01T00:00:00+00:00"},
            {"event": "workload_exit", "key": "preflight", "pid": 42, "returncode": 0, "at": "2026-01-01T00:00:03+00:00"},
        ],
    }
    assert s3_preflight_validation(preflight, frozen, raw)["viable"]
    preflight["steps"][1]["result"]["model_digests"]["model"] = "wrong"
    result = s3_preflight_validation(preflight, frozen, raw)
    assert not result["viable"]
    assert not result["checks"]["model_digests_match"]


def test_s3_preflight_rejects_changed_ollama_service():
    frozen = {
        "runtime_config": {
            "analyst_model": "ollama/model",
            "model_digest": "model-sha",
            "embedding_model": "embedding",
            "embedding_digest": "embedding-sha",
        },
        "service_preflight": {
            "ollama_version": {"version": "1"},
            "qdrant_service": {"title": "qdrant", "version": "1", "commit": "abc"},
            "sec_health": {"status_code": 200},
        },
        "index_before": {"points": 1},
        "historical_index_before": {"points": 2},
    }
    results = {
        "repository_and_freeze_checks": {"tracked_status": ""},
        "local_service_identity": {
            "ollama_version": {"version": "2"},
            "model_digests": {"model": "model-sha", "embedding": "embedding-sha"},
            "qdrant": {"title": "qdrant", "version": "1", "commit": "abc"},
        },
        "sec_service_health": {"status_code": 200},
        "index_identity": {"current": {"points": 1}, "historical": {"points": 2}},
        "planner_import_and_construction": {},
        "unchanged_30_second_settle": None,
    }
    preflight = {
        "implementation_sha": "capture-sha",
        "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": "2026-01-01T00:00:02+00:00",
        "steps": [
            {"name": name, "status": "ok", "result": result}
            for name, result in results.items()
        ],
        "errors": [],
    }
    raw = {
        "header": {"implementation_sha": "capture-sha"},
        "events": [
            {"event": "workload_start", "key": "preflight", "pid": 42, "at": "2026-01-01T00:00:00+00:00"},
            {"event": "workload_exit", "key": "preflight", "pid": 42, "returncode": 0, "at": "2026-01-01T00:00:03+00:00"},
        ],
    }
    result = s3_preflight_validation(preflight, frozen, raw)
    assert not result["viable"]
    assert not result["checks"]["ollama_identity_matches"]


def test_failed_s3_event_binding_rejects_preflight():
    frozen = {
        "runtime_config": {"analyst_model": "ollama/model", "model_digest": "m", "embedding_model": "embed", "embedding_digest": "e"},
        "service_preflight": {"ollama_version": {"version": "1"}, "qdrant_service": {"title": "q", "version": "1", "commit": "c"}, "sec_health": {"status_code": 200}},
        "index_before": {}, "historical_index_before": {},
    }
    results = {
        "repository_and_freeze_checks": {"tracked_status": ""},
        "local_service_identity": {"ollama_version": {"version": "1"}, "model_digests": {"model": "m", "embed": "e"}, "qdrant": {"title": "q", "version": "1", "commit": "c"}},
        "sec_service_health": {"status_code": 200}, "index_identity": {"current": {}, "historical": {}},
        "planner_import_and_construction": {}, "unchanged_30_second_settle": None,
    }
    preflight = {"implementation_sha": "capture", "started_at": "2026-01-01T00:00:01+00:00", "finished_at": "2026-01-01T00:00:02+00:00", "steps": [{"name": k, "status": "ok", "result": v} for k,v in results.items()], "errors": []}
    raw = {"header": {"implementation_sha": "capture"}, "events": [
        {"event": "workload_start", "key": "preflight", "pid": 1, "at": "2026-01-01T00:00:00+00:00"},
        {"event": "workload_exit", "key": "preflight", "pid": 1, "returncode": 1, "at": "2026-01-01T00:00:03+00:00"},
    ]}
    result = s3_preflight_validation(preflight, frozen, raw)
    assert not result["viable"]
    assert not result["checks"]["bound_to_successful_s3_event"]


def test_global_hard_controls_cover_every_scenario_and_only_owned_s6_browser():
    base = {"samples": [{"ac_power": True, "low_power_mode": 0, "elapsed_seconds": 1, "processes": []}], "events": []}
    raw = {"S4_SUSTAINED_CPU_INTERFERENCE": deepcopy(base), "S6_BROWSER_WORKLOAD": deepcopy(base)}
    raw["S6_BROWSER_WORKLOAD"]["samples"][0]["processes"] = [{"category": "user_browser_workload", "pid": 8, "controlled_external": True}]
    raw["S6_BROWSER_WORKLOAD"]["events"] = [
        {"event": "workload_start", "key": "browser", "pid": 8, "elapsed_seconds": 0},
        {"event": "workload_stop", "key": "browser", "pid": 8, "returncode": 0, "elapsed_seconds": 2},
    ]
    assert global_hard_control_validation(raw)["viable"]
    raw["S4_SUSTAINED_CPU_INTERFERENCE"]["samples"][0]["ac_power"] = False
    assert not global_hard_control_validation(raw)["viable"]


def test_incomplete_registered_capture_is_not_viable(tmp_path):
    prereg = {"scope": {"sample_interval_seconds": 1}, "scenarios": {"S": {"duration_seconds": 2}}}
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}")
    raw = {"S": {"header": {"scenario": "S", "preregistration_sha256": __import__('hashlib').sha256(b'{}').hexdigest(), "duration_seconds": 2, "sample_interval_seconds": 1}, "samples": [{"scenario": "S", "elapsed_seconds": 0}]}}
    result = scenario_capture_validation(raw, prereg, prereg_path)
    assert not result["viable"]
    assert not result["scenarios"]["S"]["checks"]["sample_count_matches"]


def test_short_burst_validation_requires_six_successful_observed_workloads():
    events = []
    samples = []
    for index in range(1, 7):
        key = f"short_{index}"
        start = 15 + (index - 1) * 15
        pid = 100 + index
        events.extend([
            {"event": "workload_start", "key": key, "pid": pid, "elapsed_seconds": start},
            {"event": "workload_exit", "key": key, "pid": pid, "returncode": 0, "elapsed_seconds": start + 2},
        ])
        samples.append({"processes": [{"pid": pid, "controlled_external": True, "category": "unrelated_external_workload", "cpu": 95.0}]})
    prereg = {"scenarios": {"S5_SHORT_CPU_BURSTS": {"burst_count": 6, "warmup_seconds": 15, "burst_start_interval_seconds": 15, "burst_seconds": 1.5}}}
    raw = {"events": events, "samples": samples}
    assert short_burst_validation(raw, prereg)["viable"]
    raw["events"][-1]["returncode"] = 1
    assert not short_burst_validation(raw, prereg)["viable"]


def test_comparison_float_canonicalization_is_recursive():
    value = {"a": [154.12666666666667, -0.0], "b": 1}
    assert canonicalize(value) == {"a": [154.126666667, 0.0], "b": 1}


def test_preserved_reopened_terminal_attempt_is_not_viable():
    path = Path(
        "artifacts/evals/workload_control/v2/calibration/"
        "36b27b41422ee866ce453d17c3af8543d857a7b2/invalid_protocol_attempts/"
        "S2_attempt_3_reopened_at_sample_891.jsonl.gz"
    )
    result = terminal_only_viability(read_raw(path))
    assert not result["viable"]
    assert result["active_supervision_process_records"] > 0
    assert result["classification"] == "not_demonstrated"
    assert f"Observed {result['active_supervision_process_records']}" in result["evidence"]
