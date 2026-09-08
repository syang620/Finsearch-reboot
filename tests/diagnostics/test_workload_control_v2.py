import json
from pathlib import Path

from scripts.diagnostics.workload_control_v2 import (
    classify_process,
    evaluate_samples,
    load_preregistration,
    summarize_evaluation,
)


PREREGISTRATION = Path("docs/evals/workload_control_v2_preregistration.json")


def sample(index, cpu=0, executable="/usr/bin/other", command=None, **overrides):
    process = {
        "pid": 100,
        "ppid": 1,
        "cpu": cpu,
        "executable": executable,
        "command_line": command,
        "ancestors": [],
    }
    value = {
        "index": index,
        "observed_at": f"t{index}",
        "ac_power": True,
        "low_power_mode": 0,
        "browser_process_count": 0,
        "frozen_v1_cpu_violation": cpu >= 50,
        "processes": [process] if cpu else [],
    }
    value.update(overrides)
    return value


def test_preregistration_is_deterministic_and_has_fixed_candidates():
    first = load_preregistration(PREREGISTRATION)
    second = json.loads(PREREGISTRATION.read_text())
    assert first == second
    assert [item["id"] for item in first["candidate_policies"]] == [
        "A_INSTANTANEOUS_CURRENT",
        "B_CONSECUTIVE_3",
        "B_CONSECUTIVE_5",
        "B_CONSECUTIVE_10",
        "C_OCCUPANCY_5_OF_30",
        "C_OCCUPANCY_12_OF_60",
        "D_BURDEN_25_OVER_30",
        "D_BURDEN_20_OVER_60",
    ]


def test_process_classification_does_not_inherit_application_exemptions():
    backend = {
        "executable": "/Applications/Docker.app/Contents/MacOS/com.docker.backend",
        "command_line": "com.docker.backend",
        "ancestors": [],
    }
    renderer = {
        "executable": "/Applications/Docker.app/Contents/MacOS/Docker Desktop.app/Contents/Frameworks/Docker Desktop Helper (Renderer).app/Contents/MacOS/Docker Desktop Helper (Renderer)",
        "command_line": "--type=renderer",
        "ancestors": [],
    }
    assert classify_process(backend)[0] == "required_service_host"
    assert classify_process(renderer) == (
        "unrelated_external_workload",
        "unrelated_external_workload:docker_desktop_ui",
    )


def test_browser_and_supervision_are_visible_not_exempt():
    chrome = {"executable": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
    codex = {"executable": "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Helper"}
    assert classify_process(chrome)[0] == "user_browser_workload"
    assert classify_process(codex) == (
        "run_supervision_ui_tooling",
        "run_supervision_ui_tooling:chatgpt_codex",
    )


def test_unknown_is_retained_and_cpu_scored():
    preregistration = load_preregistration(PREREGISTRATION)
    evaluated = evaluate_samples([sample(0, cpu=75)], preregistration)
    process = evaluated[0]["processes"][0]
    assert process["category"] == "unknown"
    assert evaluated[0]["candidates"]["A_INSTANTANEOUS_CURRENT"]["cpu_violation"]


def test_consecutive_policy_ignores_two_samples_and_detects_third():
    preregistration = load_preregistration(PREREGISTRATION)
    evaluated = evaluate_samples([sample(i, cpu=80) for i in range(3)], preregistration)
    values = [row["candidates"]["B_CONSECUTIVE_3"]["cpu_violation"] for row in evaluated]
    assert values == [False, False, True]
    assert not evaluated[-1]["candidates"]["B_CONSECUTIVE_5"]["cpu_violation"]


def test_consecutive_policy_resets_on_gap():
    preregistration = load_preregistration(PREREGISTRATION)
    samples = [sample(0, 80), sample(1, 80), sample(2, 0), sample(3, 80), sample(4, 80)]
    evaluated = evaluate_samples(samples, preregistration)
    assert not any(row["candidates"]["B_CONSECUTIVE_3"]["cpu_violation"] for row in evaluated)


def test_single_extreme_spike_is_not_rolling_interference():
    preregistration = load_preregistration(PREREGISTRATION)
    evaluated = evaluate_samples([sample(0, cpu=800)], preregistration)
    for candidate_id in (
        "C_OCCUPANCY_5_OF_30",
        "C_OCCUPANCY_12_OF_60",
        "D_BURDEN_25_OVER_30",
        "D_BURDEN_20_OVER_60",
    ):
        assert not evaluated[0]["candidates"][candidate_id]["cpu_violation"]


def test_fifteen_sustained_samples_detect_every_eligible_family():
    preregistration = load_preregistration(PREREGISTRATION)
    evaluated = evaluate_samples([sample(i, cpu=100) for i in range(15)], preregistration)
    last = evaluated[-1]["candidates"]
    assert all(
        last[candidate["id"]]["cpu_violation"]
        for candidate in preregistration["candidate_policies"]
        if candidate["eligible_for_selection"]
    )


def test_browser_hard_rule_is_independent_of_cpu():
    preregistration = load_preregistration(PREREGISTRATION)
    value = sample(0, cpu=0, browser_process_count=1)
    evaluated = evaluate_samples([value], preregistration)[0]["candidates"]
    assert all(item["hard_violation"] and "browser" in item["hard_reasons"] for item in evaluated.values())


def test_replay_and_summary_are_deterministic():
    preregistration = load_preregistration(PREREGISTRATION)
    samples = [sample(i, cpu=80 if i < 3 else 0) for i in range(5)]
    first = evaluate_samples(samples, preregistration)
    second = evaluate_samples(samples, preregistration)
    assert first == second
    assert summarize_evaluation(first) == summarize_evaluation(second)
    assert summarize_evaluation(first)["B_CONSECUTIVE_3"]["cpu_violation_episodes"] == 1
