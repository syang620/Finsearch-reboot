import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evals.agents import run_semantic_baseline_v2_2 as launcher
from scripts.evals.agents import semantic_workload_control_v2 as control


class Awake:
    pid = 4242

    def poll(self):
        return None


def process(cpu):
    return [{
        "pid": 100,
        "ppid": 1,
        "cpu": cpu,
        "executable": "/usr/bin/unrelated",
        "command_line": "unrelated",
        "ancestors": [],
    }]


def test_monitor_applies_exact_b10_and_preserves_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(control, "CADENCE_TOLERANCE_SECONDS", 0.1)
    monkeypatch.setattr(control, "controls", lambda: {
        "ac_power": True,
        "low_power_mode": 0,
        "browser_process_count": 0,
        "heavy_non_model_processes": [{"cpu": 75}],
    })
    monkeypatch.setattr(control, "capture_processes", lambda *args: process(75))
    raw = tmp_path / "raw.jsonl"
    monitor = control.WorkloadControlV2Monitor(
        raw, launcher.PREREGISTRATION, launcher.CONTROL_CONTRACT, Awake()
    )
    monitor.start()
    while monitor.samples < 10:
        monitor.stop_event.wait(0.002)
    summary = monitor.stop()
    assert not summary["valid"]
    assert summary["first_violation"]["sample_index"] == 9
    assert summary["first_violation"]["cpu_violation"]
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    assert records[0]["type"] == "header" and records[-1]["type"] == "footer"
    assert sum(row.get("type") == "sample" for row in records) == summary["sample_count"]


def test_monitor_does_not_treat_short_burst_as_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(control, "CADENCE_TOLERANCE_SECONDS", 0.1)
    monkeypatch.setattr(control, "controls", lambda: {
        "ac_power": True, "low_power_mode": 0,
        "browser_process_count": 0, "heavy_non_model_processes": [],
    })
    calls = {"count": 0}

    def captured(*args):
        calls["count"] += 1
        return process(75 if calls["count"] <= 6 else 0)

    monkeypatch.setattr(control, "capture_processes", captured)
    monitor = control.WorkloadControlV2Monitor(
        tmp_path / "raw.jsonl", launcher.PREREGISTRATION,
        launcher.CONTROL_CONTRACT, Awake(),
    )
    monitor.start()
    while monitor.samples < 12:
        monitor.stop_event.wait(0.002)
    summary = monitor.stop()
    assert summary["valid"] and summary["first_violation"] is None


def test_terminal_only_rejects_active_supervision_but_allows_inert_handler():
    active = {
        "category": "run_supervision_ui_tooling", "cpu": 0,
        "executable": "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT",
        "ancestors": [],
    }
    assert not control.terminal_only_result({"processes": [active]})["valid"]
    handler = {
        "category": "run_supervision_ui_tooling", "cpu": 0,
        "pid": 12, "executable": "/tmp/browser_crashpad_handler",
        "ancestors": [{"pid": 1, "executable": "/sbin/launchd"}],
    }
    result = control.terminal_only_result({"processes": [handler]})
    assert result["valid"] and result["retained_inert_crash_handler_count"] == 1


def test_opt_in_is_explicit_and_hash_bound(monkeypatch):
    args = SimpleNamespace(workload_control_v2=control.POLICY, integration_approval=Path("approval"))
    monkeypatch.setattr(launcher.frozen, "clean_checkout", lambda: None)
    git_calls = []

    def git(*arguments):
        git_calls.append(arguments)
        return "a" * 40 if arguments[0] == "rev-parse" else ""

    monkeypatch.setattr(launcher, "git", git)
    expected = {
        launcher.PREREGISTRATION: launcher.EXPECTED_PREREGISTRATION_SHA256,
        launcher.CONTROL_CONTRACT: launcher.EXPECTED_CONTRACT_SHA256,
        launcher.CONTROL_IMPLEMENTATION: launcher.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        launcher.CONTROL_COLLECTOR: launcher.EXPECTED_CONTROL_COLLECTOR_SHA256,
        launcher.CONTROL_OBSERVER: launcher.EXPECTED_CONTROL_OBSERVER_SHA256,
        launcher.FROZEN_PROVENANCE: launcher.EXPECTED_FROZEN_PROVENANCE_SHA256,
        launcher.LAUNCHER: "launcher",
        launcher.ADAPTER: "adapter",
    }
    monkeypatch.setattr(launcher, "file_sha", lambda path: expected[path])
    approval = {
        "status": "approved_for_one_semantic_v2_control_v2_attempt",
        "reviewed_commit": "b" * 40,
        "selected_policy": control.POLICY,
        "integration_launcher_sha256": "launcher",
        "integration_adapter_sha256": "adapter",
        "control_contract_sha256": launcher.EXPECTED_CONTRACT_SHA256,
        "control_implementation_sha256": launcher.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        "control_collector_sha256": launcher.EXPECTED_CONTROL_COLLECTOR_SHA256,
        "control_observer_sha256": launcher.EXPECTED_CONTROL_OBSERVER_SHA256,
    }
    monkeypatch.setattr(launcher.frozen, "committed_approval", lambda path: approval)
    verified = []
    monkeypatch.setattr(launcher.frozen, "verify_remote_review", lambda record: verified.append(record))
    head, review = launcher.verify_opt_in(args)
    assert head == "a" * 40 and review == approval and verified == [approval]
    diff_call = next(call for call in git_calls if call[0] == "diff")
    assert str(launcher.CONTROL_COLLECTOR) in diff_call
    assert str(launcher.CONTROL_OBSERVER) in diff_call
    args.workload_control_v2 = "A_INSTANTANEOUS_CURRENT"
    with pytest.raises(ValueError, match="Explicit"):
        launcher.verify_opt_in(args)


@pytest.mark.parametrize("reviewed", ["short", "A" * 40, "g" * 40, None, 123])
def test_integration_approval_requires_full_lowercase_sha(monkeypatch, reviewed):
    args = SimpleNamespace(workload_control_v2=control.POLICY, integration_approval=Path("approval"))
    monkeypatch.setattr(launcher.frozen, "clean_checkout", lambda: None)
    monkeypatch.setattr(launcher, "git", lambda *args: "a" * 40 if args[0] == "rev-parse" else "")
    monkeypatch.setattr(launcher, "file_sha", lambda path: {
        launcher.PREREGISTRATION: launcher.EXPECTED_PREREGISTRATION_SHA256,
        launcher.CONTROL_CONTRACT: launcher.EXPECTED_CONTRACT_SHA256,
        launcher.CONTROL_IMPLEMENTATION: launcher.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        launcher.CONTROL_COLLECTOR: launcher.EXPECTED_CONTROL_COLLECTOR_SHA256,
        launcher.CONTROL_OBSERVER: launcher.EXPECTED_CONTROL_OBSERVER_SHA256,
        launcher.FROZEN_PROVENANCE: launcher.EXPECTED_FROZEN_PROVENANCE_SHA256,
        launcher.LAUNCHER: "launcher", launcher.ADAPTER: "adapter",
    }[path])
    monkeypatch.setattr(launcher.frozen, "committed_approval", lambda path: {
        "status": "approved_for_one_semantic_v2_control_v2_attempt",
        "reviewed_commit": reviewed, "selected_policy": control.POLICY,
        "integration_launcher_sha256": "launcher", "integration_adapter_sha256": "adapter",
        "control_contract_sha256": launcher.EXPECTED_CONTRACT_SHA256,
        "control_implementation_sha256": launcher.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        "control_collector_sha256": launcher.EXPECTED_CONTROL_COLLECTOR_SHA256,
        "control_observer_sha256": launcher.EXPECTED_CONTROL_OBSERVER_SHA256,
    })
    with pytest.raises(ValueError, match="approval mismatch"):
        launcher.verify_opt_in(args)


def test_adapter_restores_frozen_launcher_hooks(monkeypatch):
    async def settled(stage):
        return {"stage": stage}

    monkeypatch.setattr(launcher.frozen, "settle_preflight", settled)
    original = {name: getattr(launcher.frozen, name) for name in (
        "verify_launcher", "settle_preflight", "controls", "service_preflight",
        "check_controls", "finalize_validity"
    )}
    started = []
    monitor = SimpleNamespace(trigger=None, error=None)
    monkeypatch.setattr(launcher, "verify_frozen_launcher", lambda approval: ({}, {}, {}))
    with launcher.frozen_launcher_adapter(lambda: started.append(True), monitor):
        asyncio.run(launcher.frozen.settle_preflight("index_verification_and_planner_setup"))
        assert started == [True]
        assert launcher.frozen.finalize_validity is launcher.provisional_validity
    for name, value in original.items():
        assert getattr(launcher.frozen, name) is value


def test_adapter_stops_before_next_case_after_control_trigger(monkeypatch):
    async def settled(stage):
        return {"stage": stage}

    monkeypatch.setattr(launcher.frozen, "settle_preflight", settled)
    monkeypatch.setattr(launcher.frozen, "controls", lambda: {"ac_power": True, "low_power_mode": 0})
    monitor = SimpleNamespace(trigger=None, error=None)
    with launcher.frozen_launcher_adapter(lambda: None, monitor):
        asyncio.run(launcher.frozen.settle_preflight("index_verification_and_planner_setup"))
        for _ in range(58):
            launcher.frozen.controls()  # before case
            launcher.frozen.controls()  # after case
        launcher.frozen.controls()  # case 59 starts
        monitor.trigger = {"sample_index": 9}
        launcher.frozen.controls()  # after the in-flight case is retained
        with pytest.raises(RuntimeError, match="invalidated"):
            launcher.frozen.controls()  # next case cannot start


def test_service_identity_is_exact_except_latency(tmp_path, monkeypatch):
    expected = {
        "ollama_version": {"version": "1"},
        "model_identities": {"model": {"digest": "abc"}},
        "qdrant_service": {"title": "q", "version": "2", "commit": "def"},
        "sec_health": {"status_code": 200, "wall_ms": 1},
        "reranker_health": {
            "metadata": {"applied_backend": "qwen3_api", "fallback_used": False},
            "requested_model": "reranker", "wall_ms": 2,
        },
    }
    provenance = tmp_path / "started.json"
    provenance.write_text(json.dumps({"service_preflight": expected}))
    monkeypatch.setattr(launcher, "FROZEN_PROVENANCE", provenance)
    observed = json.loads(json.dumps(expected))
    observed["sec_health"]["wall_ms"] = 999
    assert launcher.verified_service_preflight(lambda: observed) == observed
    observed["qdrant_service"]["version"] = "changed"
    with pytest.raises(ValueError, match="qdrant_service.*False"):
        launcher.verified_service_preflight(lambda: observed)


def test_finalizer_cannot_turn_failed_control_into_baseline(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "completion.json").write_text(json.dumps({
        "invalidity_reasons": [], "control_violations": [],
    }))
    monkeypatch.setattr(launcher, "file_sha", lambda path: "digest")
    launcher.finalize_output(out, {
        "valid": False, "sample_count": 10, "first_violation": {"sample_index": 9},
    }, {"reviewed_commit": "a" * 40})
    completion = json.loads((out / "workload_control_v2_completion.json").read_text())
    assert completion["status"] == "invalid_diagnostic"
    assert not completion["official_baseline_eligible"]
    assert completion["invalidity_reasons"] == ["workload_control_v2_failed"]


def test_provisional_validity_never_claims_eligibility_and_keeps_hard_failures():
    ending = {
        "captured_cases": ["A"], "evaluation_errors": [],
        "control_violations": [], "model_identities_unchanged": True,
        "index_unchanged": True,
    }
    assert not launcher.provisional_validity(ending, ["A"], ["A"])
    assert ending["status"] == "pending_workload_control_v2"
    assert ending["invalidity_reasons"] == ["workload_control_v2_pending"]
    ending["control_violations"] = [{"reason": "AC power"}]
    assert not launcher.provisional_validity(ending, ["A"], ["A"])
    assert "hard_control_violations" in ending["invalidity_reasons"]


def test_preactivation_failure_is_not_masked(monkeypatch, tmp_path):
    args = SimpleNamespace(out_root=tmp_path)
    monkeypatch.setattr(launcher, "verify_opt_in", lambda args: ("a" * 40, {}))
    monkeypatch.setattr(launcher, "frozen_launcher_adapter", lambda *args: nullcontext())

    async def fail(args):
        raise ValueError("original preflight failure")

    monkeypatch.setattr(launcher.frozen, "run_once", fail)

    class Process:
        pid = 42

        def terminate(self):
            pass

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(ValueError, match="original preflight failure"):
        asyncio.run(launcher.run(args))
