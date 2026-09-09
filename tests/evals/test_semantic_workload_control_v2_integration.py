import asyncio
from contextlib import contextmanager, nullcontext
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


def finalization_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "completion.json").write_text(json.dumps({
        "invalidity_reasons": ["workload_control_v2_pending"],
        "official_baseline_eligible": False,
    }))
    (out / "deterministic.jsonl").write_text('{"case_id":"A"}\n')
    return out


def test_dataset_load_failure_leaves_frozen_completion_ineligible(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "load_dataset", lambda path: (_ for _ in ()).throw(OSError("dataset load")))
    with pytest.raises(OSError, match="dataset load"):
        launcher.finalize_artifacts(out, {"valid": True}, None)
    completion = json.loads((out / "completion.json").read_text())
    assert completion["official_baseline_eligible"] is False
    assert not (out / "workload_control_v2_completion.json").exists()


def test_summary_write_failure_leaves_no_eligible_completion(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "load_dataset", lambda path: ([{"id": "A"}], {}))
    monkeypatch.setattr(launcher.frozen, "deterministic_breakdowns", lambda cases, rows: {"cases": len(rows)})
    original_save = launcher.frozen.save

    def fail_summary(path, value):
        if Path(path).name == "deterministic_summary.json":
            raise OSError("summary write")
        return original_save(path, value)

    monkeypatch.setattr(launcher.frozen, "save", fail_summary)
    with pytest.raises(OSError, match="summary write"):
        launcher.finalize_artifacts(out, {"valid": True}, None)
    assert not (out / "workload_control_v2_completion.json").exists()


def test_manifest_write_failure_leaves_no_eligible_completion(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "load_dataset", lambda path: ([{"id": "A"}], {}))
    monkeypatch.setattr(launcher.frozen, "deterministic_breakdowns", lambda cases, rows: {"cases": len(rows)})
    original_save = launcher.frozen.save

    def fail_manifest(path, value):
        if Path(path).name == "workload_control_v2_files_sha256.json":
            raise OSError("manifest write")
        return original_save(path, value)

    monkeypatch.setattr(launcher.frozen, "save", fail_manifest)
    with pytest.raises(OSError, match="manifest write"):
        launcher.finalize_artifacts(out, {"valid": True}, None)
    assert not (out / "workload_control_v2_completion.json").exists()


def test_evidence_copy_failure_leaves_no_eligible_completion(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"type":"header"}\n')
    monkeypatch.setattr(launcher.shutil, "copyfile", lambda source, destination: (_ for _ in ()).throw(OSError("evidence copy")))
    with pytest.raises(OSError, match="evidence copy"):
        launcher.copy_raw_evidence(raw, out)
    assert not (out / "workload_control_v2_completion.json").exists()


def test_raw_evidence_fsync_failure_leaves_no_eligible_completion(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"type":"footer"}\n')
    monkeypatch.setattr(launcher, "sync_file_and_parent", lambda path: (_ for _ in ()).throw(OSError("raw fsync")))
    with pytest.raises(OSError, match="raw fsync"):
        launcher.copy_raw_evidence(raw, out)
    assert not (out / "workload_control_v2_completion.json").exists()


def test_summary_fsync_failure_leaves_no_eligible_completion(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "load_dataset", lambda path: ([{"id": "A"}], {}))
    monkeypatch.setattr(launcher.frozen, "deterministic_breakdowns", lambda cases, rows: {"cases": len(rows)})
    monkeypatch.setattr(
        launcher,
        "sync_file_and_parent",
        lambda path: (_ for _ in ()).throw(OSError("summary fsync"))
        if Path(path).name == "deterministic_summary.json" else None,
    )
    with pytest.raises(OSError, match="summary fsync"):
        launcher.finalize_artifacts(out, {"valid": True}, None)
    assert not (out / "workload_control_v2_completion.json").exists()


def test_manifest_fsync_failure_leaves_no_eligible_completion(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "load_dataset", lambda path: ([{"id": "A"}], {}))
    monkeypatch.setattr(launcher.frozen, "deterministic_breakdowns", lambda cases, rows: {"cases": len(rows)})
    def fail_manifest_sync(path):
        if Path(path).name == "workload_control_v2_files_sha256.json":
            raise OSError("manifest fsync")
    monkeypatch.setattr(launcher, "sync_file_and_parent", fail_manifest_sync)
    with pytest.raises(OSError, match="manifest fsync"):
        launcher.finalize_artifacts(out, {"valid": True}, None)
    assert not (out / "workload_control_v2_completion.json").exists()


@pytest.mark.parametrize("member_name", ["started.json", "completion.json", "files_sha256.json"])
def test_manifest_member_fsync_failure_leaves_no_eligible_completion(tmp_path, monkeypatch, member_name):
    out = finalization_output(tmp_path)
    for name in ("started.json", "files_sha256.json"):
        (out / name).write_text("{}")
    monkeypatch.setattr(launcher, "load_dataset", lambda path: ([{"id": "A"}], {}))
    monkeypatch.setattr(launcher.frozen, "deterministic_breakdowns", lambda cases, rows: {"cases": len(rows)})

    def fail_member_sync(path):
        if Path(path).name == member_name:
            raise OSError(f"{member_name} fsync")

    monkeypatch.setattr(launcher, "sync_file_and_parent", fail_member_sync)
    with pytest.raises(OSError, match=f"{member_name} fsync"):
        launcher.finalize_artifacts(out, {"valid": True}, None)
    assert not (out / "workload_control_v2_completion.json").exists()


def test_closed_raw_evidence_includes_footer_and_final_sample(tmp_path):
    raw = tmp_path / "raw.jsonl"
    out = tmp_path / "out"
    out.mkdir()
    raw.write_text('{"type":"header"}\n{"type":"sample","index":7}\n{"type":"footer","sample_count":8}\n')
    launcher.copy_raw_evidence(raw, out)
    assert (out / "workload_control_v2.jsonl").read_text().endswith(
        '{"type":"footer","sample_count":8}\n'
    )


def test_invalid_final_monitor_verdict_skips_summary_and_manifest(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "load_dataset", lambda path: pytest.fail("invalid monitor must skip summary"))
    launcher.finalize_artifacts(
        out,
        {"valid": False, "sample_count": 12, "first_violation": {"sample_index": 11}},
    )
    assert not (out / "deterministic_summary.json").exists()
    assert not (out / "workload_control_v2_files_sha256.json").exists()


def test_authoritative_post_rename_durability_failure_removes_visible_record(tmp_path, monkeypatch):
    path = tmp_path / "workload_control_v2_completion.json"
    calls = {"count": 0}
    original_fsync = launcher.os.fsync

    def fail_directory_fsync(descriptor):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("directory durability failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(launcher.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="directory durability failed"):
        launcher.publish_authoritative(path, {"official_baseline_eligible": True})
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    assert not path.with_name(path.name + ".ineligible.tmp").exists()


def test_monitor_violation_during_finalization_is_ineligible(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "file_sha", lambda path: "digest")
    launcher.finalize_output(
        out,
        {"valid": False, "sample_count": 11, "first_violation": {"sample_index": 10}},
        {"reviewed_commit": "a" * 40},
    )
    completion = json.loads((out / "workload_control_v2_completion.json").read_text())
    assert completion["status"] == "invalid_diagnostic"
    assert completion["official_baseline_eligible"] is False


def test_monitor_shutdown_failure_is_ineligible_after_artifact_preparation(tmp_path, monkeypatch):
    out = finalization_output(tmp_path)
    monkeypatch.setattr(launcher, "file_sha", lambda path: "digest")
    launcher.finalize_output(
        out,
        {"valid": False, "sample_count": 12, "first_violation": None},
        {"reviewed_commit": "a" * 40},
        RuntimeError("monitor shutdown failed"),
    )
    completion = json.loads((out / "workload_control_v2_completion.json").read_text())
    assert completion["status"] == "invalid_diagnostic"
    assert completion["official_baseline_eligible"] is False
    assert completion["artifact_finalization_error"]["type"] == "RuntimeError"


def test_run_orders_closed_monitor_before_artifact_publication(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    events = []
    head = "a" * 40

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            self.raw_path = Path(args[0])

        def start(self):
            events.append("start")
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path.write_text("raw")

        def stop(self):
            events.append("stop")
            return {"valid": True, "sample_count": 1, "first_violation": None}

    class Awake:
        def terminate(self):
            events.append("awake_terminate")

        def wait(self, timeout):
            events.append("awake_wait")

    monkeypatch.setattr(launcher, "WorkloadControlV2Monitor", FakeMonitor)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: Awake())
    monkeypatch.setattr(launcher, "verify_opt_in", lambda args: (head, {}))
    monkeypatch.setattr(launcher, "file_sha", lambda path: "digest")

    @contextmanager
    def adapter(start_monitor, monitor):
        start_monitor()
        yield

    monkeypatch.setattr(launcher, "frozen_launcher_adapter", adapter)

    async def run_once(args):
        out = args.out_root / head
        out.mkdir(parents=True)
        (out / "completion.json").write_text(json.dumps({"invalidity_reasons": []}))

    monkeypatch.setattr(launcher.frozen, "run_once", run_once)
    monkeypatch.setattr(launcher, "copy_raw_evidence", lambda raw, out: events.append("copy") or "digest")
    monkeypatch.setattr(launcher, "finalize_artifacts", lambda out, summary, error=None: events.append("artifacts"))
    monkeypatch.setattr(launcher, "finalize_output", lambda out, summary, review, error=None: events.append("finalize"))

    asyncio.run(launcher.run(SimpleNamespace(out_root=tmp_path / "out")))
    assert events == ["start", "stop", "awake_terminate", "awake_wait", "copy", "artifacts", "finalize"]


def test_run_preserves_closed_raw_evidence_on_monitor_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    head = "c" * 40
    copied = []

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            self.raw_path = Path(args[0])

        def start(self):
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path.write_text('{"type":"footer","sample_count":1}\n')

        def stop(self):
            return {"valid": False, "sample_count": 1, "first_violation": None, "monitor_error": "collector failed"}

    class Awake:
        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    monkeypatch.setattr(launcher, "WorkloadControlV2Monitor", FakeMonitor)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: Awake())
    monkeypatch.setattr(launcher, "verify_opt_in", lambda args: (head, {}))
    monkeypatch.setattr(launcher, "file_sha", lambda path: "digest")

    @contextmanager
    def adapter(start_monitor, monitor):
        start_monitor()
        yield

    monkeypatch.setattr(launcher, "frozen_launcher_adapter", adapter)

    async def run_once(args):
        out = args.out_root / head
        out.mkdir(parents=True)
        (out / "completion.json").write_text(json.dumps({"invalidity_reasons": []}))

    monkeypatch.setattr(launcher.frozen, "run_once", run_once)
    monkeypatch.setattr(launcher, "copy_raw_evidence", lambda raw, out: copied.append(out / "workload_control_v2.jsonl") or "digest")
    monkeypatch.setattr(launcher, "finalize_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher, "finalize_artifacts", lambda *args, **kwargs: pytest.fail("invalid monitor must skip official artifacts"))

    with pytest.raises(RuntimeError, match="collector failed"):
        asyncio.run(launcher.run(SimpleNamespace(out_root=tmp_path / "out")))
    assert copied == [tmp_path / "out" / head / "workload_control_v2.jsonl"]


def test_run_monitor_shutdown_failure_publishes_only_ineligible_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    head = "b" * 40

    class FakeMonitor:
        def __init__(self, *args, **kwargs):
            self.raw_path = Path(args[0])

        def start(self):
            self.raw_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_path.write_text("raw")

        def stop(self):
            raise RuntimeError("monitor shutdown failed")

    class Awake:
        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    monkeypatch.setattr(launcher, "WorkloadControlV2Monitor", FakeMonitor)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: Awake())
    monkeypatch.setattr(launcher, "verify_opt_in", lambda args: (head, {}))
    monkeypatch.setattr(launcher, "file_sha", lambda path: "digest")

    @contextmanager
    def adapter(start_monitor, monitor):
        start_monitor()
        yield

    monkeypatch.setattr(launcher, "frozen_launcher_adapter", adapter)

    async def run_once(args):
        out = args.out_root / head
        out.mkdir(parents=True)
        (out / "completion.json").write_text(json.dumps({"invalidity_reasons": []}))

    monkeypatch.setattr(launcher.frozen, "run_once", run_once)
    monkeypatch.setattr(launcher, "finalize_artifacts", lambda out, summary, error=None: None)
    with pytest.raises(RuntimeError, match="monitor shutdown failed"):
        asyncio.run(launcher.run(SimpleNamespace(out_root=tmp_path / "out")))
    completion = json.loads((tmp_path / "out" / head / "workload_control_v2_completion.json").read_text())
    assert completion["official_baseline_eligible"] is False
    assert completion["status"] == "invalid_diagnostic"


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
