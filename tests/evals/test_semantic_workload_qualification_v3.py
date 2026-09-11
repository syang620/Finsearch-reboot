import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evals.agents import run_semantic_baseline_v2_1 as frozen
from scripts.evals.agents import run_semantic_baseline_v2_4 as launcher
from scripts.evals.agents import semantic_workload_qualification_v3 as qualification


def completion(**overrides):
    value = {
        "status": "pending_workload_qualification_v3",
        "official_baseline_eligible": False,
        "invalidity_reasons": ["workload_qualification_v3_pending"],
        "captured_cases": ["A"],
        "capture_complete": True,
        "evaluation_errors": [],
        "control_violations": [],
        "controls_after": {"ac_power": True, "low_power_mode": 0},
        "model_identities_unchanged": True,
        "index_unchanged": True,
    }
    value.update(overrides)
    return value


def monitor(**overrides):
    value = {
        "sample_count": 20,
        "first_violation": None,
        "awake_protection_active_through_final_sample": True,
        "cadence_within_frozen_tolerance": True,
        "monitor_error": None,
    }
    value.update(overrides)
    return value


def observations(**overrides):
    value = {
        "sample_count": 20,
        "browser_samples": 0,
        "active_supervision_samples": 0,
        "sustained_external_cpu_samples": 0,
        "ac_power_violation_samples": 0,
        "low_power_mode_violation_samples": 0,
        "header_present": True,
        "footer_present": True,
        "header_count": 1,
        "footer_count": 1,
        "indices_contiguous": True,
        "framing_valid": True,
    }
    value.update(overrides)
    return value


def test_quiet_complete_run_meets_both_dimensions_pending_activation():
    record = qualification.qualification_record(
        completion(), monitor(), observations(), {"reviewed_commit": "a" * 40}
    )
    assert record["status"] == "complete"
    assert record["eligibility"]["answer_quality"] == {
        "requirements_met": True,
        "eligible": False,
        "activation_required": True,
        "reasons": [],
    }
    assert record["eligibility"]["controlled_latency"] == {
        "requirements_met": True,
        "eligible": False,
        "activation_required": True,
        "reasons": [],
    }
    assert "official_baseline_eligible" not in record
    assert "invalidity_reasons" not in record


def test_candidate_has_no_execution_approval():
    assert not launcher.APPROVAL.exists()
    contract = json.loads(launcher.CONTRACT.read_text())
    assert contract["status"] == "inactive_candidate_pending_review"
    assert "grants no semantic execution" in contract["execution_gate"]


def test_ambient_workload_only_disqualifies_controlled_latency():
    record = qualification.qualification_record(
        completion(case_outcomes=["analyst_timeout"]),
        monitor(cadence_within_frozen_tolerance=False),
        observations(
            browser_samples=2,
            active_supervision_samples=20,
            sustained_external_cpu_samples=3,
        ),
        {},
    )
    assert record["eligibility"]["answer_quality"] == {
        "requirements_met": True,
        "eligible": False,
        "activation_required": True,
        "reasons": [],
    }
    assert record["eligibility"]["controlled_latency"] == {
        "requirements_met": False,
        "eligible": False,
        "activation_required": False,
        "reasons": [
            "browser_present",
            "active_supervision_ui",
            "sustained_external_cpu",
            "sampling_cadence_out_of_tolerance",
        ],
    }
    assert record["case_outcomes"] == ["analyst_timeout"]


def test_incomplete_answer_evidence_fails_both_dimensions():
    record = qualification.qualification_record(
        completion(
            capture_complete=False,
            invalidity_reasons=[
                "incomplete_or_duplicate_capture",
                "workload_qualification_v3_pending",
            ],
        ),
        monitor(),
        observations(),
        {},
    )
    assert record["status"] == "incomplete_diagnostic"
    assert record["eligibility"]["answer_quality"] == {
        "requirements_met": False,
        "eligible": False,
        "activation_required": False,
        "reasons": ["incomplete_or_duplicate_capture"],
    }
    assert record["eligibility"]["controlled_latency"] == {
        "requirements_met": False,
        "eligible": False,
        "activation_required": False,
        "reasons": ["answer_quality_ineligible"],
    }


def test_monitor_capture_and_power_fail_closed_for_answer_quality():
    record = qualification.qualification_record(
        completion(control_violations=[{"reason": "power"}]),
        monitor(
            sample_count=0,
            awake_protection_active_through_final_sample=False,
            monitor_error="collector stopped",
        ),
        observations(
            sample_count=0,
            footer_present=False,
            footer_count=0,
            ac_power_violation_samples=1,
        ),
        {},
    )
    reasons = record["eligibility"]["answer_quality"]["reasons"]
    assert reasons == [
        "power_control_violation",
        "monitor_capture_incomplete",
        "awake_protection_failed",
    ]
    assert record["status"] == "incomplete_diagnostic"


def test_final_power_sample_fails_answer_requirements_closed():
    record = qualification.qualification_record(
        completion(controls_after={"ac_power": False, "low_power_mode": 0}),
        monitor(),
        observations(),
        {},
    )
    assert record["eligibility"]["answer_quality"] == {
        "requirements_met": False,
        "eligible": False,
        "activation_required": False,
        "reasons": ["power_control_violation"],
    }
    assert record["eligibility"]["controlled_latency"]["requirements_met"] is False


def test_operation_error_fails_answer_quality_without_generic_flag():
    record = qualification.qualification_record(
        completion(), monitor(), observations(), {}, RuntimeError("failed")
    )
    assert not record["eligibility"]["answer_quality"]["requirements_met"]
    assert "execution_or_finalization_failed" in record["eligibility"][
        "answer_quality"
    ]["reasons"]
    assert record["operation_error"] == {
        "type": "RuntimeError",
        "message": "Operation or artifact finalization failed",
    }
    assert "official_baseline_eligible" not in record


def test_provisional_completion_is_always_fail_closed():
    ending = {
        "captured_cases": ["A"],
        "evaluation_errors": [],
        "control_violations": [],
        "controls_after": {"ac_power": True, "low_power_mode": 0},
        "model_identities_unchanged": True,
        "index_unchanged": True,
    }
    assert not qualification.provisional_validity(ending, ["A"], ["A"])
    assert ending["capture_complete"]
    assert ending["official_baseline_eligible"] is False
    assert ending["invalidity_reasons"] == [
        "workload_qualification_v3_pending"
    ]


def test_adapter_never_stops_cases_for_ambient_monitor_trigger(monkeypatch):
    calls = []

    async def settle(stage):
        return {"stage": stage}

    def controls():
        calls.append("control")
        return {"ac_power": True, "low_power_mode": 0}

    monkeypatch.setattr(frozen, "settle_preflight", settle)
    monkeypatch.setattr(frozen, "controls", controls)
    original = {
        name: getattr(frozen, name)
        for name in (
            "verify_launcher",
            "settle_preflight",
            "service_preflight",
            "check_controls",
            "finalize_validity",
            "controls",
        )
    }
    started = []
    with qualification.frozen_launcher_adapter(
        lambda: started.append(True), lambda path: None
    ):
        asyncio.run(frozen.settle_preflight("index_verification_and_planner_setup"))
        for _ in range(120):
            frozen.controls()
    assert started == [True]
    assert len(calls) == 120
    for name, value in original.items():
        assert getattr(frozen, name) is value


def test_adapter_routes_service_preflight_through_exact_verifier(monkeypatch):
    observed = {"service": "observed"}
    calls = []

    def service_preflight(config):
        calls.append(("preflight", config))
        return observed

    def verified_service_preflight(operation):
        calls.append(("verifier", None))
        return operation()

    monkeypatch.setattr(frozen, "service_preflight", service_preflight)
    monkeypatch.setattr(
        qualification.legacy,
        "verified_service_preflight",
        verified_service_preflight,
    )
    with qualification.frozen_launcher_adapter(lambda: None, lambda path: None):
        assert frozen.service_preflight("config") is observed
    assert calls == [("verifier", None), ("preflight", "config")]


def test_raw_analysis_aggregates_all_performance_reasons(tmp_path):
    path = tmp_path / "raw.jsonl"
    records = [
        {"type": "header"},
        {
            "type": "sample",
            "index": 0,
            "selected_policy_result": {
                "hard_reasons": ["browser"],
                "cpu_violation": True,
            },
            "terminal_only_result": {"valid": False},
        },
        {
            "type": "sample",
            "index": 1,
            "selected_policy_result": {
                "hard_reasons": ["ac_power", "low_power_mode"],
                "cpu_violation": False,
            },
            "terminal_only_result": {"valid": True},
        },
        {"type": "footer"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert qualification.analyze_raw(path) == {
        "sample_count": 2,
        "browser_samples": 1,
        "active_supervision_samples": 1,
        "sustained_external_cpu_samples": 1,
        "ac_power_violation_samples": 1,
        "low_power_mode_violation_samples": 1,
        "header_present": True,
        "footer_present": True,
        "header_count": 1,
        "footer_count": 1,
        "indices_contiguous": True,
        "framing_valid": True,
    }


def test_authoritative_record_has_no_generic_eligibility(tmp_path):
    path = tmp_path / qualification.COMPLETION_NAME
    record = qualification.qualification_record(
        completion(), monitor(), observations(), {}
    )
    qualification.publish_authoritative(path, record)
    loaded = json.loads(path.read_text())
    assert loaded == record
    assert "official_baseline_eligible" not in loaded
    assert not loaded["eligibility"]["answer_quality"]["eligible"]
    assert not loaded["eligibility"]["controlled_latency"]["eligible"]


def test_directory_sync_failure_removes_visible_eligibility(tmp_path, monkeypatch):
    path = tmp_path / qualification.COMPLETION_NAME
    record = qualification.qualification_record(
        completion(), monitor(), observations(), {}
    )
    original_fsync = qualification.os.fsync
    calls = {"count": 0}

    def fail_first_directory_sync(descriptor):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("directory sync failed")
        return original_fsync(descriptor)

    monkeypatch.setattr(qualification.os, "fsync", fail_first_directory_sync)
    try:
        qualification.publish_authoritative(path, record)
    except OSError as exc:
        assert str(exc) == "directory sync failed"
    else:
        raise AssertionError("publication should fail")
    assert not path.exists()


def test_cleanup_failure_replaces_candidate_with_fail_closed_record(
    tmp_path, monkeypatch
):
    path = tmp_path / qualification.COMPLETION_NAME
    record = qualification.qualification_record(
        completion(), monitor(), observations(), {}
    )
    original_fsync = qualification.os.fsync
    original_unlink = Path.unlink
    calls = {"count": 0}

    def fail_first_directory_sync(descriptor):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("directory sync failed")
        return original_fsync(descriptor)

    def refuse_candidate_cleanup(self, *args, **kwargs):
        if self == path:
            raise PermissionError("candidate cleanup failed")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(qualification.os, "fsync", fail_first_directory_sync)
    monkeypatch.setattr(Path, "unlink", refuse_candidate_cleanup)
    try:
        qualification.publish_authoritative(path, record)
    except OSError as exc:
        assert str(exc) == "directory sync failed"
    else:
        raise AssertionError("publication should fail")
    failed = json.loads(path.read_text())
    assert failed["status"] == "incomplete_diagnostic"
    for dimension in failed["eligibility"].values():
        assert dimension["requirements_met"] is False
        assert dimension["eligible"] is False
        assert dimension["activation_required"] is False
        assert "authoritative_publication_failed" in dimension["reasons"]


def test_frozen_performance_dependency_change_is_rejected(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "file_sha",
        lambda path: "changed"
        if path == launcher.legacy.CONTROL_IMPLEMENTATION
        else {
            launcher.legacy.PREREGISTRATION:
                launcher.legacy.EXPECTED_PREREGISTRATION_SHA256,
            launcher.legacy.CONTROL_CONTRACT:
                launcher.legacy.EXPECTED_CONTRACT_SHA256,
            launcher.legacy.CONTROL_COLLECTOR:
                launcher.legacy.EXPECTED_CONTROL_COLLECTOR_SHA256,
            launcher.legacy.CONTROL_OBSERVER:
                launcher.legacy.EXPECTED_CONTROL_OBSERVER_SHA256,
            launcher.legacy.ADAPTER:
                qualification.PERFORMANCE_ADAPTER_SHA256,
            launcher.legacy.FROZEN_PROVENANCE:
                launcher.legacy.EXPECTED_FROZEN_PROVENANCE_SHA256,
        }[path],
    )
    try:
        launcher.verify_frozen_performance_dependencies()
    except ValueError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("changed performance dependency should fail")


def test_frozen_service_provenance_change_is_rejected(monkeypatch):
    expected = {
        launcher.legacy.PREREGISTRATION:
            launcher.legacy.EXPECTED_PREREGISTRATION_SHA256,
        launcher.legacy.CONTROL_CONTRACT:
            launcher.legacy.EXPECTED_CONTRACT_SHA256,
        launcher.legacy.CONTROL_IMPLEMENTATION:
            launcher.legacy.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        launcher.legacy.CONTROL_COLLECTOR:
            launcher.legacy.EXPECTED_CONTROL_COLLECTOR_SHA256,
        launcher.legacy.CONTROL_OBSERVER:
            launcher.legacy.EXPECTED_CONTROL_OBSERVER_SHA256,
        launcher.legacy.ADAPTER: qualification.PERFORMANCE_ADAPTER_SHA256,
        launcher.legacy.FROZEN_PROVENANCE:
            launcher.legacy.EXPECTED_FROZEN_PROVENANCE_SHA256,
    }
    monkeypatch.setattr(
        launcher,
        "file_sha",
        lambda path: "changed"
        if path == launcher.legacy.FROZEN_PROVENANCE
        else expected[path],
    )
    try:
        launcher.verify_frozen_performance_dependencies()
    except ValueError as exc:
        assert "identity changed" in str(exc)
    else:
        raise AssertionError("changed service provenance should fail")


def test_review_guard_covers_every_executable_dependency():
    attestation = Path("docs/evals/index_attestation.json")
    reviewed = set(launcher.approval_reviewed_paths(attestation))
    assert {
        launcher.frozen.LAUNCHER,
        launcher.frozen.CONTRACT,
        launcher.legacy.PREREGISTRATION,
        launcher.legacy.CONTROL_CONTRACT,
        launcher.legacy.CONTROL_IMPLEMENTATION,
        launcher.legacy.CONTROL_COLLECTOR,
        launcher.legacy.CONTROL_OBSERVER,
        launcher.legacy.ADAPTER,
        launcher.legacy.LAUNCHER,
        launcher.legacy.FROZEN_PROVENANCE,
        launcher.canonical.LAUNCHER,
        launcher.canonical.CANONICAL_VERIFIER,
        attestation,
        launcher.LAUNCHER,
        launcher.ADAPTER,
        launcher.CONTRACT,
    } == reviewed


class FakeAwake:
    pid = 321

    def terminate(self):
        pass

    def wait(self, timeout):
        return 0


class ViolatingMonitor:
    def __init__(self, raw, *args, **kwargs):
        self.raw = Path(raw)

    def start(self):
        self.raw.parent.mkdir(parents=True)
        self.raw.write_text(
            json.dumps({"type": "header"})
            + "\n"
            + json.dumps(
                {
                    "type": "sample",
                    "index": 0,
                    "selected_policy_result": {
                        "hard_reasons": ["browser"],
                        "cpu_violation": False,
                    },
                    "terminal_only_result": {"valid": False},
                }
            )
            + "\n"
            + json.dumps({"type": "footer"})
            + "\n"
        )

    def stop(self):
        return {
            "sample_count": 1,
            "awake_protection_active_through_final_sample": True,
            "cadence_within_frozen_tolerance": False,
            "monitor_error": None,
        }


def test_run_keeps_all_cases_after_workload_violation(tmp_path, monkeypatch):
    head = "a" * 40
    args = SimpleNamespace(out_root=tmp_path / "out")
    calls = []

    async def settle(stage):
        return {"stage": stage}

    async def run_once(_args):
        await frozen.settle_preflight("index_verification_and_planner_setup")
        for index in range(60):
            calls.append(index)
        out = args.out_root / head
        out.mkdir(parents=True)
        (out / "completion.json").write_text(
            json.dumps(completion(captured_cases=[str(i) for i in range(60)]))
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(frozen, "settle_preflight", settle)
    monkeypatch.setattr(qualification.subprocess, "Popen", lambda *a, **k: FakeAwake())
    monkeypatch.setattr(qualification, "WorkloadControlV2Monitor", ViolatingMonitor)
    monkeypatch.setattr(
        qualification,
        "prepare_answer_artifacts",
        lambda out: (Path(out) / "deterministic_summary.json").write_text("{}"),
    )
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    asyncio.run(
        qualification.run(
            args,
            head,
            {},
            run_once,
            lambda path: None,
            contract,
        )
    )
    assert calls == list(range(60))
    final = json.loads(
        (
            args.out_root
            / head
            / qualification.COMPLETION_NAME
        ).read_text()
    )
    assert final["eligibility"]["answer_quality"] == {
        "requirements_met": True,
        "eligible": False,
        "activation_required": True,
        "reasons": [],
    }
    assert final["eligibility"]["controlled_latency"] == {
        "requirements_met": False,
        "eligible": False,
        "activation_required": False,
        "reasons": [
            "browser_present",
            "active_supervision_ui",
            "sampling_cadence_out_of_tolerance",
        ],
    }
