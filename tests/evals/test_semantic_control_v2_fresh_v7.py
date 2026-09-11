import hashlib
from pathlib import Path

import pytest

from scripts.evals.agents import run_semantic_baseline_v2_2 as control
from scripts.evals.agents import run_semantic_baseline_v2_3 as launcher
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v7 as operation
from scripts.operations import semantic_v7_environment as environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_fresh_v6_wrapper_and_approval_match_retirement_record():
    retirement = launcher.json.loads(launcher.RETIREMENT.read_text())

    assert retirement["status"] == "retired_currently_unexecutable"
    assert retirement["consumed"] is False
    assert sha(retirement["wrapper_path"]) == retirement["wrapper_sha256"]
    assert sha(retirement["approval_path"]) == retirement["approval_sha256"]
    assert "Never repurpose" in retirement["immutability_rule"]


def test_fresh_v7_candidate_has_no_approval_or_execution_authority():
    contract = launcher.canonical.load_contract()

    assert contract["status"] == "inactive_v7_candidate"
    assert contract["authority"] == "none"
    assert not operation.AUTH.exists()
    assert "fresh_v6" not in operation.DEPENDENCY.name
    assert (
        "import run_authorized_semantic_v2_control_v2_fresh_v6"
        not in Path(operation.__file__).read_text()
    )


def test_v7_environment_preserves_inherited_precedence_and_redacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        environment.os,
        "environ",
        {
            "SEC_USER_AGENT": "inherited@example.org",
            "PYTHONPATH": "/exact/path",
        },
    )
    env_file = tmp_path / "runtime.env"
    env_file.write_text(
        "SEC_USER_AGENT=file@example.org\nDASHSCOPE_API_KEY=secret-value\n"
    )

    effective, contract = environment.freeze_effective_child_environment(env_file)

    assert effective["SEC_USER_AGENT"] == "inherited@example.org"
    assert effective["DASHSCOPE_API_KEY"] == "secret-value"
    assert contract["required_key_sources"]["SEC_USER_AGENT"] == "inherited"
    assert contract["required_key_sources"]["DASHSCOPE_API_KEY"] == "env_file"
    assert "inherited@example.org" not in str(contract)
    assert "secret-value" not in str(contract)


def test_unclassified_values_cannot_enter_v7_environment_fingerprint():
    first = environment.environment_contract(
        {"DATABASE_URL": "first-secret"}, {"DATABASE_URL": "first-secret"}, False
    )
    second = environment.environment_contract(
        {"DATABASE_URL": "second-secret"}, {"DATABASE_URL": "second-secret"}, False
    )

    assert first["fingerprint_sha256"] == second["fingerprint_sha256"]
    assert "first-secret" not in str(first)
    assert "second-secret" not in str(second)


def test_registration_fails_inactive_before_any_marker(monkeypatch, tmp_path):
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(operation, "AUTH", tmp_path / "missing-approval.json")
    monkeypatch.setattr(operation, "MARKER", marker)
    monkeypatch.setattr(operation.helper, "interpreter_identity", lambda: {})

    with pytest.raises(RuntimeError, match="candidate is inactive"):
        operation.registration()
    assert not marker.exists()


def test_inactive_run_stops_before_qdrant_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(operation, "AUTH", tmp_path / "missing-approval.json")
    monkeypatch.setattr(operation.helper, "interpreter_identity", lambda: {})
    monkeypatch.setattr(
        operation.environment,
        "freeze_effective_child_environment",
        lambda env_file: ({}, {}),
    )
    monkeypatch.setattr(operation, "_configure_helper", lambda: None)
    monkeypatch.setattr(
        operation,
        "canonical_qdrant_preflight",
        lambda: pytest.fail("inactive candidate queried Qdrant"),
    )

    with pytest.raises(RuntimeError, match="candidate is inactive"):
        operation.run()


def test_child_argv_is_exact_and_contains_no_env_file():
    argv = operation.build_child_argv()

    assert argv == [
        str(operation.INTERPRETER),
        "-u",
        str(operation.LAUNCHER),
        "--approval",
        str(operation.QUALITY),
        "--integration-approval",
        str(operation.AUTH),
        "--workload-control-v2",
        "B_CONSECUTIVE_10",
        "--out-root",
        str(operation.STAGING),
        "--index-attestation",
        str(operation.INDEX_ATTESTATION),
    ]
    assert "--env-file" not in argv


def test_successful_after_guard_records_exact_identity():
    snapshot = {
        "collection": "expected",
        "points": 948,
        "payload_vectors_sha256": "f" * 64,
        "config": {"exact": True},
    }

    class Guard:
        def verify_after(self):
            return snapshot, []

    ending = {}
    launcher.close_index_guard(ending, Guard(), snapshot)

    assert ending["canonical_index_after_verified"] is True
    assert ending["index_unchanged"] is True
    assert ending["index_after"] == snapshot


def test_after_guard_failure_withholds_official_eligibility():
    class MutatedGuard:
        def verify_after(self):
            raise ValueError("fingerprint changed")

    ending = {
        "captured_cases": ["case"],
        "evaluation_errors": [],
        "control_violations": [],
        "model_identities_unchanged": True,
    }
    launcher.close_index_guard(
        ending,
        MutatedGuard(),
        {"before": True},
    )
    valid = control.provisional_validity(ending, ["case"], ["case"])

    assert valid is False
    assert ending["canonical_index_after_verified"] is False
    assert ending["index_unchanged"] is False
    assert ending["official_baseline_eligible"] is False
    assert "index_unchanged_not_verified" in ending["invalidity_reasons"]
    assert "index_verification_error" in ending["invalidity_reasons"]
