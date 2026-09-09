import json
from pathlib import Path
import signal
import sys

import pytest

from scripts.operations import run_authorized_semantic_v2_control_v2 as operation
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh as fresh_operation


def setup_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("AUTH", "WRAPPER", "LAUNCHER"):
        path = tmp_path / name.lower()
        path.write_text("{}")
        monkeypatch.setattr(operation, name, path)
    for name in ("MARKER", "OUTCOME", "LOG", "STAGING", "QUALITY"):
        monkeypatch.setattr(operation, name, tmp_path / name.lower())
    monkeypatch.setattr(operation, "registration", lambda: ({}, "a" * 40))


def test_registration_is_committed_and_hash_bound(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper = tmp_path / "wrapper.py"
    launcher = tmp_path / "launcher.py"
    approval_path = tmp_path / "approval.json"
    wrapper.write_text("wrapper")
    launcher.write_text("launcher")
    for name, value in (("WRAPPER", wrapper), ("LAUNCHER", launcher), ("AUTH", approval_path)):
        monkeypatch.setattr(operation, name, value)
    monkeypatch.setattr(operation, "MARKER", tmp_path / "marker")
    monkeypatch.setattr(operation, "STAGING", tmp_path / "staging")
    approval = {
        "authorization_id": operation.AUTHORIZATION_ID,
        "status": "approved_for_one_semantic_v2_control_v2_attempt",
        "max_new_attempts": 1,
        "selected_policy": "B_CONSECUTIVE_10",
        "operation_wrapper": str(wrapper),
        "operation_wrapper_sha256": operation.sha(wrapper),
        "integration_launcher_sha256": operation.sha(launcher),
        "consumption_marker": str(operation.MARKER),
        "staging_output_root": str(operation.STAGING),
    }
    approval_path.write_text(json.dumps(approval))
    monkeypatch.setattr(operation, "git", lambda *args: "b" * 40 if args[0] == "rev-parse" else "")
    monkeypatch.setattr(operation.subprocess, "check_output", lambda command: approval_path.read_bytes())
    assert operation.registration() == (approval, "b" * 40)
    approval["max_new_attempts"] = 2
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="authorization changed"):
        operation.registration()


def test_fresh_registration_is_distinct_and_source_review_bound():
    approval = json.loads(fresh_operation.AUTH.read_text())
    assert approval["authorization_id"] == fresh_operation.AUTHORIZATION_ID
    assert approval["reviewed_commit"] == fresh_operation.REVIEWED_SOURCE_HEAD
    assert approval["operation_wrapper"] == str(fresh_operation.WRAPPER)
    assert approval["operation_wrapper_sha256"] == fresh_operation.sha(fresh_operation.WRAPPER)
    assert approval["integration_launcher_sha256"] == fresh_operation.sha(fresh_operation.LAUNCHER)
    for name in ("AUTH", "WRAPPER", "MARKER", "OUTCOME", "LOG", "STAGING"):
        assert getattr(fresh_operation, name) != getattr(operation, name)


def test_marker_is_exclusive_and_durable(tmp_path, monkeypatch):
    synced = []
    original = operation.os.fsync
    monkeypatch.setattr(operation.os, "fsync", lambda descriptor: (synced.append(descriptor), original(descriptor)))
    marker = tmp_path / "marker.json"
    operation.write_once(marker, {"status": "consumed"})
    before = marker.read_bytes()
    assert len(synced) == 2
    with pytest.raises(FileExistsError):
        operation.write_once(marker, {"status": "reset"})
    assert marker.read_bytes() == before


@pytest.mark.parametrize("code", [0, 1, 130])
def test_child_result_never_resets_permission(tmp_path, monkeypatch, code):
    setup_operation(tmp_path, monkeypatch)
    calls = []

    class Child:
        def wait(self):
            return code

    def spawn(command, **kwargs):
        assert operation.MARKER.exists()
        calls.append(command)
        return Child()

    monkeypatch.setattr(operation.subprocess, "Popen", spawn)
    assert operation.run(Path("index.json"), Path("local.env")) == code
    expected = [
        sys.executable, "-u", str(operation.LAUNCHER),
        "--approval", str(operation.QUALITY),
        "--integration-approval", str(operation.AUTH),
        "--workload-control-v2", "B_CONSECUTIVE_10",
        "--out-root", str(operation.STAGING),
        "--index-manifest", "index.json",
        "--env-file", "local.env",
    ]
    assert calls == [expected]
    marker = operation.MARKER.read_bytes()
    with pytest.raises(FileExistsError):
        operation.run(Path("index.json"))
    assert operation.MARKER.read_bytes() == marker


def test_launch_failure_remains_consumed(tmp_path, monkeypatch):
    setup_operation(tmp_path, monkeypatch)
    monkeypatch.setattr(operation.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn")))
    with pytest.raises(OSError):
        operation.run(Path("index.json"))
    assert operation.MARKER.exists()
    assert json.loads(operation.OUTCOME.read_text())["error_type"] == "OSError"


def test_interrupt_is_forwarded_and_outcome_preserved(tmp_path, monkeypatch):
    setup_operation(tmp_path, monkeypatch)
    signals = []

    class Child:
        calls = 0

        def wait(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            return 130

        def poll(self):
            return None

        def send_signal(self, value):
            signals.append(value)

    monkeypatch.setattr(operation.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(KeyboardInterrupt):
        operation.run(Path("index.json"))
    assert signals == [signal.SIGINT]
    assert json.loads(operation.OUTCOME.read_text())["child_returncode"] == 130
