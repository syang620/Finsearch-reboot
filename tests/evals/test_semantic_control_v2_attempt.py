import json
from pathlib import Path
import signal
import sys

import pytest

from scripts.operations import run_authorized_semantic_v2_control_v2 as operation
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh as fresh_operation
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v2 as sigterm_operation


def setup_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("AUTH", "WRAPPER", "LAUNCHER"):
        path = tmp_path / name.lower()
        path.write_text("{}")
        monkeypatch.setattr(operation, name, path)
    for name in ("MARKER", "OUTCOME", "LOG", "STAGING", "QUALITY"):
        monkeypatch.setattr(operation, name, tmp_path / name.lower())
    monkeypatch.setattr(operation, "registration", lambda: ({}, "a" * 40))


def setup_sigterm_operation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("AUTH", "WRAPPER", "LAUNCHER"):
        path = tmp_path / name.lower()
        path.write_text("{}")
        monkeypatch.setattr(sigterm_operation, name, path)
    for name in ("MARKER", "OUTCOME", "LOG", "STAGING", "QUALITY"):
        monkeypatch.setattr(sigterm_operation, name, tmp_path / name.lower())
    monkeypatch.setattr(sigterm_operation, "registration", lambda: ({}, "c" * 40))


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


def test_historical_fresh_registration_is_distinct_and_now_stale():
    approval = json.loads(fresh_operation.AUTH.read_text())
    assert approval["authorization_id"] == fresh_operation.AUTHORIZATION_ID
    assert approval["reviewed_commit"] == fresh_operation.REVIEWED_SOURCE_HEAD
    assert approval["operation_wrapper"] == str(fresh_operation.WRAPPER)
    assert approval["operation_wrapper_sha256"] == fresh_operation.sha(fresh_operation.WRAPPER)
    assert approval["integration_launcher_sha256"] != fresh_operation.sha(fresh_operation.LAUNCHER)
    for name in ("AUTH", "WRAPPER", "MARKER", "OUTCOME", "LOG", "STAGING"):
        assert getattr(fresh_operation, name) != getattr(operation, name)


def test_existing_fresh_v2_approval_rejects_launcher_hash_before_consumption(monkeypatch):
    approval_path = sigterm_operation.AUTH

    def git(*args):
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return "c" * 40
        return ""

    monkeypatch.setattr(sigterm_operation, "git", git)
    monkeypatch.setattr(
        sigterm_operation.subprocess,
        "check_output",
        lambda command: approval_path.read_bytes(),
    )
    with pytest.raises(ValueError, match="Registered fresh-v2 control-v2 authorization changed"):
        sigterm_operation.registration()
    for path in (sigterm_operation.MARKER, sigterm_operation.OUTCOME, sigterm_operation.LOG):
        assert not path.exists()
    assert not sigterm_operation.STAGING.exists()


def test_sigterm_registration_paths_are_new_and_inactive():
    for name in ("AUTH", "WRAPPER", "MARKER", "OUTCOME", "LOG", "STAGING"):
        assert getattr(sigterm_operation, name) not in {
            getattr(operation, name), getattr(fresh_operation, name),
        }
    assert sigterm_operation.AUTHORIZATION_ID not in {
        operation.AUTHORIZATION_ID, fresh_operation.AUTHORIZATION_ID,
    }


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


def install_sigterm_handler(monkeypatch):
    installed = []
    previous = object()

    def install(signum, handler):
        installed.append((signum, handler))
        return previous

    monkeypatch.setattr(sigterm_operation.signal, "signal", install)
    return installed, previous


def test_sigterm_is_forwarded_once_and_outcome_separates_exit_codes(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    installed, previous = install_sigterm_handler(monkeypatch)
    forwarded = []
    monkeypatch.setattr(sigterm_operation.os, "kill", lambda pid, signum: forwarded.append((pid, signum)))

    class Child:
        pid = 42

        def wait(self):
            handler = installed[0][1]
            handler(signal.SIGTERM, None)
            handler(signal.SIGTERM, None)
            return -signal.SIGTERM

    monkeypatch.setattr(sigterm_operation.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert sigterm_operation.run(Path("index.json")) == 143
    outcome = json.loads(sigterm_operation.OUTCOME.read_text())
    assert forwarded == [(42, signal.SIGTERM)]
    assert outcome["child_returncode"] == -signal.SIGTERM
    assert outcome["wrapper_exit_code"] == 143
    assert outcome["error_type"] == "SIGTERM"
    assert outcome["termination"]["received"]
    assert outcome["termination"]["received_at"]
    assert outcome["termination"]["forwarded_to_child"]
    assert outcome["termination"]["forwarded_at"]
    assert installed[-1] == (signal.SIGTERM, previous)


def test_sigterm_during_popen_assignment_is_rechecked_and_forwarded(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    installed, _ = install_sigterm_handler(monkeypatch)
    forwarded = []
    monkeypatch.setattr(sigterm_operation.os, "kill", lambda pid, signum: forwarded.append((pid, signum)))

    class Child:
        pid = 84

        def wait(self):
            return -signal.SIGTERM

    def spawn(*args, **kwargs):
        installed[0][1](signal.SIGTERM, None)
        return Child()

    monkeypatch.setattr(sigterm_operation.subprocess, "Popen", spawn)
    assert sigterm_operation.run(Path("index.json")) == 143
    outcome = json.loads(sigterm_operation.OUTCOME.read_text())
    assert forwarded == [(84, signal.SIGTERM)]
    assert outcome["termination"]["forwarded_to_child"]


def test_pending_sigterm_is_latched_before_popen(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    install_sigterm_handler(monkeypatch)
    monkeypatch.setattr(sigterm_operation.signal, "sigpending", lambda: {signal.SIGTERM})
    monkeypatch.setattr(
        sigterm_operation.subprocess, "Popen",
        lambda *args, **kwargs: pytest.fail("Pending SIGTERM must be latched before Popen"),
    )
    assert sigterm_operation.run(Path("index.json")) == 143
    outcome = json.loads(sigterm_operation.OUTCOME.read_text())
    assert not outcome["child_started"]
    assert outcome["stage"] == "terminated_prelaunch"
    assert outcome["termination"]["received"]


def test_sigterm_forward_failure_preserves_attempt_timestamp(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    installed, _ = install_sigterm_handler(monkeypatch)

    def fail_forward(pid, signum):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(sigterm_operation.os, "kill", fail_forward)

    class Child:
        pid = 85

        def wait(self):
            installed[0][1](signal.SIGTERM, None)
            return 0

    monkeypatch.setattr(sigterm_operation.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert sigterm_operation.run(Path("index.json")) == 143
    termination = json.loads(sigterm_operation.OUTCOME.read_text())["termination"]
    assert termination["forward_attempted"]
    assert termination["forward_attempted_at"]
    assert not termination["forwarded_to_child"]
    assert termination["forwarded_at"] is None
    assert termination["forward_error_type"] == "ProcessLookupError"


def test_sigterm_after_consumption_skips_child_and_persists_outcome(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    installed, _ = install_sigterm_handler(monkeypatch)
    original_write_once = sigterm_operation.write_once

    def write_once(path, record):
        original_write_once(path, record)
        if Path(path) == sigterm_operation.MARKER:
            installed[0][1](signal.SIGTERM, None)

    monkeypatch.setattr(sigterm_operation, "write_once", write_once)
    monkeypatch.setattr(
        sigterm_operation.subprocess, "Popen",
        lambda *args, **kwargs: pytest.fail("SIGTERM before launch must skip child creation"),
    )
    assert sigterm_operation.run(Path("index.json")) == 143
    outcome = json.loads(sigterm_operation.OUTCOME.read_text())
    assert sigterm_operation.MARKER.exists()
    assert not outcome["child_started"]
    assert outcome["child_returncode"] is None
    assert outcome["wrapper_exit_code"] == 143
    assert outcome["stage"] == "terminated_prelaunch"


def test_sigterm_candidate_preserves_keyboard_interrupt_behavior(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    install_sigterm_handler(monkeypatch)
    forwarded = []

    class Child:
        pid = 126
        calls = 0

        def wait(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            return 130

        def send_signal(self, value):
            forwarded.append(value)

    monkeypatch.setattr(sigterm_operation.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(KeyboardInterrupt):
        sigterm_operation.run(Path("index.json"))
    outcome = json.loads(sigterm_operation.OUTCOME.read_text())
    assert forwarded == [signal.SIGINT]
    assert outcome["error_type"] == "KeyboardInterrupt"
    assert outcome["child_returncode"] == 130
    assert outcome["wrapper_exit_code"] is None


def test_sigterm_handler_is_restored_if_outcome_persistence_fails(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    installed, previous = install_sigterm_handler(monkeypatch)
    original_write_once = sigterm_operation.write_once

    def write_once(path, record):
        if Path(path) == sigterm_operation.OUTCOME:
            raise OSError("outcome write failed")
        original_write_once(path, record)

    class Child:
        pid = 168

        def wait(self):
            return 0

    monkeypatch.setattr(sigterm_operation, "write_once", write_once)
    monkeypatch.setattr(sigterm_operation.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(OSError, match="outcome write failed"):
        sigterm_operation.run(Path("index.json"))
    assert installed[-1] == (signal.SIGTERM, previous)


def test_pending_sigterm_is_latched_at_finalization_cutoff(tmp_path, monkeypatch):
    setup_sigterm_operation(tmp_path, monkeypatch)
    install_sigterm_handler(monkeypatch)
    pending = iter((set(), {signal.SIGTERM}))
    monkeypatch.setattr(sigterm_operation.signal, "sigpending", lambda: next(pending))

    class Child:
        pid = 210

        def wait(self):
            return 0

    monkeypatch.setattr(sigterm_operation.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert sigterm_operation.run(Path("index.json")) == 143
    outcome = json.loads(sigterm_operation.OUTCOME.read_text())
    assert outcome["child_returncode"] == 0
    assert outcome["wrapper_exit_code"] == 143
    assert outcome["termination"]["received"]
    assert outcome["termination"]["observation_closed_at"]
