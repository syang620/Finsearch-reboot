import json
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as helper
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v6 as operation


def _valid_environment(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "FinSearch tests (tests@example.org)")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-reranker-credential")
    monkeypatch.delenv("QWEN3_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("SEC_METRIC_FIXTURE_ROOT", raising=False)


def _base_child_argv(base, index_manifest, env_file=None):
    return base.build_child_argv(
        index_manifest,
        env_file,
        interpreter=base.sys.executable,
        launcher=base.LAUNCHER,
        quality_approval=base.QUALITY,
        integration_approval=base.AUTH,
        staging_root=base.STAGING,
    )


def test_missing_sec_user_agent_fails_before_marker(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-reranker-credential")
    monkeypatch.delenv("SEC_METRIC_FIXTURE_ROOT", raising=False)
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(helper, "MARKER", marker)
    monkeypatch.setattr(helper, "dependency_preflight", operation.dependency_preflight)
    monkeypatch.setattr(helper, "_BASE_WRITE_ONCE", lambda *args: pytest.fail("marker write"))
    operation._ENV_FILE = None

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT is required"):
        helper._write_once(marker, {"status": "consumed"})
    assert not marker.exists()


def test_missing_reranker_credential_fails_before_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("SEC_USER_AGENT", "FinSearch tests (tests@example.org)")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN3_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("SEC_METRIC_FIXTURE_ROOT", raising=False)
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(helper, "MARKER", marker)
    monkeypatch.setattr(helper, "dependency_preflight", operation.dependency_preflight)
    monkeypatch.setattr(helper, "_BASE_WRITE_ONCE", lambda *args: pytest.fail("marker write"))
    operation._ENV_FILE = None

    with pytest.raises(RuntimeError, match="reranker credential"):
        helper._write_once(marker, {"status": "consumed"})
    assert not marker.exists()


def test_fixture_environment_is_rejected_before_marker(monkeypatch, tmp_path):
    _valid_environment(monkeypatch)
    monkeypatch.setenv("SEC_METRIC_FIXTURE_ROOT", str(tmp_path))
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(helper, "MARKER", marker)
    monkeypatch.setattr(helper, "dependency_preflight", operation.dependency_preflight)
    monkeypatch.setattr(helper, "_BASE_WRITE_ONCE", lambda *args: pytest.fail("marker write"))
    operation._ENV_FILE = None

    with pytest.raises(RuntimeError, match="SEC_METRIC_FIXTURE_ROOT"):
        helper._write_once(marker, {"status": "consumed"})
    assert not marker.exists()


def test_valid_explicit_env_file_passes_without_recording_values(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN3_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("SEC_METRIC_FIXTURE_ROOT", raising=False)
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "SEC_USER_AGENT=FinSearch tests (tests@example.org)\n"
        "QWEN3_RERANK_API_KEY=test-reranker-credential\n"
    )

    result = operation.runtime_environment_preflight(env_file)

    assert result["env_file_supplied"] is True
    assert result["sec_user_agent_present"] is True
    assert result["reranker_credential_sources"] == ["QWEN3_RERANK_API_KEY"]
    assert "tests@example.org" not in str(result)
    assert "test-reranker-credential" not in str(result)


def test_missing_env_file_fails_non_consumingly(monkeypatch, tmp_path):
    _valid_environment(monkeypatch)
    missing = tmp_path / "missing.env"
    with pytest.raises(RuntimeError, match="missing or unreadable"):
        operation.runtime_environment_preflight(missing)
    assert not missing.exists()


def test_preflight_preserves_inherited_pythonpath(monkeypatch):
    _valid_environment(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", "/shadow/site-packages")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"executable": str(operation.INTERPRETER)}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(operation.subprocess, "run", run)
    result = operation.runtime_environment_preflight()

    assert calls[0][1]["env"]["PYTHONPATH"] == "/shadow/site-packages"
    assert result["env_file_supplied"] is False


def test_import_preflight_uses_the_frozen_mapping_and_launcher_script_directory(monkeypatch):
    monkeypatch.setattr(helper.sys, "executable", str(helper.INTERPRETER))
    frozen = operation.MappingProxyType({"PYTHONPATH": "/shadow/site-packages"})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"executable": str(helper.INTERPRETER)}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(helper.subprocess, "run", run)
    helper.dependency_preflight(preflight_env=frozen)

    assert calls[0][1]["env"] is frozen
    assert calls[0][1]["env"]["PYTHONPATH"] == "/shadow/site-packages"
    assert f"sys.path[0] = {str(helper.LAUNCHER.resolve().parent)!r}" in calls[0][0][2]


def test_run_passes_frozen_environment_and_does_not_repass_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "SEC_USER_AGENT=FinSearch tests (tests@example.org)\n"
        "DASHSCOPE_API_KEY=test-reranker-credential\n"
    )
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN3_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("SEC_METRIC_FIXTURE_ROOT", raising=False)
    captured = {}
    monkeypatch.setattr(operation, "_configure_helper", lambda: None)
    monkeypatch.setattr(operation, "dependency_preflight", lambda: {"ok": True})

    def base_run(index_manifest, env_file, child_env):
        captured.update(index_manifest=index_manifest, env_file=env_file, child_env=child_env)
        return 0

    monkeypatch.setattr(operation.helper.base, "run", base_run)
    assert operation.run("index.json", env_file) == 0
    assert captured["env_file"] is None
    assert captured["child_env"]["SEC_USER_AGENT"] == "FinSearch tests (tests@example.org)"
    assert captured["child_env"]["DASHSCOPE_API_KEY"] == "test-reranker-credential"


def test_effective_environment_is_unchanged_after_env_file_mutation(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN3_RERANK_API_KEY", raising=False)
    env_file = tmp_path / "local.env"
    env_file.write_text("SEC_USER_AGENT=original@example.org\n")

    effective = operation._effective_child_environment(env_file)
    env_file.write_text("SEC_USER_AGENT=changed@example.org\nSEC_METRIC_FIXTURE_ROOT=/tmp\n")

    assert effective["SEC_USER_AGENT"] == "original@example.org"
    assert "SEC_METRIC_FIXTURE_ROOT" not in effective
    with pytest.raises(TypeError):
        effective["SEC_USER_AGENT"] = "mutated@example.org"


def test_environment_contract_preserves_inherited_values_and_redacts_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(operation.os, "environ", {
        "INHERITED": "stable",
        "SEC_USER_AGENT": "inherited-agent@example.org",
    })
    env_file = tmp_path / "local.env"
    env_file.write_text(
        "SEC_USER_AGENT=file-agent@example.org\n"
        "DASHSCOPE_API_KEY=file-secret\n"
        "VISIBLE=from-file\n"
    )

    effective, contract = operation._freeze_effective_child_environment(env_file)

    assert effective["SEC_USER_AGENT"] == "inherited-agent@example.org"
    assert effective["DASHSCOPE_API_KEY"] == "file-secret"
    assert contract["required_key_sources"] == {
        "SEC_USER_AGENT": "inherited",
        "DASHSCOPE_API_KEY": "env_file",
        "QWEN3_RERANK_API_KEY": "absent",
        "SEC_METRIC_FIXTURE_ROOT": "absent",
    }
    entries = [
        {"name": "DASHSCOPE_API_KEY", "source": "env_file"},
        {"name": "INHERITED", "source": "inherited", "value_sha256": hashlib.sha256(b"stable").hexdigest()},
        {"name": "QWEN3_RERANK_API_KEY", "source": "absent"},
        {"name": "SEC_METRIC_FIXTURE_ROOT", "source": "absent"},
        {"name": "SEC_USER_AGENT", "source": "inherited"},
        {"name": "VISIBLE", "source": "env_file", "value_sha256": hashlib.sha256(b"from-file").hexdigest()},
    ]
    fingerprint_input = {
        "version": operation.EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION,
        "env_file_supplied": True,
        "entries": entries,
        "required_key_sources": contract["required_key_sources"],
    }
    assert contract["fingerprint_sha256"] == hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rendered = json.dumps(contract)
    assert "file-secret" not in rendered
    assert "inherited-agent@example.org" not in rendered


def test_runtime_preflight_uses_the_same_frozen_mapping_and_explicit_source_flag(monkeypatch):
    frozen = {"SEC_USER_AGENT": "agent", "DASHSCOPE_API_KEY": "credential"}
    contract = operation._environment_contract(frozen, {}, True)
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"executable": str(operation.INTERPRETER)}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(operation.subprocess, "run", run)
    result = operation.runtime_environment_preflight(
        child_env=frozen, environment_contract=contract, env_file_supplied=True
    )

    assert calls[0]["env"] is frozen
    assert result["env_file_supplied"] is True
    assert result["effective_environment"] == contract


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (subprocess.TimeoutExpired(["python"], 1), "timed out"),
        (OSError("boom"), "could not start"),
    ],
)
def test_runtime_preflight_fails_closed_on_timeout_or_crash(monkeypatch, failure, message):
    _valid_environment(monkeypatch)

    def run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(operation.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=message):
        operation.runtime_environment_preflight()


def test_runtime_preflight_rejects_nonzero_malformed_and_unexpected_interpreter(monkeypatch):
    _valid_environment(monkeypatch)
    responses = [
        SimpleNamespace(returncode=2, stdout="secret-value", stderr="secret-value"),
        SimpleNamespace(returncode=0, stdout="not json", stderr=""),
        SimpleNamespace(returncode=0, stdout=json.dumps({"executable": "/wrong/python"}), stderr=""),
    ]
    monkeypatch.setattr(operation.subprocess, "run", lambda *_args, **_kwargs: responses.pop(0))

    with pytest.raises(RuntimeError, match="runtime environment contract rejected") as error:
        operation.runtime_environment_preflight()
    assert "secret-value" not in str(error.value)
    with pytest.raises(RuntimeError, match="invalid metadata"):
        operation.runtime_environment_preflight()
    with pytest.raises(RuntimeError, match="expected"):
        operation.runtime_environment_preflight()


def test_reviewed_blob_check_rejects_mismatch_symlink_and_untracked_path(monkeypatch, tmp_path):
    path = tmp_path / "reviewed.py"
    path.write_text("current\n")
    monkeypatch.setattr(operation.subprocess, "check_output", lambda *_args, **_kwargs: b"reviewed\n")
    with pytest.raises(ValueError, match="differs from its reviewed Git blob"):
        operation._verify_reviewed_regular_tracked_blob("a" * 40, path, "test path")

    link = tmp_path / "reviewed-link.py"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular file, not a symlink"):
        operation._verify_reviewed_regular_tracked_blob("a" * 40, link, "test path")

    def untracked(command, **_kwargs):
        if command[1] == "ls-files":
            raise subprocess.CalledProcessError(1, command)
        return b"current\n"

    monkeypatch.setattr(operation.subprocess, "check_output", untracked)
    with pytest.raises(ValueError, match="not a tracked reviewed path"):
        operation._verify_reviewed_regular_tracked_blob("a" * 40, path, "test path")


def test_child_argv_is_centralized_without_an_env_file_argument():
    argv = operation.build_child_argv("index.json")
    assert argv == _base_child_argv(operation.helper.base, "index.json")
    assert "--env-file" not in argv
    metadata = operation._safe_child_argv_metadata(argv)
    assert metadata["contains_env_file_argument"] is False
    assert "index.json" not in json.dumps(metadata)


def test_actual_child_popen_uses_the_centralized_argv_cwd_and_frozen_environment(
    monkeypatch, tmp_path
):
    base = operation.helper.base
    for name in ("AUTH", "WRAPPER", "LAUNCHER", "QUALITY"):
        path = tmp_path / name.lower()
        path.write_text("{}")
        monkeypatch.setattr(base, name, path)
    for name in ("MARKER", "OUTCOME", "LOG", "STAGING"):
        monkeypatch.setattr(base, name, tmp_path / name.lower())
    monkeypatch.setattr(base, "registration", lambda: ({}, "a" * 40))
    frozen = operation.MappingProxyType({"SEC_USER_AGENT": "agent"})
    captured = {}

    class Child:
        pid = 123

        def wait(self):
            return 0

    def popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return Child()

    monkeypatch.setattr(base.subprocess, "Popen", popen)
    assert base.run("index.json", None, frozen) == 0
    assert captured["argv"] == _base_child_argv(base, "index.json")
    assert captured["kwargs"]["env"] is frozen
    assert captured["kwargs"]["cwd"] == Path(base.__file__).resolve().parents[2]
