import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as operation


def test_wrong_interpreter_is_rejected_before_consumption(monkeypatch):
    monkeypatch.setattr(operation.sys, "executable", "/usr/local/bin/python3")
    with pytest.raises(RuntimeError, match="Exact finsearch-arm interpreter required"):
        operation.interpreter_identity()


def test_dependency_preflight_uses_exact_interpreter_and_records_identity(monkeypatch):
    monkeypatch.setattr(operation.sys, "executable", str(operation.INTERPRETER))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "executable": str(operation.INTERPRETER.resolve()),
                "python_version": "3.11.14",
                "prefix": str(operation.INTERPRETER.parent.parent),
                "base_prefix": str(operation.INTERPRETER.parent.parent),
                "virtual_env": None,
                "conda_prefix": str(operation.INTERPRETER.parent.parent),
                "path": "validated-path",
                "modules": list(operation.REQUIRED_MODULES),
                "packages": {name: "test" for name in operation.REQUIRED_PACKAGES},
            }) + "\n",
            stderr="",
        )

    monkeypatch.setattr(operation.subprocess, "run", run)
    result = operation.dependency_preflight()
    assert calls[0][0][:2] == [str(operation.INTERPRETER), "-c"]
    assert "PYTHONPATH" in calls[0][1]["env"]
    assert result["approved_interpreter"]["executable"] == str(operation.INTERPRETER.resolve())
    assert result["required_modules"] == list(operation.REQUIRED_MODULES)


def test_missing_dependency_fails_before_marker_write(tmp_path, monkeypatch):
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(operation, "MARKER", marker)
    operation._PREFLIGHT_RESULT = None
    writes = []
    monkeypatch.setattr(operation, "_BASE_WRITE_ONCE", lambda *args: writes.append(args))

    def fail():
        raise RuntimeError("Exact-interpreter dependency preflight failed: requests")

    monkeypatch.setattr(operation, "dependency_preflight", fail)
    with pytest.raises(RuntimeError, match="requests"):
        operation._write_once(marker, {"status": "consumed"})
    assert writes == []
    assert not marker.exists()


def test_preflight_is_attached_before_marker_and_outcome(tmp_path, monkeypatch):
    marker = tmp_path / "marker.json"
    outcome = tmp_path / "outcome.json"
    monkeypatch.setattr(operation, "MARKER", marker)
    monkeypatch.setattr(operation, "OUTCOME", outcome)
    records = []
    preflight = {"executable": str(operation.INTERPRETER), "packages": {"requests": "2.32.5"}}
    monkeypatch.setattr(operation, "dependency_preflight", lambda: preflight)

    def write(path, record):
        records.append((Path(path), dict(record)))

    monkeypatch.setattr(operation, "_BASE_WRITE_ONCE", write)
    operation._PREFLIGHT_RESULT = None
    marker_record = {"status": "consumed"}
    operation._write_once(marker, marker_record)
    outcome_record = {"stage": "finished"}
    operation._write_once(outcome, outcome_record)

    assert records[0][0] == marker
    assert records[0][1]["interpreter_preflight"] == preflight
    assert records[1][0] == outcome
    assert records[1][1]["interpreter_preflight"] == preflight
