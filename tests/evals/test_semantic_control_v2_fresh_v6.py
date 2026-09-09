import pytest

from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as helper
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v6 as operation


def _valid_environment(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "FinSearch tests (tests@example.org)")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-reranker-credential")
    monkeypatch.delenv("QWEN3_RERANK_API_KEY", raising=False)
    monkeypatch.delenv("SEC_METRIC_FIXTURE_ROOT", raising=False)


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
