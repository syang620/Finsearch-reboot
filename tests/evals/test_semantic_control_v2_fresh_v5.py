import hashlib
from pathlib import Path

import pytest

from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as helper
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v5 as operation


def review_body(commit="a" * 40):
    return f"Codex Review: Didn't find any major issues.\n\n**Reviewed commit:** `{commit}`\n"


def fake_remote(monkeypatch, body, commit):
    def github_json(endpoint, paginate=False):
        if "/issues/comments/" in endpoint:
            return {
                "id": 42,
                "html_url": "https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-42",
                "issue_url": "https://api.github.com/repos/syang620/Finsearch-reboot/issues/31",
                "user": {"login": operation.REVIEW_AUTHOR, "type": "Bot"},
                "body": body,
            }
        if endpoint.endswith("/pulls/31"):
            return {"base": {"repo": {"full_name": operation.REVIEW_REPOSITORY}}, "state": "open"}
        return []

    monkeypatch.setattr(operation.provenance, "github_json", github_json)
    return {
        "pull_request": 31,
        "review_comment_id": 42,
        "review_url": "https://github.com/syang620/Finsearch-reboot/pull/31#issuecomment-42",
        "review_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "reviewed_commit": commit,
    }


def test_valid_remote_review_attestation_passes(monkeypatch):
    commit = "a" * 40
    body = review_body(commit)
    approval = fake_remote(monkeypatch, body, commit)
    result = operation._remote_attestation(approval, "integration")
    assert result["reviewed_commit"] == commit
    assert result["body_hash_matches_attestation"] is True


def test_cosmetic_remote_body_drift_is_preserved_not_silently_accepted(monkeypatch):
    commit = "a" * 40
    attested = review_body(commit)
    current = attested + "\n<!-- harmless bot formatting -->\n"
    approval = fake_remote(monkeypatch, current, commit)
    approval["review_body_sha256"] = hashlib.sha256(attested.encode()).hexdigest()
    result = operation._remote_attestation(approval, "integration")
    assert result["body_hash_matches_attestation"] is False
    assert result["remote_body_sha256"] == hashlib.sha256(current.encode()).hexdigest()


def test_unreachable_remote_review_fails_before_marker(tmp_path, monkeypatch):
    marker = tmp_path / "marker.json"
    monkeypatch.setattr(helper, "MARKER", marker)
    monkeypatch.setattr(helper, "_BASE_WRITE_ONCE", lambda *args: pytest.fail("marker write"))

    def fail():
        raise RuntimeError("remote review unavailable")

    monkeypatch.setattr(helper, "dependency_preflight", fail)
    with pytest.raises(RuntimeError, match="remote review unavailable"):
        helper._write_once(marker, {"status": "consumed"})
    assert not marker.exists()


def test_remote_preflight_failure_is_non_consuming(monkeypatch):
    monkeypatch.setattr(operation, "_BASE_DEPENDENCY_PREFLIGHT", lambda: {"interpreter": "ok"})
    monkeypatch.setattr(operation, "remote_review_preflight", lambda: (_ for _ in ()).throw(ValueError("stale review")))
    with pytest.raises(ValueError, match="stale review"):
        operation.dependency_preflight()

