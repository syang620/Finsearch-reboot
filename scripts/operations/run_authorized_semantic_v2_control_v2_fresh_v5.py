"""Fresh-v5 wrapper: all remote attestations pass before marker consumption."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.evals.retrieval import run_benchmark_v3 as provenance
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as helper


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_fresh_v5_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v5.py")
DEPENDENCY = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v4.py")
QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
MARKER = Path(".cache/semantic_v2_control_v2_fresh_v5_20260909.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_fresh_v5_20260909.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_fresh_v5_20260909.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_fresh_v5_20260909")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-FRESH-V5-20260909"
REVIEW_AUTHOR = "chatgpt-codex-connector[bot]"
REVIEW_REPOSITORY = "syang620/Finsearch-reboot"
_BASE_DEPENDENCY_PREFLIGHT = helper.dependency_preflight


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _remote_attestation(approval, label):
    pr = approval.get("pull_request")
    comment_id = approval.get("review_comment_id")
    reviewed = approval.get("reviewed_commit")
    if type(pr) is not int or pr <= 0 or type(comment_id) is not int or comment_id <= 0:
        raise ValueError(f"{label} attestation requires numeric PR/comment identities")
    if not isinstance(reviewed, str) or not re.fullmatch(r"[0-9a-f]{40}", reviewed):
        raise ValueError(f"{label} attestation requires an exact reviewed commit")
    prefix = f"repos/{REVIEW_REPOSITORY}"
    comment = provenance.github_json(f"{prefix}/issues/comments/{comment_id}")
    expected_url = f"https://github.com/{REVIEW_REPOSITORY}/pull/{pr}#issuecomment-{comment_id}"
    if (
        comment.get("id") != comment_id
        or comment.get("html_url") != expected_url
        or comment.get("issue_url") != f"https://api.github.com/{prefix}/issues/{pr}"
        or approval.get("review_url") != expected_url
    ):
        raise ValueError(f"{label} review URL/PR/comment provenance mismatch")
    author = comment.get("user") or {}
    if author.get("login") != REVIEW_AUTHOR or author.get("type") != "Bot":
        raise ValueError(f"{label} review is not from the expected Codex reviewer")
    body = comment.get("body", "")
    match = re.search(r"\*\*Reviewed commit:\*\*\s*`([0-9a-f]{10,40})`", body)
    if (
        not match
        or not reviewed.startswith(match.group(1))
        or not body.strip().startswith("Codex Review: Didn't find any major issues.")
    ):
        raise ValueError(f"{label} is not a clean review of the attested commit")
    pull = provenance.github_json(f"{prefix}/pulls/{pr}")
    if pull.get("base", {}).get("repo", {}).get("full_name") != REVIEW_REPOSITORY or pull.get("state") != "open":
        raise ValueError(f"{label} review does not belong to the open benchmark PR")
    reviews = provenance.github_json(f"{prefix}/pulls/{pr}/reviews", paginate=True)
    if any(r.get("commit_id") == reviewed and r.get("state") == "CHANGES_REQUESTED" for r in reviews):
        raise ValueError(f"{label} attested commit has requested changes")
    findings = provenance.github_json(f"{prefix}/pulls/{pr}/comments", paginate=True)
    if any(
        (r.get("user") or {}).get("login") == REVIEW_AUTHOR
        and r.get("original_commit_id", r.get("commit_id")) == reviewed
        for r in findings
    ):
        raise ValueError(f"{label} attested commit has inline Codex findings")
    actual_hash = hashlib.sha256(body.encode()).hexdigest()
    return {
        "label": label,
        "pull_request": pr,
        "comment_id": comment_id,
        "review_url": expected_url,
        "reviewer": author.get("login"),
        "reviewed_commit": reviewed,
        "attested_body_sha256": approval.get("review_body_sha256"),
        "remote_body_sha256": actual_hash,
        "body_hash_matches_attestation": actual_hash == approval.get("review_body_sha256"),
        "body_hash_policy": "semantic clean/head identity is authoritative; hash drift is preserved as diagnostic metadata",
    }


def remote_review_preflight():
    approval = json.loads(AUTH.read_text())
    quality = json.loads(QUALITY.read_text())
    return {
        "integration_review": _remote_attestation(approval, "integration"),
        "quality_review": _remote_attestation(quality, "quality"),
    }


def dependency_preflight():
    result = _BASE_DEPENDENCY_PREFLIGHT()
    result["remote_reviews"] = remote_review_preflight()
    return result


def registration():
    helper.interpreter_identity()
    if helper.base.git("status", "--porcelain"):
        raise ValueError("Clean committed authorization required")
    if AUTH.read_bytes() != subprocess.check_output(["git", "show", f"HEAD:{AUTH}"]):
        raise ValueError("Authorization must match committed bytes")
    approval = json.loads(AUTH.read_text())
    reviewed = approval.get("reviewed_commit")
    if (
        approval.get("authorization_id") != AUTHORIZATION_ID
        or approval.get("status") != "approved_for_one_semantic_v2_control_v2_attempt"
        or approval.get("max_new_attempts") != 1
        or approval.get("selected_policy") != "B_CONSECUTIVE_10"
        or approval.get("review_status") != "completed_clean"
        or approval.get("operation_wrapper") != str(WRAPPER)
        or approval.get("operation_wrapper_sha256") != sha(WRAPPER)
        or approval.get("wrapper_dependency") != str(DEPENDENCY)
        or approval.get("wrapper_dependency_sha256") != sha(DEPENDENCY)
        or approval.get("quality_approval_sha256") != sha(QUALITY)
        or approval.get("interpreter_path") != str(helper.INTERPRETER)
        or approval.get("interpreter_sha256") != sha(helper.INTERPRETER)
        or approval.get("consumption_marker") != str(MARKER)
        or approval.get("launch_outcome") != str(OUTCOME)
        or approval.get("local_console_log") != str(LOG)
        or approval.get("staging_output_root") != str(STAGING)
    ):
        raise ValueError("Registered fresh-v5 control-v2 authorization changed")
    helper.base.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    if helper.base.git("diff", reviewed, "--", str(DEPENDENCY)):
        raise ValueError("Fresh-v4 dependency changed after fresh-v5 review")
    return approval, helper.base.git("rev-parse", "HEAD")


def _configure_helper():
    for name, value in {
        "AUTH": AUTH,
        "WRAPPER": WRAPPER,
        "DEPENDENCY": DEPENDENCY,
        "MARKER": MARKER,
        "OUTCOME": OUTCOME,
        "LOG": LOG,
        "STAGING": STAGING,
        "QUALITY": QUALITY,
        "AUTHORIZATION_ID": AUTHORIZATION_ID,
    }.items():
        setattr(helper, name, value)
    helper.registration = registration
    helper.dependency_preflight = dependency_preflight
    helper._configure_base()


def run(index_manifest, env_file=None):
    _configure_helper()
    return helper.base.run(index_manifest, env_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.index_manifest, arguments.env_file))
