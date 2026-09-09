"""Fresh-v6 wrapper with runtime-environment validation before consumption.

This wrapper is intentionally a new authorization generation.  Fresh-v5 is
permanently exhausted; v6 adds the local runtime environment contract that v5
could not validate before writing its one-use marker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from scripts.evals.retrieval import run_benchmark_v3 as provenance
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as helper


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_fresh_v6_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v6.py")
DEPENDENCY = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v5.py")
INTERPRETER = helper.INTERPRETER
MARKER = Path(".cache/semantic_v2_control_v2_fresh_v6_20260909.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_fresh_v6_20260909.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_fresh_v6_20260909.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_fresh_v6_20260909")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-FRESH-V6-20260909"

QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
REVIEW_AUTHOR = "chatgpt-codex-connector[bot]"
REVIEW_REPOSITORY = "syang620/Finsearch-reboot"
_BASE_DEPENDENCY_PREFLIGHT = helper.dependency_preflight
_ENV_FILE = None
_PREFLIGHT_RESULT = None


def _environment_script(env_file):
    return f"""
import json
import os
from pathlib import Path
import sys

env_file = {repr(str(env_file) if env_file is not None else None)}
if env_file is not None:
    path = Path(env_file)
    if not path.is_file() or not os.access(path, os.R_OK):
        raise RuntimeError("runtime env file is missing or unreadable")
    from dotenv import load_dotenv
    load_dotenv(path, override=False)

if os.getenv("SEC_METRIC_FIXTURE_ROOT"):
    raise RuntimeError("SEC_METRIC_FIXTURE_ROOT is forbidden for the live baseline")
if not os.getenv("SEC_USER_AGENT", "").strip():
    raise RuntimeError("SEC_USER_AGENT is required")

credential_sources = [
    name for name in ("QWEN3_RERANK_API_KEY", "DASHSCOPE_API_KEY")
    if os.getenv(name, "").strip()
]
if not credential_sources:
    raise RuntimeError("an existing reranker credential is required")

print(json.dumps({{
    "executable": sys.executable,
    "env_file_supplied": env_file is not None,
    "sec_user_agent_present": True,
    "reranker_credential_sources": credential_sources,
    "fixture_root_forbidden": True,
}}))
"""


def runtime_environment_preflight(env_file=None):
    """Validate the effective child environment without recording secrets."""
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    project_path = os.pathsep.join((str(repo_root), str(repo_root / "src")))
    if env.get("PYTHONPATH"):
        project_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = project_path
    completed = subprocess.run(
        [str(INTERPRETER), "-c", _environment_script(env_file)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Exact-interpreter runtime environment preflight failed with exit "
            f"{completed.returncode}: {detail}"
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Runtime environment preflight returned invalid metadata") from exc
    actual = Path(result.get("executable", "")).resolve()
    expected = Path(INTERPRETER).resolve()
    if actual != expected:
        raise RuntimeError(f"Runtime environment preflight used {actual}, expected {expected}")
    result["interpreter"] = str(actual)
    return result


def _remote_review_preflight():
    approval = json.loads(AUTH.read_text())
    quality = json.loads(QUALITY.read_text())
    return {
        "integration_review": _remote_attestation(approval, "integration"),
        "quality_review": _remote_attestation(quality, "quality"),
    }


def _remote_attestation(approval, label):
    """Reuse v5's immutable remote-review contract without its globals."""
    pr = approval.get("pull_request")
    comment_id = approval.get("review_comment_id")
    reviewed = approval.get("reviewed_commit")
    if type(pr) is not int or type(comment_id) is not int or comment_id <= 0:
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
    body = comment.get("body", "")
    match = re.search(r"\*\*Reviewed commit:\*\*\s*`([0-9a-f]{10,40})`", body)
    if (
        author.get("login") != REVIEW_AUTHOR
        or author.get("type") != "Bot"
        or not match
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
    if actual_hash != approval.get("review_body_sha256"):
        raise ValueError(f"{label} review body differs from its immutable committed attestation")
    return {"label": label, "reviewed_commit": reviewed, "review_body_sha256": actual_hash}


def dependency_preflight():
    global _PREFLIGHT_RESULT
    result = runtime_environment_preflight(_ENV_FILE)
    result["dependency_and_review_preflight"] = _BASE_DEPENDENCY_PREFLIGHT()
    result["remote_reviews"] = _remote_review_preflight()
    from scripts.evals.agents import run_semantic_baseline_v2_2 as launcher
    args = SimpleNamespace(
        workload_control_v2="B_CONSECUTIVE_10",
        integration_approval=AUTH,
        approval=QUALITY,
    )
    head, review = launcher.verify_opt_in(args)
    launcher.verify_frozen_launcher(QUALITY)
    result["frozen_runtime_preflight"] = {
        "implementation_sha": head,
        "integration_reviewed_commit": review["reviewed_commit"],
        "launcher_sha256": sha(launcher.LAUNCHER),
        "adapter_sha256": sha(launcher.ADAPTER),
        "control_contract_sha256": sha(launcher.CONTROL_CONTRACT),
        "control_implementation_sha256": sha(launcher.CONTROL_IMPLEMENTATION),
        "control_collector_sha256": sha(launcher.CONTROL_COLLECTOR),
        "control_observer_sha256": sha(launcher.CONTROL_OBSERVER),
    }
    _PREFLIGHT_RESULT = result
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        or approval.get("review_comment_id") is None
        or approval.get("review_url") is None
        or approval.get("review_body_sha256") is None
        or not isinstance(reviewed, str)
        or approval.get("operation_wrapper") != str(WRAPPER)
        or approval.get("operation_wrapper_sha256") != sha(WRAPPER)
        or approval.get("wrapper_dependency") != str(DEPENDENCY)
        or approval.get("wrapper_dependency_sha256") != sha(DEPENDENCY)
        or approval.get("quality_approval_sha256") != sha(QUALITY)
        or approval.get("integration_launcher_sha256") != sha(helper.LAUNCHER)
        or approval.get("integration_adapter_sha256") != sha(helper.ADAPTER)
        or approval.get("control_contract_sha256") != sha(helper.CONTROL_CONTRACT)
        or approval.get("control_implementation_sha256") != sha(helper.CONTROL_IMPLEMENTATION)
        or approval.get("control_collector_sha256") != sha(helper.CONTROL_COLLECTOR)
        or approval.get("control_observer_sha256") != sha(helper.CONTROL_OBSERVER)
        or approval.get("interpreter_path") != str(INTERPRETER)
        or approval.get("interpreter_sha256") != sha(INTERPRETER)
        or approval.get("consumption_marker") != str(MARKER)
        or approval.get("launch_outcome") != str(OUTCOME)
        or approval.get("local_console_log") != str(LOG)
        or approval.get("staging_output_root") != str(STAGING)
    ):
        raise ValueError("Registered fresh-v6 control-v2 authorization changed")
    helper.base.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    if helper.base.git("diff", reviewed, "--", str(DEPENDENCY)):
        raise ValueError("Fresh-v5 dependency changed after fresh-v6 review")
    return approval, helper.base.git("rev-parse", "HEAD")


def _configure_helper():
    """Point the reviewed v4 machinery at fresh-v6 identities and paths."""
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
    global _ENV_FILE
    _ENV_FILE = env_file
    _configure_helper()
    # This explicit pass is non-consuming.  The inherited marker hook repeats
    # it immediately before the exclusive marker write as a defense in depth.
    dependency_preflight()
    return helper.base.run(index_manifest, env_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.index_manifest, arguments.env_file))
