"""Independent effective-environment and review checks for fresh-v7."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType

from scripts.evals.retrieval import run_benchmark_v3 as provenance


EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION = "2"
PREFLIGHT_TIMEOUT_SECONDS = 30
REQUIRED_ENVIRONMENT_KEYS = (
    "SEC_USER_AGENT",
    "DASHSCOPE_API_KEY",
    "QWEN3_RERANK_API_KEY",
    "SEC_METRIC_FIXTURE_ROOT",
)
HASHABLE_ENVIRONMENT_KEYS = frozenset(
    (
        "PYTHONPATH",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "QDRANT_COLLECTION_NAME",
        "TABLES_DIR",
        "LITELLM_GPT_MODEL",
        "LITELLM_CLAUDE_MODEL",
        "LITELLM_GEMINI_MODEL",
        "LLM_FALLBACK_MODELS",
    )
)
REVIEW_AUTHOR = "chatgpt-codex-connector[bot]"
REVIEW_REPOSITORY = "syang620/Finsearch-reboot"


def _environment_script():
    return """
import json
import os
import sys

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
print(json.dumps({
    "executable": sys.executable,
    "sec_user_agent_present": True,
    "reranker_credential_sources": credential_sources,
    "fixture_root_forbidden": True,
}))
"""


def environment_contract(effective, inherited, env_file_supplied):
    required_sources = {}
    entries = []
    for name in sorted(set(effective) | set(REQUIRED_ENVIRONMENT_KEYS)):
        if name not in effective:
            source = "absent"
        elif name in inherited:
            source = "inherited"
        else:
            source = "env_file"
        entry = {"name": name, "source": source}
        if source != "absent" and name in HASHABLE_ENVIRONMENT_KEYS:
            entry["value_sha256"] = hashlib.sha256(effective[name].encode()).hexdigest()
        entries.append(entry)
        if name in REQUIRED_ENVIRONMENT_KEYS:
            required_sources[name] = source
    fingerprint_input = {
        "version": EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION,
        "env_file_supplied": env_file_supplied,
        "entries": entries,
        "required_key_sources": required_sources,
    }
    return {
        "effective_environment_contract_version": EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION,
        "env_file_supplied": env_file_supplied,
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(
                fingerprint_input, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "required_key_sources": required_sources,
    }


def freeze_effective_child_environment(env_file):
    inherited = os.environ.copy()
    if env_file is None:
        effective = inherited.copy()
    else:
        path = Path(env_file)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise RuntimeError("runtime env file is missing or unreadable")
        from dotenv import load_dotenv

        try:
            load_dotenv(path, override=False)
            effective = os.environ.copy()
        finally:
            os.environ.clear()
            os.environ.update(inherited)
    return MappingProxyType(effective), environment_contract(
        effective, inherited, env_file is not None
    )


def _safe_runtime_failure_detail(detail):
    for message in (
        "SEC_METRIC_FIXTURE_ROOT is forbidden for the live baseline",
        "SEC_USER_AGENT is required",
        "an existing reranker credential is required",
    ):
        if message in detail:
            return message
    return "runtime environment contract rejected"


def runtime_environment_preflight(
    interpreter,
    *,
    child_env,
    environment_contract_record,
    env_file_supplied,
):
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", _environment_script()],
            cwd=repo_root,
            env=child_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Runtime environment preflight timed out") from exc
    except OSError as exc:
        raise RuntimeError("Runtime environment preflight could not start") from exc
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Exact-interpreter runtime environment preflight failed with exit "
            f"{completed.returncode}: {_safe_runtime_failure_detail(detail)}"
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Runtime environment preflight returned invalid metadata") from exc
    actual = Path(result.get("executable", "")).resolve()
    expected = Path(interpreter).resolve()
    if actual != expected:
        raise RuntimeError("Runtime environment preflight used an unexpected interpreter")
    result["env_file_supplied"] = env_file_supplied
    result["interpreter"] = str(actual)
    result["effective_environment"] = environment_contract_record
    return result


def remote_attestation(approval, label):
    pr = approval.get("pull_request")
    comment_id = approval.get("review_comment_id")
    reviewed = approval.get("reviewed_commit")
    if type(pr) is not int or type(comment_id) is not int or comment_id <= 0:
        raise ValueError(f"{label} attestation requires numeric PR/comment identities")
    if not isinstance(reviewed, str) or not re.fullmatch(r"[0-9a-f]{40}", reviewed):
        raise ValueError(f"{label} attestation requires an exact reviewed commit")
    prefix = f"repos/{REVIEW_REPOSITORY}"
    comment = provenance.github_json(f"{prefix}/issues/comments/{comment_id}")
    expected_url = (
        f"https://github.com/{REVIEW_REPOSITORY}/pull/{pr}#issuecomment-{comment_id}"
    )
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
    if (
        pull.get("base", {}).get("repo", {}).get("full_name") != REVIEW_REPOSITORY
        or pull.get("state") != "open"
    ):
        raise ValueError(f"{label} review does not belong to the open benchmark PR")
    reviews = provenance.github_json(f"{prefix}/pulls/{pr}/reviews", paginate=True)
    if any(
        review.get("commit_id") == reviewed
        and review.get("state") == "CHANGES_REQUESTED"
        for review in reviews
    ):
        raise ValueError(f"{label} attested commit has requested changes")
    findings = provenance.github_json(f"{prefix}/pulls/{pr}/comments", paginate=True)
    if any(
        (finding.get("user") or {}).get("login") == REVIEW_AUTHOR
        and finding.get("original_commit_id", finding.get("commit_id")) == reviewed
        for finding in findings
    ):
        raise ValueError(f"{label} attested commit has inline Codex findings")
    actual_hash = hashlib.sha256(body.encode()).hexdigest()
    if actual_hash != approval.get("review_body_sha256"):
        raise ValueError(f"{label} review body differs from committed attestation")
    return {
        "label": label,
        "reviewed_commit": reviewed,
        "review_body_sha256": actual_hash,
    }


def verify_reviewed_regular_tracked_blob(reviewed, path, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file, not a symlink")
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            stderr=subprocess.DEVNULL,
        )
        reviewed_bytes = subprocess.check_output(["git", "show", f"{reviewed}:{path}"])
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{label} is not a tracked reviewed path") from exc
    if path.read_bytes() != reviewed_bytes:
        raise ValueError(f"{label} differs from its reviewed Git blob")
