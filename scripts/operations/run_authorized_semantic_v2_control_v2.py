"""Consume the single control-v2 baseline permission before launching it.

Only committed local identity checks precede the durable marker.  Remote review,
imports, service preflight, monitoring, and child launch happen after consumption.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2.py")
LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_2.py")
MARKER = Path(".cache/semantic_v2_control_v2_20260909.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_20260909.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_20260909.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_20260909")
QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-20260909"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def write_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def registration():
    if git("status", "--porcelain"):
        raise ValueError("Clean committed authorization required")
    if AUTH.read_bytes() != subprocess.check_output(["git", "show", f"HEAD:{AUTH}"]):
        raise ValueError("Authorization must match committed bytes")
    approval = json.loads(AUTH.read_text())
    if (approval.get("authorization_id") != AUTHORIZATION_ID
        or approval.get("status") != "approved_for_one_semantic_v2_control_v2_attempt"
        or approval.get("max_new_attempts") != 1
        or approval.get("selected_policy") != "B_CONSECUTIVE_10"
        or approval.get("operation_wrapper") != str(WRAPPER)
        or approval.get("operation_wrapper_sha256") != sha(WRAPPER)
        or approval.get("integration_launcher_sha256") != sha(LAUNCHER)
        or approval.get("consumption_marker") != str(MARKER)
        or approval.get("staging_output_root") != str(STAGING)):
        raise ValueError("Registered control-v2 authorization changed")
    return approval, git("rev-parse", "HEAD")


def run(index_manifest, env_file=None):
    approval, head = registration()
    marker = {
        "status": "consumed",
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": sha(AUTH),
        "operation_wrapper_sha256": sha(WRAPPER),
        "integration_launcher_sha256": sha(LAUNCHER),
        "implementation_sha": head,
        "consumed_at": now(),
        "policy": "One invocation including all preflight failures; never delete, reset, or replace this marker.",
    }
    write_once(MARKER, marker)
    result = {
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": sha(AUTH),
        "implementation_sha": head,
        "started_at": now(),
        "stage": "post_consumption_prelaunch",
        "child_started": False,
        "child_returncode": None,
        "error_type": None,
    }
    child = None
    try:
        command = [
            sys.executable, "-u", str(LAUNCHER),
            "--approval", str(QUALITY),
            "--integration-approval", str(AUTH),
            "--workload-control-v2", "B_CONSECUTIVE_10",
            "--out-root", str(STAGING),
            "--index-manifest", str(index_manifest),
        ]
        if env_file is not None:
            command += ["--env-file", str(env_file)]
        result["stage"] = "launcher"
        log_descriptor = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(log_descriptor, "w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            result["child_started"] = True
            result["child_returncode"] = child.wait()
        result["stage"] = "finished"
        return result["child_returncode"]
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            result["child_returncode"] = child.wait()
        raise
    finally:
        result["finished_at"] = now()
        result["policy"] = (
            "Permission remains consumed regardless of preflight, captured cases, "
            "control validity, interruption, or exit status."
        )
        write_once(OUTCOME, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.index_manifest, arguments.env_file))
