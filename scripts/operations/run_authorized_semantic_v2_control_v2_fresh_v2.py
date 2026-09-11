"""SIGTERM-safe candidate for a future single control-v2 baseline permission.

The separately reviewed approval named below does not exist in this candidate.
Only committed local identity checks and SIGTERM handler installation may precede
the durable marker. Remote review, imports, service preflight, monitoring, and
child launch happen after consumption.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_fresh_v2_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v2.py")
LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_2.py")
MARKER = Path(".cache/semantic_v2_control_v2_fresh_v2_20260909.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_fresh_v2_20260909.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_fresh_v2_20260909.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_fresh_v2_20260909")
QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-FRESH-V2-20260909"
SIGTERM_EXIT_CODE = 128 + signal.SIGTERM


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
    reviewed = approval.get("reviewed_commit")
    if (approval.get("authorization_id") != AUTHORIZATION_ID
        or approval.get("status") != "approved_for_one_semantic_v2_control_v2_attempt"
        or approval.get("max_new_attempts") != 1
        or approval.get("selected_policy") != "B_CONSECUTIVE_10"
        or not isinstance(reviewed, str)
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed)
        or approval.get("operation_wrapper") != str(WRAPPER)
        or approval.get("operation_wrapper_sha256") != sha(WRAPPER)
        or approval.get("integration_launcher_sha256") != sha(LAUNCHER)
        or approval.get("consumption_marker") != str(MARKER)
        or approval.get("staging_output_root") != str(STAGING)):
        raise ValueError("Registered fresh-v2 control-v2 authorization changed")
    git("merge-base", "--is-ancestor", reviewed, "HEAD")
    if git("diff", reviewed, "--", str(WRAPPER)):
        raise ValueError("Fresh-v2 operation wrapper changed after review")
    return approval, git("rev-parse", "HEAD")


def build_child_argv(
    index_manifest,
    env_file,
    *,
    interpreter,
    launcher,
    quality_approval,
    integration_approval,
    staging_root,
):
    """Build the launcher command solely from its explicit execution contract."""
    command = [
        str(interpreter), "-u", str(launcher),
        "--approval", str(quality_approval),
        "--integration-approval", str(integration_approval),
        "--workload-control-v2", "B_CONSECUTIVE_10",
        "--out-root", str(staging_root),
        "--index-manifest", str(index_manifest),
    ]
    if env_file is not None:
        command += ["--env-file", str(env_file)]
    return command


def run(index_manifest, env_file=None, child_env=None):
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
    termination = {
        "signal": "SIGTERM",
        "signal_number": signal.SIGTERM,
        "received": False,
        "received_at": None,
        "forward_attempted": False,
        "forward_attempted_at": None,
        "forwarded_to_child": False,
        "forwarded_at": None,
        "forward_error_type": None,
        "observation_closed_at": None,
        "finalization_policy": (
            "SIGTERM is blocked at the observation cutoff through exclusive outcome fsync; "
            "pending SIGTERM is latched at the cutoff and later delivery is ignored."
        ),
    }
    result = {
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": sha(AUTH),
        "implementation_sha": head,
        "started_at": now(),
        "stage": "pre_consumption",
        "child_started": False,
        "child_returncode": None,
        "wrapper_exit_code": None,
        "error_type": None,
        "termination": termination,
    }
    child = None
    active_child_pid = None

    def forward_sigterm():
        if active_child_pid is None or termination["forward_attempted"]:
            return
        termination["forward_attempted"] = True
        termination["forward_attempted_at"] = now()
        try:
            os.kill(active_child_pid, signal.SIGTERM)
        except OSError as exc:
            termination["forward_error_type"] = type(exc).__name__
        else:
            termination["forwarded_to_child"] = True
            termination["forwarded_at"] = now()

    def handle_sigterm(signum, _frame):
        if not termination["received"]:
            termination["received"] = True
            termination["received_at"] = now()
            termination["signal_number"] = signum
        forward_sigterm()

    consumed = False
    pending_error = None
    pending_traceback = None
    previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        write_once(MARKER, marker)
        consumed = True
        result["stage"] = "post_consumption_prelaunch"
        if not termination["received"]:
            command = build_child_argv(
                index_manifest,
                env_file,
                interpreter=sys.executable,
                launcher=LAUNCHER,
                quality_approval=QUALITY,
                integration_approval=AUTH,
                staging_root=STAGING,
            )
            result["stage"] = "launcher"
            log_descriptor = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(log_descriptor, "w") as log:
                previous_spawn_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
                try:
                    if signal.SIGTERM in signal.sigpending():
                        handle_sigterm(signal.SIGTERM, None)
                    if not termination["received"]:
                        popen_options = {
                            "stdout": log,
                            "stderr": subprocess.STDOUT,
                            "cwd": Path(__file__).resolve().parents[2],
                        }
                        if child_env is not None:
                            popen_options["env"] = child_env
                        child = subprocess.Popen(command, **popen_options)
                        active_child_pid = child.pid
                        result["child_started"] = True
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous_spawn_mask)
                if child is not None:
                    forward_sigterm()
                    result["child_returncode"] = child.wait()
                    active_child_pid = None
            if child is not None:
                result["stage"] = "finished"
                result["wrapper_exit_code"] = result["child_returncode"]
    except BaseException as exc:
        pending_error = exc
        pending_traceback = exc.__traceback__
        result["error_type"] = type(exc).__name__
        if child is not None and active_child_pid is not None:
            if termination["received"]:
                forward_sigterm()
            else:
                child.send_signal(signal.SIGINT)
            result["child_returncode"] = child.wait()
            active_child_pid = None
    finally:
        previous_finalization_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            termination["observation_closed_at"] = now()
            if signal.SIGTERM in signal.sigpending():
                handle_sigterm(signal.SIGTERM, None)
            if consumed:
                if termination["received"]:
                    result["stage"] = "terminated" if result["child_started"] else "terminated_prelaunch"
                    result["error_type"] = "SIGTERM"
                    result["wrapper_exit_code"] = SIGTERM_EXIT_CODE
                result["finished_at"] = now()
                result["policy"] = (
                    "Permission remains consumed regardless of preflight, captured cases, "
                    "control validity, interruption, signal, or exit status."
                )
                write_once(OUTCOME, result)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_finalization_mask)
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
    if termination["received"]:
        return SIGTERM_EXIT_CODE
    if pending_error is not None:
        raise pending_error.with_traceback(pending_traceback)
    return result["wrapper_exit_code"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.index_manifest, arguments.env_file))
