"""Fresh-v3 one-use authorization wrapper for the semantic control-v2 baseline.

This wrapper preserves the reviewed SIGTERM-safe v2 operation while binding a
new immutable authorization namespace to the post-fix launcher head and clean
Codex review result. It must not be invoked without separate execution approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v2 as base


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_fresh_v3_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v3.py")
DEPENDENCY = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v2.py")
LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_2.py")
ADAPTER = Path("scripts/evals/agents/semantic_workload_control_v2.py")
CONTROL_CONTRACT = Path("docs/evals/workload_control_v2_contract.json")
CONTROL_IMPLEMENTATION = Path("scripts/diagnostics/workload_control_v2.py")
CONTROL_COLLECTOR = Path("scripts/diagnostics/run_workload_control_v2_calibration.py")
CONTROL_OBSERVER = Path("scripts/diagnostics/observe_semantic_workload.py")
FROZEN_PROVENANCE = Path("artifacts/evals/semantic_answer/v2/controlled_baselines/488e112a64b51fb2a5ad159194df993b2ff04f11/started.json")
MARKER = Path(".cache/semantic_v2_control_v2_fresh_v3_20260909.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_fresh_v3_20260909.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_fresh_v3_20260909.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_fresh_v3_20260909")
QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-FRESH-V3-20260909"
REVIEWED_COMMIT = "577eb02f5a90953a145b26eae12b42241e76c441"
CLEAN_REVIEW_BODY_SHA256 = "ce09de4fbbd93ea23f73fe2eef674a436fc29dcd422bf058a788ca3d1017ae74"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registration():
    if base.git("status", "--porcelain"):
        raise ValueError("Clean committed authorization required")
    if AUTH.read_bytes() != base.subprocess.check_output(["git", "show", f"HEAD:{AUTH}"]):
        raise ValueError("Authorization must match committed bytes")
    approval = json.loads(AUTH.read_text())
    reviewed = approval.get("reviewed_commit")
    if (
        approval.get("authorization_id") != AUTHORIZATION_ID
        or approval.get("status") != "approved_for_one_semantic_v2_control_v2_attempt"
        or approval.get("max_new_attempts") != 1
        or approval.get("selected_policy") != "B_CONSECUTIVE_10"
        or approval.get("review_status") != "completed_clean"
        or approval.get("review_comment_id") != 5578217788
        or approval.get("review_body_sha256") != CLEAN_REVIEW_BODY_SHA256
        or reviewed != REVIEWED_COMMIT
        or not isinstance(reviewed, str)
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed)
        or approval.get("operation_wrapper") != str(WRAPPER)
        or approval.get("operation_wrapper_sha256") != sha(WRAPPER)
        or approval.get("wrapper_dependency") != str(DEPENDENCY)
        or approval.get("wrapper_dependency_sha256") != sha(DEPENDENCY)
        or approval.get("integration_launcher_sha256") != sha(LAUNCHER)
        or approval.get("integration_adapter_sha256") != sha(ADAPTER)
        or approval.get("control_contract_sha256") != sha(CONTROL_CONTRACT)
        or approval.get("control_implementation_sha256") != sha(CONTROL_IMPLEMENTATION)
        or approval.get("control_collector_sha256") != sha(CONTROL_COLLECTOR)
        or approval.get("control_observer_sha256") != sha(CONTROL_OBSERVER)
        or approval.get("consumption_marker") != str(MARKER)
        or approval.get("launch_outcome") != str(OUTCOME)
        or approval.get("local_console_log") != str(LOG)
        or approval.get("staging_output_root") != str(STAGING)
    ):
        raise ValueError("Registered fresh-v3 control-v2 authorization changed")
    base.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    if base.git("diff", reviewed, "--", str(LAUNCHER)):
        raise ValueError("Reviewed semantic launcher changed after fresh-v3 review")
    return approval, base.git("rev-parse", "HEAD")


def _configure_base():
    for name, value in {
        "AUTH": AUTH,
        "WRAPPER": WRAPPER,
        "LAUNCHER": LAUNCHER,
        "MARKER": MARKER,
        "OUTCOME": OUTCOME,
        "LOG": LOG,
        "STAGING": STAGING,
        "QUALITY": QUALITY,
        "AUTHORIZATION_ID": AUTHORIZATION_ID,
    }.items():
        setattr(base, name, value)
    base.registration = registration


def run(index_manifest, env_file=None):
    _configure_base()
    return base.run(index_manifest, env_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.index_manifest, arguments.env_file))
