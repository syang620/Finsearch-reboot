"""Inactive fresh-v7 one-use wrapper candidate.

Fresh-v6 remains byte-for-byte preserved and retired.  This candidate replaces
only its unavailable serialized-cache input with the separately reviewed,
read-only canonical Qdrant identity contract.  No approval exists yet.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from qdrant_client import QdrantClient

from scripts.evals.agents import run_semantic_baseline_v2_3 as launcher
from scripts.evals.retrieval import canonical_qdrant_v7 as canonical
from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v4 as helper
from scripts.operations import semantic_v7_environment as environment


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_fresh_v7_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v7.py")
ENVIRONMENT_HELPER = Path("scripts/operations/semantic_v7_environment.py")
DEPENDENCY = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v4.py")
V2_DEPENDENCY = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v2.py")
LAUNCHER = launcher.LAUNCHER
CONTROL_LAUNCHER = launcher.CONTROL_LAUNCHER
CANONICAL_VERIFIER = Path("scripts/evals/retrieval/canonical_qdrant_v7.py")
INDEX_ATTESTATION = canonical.CONTRACT
RETIREMENT = launcher.RETIREMENT
INTERPRETER = helper.INTERPRETER
MARKER = Path(".cache/semantic_v2_control_v2_fresh_v7_20260910.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_fresh_v7_20260910.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_fresh_v7_20260910.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_fresh_v7_20260910")
QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-FRESH-V7-20260910"
EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION = (
    environment.EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION
)
_V4_DEPENDENCY_PREFLIGHT = helper.dependency_preflight

_CHILD_ENV = None
_ENVIRONMENT_CONTRACT = None
_ENV_FILE_SUPPLIED = False
_PREFLIGHT_RESULT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _base_child_argv(
    index_attestation,
    env_file,
    *,
    interpreter,
    launcher,
    quality_approval,
    integration_approval,
    staging_root,
):
    if env_file is not None:
        raise ValueError("Fresh-v7 never passes an env-file path to the child")
    return [
        str(interpreter),
        "-u",
        str(launcher),
        "--approval",
        str(quality_approval),
        "--integration-approval",
        str(integration_approval),
        "--workload-control-v2",
        "B_CONSECUTIVE_10",
        "--out-root",
        str(staging_root),
        "--index-attestation",
        str(index_attestation),
    ]


def build_child_argv(index_attestation=INDEX_ATTESTATION):
    """Return the exact v7 child argv with no mutable env-file argument."""
    return _base_child_argv(
        index_attestation,
        None,
        interpreter=INTERPRETER,
        launcher=LAUNCHER,
        quality_approval=QUALITY,
        integration_approval=AUTH,
        staging_root=STAGING,
    )


def _safe_child_argv_metadata(argv):
    return {
        "argv_sha256": hashlib.sha256(
            b"\0".join(os.fsencode(part) for part in argv)
        ).hexdigest(),
        "argument_count": len(argv),
        "contains_env_file_argument": "--env-file" in argv,
        "interpreter": argv[0],
        "cwd": str(Path(__file__).resolve().parents[2]),
    }


def canonical_qdrant_preflight():
    """Verify the exact surviving index read-only and return safe metadata."""
    client = QdrantClient(host="127.0.0.1", port=6333, timeout=120)
    try:
        snapshot, records = canonical.verify_snapshot(client)
    finally:
        client.close()
    return {
        "collection": snapshot["collection"],
        "points": snapshot["points"],
        "payload_vectors_sha256": snapshot["payload_vectors_sha256"],
        "dense_vector_dimensions": snapshot["config"]["params"]["vectors"][
            "dense"
        ]["size"],
        "complete_records": len(records),
        "index_attestation_sha256": sha(INDEX_ATTESTATION),
        "canonical_qdrant_verifier_sha256": sha(CANONICAL_VERIFIER),
    }


def _remote_review_preflight():
    approval = json.loads(AUTH.read_text())
    quality = json.loads(QUALITY.read_text())
    return {
        "integration_review": environment.remote_attestation(approval, "integration"),
        "quality_review": environment.remote_attestation(quality, "quality"),
    }


def dependency_preflight():
    global _PREFLIGHT_RESULT
    result = environment.runtime_environment_preflight(
        INTERPRETER,
        child_env=_CHILD_ENV,
        environment_contract_record=_ENVIRONMENT_CONTRACT,
        env_file_supplied=_ENV_FILE_SUPPLIED,
    )
    result["dependency_preflight"] = _V4_DEPENDENCY_PREFLIGHT(
        preflight_env=_CHILD_ENV
    )
    result["canonical_qdrant"] = canonical_qdrant_preflight()
    result["remote_reviews"] = _remote_review_preflight()
    args = argparse.Namespace(
        workload_control_v2="B_CONSECUTIVE_10",
        integration_approval=AUTH,
        approval=QUALITY,
        index_attestation=INDEX_ATTESTATION,
    )
    head, review = launcher.verify_opt_in(args)
    launcher.verify_frozen_launcher_v7(QUALITY)
    result["frozen_runtime_preflight"] = {
        "implementation_sha": head,
        "integration_reviewed_commit": review["reviewed_commit"],
        "launcher_sha256": sha(LAUNCHER),
        "control_launcher_sha256": sha(CONTROL_LAUNCHER),
        "canonical_qdrant_verifier_sha256": sha(CANONICAL_VERIFIER),
        "index_attestation_sha256": sha(INDEX_ATTESTATION),
    }
    result["child_argv"] = _safe_child_argv_metadata(build_child_argv())
    _PREFLIGHT_RESULT = result
    return result


def registration():
    helper.interpreter_identity()
    if not AUTH.exists():
        raise RuntimeError("Fresh-v7 candidate is inactive; no approval exists")
    if helper.base.git("status", "--porcelain"):
        raise ValueError("Clean committed authorization required")
    if AUTH.read_bytes() != helper.subprocess.check_output(
        ["git", "show", f"HEAD:{AUTH}"]
    ):
        raise ValueError("Authorization must match committed bytes")
    approval = json.loads(AUTH.read_text())
    reviewed = approval.get("reviewed_commit")
    if (
        approval.get("authorization_id") != AUTHORIZATION_ID
        or approval.get("status")
        != "approved_for_one_semantic_v2_control_v2_fresh_v7_attempt"
        or approval.get("max_new_attempts") != 1
        or approval.get("selected_policy") != "B_CONSECUTIVE_10"
        or approval.get("review_status") != "completed_clean"
        or not isinstance(reviewed, str)
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed)
        or approval.get("operation_wrapper") != str(WRAPPER)
        or approval.get("operation_wrapper_sha256") != sha(WRAPPER)
        or approval.get("wrapper_dependency") != str(DEPENDENCY)
        or approval.get("wrapper_dependency_sha256") != sha(DEPENDENCY)
        or approval.get("environment_helper_sha256") != sha(ENVIRONMENT_HELPER)
        or approval.get("v2_dependency_sha256") != sha(V2_DEPENDENCY)
        or approval.get("integration_launcher_sha256") != sha(LAUNCHER)
        or approval.get("control_launcher_sha256") != sha(CONTROL_LAUNCHER)
        or approval.get("integration_adapter_sha256") != sha(helper.ADAPTER)
        or approval.get("canonical_qdrant_verifier_sha256")
        != sha(CANONICAL_VERIFIER)
        or approval.get("index_attestation_sha256") != sha(INDEX_ATTESTATION)
        or approval.get("fresh_v6_retirement_sha256") != sha(RETIREMENT)
        or approval.get("quality_approval_sha256") != sha(QUALITY)
        or approval.get("interpreter_path") != str(INTERPRETER)
        or approval.get("interpreter_sha256") != sha(INTERPRETER)
        or approval.get("consumption_marker") != str(MARKER)
        or approval.get("launch_outcome") != str(OUTCOME)
        or approval.get("local_console_log") != str(LOG)
        or approval.get("staging_output_root") != str(STAGING)
        or approval.get("effective_environment_contract_version")
        != EFFECTIVE_ENVIRONMENT_CONTRACT_VERSION
        or approval.get("preflight_implementation_sha256") != sha(WRAPPER)
    ):
        raise ValueError("Registered fresh-v7 control-v2 authorization changed")
    helper.base.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    for path, label in (
        (WRAPPER, "Fresh-v7 wrapper"),
        (ENVIRONMENT_HELPER, "Fresh-v7 environment helper"),
        (DEPENDENCY, "Fresh-v4 dependency"),
        (V2_DEPENDENCY, "Fresh-v2 dependency"),
        (LAUNCHER, "Fresh-v7 launcher"),
        (CONTROL_LAUNCHER, "Control-v2 launcher"),
        (CANONICAL_VERIFIER, "Canonical Qdrant verifier"),
        (INDEX_ATTESTATION, "Fresh-v7 index attestation"),
        (RETIREMENT, "Fresh-v6 retirement record"),
    ):
        environment.verify_reviewed_regular_tracked_blob(reviewed, path, label)
    return approval, helper.base.git("rev-parse", "HEAD")


def _configure_helper():
    for name, value in {
        "AUTH": AUTH,
        "WRAPPER": WRAPPER,
        "DEPENDENCY": DEPENDENCY,
        "LAUNCHER": LAUNCHER,
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
    helper.base.build_child_argv = _base_child_argv


def run(env_file=None):
    global _CHILD_ENV, _ENVIRONMENT_CONTRACT, _ENV_FILE_SUPPLIED
    _CHILD_ENV, _ENVIRONMENT_CONTRACT = environment.freeze_effective_child_environment(
        env_file
    )
    _ENV_FILE_SUPPLIED = env_file is not None
    _configure_helper()
    registration()
    dependency_preflight()
    return helper.base.run(INDEX_ATTESTATION, None, _CHILD_ENV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.env_file))
