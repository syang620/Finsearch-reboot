"""Fresh-v4 one-use wrapper with exact-interpreter dependency preflight.

Fresh-v3 is permanently exhausted.  This version is intentionally separate:
the exact finsearch-arm interpreter and the complete launcher import chain must
pass a non-consuming preflight before the inherited SIGTERM-safe operation can
write its one-use marker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.operations import run_authorized_semantic_v2_control_v2_fresh_v2 as base


AUTH = Path("docs/evals/semantic_answer_v2_control_v2_fresh_v4_approval.json")
WRAPPER = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v4.py")
DEPENDENCY = Path("scripts/operations/run_authorized_semantic_v2_control_v2_fresh_v2.py")
LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_2.py")
ADAPTER = Path("scripts/evals/agents/semantic_workload_control_v2.py")
CONTROL_CONTRACT = Path("docs/evals/workload_control_v2_contract.json")
CONTROL_IMPLEMENTATION = Path("scripts/diagnostics/workload_control_v2.py")
CONTROL_COLLECTOR = Path("scripts/diagnostics/run_workload_control_v2_calibration.py")
CONTROL_OBSERVER = Path("scripts/diagnostics/observe_semantic_workload.py")
FROZEN_PROVENANCE = Path("artifacts/evals/semantic_answer/v2/controlled_baselines/488e112a64b51fb2a5ad159194df993b2ff04f11/started.json")
INTERPRETER = Path("/Users/shicheny/miniforge3/envs/finsearch-arm/bin/python")
MARKER = Path(".cache/semantic_v2_control_v2_fresh_v4_20260909.consumed.json")
OUTCOME = Path(".cache/semantic_v2_control_v2_fresh_v4_20260909.launch_outcome.json")
LOG = Path(".cache/semantic_v2_control_v2_fresh_v4_20260909.console.log")
STAGING = Path(".cache/semantic_v2_control_v2_fresh_v4_20260909")
QUALITY = Path("docs/evals/semantic_answer_v2_quality_approval.json")
AUTHORIZATION_ID = "SEMANTIC-V2-CONTROL-V2-FRESH-V4-20260909"

REQUIRED_MODULES = (
    "requests",
    "qdrant_client",
    "dotenv",
    "evals.semantic_dataset_v2",
    "scripts.evals.agents.run_semantic_baseline_v2_2",
    "scripts.evals.agents.semantic_workload_control_v2",
    "scripts.diagnostics.run_workload_control_v2_calibration",
    "scripts.diagnostics.observe_semantic_workload",
    "agents.planner.interactive_target_resolution",
    "agents.orchestrator.agent_orchestrator",
    # These are imported lazily when the first retrieval case starts and when
    # the child launches the stdio MCP server.
    "agents.retrieval.mcp_client",
    "mcp_server.server",
)
REQUIRED_PACKAGES = (
    "requests",
    "qdrant-client",
    "python-dotenv",
    "langchain-ollama",
    "langgraph",
    "langgraph-checkpoint-sqlite",
)
_PREFLIGHT_RESULT = None
_BASE_WRITE_ONCE = base.write_once


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpreter_identity():
    actual = Path(sys.executable).resolve()
    expected = INTERPRETER.resolve()
    if actual != expected:
        raise RuntimeError(
            f"Exact finsearch-arm interpreter required: expected {expected}, got {actual}"
        )
    return {
        "executable": str(actual),
        "python_version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def _preflight_script():
    modules = repr(REQUIRED_MODULES)
    packages = repr(REQUIRED_PACKAGES)
    return f"""
import importlib
import importlib.metadata
import json
import os
import sys

modules = {modules}
for name in modules:
    importlib.import_module(name)
packages = {packages}
versions = {{name: importlib.metadata.version(name) for name in packages}}
print(json.dumps({{
    'executable': sys.executable,
    'python_version': sys.version,
    'prefix': sys.prefix,
    'base_prefix': sys.base_prefix,
    'virtual_env': os.environ.get('VIRTUAL_ENV'),
    'conda_prefix': os.environ.get('CONDA_PREFIX'),
    'path': os.environ.get('PATH'),
    'modules': list(modules),
    'packages': versions,
}}))
"""


def dependency_preflight(preflight_env=None):
    identity = interpreter_identity()
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(preflight_env) if preflight_env is not None else os.environ.copy()
    if preflight_env is None:
        project_path = os.pathsep.join((str(repo_root), str(repo_root / "src")))
        if env.get("PYTHONPATH"):
            project_path += os.pathsep + env["PYTHONPATH"]
        env["PYTHONPATH"] = project_path
    completed = subprocess.run(
        [str(INTERPRETER), "-c", _preflight_script()],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Exact-interpreter dependency preflight failed with exit "
            f"{completed.returncode}: {detail}"
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exact-interpreter dependency preflight returned invalid metadata") from exc
    if Path(result.get("executable", "")).resolve() != INTERPRETER.resolve():
        raise RuntimeError("Dependency preflight used an unexpected interpreter")
    result["approved_interpreter"] = identity
    result["required_modules"] = list(REQUIRED_MODULES)
    result["required_packages"] = list(REQUIRED_PACKAGES)
    return result


def registration():
    interpreter_identity()
    if base.git("status", "--porcelain"):
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
        or approval.get("interpreter_path") != str(INTERPRETER)
        or approval.get("interpreter_sha256") != sha(INTERPRETER)
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
        raise ValueError("Registered fresh-v4 control-v2 authorization changed")
    base.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    if base.git("diff", reviewed, "--", str(LAUNCHER)):
        raise ValueError("Reviewed semantic launcher changed after fresh-v4 review")
    return approval, base.git("rev-parse", "HEAD")


def _write_once(path, record):
    global _PREFLIGHT_RESULT
    if Path(path) == MARKER:
        _PREFLIGHT_RESULT = dependency_preflight()
        record["interpreter_preflight"] = _PREFLIGHT_RESULT
    elif Path(path) == OUTCOME and _PREFLIGHT_RESULT is not None:
        record["interpreter_preflight"] = _PREFLIGHT_RESULT
    return _BASE_WRITE_ONCE(path, record)


def _configure_base():
    interpreter_identity()
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
    # The inherited operation builds its child command from base.sys.executable.
    # Pin that command to the already-validated absolute interpreter path.
    base.sys.executable = str(INTERPRETER)
    base.registration = registration
    base.write_once = _write_once


def run(index_manifest, env_file=None, child_env=None):
    _configure_base()
    return base.run(index_manifest, env_file, child_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.index_manifest, arguments.env_file))
