"""Opt-in workload-control-v2 adapter around the frozen semantic-v2.1 launcher.

The semantic case loop, runtime, scorer, schedule, retries, models, and 120-second
timeout remain in ``run_semantic_baseline_v2_1``.  This adapter changes only the
operational workload observation and final baseline-eligibility decision.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import os

from evals.semantic_dataset_v2 import load_dataset, sha
from scripts.evals.agents import run_semantic_baseline_v2_1 as frozen
from scripts.evals.agents.semantic_workload_control_v2 import POLICY, WorkloadControlV2Monitor


PR33_REVIEWED_HEAD = "6101e3fc32a49460eef0b0a81ddb3c7c418d93f5"
PREREGISTRATION = Path("docs/evals/workload_control_v2_preregistration.json")
CONTROL_CONTRACT = Path("docs/evals/workload_control_v2_contract.json")
CONTROL_IMPLEMENTATION = Path("scripts/diagnostics/workload_control_v2.py")
CONTROL_COLLECTOR = Path("scripts/diagnostics/run_workload_control_v2_calibration.py")
CONTROL_OBSERVER = Path("scripts/diagnostics/observe_semantic_workload.py")
FROZEN_PROVENANCE = Path("artifacts/evals/semantic_answer/v2/controlled_baselines/488e112a64b51fb2a5ad159194df993b2ff04f11/started.json")
EXPECTED_PREREGISTRATION_SHA256 = "0c6c4ac93cee1b23e33ea2ea93ded0a9b56092cd9898045b44a0bb7595463483"
EXPECTED_CONTRACT_SHA256 = "903057b038684a34655dfd6884a471ade4472d6b36047cee94669ec9cdb57e76"
EXPECTED_CONTROL_IMPLEMENTATION_SHA256 = "3cf81da3c8cd4abeb4c8304f0f501af2b2262555def8e250b785cc8f50f78e63"
EXPECTED_CONTROL_COLLECTOR_SHA256 = "eb950de1a530666de29361850665c2d473c18f9d2b7b6d8a112c9cfafb75f60a"
EXPECTED_CONTROL_OBSERVER_SHA256 = "96bc78eb20e6200105d128190b66147bc18d5be6c7431359f7b45ecf62d8ed00"
EXPECTED_FROZEN_PROVENANCE_SHA256 = "0ae5a69d1935ad7980b158adc98b5547f505b97602354661cdb47ac2286ab730"
LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_2.py")
ADAPTER = Path("scripts/evals/agents/semantic_workload_control_v2.py")
INTEGRATION_APPROVAL = Path("docs/evals/semantic_answer_v2_control_v2_approval.json")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_opt_in(args):
    if args.workload_control_v2 != POLICY:
        raise ValueError(f"Explicit --workload-control-v2 {POLICY} opt-in required")
    frozen.clean_checkout()
    head = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", PR33_REVIEWED_HEAD, "HEAD")
    expected = {
        PREREGISTRATION: EXPECTED_PREREGISTRATION_SHA256,
        CONTROL_CONTRACT: EXPECTED_CONTRACT_SHA256,
        CONTROL_IMPLEMENTATION: EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        CONTROL_COLLECTOR: EXPECTED_CONTROL_COLLECTOR_SHA256,
        CONTROL_OBSERVER: EXPECTED_CONTROL_OBSERVER_SHA256,
        FROZEN_PROVENANCE: EXPECTED_FROZEN_PROVENANCE_SHA256,
    }
    for path, digest in expected.items():
        if file_sha(path) != digest:
            raise ValueError(f"Frozen workload-control-v2 identity changed: {path}")
    contract = json.loads(CONTROL_CONTRACT.read_text())
    if contract.get("selected_policy") != POLICY or contract.get("status") != "validated_frozen_candidate":
        raise ValueError("Validated workload-control-v2 contract changed")
    approval = frozen.committed_approval(args.integration_approval)
    reviewed = approval.get("reviewed_commit")
    if (approval.get("status") != "approved_for_one_semantic_v2_control_v2_attempt"
        or not isinstance(reviewed, str)
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed)
        or approval.get("selected_policy") != POLICY
        or approval.get("integration_launcher_sha256") != file_sha(LAUNCHER)
        or approval.get("integration_adapter_sha256") != file_sha(ADAPTER)
        or approval.get("control_contract_sha256") != file_sha(CONTROL_CONTRACT)
        or approval.get("control_implementation_sha256") != file_sha(CONTROL_IMPLEMENTATION)
        or approval.get("control_collector_sha256") != file_sha(CONTROL_COLLECTOR)
        or approval.get("control_observer_sha256") != file_sha(CONTROL_OBSERVER)):
        raise ValueError("Committed workload-control-v2 integration approval mismatch")
    git("merge-base", "--is-ancestor", reviewed, "HEAD")
    if git("diff", reviewed, "--", str(LAUNCHER), str(ADAPTER),
           str(CONTROL_IMPLEMENTATION), str(CONTROL_COLLECTOR), str(CONTROL_OBSERVER)):
        raise ValueError("Integration behavior changed after its full-SHA review")
    frozen.verify_remote_review(approval)
    return head, approval


def hard_control_only(state):
    if state.get("ac_power") is not True or state.get("low_power_mode") != 0:
        raise ValueError("AC power and Low Power Mode off required")


def verified_service_preflight(original):
    observed = original()
    expected = json.loads(FROZEN_PROVENANCE.read_text())["service_preflight"]
    qdrant_keys = ("title", "version", "commit")
    checks = {
        "ollama_version": observed.get("ollama_version") == expected.get("ollama_version"),
        "model_identities": observed.get("model_identities") == expected.get("model_identities"),
        "qdrant_service": all(
            observed.get("qdrant_service", {}).get(key) == expected.get("qdrant_service", {}).get(key)
            for key in qdrant_keys
        ),
        "sec_status": observed.get("sec_health", {}).get("status_code") == expected.get("sec_health", {}).get("status_code"),
        "reranker_backend": (
            observed.get("reranker_health", {}).get("metadata", {}).get("applied_backend")
            == expected.get("reranker_health", {}).get("metadata", {}).get("applied_backend")
            and observed.get("reranker_health", {}).get("metadata", {}).get("fallback_used") is False
            and observed.get("reranker_health", {}).get("requested_model")
            == expected.get("reranker_health", {}).get("requested_model")
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen service identity mismatch: {checks}")
    return observed


def provisional_validity(ending, expected_ids, evaluated_ids):
    """Preserve v2.1 checks but never claim eligibility before v2 closes."""
    reasons = []
    captured = ending["captured_cases"]
    if len(captured) != len(set(captured)) or set(captured) != set(expected_ids):
        reasons.append("incomplete_or_duplicate_capture")
    if len(evaluated_ids) != len(set(evaluated_ids)) or set(evaluated_ids) != set(expected_ids):
        reasons.append("incomplete_or_duplicate_evaluation")
    if ending.get("evaluation_errors"):
        reasons.append("evaluation_errors")
    if ending.get("control_violations"):
        reasons.append("hard_control_violations")
    for field in ("model_identities_unchanged", "index_unchanged"):
        if ending.get(field) is not True:
            reasons.append(field + "_not_verified")
    for field in ("model_verification_error", "index_verification_error", "runtime_cleanup_error"):
        if ending.get(field):
            reasons.append(field)
    ending.update(
        status="pending_workload_control_v2",
        official_baseline_eligible=False,
        invalidity_reasons=[*reasons, "workload_control_v2_pending"],
        capture_complete=len(captured) == len(expected_ids) and set(captured) == set(expected_ids),
    )
    return False


@contextmanager
def frozen_launcher_adapter(start_monitor, monitor):
    original = {
        "verify_launcher": frozen.verify_launcher,
        "settle_preflight": frozen.settle_preflight,
        "controls": frozen.controls,
        "service_preflight": frozen.service_preflight,
        "check_controls": frozen.check_controls,
        "finalize_validity": frozen.finalize_validity,
    }
    settled = original["settle_preflight"]
    control_calls = 0
    monitor_active = False

    async def settle(stage):
        nonlocal control_calls, monitor_active
        record = await settled(stage)
        if stage == "index_verification_and_planner_setup":
            start_monitor()
            control_calls = 0
            monitor_active = True
        return record

    def observed_controls():
        nonlocal control_calls
        state = original["controls"]()
        if not monitor_active:
            return state
        call_index = control_calls
        control_calls += 1
        # The frozen loop calls controls before and after each of 60 cases, then
        # once for final closure. Preserve a completed in-flight case, but stop
        # before launching another case after the monitor has invalidated the run.
        if ((monitor.trigger is not None or monitor.error is not None)
            and call_index % 2 == 0 and call_index < 120):
            raise RuntimeError("Workload-control-v2 invalidated the attempt")
        return state

    frozen.verify_launcher = lambda approval: verify_frozen_launcher(approval)
    frozen.controls = observed_controls
    frozen.service_preflight = lambda config: verified_service_preflight(
        lambda: original["service_preflight"](config)
    )
    frozen.check_controls = hard_control_only
    frozen.settle_preflight = settle
    frozen.finalize_validity = provisional_validity
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(frozen, name, value)


def verify_frozen_launcher(approval_path):
    """Apply the original v2.1 freeze while allowing only this versioned adapter."""
    frozen.clean_checkout()
    approval = frozen.committed_approval(approval_path)
    manifest_path = frozen.DATA / "optimization_manifest.json"
    if (file_sha(manifest_path) != frozen.ORIGINAL_MANIFEST_SHA
        or approval.get("status") != "approved_for_narrow_semantic_v2_baseline"
        or approval.get("optimization_manifest_sha256") != frozen.ORIGINAL_MANIFEST_SHA):
        raise ValueError("Original v2 optimization freeze/approval changed")
    manifest = json.loads(manifest_path.read_text())
    frozen.verify_files(frozen.DATA, manifest["files_sha256"])
    frozen.verify_files(".", manifest["code_sha256"])
    git("merge-base", "--is-ancestor", frozen.ORIGINAL_FREEZE, "HEAD")
    changed = set(git(
        "diff", "--name-only", frozen.ORIGINAL_FREEZE, "--", "src",
        "scripts/evals/agents", "scripts/evals/retrieval",
        "data/evals/semantic_answer",
    ).splitlines())
    allowed = {str(frozen.LAUNCHER), str(frozen.CONTRACT), str(LAUNCHER), str(ADAPTER)}
    if changed - allowed:
        raise ValueError("Frozen semantic inputs or behavior changed outside the v2 adapter")
    frozen.verify_remote_review(approval)
    launcher = frozen.committed_approval(frozen.CONTRACT)
    if (launcher.get("status") != "approved_preflight_only_launcher_v2_1"
        or launcher.get("optimization_manifest_sha256") != frozen.ORIGINAL_MANIFEST_SHA
        or launcher.get("quality_approval_sha256") != sha(approval_path)
        or launcher.get("launcher_sha256") != sha(frozen.LAUNCHER)
        or launcher.get("dataset_sha256") != manifest["dataset_sha256"]
        or launcher.get("judge_enabled") != manifest["judge_enabled"]
        or launcher.get("settle_seconds") != frozen.SETTLE_SECONDS):
        raise ValueError("Frozen v2.1 launcher identity changed")
    frozen.verify_remote_review(launcher)
    return manifest, approval, launcher


def finalize_output(out, monitor_summary, review):
    completion_path = out / "completion.json"
    completion = json.loads(completion_path.read_text())
    reasons = [
        reason for reason in completion.get("invalidity_reasons", [])
        if reason != "workload_control_v2_pending"
    ]
    if not monitor_summary["valid"]:
        reasons.append("workload_control_v2_failed")
    final = {
        **completion,
        "workload_control_v2": {
            **monitor_summary,
            "contract_sha256": file_sha(CONTROL_CONTRACT),
            "control_implementation_sha256": file_sha(CONTROL_IMPLEMENTATION),
            "control_collector_sha256": file_sha(CONTROL_COLLECTOR),
            "control_observer_sha256": file_sha(CONTROL_OBSERVER),
            "integration_launcher_sha256": file_sha(LAUNCHER),
            "integration_adapter_sha256": file_sha(ADAPTER),
            "integration_review": review,
        },
        "invalidity_reasons": reasons,
        "status": "complete" if not reasons else "invalid_diagnostic",
        "official_baseline_eligible": not reasons,
        "supersedes_operational_eligibility_in": "completion.json",
    }
    frozen.save(out / "workload_control_v2_completion.json", final)
    if not reasons:
        cases, _ = load_dataset(frozen.DATA)
        rows = [json.loads(line) for line in (out / "deterministic.jsonl").read_text().splitlines()]
        frozen.save(out / "deterministic_summary.json", frozen.deterministic_breakdowns(cases, rows))
    frozen.save(out / "workload_control_v2_files_sha256.json", {
        path.name: sha(path) for path in sorted(out.iterdir())
        if path.is_file() and path.name != "workload_control_v2_files_sha256.json"
    })


async def run(args):
    head, review = verify_opt_in(args)
    out = args.out_root / head
    control_root = Path(".cache/semantic_answer_v2_control_v2") / head
    raw = control_root / "workload_control_v2.jsonl"
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    monitor = WorkloadControlV2Monitor(
        raw, PREREGISTRATION, CONTROL_CONTRACT, awake,
        provenance={
            "implementation_sha": head,
            "control_implementation_sha256": file_sha(CONTROL_IMPLEMENTATION),
            "control_collector_sha256": file_sha(CONTROL_COLLECTOR),
            "control_observer_sha256": file_sha(CONTROL_OBSERVER),
            "integration_launcher_sha256": file_sha(LAUNCHER),
            "integration_adapter_sha256": file_sha(ADAPTER),
            "integration_review": review,
        },
    )
    started = False

    def start_monitor():
        nonlocal started
        monitor.start()
        started = True

    failure = None
    try:
        with frozen_launcher_adapter(start_monitor, monitor):
            await frozen.run_once(args)
    except BaseException as exc:
        failure = exc
    finally:
        summary = monitor.stop() if started else None
        awake.terminate()
        awake.wait(timeout=10)
    if summary is None:
        if failure is not None:
            raise failure
        raise RuntimeError("Workload-control-v2 did not reach the settled activation boundary")
    if not out.exists():
        raise RuntimeError("Frozen semantic launcher produced no output")
    destination = out / "workload_control_v2.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(raw, destination)
    summary["raw_sha256"] = file_sha(destination)
    finalize_output(out, summary, review)
    if failure is not None:
        raise failure


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, default=Path("docs/evals/semantic_answer_v2_quality_approval.json"))
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/evals/semantic_answer/v2/baselines"))
    parser.add_argument("--workload-control-v2", required=True)
    parser.add_argument("--integration-approval", type=Path, default=INTEGRATION_APPROVAL)
    args = parser.parse_args(argv)
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
