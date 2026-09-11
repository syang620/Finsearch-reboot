"""Inactive semantic-v2 launcher with split workload qualification.

No approval file exists for this candidate.  This module therefore cannot run
semantic cases until a later exact-head review and approval commit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re

from evals.semantic_dataset_v2 import sha
from scripts.evals.agents import run_semantic_baseline_v2_1 as frozen
from scripts.evals.agents import run_semantic_baseline_v2_2 as legacy
from scripts.evals.agents import run_semantic_baseline_v2_3 as canonical
from scripts.evals.agents import semantic_workload_qualification_v3 as qualification


LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_4.py")
ADAPTER = Path("scripts/evals/agents/semantic_workload_qualification_v3.py")
CONTRACT = Path("docs/evals/workload_qualification_v3_contract.json")
APPROVAL = Path("docs/evals/semantic_answer_v2_workload_qualification_v3_approval.json")


def file_sha(path):
    return legacy.file_sha(path)


def verify_frozen_performance_dependencies():
    expected = {
        legacy.PREREGISTRATION: legacy.EXPECTED_PREREGISTRATION_SHA256,
        legacy.CONTROL_CONTRACT: legacy.EXPECTED_CONTRACT_SHA256,
        legacy.CONTROL_IMPLEMENTATION: legacy.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        legacy.CONTROL_COLLECTOR: legacy.EXPECTED_CONTROL_COLLECTOR_SHA256,
        legacy.CONTROL_OBSERVER: legacy.EXPECTED_CONTROL_OBSERVER_SHA256,
        legacy.ADAPTER: qualification.PERFORMANCE_ADAPTER_SHA256,
        legacy.FROZEN_PROVENANCE: legacy.EXPECTED_FROZEN_PROVENANCE_SHA256,
    }
    for path, digest in expected.items():
        if file_sha(path) != digest:
            raise ValueError(f"Frozen workload-control-v2 identity changed: {path}")


def approval_reviewed_paths(index_attestation):
    return (
        frozen.LAUNCHER,
        frozen.CONTRACT,
        legacy.PREREGISTRATION,
        legacy.CONTROL_CONTRACT,
        legacy.CONTROL_IMPLEMENTATION,
        legacy.CONTROL_COLLECTOR,
        legacy.CONTROL_OBSERVER,
        legacy.ADAPTER,
        legacy.LAUNCHER,
        legacy.FROZEN_PROVENANCE,
        canonical.LAUNCHER,
        canonical.CANONICAL_VERIFIER,
        Path(index_attestation),
        LAUNCHER,
        ADAPTER,
        CONTRACT,
    )


def reviewed_diff(reviewed, paths):
    return legacy.git(
        "--literal-pathspecs",
        "diff",
        reviewed,
        "--",
        *(str(path) for path in paths),
    )


def verify_opt_in(args):
    if args.workload_qualification_v3 != qualification.POLICY:
        raise ValueError(
            f"Explicit --workload-qualification-v3 {qualification.POLICY} opt-in required"
        )
    frozen.clean_checkout()
    verify_frozen_performance_dependencies()
    frozen.committed_approval(args.index_attestation)
    head = legacy.git("rev-parse", "HEAD")
    approval = frozen.committed_approval(args.qualification_approval)
    reviewed = approval.get("reviewed_commit")
    expected = {
        "status": "approved_for_split_workload_qualification_v3",
        "selected_policy": qualification.POLICY,
        "performance_policy": qualification.PERFORMANCE_POLICY,
        "performance_preregistration_sha256": file_sha(legacy.PREREGISTRATION),
        "performance_contract_sha256": file_sha(legacy.CONTROL_CONTRACT),
        "performance_implementation_sha256": file_sha(legacy.CONTROL_IMPLEMENTATION),
        "performance_collector_sha256": file_sha(legacy.CONTROL_COLLECTOR),
        "performance_adapter_sha256": file_sha(legacy.ADAPTER),
        "service_provenance_sha256": file_sha(legacy.FROZEN_PROVENANCE),
        "legacy_control_launcher_sha256": file_sha(legacy.LAUNCHER),
        "integration_launcher_sha256": file_sha(LAUNCHER),
        "integration_adapter_sha256": file_sha(ADAPTER),
        "qualification_contract_sha256": file_sha(CONTRACT),
        "canonical_launcher_sha256": file_sha(canonical.LAUNCHER),
        "canonical_qdrant_verifier_sha256": file_sha(canonical.CANONICAL_VERIFIER),
        "index_attestation_sha256": file_sha(args.index_attestation),
    }
    if (
        not isinstance(reviewed, str)
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed)
        or any(approval.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Committed workload-qualification-v3 approval mismatch")
    legacy.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    reviewed_paths = approval_reviewed_paths(args.index_attestation)
    if reviewed_diff(reviewed, reviewed_paths):
        raise ValueError("Workload-qualification-v3 behavior changed after review")
    frozen.verify_remote_review(approval)
    return head, approval


def verify_frozen_launcher_v3(approval_path):
    """Keep frozen semantic behavior while admitting the reviewed v3 shell."""
    frozen.clean_checkout()
    verify_frozen_performance_dependencies()
    approval = frozen.committed_approval(approval_path)
    manifest_path = frozen.DATA / "optimization_manifest.json"
    if (
        file_sha(manifest_path) != frozen.ORIGINAL_MANIFEST_SHA
        or approval.get("status") != "approved_for_narrow_semantic_v2_baseline"
        or approval.get("optimization_manifest_sha256") != frozen.ORIGINAL_MANIFEST_SHA
    ):
        raise ValueError("Original v2 optimization freeze/approval changed")
    manifest = json.loads(manifest_path.read_text())
    frozen.verify_files(frozen.DATA, manifest["files_sha256"])
    frozen.verify_files(".", manifest["code_sha256"])
    legacy.git("merge-base", "--is-ancestor", frozen.ORIGINAL_FREEZE, "HEAD")
    changed = set(
        legacy.git(
            "diff",
            "--name-only",
            frozen.ORIGINAL_FREEZE,
            "--",
            "src",
            "scripts/evals/agents",
            "scripts/evals/retrieval",
            "data/evals/semantic_answer",
        ).splitlines()
    )
    allowed = {
        str(frozen.LAUNCHER),
        str(frozen.CONTRACT),
        str(legacy.LAUNCHER),
        str(legacy.ADAPTER),
        str(canonical.LAUNCHER),
        str(canonical.CANONICAL_VERIFIER),
        str(LAUNCHER),
        str(ADAPTER),
    }
    if changed - allowed:
        raise ValueError("Frozen semantic behavior changed outside reviewed adapters")
    frozen.verify_remote_review(approval)
    launcher = frozen.committed_approval(frozen.CONTRACT)
    if (
        launcher.get("status") != "approved_preflight_only_launcher_v2_1"
        or launcher.get("optimization_manifest_sha256") != frozen.ORIGINAL_MANIFEST_SHA
        or launcher.get("quality_approval_sha256") != sha(approval_path)
        or launcher.get("launcher_sha256") != sha(frozen.LAUNCHER)
        or launcher.get("dataset_sha256") != manifest["dataset_sha256"]
        or launcher.get("judge_enabled") != manifest["judge_enabled"]
        or launcher.get("settle_seconds") != frozen.SETTLE_SECONDS
    ):
        raise ValueError("Frozen v2.1 launcher identity changed")
    frozen.verify_remote_review(launcher)
    return manifest, approval, launcher


async def run(args):
    head, review = verify_opt_in(args)
    await qualification.run(
        args,
        head,
        review,
        canonical.run_once,
        verify_frozen_launcher_v3,
        CONTRACT,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approval",
        type=Path,
        default=Path("docs/evals/semantic_answer_v2_quality_approval.json"),
    )
    parser.add_argument("--index-attestation", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workload-qualification-v3", required=True)
    parser.add_argument("--qualification-approval", type=Path, default=APPROVAL)
    args = parser.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
