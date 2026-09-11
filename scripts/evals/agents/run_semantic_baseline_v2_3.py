"""Inactive fresh-v7 adapter for the surviving canonical Qdrant index.

The semantic case loop and workload-control-v2 behavior remain inherited from
the frozen v2.1/v2.2 launchers.  Only the unavailable build-cache provenance
check is replaced by an exact read-only Qdrant identity guard before and after
the attempt.  This module grants no execution authority.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import requests
from qdrant_client import QdrantClient

from evals.semantic_dataset_v2 import load_dataset, load_numeric_catalog, read, sha
from scripts.evals.agents import run_semantic_baseline_v2_1 as frozen
from scripts.evals.agents import run_semantic_baseline_v2_2 as control
from scripts.evals.retrieval import canonical_qdrant_v7 as canonical


LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_3.py")
CONTROL_LAUNCHER = Path("scripts/evals/agents/run_semantic_baseline_v2_2.py")
INTEGRATION_APPROVAL = Path(
    "docs/evals/semantic_answer_v2_control_v2_fresh_v7_approval.json"
)
INDEX_ATTESTATION = canonical.CONTRACT
CANONICAL_VERIFIER = Path("scripts/evals/retrieval/canonical_qdrant_v7.py")
RETIREMENT = Path(
    "docs/evals/semantic_answer_v2_control_v2_fresh_v6_retirement.json"
)


def file_sha(path):
    return control.file_sha(path)


def verify_opt_in(args):
    if args.workload_control_v2 != control.POLICY:
        raise ValueError(
            f"Explicit --workload-control-v2 {control.POLICY} opt-in required"
        )
    frozen.clean_checkout()
    head = control.git("rev-parse", "HEAD")
    control.git("merge-base", "--is-ancestor", control.PR33_REVIEWED_HEAD, "HEAD")
    expected = {
        control.PREREGISTRATION: control.EXPECTED_PREREGISTRATION_SHA256,
        control.CONTROL_CONTRACT: control.EXPECTED_CONTRACT_SHA256,
        control.CONTROL_IMPLEMENTATION: control.EXPECTED_CONTROL_IMPLEMENTATION_SHA256,
        control.CONTROL_COLLECTOR: control.EXPECTED_CONTROL_COLLECTOR_SHA256,
        control.CONTROL_OBSERVER: control.EXPECTED_CONTROL_OBSERVER_SHA256,
        control.FROZEN_PROVENANCE: control.EXPECTED_FROZEN_PROVENANCE_SHA256,
    }
    for path, digest in expected.items():
        if file_sha(path) != digest:
            raise ValueError(f"Frozen workload-control-v2 identity changed: {path}")
    contract = canonical.load_contract(args.index_attestation)
    approval = frozen.committed_approval(args.integration_approval)
    reviewed = approval.get("reviewed_commit")
    if (
        approval.get("status")
        != "approved_for_one_semantic_v2_control_v2_fresh_v7_attempt"
        or not isinstance(reviewed, str)
        or not re.fullmatch(r"[0-9a-f]{40}", reviewed)
        or approval.get("selected_policy") != control.POLICY
        or approval.get("integration_launcher_sha256") != file_sha(LAUNCHER)
        or approval.get("control_launcher_sha256") != file_sha(CONTROL_LAUNCHER)
        or approval.get("integration_adapter_sha256") != file_sha(control.ADAPTER)
        or approval.get("canonical_qdrant_verifier_sha256")
        != file_sha(CANONICAL_VERIFIER)
        or approval.get("index_attestation_sha256") != file_sha(args.index_attestation)
        or approval.get("fresh_v6_retirement_sha256") != file_sha(RETIREMENT)
    ):
        raise ValueError("Committed fresh-v7 integration approval mismatch")
    control.git("merge-base", "--is-ancestor", reviewed, "HEAD")
    reviewed_paths = (
        LAUNCHER,
        CONTROL_LAUNCHER,
        control.ADAPTER,
        CANONICAL_VERIFIER,
        args.index_attestation,
        RETIREMENT,
    )
    if control.git("diff", reviewed, "--", *(str(path) for path in reviewed_paths)):
        raise ValueError("Fresh-v7 integration behavior changed after exact-head review")
    frozen.verify_remote_review(approval)
    if contract["collection"] != "finsearch_benchmark_v2_39c8d01ee5c71710":
        raise ValueError("Fresh-v7 collection identity changed")
    return head, approval


def verify_frozen_launcher_v7(approval_path):
    """Keep the v2.1 freeze while admitting both reviewed adapter generations."""
    frozen.clean_checkout()
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
    control.git("merge-base", "--is-ancestor", frozen.ORIGINAL_FREEZE, "HEAD")
    changed = set(
        control.git(
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
        str(CONTROL_LAUNCHER),
        str(LAUNCHER),
        str(control.ADAPTER),
        str(CANONICAL_VERIFIER),
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


def close_index_guard(ending, guard, before):
    """Record a second exact primary-index verification before eligibility."""
    try:
        after, _ = guard.verify_after()
        ending.update(
            index_after=after,
            canonical_index_after_verified=True,
            index_unchanged=(after == before),
        )
    except Exception as exc:
        ending.update(
            canonical_index_after_verified=False,
            index_unchanged=False,
            index_verification_error=f"{type(exc).__name__}: {exc}",
        )


async def run_once(args):
    freeze, approval, launcher = frozen.verify_launcher(args.approval)
    cases, counts = load_dataset(frozen.DATA)
    config = json.loads(
        Path("data/evals/semantic_answer/v1/evaluation_config.json").read_text()
    )
    head = control.git("rev-parse", "HEAD")
    out = args.out_root / head
    cache = Path(".cache/semantic_answer_v2") / head
    if args.out_root.exists() and any(args.out_root.glob("*/started.json")):
        raise ValueError("A semantic v2 baseline was already started; do not rerun")
    if out.exists() or cache.exists():
        raise ValueError("SHA-keyed baseline already attempted; do not overwrite")
    environment = frozen.runtime_environment(config, cache)
    initial_settle = await frozen.settle_preflight("imports_and_configuration")
    state = initial_settle["after"]
    services = frozen.service_preflight(config)
    client = QdrantClient(host="127.0.0.1", port=6333, timeout=120)
    guard = canonical.FrozenIndexGuard(
        client, canonical.load_contract(args.index_attestation)
    )
    before, records = guard.verify_before()
    historical, _ = frozen.snapshot(client, config["historical_collection"])
    frozen.verify_index(
        records, read("data/evals/retrieval/benchmark_v2/corpus.jsonl")
    )
    old_reference = json.loads(
        Path(
            "artifacts/evals/retrieval/benchmark_v2/baselines/"
            "54d31917c27263ea25ebf5d905caa0a4ee34f74f/manifest.json"
        ).read_text()
    )
    if historical != old_reference["historical_index_after"]:
        raise ValueError("Historical collection changed")
    schedule = sorted(cases, key=lambda case: case["id"])
    random.Random(config["order_seed"]).shuffle(schedule)
    catalog = load_numeric_catalog(frozen.DATA)
    from agents.planner.interactive_target_resolution import InteractivePlannerAgent
    from agents.orchestrator.agent_orchestrator import (
        aclose_orchestrator_runtime,
        run_multi_agent_orchestration,
    )

    planner = InteractivePlannerAgent(model=config["planner_model"], log_timing=False)
    final_settle = await frozen.settle_preflight(
        "index_verification_and_planner_setup"
    )
    out.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=False)
    manifest = {
        "launcher_contract": launcher,
        "launcher_contract_sha256": sha(frozen.CONTRACT),
        "preflight_settling": [initial_settle, final_settle],
        "implementation_sha": head,
        "production_identical_to": frozen.BASE,
        "started_at": frozen.now(),
        "dataset_sha256": sha(frozen.DATA / "queries.jsonl"),
        "optimization_manifest_sha256": sha(frozen.DATA / "optimization_manifest.json"),
        "quality_approval": approval,
        "controls_before": state,
        "service_preflight": services,
        "hardware": frozen.hardware(),
        "environment": environment,
        "schedule": [case["id"] for case in schedule],
        "dataset_composition": counts,
        "runtime_config": {
            key: value
            for key, value in config.items()
            if key not in {"judge", "audit", "runtime_policy", "base_runtime_sha"}
        },
        "historical_runtime_config_sha256": sha(
            "data/evals/semantic_answer/v1/evaluation_config.json"
        ),
        "python": os.sys.version.split()[0],
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("pytest", "requests", "qdrant-client", "langchain-ollama")
        },
        "index_before": before,
        "historical_index_before": historical,
        "index_origin": canonical.load_contract(args.index_attestation),
        "index_limitation": (
            "The historically fingerprinted surviving collection is evaluated; "
            "the deleted serialized build cache is not byte-reproducible."
        ),
        "policy": (
            "One sequential pass, unchanged runtime retries and 120s analyst timeout; "
            "no harness retries or clarification follow-ups; preserve all failures."
        ),
    }
    frozen.save(out / "started.json", manifest)
    rows = []
    captured = []
    evaluation_errors = []
    control_violations = []
    try:
        for index, case in enumerate(schedule):
            started = frozen.now()
            timer = time.perf_counter()
            before_state = frozen.controls()
            print(f'START {index + 1}/60 {case["id"]}', flush=True)
            try:
                output = await run_multi_agent_orchestration(
                    case["user_query"],
                    planner=planner,
                    analyst_model=config["analyst_model"],
                    tables_dir="data/evals/semantic_answer/v1/tables",
                    debug=False,
                    include_evidence_trace=True,
                )
            except Exception as exc:
                output = {
                    "ok": False,
                    "status": "harness_captured_runtime_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            elapsed = (time.perf_counter() - timer) * 1000
            after_state = frozen.controls()
            frozen.append(
                out / "raw_answers.jsonl",
                {
                    "case_id": case["id"],
                    "started_at": started,
                    "wall_ms": elapsed,
                    "controls_before": before_state,
                    "controls_after": after_state,
                    "output": output,
                },
            )
            captured.append(case["id"])
            for when, check in (("before", before_state), ("after", after_state)):
                try:
                    frozen.check_controls(check)
                except ValueError as exc:
                    control_violations.append(
                        {"case_id": case["id"], "when": when, "reason": str(exc)}
                    )
            try:
                row = frozen.deterministic_case(case, output, catalog)
                rows.append(row)
                frozen.append(out / "deterministic.jsonl", row)
            except Exception as exc:
                record = {
                    "case_id": case["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                evaluation_errors.append(record)
                frozen.append(out / "evaluation_errors.jsonl", record)
            print(
                f'END {case["id"]} {output.get("status")} {elapsed / 1000:.1f}s',
                flush=True,
            )
    finally:
        ending = {
            "finished_at": frozen.now(),
            "captured_cases": captured,
            "status": "complete" if len(captured) == len(cases) else "incomplete",
            "evaluation_errors": evaluation_errors,
            "control_violations": control_violations,
            "controls_after": frozen.controls(),
        }
        try:
            await aclose_orchestrator_runtime()
        except Exception as exc:
            ending["runtime_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
            response.raise_for_status()
            identities = {
                model["name"]: {
                    key: model.get(key) for key in ("name", "digest", "size")
                }
                for model in response.json()["models"]
                if model["name"] in services["model_identities"]
            }
            ending.update(
                model_identities_after=identities,
                model_identities_unchanged=(
                    identities == services["model_identities"]
                ),
            )
        except Exception as exc:
            ending["model_verification_error"] = f"{type(exc).__name__}: {exc}"
        close_index_guard(ending, guard, before)
        try:
            historical_after, _ = frozen.snapshot(
                client, config["historical_collection"]
            )
            ending["historical_index_after"] = historical_after
            ending["index_unchanged"] = (
                ending.get("index_unchanged") is True
                and historical_after == historical
            )
        except Exception as exc:
            previous = ending.get("index_verification_error")
            detail = f"{type(exc).__name__}: {exc}"
            ending["index_unchanged"] = False
            ending["index_verification_error"] = (
                f"{previous}; historical snapshot: {detail}"
                if previous
                else f"historical snapshot: {detail}"
            )
        valid = frozen.finalize_validity(
            ending, [case["id"] for case in cases], [row["case_id"] for row in rows]
        )
        frozen.save(out / "completion.json", ending)
        client.close()
        if valid:
            frozen.save(
                out / "deterministic_summary.json",
                frozen.deterministic_breakdowns(cases, rows),
            )
        frozen.save(
            out / "files_sha256.json",
            {
                path.name: sha(path)
                for path in sorted(out.iterdir())
                if path.is_file()
            },
        )


@contextmanager
def v7_adapter():
    """Temporarily route the unchanged v2.2 control shell through v2.3."""
    original = {
        "verify_opt_in": control.verify_opt_in,
        "verify_frozen_launcher": control.verify_frozen_launcher,
        "launcher": control.LAUNCHER,
        "integration_approval": control.INTEGRATION_APPROVAL,
        "run_once": frozen.run_once,
    }
    control.verify_opt_in = verify_opt_in
    control.verify_frozen_launcher = verify_frozen_launcher_v7
    control.LAUNCHER = LAUNCHER
    control.INTEGRATION_APPROVAL = INTEGRATION_APPROVAL
    frozen.run_once = run_once
    try:
        yield
    finally:
        control.verify_opt_in = original["verify_opt_in"]
        control.verify_frozen_launcher = original["verify_frozen_launcher"]
        control.LAUNCHER = original["launcher"]
        control.INTEGRATION_APPROVAL = original["integration_approval"]
        frozen.run_once = original["run_once"]


async def run(args):
    with v7_adapter():
        await control.run(args)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approval",
        type=Path,
        default=Path("docs/evals/semantic_answer_v2_quality_approval.json"),
    )
    parser.add_argument("--index-attestation", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workload-control-v2", required=True)
    parser.add_argument(
        "--integration-approval", type=Path, default=INTEGRATION_APPROVAL
    )
    args = parser.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
