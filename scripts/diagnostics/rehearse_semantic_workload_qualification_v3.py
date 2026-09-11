"""Observe workload qualification without running semantic benchmark cases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import time
import uuid

from evals.semantic_dataset_v2 import sha
from scripts.evals.agents import run_semantic_baseline_v2_1 as frozen
from scripts.evals.agents import run_semantic_baseline_v2_2 as legacy
from scripts.evals.agents.semantic_workload_control_v2 import WorkloadControlV2Monitor
from scripts.evals.agents import semantic_workload_qualification_v3 as qualification


DEFAULT_ROOT = (
    Path.home() / ".local/share/finsearch/semantic-qualification/rehearsals"
)
CONTRACT = Path("docs/evals/workload_qualification_v3_contract.json")
QUALIFICATION_ADAPTER = Path(
    "scripts/evals/agents/semantic_workload_qualification_v3.py"
)
REHEARSAL = Path("scripts/diagnostics/rehearse_semantic_workload_qualification_v3.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def private_directory(root):
    root = Path(root).expanduser().absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("rehearsal output root must be an owned directory")
    root.chmod(0o700)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / f"{name}-{uuid.uuid4().hex[:8]}"
    target.mkdir(mode=0o700)
    return target


def write_private(path, record):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_frozen_inputs():
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
        if sha(path) != digest:
            raise ValueError(f"Frozen workload-control-v2 identity changed: {path}")
    contract = json.loads(CONTRACT.read_text())
    if (
        contract.get("version") != qualification.SCHEMA_VERSION
        or contract.get("selected_policy") != qualification.POLICY
        or contract.get("performance_policy") != qualification.PERFORMANCE_POLICY
    ):
        raise ValueError("Workload-qualification-v3 contract mismatch")
    frozen.clean_checkout()


def run(args):
    if not 10 <= args.duration_seconds <= 3600:
        raise ValueError("duration must be between 10 and 3600 seconds")
    verify_frozen_inputs()
    implementation_sha = legacy.git("rev-parse", "HEAD")
    target = private_directory(args.out_root)
    raw = target / qualification.RAW_NAME
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    monitor = WorkloadControlV2Monitor(
        raw,
        legacy.PREREGISTRATION,
        legacy.CONTROL_CONTRACT,
        awake,
        provenance={
            "rehearsal": True,
            "split_policy": qualification.POLICY,
            "contract_sha256": sha(CONTRACT),
            "performance_adapter_sha256": sha(legacy.ADAPTER),
            "service_provenance_sha256": sha(legacy.FROZEN_PROVENANCE),
            "qualification_adapter_sha256": sha(QUALIFICATION_ADAPTER),
            "rehearsal_sha256": sha(REHEARSAL),
            "implementation_sha": implementation_sha,
        },
    )
    technical_error = None
    summary = None
    started_at = now()
    try:
        monitor.start()
        deadline = time.monotonic() + args.duration_seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    except BaseException as exc:
        technical_error = exc
    finally:
        if monitor.thread is not None:
            try:
                summary = monitor.stop()
            except BaseException as exc:
                technical_error = technical_error or exc
        try:
            awake.terminate()
            awake.wait(timeout=10)
        except BaseException as exc:
            technical_error = technical_error or exc
    if raw.exists():
        raw.chmod(0o600)
    if summary is None:
        summary = {
            "sample_count": 0,
            "awake_protection_active_through_final_sample": False,
            "cadence_within_frozen_tolerance": False,
            "monitor_error": (
                f"{type(technical_error).__name__}: {technical_error}"
                if technical_error
                else "monitor did not produce a summary"
            ),
        }
    try:
        observations = qualification.analyze_raw(raw) if raw.exists() else {}
    except BaseException as exc:
        technical_error = technical_error or exc
        observations = {}
    observations = {
        "sample_count": 0,
        "header_present": False,
        "footer_present": False,
        "header_count": 0,
        "footer_count": 0,
        "indices_contiguous": False,
        "framing_valid": False,
        **observations,
    }
    capture_complete = bool(
        technical_error is None
        and summary.get("monitor_error") is None
        and observations.get("header_count") == 1
        and observations.get("footer_count") == 1
        and observations.get("sample_count", 0) > 0
        and observations.get("indices_contiguous") is True
        and observations.get("framing_valid") is True
        and observations.get("sample_count") == summary.get("sample_count")
        and summary.get("awake_protection_active_through_final_sample") is True
    )
    answer_requirements_met = bool(
        capture_complete
        and not observations.get("ac_power_violation_samples")
        and not observations.get("low_power_mode_violation_samples")
    )
    performance_reasons = qualification.latency_reasons(
        answer_requirements_met, summary, observations
    )
    record = {
        "schema_version": qualification.SCHEMA_VERSION,
        "policy": qualification.POLICY,
        "status": "complete" if capture_complete else "failed",
        "started_at": started_at,
        "finished_at": now(),
        "requested_duration_seconds": args.duration_seconds,
        "capture_complete": capture_complete,
        "controlled_latency": {
            "requirements_met": answer_requirements_met and not performance_reasons,
            "eligible": False,
            "activation_required": answer_requirements_met and not performance_reasons,
            "reasons": performance_reasons,
        },
        "workload_observation": {
            **summary,
            "aggregates": observations,
            "raw_sha256": sha(raw) if raw.exists() else None,
        },
        "contract_sha256": sha(CONTRACT),
        "performance_adapter_sha256": sha(legacy.ADAPTER),
        "service_provenance_sha256": sha(legacy.FROZEN_PROVENANCE),
        "qualification_adapter_sha256": sha(QUALIFICATION_ADAPTER),
        "rehearsal_sha256": sha(REHEARSAL),
        "implementation_sha": implementation_sha,
        "execution_authority": False,
    }
    if technical_error is not None:
        record["technical_error"] = {
            "type": type(technical_error).__name__,
            "message": str(technical_error),
        }
    summary_path = target / "rehearsal_summary.json"
    write_private(summary_path, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "controlled_latency": record["controlled_latency"],
                "sample_count": summary.get("sample_count"),
                "output": str(target),
            }
        )
    )
    if not capture_complete:
        return 1
    if args.require_controlled_latency and performance_reasons:
        return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--require-controlled-latency", action="store_true")
    args = parser.parse_args(argv)
    previous = os.umask(0o077)
    try:
        return run(args)
    finally:
        os.umask(previous)


if __name__ == "__main__":
    raise SystemExit(main())
