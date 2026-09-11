"""Split semantic answer-quality and controlled-latency qualification.

This inactive adapter reuses the frozen workload-control-v2 observer and B10
classifier.  Ambient workload never stops case capture.  It can disqualify
controlled latency without disqualifying otherwise complete answer evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess

from evals.semantic_dataset_v2 import load_dataset, sha
from scripts.evals.agents import run_semantic_baseline_v2_1 as frozen
from scripts.evals.agents import run_semantic_baseline_v2_2 as legacy
from scripts.evals.agents.semantic_workload_control_v2 import (
    POLICY as PERFORMANCE_POLICY,
    WorkloadControlV2Monitor,
)


POLICY = "SPLIT_QUALITY_LATENCY_V1"
SCHEMA_VERSION = "3.0.0"
RAW_NAME = "workload_qualification_v3.jsonl"
COMPLETION_NAME = "workload_qualification_v3_completion.json"
MANIFEST_NAME = "workload_qualification_v3_files_sha256.json"
PERFORMANCE_ADAPTER_SHA256 = (
    "7fcef1b2c66fcb0226ee10f0036f6166203d0ee128be459d7fa4ffbd4dd62eb5"
)


def _append_unique(values, value):
    if value not in values:
        values.append(value)


def _completion(out):
    path = Path(out) / "completion.json"
    if not path.exists():
        return {
            "status": "incomplete",
            "invalidity_reasons": ["missing_frozen_completion"],
            "captured_cases": [],
        }
    return json.loads(path.read_text())


def analyze_raw(path):
    """Return bounded aggregate facts from the closed v2-compatible JSONL."""
    observations = {
        "sample_count": 0,
        "browser_samples": 0,
        "active_supervision_samples": 0,
        "sustained_external_cpu_samples": 0,
        "ac_power_violation_samples": 0,
        "low_power_mode_violation_samples": 0,
        "header_present": False,
        "footer_present": False,
        "header_count": 0,
        "footer_count": 0,
        "indices_contiguous": False,
        "framing_valid": False,
    }
    indices = []
    record_types = []
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            kind = record.get("type")
            record_types.append(kind)
            if kind == "header":
                observations["header_present"] = True
                observations["header_count"] += 1
            elif kind == "footer":
                observations["footer_present"] = True
                observations["footer_count"] += 1
            elif kind == "sample":
                observations["sample_count"] += 1
                indices.append(record.get("index"))
                selected = record.get("selected_policy_result", {})
                hard = selected.get("hard_reasons", [])
                observations["browser_samples"] += int("browser" in hard)
                observations["ac_power_violation_samples"] += int("ac_power" in hard)
                observations["low_power_mode_violation_samples"] += int(
                    "low_power_mode" in hard
                )
                observations["sustained_external_cpu_samples"] += int(
                    selected.get("cpu_violation") is True
                )
                observations["active_supervision_samples"] += int(
                    record.get("terminal_only_result", {}).get("valid") is False
                )
    observations["indices_contiguous"] = indices == list(range(len(indices)))
    observations["framing_valid"] = bool(
        record_types
        and record_types[0] == "header"
        and record_types[-1] == "footer"
    )
    return observations


def quality_reasons(completion, monitor_summary, observations, operation_error=None):
    reasons = [
        reason
        for reason in completion.get("invalidity_reasons", [])
        if reason != "workload_qualification_v3_pending"
    ]
    if completion.get("capture_complete") is not True:
        _append_unique(reasons, "incomplete_or_duplicate_capture")
    if completion.get("evaluation_errors"):
        _append_unique(reasons, "evaluation_errors")
    for field in ("model_identities_unchanged", "index_unchanged"):
        if completion.get(field) is not True:
            _append_unique(reasons, field + "_not_verified")
    for field in (
        "model_verification_error",
        "index_verification_error",
        "runtime_cleanup_error",
    ):
        if completion.get(field):
            _append_unique(reasons, field)
    if completion.get("control_violations"):
        _append_unique(reasons, "power_control_violation")
    if monitor_summary.get("monitor_error"):
        _append_unique(reasons, "monitor_capture_incomplete")
    if (
        observations.get("header_count") != 1
        or observations.get("footer_count") != 1
        or observations.get("sample_count", 0) <= 0
        or observations.get("indices_contiguous") is not True
        or observations.get("framing_valid") is not True
        or observations.get("sample_count") != monitor_summary.get("sample_count")
    ):
        _append_unique(reasons, "monitor_capture_incomplete")
    if not monitor_summary.get("awake_protection_active_through_final_sample"):
        _append_unique(reasons, "awake_protection_failed")
    if observations.get("ac_power_violation_samples"):
        _append_unique(reasons, "power_control_violation")
    if observations.get("low_power_mode_violation_samples"):
        _append_unique(reasons, "power_control_violation")
    if operation_error is not None:
        _append_unique(reasons, "execution_or_finalization_failed")
    return reasons


def latency_reasons(answer_requirements_met, monitor_summary, observations):
    reasons = []
    if not answer_requirements_met:
        reasons.append("answer_quality_ineligible")
    if observations.get("browser_samples"):
        reasons.append("browser_present")
    if observations.get("active_supervision_samples"):
        reasons.append("active_supervision_ui")
    if observations.get("sustained_external_cpu_samples"):
        reasons.append("sustained_external_cpu")
    if (
        observations.get("ac_power_violation_samples")
        or observations.get("low_power_mode_violation_samples")
    ):
        reasons.append("power_control_violation")
    if monitor_summary.get("cadence_within_frozen_tolerance") is not True:
        reasons.append("sampling_cadence_out_of_tolerance")
    if monitor_summary.get("monitor_error"):
        reasons.append("monitor_error")
    return reasons


def qualification_record(
    completion,
    monitor_summary,
    observations,
    provenance,
    operation_error=None,
):
    answer_reasons = quality_reasons(
        completion, monitor_summary, observations, operation_error
    )
    answer_requirements_met = not answer_reasons
    performance_reasons = latency_reasons(
        answer_requirements_met, monitor_summary, observations
    )
    latency_requirements_met = answer_requirements_met and not performance_reasons
    capture_complete = completion.get("capture_complete") is True
    status = (
        "complete"
        if (
            capture_complete
            and operation_error is None
            and "monitor_capture_incomplete" not in answer_reasons
        )
        else "incomplete_diagnostic"
    )
    record = {
        **completion,
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY,
        "status": status,
        "eligibility": {
            "answer_quality": {
                "requirements_met": answer_requirements_met,
                "eligible": False,
                "activation_required": answer_requirements_met,
                "reasons": answer_reasons,
            },
            "controlled_latency": {
                "requirements_met": latency_requirements_met,
                "eligible": False,
                "activation_required": latency_requirements_met,
                "reasons": performance_reasons,
            },
        },
        "workload_observation": {
            **monitor_summary,
            "legacy_performance_policy": PERFORMANCE_POLICY,
            "aggregates": observations,
            "provenance": provenance,
        },
        "claim_policy": {
            "answer_quality": (
                "Publication requires a separate reviewed activation artifact after "
                "eligibility.answer_quality.requirements_met is true. "
                "Captured analyst, model, and tool failures remain measured outcomes."
            ),
            "latency": (
                "Timings remain diagnostic until controlled-latency requirements are "
                "met and a separate reviewed activation artifact exists."
            ),
        },
        "supersedes_operational_eligibility_in": "completion.json",
    }
    record.pop("official_baseline_eligible", None)
    record.pop("invalidity_reasons", None)
    if operation_error is not None:
        record["operation_error"] = {
            "type": type(operation_error).__name__,
            "message": "Operation or artifact finalization failed",
        }
    return record


def provisional_validity(ending, expected_ids, evaluated_ids):
    reasons = []
    captured = ending["captured_cases"]
    if len(captured) != len(set(captured)) or set(captured) != set(expected_ids):
        reasons.append("incomplete_or_duplicate_capture")
    if (
        len(evaluated_ids) != len(set(evaluated_ids))
        or set(evaluated_ids) != set(expected_ids)
    ):
        reasons.append("incomplete_or_duplicate_evaluation")
    if ending.get("evaluation_errors"):
        reasons.append("evaluation_errors")
    if ending.get("control_violations"):
        reasons.append("power_control_violation")
    for field in ("model_identities_unchanged", "index_unchanged"):
        if ending.get(field) is not True:
            reasons.append(field + "_not_verified")
    for field in (
        "model_verification_error",
        "index_verification_error",
        "runtime_cleanup_error",
    ):
        if ending.get(field):
            reasons.append(field)
    ending.update(
        status="pending_workload_qualification_v3",
        official_baseline_eligible=False,
        invalidity_reasons=[*reasons, "workload_qualification_v3_pending"],
        capture_complete=(
            len(captured) == len(expected_ids) and set(captured) == set(expected_ids)
        ),
    )
    return False


@contextmanager
def frozen_launcher_adapter(start_monitor, verify_launcher):
    """Observe the full run without aborting on ambient workload findings."""
    original = {
        "verify_launcher": frozen.verify_launcher,
        "settle_preflight": frozen.settle_preflight,
        "service_preflight": frozen.service_preflight,
        "check_controls": frozen.check_controls,
        "finalize_validity": frozen.finalize_validity,
    }
    settled = original["settle_preflight"]

    async def settle(stage):
        record = await settled(stage)
        if stage == "index_verification_and_planner_setup":
            start_monitor()
        return record

    frozen.verify_launcher = verify_launcher
    frozen.settle_preflight = settle
    frozen.service_preflight = lambda config: legacy.verified_service_preflight(
        lambda: original["service_preflight"](config)
    )
    frozen.check_controls = legacy.hard_control_only
    frozen.finalize_validity = provisional_validity
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(frozen, name, value)


def copy_raw_evidence(raw, out):
    destination = Path(out) / RAW_NAME
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(raw, destination)
    legacy.sync_file_and_parent(destination)
    return destination


def prepare_answer_artifacts(out):
    cases, _ = load_dataset(frozen.DATA)
    rows = [
        json.loads(line)
        for line in (Path(out) / "deterministic.jsonl").read_text().splitlines()
    ]
    summary_path = Path(out) / "deterministic_summary.json"
    frozen.save(summary_path, frozen.deterministic_breakdowns(cases, rows))
    legacy.sync_file_and_parent(summary_path)


def publish_manifest(out):
    out = Path(out)
    path = out / MANIFEST_NAME
    manifest = {
        member.name: sha(member)
        for member in sorted(out.iterdir())
        if member.is_file() and member.name not in {MANIFEST_NAME, COMPLETION_NAME}
    }
    frozen.save(path, manifest)
    for member in manifest:
        legacy.sync_file_and_parent(out / member)
    legacy.sync_file_and_parent(path)


def publish_authoritative(path, record):
    """Exclusively publish the split-requirements record without a generic flag."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        published = True
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException as original_error:
        if published:
            cleanup_error = None
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_error = exc
                fallback = path.with_name(path.name + ".ineligible.tmp")
                try:
                    failed = json.loads(json.dumps(record))
                    failed["status"] = "incomplete_diagnostic"
                    for dimension in failed.get("eligibility", {}).values():
                        dimension["requirements_met"] = False
                        dimension["eligible"] = False
                        dimension["activation_required"] = False
                        _append_unique(
                            dimension.setdefault("reasons", []),
                            "authoritative_publication_failed",
                        )
                    descriptor = os.open(
                        fallback, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    with os.fdopen(descriptor, "w") as stream:
                        json.dump(
                            failed,
                            stream,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=False,
                        )
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(fallback, path)
                except BaseException as replacement_error:
                    cleanup_error = replacement_error
            try:
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise original_error from cleanup_error
        raise original_error
    finally:
        for candidate in (
            temporary,
            path.with_name(path.name + ".ineligible.tmp"),
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


async def run(args, head, review, run_once, verify_launcher, contract_path):
    out = args.out_root / head
    observation_root = Path(".cache/semantic_workload_qualification_v3") / head
    raw = observation_root / RAW_NAME
    awake = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    monitor = WorkloadControlV2Monitor(
        raw,
        legacy.PREREGISTRATION,
        legacy.CONTROL_CONTRACT,
        awake,
        provenance={
            "implementation_sha": head,
            "split_policy": POLICY,
            "contract_sha256": sha(contract_path),
            "performance_adapter_sha256": PERFORMANCE_ADAPTER_SHA256,
            "integration_review": review,
        },
    )
    started = False

    def start_monitor():
        nonlocal started
        monitor.start()
        started = True

    operation_error = None
    summary = None
    try:
        with frozen_launcher_adapter(start_monitor, verify_launcher):
            await run_once(args)
    except BaseException as exc:
        operation_error = exc
    if started:
        try:
            summary = monitor.stop()
        except BaseException as exc:
            operation_error = operation_error or exc
            summary = {
                "selected_policy": PERFORMANCE_POLICY,
                "sample_count": 0,
                "awake_protection_active_through_final_sample": False,
                "cadence_within_frozen_tolerance": False,
                "monitor_error": f"{type(exc).__name__}: {exc}",
            }
    try:
        awake.terminate()
        awake.wait(timeout=10)
    except BaseException as exc:
        operation_error = operation_error or exc

    if summary is None or not out.exists() or not raw.exists():
        if operation_error is not None:
            raise operation_error
        raise RuntimeError("Workload qualification did not reach artifact closure")

    copied = None
    observations = {
        "sample_count": 0,
        "header_present": False,
        "footer_present": False,
    }
    try:
        copied = copy_raw_evidence(raw, out)
        observations = analyze_raw(copied)
        completion = _completion(out)
        provisional = qualification_record(
            completion, summary, observations, review, operation_error
        )
        if provisional["eligibility"]["answer_quality"]["requirements_met"]:
            prepare_answer_artifacts(out)
        publish_manifest(out)
    except BaseException as exc:
        operation_error = operation_error or exc

    completion = _completion(out)
    final = qualification_record(
        completion, summary, observations, review, operation_error
    )
    if copied is not None:
        final["workload_observation"]["raw_sha256"] = sha(copied)
    publish_authoritative(out / COMPLETION_NAME, final)
    if operation_error is not None:
        raise operation_error
