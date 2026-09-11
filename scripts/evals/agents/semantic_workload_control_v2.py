"""Runtime adapter for the frozen semantic workload-control-v2 contract.

This module records control evidence only.  It does not call the semantic system,
change case scheduling, or alter benchmark outputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from scripts.diagnostics.run_workload_control_v2_calibration import capture_processes
from scripts.diagnostics.workload_control_v2 import CandidateEvaluator, load_preregistration
from scripts.evals.retrieval.run_benchmark_v3 import controls


POLICY = "B_CONSECUTIVE_10"
INTERVAL_SECONDS = 1.0
CADENCE_TOLERANCE_SECONDS = 0.25


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(stream, record):
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


def terminal_only_result(sample):
    """Match the reviewed S2 rule: permit only inert launchd crash handlers."""
    supervision = [
        process for process in sample.get("processes", [])
        if process.get("category") == "run_supervision_ui_tooling"
    ]
    active = []
    inert = []
    for process in supervision:
        is_handler = Path(str(process.get("executable") or "")).name == "browser_crashpad_handler"
        launchd_owned = any(
            str(ancestor.get("executable") or "").lower() == "/sbin/launchd"
            for ancestor in process.get("ancestors", [])
            if ancestor.get("pid") != process.get("pid")
        )
        if is_handler and float(process.get("cpu", 0.0)) == 0.0 and launchd_owned:
            inert.append(process)
        else:
            active.append(process)
    return {
        "valid": not active,
        "active_supervision_processes": active,
        "retained_inert_crash_handler_count": len(inert),
    }


class WorkloadControlV2Monitor:
    """Continuously apply the reviewed B10 rule and retain every raw sample."""

    def __init__(self, raw_path, preregistration_path, contract_path, awake_process, provenance=None):
        self.raw_path = Path(raw_path)
        self.preregistration_path = Path(preregistration_path)
        self.contract_path = Path(contract_path)
        self.awake_process = awake_process
        self.provenance = dict(provenance or {})
        self.preregistration = load_preregistration(self.preregistration_path)
        self.evaluator = CandidateEvaluator(self.preregistration)
        self.stop_event = threading.Event()
        self.started_event = threading.Event()
        self.thread = None
        self.samples = 0
        self.trigger = None
        self.error = None
        self.max_cadence_error_seconds = 0.0
        self.awake_active_all_samples = True
        self.inert_crash_handler_records = 0
        self.started_at = None
        self.finished_at = None

    def start(self):
        if self.thread is not None or self.raw_path.exists():
            raise FileExistsError(self.raw_path)
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="semantic-workload-control-v2", daemon=True)
        self.thread.start()
        if not self.started_event.wait(timeout=10):
            raise RuntimeError("Workload-control-v2 monitor did not start")
        if self.error:
            raise RuntimeError("Workload-control-v2 monitor failed at startup") from self.error

    def _run(self):
        started = time.monotonic()
        self.started_at = now()
        try:
            with self.raw_path.open("x") as stream:
                _write(stream, {
                    "type": "header",
                    "started_at": self.started_at,
                    "selected_policy": POLICY,
                    "sample_interval_seconds": INTERVAL_SECONDS,
                    "preregistration_sha256": sha256(self.preregistration_path),
                    "contract_sha256": sha256(self.contract_path),
                    "integration_provenance": self.provenance,
                    "observer_pid": os.getpid(),
                    "awake_pid": self.awake_process.pid,
                    "policy": "Continuous control evidence only; semantic behavior is owned by the frozen launcher.",
                })
                self.started_event.set()
                index = 0
                while not self.stop_event.is_set():
                    elapsed = time.monotonic() - started
                    cadence_error = abs(elapsed - index * INTERVAL_SECONDS)
                    self.max_cadence_error_seconds = max(self.max_cadence_error_seconds, cadence_error)
                    frozen = controls()
                    sample = {
                        "type": "sample",
                        "index": index,
                        "observed_at": now(),
                        "elapsed_seconds": elapsed,
                        "ac_power": frozen.get("ac_power"),
                        "low_power_mode": frozen.get("low_power_mode"),
                        "browser_process_count": frozen.get("browser_process_count", 0),
                        "frozen_v1_cpu_violation": bool(frozen.get("heavy_non_model_processes")),
                        "frozen_v1_heavy_processes": frozen.get("heavy_non_model_processes", []),
                        "awake_protection": {
                            "pid": self.awake_process.pid,
                            "active": self.awake_process.poll() is None,
                        },
                        "processes": capture_processes(os.getpid(), set()),
                    }
                    classified, candidates = self.evaluator.evaluate(sample)
                    selected = candidates[POLICY]
                    terminal = terminal_only_result(classified)
                    classified["selected_policy_result"] = selected
                    classified["terminal_only_result"] = terminal
                    _write(stream, classified)
                    self.samples += 1
                    self.awake_active_all_samples &= classified["awake_protection"]["active"]
                    self.inert_crash_handler_records += terminal["retained_inert_crash_handler_count"]
                    if (selected["invalid"] or not terminal["valid"]) and self.trigger is None:
                        self.trigger = {
                            "sample_index": index,
                            "observed_at": classified["observed_at"],
                            "terminal_only_violation": not terminal["valid"],
                            **selected,
                        }
                    index += 1
                    self.stop_event.wait(max(0, started + index * INTERVAL_SECONDS - time.monotonic()))
                self.finished_at = now()
                _write(stream, {
                    "type": "footer",
                    "finished_at": self.finished_at,
                    "sample_count": self.samples,
                    "selected_policy": POLICY,
                    "first_violation": self.trigger,
                    "retained_inert_crash_handler_records": self.inert_crash_handler_records,
                    "awake_protection_active_through_final_sample": self.awake_active_all_samples,
                    "awake_protection_returncode_before_cleanup": self.awake_process.poll(),
                    "max_cadence_error_seconds": self.max_cadence_error_seconds,
                })
        except BaseException as exc:
            self.error = exc
            self.started_event.set()

    def stop(self):
        if self.thread is None:
            raise RuntimeError("Workload-control-v2 monitor was not started")
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Workload-control-v2 monitor did not stop")
        summary = {
            "selected_policy": POLICY,
            "sample_count": self.samples,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "first_violation": self.trigger,
            "retained_inert_crash_handler_records": self.inert_crash_handler_records,
            "awake_protection_active_through_final_sample": self.awake_active_all_samples,
            "max_cadence_error_seconds": self.max_cadence_error_seconds,
            "cadence_within_frozen_tolerance": (
                self.samples > 0 and self.max_cadence_error_seconds <= CADENCE_TOLERANCE_SECONDS
            ),
            "monitor_error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
        }
        summary["valid"] = bool(
            summary["sample_count"]
            and summary["first_violation"] is None
            and summary["awake_protection_active_through_final_sample"]
            and summary["cadence_within_frozen_tolerance"]
            and summary["monitor_error"] is None
        )
        summary["raw_sha256"] = sha256(self.raw_path) if self.raw_path.exists() else None
        return summary
