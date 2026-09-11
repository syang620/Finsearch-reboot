import json
from pathlib import Path
from types import SimpleNamespace

from scripts.diagnostics import rehearse_semantic_workload_qualification_v3 as rehearsal


class Awake:
    def terminate(self):
        pass

    def wait(self, timeout):
        return 0


class Monitor:
    def __init__(self, raw, *args, **kwargs):
        self.raw = Path(raw)
        self.thread = None

    def start(self):
        self.thread = object()
        self.raw.write_text(
            json.dumps({"type": "header"})
            + "\n"
            + json.dumps(
                {
                    "type": "sample",
                    "index": 0,
                    "selected_policy_result": {
                        "hard_reasons": ["browser"],
                        "cpu_violation": False,
                    },
                    "terminal_only_result": {"valid": False},
                }
            )
            + "\n"
            + json.dumps({"type": "footer"})
            + "\n"
        )

    def stop(self):
        return {
            "sample_count": 1,
            "awake_protection_active_through_final_sample": True,
            "cadence_within_frozen_tolerance": True,
            "monitor_error": None,
        }


def arguments(tmp_path, require=False):
    return SimpleNamespace(
        duration_seconds=10,
        out_root=tmp_path,
        require_controlled_latency=require,
    )


def setup(monkeypatch):
    monkeypatch.setattr(rehearsal, "verify_frozen_inputs", lambda: None)
    monkeypatch.setattr(rehearsal.subprocess, "Popen", lambda *args, **kwargs: Awake())
    monkeypatch.setattr(rehearsal, "WorkloadControlV2Monitor", Monitor)
    values = iter([0, 10, 11])
    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(rehearsal.time, "sleep", lambda seconds: None)


def test_completed_unqualified_rehearsal_is_observational_success(tmp_path, monkeypatch):
    setup(monkeypatch)
    assert rehearsal.run(arguments(tmp_path)) == 0
    targets = list(tmp_path.iterdir())
    assert len(targets) == 1
    summary_path = targets[0] / "rehearsal_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["capture_complete"]
    assert summary["controlled_latency"] == {
        "eligible": False,
        "reasons": ["browser_present", "active_supervision_ui"],
    }
    assert summary["execution_authority"] is False
    assert summary_path.stat().st_mode & 0o777 == 0o600
    assert (targets[0] / "workload_qualification_v3.jsonl").stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.rglob("consumed.json"))


def test_required_controlled_latency_uses_distinct_exit_code(tmp_path, monkeypatch):
    setup(monkeypatch)
    assert rehearsal.run(arguments(tmp_path, require=True)) == 2


def test_duration_is_bounded(tmp_path):
    args = arguments(tmp_path)
    args.duration_seconds = 9
    try:
        rehearsal.run(args)
    except ValueError as exc:
        assert str(exc) == "duration must be between 10 and 3600 seconds"
    else:
        raise AssertionError("short rehearsal should fail")


def test_output_root_symlink_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    try:
        rehearsal.private_directory(link)
    except ValueError as exc:
        assert str(exc) == "rehearsal output root must be an owned directory"
    else:
        raise AssertionError("symlink output root should fail")
