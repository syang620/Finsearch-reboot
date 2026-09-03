from __future__ import annotations

from scripts import run_multi_agent_orchestration


def test_degraded_run_prints_usable_answer_and_exits_successfully(capsys) -> None:
    result = {
        "status": "degraded",
        "run_id": "run-degraded",
        "analyst": {"ok": True, "answer": "Usable answer with degraded coverage."},
        "degradation": {
            "active": True,
            "affected_lanes": ["kb"],
            "notice": "KB evidence coverage is incomplete.",
        },
        "orchestrator_trace": {"total_ms": 1250},
    }

    exit_code = run_multi_agent_orchestration._print_output(result, as_json=False)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "Run status: degraded",
        "KB evidence coverage is incomplete.",
        "Usable answer with degraded coverage.",
        "Total runtime: 1.25 s",
    ]
