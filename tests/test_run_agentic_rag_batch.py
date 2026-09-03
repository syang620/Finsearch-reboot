from __future__ import annotations

import argparse
import asyncio
import json

from scripts import run_agentic_rag_batch


def test_completed_and_degraded_runs_are_serialized_as_success(tmp_path, monkeypatch) -> None:
    statuses = {
        "completed": "completed",
        "degraded": "degraded",
        "failed": "failed",
        "interrupted": "interrupted",
    }
    input_path = tmp_path / "prompts.json"
    input_path.write_text(
        json.dumps(
            {
                "items": [
                    {"id": row_id, "prompt": f"prompt for {row_id}"}
                    for row_id in statuses
                ]
            }
        ),
        encoding="utf-8",
    )

    async def fake_run_one(prompt_row, *, analyst_model, tables_dir, debug):
        del analyst_model, tables_dir, debug
        status = statuses[prompt_row["id"]]
        ok = status in {"completed", "degraded"}
        return {
            "query_id": prompt_row["id"],
            "prompt": prompt_row["prompt"],
            "status": status,
            "run_id": f"run-{prompt_row['id']}",
            "elapsed_ms": 1,
            "ok": ok,
            "answer": "answer" if ok else "",
            "error": None,
            "run_output": {
                "status": status,
                "analyst": {"ok": ok, "answer": "answer" if ok else ""},
            },
        }

    monkeypatch.setattr(run_agentic_rag_batch, "_run_one", fake_run_one)
    args = argparse.Namespace(
        input_file=str(input_path),
        out_dir=str(tmp_path / "output"),
        analyst_model="test-model",
        tables_dir="test-tables",
        max_prompts=0,
        start_at=0,
        no_debug=True,
        skip_interrupted=False,
        max_runtime_seconds=0,
    )

    report = asyncio.run(run_agentic_rag_batch._run_batch(args))

    assert report["summary"]["completed_count"] == 2
    assert report["summary"]["status_counts"] == {
        "completed": 1,
        "degraded": 1,
        "failed": 1,
        "interrupted": 1,
    }

    result_rows = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (tmp_path / "output" / "agentic_rag_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert result_rows["completed"]["ok"] is True
    assert result_rows["completed"]["error"] is None
    assert result_rows["degraded"]["ok"] is True
    assert result_rows["degraded"]["error"] is None
    assert result_rows["failed"]["ok"] is False
    assert result_rows["failed"]["error"] == "non-completed-status:failed"
    assert result_rows["interrupted"]["ok"] is False
    assert result_rows["interrupted"]["error"] == "non-completed-status:interrupted"
