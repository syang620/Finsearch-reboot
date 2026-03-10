#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.orchestrator import run_multi_agent_orchestration


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the agentic RAG pipeline over prompts from a JPMorgan-style prompt bank "
            "and save per-query answers plus traces."
        )
    )
    p.add_argument(
        "--input-file",
        default="artifacts/jpmorgan_chase_single_company_gold_answers_with_chunk_ids.json",
        help="Path to prompt JSON file containing a list of prompt items.",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/reports/jpmorgan_chase_agentic_rag",
        help="Directory to write results and traces.",
    )
    p.add_argument(
        "--analyst-model",
        default="qwen2.5-14b-instruct-1m",
        help="Analyst model used by orchestrator for final synthesis.",
    )
    p.add_argument(
        "--tables-dir",
        default="data/chunked",
        help="Tables/text chunk directory for downstream table hydration.",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Optional cap on number of prompts to process. 0 means process all.",
    )
    p.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="Optional 0-based start index in prompt list.",
    )
    p.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable detailed debug output from orchestrator/analyst.",
    )
    p.add_argument(
        "--skip-interrupted",
        action="store_true",
        help="Skip prompts that return interrupted status instead of treating as error.",
    )
    p.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0,
        help="Optional runtime limit in seconds for the full batch. 0 means no limit.",
    )
    return p


def _load_prompt_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = payload.get("items")
    if isinstance(items, list):
        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            prompt = item.get("prompt")
            if not isinstance(prompt, str):
                continue
            row = {
                "id": str(item.get("id") or item.get("qid") or item.get("question_id") or "").strip(),
                "prompt": prompt.strip(),
                "gold_answer": item.get("gold_answer", ""),
                "source_chunk_ids": item.get("source_chunk_ids") or [],
                "theme": item.get("theme", ""),
                "source_chunks": item.get("source_chunks", []),
                "mapping_note": item.get("mapping_note"),
            }
            if not row["id"]:
                row["id"] = f"item_{len(rows)+1}"
            rows.append(row)
        return rows

    if not isinstance(payload, list):
        raise ValueError("Input JSON must contain an 'items' array or be a list of prompt rows.")

    rows: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            continue
        row = {
            "id": str(item.get("id") or item.get("qid") or item.get("question_id") or f"item_{len(rows)+1}").strip(),
            "prompt": prompt.strip(),
            "gold_answer": item.get("gold_answer", ""),
            "source_chunk_ids": item.get("source_chunk_ids") or [],
            "theme": item.get("theme", ""),
            "source_chunks": item.get("source_chunks", []),
            "mapping_note": item.get("mapping_note"),
        }
        rows.append(row)
    return rows


def _read_payload(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict) and not isinstance(payload, list):
        raise ValueError("Prompt file must contain a JSON object or array.")
    return payload


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    return str(value)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _run_one(prompt_row: Dict[str, Any], *, analyst_model: str, tables_dir: str, debug: bool) -> Dict[str, Any]:
    query = prompt_row["prompt"]
    row_id = prompt_row["id"]
    start = time.perf_counter()

    run_output: Dict[str, Any] = {}
    try:
        run_output = await run_multi_agent_orchestration(
            query,
            analyst_model=analyst_model,
            tables_dir=tables_dir,
            debug=debug,
        )
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {
            "query_id": row_id,
            "prompt": query,
            "status": "error",
            "run_id": None,
            "elapsed_ms": elapsed_ms,
            "ok": False,
            "answer": "",
            "error": str(exc),
            "run_output": None,
        }

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    status = str(run_output.get("status") or "unknown").strip().lower()
    analyst = run_output.get("analyst") or {}
    answer = analyst.get("answer", "") if isinstance(analyst, dict) else ""

    return {
        "query_id": row_id,
        "prompt": query,
        "status": status,
        "run_id": run_output.get("run_id"),
        "elapsed_ms": elapsed_ms,
        "ok": bool(analyst.get("ok", False)) if isinstance(analyst, dict) else False,
        "answer": answer,
        "error": None,
        "run_output": run_output,
    }


async def _run_batch(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.perf_counter()

    source = Path(args.input_file)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    payload = _read_payload(source)
    rows = _load_prompt_rows(payload)

    if args.start_at:
        rows = rows[args.start_at :]
    if args.max_prompts > 0:
        rows = rows[: args.max_prompts]

    if not rows:
        raise ValueError("No valid prompt rows found in input file.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = out_dir / "prompt_rows.jsonl"
    results_path = out_dir / "agentic_rag_results.jsonl"
    traces_path = out_dir / "agentic_rag_traces.jsonl"
    full_path = out_dir / "agentic_rag_full_runs.jsonl"
    summary_path = out_dir / "agentic_rag_summary.json"

    _write_jsonl(prompts_path, [_to_jsonable(r) for r in rows])

    result_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []

    status_counts: Dict[str, int] = {}
    completed = 0

    for idx, row in enumerate(rows, start=1):
        item_debug = not args.no_debug
        current = await _run_one(
            row,
            analyst_model=args.analyst_model,
            tables_dir=args.tables_dir,
            debug=item_debug,
        )
        current.update(
            {
                "source_chunk_ids": row.get("source_chunk_ids") or [],
                "theme": row.get("theme") or "",
                "gold_answer": row.get("gold_answer") or "",
            }
        )

        if args.max_runtime_seconds > 0 and (time.perf_counter() - start) > args.max_runtime_seconds:
            raise TimeoutError(f"Batch exceeded max-runtime-seconds={args.max_runtime_seconds}")

        status = str(current.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "completed":
            completed += 1
        else:
            if not (args.skip_interrupted and status == "interrupted"):
                current["error"] = current.get("error") or f"non-completed-status:{status}"

        result_rows.append(
            {
                "query_id": current.get("query_id"),
                "prompt": current.get("prompt"),
                "theme": row.get("theme") or "",
                "source_chunk_ids": row.get("source_chunk_ids") or [],
                "status": status,
                "run_id": current.get("run_id"),
                "elapsed_ms": current.get("elapsed_ms", 0),
                "ok": current.get("ok", False),
                "answer": current.get("answer", ""),
                "error": current.get("error"),
                "analyst_ok": bool((current.get("run_output") or {}).get("analyst", {}).get("ok", False)
                                  if isinstance((current.get("run_output") or {}).get("analyst"), dict)
                                  else False),
                "has_retrieval": bool((current.get("run_output") or {}).get("retrieval", {})),
            }
        )

        run_output = current.get("run_output")
        if isinstance(run_output, dict):
            trace = run_output.get("orchestrator_trace") if isinstance(run_output.get("orchestrator_trace"), dict) else {}
            trace_rows.append(
                {
                    "query_id": current.get("query_id"),
                    "run_id": current.get("run_id"),
                    "prompt": current.get("prompt"),
                    "status": status,
                    "trace": _to_jsonable(trace),
                    "interrupt": _to_jsonable(run_output.get("interrupt", [])),
                    "planner_dump": _to_jsonable(run_output.get("planner", {})),
                    "retrieval_dump": _to_jsonable(run_output.get("retrieval", {})),
                    "planner_turn": _to_jsonable(run_output.get("planner_turn", {})),
                }
            )
            full_rows.append(_to_jsonable(run_output))
        else:
            full_rows.append({"query_id": current.get("query_id"), "error": current.get("error", "unknown")})

        print(f"[{idx}/{len(rows)}] {row['id']}: {status} run_id={current.get('run_id')} elapsed_ms={current.get('elapsed_ms')}")

    total_ms = int((time.perf_counter() - start) * 1000)

    summary = {
        "input_file": str(source.resolve()),
        "out_dir": str(out_dir.resolve()),
        "analyst_model": args.analyst_model,
        "tables_dir": args.tables_dir,
        "prompt_count": len(rows),
        "completed_count": completed,
        "status_counts": status_counts,
        "elapsed_ms": total_ms,
        "results": [r["query_id"] for r in result_rows],
    }

    _write_jsonl(results_path, result_rows)
    _write_jsonl(traces_path, trace_rows)
    _write_jsonl(full_path, full_rows)
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "summary": summary,
        "results_path": str(results_path),
        "traces_path": str(traces_path),
        "full_path": str(full_path),
        "summary_path": str(summary_path),
    }


def main() -> int:
    args = _build_parser().parse_args()
    report = asyncio.run(_run_batch(args))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
