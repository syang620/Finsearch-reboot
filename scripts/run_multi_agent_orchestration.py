#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.orchestrator import run_multi_agent_orchestration, resume_multi_agent_orchestration


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run planner -> retrieval -> analyst orchestration from the CLI."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--query",
        help="Natural language query to run through the full orchestration.",
    )
    mode.add_argument(
        "--resume-run-id",
        help="Resume a previously interrupted run using this run_id.",
    )

    parser.add_argument("--analyst-model", default="qwen2.5-14b-instruct-1m", help="Analyst model for the final agent.")
    parser.add_argument("--tables-dir", default="data/chunked", help="Directory containing chunked table/text files.")
    parser.add_argument(
        "--answer",
        action="append",
        default=[],
        help="Clarification answer. Can be used multiple times (one per clarification question).",
    )
    parser.add_argument(
        "--answers-json",
        default="",
        help="Optional JSON array of clarification answers, used after --resume or when interruptions occur.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable detailed debug output from orchestrator/analyst.",
    )
    parser.add_argument("--json", action="store_true", help="Emit full JSON output.")
    return parser


def _normalize_prefilled_answers(*, answers: List[str], answers_json: str) -> List[str]:
    normalized: List[str] = []
    if answers:
        normalized.extend([str(a) for a in answers if str(a).strip() != ""])

    if answers_json:
        try:
            raw = json.loads(answers_json)
        except Exception as exc:
            raise ValueError(f"--answers-json must be a JSON array: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("--answers-json must parse to a JSON array.")
        normalized.extend([str(x) for x in raw if str(x).strip() != ""])

    return normalized


def _extract_prompt_questions(interrupt: dict[str, Any]) -> List[str]:
    value = interrupt.get("value") if isinstance(interrupt, dict) else None
    if not isinstance(value, dict):
        return []

    request = value.get("clarification_request")
    if not isinstance(request, dict):
        return []

    questions = request.get("questions")
    if isinstance(questions, list):
        return [str(q) for q in questions if str(q).strip()]
    if request.get("question"):
        return [str(request["question"])]
    return []


def _prompt_for_resume_answers(interrupts: list[dict[str, Any]], answer_queue: deque[str]) -> List[str]:
    if not interrupts:
        return []

    if not sys.stdin.isatty():
        raise RuntimeError("Run is waiting for clarification answers, but stdin is not interactive.")

    answers: List[str] = []
    for interrupt in interrupts:
        request = interrupt.get("value") or {}
        request = request.get("clarification_request") if isinstance(request, dict) else {}
        reason = request.get("reason") if isinstance(request, dict) else None

        if reason:
            print(f"Clarification required: {reason}")
            print("")

        questions = _extract_prompt_questions(interrupt)
        if not questions:
            questions = ["Please provide a clarification to continue."]

        for q in questions:
            if answer_queue:
                answers.append(answer_queue.popleft())
                continue
            try:
                ans = input(f"{q}\\n> ")
            except EOFError:
                raise RuntimeError("No clarification answer provided.")
            answers.append(ans.strip())

    return answers


async def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    prefilled_answers = deque(
        _normalize_prefilled_answers(answers=args.answer, answers_json=args.answers_json)
    )

    if args.resume_run_id:
        result = await resume_multi_agent_orchestration(args.resume_run_id, list(prefilled_answers))
        if prefilled_answers:
            prefilled_answers.clear()
    else:
        result = await run_multi_agent_orchestration(
            user_query=args.query,
            analyst_model=args.analyst_model,
            tables_dir=args.tables_dir,
            debug=not args.no_debug,
        )

    while result.get("status") == "interrupted":
        interrupts = result.get("interrupt") or []
        answers = _prompt_for_resume_answers(interrupts=interrupts, answer_queue=prefilled_answers)
        if not answers:
            break

        prefilled_answers.clear()
        result = await resume_multi_agent_orchestration(result["run_id"], answers)

    return result


def _print_output(result: dict[str, Any], as_json: bool) -> int:
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    status = result.get("status") or "unknown"

    orchestrator_trace = result.get("orchestrator_trace") or {}
    total_ms = orchestrator_trace.get("total_ms")

    if status not in {"completed", "degraded"}:
        print(f"Run status: {status}")
        print(f"Run ID: {result.get('run_id')}")
        if total_ms is not None:
            print(f"Total runtime: {int(total_ms)/1000:.2f} s")
        if status == "interrupted":
            print("Run is interrupted. Re-run with --resume-run-id and answers.")
        return 1

    analyst = result.get("analyst") or {}
    if status == "degraded":
        print("Run status: degraded")
        degradation = result.get("degradation") or {}
        notice = degradation.get("notice")
        if isinstance(notice, str) and notice.strip():
            print(notice.strip())

    answer = analyst.get("answer", "")
    if isinstance(answer, str):
        print(answer)
    else:
        print(json.dumps(answer, ensure_ascii=False))

    if total_ms is not None:
        print(f"Total runtime: {int(total_ms)/1000:.2f} s")

    return 0 if analyst.get("ok", False) else 2


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = asyncio.run(_run_pipeline(args))
    return _print_output(result, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
