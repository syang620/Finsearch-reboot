#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.planner import InteractivePlannerAgent
from agents.planner.evaluation import (
    default_cases_path,
    load_planner_eval_cases,
    run_planner_cases,
    serialize_results,
    summarize_results,
)


def _parse_csv(values: str | None) -> list[str] | None:
    if not values:
        return None
    items = [item.strip() for item in values.split(",") if item.strip()]
    return items or None


def _path_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _case_set_name(cases_path: str | Path) -> str:
    path = Path(cases_path)
    return path.name[:-5] if path.name.endswith(".json") else path.stem


def _default_out_path(cases_path: str | Path, model: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "artifacts"
        / "evals"
        / "agents"
        / "planner"
        / _case_set_name(cases_path)
        / "runs"
        / f"{timestamp}_{_path_token(model)}.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate planner routing behavior.")
    parser.add_argument(
        "--cases-path",
        default=str(default_cases_path()),
        help=(
            "Planner eval JSON file. Defaults to data/evals/agents/planner_eval_cases.json "
            "when present, otherwise planner_routing_core.v1.json."
        ),
    )
    parser.add_argument(
        "--model",
        default="ollama/qwen2.5:14b-instruct",
        help="Planner model passed to InteractivePlannerAgent.",
    )
    parser.add_argument(
        "--priority",
        default=None,
        help='Comma-separated priority filter, e.g. "P0". Omit to run all cases.',
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Comma-separated category filter. Omit to run all categories.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help=(
            "Path for raw planner eval results JSON artifact. Defaults to "
            "artifacts/evals/agents/planner/<case_set>/runs/<timestamp>_<model>.json."
        ),
    )
    return parser


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases = load_planner_eval_cases(
        args.cases_path,
        categories=_parse_csv(args.category),
        priorities=_parse_csv(args.priority),
    )
    if not cases:
        print("No planner eval cases matched the requested filters.", file=sys.stderr)
        return 2

    planner = InteractivePlannerAgent(model=args.model, log_timing=False)
    results = await run_planner_cases(planner, cases)
    out_path = Path(args.out_path) if args.out_path else _default_out_path(
        args.cases_path,
        args.model,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serialize_results(results), f, ensure_ascii=False, indent=2)

    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        detail = (
            f"{marker} {result.case.category}/{result.case.id} "
            f"route={result.actual_route!r} "
            f"structured_fact_count={result.actual_structured_fact_count}"
        )
        print(detail)
        for failure in result.failures:
            print(f"  - {failure}")

    summary = summarize_results(results)
    print("\nCategory summaries:")
    for category in sorted(summary["categories"]):
        item = summary["categories"][category]
        print(
            f"  {category}: {item['passed']}/{item['total']} "
            f"passed ({item['accuracy']:.1%})"
        )

    print(
        "\nOverall accuracy: "
        f"{summary['passed']}/{summary['total']} ({summary['accuracy']:.1%})"
    )
    print(f"Raw planner outputs written to: {out_path}")
    return 0 if summary["failed"] == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
