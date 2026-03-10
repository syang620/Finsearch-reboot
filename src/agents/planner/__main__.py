from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with the packaged SEC planner agent from the terminal.",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Initial user query. If omitted, the CLI will prompt for one.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b-instruct-1m",
        help="Planner model to use for target resolution and downstream planning.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum clarification rounds to run in chat mode.",
    )
    parser.add_argument(
        "--disable-query-expansion",
        action="store_true",
        help="Disable taxonomy query expansion in the downstream full planner.",
    )
    parser.add_argument(
        "--no-full-planner",
        action="store_true",
        help="Stop after target resolution instead of running the downstream crawl-mode planner.",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Print the full final turn object as JSON.",
    )
    return parser


def _prompt_for_query() -> str:
    while True:
        query = input("Planner query\n> ").strip()
        if query:
            return query
        print("Query must be non-empty.", file=sys.stderr)


def _print_turn_summary(turn: Dict[str, Any]) -> None:
    summary = {
        "status": turn.get("status"),
        "clarification_request": turn.get("clarification_request"),
        "retrieval_plan": ((turn.get("target_resolution") or {}).get("retrieval_plan")),
        "target_resolution": turn.get("target_resolution"),
        "downstream_skipped_reason": turn.get("downstream_skipped_reason"),
        "has_full_plan": turn.get("full_plan") is not None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from .interactive_target_resolution import InteractivePlannerAgent

    user_query = str(args.query or "").strip() or _prompt_for_query()
    agent = InteractivePlannerAgent(
        model=args.model,
        enable_query_expansion=not bool(args.disable_query_expansion),
        log_timing=False,
        auto_run_full_planner=not bool(args.no_full_planner),
    )
    turn = agent.chat(user_query, max_rounds=max(1, int(args.max_rounds)))

    if args.full_json:
        print(json.dumps(turn, indent=2, ensure_ascii=False))
    else:
        _print_turn_summary(turn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
