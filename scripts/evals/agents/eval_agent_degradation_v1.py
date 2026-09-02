#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.agent_eval_degradation_v1 import (
    evaluate_degradation_matrix,
    load_degradation_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic PR5 evidence-degradation evaluation."
    )
    parser.add_argument(
        "--eval-path",
        default="data/evals/agents/v1/agent_eval_degradation_v1.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/evals/agents/v1/degradation",
    )
    args = parser.parse_args()

    result = evaluate_degradation_matrix(load_degradation_cases(args.eval_path))
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if result["summary"]["overall_graceful_degradation_behavior_accuracy"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
