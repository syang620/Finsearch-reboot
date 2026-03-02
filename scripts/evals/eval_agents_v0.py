#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evals.agent_eval_runner import AgentEvalThresholds, run_agent_eval
from evals.ragas_agent_metrics import RagasAgentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run end-to-end agent evaluation (deterministic + optional RAGAS).")
    p.add_argument("--eval-path", default="data/evals/agents/v0/agent_eval_v0.jsonl")
    p.add_argument("--out-dir", default="artifacts/evals/agents/v0")

    p.add_argument("--analyst-model", default="qwen3:14b")
    p.add_argument("--planner-model", default="")

    p.add_argument("--enable-ragas", action="store_true", default=True)
    p.add_argument("--disable-ragas", action="store_true", default=False)

    p.add_argument("--ragas-ollama-base-url", default=os.getenv("RAGAS_OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--ragas-judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "llama3.1:8b-instruct"))
    p.add_argument("--ragas-embed-model", default=os.getenv("RAGAS_EMBED_MODEL", "nomic-embed-text"))
    p.add_argument("--ragas-timeout-s", type=int, default=120)
    p.add_argument("--enable-context-precision", action="store_true", default=False)
    p.add_argument("--enable-context-recall", action="store_true", default=False)

    p.add_argument("--deterministic-threshold", type=float, default=0.90)
    p.add_argument("--max-critical-failure-rate", type=float, default=0.05)
    p.add_argument("--ragas-threshold-faithfulness", type=float, default=0.75)
    p.add_argument("--ragas-threshold-answer-relevancy", type=float, default=0.75)
    p.add_argument("--max-ragas-error-rate", type=float, default=0.10)

    p.add_argument("--fail-fast", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()

    enable_ragas = bool(args.enable_ragas and not args.disable_ragas)

    ragas_config = RagasAgentConfig(
        ollama_base_url=args.ragas_ollama_base_url,
        judge_model=args.ragas_judge_model,
        embed_model=args.ragas_embed_model,
        timeout_s=args.ragas_timeout_s,
        enable_context_precision=bool(args.enable_context_precision),
        enable_context_recall=bool(args.enable_context_recall),
    )

    thresholds = AgentEvalThresholds(
        deterministic_score_min=float(args.deterministic_threshold),
        max_critical_failure_rate=float(args.max_critical_failure_rate),
        ragas_faithfulness_min=float(args.ragas_threshold_faithfulness),
        ragas_answer_relevancy_min=float(args.ragas_threshold_answer_relevancy),
        max_ragas_error_rate=float(args.max_ragas_error_rate),
    )

    summary, _rows, errors = run_agent_eval(
        eval_path=args.eval_path,
        out_dir=args.out_dir,
        analyst_model=args.analyst_model,
        planner_model=(args.planner_model.strip() or None),
        enable_ragas=enable_ragas,
        ragas_config=ragas_config,
        thresholds=thresholds,
        fail_fast=bool(args.fail_fast),
    )

    print(json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"output_dir={Path(args.out_dir).resolve()}")
    print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()
