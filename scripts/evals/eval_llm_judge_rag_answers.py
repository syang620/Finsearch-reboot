#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.llm_judge_runner import run_llm_judge_eval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation over RAG final answers.")
    p.add_argument("--gold-path", default="data/evals/retrieval/aapl_2025_10k_gold_chunks.json")
    p.add_argument("--rag-path", default="artifacts/retrieval_v2_single_pass_aapl_2025_answers.json")
    p.add_argument("--out-dir", default="artifacts/evals/llm_judge/aapl_2025_10k")
    p.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    p.add_argument("--judge-mode", choices=("answer_only", "evidence_based"), default="answer_only")
    p.add_argument("--max-evidence-chunks", type=int, default=6)
    p.add_argument("--max-chars-per-chunk", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--fail-fast", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary, _rows, errors = run_llm_judge_eval(
        gold_path=args.gold_path,
        rag_path=args.rag_path,
        out_dir=args.out_dir,
        judge_model=args.judge_model,
        judge_mode=args.judge_mode,
        max_evidence_chunks=args.max_evidence_chunks,
        max_chars_per_chunk=args.max_chars_per_chunk,
        limit=args.limit,
        fail_fast=bool(args.fail_fast),
        base_url=args.base_url,
        timeout=float(args.timeout_s),
    )

    print(json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"output_dir={Path(args.out_dir).resolve()}")
    print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()
