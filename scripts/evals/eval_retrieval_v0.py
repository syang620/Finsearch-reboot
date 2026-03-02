#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from evals.ragas_retrieval_metrics import RagasRetrievalConfig
from evals.retrieval_eval_runner import run_retrieval_eval


def _parse_csv_ints(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_csv_strs(raw: str) -> List[str]:
    out: List[str] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run retrieval-only evaluation (deterministic + ragas context metrics).")
    p.add_argument("--eval-path", default="data/evals/retrieval/table/table_eval_v1.jsonl")
    p.add_argument("--out-dir", default="artifacts/evals/retrieval/v0")
    p.add_argument("--eval-mode", choices=["auto", "table", "text"], default="auto")

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--k-values", default="1,3,5,10")
    p.add_argument(
        "--doc-types",
        default="",
        help="Optional override doc_types (CSV). Leave empty to use mode defaults.",
    )
    p.add_argument("--min-total-score", type=int, default=0)

    p.add_argument("--default-ticker", default="AAPL")
    p.add_argument("--default-fiscal-year", type=int, default=2024)
    p.add_argument("--default-form-type", default="10-K")

    p.add_argument("--enable-ragas", action="store_true", default=True)
    p.add_argument("--disable-ragas", action="store_true", default=False)

    p.add_argument("--ragas-ollama-base-url", default=os.getenv("RAGAS_OLLAMA_BASE_URL", "http://localhost:11434"))
    p.add_argument("--ragas-judge-model", default=os.getenv("RAGAS_JUDGE_MODEL", "llama3.1:8b-instruct"))
    p.add_argument("--ragas-embed-model", default=os.getenv("RAGAS_EMBED_MODEL", "nomic-embed-text"))
    p.add_argument("--ragas-timeout-s", type=int, default=120)
    p.add_argument("--enable-context-recall", action="store_true", default=False)

    p.add_argument("--fail-fast", action="store_true", default=False)
    return p


def main() -> None:
    args = build_parser().parse_args()

    enable_ragas = bool(args.enable_ragas and not args.disable_ragas)

    k_values = _parse_csv_ints(args.k_values)
    doc_types = _parse_csv_strs(args.doc_types)

    ragas_config = RagasRetrievalConfig(
        ollama_base_url=args.ragas_ollama_base_url,
        judge_model=args.ragas_judge_model,
        embed_model=args.ragas_embed_model,
        timeout_s=args.ragas_timeout_s,
        enable_context_recall=bool(args.enable_context_recall),
    )

    summary, _rows, errors = run_retrieval_eval(
        eval_path=args.eval_path,
        out_dir=args.out_dir,
        eval_mode=args.eval_mode,
        top_k=int(args.top_k),
        k_values=k_values,
        default_ticker=args.default_ticker,
        default_fiscal_year=int(args.default_fiscal_year),
        default_form_type=args.default_form_type,
        default_doc_types=doc_types or None,
        min_total_score=int(args.min_total_score),
        enable_ragas=enable_ragas,
        ragas_config=ragas_config,
        fail_fast=bool(args.fail_fast),
    )

    print(json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"output_dir={Path(args.out_dir).resolve()}")
    print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()
