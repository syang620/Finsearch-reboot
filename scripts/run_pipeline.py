#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from qdrant_client import QdrantClient

from rag10kq.pipeline import (
    ExpansionConfig,
    FinanceRAGPipeline,
    GenerationConfig,
    LexicalScoringConfig,
    PipelineConfig,
    RetrievalConfig,
    RerankConfig,
)


def _split_csv(value: str | None) -> List[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_allowed_line_items(path: Path) -> Sequence[str]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]

    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]

    if isinstance(data, dict):
        for key in ("line_items", "items", "terms"):
            val = data.get(key)
            if isinstance(val, list):
                return [str(item).strip() for item in val if str(item).strip()]

    raise ValueError(f"Unsupported allowed-line-items format in {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Finance RAG pipeline.")
    parser.add_argument("--query", required=True, help="User query to answer.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--fiscal-year", required=True, type=int, help="Fiscal year, e.g. 2024.")
    parser.add_argument("--form-type", help="Form type, e.g. 10-K or 10-Q.")
    parser.add_argument("--collection", default="sec_docs", help="Qdrant collection name.")
    parser.add_argument("--tables-dir", default="data/chunked", help="Directory with table JSONL files.")
    parser.add_argument(
        "--allowed-line-items-file",
        required=True,
        help="Path to a JSON array or newline-delimited list of allowed line items.",
    )
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host.")
    parser.add_argument("--qdrant-port", default=6333, type=int, help="Qdrant port.")
    parser.add_argument("--doc-types", help="Comma-separated doc types filter.")
    parser.add_argument("--expansion-model", default="qwen3:4b-instruct", help="Ollama model for expansion.")
    parser.add_argument("--bge-model", default="BAAI/bge-m3", help="BGE-M3 model name.")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3", help="Reranker model name.")
    parser.add_argument("--gen-model", default="deepseek-r1:14b", help="Ollama model for generation.")
    parser.add_argument("--max-expansions", type=int, default=3, help="Max query expansions.")
    parser.add_argument("--top-k", type=int, default=50, help="Hybrid retrieval top_k.")
    parser.add_argument("--rerank-top-k", type=int, default=10, help="Reranker top_k.")
    parser.add_argument("--num-predict", type=int, default=1024, help="Generation num_predct.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--json", action="store_true", help="Print full JSON result.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    allowed_items = _load_allowed_line_items(Path(args.allowed_line_items_file))

    config = PipelineConfig(
        allowed_line_items=allowed_items,
        expansion=ExpansionConfig(model=args.expansion_model, max_expansions=args.max_expansions),
        retrieval=RetrievalConfig(
            collection_name=args.collection,
            bge_m3_model=args.bge_model,
            top_k=args.top_k,
        ),
        rerank=RerankConfig(model_name=args.reranker_model, top_k=args.rerank_top_k),
        lexical_scoring=LexicalScoringConfig(tables_dir=args.tables_dir),
        generation=GenerationConfig(
            model=args.gen_model,
            num_predct=args.num_predict,
            temperature=args.temperature,
        ),
    )

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    pipeline = FinanceRAGPipeline(client, config)

    result = pipeline.run(
        args.query,
        ticker=args.ticker,
        fiscal_year=args.fiscal_year,
        form_type=args.form_type,
        doc_types=_split_csv(args.doc_types),
    )

    if args.json:
        payload = {
            "answer": result.answer,
            "selected_tables": result.selected_tables,
            "queries": result.queries,
            "rerank_query": result.rerank_query,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
