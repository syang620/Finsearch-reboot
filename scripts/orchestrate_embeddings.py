#!/usr/bin/env python
"""
Orchestrate embedding of SEC 10-K / 10-Q chunks using ingestion.sec_embedder.

This script:
  - Locates text chunk files (prefer *.text.split.jsonl, fallback to *.text.jsonl)
    and table summary files (*.tables.summaries.jsonl) for given filings.
  - Builds text / table / row docs via ingestion.sec_embedder.
  - Calls an embedding endpoint (default: Ollama-style /api/embed with
    model=qwen3-embedding:8b) to add embeddings.
  - Writes embedded JSONL files under a per-prefix directory in data/embedding:

      data/embedding/{PREFIX}/
        {PREFIX}.text.embedded.jsonl
        {PREFIX}.tables.embedded.jsonl
        {PREFIX}.tables.rows.embedded.jsonl

where PREFIX is:
  - 10-K : {TICKER}_10-K_{YEAR}
  - 10-Q : {TICKER}_10-Q_{YEAR}{QUARTER}  (e.g., AAPL_10-Q_2025Q1)

Example:

  PYTHONPATH=src python scripts/orchestrate_embeddings.py \\
      --tickers AAPL \\
      --forms 10-K \\
      --years 2024 \\
      --api-url http://localhost:11434/api/embed \\
      --model qwen3-embedding:8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ingestion.sec_embedder import (
    build_table_and_row_docs,
    build_text_docs,
    embed_docs,
)
from _common import load_tickers


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed SEC 10-K/10-Q chunks and table summaries and write "
            "embedded docs to data/embedding."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers (e.g., AAPL MSFT).",
    )
    parser.add_argument(
        "--from-file",
        help="Optional path to a text file with one ticker per line.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Form types to process (e.g., 10-K 10-Q).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Fiscal years to process for 10-K filings.",
    )
    parser.add_argument(
        "--quarters",
        nargs="+",
        help=(
            "Quarters to process for 10-Q, formatted as YEAR_QN "
            "(e.g., 2025_Q1 2025_Q2)."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunked",
        help="Directory containing text chunk JSONL files (default: data/chunked).",
    )
    parser.add_argument(
        "--table-summaries-dir",
        default="data/chunked/table_summaries",
        help=(
            "Directory containing table summary JSONL files "
            "(default: data/chunked/table_summaries)."
        ),
    )
    parser.add_argument(
        "--out-root",
        default="data/embedding",
        help="Root directory for per-prefix embedding outputs (default: data/embedding).",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:11434/api/embed",
        help="Embedding API URL (default: http://localhost:11434/api/embed).",
    )
    parser.add_argument(
        "--model",
        default="qwen3-embedding:8b",
        help="Embedding model name (default: qwen3-embedding:8b).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding requests (default: 16).",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        choices=["text", "tables", "rows"],
        default=["text", "tables", "rows"],
        help=(
            "Which logical doc types to embed. "
            "Choices: text, tables, rows (default: all)."
        ),
    )
    parser.add_argument(
        "--company-name",
        help="Optional company name to include in metadata for all filings.",
    )
    return parser.parse_args(argv)


def _parse_quarter(spec: str) -> Tuple[int, str]:
    """
    Parse a quarter spec of the form '2025_Q1' into (year, 'Q1').
    """
    try:
        year_str, q_label = spec.split("_", 1)
        year = int(year_str)
    except Exception:
        raise ValueError(f"Invalid quarter specifier: {spec!r}") from None
    return year, q_label


def _build_prefix(ticker: str, form: str, year: int, quarter: str | None) -> str:
    ticker = ticker.upper()
    form = form.upper()
    if form == "10-K":
        return f"{ticker}_10-K_{year}"
    if form == "10-Q":
        if not quarter:
            raise ValueError("Quarter must be provided for 10-Q filings.")
        return f"{ticker}_10-Q_{year}{quarter}"
    raise ValueError(f"Unsupported form: {form}")


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _find_text_chunk_path(chunks_dir: Path, prefix: str) -> Path | None:
    split_path = chunks_dir / f"{prefix}.text.split.jsonl"
    if split_path.is_file():
        return split_path
    plain_path = chunks_dir / f"{prefix}.text.jsonl"
    if plain_path.is_file():
        return plain_path
    return None


def _embed_for_prefix(
    prefix: str,
    ticker: str | None,
    form_type: str | None,
    fiscal_year: int | None,
    quarter_label: str | None,
    chunks_dir: Path,
    table_summaries_dir: Path,
    out_root: Path,
    api_url: str,
    model: str,
    batch_size: int,
    file_types: List[str],
    company_name: str | None,
) -> None:
    common_meta: Dict[str, Any] = {
        "prefix": prefix,
    }
    if ticker:
        common_meta["ticker"] = ticker
    if company_name:
        common_meta["company_name"] = company_name
    if form_type:
        common_meta["form_type"] = form_type
    if fiscal_year is not None:
        common_meta["fiscal_year"] = fiscal_year
    if quarter_label:
        common_meta["quarter"] = quarter_label

    out_dir = out_root / prefix

    # --- Text chunks ---
    if "text" in file_types:
        text_path = _find_text_chunk_path(chunks_dir, prefix)
        if not text_path:
            print(f"[WARN] No text chunk file found for {prefix} in {chunks_dir}")
        else:
            print(f"[INFO] Building text docs from {text_path}")
            text_docs = build_text_docs(text_path=text_path, common_meta=common_meta)
            if not text_docs:
                print(f"[WARN] No text docs built for {prefix}")
            else:
                embedded_text = embed_docs(
                    text_docs,
                    api_url=api_url,
                    model=model,
                    batch_size=batch_size,
                )
                out_path = out_dir / f"{prefix}.text.embedded.jsonl"
                print(f"[INFO] Writing {len(embedded_text)} text embeddings to {out_path}")
                _write_jsonl(out_path, embedded_text)

    # --- Table + row docs (from table summaries) ---
    tables_needed = "tables" in file_types
    rows_needed = "rows" in file_types
    if tables_needed or rows_needed:
        ts_path = table_summaries_dir / f"{prefix}.tables.summaries.jsonl"
        if not ts_path.is_file():
            print(f"[WARN] No table summaries file found for {prefix} at {ts_path}")
        else:
            print(f"[INFO] Building table/row docs from {ts_path}")
            docs_dict = build_table_and_row_docs(
                table_summaries_path=ts_path,
                common_meta=common_meta,
            )
            table_docs = docs_dict.get("tables", [])
            row_docs = docs_dict.get("rows", [])

            if tables_needed and table_docs:
                embedded_tables = embed_docs(
                    table_docs,
                    api_url=api_url,
                    model=model,
                    batch_size=batch_size,
                )
                out_path = out_dir / f"{prefix}.tables.embedded.jsonl"
                print(
                    f"[INFO] Writing {len(embedded_tables)} table embeddings to {out_path}",
                )
                _write_jsonl(out_path, embedded_tables)
            elif tables_needed:
                print(f"[WARN] No table docs built for {prefix}")

            if rows_needed and row_docs:
                embedded_rows = embed_docs(
                    row_docs,
                    api_url=api_url,
                    model=model,
                    batch_size=batch_size,
                )
                out_path = out_dir / f"{prefix}.tables.rows.embedded.jsonl"
                print(
                    f"[INFO] Writing {len(embedded_rows)} table-row embeddings to {out_path}",
                )
                _write_jsonl(out_path, embedded_rows)
            elif rows_needed:
                print(f"[WARN] No table-row docs built for {prefix}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tickers = load_tickers(
        tickers=args.tickers,
        from_file=args.from_file,
        required=True,
    )
    chunks_dir = Path(args.chunks_dir)
    table_summaries_dir = Path(args.table_summaries_dir)
    out_root = Path(args.out_root)

    for ticker in tickers:
        print(f"\n=== Embedding filings for {ticker} ===")
        for form in args.forms:
            form_norm = form.upper()

            if form_norm == "10-K":
                if not args.years:
                    print(
                        f"[WARN] No --years provided; skipping 10-K embeddings for {ticker}.",
                    )
                    continue
                for year in args.years:
                    prefix = _build_prefix(
                        ticker=ticker,
                        form=form_norm,
                        year=year,
                        quarter=None,
                    )
                    print(f"[INFO] Processing 10-K prefix {prefix}")
                    _embed_for_prefix(
                        prefix=prefix,
                        ticker=ticker,
                        form_type=form_norm,
                        fiscal_year=year,
                        quarter_label=None,
                        chunks_dir=chunks_dir,
                        table_summaries_dir=table_summaries_dir,
                        out_root=out_root,
                        api_url=args.api_url,
                        model=args.model,
                        batch_size=args.batch_size,
                        file_types=args.file_types,
                        company_name=args.company_name,
                    )

            elif form_norm == "10-Q":
                if not args.quarters:
                    print(
                        f"[WARN] No --quarters provided; skipping 10-Q embeddings for {ticker}.",
                    )
                    continue
                for q_spec in args.quarters:
                    try:
                        year, q_label = _parse_quarter(q_spec)
                    except ValueError as exc:
                        print(f"[WARN] {exc}; skipping.")
                        continue
                    prefix = _build_prefix(
                        ticker=ticker,
                        form=form_norm,
                        year=year,
                        quarter=q_label,
                    )
                    print(f"[INFO] Processing 10-Q prefix {prefix}")
                    _embed_for_prefix(
                        prefix=prefix,
                        ticker=ticker,
                        form_type=form_norm,
                        fiscal_year=year,
                        quarter_label=q_label,
                        chunks_dir=chunks_dir,
                        table_summaries_dir=table_summaries_dir,
                        out_root=out_root,
                        api_url=args.api_url,
                        model=args.model,
                        batch_size=args.batch_size,
                        file_types=args.file_types,
                        company_name=args.company_name,
                    )

            else:
                print(f"[WARN] Unsupported form type: {form_norm}; skipping.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
