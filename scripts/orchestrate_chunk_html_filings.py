#!/usr/bin/env python
"""
Orchestrate chunking of 10-K and 10-Q HTML filings using rag10kq.sec_chunker.

This script expects HTML filings to already exist under data/html_filings,
typically created by scripts/orchestrate_html_downloads.py.

For each requested ticker / form / year or quarter, it:
  1) Locates the corresponding HTML file in data/html_filings.
  2) Invokes `python -m rag10kq.sec_chunker` to produce text + table chunks.

Example:
    PYTHONPATH=src python scripts/orchestrate_chunk_html_filings.py \\
        --tickers AAPL MSFT \\
        --forms 10-K 10-Q \\
        --years 2024 2025 \\
        --quarters 2025_Q1 2025_Q2 \\
        --out-dir data/chunked
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate 10-K/10-Q HTML filings under data/html_filings and "
            "chunk them via rag10kq.sec_chunker."
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
        help="Fiscal years to process (applies to 10-K). If omitted, process all 10-K HTMLs found.",
    )
    parser.add_argument(
        "--quarters",
        nargs="+",
        help=(
            "Quarters to process for 10-Q, formatted as YEAR_QN (e.g., 2025_Q1 2025_Q2). "
            "If omitted, process all 10-Q HTMLs found."
        ),
    )
    parser.add_argument(
        "--html-root",
        default="data/html_filings",
        help="Root directory where HTML filings are stored.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/chunked",
        help="Output directory for JSONL chunk files (default: data/chunked).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking rag10kq.sec_chunker.",
    )
    parser.add_argument(
        "--run-splitter",
        action="store_true",
        default=True,
        help=(
            "If set, run rag10kq.chunk_splitter on each generated text chunk file "
            "to produce *.text.split.jsonl."
        ),
    )
    parser.add_argument(
        "--split-max-tokens",
        type=int,
        default=1200,
        help="Maximum tokens per chunk when running the splitter (default: 1200).",
    )
    parser.add_argument(
        "--split-overlap-paragraphs",
        type=int,
        default=1,
        help="Paragraph overlap between split chunks (default: 1).",
    )
    parser.add_argument(
        "--split-encoding-model",
        type=str,
        default="text-embedding-3-large",
        help=(
            "Model name passed to tiktoken when running the splitter "
            "(default: text-embedding-3-large)."
        ),
    )
    return parser.parse_args(argv)


def _load_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(args.tickers)

    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as handle:
            for line in handle:
                ticker = line.strip()
                if ticker:
                    tickers.append(ticker)

    if not tickers:
        raise SystemExit("Provide --tickers or --from-file with at least one ticker.")

    return sorted({t.upper() for t in tickers})


def _iter_k_jobs(
    html_dir: Path,
    years: Sequence[int] | None,
) -> Iterable[Tuple[Path, int]]:
    """
    Yield (html_path, year) pairs for 10-K filings in html_dir.
    """
    if years:
        for year in years:
            path = html_dir / f"10-K_{year}.html"
            yield path, year
    else:
        for path in sorted(html_dir.glob("10-K_*.html")):
            # Expect file name like 10-K_2024.html
            stem = path.stem  # e.g., "10-K_2024"
            try:
                _, year_str = stem.split("_", 1)
                year = int(year_str)
            except Exception:
                continue
            yield path, year


def _iter_q_jobs(
    html_dir: Path,
    quarters: Sequence[str] | None,
) -> Iterable[Tuple[Path, int, str]]:
    """
    Yield (html_path, year, quarter_label) for 10-Q filings in html_dir.
    quarter_label is like 'Q1', 'Q2', etc.
    """
    if quarters:
        for q in quarters:
            # Expect something like "2025_Q1"
            try:
                year_str, q_label = q.split("_", 1)
                year = int(year_str)
            except Exception:
                print(f"Skipping invalid quarter specifier: {q}")
                continue
            path = html_dir / f"10-Q_{year}_{q_label}.html"
            yield path, year, q_label
    else:
        for path in sorted(html_dir.glob("10-Q_*.html")):
            # Expect file name like 10-Q_2025_Q1.html
            stem = path.stem  # e.g., "10-Q_2025_Q1"
            try:
                _, rest = stem.split("_", 1)  # "2025_Q1"
                year_str, q_label = rest.split("_", 1)
                year = int(year_str)
            except Exception:
                continue
            yield path, year, q_label


def _run_chunker(
    python_exe: str,
    html_path: Path,
    ticker: str,
    form_type: str,
    year: int,
    out_dir: Path,
    quarter: str | None = None,
) -> None:
    cmd = [
        python_exe,
        "-m",
        "rag10kq.sec_chunker",
        "--html-file",
        str(html_path),
        "--ticker",
        ticker,
        "--form",
        form_type,
        "--year",
        str(year),
        "--out-dir",
        str(out_dir),
    ]
    if quarter:
        cmd.extend(["--quarter", quarter])
    print(f"Running chunker: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_splitter(
    python_exe: str,
    text_path: Path,
    split_path: Path,
    max_tokens: int,
    overlap_paragraphs: int,
    encoding_model: str,
) -> None:
    if not text_path.is_file():
        print(f"    Splitter input not found (skipping): {text_path}")
        return
    if split_path.is_file():
        print(f"    Split output already exists (skipping): {split_path}")
        return

    cmd = [
        python_exe,
        "-m",
        "rag10kq.chunk_splitter",
        "--in-file",
        str(text_path),
        "--out-file",
        str(split_path),
        "--max-tokens",
        str(max_tokens),
        "--overlap-paragraphs",
        str(overlap_paragraphs),
        "--encoding-model",
        encoding_model,
    ]
    print(f"    Running splitter: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tickers = _load_tickers(args)
    html_root = Path(args.html_root)
    out_dir = Path(args.out_dir)

    for ticker in tickers:
        print(f"\n=== Chunking filings for {ticker} ===")
        for form_type in args.forms:
            form_type_norm = form_type.upper()
            form_dir = html_root / ticker / form_type_norm

            if not form_dir.exists():
                print(f"  Skipping {form_type_norm} for {ticker}: {form_dir} does not exist.")
                continue

            if form_type_norm.startswith("10-K"):
                print(f"  Processing 10-K HTMLs in {form_dir} ...")
                for html_path, year in _iter_k_jobs(form_dir, args.years):
                    if not html_path.is_file():
                        print(f"    Missing file (skipping): {html_path}")
                        continue
                    _run_chunker(
                        python_exe=args.python,
                        html_path=html_path,
                        ticker=ticker,
                        form_type="10-K",
                        year=year,
                        out_dir=out_dir,
                    )
                    if args.run_splitter:
                        prefix = f"{ticker}_10-K_{year}"
                        text_path = out_dir / f"{prefix}.text.jsonl"
                        split_path = out_dir / f"{prefix}.text.split.jsonl"
                        _run_splitter(
                            python_exe=args.python,
                            text_path=text_path,
                            split_path=split_path,
                            max_tokens=args.split_max_tokens,
                            overlap_paragraphs=args.split_overlap_paragraphs,
                            encoding_model=args.split_encoding_model,
                        )

            elif form_type_norm.startswith("10-Q"):
                print(f"  Processing 10-Q HTMLs in {form_dir} ...")
                for html_path, year, q_label in _iter_q_jobs(form_dir, args.quarters):
                    if not html_path.is_file():
                        print(f"    Missing file (skipping): {html_path}")
                        continue
                    _run_chunker(
                        python_exe=args.python,
                        html_path=html_path,
                        ticker=ticker,
                        form_type="10-Q",
                        year=year,
                        out_dir=out_dir,
                        quarter=q_label,
                    )
                    if args.run_splitter:
                        prefix = f"{ticker}_10-Q_{year}{q_label}"
                        text_path = out_dir / f"{prefix}.text.jsonl"
                        split_path = out_dir / f"{prefix}.text.split.jsonl"
                        _run_splitter(
                            python_exe=args.python,
                            text_path=text_path,
                            split_path=split_path,
                            max_tokens=args.split_max_tokens,
                            overlap_paragraphs=args.split_overlap_paragraphs,
                            encoding_model=args.split_encoding_model,
                        )

            else:
                print(f"  Unsupported form type: {form_type_norm}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
