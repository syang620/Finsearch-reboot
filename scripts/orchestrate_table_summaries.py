#!/usr/bin/env python
"""
Orchestrate table summarization using ingestion.tables_summarizer.

This script scans table chunk files in data/chunked (or a custom directory),
derives filing prefixes from *.tables.jsonl filenames, and invokes
`python -m ingestion.tables_summarizer` for each prefix.

Summaries are written to a separate directory (default: data/chunked/table_summaries)
with filenames that preserve the original prefix, e.g.:

  data/chunked/AAPL_10-K_2024.tables.jsonl
    -> data/chunked/table_summaries/AAPL_10-K_2024.tables.summaries.jsonl

Typical usage:

  PYTHONPATH=src python scripts/orchestrate_table_summaries.py \\
      --tickers AAPL MSFT AMZN \\
      --api-url http://localhost:11434/api/generate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from _common import load_tickers_set_optional


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate *.tables.jsonl files under a chunks directory and "
            "summarize them via ingestion.tables_summarizer."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional list of tickers to include (e.g., AAPL MSFT). If omitted, process all prefixes.",
    )
    parser.add_argument(
        "--from-file",
        help="Optional path to a text file with one ticker per line.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunked",
        help="Directory containing chunk JSONL files (default: data/chunked).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/chunked/table_summaries",
        help="Directory to write per-prefix table summary JSONL files.",
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="HTTP endpoint for the LLM (e.g., http://localhost:11434/api/generate).",
    )
    parser.add_argument(
        "--model",
        default="minimax-m2:cloud",
        help="LLM model name (default: minimax-m2:cloud).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-tables",
        type=int,
        help="Optional cap on tables per prefix to summarize.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking ingestion.tables_summarizer.",
    )
    return parser.parse_args(argv)


def _find_prefixes(chunks_dir: Path, tickers: Set[str] | None) -> List[str]:
    """
    Discover filing prefixes from *.tables.jsonl filenames in chunks_dir.

    For a file like:
        AAPL_10-K_2024.tables.jsonl
    we derive a prefix:
        AAPL_10-K_2024
    """
    prefixes: List[str] = []

    for path in sorted(chunks_dir.glob("*.tables.jsonl")):
        stem = path.stem  # e.g., "AAPL_10-K_2024.tables"
        if not stem.endswith(".tables"):
            continue
        prefix = stem[: -len(".tables")]

        if tickers:
            # Expect prefix to start with "<TICKER>_"
            prefix_ticker = prefix.split("_", 1)[0].upper()
            if prefix_ticker not in tickers:
                continue

        prefixes.append(prefix)

    return prefixes


def _run_tables_summarizer_for_prefix(
    python_exe: str,
    prefix: str,
    chunks_dir: Path,
    out_dir: Path,
    api_url: str,
    model: str,
    temperature: float,
    max_tables: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}.tables.summaries.jsonl"

    cmd = [
        python_exe,
        "-m",
        "ingestion.tables_summarizer",
        "--prefixes",
        prefix,
        "--chunks-dir",
        str(chunks_dir),
        "--api-url",
        api_url,
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--output-jsonl",
        str(out_path),
    ]

    if max_tables is not None:
        cmd.extend(["--max-tables", str(max_tables)])

    print(f"Running tables_summarizer for {prefix}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)

    tickers = load_tickers_set_optional(
        tickers=args.tickers,
        from_file=args.from_file,
    )
    prefixes = _find_prefixes(chunks_dir, tickers)

    if not prefixes:
        print(f"No *.tables.jsonl files found in {chunks_dir} matching the given filters.")
        return 0

    for prefix in prefixes:
        _run_tables_summarizer_for_prefix(
            python_exe=args.python,
            prefix=prefix,
            chunks_dir=chunks_dir,
            out_dir=out_dir,
            api_url=args.api_url,
            model=args.model,
            temperature=args.temperature,
            max_tables=args.max_tables,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
