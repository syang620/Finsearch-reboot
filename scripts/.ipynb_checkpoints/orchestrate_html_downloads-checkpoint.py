#!/usr/bin/env python
"""
Orchestrate downloading 10-K and 10-Q HTML filings for multiple tickers.

This is a light wrapper around rag10kq.sec_html_filings.download_company_filings_html.

Example:
    python scripts/orchestrate_html_downloads.py --tickers AAPL MSFT AMZN
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

from rag10kq.sec_html_filings import download_company_filings_html


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10-K/10-Q HTML filings for one or more tickers.",
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
        "--output-dir",
        default="data/html_filings",
        help="Root output directory for downloaded filings.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Form types to download (e.g., 10-K 10-Q).",
    )
    parser.add_argument(
        "--per-form",
        type=int,
        default=2,
        help="Number of recent filings per form type.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between HTTP requests.",
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

    unique_tickers = sorted({ticker.upper() for ticker in tickers})
    return unique_tickers


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tickers = _load_tickers(args)

    for ticker in tickers:
        print(f"\n=== Downloading filings for {ticker} ===")
        download_company_filings_html(
            ticker_or_cik=ticker,
            form_types=args.forms,
            filings_per_form=args.per_form,
            output_root=args.output_dir,
            sleep_seconds=args.sleep_seconds,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

