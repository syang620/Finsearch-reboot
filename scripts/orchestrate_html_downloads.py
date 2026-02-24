#!/usr/bin/env python
"""
Orchestrate downloading 10-K and 10-Q HTML filings for multiple tickers.

This is a light wrapper around ingestion.sec_html_fetcher.download_company_filings_html.

Example:
    python scripts/orchestrate_html_downloads.py --tickers AAPL MSFT AMZN
"""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence

from ingestion.sec_html_fetcher import download_company_filings_html
from _common import load_tickers


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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tickers = load_tickers(
        tickers=args.tickers,
        from_file=args.from_file,
        required=True,
    )

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
