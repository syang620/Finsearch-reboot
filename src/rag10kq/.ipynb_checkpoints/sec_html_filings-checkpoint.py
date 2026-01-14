#!/usr/bin/env python
"""
Simple helpers to download 10-K and 10-Q HTML filings from SEC EDGAR.

This is intentionally minimal and mirrors the style of scripts/download_xbrl_sec.py.

Typical usage:
    python -m rag10kq.sec_html_filings --ticker AAPL --output-dir data/html_filings
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Iterable, List, Sequence

import requests

# SEC requires a User-Agent header
HEADERS = {
    "User-Agent": "FinSearch Research finsearch@example.com",
}

SEC_CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
}


def get_cik(ticker: str) -> str:
    """Get CIK for a ticker symbol (zero-padded to 10 digits)."""
    ticker = ticker.upper()
    if ticker in SEC_CIK_MAP:
        return SEC_CIK_MAP[ticker]

    url = (
        "https://www.sec.gov/cgi-bin/browse-edgar"
        f"?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&count=1&output=json"
    )
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    if "CIK" not in data:
        raise ValueError(f"Could not find CIK for ticker: {ticker}")

    cik = str(data["CIK"]).zfill(10)
    return cik


def get_recent_filings(cik: str, form_type: str, count: int = 5) -> List[Dict]:
    """
    Get recent filings for a company from the submissions API.

    Returns a list of dicts with keys:
    - accessionNumber
    - filingDate
    - reportDate
    - primaryDocument
    - primaryDocDescription
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    recent_filings = data.get("filings", {}).get("recent", {})
    accession_numbers = recent_filings.get("accessionNumber", [])
    forms = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    report_dates = recent_filings.get("reportDate", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    primary_descriptions = recent_filings.get("primaryDocDescription", [])

    filings: List[Dict] = []
    for index in range(len(accession_numbers)):
        if forms[index] != form_type:
            continue

        filing = {
            "accessionNumber": accession_numbers[index],
            "filingDate": filing_dates[index],
            "reportDate": report_dates[index],
            "primaryDocument": primary_documents[index],
            "primaryDocDescription": primary_descriptions[index],
        }
        filings.append(filing)
        if len(filings) >= count:
            break

    return filings


def build_html_url(cik: str, filing: Dict) -> str:
    """Build the EDGAR URL for the primary HTML filing document."""
    cik_int_str = str(int(cik))
    accession_no_dashes = filing["accessionNumber"].replace("-", "")
    primary_document = filing["primaryDocument"]
    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int_str}/{accession_no_dashes}/{primary_document}"
    )
    return url


def download_filing_html(
    cik: str,
    filing: Dict,
    output_dir: str,
    form_type: str,
    sleep_seconds: float = 0.2,
) -> str:
    """Download a single filing's primary HTML document."""
    url = build_html_url(cik, filing)
    os.makedirs(output_dir, exist_ok=True)

    accession_number = filing["accessionNumber"]
    filename = f"{form_type}_{accession_number}.html"
    safe_filename = filename.replace("/", "_")
    output_path = os.path.join(output_dir, safe_filename)

    print(f"Downloading {form_type} {accession_number} from {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    with open(output_path, "wb") as file:
        file.write(response.content)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return output_path


def download_company_filings_html(
    ticker_or_cik: str,
    form_types: Sequence[str] = ("10-K", "10-Q"),
    filings_per_form: int = 2,
    output_root: str = "data/html_filings",
    sleep_seconds: float = 0.2,
) -> Dict[str, List[str]]:
    """
    Download recent 10-K/10-Q HTML filings for a company.

    Returns a mapping form_type -> list of local file paths.
    """
    if ticker_or_cik.isdigit():
        cik = ticker_or_cik.zfill(10)
        label = f"CIK{cik}"
    else:
        cik = get_cik(ticker_or_cik)
        label = ticker_or_cik.upper()

    print(f"Company: {label}")
    print(f"CIK: {cik}")

    results: Dict[str, List[str]] = {}
    for form_type in form_types:
        print(f"\nFetching recent {form_type} filings...")
        filings = get_recent_filings(cik, form_type, count=filings_per_form)
        if not filings:
            print(f"  No recent {form_type} filings found.")
            results[form_type] = []
            continue

        form_output_dir = os.path.join(output_root, label, form_type)
        downloaded_paths: List[str] = []
        for filing in filings:
            path = download_filing_html(
                cik=cik,
                filing=filing,
                output_dir=form_output_dir,
                form_type=form_type,
                sleep_seconds=sleep_seconds,
            )
            print(f"  Saved to {path}")
            downloaded_paths.append(path)

        results[form_type] = downloaded_paths

    return results


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10-K and 10-Q HTML filings from SEC EDGAR",
    )
    parser.add_argument(
        "--ticker",
        help="Company ticker symbol (e.g., AAPL). If omitted, use --cik.",
    )
    parser.add_argument(
        "--cik",
        help="Company CIK (10 digits, zero-padded). If provided, overrides --ticker.",
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


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.ticker and not args.cik:
        raise SystemExit("Provide either --ticker or --cik.")

    identifier = args.cik if args.cik else args.ticker
    assert identifier is not None

    download_company_filings_html(
        ticker_or_cik=identifier,
        form_types=args.forms,
        filings_per_form=args.per_form,
        output_root=args.output_dir,
        sleep_seconds=args.sleep_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

