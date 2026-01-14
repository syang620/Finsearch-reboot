#!/usr/bin/env python
"""
Download XBRL files from SEC using ExtractorAPI (sec-api.io)
"""
from __future__ import annotations
import argparse
import json
import os
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime

API_KEY = "b002a6feff8a952beb85391bf76e267ac605832b"
QUERY_API_URL = "https://api.sec-api.io"
XBRL_API_URL = "https://api.sec-api.io/xbrl-to-json"


def search_filings(ticker: str, form_type: str, year: int, quarter: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search for SEC filings by ticker, form type, and year/quarter.

    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        form_type: Filing type ('10-K' or '10-Q')
        year: Fiscal year
        quarter: Quarter (1-4) for 10-Q filings, None for 10-K

    Returns:
        List of filing metadata dictionaries
    """
    # Build date range query
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Build query
    query = f'ticker:{ticker} AND formType:"{form_type}" AND filedAt:[{start_date} TO {end_date}]'

    payload = {
        "query": query,
        "from": "0",
        "size": "50",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Add API key as query parameter
    url = f"{QUERY_API_URL}?token={API_KEY}"

    print(f"Searching for {ticker} {form_type} filings in {year}...")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()
    filings = data.get("filings", [])

    # Filter by quarter if specified (for 10-Q)
    if quarter and form_type == "10-Q":
        filtered = []
        for filing in filings:
            period = filing.get("periodOfReport", "")
            if period:
                # Parse period and determine quarter
                period_date = datetime.strptime(period, "%Y-%m-%d")
                filing_quarter = (period_date.month - 1) // 3 + 1
                if filing_quarter == quarter:
                    filtered.append(filing)
        filings = filtered

    return filings


def download_xbrl(accession_no: str, output_path: str) -> Dict[str, Any]:
    """
    Download XBRL data as JSON using accession number.

    Args:
        accession_no: Filing accession number
        output_path: Path to save JSON file

    Returns:
        XBRL data as dictionary
    """
    url = f"{XBRL_API_URL}?accession-no={accession_no}&token={API_KEY}"

    print(f"  Downloading XBRL for accession {accession_no}...")
    response = requests.get(url)
    response.raise_for_status()

    xbrl_data = response.json()

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(xbrl_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return xbrl_data


def download_filing(ticker: str, form_type: str, year: int, quarter: Optional[int] = None,
                   output_dir: str = "data/xbrl") -> Optional[str]:
    """
    Search and download XBRL data for a specific filing.

    Args:
        ticker: Company ticker symbol
        form_type: Filing type ('10-K' or '10-Q')
        year: Fiscal year
        quarter: Quarter (1-4) for 10-Q, None for 10-K
        output_dir: Output directory

    Returns:
        Path to saved XBRL JSON file, or None if not found
    """
    # Search for filing
    filings = search_filings(ticker, form_type, year, quarter)

    if not filings:
        print(f"  No {form_type} filings found for {ticker} in {year}" +
              (f" Q{quarter}" if quarter else ""))
        return None

    # Use the most recent filing
    filing = filings[0]
    accession_no = filing["accessionNo"]
    filed_at = filing["filedAt"]
    period = filing.get("periodOfReport", "unknown")

    print(f"  Found filing:")
    print(f"    Accession: {accession_no}")
    print(f"    Filed: {filed_at}")
    print(f"    Period: {period}")

    # Create output filename
    if quarter:
        filename = f"{ticker}_{form_type}_{year}_Q{quarter}.json"
    else:
        filename = f"{ticker}_{form_type}_{year}.json"

    output_path = os.path.join(output_dir, filename)

    # Download XBRL
    download_xbrl(accession_no, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download XBRL files from SEC")
    parser.add_argument("--ticker", default="AAPL", help="Company ticker symbol")
    parser.add_argument("--form", choices=["10-K", "10-Q"], required=True,
                       help="Filing form type")
    parser.add_argument("--year", type=int, required=True, help="Fiscal year")
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4],
                       help="Quarter (for 10-Q only)")
    parser.add_argument("--output-dir", default="data/xbrl",
                       help="Output directory")

    args = parser.parse_args()

    # Validate quarter for 10-Q
    if args.form == "10-Q" and not args.quarter:
        parser.error("--quarter is required for 10-Q filings")
    if args.form == "10-K" and args.quarter:
        parser.error("--quarter should not be specified for 10-K filings")

    # Download filing
    output_path = download_filing(
        ticker=args.ticker,
        form_type=args.form,
        year=args.year,
        quarter=args.quarter,
        output_dir=args.output_dir
    )

    if output_path:
        print(f"\n✅ Successfully downloaded XBRL data to: {output_path}")
    else:
        print(f"\n❌ Failed to download XBRL data")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
