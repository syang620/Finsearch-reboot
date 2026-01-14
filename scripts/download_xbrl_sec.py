#!/usr/bin/env python
"""
Download XBRL files from SEC's free Company Facts API
No API key required!
"""
from __future__ import annotations
import argparse
import json
import os
import requests
import time
from typing import Optional, Dict, Any

# SEC requires a User-Agent header
HEADERS = {
    "User-Agent": "FinSearch Research finsearch@example.com"
}

SEC_CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605"
}


def get_cik(ticker: str) -> str:
    """Get CIK for a ticker symbol."""
    ticker = ticker.upper()
    if ticker in SEC_CIK_MAP:
        return SEC_CIK_MAP[ticker]

    # Try to fetch from SEC
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&count=1&output=json"
    response = requests.get(url, headers=HEADERS)
    if response.ok:
        data = response.json()
        if "CIK" in data:
            cik = str(data["CIK"]).zfill(10)
            return cik

    raise ValueError(f"Could not find CIK for ticker: {ticker}")


def download_company_facts(cik: str, output_path: str) -> Dict[str, Any]:
    """
    Download all company facts (XBRL data) for a CIK.

    This includes all financial data across all filings.

    Args:
        cik: Company CIK (10 digits, zero-padded)
        output_path: Path to save JSON file

    Returns:
        Company facts data as dictionary
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    print(f"Downloading company facts for CIK {cik}...")
    print(f"  URL: {url}")

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return data


def download_filing_summary(cik: str, accession: str, output_path: str) -> Dict[str, Any]:
    """
    Download XBRL filing summary for a specific filing.

    Args:
        cik: Company CIK (10 digits)
        accession: Accession number (with dashes removed)
        output_path: Path to save JSON file

    Returns:
        Filing summary data
    """
    # Remove dashes from accession number
    accession_clean = accession.replace("-", "")

    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{cik}/{accession_clean}/filing-summary.json"

    print(f"Downloading filing summary...")
    print(f"  URL: {url}")

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 404:
        print("  Filing summary not available via this endpoint")
        return {}

    response.raise_for_status()
    data = response.json()

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return data


def get_recent_filings(cik: str, form_type: str, count: int = 10) -> list:
    """
    Get recent filings for a company.

    Args:
        cik: Company CIK
        form_type: Filing type (10-K, 10-Q, etc.)
        count: Number of filings to retrieve

    Returns:
        List of filing metadata
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    print(f"Fetching recent {form_type} filings for CIK {cik}...")

    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    recent_filings = data.get("filings", {}).get("recent", {})

    # Parse filings
    filings = []
    for i in range(len(recent_filings.get("accessionNumber", []))):
        if recent_filings["form"][i] == form_type:
            filing = {
                "accessionNumber": recent_filings["accessionNumber"][i],
                "filingDate": recent_filings["filingDate"][i],
                "reportDate": recent_filings["reportDate"][i],
                "primaryDocument": recent_filings["primaryDocument"][i],
                "primaryDocDescription": recent_filings["primaryDocDescription"][i]
            }
            filings.append(filing)

            if len(filings) >= count:
                break

    return filings


def main():
    parser = argparse.ArgumentParser(description="Download XBRL data from SEC (free API)")
    parser.add_argument("--ticker", default="AAPL", help="Company ticker symbol")
    parser.add_argument("--cik", help="Company CIK (if known)")
    parser.add_argument("--output-dir", default="data/xbrl",
                       help="Output directory")
    parser.add_argument("--list-filings", action="store_true",
                       help="List recent filings")
    parser.add_argument("--form", help="Form type filter (10-K, 10-Q)")

    args = parser.parse_args()

    # Get CIK
    if args.cik:
        cik = args.cik.zfill(10)
    else:
        cik = get_cik(args.ticker)

    print(f"Company: {args.ticker}")
    print(f"CIK: {cik}")

    # List filings if requested
    if args.list_filings:
        form_types = [args.form] if args.form else ["10-K", "10-Q"]
        for form_type in form_types:
            print(f"\nRecent {form_type} filings:")
            filings = get_recent_filings(cik, form_type, count=5)
            for filing in filings:
                print(f"  {filing['reportDate']} - {filing['accessionNumber']} - {filing['filingDate']}")
        return 0

    # Download company facts
    output_path = os.path.join(args.output_dir, f"{args.ticker}_CIK{cik}_facts.json")
    data = download_company_facts(cik, output_path)

    # Print summary
    entity_name = data.get("entityName", "Unknown")
    facts = data.get("facts", {})

    print(f"\n✅ Downloaded XBRL data for {entity_name}")
    print(f"   Contains {len(facts)} fact taxonomies:")
    for taxonomy in facts:
        print(f"     - {taxonomy}: {len(facts[taxonomy])} concepts")

    return 0


if __name__ == "__main__":
    exit(main())
