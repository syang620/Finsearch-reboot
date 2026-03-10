#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests


# SEC expects a descriptive User-Agent for automated access.
DEFAULT_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "FinSearch Downloader (contact@example.com)",
)

# SEC ticker file often uses BRK-B while user input may be BRK.B.
TICKER_ALIAS = {
    "BRK.B": "BRK-B",
}


def _parse_symbol_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in tokens:
        for part in re.split(r"[,\s]+", str(raw).strip()):
            if part:
                out.append(part.upper())
    return out


def _read_symbols_file(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return _parse_symbol_tokens([text])


def _get_json(url: str, headers: Dict[str, str], timeout: int = 60, retries: int = 3) -> Dict:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)
    assert last_err is not None
    raise last_err


def _download_file(
    url: str,
    out_path: Path,
    headers: Dict[str, str],
    timeout: int = 120,
    retries: int = 3,
) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            return
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)
    assert last_err is not None
    raise last_err


def ticker_to_cik(
    ticker: str,
    ticker_map: Dict,
) -> str:
    """
    Return zero-padded 10-digit CIK for a ticker.
    """
    target = ticker.upper().strip()
    for rec in ticker_map.values():
        if str(rec.get("ticker", "")).upper() == target:
            return f"{int(rec['cik_str']):010d}"
    raise ValueError(f"Ticker not found in SEC company_tickers.json: {ticker}")


def find_primary_filing_url(
    cik10: str,
    fiscal_year: int,
    headers: Dict[str, str],
    form: str = "10-K",
) -> Dict[str, str]:
    """
    Find primary filing HTML URL by matching SEC recent filings where reportDate
    starts with fiscal_year (e.g., 2024-xx-xx for fiscal_year=2024).
    """
    sub_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    sub = _get_json(sub_url, headers=headers)

    def _pick_best(candidates: List[int], source: Dict[str, list], source_name: str) -> Dict[str, str]:
        filing_dates = source.get("filingDate", [])
        report_dates = source.get("reportDate", [])
        accession_numbers = source.get("accessionNumber", [])
        primary_docs = source.get("primaryDocument", [])
        if not candidates:
            raise ValueError(
                f"No {form} found in {source_name} for CIK={cik10} with reportDate in {fiscal_year}.",
            )

        best = max(candidates, key=lambda i: filing_dates[i] or "")
        accession = accession_numbers[best]
        accession_nodash = accession.replace("-", "")
        primary_doc = primary_docs[best]

        cik_no_leading = str(int(cik10))
        base = f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading}/{accession_nodash}/"
        return {
            "cik10": cik10,
            "fiscal_year": str(fiscal_year),
            "reportDate": report_dates[best],
            "filingDate": filing_dates[best],
            "accessionNumber": accession,
            "primaryDocument": primary_doc,
            "primary_html_url": base + primary_doc,
            "full_submission_txt_url": base + f"{accession}.txt",
            "index_url": base + f"{accession}-index.html",
            "source": source_name,
        }

    filings = sub.get("filings", {})
    recent = filings.get("recent", {})
    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    candidates: List[int] = []
    for i, (f, rd) in enumerate(zip(forms, report_dates)):
        if f != form:
            continue
        if not rd or not str(rd).startswith(f"{fiscal_year}-"):
            continue
        candidates.append(i)

    if candidates:
        return {k: v for k, v in _pick_best(candidates, recent, "recent").items() if k != "source"}

    # Some older filings (including 2024 year-end reports for some tickers) may
    # no longer be in filings.recent. Fall back to the archived filings list.
    archive_files = filings.get("files", [])
    for archive in archive_files:
        archive_name = archive.get("name")
        if not archive_name:
            continue
        archive_data = _get_json(
            f"https://data.sec.gov/submissions/{archive_name}",
            headers=headers,
        )
        forms = archive_data.get("form", [])
        report_dates = archive_data.get("reportDate", [])
        candidates = []
        for i, (f, rd) in enumerate(zip(forms, report_dates)):
            if f != form:
                continue
            if not rd or not str(rd).startswith(f"{fiscal_year}-"):
                continue
            candidates.append(i)
        if candidates:
            chosen = _pick_best(candidates, archive_data, archive_name)
            return {k: v for k, v in chosen.items() if k != "source"}

    raise ValueError(f"No {form} found for CIK={cik10} with reportDate in {fiscal_year}.")


def _parse_years(raw: str) -> List[int]:
    years: List[int] = []
    for part in re.split(r"[,\s]+", str(raw).strip()):
        if not part:
            continue
        years.append(int(part))
    if not years:
        raise ValueError("No fiscal years provided.")
    return years


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Download SEC filing primary HTML documents by ticker and fiscal year "
            "(reportDate year match)."
        ),
    )
    p.add_argument(
        "--tickers",
        nargs="*",
        default=[],
        help="Ticker list (space/comma separated), e.g. AAPL MSFT or 'AAPL,MSFT'.",
    )
    p.add_argument(
        "--tickers-file",
        help="Optional path to text file containing tickers (comma/space/newline separated).",
    )
    p.add_argument(
        "--exclude-tickers",
        default="",
        help="Tickers to exclude (comma/space separated).",
    )
    p.add_argument(
        "--fiscal-years",
        default="2024",
        help="Fiscal years to fetch, comma/space separated (default: 2024).",
    )
    p.add_argument(
        "--form",
        default="10-K",
        help="Form type to fetch (default: 10-K).",
    )
    p.add_argument(
        "--out-dir",
        default="data/sec_10k_downloads",
        help="Output root directory (default: data/sec_10k_downloads).",
    )
    p.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="SEC User-Agent header (default from SEC_USER_AGENT or fallback string).",
    )
    p.add_argument(
        "--also-download-full-submission-txt",
        action="store_true",
        help="Also download full submission text (.txt).",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between SEC requests (default: 0.2).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve URLs and write manifest, but do not download files.",
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    symbols: List[str] = []
    symbols.extend(_parse_symbol_tokens(args.tickers))
    if args.tickers_file:
        symbols.extend(_read_symbols_file(Path(args.tickers_file)))
    symbols = list(dict.fromkeys(symbols))  # stable dedupe
    if not symbols:
        raise SystemExit("No tickers provided. Use --tickers and/or --tickers-file.")

    exclude = set(_parse_symbol_tokens([args.exclude_tickers])) if args.exclude_tickers else set()
    tickers = [t for t in symbols if t not in exclude]
    years = _parse_years(args.fiscal_years)

    headers = {
        "User-Agent": args.user_agent.strip(),
        "Accept-Encoding": "gzip, deflate",
    }

    if not headers["User-Agent"] or "contact@example.com" in headers["User-Agent"]:
        print(
            "[WARN] Using placeholder User-Agent. Set --user-agent or SEC_USER_AGENT with real contact info.",
        )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ticker_map = _get_json("https://www.sec.gov/files/company_tickers.json", headers=headers)

    results: List[Dict] = []
    print(f"Processing tickers={len(tickers)} years={years} form={args.form}")

    for idx, original_ticker in enumerate(tickers, start=1):
        lookup_ticker = TICKER_ALIAS.get(original_ticker, original_ticker)
        ticker_dir = out_root / original_ticker.replace(".", "_")
        ticker_dir.mkdir(parents=True, exist_ok=True)

        rec: Dict[str, object] = {
            "ticker": original_ticker,
            "lookup_ticker": lookup_ticker,
            "downloads": [],
            "status": "ok",
        }

        try:
            cik10 = ticker_to_cik(lookup_ticker, ticker_map)
            rec["cik10"] = cik10
        except Exception as exc:
            rec["status"] = "lookup_failed"
            rec["error"] = str(exc)
            results.append(rec)
            print(f"[{idx}/{len(tickers)}] {original_ticker}: lookup_failed: {exc}")
            continue

        for fy in years:
            item: Dict[str, object] = {"fiscal_year": fy}
            try:
                meta = find_primary_filing_url(
                    cik10=cik10,
                    fiscal_year=fy,
                    headers=headers,
                    form=args.form,
                )

                safe_ticker = original_ticker.replace(".", "_")
                accession_nodash = str(meta["accessionNumber"]).replace("-", "")
                html_path = ticker_dir / (
                    f"{safe_ticker}_FY{fy}_{args.form}_{meta['reportDate']}_{accession_nodash}.html"
                )
                txt_path = ticker_dir / (
                    f"{safe_ticker}_FY{fy}_{args.form}_{meta['reportDate']}_{accession_nodash}_full.txt"
                )

                item.update(meta)

                if args.dry_run:
                    item["status"] = "resolved"
                    item["saved_path"] = str(html_path)
                    if args.also_download_full_submission_txt:
                        item["saved_txt_path"] = str(txt_path)
                    print(f"[{idx}/{len(tickers)}] {original_ticker} FY{fy}: resolved")
                else:
                    _download_file(meta["primary_html_url"], html_path, headers=headers)
                    item["saved_path"] = str(html_path)
                    item["status"] = "downloaded"
                    if args.also_download_full_submission_txt:
                        _download_file(meta["full_submission_txt_url"], txt_path, headers=headers)
                        item["saved_txt_path"] = str(txt_path)
                    print(
                        f"[{idx}/{len(tickers)}] {original_ticker} FY{fy}: "
                        f"downloaded ({meta['filingDate']}, {meta['accessionNumber']})",
                    )
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)
                print(f"[{idx}/{len(tickers)}] {original_ticker} FY{fy}: failed: {exc}")

            rec["downloads"].append(item)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        statuses = [d.get("status") for d in rec["downloads"]]
        if all(s == "failed" for s in statuses):
            rec["status"] = "all_failed"
        elif any(s == "failed" for s in statuses):
            rec["status"] = "partial"

        results.append(rec)

    # Summary + manifest
    attempts = len(tickers) * len(years)
    downloaded = sum(
        1 for r in results for d in r.get("downloads", []) if d.get("status") == "downloaded"
    )
    resolved = sum(
        1 for r in results for d in r.get("downloads", []) if d.get("status") == "resolved"
    )
    failed = sum(1 for r in results for d in r.get("downloads", []) if d.get("status") == "failed")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = out_root / f"manifest_{args.form.replace('/', '-')}_{stamp}.json"
    manifest = {
        "generated_utc": stamp,
        "out_dir": str(out_root),
        "form": args.form,
        "fiscal_years": years,
        "dry_run": bool(args.dry_run),
        "attempts": attempts,
        "downloaded": downloaded,
        "resolved": resolved,
        "failed": failed,
        "excluded": sorted(exclude),
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(
        f"tickers={len(tickers)} attempts={attempts} downloaded={downloaded} "
        f"resolved={resolved} failed={failed}",
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
