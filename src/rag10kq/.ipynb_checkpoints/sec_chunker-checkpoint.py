#!/usr/bin/env python
"""
sec_chunker.py

Given a ticker, form type (e.g. 10-K), and fiscal year,
1) fetch the filing via edgartools
2) parse it with sec-parser into a semantic tree
3) produce:
   - two-level text chunks (item + subsection)
   - table chunks (with HTML + structured data + text)

Outputs two JSONL files in the given output directory:
   <TICKER>_<FORM>_<YEAR>.text.jsonl
   <TICKER>_<FORM>_<YEAR>.tables.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sec_parser as sp
from edgar import Company
from sec_parser.semantic_elements import (
    TextElement,
    SupplementaryText,
    TitleElement,
    TopSectionTitle,
    TableElement,
)
from sec_parser.semantic_elements.table_element.table_parser import TableParser


# ---------------------------------------------------------------------
# Text chunking helpers
# ---------------------------------------------------------------------

def _parse_date(value) -> Optional[date]:
    """Best-effort conversion of value to a date object."""
    if not value:
        return None

    # If it's already a date/datetime-like object
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        # It might be a pandas Timestamp or datetime; convert to date
        try:
            return value.date()  # pandas Timestamp / datetime
        except Exception:
            return value  # probably already a date

    # Otherwise, try ISO-8601 parse from string
    try:
        return date.fromisoformat(str(value).strip())
    except Exception:
        return None

def _get_year(value) -> Optional[int]:
    d = _parse_date(value)
    return d.year if d else None


def normalize_ws(text: str | None) -> str:
    """Collapse whitespace + NBSP to single spaces."""
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


ITEM_RE = re.compile(r"Item\s+(\d+[A-Z]?)\.\s*(.*)", re.IGNORECASE)


def parse_item_id_and_title(raw_title: str) -> Tuple[Optional[str], str]:
    """
    Turn "Item 7A.    Quantitative and Qualitative ..." into:
      ("7A", "Item 7A. Quantitative and Qualitative ...")
    """
    title = normalize_ws(raw_title)
    m = ITEM_RE.search(title)
    if not m:
        return None, title
    item_id = m.group(1).upper()
    rest = m.group(2).strip()
    full_title = f"Item {item_id}. {rest}".strip()
    return item_id, full_title


def get_edgar10k_parser_cls():
    """
    Return a sec_parser parser class in a way that works across versions.

    Newer sec-parser versions expose Edgar10QParser but not always Edgar10KParser.
    For our purposes, we fall back to Edgar10QParser when Edgar10KParser
    is unavailable.
    """
    # Preferred: root-level Edgar10KParser
    if hasattr(sp, "Edgar10KParser"):
        return sp.Edgar10KParser

    # Fallbacks: processing_engine Edgar10KParser, then Edgar10QParser
    try:
        from sec_parser.processing_engine import Edgar10KParser  # type: ignore
        return Edgar10KParser
    except Exception:
        pass

    try:
        from sec_parser.processing_engine.core import Edgar10KParser  # type: ignore
        return Edgar10KParser
    except Exception:
        pass

    try:
        from sec_parser.processing_engine.core import Edgar10QParser  # type: ignore
        return Edgar10QParser
    except Exception as exc:
        raise RuntimeError(
            "Could not locate an Edgar parser in sec_parser "
            "(expected Edgar10KParser or Edgar10QParser). "
            "Check your sec-parser installation."
        ) from exc


def collect_node_paragraphs(node) -> List[str]:
    """
    Collect all TextElement + SupplementaryText under a TreeNode's subtree.
    We ignore TitleElement text here so the chunk body is mostly paragraphs.
    """
    paragraphs: List[str] = []

    def _walk(n):
        el = n.semantic_element
        if isinstance(el, (TextElement, SupplementaryText)):
            text = normalize_ws(el.text or "")
            if text:
                paragraphs.append(text)
        for child in n.children:
            _walk(child)

    _walk(node)
    return paragraphs


@dataclass
class TextChunk:
    level: str               # "item" or "subsection"
    item_id: str             # e.g. "1", "1A", "7"
    heading_path: List[str]  # ["Item 7...", "Segment Operating Performance"]
    text: str                # body text


def parse_html_to_tree(html: str):
    """Parse 10-K/10-Q HTML into a sec_parser SemanticTree."""
    ParserCls = get_edgar10k_parser_cls()
    parser = ParserCls()
    elements = parser.parse(html)
    tree = sp.TreeBuilder().build(elements)
    return tree


def build_two_level_text_chunks(tree) -> List[TextChunk]:
    """
    Build 2-level text chunks:

      level="item":
        - One chunk per Item N (intro paragraphs before first subsection heading).

      level="subsection":
        - One chunk per TitleElement directly under an Item.
    """
    chunks: List[TextChunk] = []

    for node in tree.nodes:
        el = node.semantic_element
        if not isinstance(el, TopSectionTitle):
            continue

        raw_title = el.text or ""
        title_norm = normalize_ws(raw_title)

        # Skip PART I / PART II etc; only care about Items
        if not title_norm.lower().startswith("item"):
            continue

        item_id, item_title = parse_item_id_and_title(title_norm)
        if not item_id:
            # Fallback: treat entire title as item_title, synthetic ID
            item_id = title_norm

        # ------------------- item-level intro chunk -------------------
        intro_paragraphs: List[str] = []
        for child in node.children:
            cel = child.semantic_element
            # First TitleElement marks start of subsections
            if isinstance(cel, TitleElement):
                break
            if isinstance(cel, (TextElement, SupplementaryText)):
                intro_paragraphs.extend(collect_node_paragraphs(child))

        if intro_paragraphs:
            chunks.append(
                TextChunk(
                    level="item",
                    item_id=item_id,
                    heading_path=[item_title],
                    text="\n\n".join(intro_paragraphs),
                )
            )

        # ---------------- subsection-level chunks --------------------
        for child in node.children:
            cel = child.semantic_element
            if not isinstance(cel, TitleElement):
                continue

            subsection_title = normalize_ws(cel.text or "")
            paragraphs = collect_node_paragraphs(child)
            if not paragraphs:
                continue

            chunks.append(
                TextChunk(
                    level="subsection",
                    item_id=item_id,
                    heading_path=[item_title, subsection_title],
                    text="\n\n".join(paragraphs),
                )
            )

    return chunks


# ---------------------------------------------------------------------
# Table chunking helpers
# ---------------------------------------------------------------------

@dataclass
class TableChunk:
    chunk_type: str           # "table"
    item_id: Optional[str]    # "7", "7A", etc., or None
    item_title: Optional[str]
    section_title: Optional[str]
    heading_path: List[str]
    text: str                 # markdown or text representation
    table_html: str
    table_dict: Dict[str, Any]
    meta: Dict[str, Any]


def table_element_to_df_and_text(table_el: TableElement):
    """
    Convert a TableElement to (DataFrame | None, text_for_embedding).
    """
    html_tag = getattr(table_el, "html_tag", None)
    html = html_tag.get_source_code() if html_tag is not None else ""

    df = None
    try:
        if html:
            df = TableParser(html).parse_as_df()
    except Exception:
        df = None

    if df is not None:
        try:
            text = df.to_markdown(index=False)
        except Exception:
            text = normalize_ws(table_el.text or "") or html
    else:
        text = normalize_ws(table_el.text or "") or html

    return df, html, text


def get_heading_path(node) -> List[str]:
    """Collect heading texts from ancestors of `node`."""
    path: List[str] = []
    cur = node
    while cur is not None:
        el = cur.semantic_element
        if isinstance(el, (TopSectionTitle, TitleElement)):
            txt = normalize_ws(el.text or "")
            if txt:
                path.append(txt)
        cur = cur.parent
    path.reverse()
    return path


def build_table_chunks(tree, base_meta: Optional[Dict[str, Any]] = None) -> List[TableChunk]:
    """
    Walk a sec_parser SemanticTree and return a list of TableChunk objects.
    """
    if base_meta is None:
        base_meta = {}

    chunks: List[TableChunk] = []

    for node in tree.nodes:
        el = node.semantic_element
        if not isinstance(el, TableElement):
            continue

        heading_path = get_heading_path(node)

        # Identify item_id / item_title from heading path
        item_id: Optional[str] = None
        item_title: Optional[str] = None
        for h in heading_path:
            if h.lower().startswith("item "):
                item_id, parsed = parse_item_id_and_title(h)
                item_title = parsed
                break

        # Innermost section title (last non-Item heading)
        section_title: Optional[str] = None
        if heading_path:
            for t in reversed(heading_path):
                if t.lower().startswith("item "):
                    continue
                section_title = t
                break

        df, html, text = table_element_to_df_and_text(el)
        if df is not None:
            table_dict = df.to_dict(orient="split")  # columns/index/data
        else:
            table_dict = {"columns": [], "index": [], "data": []}

        meta = {
            **base_meta,
            "item_id": item_id,
            "item_title": item_title,
            "section_title": section_title,
        }

        chunks.append(
            TableChunk(
                chunk_type="table",
                item_id=item_id,
                item_title=item_title,
                section_title=section_title,
                heading_path=heading_path,
                text=text,
                table_html=html,
                table_dict=table_dict,
                meta=meta,
            )
        )

    return chunks


# ---------------------------------------------------------------------
# Filing fetcher (edgar)
# ---------------------------------------------------------------------

def fetch_filing_html(
    ticker: str,
    form: str,
    fiscal_year: int,
    identity: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Use edgar.Company to fetch the filing HTML for given ticker/form/year.

    We try to find the 10-K/10-Q whose report_date.year == fiscal_year.
    If none match, we fall back to the latest filing for that form.
    """
    set_identity(identity)
    company = Company(ticker)

    filings = company.get_filings(form=form)
    candidates = []
    for filing in filings:
        rd_year = _get_year(getattr(filing, "report_date", None))
        if rd_year == fiscal_year:
            candidates.append(filing)

    if candidates:
        # Among the filings for that fiscal year, pick the one with the latest filing_date
        def _filing_date_or_min(f):
            fd = _parse_date(getattr(f, "filing_date", None))
            return fd or date.min

        filing = max(candidates, key=_filing_date_or_min)
    else:
        # Fallback: just take the latest of that form
        filing = filings.latest()

    html = filing.html()

    # filing_date may be str or date-like; normalize to ISO string if possible
    fd = _parse_date(getattr(filing, "filing_date", None))
    filing_date_iso = fd.isoformat() if fd else str(getattr(filing, "filing_date", ""))

    meta = {
        "cik": company.cik,
        "ticker": ticker.upper(),
        "company_name": company.name,
        "form_type": form,
        "fiscal_year": fiscal_year,
        "filing_date": filing_date_iso,
        "accession": filing.accession_number,
    }
    return html, meta


# ---------------------------------------------------------------------
# Chunk + save helpers
# ---------------------------------------------------------------------

def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError("Expected a dataclass instance")


def save_jsonl(chunks: List[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            obj = dataclass_to_dict(c)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch SEC filing and chunk into text + table chunks."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--form", required=True, help="Form type, e.g. 10-K or 10-Q")
    parser.add_argument("--year", type=int, required=True, help="Fiscal year, e.g. 2025")
    parser.add_argument(
        "--email",
        required=True,
        help="Your contact email for SEC User-Agent (edgar.set_identity).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="chunks",
        help="Output directory for JSONL files.",
    )

    args = parser.parse_args()
    ticker = args.ticker
    form = args.form
    year = args.year
    email = args.email
    out_dir = Path(args.out_dir)

    identity = f"{ticker} filing chunker <{email}>"
    print(f"Fetching {form} for {ticker} FY {year} with identity: {identity}")

    html, filing_meta = fetch_filing_html(ticker, form, year, identity)
    print("Filing meta:", filing_meta)

    print("Parsing HTML into semantic tree...")
    tree = parse_html_to_tree(html)

    print("Building text chunks...")
    text_chunks = build_two_level_text_chunks(tree)

    print("Building table chunks...")
    table_chunks = build_table_chunks(tree, base_meta=filing_meta)

    doc_prefix = f"{filing_meta['ticker']}_{filing_meta['form_type']}_{filing_meta['fiscal_year']}"
    text_path = out_dir / f"{doc_prefix}.text.jsonl"
    table_path = out_dir / f"{doc_prefix}.tables.jsonl"

    print(f"Saving {len(text_chunks)} text chunks to {text_path}")
    save_jsonl(text_chunks, text_path)

    print(f"Saving {len(table_chunks)} table chunks to {table_path}")
    save_jsonl(table_chunks, table_path)

    print("Done.")


if __name__ == "__main__":
    main()
