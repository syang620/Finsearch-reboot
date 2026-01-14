from __future__ import annotations
import re, json, os
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import pandas as pd
from .utils import FilingMeta, Chunk, hash_text

ITEM_RE = re.compile(r"^\s*Item\s+([0-9]+[A]?)\.", re.I)

def _nearest_headings(section_stack: List[Tuple[int,str]], level: int, title: str):
    # maintain a stack of (level, title)
    while section_stack and section_stack[-1][0] >= level:
        section_stack.pop()
    section_stack.append((level, title))
    return [t for _, t in section_stack]

def _section_path_for(tag) -> List[str]:
    # best-effort: collect preceding headings
    path = []
    cur = tag
    while cur:
        prev = cur.find_previous(["h1","h2","h3","h4","h5","h6"])
        if not prev: break
        level = int(prev.name[1])
        title = re.sub(r"\s+", " ", prev.get_text(" ", strip=True))
        path.insert(0, f"H{level}: {title}")
        cur = prev
    return path

def html_to_markdown_text(block) -> str:
    # Convert a bs4 element to Markdown, preserving paragraphs and links.
    html = str(block)
    m = md(html, strip=["style","script"])
    # normalize whitespace
    m = re.sub(r"\n{3,}", "\n\n", m).strip()
    return m

def extract_tables(html_path: str, filing: FilingMeta) -> List[Dict[str,Any]]:
    html = open(html_path, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")
    tables = []
    for i, tbl in enumerate(soup.find_all("table")):
        # locate caption or nearby heading
        caption = None
        if tbl.find("caption"):
            caption = tbl.find("caption").get_text(" ", strip=True)
        else:
            # nearest previous heading
            heads = _section_path_for(tbl)
            caption = heads[-1] if heads else "Table"
        # parse table to DataFrame
        try:
            df_list = pd.read_html(str(tbl))
            if not df_list:
                continue
            df = df_list[0]
        except Exception:
            continue
        # flatten headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                          for tup in df.columns.values]
        df.columns = [str(c).strip() for c in df.columns]
        # first column considered row labels
        if df.shape[1] >= 2:
            row_labels = df.iloc[:,0].astype(str).str.strip().tolist()
            data_cols = df.columns[1:]
        else:
            row_labels = [str(x) for x in range(len(df))]
            data_cols = df.columns

        # attempt statement type heuristics
        stype = None
        cap_low = (caption or "").lower()
        if "operations" in cap_low or "income" in cap_low:
            stype = "income_statement"
        elif "balance sheet" in cap_low or "financial position" in cap_low:
            stype = "balance_sheet"
        elif "cash flows" in cap_low:
            stype = "cash_flow"
        # simple currency/scaling extraction
        currency = "USD"
        scale = None
        head_text = tbl.get_text(" ", strip=True)
        if "in millions" in head_text.lower():
            scale = "millions"

        # columns with possible period labels
        columns = []
        for j, col in enumerate(data_cols):
            columns.append({
                "id": f"col_{i}_{j}",
                "label": str(col),
                "period_start": None,
                "period_end": None
            })

        rows = []
        for r, label in enumerate(row_labels):
            values = {}
            for j, col in enumerate(data_cols):
                try:
                    val = df.iloc[r, j+1]
                except Exception:
                    val = None
                # coerce numeric if possible
                if isinstance(val, str):
                    s = val.replace(",", "").replace("(", "-").replace(")", "").strip()
                    try:
                        val_num = float(s)
                        val = val_num
                    except Exception:
                        pass
                rows.append if False else None
                values[f"col_{i}_{j}"] = val
            rows.append({"label": str(label), "values": values})

        table_id = f"{os.path.basename(html_path)}_{i:03d}"
        tables.append({
            "table_id": table_id,
            "filing": {
                "ticker": filing.ticker, "form_type": filing.form_type,
                "fiscal_year": filing.fiscal_year, "period_type": filing.period_type,
                "period_start": filing.period_start, "period_end": filing.period_end,
                "source_path": filing.source_path or html_path
            },
            "location": {
                "item": None,
                "section_path": _section_path_for(tbl),
                "page_hint": None
            },
            "title": caption,
            "statement_type": stype,
            "currency": currency,
            "scale": scale,
            "columns": columns,
            "rows": rows
        })
    return tables

def extract_text_blocks(html_path: str, filing: FilingMeta) -> List[Chunk]:
    html = open(html_path, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")

    blocks = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","h6","p","div","li"]):
        if tag.name.startswith("h"):
            continue
        # skip tables—handled elsewhere
        if tag.find_parent("table") or tag.name == "table":
            continue
        text_md = html_to_markdown_text(tag)
        if not text_md or len(text_md.split()) < 10:
            continue
        section_path = _section_path_for(tag)
        ctext = text_md.strip()
        cid = f"narr_{hash_text(ctext)[:10]}"
        blocks.append(Chunk(
            chunk_id=cid,
            text=ctext,
            section_path=section_path,
            is_table=False,
            statement_type=None,
            filing=filing,
            extra={"source_path": filing.source_path or html_path}
        ))
    return blocks
