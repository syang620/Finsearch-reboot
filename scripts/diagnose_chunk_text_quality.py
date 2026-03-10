#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"


if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))


from ingestion.chunk_paths import resolve_chunk_file
from ingestion.sec_chunker import get_edgar10k_parser_cls, parse_html_to_tree
from sec_parser.semantic_elements import TopSectionTitle, TextElement, TitleElement


@dataclass
class FilingCheck:
    ticker: str
    form: str
    year: int
    quarter: str | None
    parser: str
    node_count: int
    top_sections: int
    title_elements: int
    text_nodes: int
    item_heading_nodes: int
    text_chunks: Dict[str, int | str | bool | None]


def _is_item_like_title(raw: str) -> bool:
    from ingestion.sec_chunker import ITEM_RE

    return bool(ITEM_RE.search((raw or "")))


def _count_nodes(tree_nodes) -> tuple[int, int, int, int]:
    top_sections = 0
    title_elements = 0
    text_nodes = 0
    item_like = 0

    for node in tree_nodes:
        el = node.semantic_element
        if isinstance(el, TopSectionTitle):
            top_sections += 1
        if isinstance(el, TitleElement):
            title_elements += 1
            txt = (el.text or "").strip()
            if _is_item_like_title(txt):
                item_like += 1
        if isinstance(el, TextElement):
            txt = (el.text or "").strip()
            if txt:
                text_nodes += 1

    return top_sections, title_elements, text_nodes, item_like


def _chunk_status(path: Path) -> Dict[str, int | str | bool | None]:
    if not path or not path.is_file():
        return {
            "exists": False,
            "bytes": 0,
            "lines": 0,
            "non_empty_lines": 0,
        }

    data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    non_empty = 0
    for line in data:
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("text"):
                non_empty += 1
        except Exception:
            pass

    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "lines": len(data),
        "non_empty_lines": non_empty,
    }


def _resolve_chunk_file(ticker: str, form: str, year: int, quarter: str | None, chunks_root: Path, suffix: str) -> Path | None:
    base = f"{ticker}_{form}_{year}"
    if quarter:
        base = f"{base}{quarter}"

    return resolve_chunk_file(chunks_root, base, f"{base}.{suffix}")


def _iter_filings(html_root: Path):
    for path in sorted((html_root).glob("*/10-K/10-K_*.html")):
        ticker = path.parent.parent.name
        stem = path.stem
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        yield path, ticker, parts[1]


def run_diagnosis(html_root: Path, chunks_root: Path, only_empty: bool, output_json: Path | None = None) -> List[FilingCheck]:
    parser_name = get_edgar10k_parser_cls().__name__
    results: List[FilingCheck] = []

    for path, ticker, year_text in _iter_filings(html_root):
        try:
            year = int(year_text)
        except Exception:
            continue

        raw = path.read_text(encoding="utf-8", errors="ignore")
        tree = parse_html_to_tree(raw)
        nodes = list(tree.nodes)
        top_sections, title_elements, text_nodes, item_like = _count_nodes(nodes)

        text_path = _resolve_chunk_file(ticker, "10-K", year, None, chunks_root, "text.jsonl")
        split_path = _resolve_chunk_file(ticker, "10-K", year, None, chunks_root, "text.split.jsonl")
        table_path = _resolve_chunk_file(ticker, "10-K", year, None, chunks_root, "tables.jsonl")

        text_status = _chunk_status(text_path or Path("/dev/null"))
        split_status = _chunk_status(split_path or Path("/dev/null"))
        table_status = _chunk_status(table_path or Path("/dev/null"))

        rec = FilingCheck(
            ticker=ticker,
            form="10-K",
            year=year,
            quarter=None,
            parser=parser_name,
            node_count=len(nodes),
            top_sections=top_sections,
            title_elements=title_elements,
            text_nodes=text_nodes,
            item_heading_nodes=item_like,
            text_chunks={
                "text": text_status,
                "split": split_status,
                "tables": table_status,
            },
        )

        empty = (rec.text_chunks["text"]["bytes"] == 0)
        if not only_empty or empty:
            results.append(rec)

        status = "EMPTY" if empty else "OK"
        print(
            f"{ticker}\t{year}\t{parser_name}\t"
            f"top={top_sections}\ttitles={title_elements}\ttext_nodes={text_nodes}\t"
            f"item_like={item_like}\t{status}\t"
            f"text={rec.text_chunks['text']['bytes']}\tsplit={rec.text_chunks['split']['bytes']}\ttables={rec.text_chunks['tables']['bytes']}"
        )

    if output_json:
        payload = [r.__dict__ for r in results]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose text chunking health across 10-K filings.")
    parser.add_argument("--html-root", default="data/html_filings")
    parser.add_argument("--chunks-root", default="data/chunked")
    parser.add_argument("--only-empty", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    output_json = Path(args.output_json) if args.output_json else None
    run_diagnosis(
        html_root=Path(args.html_root),
        chunks_root=Path(args.chunks_root),
        only_empty=args.only_empty,
        output_json=output_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
