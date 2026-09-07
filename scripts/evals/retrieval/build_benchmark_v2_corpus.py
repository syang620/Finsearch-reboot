#!/usr/bin/env python3
"""Build a NEW inspectable corpus; never touch historical sidecars or indexes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path

from ingestion.sec_chunker import (parse_html_to_tree, build_two_level_text_chunks,
                                   build_fallback_text_chunks, build_table_chunks)
from ingestion.chunk_splitter import (apply_parent_expansion, filter_levels,
                                     get_encoding, split_long_chunks)
from ingestion.sec_embedder import build_text_content

FILINGS = [("AAPL", 2024), ("AAPL", 2025), ("AMZN", 2023),
           ("AMZN", 2024), ("MSFT", 2024), ("MSFT", 2025)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    docs, sources = [], []
    encoding = get_encoding("text-embedding-3-large")
    for ticker, year in FILINGS:
        source = Path(f"data/html_filings/{ticker}/10-K/10-K_{year}.html")
        tree = parse_html_to_tree(source.read_text())
        raw = build_two_level_text_chunks(tree)
        fallback = not raw
        raw = raw or build_fallback_text_chunks(tree)
        raw = [asdict(c) for c in raw]
        chunks = split_long_chunks(encoding, filter_levels(apply_parent_expansion(raw),
                                  ("subsection",)), 1200, 1)
        prefix = f"{ticker}_10-K_{year}"
        common = dict(ticker=ticker, fiscal_year=year, form_type="10-K",
                      source_html=source.as_posix(), source_sha256=sha(source))
        for chunk in chunks:
            suffix = str(chunk["chunk_index"])
            if chunk["split_count"] > 1:
                suffix += f"::split::{chunk['split_index']}"
            docs.append(dict(id=f"{prefix}::text::{suffix}", content=build_text_content(chunk),
                             metadata={**common, "doc_type":"text_chunk",
                                       "section_path":" > ".join(chunk["heading_path"]),
                                       "item_id":chunk["item_id"],
                                       "chunk_index":chunk["chunk_index"],
                                       "split_index":chunk["split_index"]}))
        tables = build_table_chunks(tree)
        for i, table in enumerate(tables):
            docs.append(dict(id=f"{prefix}::table::{i}", content=table.text,
                             metadata={**common, "doc_type":"table", "table_index":i,
                                       "section_path":" > ".join(table.heading_path),
                                       "item_id":table.item_id}))
        sources.append({**common, "raw_text_chunks":len(raw), "text_chunks":len(chunks),
                        "tables":len(tables), "fallback_parser":fallback})
        print(prefix, len(chunks), "text", len(tables), "tables", flush=True)
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))
    docs.sort(key=lambda d:d["id"])
    for d in docs:
        d["content_sha256"] = hashlib.sha256(d["content"].encode()).hexdigest()
    corpus = out/"corpus.jsonl"
    corpus.write_text("".join(json.dumps(d, ensure_ascii=False, sort_keys=True)+"\n" for d in docs))
    manifest = {"corpus_id":"sec_retrieval_benchmark_v2", "documents":len(docs),
                "corpus_sha256":sha(corpus), "sources":sources,
                "construction":{"text":"unchanged sec_chunker and ingestion CLI splitter defaults: parent expansion, subsection, 1200 tokens, one paragraph overlap",
                                "tables":"full raw tables from unchanged sec_chunker; no generated summaries or table-row documents",
                                "scope":"all extracted chunks from six preselected tracked annual filings; no retrieval-based selection"},
                "packages":{p:importlib.metadata.version(p) for p in ["sec-parser","tiktoken","pandas","beautifulsoup4"]},
                "source_code_sha256":{p:sha(p) for p in ["src/ingestion/sec_chunker.py", "src/ingestion/chunk_splitter.py", "src/ingestion/sec_embedder.py"]}}
    (out/"corpus_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    build(parser.parse_args().out_dir)
