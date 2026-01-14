#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from tqdm import tqdm
from typing import List
from rag10kq.utils import FilingMeta, ensure_dir, to_jsonl
from rag10kq.ingest import extract_tables, extract_text_blocks
from rag10kq.chunking import chunk_narrative
from rag10kq.indexer import build_bm25, build_dense_index

def guess_meta(html_path: str) -> FilingMeta:
    name = os.path.basename(html_path).lower()
    ft = "10-k" if "10-k" in name or "10k" in name else "10-q"
    form_type = "10-K" if ft == "10-k" else "10-Q"
    # naive year guess
    year = None
    for tok in name.replace("_","-").split("-"):
        if tok.isdigit() and len(tok) == 4:
            year = int(tok); break
    return FilingMeta(ticker="AAPL", form_type=form_type, fiscal_year=year,
                      source_path=html_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html_files", nargs="+", help="Paths to 10-K / 10-Q HTML files")
    ap.add_argument("--outdir", default="artifacts", help="Output dir")
    args = ap.parse_args()
    ensure_dir(args.outdir)

    all_chunks = []
    all_tables = []
    for path in tqdm(args.html_files, desc="Ingesting"):
        meta = guess_meta(path)
        tables = extract_tables(path, meta)
        all_tables.extend(tables)
        text_blocks = extract_text_blocks(path, meta)
        chunks = chunk_narrative(text_blocks, target_tokens=500, overlap_tokens=80, min_tokens=200)
        all_chunks.extend([c.to_dict() for c in chunks])

    to_jsonl(os.path.join(args.outdir, "tables.jsonl"), all_tables)
    to_jsonl(os.path.join(args.outdir, "chunks.jsonl"), all_chunks)

    # Build indexes (narrative only)
    from rag10kq.utils import read_jsonl, Chunk, FilingMeta
    dicts = read_jsonl(os.path.join(args.outdir, "chunks.jsonl"))
    chunks_obj = []
    for d in dicts:
        f = FilingMeta(**d["filing"])
        chunks_obj.append(Chunk(
            chunk_id=d["chunk_id"], text=d["text"], section_path=d["section_path"],
            is_table=d["is_table"], statement_type=d.get("statement_type"),
            filing=f, extra=d.get("extra", {})
        ))
    # indexes dir
    idx_dir = os.path.join(args.outdir, "index")
    ensure_dir(idx_dir)
    build_bm25(chunks_obj, idx_dir)
    build_dense_index(chunks_obj, idx_dir)

    print(f"✅ Wrote {len(all_chunks)} chunks and {len(all_tables)} tables")
    print(f"   Indexes at: {idx_dir}")

if __name__ == "__main__":
    main()
