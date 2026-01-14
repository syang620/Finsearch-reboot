#!/usr/bin/env python
"""
Orchestrate ingestion of pre-computed embeddings into Qdrant.

This script mirrors the style of other orchestration scripts (e.g.
scripts/orchestrate_table_summaries.py) and parameterizes:

  - ticker (e.g., AAPL)
  - fiscal year (e.g., 2024)
  - form type (10-K or 10-Q)
  - which embedding file types to ingest (text, tables, rows)

It expects embeddings to be laid out under an embeddings root directory as:

  data/embedding/{PREFIX}/
    {PREFIX}.text.embedded.jsonl
    {PREFIX}.tables.embedded.jsonl
    {PREFIX}.tables.rows.embedded.jsonl

where PREFIX is:
  - 10-K : {TICKER}_10-K_{YEAR}
  - 10-Q : {TICKER}_10-Q_{YEAR}{QUARTER}  (e.g., AAPL_10-Q_2025Q1)

Example:

  PYTHONPATH=src python scripts/orchestrate_ingestion.py \\
      --ticker AAPL \\
      --form 10-K \\
      --year 2024 \\
      --file-types text tables rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from rag10kq.qdrant_ingester import (
    count_points,
    ensure_collection,
    get_qdrant_client,
    upsert_docs,
)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest pre-computed SEC 10-K/10-Q embeddings into Qdrant, "
            "for a given ticker, year, and embedding file types."
        ),
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol (e.g., AAPL).",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        choices=["10-K", "10-Q"],
        help="Form type to ingest embeddings for (default: 10-K).",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help="Fiscal year corresponding to the filing (e.g., 2024).",
    )
    parser.add_argument(
        "--quarter",
        help=(
            "Quarter label for 10-Q filings (e.g., Q1, Q2). "
            "Only used when --form 10-Q."
        ),
    )
    parser.add_argument(
        "--embeddings-root",
        default="data/embedding",
        help="Root directory containing per-prefix embedding subdirectories.",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        choices=["text", "tables", "rows"],
        default=["text", "tables", "rows"],
        help=(
            "Which embedding file types to ingest. "
            "Choices: text, tables, rows (default: all)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for Qdrant upserts (default: 128).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333).",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="Use HTTPS when connecting to Qdrant.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional Qdrant API key.",
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> List[Dict]:
    docs: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

def write_jsonl(path: Path, docs: Iterable[Dict[str, Any]]) -> None:
    """Write docs as JSON lines to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def combine_for_prefix(
    prefix: str,
    embeddings_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    """
    Combine embedded text, table, and table_row docs for a single filing prefix
    into one JSONL file.

    Assumes filenames like:
      {prefix}.text.embedded.jsonl
      {prefix}.tables.embedded.jsonl
      {prefix}.tables.rows.embedded.jsonl   # adjust if your name is different

    Returns the path to the combined output file.
    """
    if output_dir is None:
        output_dir = embeddings_dir

    patterns = [
        "{prefix}.text.embedded.jsonl",
        "{prefix}.tables.embedded.jsonl",
        "{prefix}.tables.rows.embedded.jsonl",  # <- adjust if needed
    ]

    all_docs: List[Dict[str, Any]] = []
    for pattern in patterns:
        path = embeddings_dir / pattern.format(prefix=prefix)
        if not path.exists():
            print(f"[WARN] File not found for prefix={prefix}: {path}")
            continue
        docs = load_jsonl(path)
        print(f"[INFO] Loaded {len(docs)} docs from {path.name}")
        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError(f"No docs found for prefix={prefix} in {embeddings_dir}")

    # (Optional) sort by id for stability
    all_docs.sort(key=lambda d: d.get("id", ""))

    out_path = output_dir / f"{prefix}.all.embedded.jsonl"
    write_jsonl(out_path, all_docs)
    print(f"[INFO] Wrote {len(all_docs)} combined docs to {out_path}")
    return out_path

def _build_prefix(ticker: str, form: str, year: int, quarter: str | None) -> str:
    ticker = ticker.upper()
    form = form.upper()
    if form == "10-K":
        return f"{ticker}_10-K_{year}"
    if form == "10-Q":
        if quarter:
            return f"{ticker}_10-Q_{year}{quarter}"
        return f"{ticker}_10-Q_{year}"
    raise ValueError(f"Unsupported form: {form}")


def _embedding_paths_for_prefix(
    embeddings_root: Path,
    prefix: str,
) -> Mapping[str, Path]:
    base_dir = embeddings_root / prefix
    return {
        "text": base_dir / f"{prefix}.text.embedded.jsonl",
        "tables": base_dir / f"{prefix}.tables.embedded.jsonl",
        "rows": base_dir / f"{prefix}.tables.rows.embedded.jsonl",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    prefix = _build_prefix(args.ticker, args.form, args.year, args.quarter)
    embeddings_root = Path(args.embeddings_root)
    paths = _embedding_paths_for_prefix(embeddings_root, prefix)

    # 1. Load requested embedding docs
    docs_by_type: MutableMapping[str, List[Dict]] = {}
    dim: int | None = None

    for ftype in args.file_types:
        path = paths[ftype]
        if not path.is_file():
            print(f"[WARN] Skipping {ftype}: file not found at {path}")
            continue

        docs = load_jsonl(path)
        if not docs:
            print(f"[WARN] Skipping {ftype}: no documents in {path}")
            continue

        this_dim = len(docs[0].get("embedding", []))
        if this_dim == 0:
            print(f"[WARN] Skipping {ftype}: first doc has empty embedding in {path}")
            continue

        if dim is None:
            dim = this_dim
        elif this_dim != dim:
            print(
                f"[WARN] Skipping {ftype}: embedding dim {this_dim} != expected {dim}",
            )
            continue

        docs_by_type[ftype] = docs

    if not docs_by_type:
        print(
            f"[ERROR] No valid embedding documents loaded for prefix {prefix}. "
            "Nothing to ingest.",
        )
        return 1

    assert dim is not None

    # 2. Create client & collections
    client = get_qdrant_client(
        host=args.host,
        port=args.port,
        https=args.https,
        api_key=args.api_key,
    )

    collection_names: Dict[str, str] = {
        "text": "sec_10k_text_chunks",
        "tables": "sec_10k_tables",
        "rows": "sec_10k_table_rows",
    }

    for ftype, docs in docs_by_type.items():
        collection_name = collection_names[ftype]
        print(
            f"[INFO] Ensuring collection '{collection_name}' (dim={dim}) "
            f"for {ftype} embeddings.",
        )
        ensure_collection(
            client,
            collection_name,
            vector_size=dim,
            distance="COSINE",
            recreate=False,
        )

        # 3. Upsert
        print(
            f"[INFO] Upserting {len(docs)} {ftype} docs from prefix {prefix} "
            f"into collection '{collection_name}' (batch_size={args.batch_size}).",
        )
        upsert_docs(
            client,
            collection_name,
            docs,
            batch_size=args.batch_size,
        )

    # 4. Optional sanity check
    for ftype in docs_by_type.keys():
        collection_name = collection_names[ftype]
        count_points(client, collection_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
