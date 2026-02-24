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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from qdrant_client import models

from ingestion.qdrant_ingester import (
    count_points,
    docs_to_points,
    ensure_collection,
    get_qdrant_client,
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
    parser.add_argument(
        "--collection-name",
        default="sec_docs_hybrid",
        help="Qdrant collection name to ingest into (default: sec_docs_hybrid).",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help=(
            "Delete and recreate the target collection if it exists. "
            "This is required when migrating from a single-vector schema "
            "to named multi-vectors."
        ),
    )
    parser.add_argument(
        "--bge-m3-model-name",
        default="BAAI/bge-m3",
        help="HuggingFace model id/path for BGE-M3 (default: BAAI/bge-m3).",
    )
    parser.add_argument(
        "--bge-m3-cache-dir",
        help="Optional cache dir for BGE-M3 model files.",
    )
    parser.add_argument(
        "--bge-m3-dense-dim",
        type=int,
        default=1024,
        help="Expected BGE-M3 dense dimension (default: 1024).",
    )
    parser.add_argument(
        "--bge-m3-allow-download",
        action="store_true",
        help=(
            "Allow downloading BGE-M3 model files if not cached. "
            "By default, the script runs in local-files-only mode."
        ),
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
    base_dir = embeddings_root / prefix

    # 1. Combine text, tables, and table-row embeddings into a single JSONL
    try:
        combined_path = combine_for_prefix(prefix, base_dir)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    docs = load_jsonl(combined_path)
    if not docs:
        print(f"[ERROR] No documents loaded from combined file {combined_path}")
        return 1

    dim = len(docs[0].get("embedding", []))
    if dim == 0:
        print(
            f"[ERROR] First document in combined file {combined_path} "
            "has an empty embedding.",
        )
        return 1

    # Compute average doc length once for BM25 (used by Qdrant BM25 inference).
    lengths = []
    for d in docs:
        content = d.get("content")
        if isinstance(content, str) and content.strip():
            lengths.append(len(content.split()))
    bm25_avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    # 2. Create client & single collection
    client = get_qdrant_client(
        host=args.host,
        port=args.port,
        https=args.https,
        api_key=args.api_key,
    )

    collection_name = args.collection_name
    vectors_config = {
        "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE),
        "bge_m3_dense": models.VectorParams(
            size=args.bge_m3_dense_dim,
            distance=models.Distance.COSINE,
        ),
    }
    sparse_vectors_config = {
        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
        "bge_m3_sparse": models.SparseVectorParams(),
    }

    if client.collection_exists(collection_name=collection_name):
        if args.recreate_collection:
            print(f"[WARN] Recreating collection '{collection_name}' (will delete existing points).")
            client.delete_collection(collection_name=collection_name)
            ensure_collection(
                client=client,
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        else:
            info = client.get_collection(collection_name=collection_name)
            existing_vectors = info.config.params.vectors
            if not isinstance(existing_vectors, dict):
                print(
                    f"[ERROR] Collection '{collection_name}' exists but is not configured "
                    "for named multi-vectors. Re-run with --recreate-collection, "
                    "or ingest into a new --collection-name.",
                )
                return 1
    else:
        print(
            f"[INFO] Creating collection '{collection_name}' with dense+BM25+BGE-M3 vectors "
            f"(dense_dim={dim}, bge_m3_dense_dim={args.bge_m3_dense_dim}).",
        )
        ensure_collection(
            client=client,
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    # 2b. Initialize BGE-M3 encoder once (required by docs_to_points defaults).
    try:
        from FlagEmbedding import BGEM3FlagModel
    except Exception as exc:
        print(
            "[ERROR] Missing dependency `FlagEmbedding`. Install it (pip install FlagEmbedding) "
            "or adjust ingestion to pass precomputed BGE-M3 vectors per-doc.",
        )
        raise

    bge_m3_model = BGEM3FlagModel(
        args.bge_m3_model_name,
        use_fp16=True,
        cache_dir=args.bge_m3_cache_dir,
        local_files_only=not args.bge_m3_allow_download,
    )

    # 3. Upsert combined docs
    print(
        f"[INFO] Upserting {len(docs)} combined docs from prefix {prefix} "
        f"into collection '{collection_name}' (batch_size={args.batch_size}).",
    )
    for start in range(0, len(docs), args.batch_size):
        batch = docs[start : start + args.batch_size]
        points = docs_to_points(
            batch,
            bm25_avg_len=bm25_avg_len,
            bge_m3_model=bge_m3_model,
        )
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        print(f"[INFO]   upserted docs {start}–{start + len(batch) - 1}")

    # 4. Optional sanity check
    count_points(client, collection_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
