#!/usr/bin/env python3
"""Validate embedding artifacts for consistency.

Usage:
    python scripts/validate_embedding_outputs.py \
      --embeddings data/embedding_batches/embeddings_unique.f16.npy \
      --all-jsonl data/embedding_batches/all_embedding_inputs.jsonl \
      --row-map data/embedding_batches/row_map.parquet \
      --unique-index data/embedding_batches/unique_index.parquet \
      --manifest data/embedding_batches/chunking_strategy_embed_inputs.jsonl.manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate embedding output files")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--all-jsonl", required=True)
    p.add_argument("--row-map", required=True)
    p.add_argument("--unique-index", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--max-preview", type=int, default=10)
    return p.parse_args()


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")


def _check_contiguous_unique(idx_df: pd.DataFrame, n_unique: int) -> bool:
    if n_unique == 0:
        return False
    vals = set(map(int, idx_df["unique_row"].tolist()))
    return len(vals) == n_unique and min(vals) == 0 and max(vals) == n_unique - 1


def _check_manifest_matches_row_map(manifest_path: Optional[Path], row_map_df: pd.DataFrame) -> Optional[str]:
    if manifest_path is None:
        return None
    if not manifest_path.exists():
        return f"manifest not found: {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest.get("total_docs", -1))
    if expected < 0:
        return "manifest missing total_docs"
    if expected != len(row_map_df):
        return (
            f"manifest total_docs mismatch: expected={expected}, "
            f"row_map_rows={len(row_map_df)}"
        )
    return None


def main() -> int:
    args = _parse_args()

    emb_path = Path(args.embeddings)
    all_jsonl = Path(args.all_jsonl)
    row_map_path = Path(args.row_map)
    unique_path = Path(args.unique_index)
    manifest_path = Path(args.manifest) if args.manifest else None

    for p in (emb_path, all_jsonl, row_map_path, unique_path):
        _ensure_file(p)

    arr = np.load(emb_path)
    row_map = pd.read_parquet(row_map_path)
    unique_index = pd.read_parquet(unique_path)
    all_rows = _count_jsonl_lines(all_jsonl)

    print("[INFO] embeddings", emb_path)
    print("shape", arr.shape)
    print("dtype", arr.dtype)
    print("[INFO] all_jsonl rows", all_rows)
    print("[INFO] row_map rows", len(row_map))
    print("[INFO] unique_index rows", len(unique_index))

    ok = True

    # Embedding cardinality checks
    if arr.shape[0] != len(unique_index):
        print(f"[FAIL] embeddings rows mismatch unique_index: {arr.shape[0]} != {len(unique_index)}")
        ok = False
    else:
        print("[PASS] embeddings rows == unique_index rows")

    # row_map unique row coverage
    if not _check_contiguous_unique(row_map, len(unique_index)):
        print("[FAIL] row_map unique_row not complete/contiguous")
        ok = False
    else:
        print("[PASS] row_map unique_row complete and contiguous")

    if row_map['id'].is_unique:
        print("[PASS] row_map ids are unique")
    else:
        dupes = len(row_map) - row_map['id'].nunique()
        print(f"[FAIL] row_map ids have duplicates: {dupes}")
        ok = False

    if unique_index['unique_row'].is_unique:
        print("[PASS] unique_index unique_row is unique")
    else:
        dupes = len(unique_index) - unique_index['unique_row'].nunique()
        print(f"[FAIL] unique_index unique_row duplicates: {dupes}")
        ok = False

    # Optional manifest check
    if manifest_path is not None:
        man_err = _check_manifest_matches_row_map(manifest_path, row_map)
        if man_err:
            print(f"[FAIL] manifest check: {man_err}")
            ok = False
        else:
            print(f"[PASS] manifest matches row_map rows: {manifest_path}")

    # Optional warning: many-to-unique compression ratio
    if all_rows:
        ratio = len(row_map) / max(1, all_rows)
        uniq_ratio = len(unique_index) / max(1, len(row_map))
        print(f"[INFO] row_map/all_rows ratio: {ratio:.4f}")
        print(f"[INFO] unique/all_rows ratio: {len(unique_index)/max(1,all_rows):.4f}")
        print(f"[INFO] unique/row_map ratio: {uniq_ratio:.4f}")

    # source / doc_type sanity sample
    if "doc_type" in row_map.columns:
        by_doc = row_map["doc_type"].value_counts().to_dict()
        print(f"[INFO] row_map doc_type distribution: {by_doc}")

    print("[RESULT]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
