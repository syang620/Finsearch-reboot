#!/usr/bin/env python
"""
Rename SEC chunk JSONL files under data/chunked to include ticker prefix.

Transforms:
    data/chunked/AAPL/10-K/10-K_2024.text.jsonl
    -> data/chunked/AAPL/10-K/AAPL_10-K_2024.text.jsonl

    data/chunked/AAPL/10-Q/10-Q_2025Q1.tables.jsonl
    -> data/chunked/AAPL/10-Q/AAPL_10-Q_2025Q1.tables.jsonl

This keeps the directory layout unchanged and only updates file basenames.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TARGET_RE = re.compile(
    r"^(?P<form>10-[A-Za-z0-9]+)_(?P<rest>.+)\.(?P<suffix>text\.split\.jsonl|text\.jsonl|tables\.jsonl)$"
)
PREFIXED_RE = re.compile(
    r"^[A-Za-z0-9._-]+_10-[A-Za-z0-9]+_.+\.(text\.split\.jsonl|text\.jsonl|tables\.jsonl)$"
)


def _is_prefixed(name: str) -> bool:
    # Already in <TICKER>_<FORM>_<REST>.* form.
    return bool(PREFIXED_RE.match(name))


def rename_chunked_files_with_ticker(
    root: Path,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    processed = 0
    renamed = 0
    skipped = 0
    conflicts = 0

    for path in sorted(root.rglob("*.jsonl")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if not TARGET_RE.match(path.name):
            continue
        if _is_prefixed(path.name):
            skipped += 1
            continue

        # Nested layout expected: <root>/<ticker>/<form>/<form>_rest...
        if len(path.parts) < 3:
            continue
        ticker = path.parent.parent.name
        if not ticker:
            continue
        if not TARGET_RE.match(path.name):
            continue

        destination = path.with_name(f"{ticker}_{path.name}")
        if destination.exists():
            if destination.stat().st_size == path.stat().st_size:
                print(f"[skip] same size exists: {path.relative_to(root)} -> {destination.relative_to(root)}")
                skipped += 1
            else:
                print(f"[conflict] different size exists: {destination}")
                conflicts += 1
            continue

        if dry_run:
            print(f"[dry-run] mv {path.relative_to(root)} -> {destination.relative_to(root)}")
        else:
            path.rename(destination)
            print(f"[renamed] {path.relative_to(root)} -> {destination.relative_to(root)}")
        renamed += 1
        processed += 1

    return processed, renamed, skipped, conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename nested chunk files to include ticker prefix.")
    parser.add_argument(
        "--root",
        default="data/chunked",
        help="Chunk root directory (default: data/chunked).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed, renamed, skipped, conflicts = rename_chunked_files_with_ticker(
        root=Path(args.root),
        dry_run=args.dry_run,
    )
    print(
        f"rename_chunked_filenames_with_ticker: processed={processed} "
        f"renamed={renamed} skipped={skipped} conflicts={conflicts} "
        f"dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
