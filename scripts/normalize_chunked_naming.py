#!/usr/bin/env python
"""
Normalize SEC chunk filenames in data/chunked.

Rules:
- Legacy flat files:
    data/chunked/AAPL_10-K_2024.text.jsonl
    -> data/chunked/AAPL/10-K/10-K_2024.text.jsonl
- Nested legacy files:
    data/chunked/AAPL/10-K/AAPL_10-K_2024.text.jsonl
    -> data/chunked/AAPL/10-K/10-K_2024.text.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_PREFIXED_RE = re.compile(
    r"^(?P<ticker>[A-Za-z0-9._-]+)_(?P<form>10-[A-Za-z0-9]+)_(?P<rest>\d{4}(?:Q[1-4])?)\.(?P<suffix>text\.split\.jsonl|text\.jsonl|tables\.jsonl)$"
)


def _canonical_for(source: Path, root: Path) -> tuple[Path, Path]:
    """Return (source, destination) for a legacy chunk filename."""
    m = _PREFIXED_RE.match(source.name)
    if not m:
        raise ValueError(f"Unrecognized chunk filename: {source}")

    ticker = m.group("ticker")
    form = m.group("form")
    rest = m.group("rest")
    suffix = m.group("suffix")
    destination = root / ticker / form / f"{form}_{rest}.{suffix}"
    return source, destination


def normalize_chunked_naming(root: Path, dry_run: bool = False) -> tuple[int, int, int, int]:
    """Normalize files and return counts."""
    processed = 0
    moved = 0
    skipped = 0
    conflicts = 0

    for path in sorted(root.rglob("*.jsonl")):
        if ".ipynb_checkpoints" in path.parts:
            continue

        if not _PREFIXED_RE.match(path.name):
            continue

        try:
            source, destination = _canonical_for(path, root)
        except ValueError:
            continue

        if source == destination:
            continue

        # If this file is already canonical, skip.
        if destination.exists() and destination.is_file():
            if destination.stat().st_size == path.stat().st_size:
                print(f"[skip] same size exists: {source.relative_to(root)} -> {destination.relative_to(root)}")
                skipped += 1
            else:
                print(f"[conflict] different size existing file: {destination}")
                conflicts += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"[dry-run] mv {source.relative_to(root)} -> {destination.relative_to(root)}")
        else:
            path.replace(destination)
            print(f"[moved] {source.relative_to(root)} -> {destination.relative_to(root)}")
        moved += 1

        processed += 1

    return processed, moved, skipped, conflicts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize chunked file naming under a chunk root.")
    parser.add_argument("--root", default="data/chunked", help="Root chunk directory.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without renaming files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed, moved, skipped, conflicts = normalize_chunked_naming(Path(args.root), dry_run=args.dry_run)
    print(
        f"normalize_chunked_naming: processed={processed} moved={moved} "
        f"skipped={skipped} conflicts={conflicts} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
