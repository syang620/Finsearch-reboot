from __future__ import annotations

import re
from pathlib import Path

from typing import Optional, Tuple


_PREFIX_RE = re.compile(r"^(?P<ticker>.+?)_(?P<form>10-[A-Za-z0-9]+)_(?P<rest>.+)$")
_QTR_RE = re.compile(r"^(?P<year>\d{4})(?P<quarter>Q[1-4])$", re.IGNORECASE)


def parse_filing_prefix(prefix: str) -> Optional[Tuple[str, str, str]]:
    """Parse filing prefix into `(ticker, form, rest)`."""
    match = _PREFIX_RE.match(prefix)
    if not match:
        return None

    return (
        match.group("ticker"),
        match.group("form"),
        match.group("rest"),
    )


def _quarter_alt_rest(rest: str) -> Optional[str]:
    m = _QTR_RE.fullmatch(rest)
    if not m:
        return None
    return f"{m.group('year')}_{m.group('quarter')}"


def _candidate_nested_names(
    prefix: str,
    form: str,
    rest: str,
    filename: str,
    *,
    ticker: str | None = None,
) -> list[str]:
    if not filename.startswith(prefix + "."):
        return [filename]

    suffix = filename[len(prefix) :]
    names = [f"{form}_{rest}{suffix}"]
    if ticker:
        names.append(f"{ticker}_{form}_{rest}{suffix}")
    alt_rest = _quarter_alt_rest(rest)
    if alt_rest:
        names.extend([f"{form}_{alt_rest}{suffix}"])
        if ticker:
            names.append(f"{ticker}_{form}_{alt_rest}{suffix}")

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for candidate in names:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _is_checkpoint(path: Path) -> bool:
    return ".ipynb_checkpoints" in path.parts


def resolve_chunk_file(base_dir: Path | str, prefix: str, filename: str) -> Optional[Path]:
    """
    Resolve a chunk/summary file path from either legacy flat layout or nested layout.

    Legacy (flat): ``<base_dir>/<prefix>.<suffix>``
    Nested: ``<base_dir>/<ticker>/<form>/<form>_<rest>.<suffix>``
    """
    base = Path(base_dir)
    direct = base / filename
    if direct.is_file():
        return direct

    parsed = parse_filing_prefix(prefix)
    if parsed:
        ticker, form, rest = parsed
        nested_dir = base / ticker / form
        for nested_name in _candidate_nested_names(
            prefix=prefix,
            form=form,
            rest=rest,
            filename=filename,
            ticker=ticker,
        ):
            candidate = nested_dir / nested_name
            if candidate.is_file():
                return candidate

    # Last resort: scan recursively for exact filename (useful for custom layouts).
    for candidate in sorted(base.rglob(filename)):
        if _is_checkpoint(candidate):
            continue
        if candidate.is_file():
            return candidate

    return None


__all__ = ["parse_filing_prefix", "resolve_chunk_file"]
