from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

_QUARTER_RE = re.compile(r"^(\d{4})_(Q[1-4])$")


def load_tickers(
    *,
    tickers: Sequence[str] | None = None,
    from_file: str | None = None,
    required: bool = False,
) -> List[str]:
    out: List[str] = []
    if tickers:
        out.extend(str(t).strip() for t in tickers if str(t).strip())

    if from_file:
        p = Path(from_file)
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                ticker = line.strip()
                if ticker:
                    out.append(ticker)

    uniq = sorted({t.upper() for t in out if t})
    if required and not uniq:
        raise SystemExit("Provide --tickers or --from-file with at least one ticker.")
    return uniq


def load_tickers_set_optional(
    *,
    tickers: Sequence[str] | None = None,
    from_file: str | None = None,
) -> Set[str] | None:
    vals = load_tickers(tickers=tickers, from_file=from_file, required=False)
    return set(vals) if vals else None


def parse_quarter_spec(spec: str) -> Tuple[int, str]:
    text = str(spec).strip().upper()
    m = _QUARTER_RE.match(text)
    if not m:
        raise ValueError(f"Invalid quarter specifier: {spec!r} (expected YEAR_QN, e.g. 2025_Q1)")
    return int(m.group(1)), m.group(2)


def parse_quarter_specs(specs: Iterable[str] | None) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for spec in specs or []:
        out.append(parse_quarter_spec(spec))
    return out
