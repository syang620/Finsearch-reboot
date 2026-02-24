from __future__ import annotations

import json
from typing import Any, List


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def accounting_terms_to_llm_digest(
    terms: List[str],
    *,
    title: str = "Accounting Terms",
    max_terms: int | None = None,
) -> str:
    """
    Format a list of accounting terms into a compact text block for an LLM prompt.
    """
    cleaned: List[str] = []
    for t in terms:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s:
            cleaned.append(s)

    if max_terms is not None:
        cleaned = cleaned[:max_terms]

    lines = [f"{title} ({len(cleaned)}):"]
    lines.extend([f"- {t}" for t in cleaned])
    return "\n".join(lines)


def accounting_terms_file_to_llm_digest(
    path: str,
    *,
    title: str = "Accounting Terms",
    max_terms: int | None = None,
) -> str:
    """
    Read an accounting terms JSON file and return an LLM-friendly digest string.
    """
    obj = _load_json(path)
    if isinstance(obj, list):
        terms = obj
    elif isinstance(obj, dict) and isinstance(obj.get("terms"), list):
        terms = obj["terms"]
    else:
        raise ValueError(f"Unrecognized accounting terms format in {path}")

    return accounting_terms_to_llm_digest(
        terms,
        title=title,
        max_terms=max_terms,
    )


__all__ = [
    "accounting_terms_file_to_llm_digest",
    "accounting_terms_to_llm_digest",
]
