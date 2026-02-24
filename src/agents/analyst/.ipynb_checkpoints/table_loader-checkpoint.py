"""Helpers for hydrating retrieved SEC table references into full table_dict payloads."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

_TABLE_DOC_ID_RE = re.compile(r"::table::(\d+)$")


def load_table_data(
    scored_entry: Dict[str, Any],
    data_dir: str,
    *,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Hydrate a retrieved table entry into a full table_dict by reading from disk.

    Accepts either:
    1) a single top-table entry, or
    2) the full retrieval output containing `top_tables` (uses first entry).
    """
    if not isinstance(scored_entry, dict):
        if verbose:
            print("Error: retrieval input is not a dict.")
        return None

    # Accept full retrieval response shape: {"top_tables": [...]}
    resolved_entry = scored_entry
    if "top_tables" in scored_entry and "table" not in scored_entry:
        top_tables = scored_entry.get("top_tables") or []
        if not isinstance(top_tables, list) or not top_tables:
            if verbose:
                print("Error: retrieval output has no top_tables entries.")
            return None
        resolved_entry = top_tables[0]

    # Accept both old ScoredPoint object shape and dict-serialized retrieval outputs.
    scored_point = resolved_entry.get("table", resolved_entry)
    payload: Dict[str, Any] = {}

    if isinstance(scored_point, dict):
        maybe_payload = scored_point.get("payload")
        payload = maybe_payload if isinstance(maybe_payload, dict) else scored_point
    else:
        maybe_payload = getattr(scored_point, "payload", None)
        payload = maybe_payload if isinstance(maybe_payload, dict) else {}

    if not payload and isinstance(resolved_entry.get("payload"), dict):
        payload = resolved_entry["payload"]

    prefix = payload.get("prefix")
    table_index = payload.get("table_index")
    doc_id = payload.get("doc_id")

    # Fallbacks from doc_id: "<prefix>::table::<index>"
    if prefix is None and isinstance(doc_id, str) and "::" in doc_id:
        prefix = doc_id.split("::", 1)[0]
    if table_index is None and isinstance(doc_id, str):
        m = _TABLE_DOC_ID_RE.search(doc_id)
        if m:
            table_index = int(m.group(1))

    # Last-resort fallback for partially flattened entries
    if prefix is None:
        prefix = resolved_entry.get("prefix")
    if table_index is None:
        table_index = resolved_entry.get("table_index")

    if prefix is None or table_index is None:
        entry_id = None
        if isinstance(scored_point, dict):
            entry_id = scored_point.get("id")
        else:
            entry_id = getattr(scored_point, "id", None)
        if verbose:
            print(f"Error: Missing metadata in payload for {entry_id}")
        return None

    file_path = Path(data_dir) / f"{prefix}.tables.jsonl"

    if not file_path.exists():
        if verbose:
            print(f"Error: Source file not found at {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == int(table_index):
                    chunk = json.loads(line)
                    return chunk.get("table_dict")
    except Exception as e:
        if verbose:
            print(f"Error reading file: {e}")
        return None

    if verbose:
        print(f"Error: Table index {table_index} out of bounds.")
    return None

