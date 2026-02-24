"""
chunks_loader.py

Helpers for loading text and table chunks produced by ingestion.sec_chunker / chunk_splitter.py.

Typical usage in a notebook:

    from chunks_loader import load_text_chunks, load_table_chunks, load_filing_chunks

    text_chunks, table_chunks = load_filing_chunks("AAPL_10-K_2025", out_dir="./chunks")

    len(text_chunks), len(table_chunks)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import pandas as pd

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Low-level JSONL loaders
# ---------------------------------------------------------------------

def load_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dicts.
    Each line is parsed with json.loads.

    Parameters
    ----------
    path : str | Path
        Path to the JSONL file.

    Returns
    -------
    List[Dict[str, Any]]
    """
    p = Path(path)
    chunks: List[Dict[str, Any]] = []

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------
# Text chunk loader
# ---------------------------------------------------------------------

def load_text_chunks(path: PathLike) -> List[Dict[str, Any]]:
    """
    Load text chunks from a *.text.jsonl or *.text.split.jsonl file.

    The JSON objects typically look like:
        {
          "level": "item" | "subsection",
          "item_id": "7",
          "heading_path": ["Item 7. ...", "Segment Operating Performance"],
          "text": "...",
          ...
        }

    Parameters
    ----------
    path : str | Path
        Path to the text chunk JSONL file.

    Returns
    -------
    List[Dict[str, Any]]
    """
    return load_jsonl(path)


# ---------------------------------------------------------------------
# Table chunk loader
# ---------------------------------------------------------------------

def load_table_chunks(path: PathLike, rebuild_df: bool = True) -> List[Dict[str, Any]]:
    """
    Load table chunks from a *.tables.jsonl file.

    Each JSON object is expected to look like (simplified):

        {
          "chunk_type": "table",
          "item_id": "7",
          "item_title": "...",
          "section_title": "...",
          "heading_path": [...],
          "text": "| col1 | col2 | ...",   # markdown text
          "table_html": "<table>...</table>",
          "table_dict": {
              "columns": [...],
              "index": [...],
              "data": [...]
          },
          "meta": {...}
        }

    If rebuild_df=True, adds a 'table_df' key with a pandas DataFrame
    reconstructed from table_dict.

    Parameters
    ----------
    path : str | Path
        Path to the table chunk JSONL file.
    rebuild_df : bool
        Whether to add 'table_df' for each chunk (default: True).

    Returns
    -------
    List[Dict[str, Any]]
    """
    raw = load_jsonl(path)
    if not rebuild_df:
        return raw

    chunks: List[Dict[str, Any]] = []
    for obj in raw:
        tbl = obj.get("table_dict")
        if isinstance(tbl, dict) and "columns" in tbl and "data" in tbl:
            try:
                obj["table_df"] = pd.DataFrame(**tbl)  # columns/index/data
            except Exception:
                # If something goes wrong, leave out table_df
                pass
        chunks.append(obj)
    return chunks


# ---------------------------------------------------------------------
# Filing-level convenience loader
# ---------------------------------------------------------------------

def load_filing_chunks(
    prefix: str,
    out_dir: PathLike = "./chunks",
    use_split_text: bool = True,
    rebuild_table_df: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load text + table chunks for a single filing, given a file prefix.

    The prefix should match how ingestion.sec_chunker saves files, e.g.:

        prefix = "AAPL_10-K_2025"

    Then this function will look for text and table files in out_dir:
        - text:
            * if use_split_text:
                <out_dir>/<prefix>.text.split.jsonl
              else:
                <out_dir>/<prefix>.text.jsonl
        - tables:
            <out_dir>/<prefix>.tables.jsonl

    Parameters
    ----------
    prefix : str
        File prefix like "AAPL_10-K_2025".
    out_dir : str | Path
        Directory where chunk files are stored (default: "./chunks").
    use_split_text : bool
        If True, prefer "<prefix>.text.split.jsonl" if it exists;
        otherwise fall back to "<prefix>.text.jsonl".
    rebuild_table_df : bool
        Whether to reconstruct pandas DataFrames in table chunks.

    Returns
    -------
    (text_chunks, table_chunks)

    Both are lists of dicts.
    """
    out_dir = Path(out_dir)

    # Resolve text path
    text_path_split = out_dir / f"{prefix}.text.split.jsonl"
    text_path_plain = out_dir / f"{prefix}.text.jsonl"

    if use_split_text and text_path_split.exists():
        text_path = text_path_split
    elif text_path_plain.exists():
        text_path = text_path_plain
    else:
        raise FileNotFoundError(
            f"Could not find text chunk file for prefix '{prefix}' in {out_dir} "
            f"(looked for {text_path_split} and {text_path_plain})"
        )

    # Resolve table path
    table_path = out_dir / f"{prefix}.tables.jsonl"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Could not find table chunk file for prefix '{prefix}' in {out_dir} "
            f"(expected {table_path})"
        )

    text_chunks = load_text_chunks(text_path)
    table_chunks = load_table_chunks(table_path, rebuild_df=rebuild_table_df)

    return text_chunks, table_chunks
