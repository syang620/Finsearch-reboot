#!/usr/bin/env python
"""
chunk_splitter.py

Post-process the text chunks from rag10kq.sec_chunker:

- Input:  JSONL file where each line is a dict like:
    {
      "level": "item" | "subsection",
      "item_id": "7",
      "heading_path": ["Item 7. ...", "Segment Operating Performance"],
      "text": "..."
      ... (any other fields are kept and copied)
    }

- Output: JSONL file where chunks whose `text` exceeds max_tokens are split
          into several smaller chunks along paragraph boundaries.

Requires:
    pip install tiktoken
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Iterable

import tiktoken


# ---------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------

def get_encoding(model_name: str = "text-embedding-3-large"):
    """
    Get a tiktoken encoding appropriate for the given model name.
    Falls back to cl100k_base if model-specific encoding isn't known.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(encoding, text: str) -> int:
    """Return token count for a given text string."""
    return len(encoding.encode(text or ""))


# ---------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------

def split_text_by_paragraph_tokens(
    encoding,
    text: str,
    max_tokens: int,
    overlap_paragraphs: int = 1,
) -> List[str]:
    """
    Split `text` into a list of chunk strings, where each chunk is at most
    `max_tokens` tokens (approx) according to tiktoken.

    Strategy:
      - Split on double newlines (paragraphs).
      - Accumulate paragraphs until adding another would exceed `max_tokens`.
      - When splitting, optionally carry over the last `overlap_paragraphs`
        paragraphs into the next chunk for context.

    Assumes: individual paragraphs are not themselves longer than max_tokens.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Simple paragraph split; you can refine later if you want
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []

    chunks: List[str] = []
    current_paras: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current_paras, current_tokens
        if not current_paras:
            return
        chunk_text = "\n\n".join(current_paras).strip()
        if chunk_text:
            chunks.append(chunk_text)
        current_paras = []
        current_tokens = 0

    for para in paras:
        para_tokens = count_tokens(encoding, para)

        # If adding this paragraph would exceed max_tokens and we already
        # have some content, flush and start a new chunk.
        if current_paras and current_tokens + para_tokens > max_tokens:
            overlap = (
                list(current_paras[-overlap_paragraphs:])
                if overlap_paragraphs > 0
                else []
            )
            flush()
            current_paras = overlap[:]
            current_tokens = sum(count_tokens(encoding, p) for p in current_paras)

        current_paras.append(para)
        current_tokens += para_tokens

    flush()
    return chunks


# ---------------------------------------------------------------------
# Main splitting routine
# ---------------------------------------------------------------------

def split_long_chunks(
    encoding,
    chunks: Iterable[Dict[str, Any]],
    max_tokens: int,
    overlap_paragraphs: int,
) -> List[Dict[str, Any]]:
    """
    Given an iterable of chunk dicts, detect those whose "text" is longer
    than `max_tokens` tokens and split them into multiple overlapping chunks.

    Returns:
        New list of chunk dicts.
    """
    new_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "") or ""
        token_count = count_tokens(encoding, text)

        if token_count <= max_tokens:
            # Add small metadata about tokens, optional
            chunk["token_len"] = token_count
            chunk["split_index"] = 0
            chunk["split_count"] = 1
            new_chunks.append(chunk)
            continue

        # Split this chunk into smaller pieces
        split_texts = split_text_by_paragraph_tokens(
            encoding,
            text,
            max_tokens=max_tokens,
            overlap_paragraphs=overlap_paragraphs,
        )

        if not split_texts:
            # If splitting somehow failed, fall back to original
            chunk["token_len"] = token_count
            chunk["split_index"] = 0
            chunk["split_count"] = 1
            new_chunks.append(chunk)
            continue

        for sub_idx, sub_text in enumerate(split_texts):
            new_chunk = dict(chunk)  # shallow copy original fields
            new_chunk["text"] = sub_text
            new_chunk["token_len"] = count_tokens(encoding, sub_text)
            new_chunk["split_index"] = sub_idx
            new_chunk["split_count"] = len(split_texts)
            new_chunk["parent_chunk_index"] = idx  # track source if you want
            new_chunks.append(new_chunk)

    return new_chunks


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def save_jsonl(path: Path, chunks: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split long text chunks (from rag10kq.sec_chunker) by token length.",
    )
    parser.add_argument(
        "--in-file",
        required=True,
        help="Input JSONL file with text chunks (e.g. AAPL_10-K_2025.text.jsonl)",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        help="Output JSONL file for split text chunks.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="Maximum tokens per chunk before splitting (default: 1200).",
    )
    parser.add_argument(
        "--overlap-paragraphs",
        type=int,
        default=1,
        help="Number of paragraphs to overlap between consecutive chunks (default: 1).",
    )
    parser.add_argument(
        "--encoding-model",
        type=str,
        default="text-embedding-3-large",
        help="Model name used to choose the tiktoken encoding (default: text-embedding-3-large).",
    )

    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    print(f"Loading chunks from {in_path} ...")
    raw_chunks = load_jsonl(in_path)
    print(f"Loaded {len(raw_chunks)} chunks")

    print(f"Initializing tiktoken encoding for {args.encoding_model} ...")
    encoding = get_encoding(args.encoding_model)

    print(
        f"Splitting chunks with max_tokens={args.max_tokens}, "
        f"overlap_paragraphs={args.overlap_paragraphs} ..."
    )
    new_chunks = split_long_chunks(
        encoding,
        raw_chunks,
        max_tokens=args.max_tokens,
        overlap_paragraphs=args.overlap_paragraphs,
    )

    print(
        f"Done. Original chunks: {len(raw_chunks)} -> New chunks: {len(new_chunks)}"
    )

    print(f"Saving new chunks to {out_path} ...")
    save_jsonl(out_path, new_chunks)
    print("All set.")


if __name__ == "__main__":
    main()
