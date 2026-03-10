#!/usr/bin/env python
"""
chunk_splitter.py

Post-process the text chunks from ingestion.sec_chunker:

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

Optional preprocessing can also:
  - filter to selected chunk levels (for example, subsection-only)
  - prepend parent item text onto subsection chunks before splitting

Requires:
    pip install tiktoken
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

def _resolve_parent_text_by_item_id(chunks: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    parents: Dict[str, str] = {}
    for chunk in chunks:
        if str(chunk.get("level") or "").lower() != "item":
            continue
        item_id = str(chunk.get("item_id") or "").strip()
        if not item_id:
            continue
        text = str(chunk.get("text") or "").strip()
        if text:
            parents[item_id] = text
    return parents


def apply_parent_expansion(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunk_list = [dict(chunk) for chunk in chunks]
    parents = _resolve_parent_text_by_item_id(chunk_list)
    out: List[Dict[str, Any]] = []

    for chunk in chunk_list:
        if str(chunk.get("level") or "").lower() != "subsection":
            out.append(chunk)
            continue

        item_id = str(chunk.get("item_id") or "").strip()
        parent_text = parents.get(item_id)
        if not parent_text:
            out.append(chunk)
            continue

        text = str(chunk.get("text") or "").strip()
        chunk["text"] = f"{parent_text}\n\n{text}".strip()
        chunk["parent_expanded"] = True
        out.append(chunk)

    return out


def filter_levels(
    chunks: Iterable[Dict[str, Any]],
    levels: Tuple[str, ...] | None,
) -> List[Dict[str, Any]]:
    chunk_list = [dict(chunk) for chunk in chunks]
    if not levels:
        return chunk_list

    wanted = {str(level).strip().lower() for level in levels if str(level).strip()}
    if not wanted:
        return chunk_list

    return [
        chunk
        for chunk in chunk_list
        if str(chunk.get("level") or "").strip().lower() in wanted
    ]

def split_text_by_paragraph_tokens(
    encoding,
    text: str,
    max_tokens: int,
    overlap_paragraphs: int = 1,
) -> List[str]:
    """
    Split `text` into a list of chunk strings, where each chunk is at most
    `max_tokens` tokens according to tiktoken.

    Strategy:
      - Split on double newlines (paragraphs).
      - Accumulate paragraphs until adding another would exceed `max_tokens`.
      - When splitting, optionally carry over the last `overlap_paragraphs`
        paragraphs into the next chunk for context.

    Assumes: individual paragraphs may be longer than max_tokens; oversized
    paragraphs are split recursively.
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

    def split_oversized_para(para: str) -> List[str]:
        """
        Hard-split a single oversized paragraph into sub-chunks <= max_tokens.
        Prefer sentence boundaries first, then fall back to token slicing.
        """
        if not para:
            return []

        sentences = re.split(r"(?<=[.!?])\\s+(?=[A-Z0-9(\\\"'])", para)
        if len(sentences) == 1:
            sentences = [para]

        sentence_chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for sent in [s.strip() for s in sentences if s and s.strip()]:
            sent_tokens = count_tokens(encoding, sent)
            if sent_tokens > max_tokens:
                if current:
                    chunk_text = " ".join(current).strip()
                    if chunk_text:
                        sentence_chunks.append(chunk_text)
                    current = []
                    current_tokens = 0

                token_ids = encoding.encode(sent)
                for start in range(0, len(token_ids), max_tokens):
                    token_slice = token_ids[start : start + max_tokens]
                    sentence_chunks.append(encoding.decode(token_slice).strip())
                continue

            if current and current_tokens + sent_tokens > max_tokens:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    sentence_chunks.append(chunk_text)
                current = []
                current_tokens = 0

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                sentence_chunks.append(chunk_text)

        if not sentence_chunks:
            return [para]
        return sentence_chunks

    for para in paras:
        para_tokens = count_tokens(encoding, para)

        # Hard split oversized paragraph.
        if para_tokens > max_tokens:
            split_paras = split_oversized_para(para)
            for split_para in split_paras:
                split_tokens = count_tokens(encoding, split_para)
                if split_tokens <= 0:
                    continue
                if current_paras and current_tokens + split_tokens > max_tokens:
                    overlap = (
                        list(current_paras[-overlap_paragraphs:])
                        if overlap_paragraphs > 0
                        else []
                    )
                    flush()
                    current_paras = overlap[:]
                    current_tokens = sum(count_tokens(encoding, p) for p in current_paras)
                if current_tokens + split_tokens > max_tokens:
                    # If even overlap does not leave room, start fresh.
                    current_paras = []
                    current_tokens = 0
                current_paras.append(split_para)
                current_tokens += split_tokens
            continue

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


def _ensure_hard_token_cap(
    encoding,
    text: str,
    max_tokens: int,
) -> List[str]:
    """
    Fallback splitter that guarantees every output chunk is under max_tokens.
    """
    text = (text or "").strip()
    if not text:
        return []

    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks: List[str] = []
    for start in range(0, len(token_ids), max_tokens):
        token_slice = token_ids[start : start + max_tokens]
        chunk_text = encoding.decode(token_slice).strip()
        if not chunk_text:
            continue
        chunks.append(chunk_text)
    return chunks


def split_text_by_tokens(
    encoding,
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    """
    Split `text` into fixed-size token windows.

    This splitter is token-window based and does not attempt structure-aware
    paragraph boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0 for token-window splitting.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0 for token-window splitting.")

    token_ids = encoding.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    step = max(1, max_tokens - overlap_tokens)
    if step <= 0:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    chunks: List[str] = []
    for start in range(0, len(token_ids), step):
        token_slice = token_ids[start : start + max_tokens]
        chunk_text = encoding.decode(token_slice).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


# ---------------------------------------------------------------------
# Main splitting routine
# ---------------------------------------------------------------------

def split_long_chunks(
    encoding,
    chunks: Iterable[Dict[str, Any]],
    max_tokens: int,
    overlap_paragraphs: int,
    split_mode: str = "paragraph",
    overlap_tokens: int = 0,
) -> List[Dict[str, Any]]:
    """
    Given an iterable of chunk dicts, detect those whose "text" is longer
    than `max_tokens` tokens and split them into multiple overlapping chunks.

    split_mode:
        - "paragraph" (default): paragraph-aware splitting.
        - "token": fixed token-window splitting.

    Returns:
        New list of chunk dicts.
    """
    new_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "") or ""
        token_count = count_tokens(encoding, text)
        source_chunk_index = chunk.get("chunk_index")
        if source_chunk_index is None:
            source_chunk_index = idx
        else:
            try:
                source_chunk_index = int(source_chunk_index)
            except Exception:
                source_chunk_index = idx

        try:
            split_mode_norm = str(split_mode).strip().lower()
        except Exception:
            split_mode_norm = "paragraph"

        split_texts: List[str]
        if token_count <= max_tokens:
            split_texts = []
        elif split_mode_norm == "token":
            split_texts = split_text_by_tokens(
                encoding,
                text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        else:
            split_texts = split_text_by_paragraph_tokens(
                encoding,
                text,
                max_tokens=max_tokens,
                overlap_paragraphs=overlap_paragraphs,
            )

        if token_count <= max_tokens:
            # Add small metadata about tokens, optional
            chunk["token_len"] = token_count
            chunk["split_index"] = 0
            chunk["split_count"] = 1
            chunk["chunk_index"] = source_chunk_index
            chunk["source_chunk_index"] = source_chunk_index
            if overlap_tokens or split_mode_norm == "token":
                chunk["split_id"] = str(source_chunk_index)
            new_chunks.append(chunk)
            continue

        hard_split_texts: List[str] = []
        for sub_text in split_texts:
            hard_split_texts.extend(_ensure_hard_token_cap(encoding, sub_text, max_tokens))
        if not hard_split_texts:
            # If fallback splitting somehow produced nothing, fall back to original
            hard_split_texts = [text]

        if not split_texts:
            # If splitting somehow failed, fall back to original
            chunk["token_len"] = token_count
            chunk["split_index"] = 0
            chunk["split_count"] = 1
            new_chunks.append(chunk)
            continue

        for sub_idx, sub_text in enumerate(hard_split_texts):
            new_chunk = dict(chunk)  # shallow copy original fields
            new_chunk["text"] = sub_text
            new_chunk["token_len"] = count_tokens(encoding, sub_text)
            new_chunk["split_index"] = sub_idx
            new_chunk["split_count"] = len(hard_split_texts)
            new_chunk["chunk_index"] = source_chunk_index
            new_chunk["source_chunk_index"] = idx
            new_chunk["split_id"] = f"{source_chunk_index}::split::{sub_idx}"
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
        description="Split long text chunks (from ingestion.sec_chunker) by token length.",
    )
    parser.add_argument(
        "--in-file",
        required=True,
        help="Input JSONL file with text chunks (e.g. data/chunked/AAPL/10-K/10-K_2025.text.jsonl)",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        help="Output JSONL file for split text chunks.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum tokens per chunk before splitting (default: 800).",
    )
    parser.add_argument(
        "--overlap-paragraphs",
        type=int,
        default=1,
        help="Number of paragraphs to overlap between consecutive chunks (default: 1).",
    )
    parser.add_argument(
        "--split-mode",
        choices=["paragraph", "token"],
        default="paragraph",
        help="Splitting mode: paragraph (existing) or token-window.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=0,
        help="Token overlap size when --split-mode=token (default: 0).",
    )
    parser.add_argument(
        "--encoding-model",
        type=str,
        default="text-embedding-3-large",
        help="Model name used to choose the tiktoken encoding (default: text-embedding-3-large).",
    )
    parser.add_argument(
        "--filter-levels",
        nargs="+",
        default=["subsection"],
        help=(
            "Chunk levels to keep before splitting (default: subsection). "
            "Use multiple values to keep more than one level."
        ),
    )
    parser.add_argument(
        "--parent-expand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prepend parent item text onto subsection chunks before splitting "
            "(default: enabled)."
        ),
    )

    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    print(f"Loading chunks from {in_path} ...")
    raw_chunks = load_jsonl(in_path)
    print(f"Loaded {len(raw_chunks)} chunks")

    work_chunks: List[Dict[str, Any]] = [dict(chunk) for chunk in raw_chunks]
    if args.parent_expand:
        work_chunks = apply_parent_expansion(work_chunks)

    level_tuple = tuple(args.filter_levels or [])
    work_chunks = filter_levels(work_chunks, level_tuple)
    print(
        f"Prepared chunks for splitting: {len(raw_chunks)} -> {len(work_chunks)} "
        f"(parent_expand={bool(args.parent_expand)}, filter_levels={list(level_tuple)})"
    )

    print(f"Initializing tiktoken encoding for {args.encoding_model} ...")
    encoding = get_encoding(args.encoding_model)

    print(
        f"Splitting chunks with max_tokens={args.max_tokens}, "
        f"overlap_paragraphs={args.overlap_paragraphs} ..."
    )
    new_chunks = split_long_chunks(
        encoding,
        work_chunks,
        max_tokens=args.max_tokens,
        overlap_paragraphs=args.overlap_paragraphs,
        split_mode=args.split_mode,
        overlap_tokens=args.overlap_tokens,
    )

    print(
        f"Done. Prepared chunks: {len(work_chunks)} -> New chunks: {len(new_chunks)}"
    )

    print(f"Saving new chunks to {out_path} ...")
    save_jsonl(out_path, new_chunks)
    print("All set.")


if __name__ == "__main__":
    main()
