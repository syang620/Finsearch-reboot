#!/usr/bin/env python
"""
Summarize SEC table chunks with an LLM for later embedding.

This script:
  1) Loads table chunks produced by sec_chunker/chunk_splitter (no notebook deps).
  2) Builds a strict JSON summarization prompt (from table_summarization_eval).
  3) Calls an LLM (default model: minimax-m2:cloud) via an HTTP API
     compatible with the Ollama /generate pattern.
  4) Writes one JSONL file with annotations per filing prefix.

Example:
    PYTHONPATH=src python -m rag10kq.tables_summarizer \\
        --prefixes AAPL_10-K_2024 AAPL_10-Q_2025Q1 \\
        --chunks-dir data/chunked \\
        --api-url http://localhost:11434/api/generate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd
import requests


DEFAULT_MODEL = "qwen2.5:7b"


# ---------------------------------------------------------------------
# Chunk loading helpers (adapted from notebooks/chunks_loader.py)
# ---------------------------------------------------------------------

PathLike = Union[str, Path]


def load_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    p = Path(path)
    chunks: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def load_text_chunks(path: PathLike) -> List[Dict[str, Any]]:
    """Load text chunks from a *.text.jsonl or *.text.split.jsonl file."""
    return load_jsonl(path)


def load_table_chunks(
    path: PathLike,
    rebuild_df: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load table chunks from a *.tables.jsonl file and optionally reconstruct DataFrames.
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
                # If reconstruction fails, skip table_df but keep the rest
                pass
        chunks.append(obj)
    return chunks


def load_filing_chunks(
    prefix: str,
    out_dir: PathLike = "./chunks",
    use_split_text: bool = True,
    rebuild_table_df: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load text + table chunks for a single filing, given a file prefix.

    The prefix should match how sec_chunker/chunk_splitter save files, e.g.:
        prefix = "AAPL_10-K_2024" or "AAPL_10-Q_2025Q1"

    This looks for:
        - text:
            * if use_split_text and <prefix>.text.split.jsonl exists:
                <out_dir>/<prefix>.text.split.jsonl
              else:
                <out_dir>/<prefix>.text.jsonl
        - tables:
            <out_dir>/<prefix>.tables.jsonl
    """
    out_dir = Path(out_dir)

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

    table_path = out_dir / f"{prefix}.tables.jsonl"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Could not find table chunk file for prefix '{prefix}' in {out_dir} "
            f"(expected {table_path})"
        )

    text_chunks = load_text_chunks(text_path)
    table_chunks = load_table_chunks(table_path, rebuild_df=rebuild_table_df)

    return text_chunks, table_chunks


def load_table_chunks_for_prefix(
    prefix: str,
    chunks_dir: Path,
) -> List[Dict[str, Any]]:
    """Load table chunks for a single filing prefix."""
    _text_chunks, table_chunks = load_filing_chunks(
        prefix=prefix,
        out_dir=str(chunks_dir),
        use_split_text=True,
        rebuild_table_df=True,
    )
    return table_chunks


# ---------------------------------------------------------------------
#  Helpers: table → Markdown (from table_summarization_eval)
# ---------------------------------------------------------------------

def df_to_markdown_for_llm(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown (no index)."""
    return df.to_markdown(index=False)


# ---------------------------------------------------------------------
#  System Prompt for summarization (from table_summarization_eval)
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a careful financial reporting analyst.

You are given:
- A markdown table extracted from an SEC 10-K or 10-Q.
- Optional filing metadata.

Your job:
1. Summarize in plain English what the entire table is about.
2. Summarize what each data row (non-header, non-section row) represents.

RULES:
- Use ONLY information from the table and metadata.
- Do NOT invent reasons, trends, or comparisons. Avoid change words like “increased”, “decreased”, “higher”, “lower”, etc.
- Do NOT repeat numeric values from the table (amounts, counts, dollar figures, etc.).
- You MAY mention years or period labels (e.g., 2022, 2023, 2024, Q1, September), copying them exactly.
- Summarize by row, NOT by column.
- Output MUST be a single valid JSON object, nothing else, starting with '{' and ending with '}'.

ROW HANDLING:
- If a row’s first cell is only a year or period label (e.g., "2024", "2024.0", "Q1"), or the row is mostly years/periods, treat it as a period header row, not a data row. Do NOT create a row_summaries entry for it.
- The first row after “Table (first row is header):” is the header row. Do NOT create a summary for the header row.
- Some rows are section headers that group later rows (e.g., labels ending with ':' like "Americas:", "Europe:", or rows with a label but no numeric data).
  - Do NOT create summaries for section headers.
  - Treat a section header as context for the rows that follow it, until the next section header.
- For a data row under a section header, build row_label as: "<section> – <metric label>".
  - Example: section "Americas:" + metric "Net sales$" → "Americas – Net sales$".
- If there is no section header, use the natural row label (first column) as row_label.
- You MUST output exactly one element in row_summaries for EACH non-header, non-section data row, in order.
- row_index is a 0-based index over these data rows:
  - The first data row has row_index = 0, the next has 1, etc.

JSON FORMAT:
{
  "table_summary": "one or two sentences about the whole table",
  "row_summaries": [
    {
      "row_index": <0-based integer index of the data row>,
      "row_label": "<row label, including section + metric if applicable>",
      "description": "one concise sentence describing what this row represents"
    }
  ]
}

EXAMPLE (ILLUSTRATIVE ONLY):

Filing metadata (example):
- Company: Apple Inc.
- Form type: 10-K
- Fiscal year: 2024
- Section path: PART II > Note – Segment Information and Geographic Data

Example table (first row is header):

| 1                     | 2        | 3    | 4        | 5    | 6        |
|:----------------------|:---------|:-----|:---------|:-----|:---------|
| 2024                  | 2024.0   | 2023 | 2023.0   | 2022 | 2022.0   |
| Americas:             |          |      |          |      |          |
| Net sales$            | 167045.0 | $    | 162560.0 | $    | 169658.0 |
| Operating income$     | 67656.0  | $    | 60508.0  | $    | 62683.0  |
| Europe:               |          |      |          |      |          |
| Net sales$            | 101328.0 | $    | 94294.0  | $    | 95118.0  |
| Operating income$     | 41790.0  | $    | 36098.0  | $    | 35233.0  |

Example JSON response for this example table:

{
  "table_summary": "The table presents segment information for Apple Inc., showing net sales and operating income for geographic segments such as Americas and Europe for fiscal years 2024, 2023, and 2022.",
  "row_summaries": [
    {
      "row_index": 0,
      "row_label": "Americas – Net sales$",
      "description": "Net sales for the Americas segment across fiscal years 2024, 2023, and 2022."
    },
    {
      "row_index": 1,
      "row_label": "Americas – Operating income$",
      "description": "Operating income for the Americas segment across fiscal years 2024, 2023, and 2022."
    },
    {
      "row_index": 2,
      "row_label": "Europe – Net sales$",
      "description": "Net sales for the Europe segment across fiscal years 2024, 2023, and 2022."
    },
    {
      "row_index": 3,
      "row_label": "Europe – Operating income$",
      "description": "Operating income for the Europe segment across fiscal years 2024, 2023, and 2022."
    }
  ]
}

When you receive a NEW table:
- Follow all rules above.
- Use section headers only as context.
- Create one row_summaries entry per data row.
- Do not repeat numeric values from the table.
- Respond with exactly one valid JSON object and nothing else.
"""


# ---------------------------------------------------------------------
#  Prompt builder (from table_summarization_eval, lightly adapted)
# ---------------------------------------------------------------------

def build_user_prompt(table_chunk: Dict[str, Any]) -> str:
    df: pd.DataFrame = table_chunk["table_df"]
    meta: Dict[str, Any] = table_chunk.get("meta", {})
    table_md = df_to_markdown_for_llm(df)
    heading_path = meta.get("heading_path", table_chunk.get("heading_path", []))

    return f"""Filing metadata:
- Company: {meta.get("company_name")}
- Ticker: {meta.get("ticker")}
- CIK: {meta.get("cik")}
- Form type: {meta.get("form_type")}
- Fiscal year: {meta.get("fiscal_year")}
- Filing date: {meta.get("filing_date")}
- Item: {meta.get("item_id")} – {meta.get("item_title")}
- Section path: { ' > '.join(heading_path) }

Table (first row is header):
{table_md}

Remember:
- Use only info in the table + metadata.
- Do not compute new metrics or trends.
- Do not repeat numeric values from the table.
- Years like 2023/2024/2025 are allowed.
- Respond with JSON only.
"""


# ---------------------------------------------------------------------
#  LLM summarization call (Ollama-style HTTP API)
# ---------------------------------------------------------------------

def summarize_table_with_llm(
    table_chunk: Dict[str, Any],
    model: str,
    api_url: str,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """
    Call an LLM to summarize a table with strict JSON output.

    The API is expected to be compatible with Ollama's /generate endpoint:
      POST api_url with JSON:
        {
          "model": model,
          "prompt": full_prompt,
          "stream": false,
          "format": "json",
          "options": {"temperature": temperature, "num_predict": 4096}
        }

    The response JSON should contain a 'response' field that is a JSON string
    matching the schema described in SYSTEM_PROMPT.
    """
    user_prompt = build_user_prompt(table_chunk)
    full_prompt = SYSTEM_PROMPT.strip() + "\n\n---\n\n" + user_prompt.strip()

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": 4096},
    }

    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()

    obj = resp.json()
    text = (obj.get("response") or "").strip()
    # print(obj)
    
    # Parse the JSON string returned by the model
    return json.loads(text)


# ---------------------------------------------------------------------
#  CLI + orchestration
# ---------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SEC table chunks for later embedding.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        required=True,
        help="Filing prefixes, e.g. AAPL_10-K_2024 AAPL_10-Q_2025Q1.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunked",
        help="Directory containing chunk JSONL files (default: data/chunked).",
    )
    parser.add_argument(
        "--output-jsonl",
        help=(
            "Path to save table summaries as JSONL. "
            "If omitted, defaults to <chunks-dir>/table_summaries.jsonl."
        ),
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="HTTP endpoint for the LLM (e.g., http://localhost:11434/api/generate).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-tables",
        type=int,
        help="Optional cap on number of tables per prefix to summarize.",
    )
    return parser.parse_args(argv)


def summarize_prefix(
    prefix: str,
    chunks_dir: Path,
    api_url: str,
    model: str,
    temperature: float,
    max_tables: int | None,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Summarize all (or up to max_tables) tables for a single filing prefix.
    """
    start_all = time.time()
    table_chunks = load_table_chunks_for_prefix(prefix, chunks_dir)
    records: List[Dict[str, Any]] = []
    # Open output file once if incremental saving is requested
    f = None
    if output_path is not None:
        # ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # "a" = append, so you can resume runs if you want
        f = output_path.open("a", encoding="utf-8")
        
    try:
        for idx, table_chunk in enumerate(table_chunks):
            if max_tables is not None and idx >= max_tables:
                break
    
            start = time.time()
            error: str | None = None
            annotation: Dict[str, Any] | None = None
    
            try:
                annotation = summarize_table_with_llm(
                    table_chunk=table_chunk,
                    model=model,
                    api_url=api_url,
                    temperature=temperature,
                )
            except Exception as exc:
                error = str(exc)
                print(error)
    
            elapsed = time.time() - start
    
            rec: Dict[str, Any] = {
                "prefix": prefix,
                "table_index": idx,
                "section_title": table_chunk.get("section_title"),
                "model": model,
                "elapsed_sec": elapsed,
                "error": error,
                "annotation": annotation,
            }
            records.append(rec)
            # --- incremental save ---
            if f is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()  # make sure it's on disk after each table
                # optional: os.fsync(f.fileno()) if you want extra safety
    finally:
        if f is not None:
            f.close()
            
    total_elapsed = time.time() - start_all
    print(
        f"[SUMMARY] prefix={prefix} "
        f"tables={len(table_chunks)} "
        f"total_elapsed={total_elapsed:.2f}s (sequential)"
    )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else chunks_dir / "table_summaries.jsonl"
    )

    all_records: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as f:
        for prefix in args.prefixes:
            print(f"\n=== Summarizing tables for prefix: {prefix} ===")
            records = summarize_prefix(
                prefix=prefix,
                chunks_dir=chunks_dir,
                api_url=args.api_url,
                model=args.model,
                temperature=args.temperature,
                max_tables=args.max_tables,
            )
            for rec in records:
                all_records.append(rec)
                f.write(json.dumps(rec, default=str) + "\n")

    print(f"\nWrote {len(all_records)} table summaries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
